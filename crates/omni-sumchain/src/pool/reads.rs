//! O2 (#104): typed reads of committed C1 pool state.
//!
//! sum-chain's read-only `pool_getUnit` and `pool_getOffer` (sum-chain #130;
//! the server at `90c49da1`) answer from committed pool rows. This module
//! builds their requests and decodes their answers **fail-closed**: every id
//! must be a `0x`-prefixed 64-digit lowercase hex string, every address a
//! canonical checksummed base58 address, every amount a canonical decimal
//! `u128`, every state one of the six the chain names, and no field may be
//! missing, extra or of another type. An answer about a different unit or
//! offer than the one asked for, and a reservation above its capacity, are
//! refused too. Nothing is defaulted or repaired.
//!
//! **`None` means the chain holds no such row.** On every network today that is
//! the only answer: genesis refuses the pool gate, and no C1 operation has an
//! executor (sum-chain #130 / #217), so no block writes pool state. A `None`
//! is not an error and not a reason to retry.

use serde::Deserialize;
use sumchain_wire::address::Address;
use sumchain_wire::compute_pool_wire::WorkItemRef;

use super::ids::OwnOffer;
use crate::rpc::JsonRpcTransport;
use crate::SumChainClient;

/// JSON-RPC method name of the unit read.
pub const METHOD_GET_UNIT: &str = "pool_getUnit";
/// JSON-RPC method name of the offer read.
pub const METHOD_GET_OFFER: &str = "pool_getOffer";

/// Why a pool read did not produce a value.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PoolReadError {
    /// The transport or the JSON-RPC envelope failed, or the server answered
    /// with a JSON-RPC error (an invalid id, or committed state the server
    /// itself found inconsistent).
    #[error("{method}: {message}")]
    Rpc {
        method: &'static str,
        message: String,
    },
    /// The server answered, but not in the shape this client accepts.
    #[error("{method}: malformed response: {reason}")]
    Malformed {
        method: &'static str,
        reason: String,
    },
}

/// A unit's lifecycle state, exactly the six the chain names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitState {
    Blocked,
    Eligible,
    Assigned,
    Accepted,
    Reassignable,
    AssignmentHalted,
}

impl UnitState {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "blocked" => Self::Blocked,
            "eligible" => Self::Eligible,
            "assigned" => Self::Assigned,
            "accepted" => Self::Accepted,
            "reassignable" => Self::Reassignable,
            "assignment_halted" => Self::AssignmentHalted,
            _ => return None,
        })
    }
}

/// One input a unit requires from a predecessor's output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequiredInput {
    pub predecessor: [u8; 32],
    pub required_output_slot_id: [u8; 32],
    pub pred_output_manifest_root: [u8; 32],
    pub required_slot_state_object_root: [u8; 32],
}

/// The assignment recorded for a unit's current-generation work item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    pub winner_offer_bond_id: [u8; 32],
    pub winner_payment_addr: Address,
}

/// The write-once accepted leaf recorded for a unit's current-generation work
/// item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedLeaf {
    pub offer_bond_id: [u8; 32],
    pub commit_bond_id: [u8; 32],
    pub accepted_bytes: u128,
}

/// One work unit, as committed state holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolUnit {
    pub job_id: [u8; 32],
    pub unit_id: [u8; 32],
    pub generation: u64,
    pub state: UnitState,
    pub predecessors: Vec<[u8; 32]>,
    pub required_inputs: Vec<RequiredInput>,
    pub assignment: Option<Assignment>,
    pub accepted_leaf: Option<AcceptedLeaf>,
}

impl PoolUnit {
    /// The current-generation work item `(job, unit, generation)`: the one the
    /// assignment and accepted leaf are recorded for, and the one an
    /// acceptance must name.
    pub fn work_item(&self) -> WorkItemRef {
        WorkItemRef {
            job_id: self.job_id,
            unit_id: self.unit_id,
            generation: self.generation,
        }
    }
}

/// The capacity reservation held against an offer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reservation {
    pub offered: u128,
    pub reserved: u128,
}

/// One bonded offer, as committed state holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolOffer {
    pub offer_bond_id: [u8; 32],
    pub identity: Address,
    pub payment_addr: Address,
    pub offered_bytes: u128,
    pub offer_seq: u64,
    pub bond_locked: u128,
    pub active: bool,
    pub reservation: Option<Reservation>,
}

/// `0x` followed by the 64 lowercase hex digits of `bytes`: the server's id
/// format, and the only one this client sends.
pub fn hex32(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(66);
    s.push_str("0x");
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// `pool_getUnit(job_id, unit_id)`.
pub fn get_unit<T: JsonRpcTransport + ?Sized>(
    transport: &T,
    job_id: &[u8; 32],
    unit_id: &[u8; 32],
) -> Result<Option<PoolUnit>, PoolReadError> {
    let m = METHOD_GET_UNIT;
    let result = transport
        .call(m, serde_json::json!([hex32(job_id), hex32(unit_id)]))
        .map_err(|e| PoolReadError::Rpc {
            method: m,
            message: e.to_string(),
        })?;
    if result.is_null() {
        return Ok(None);
    }
    let raw: RawUnit = from_value(m, result)?;
    let unit = raw.into_unit(m)?;
    if &unit.job_id != job_id || &unit.unit_id != unit_id {
        return Err(malformed(
            m,
            format!(
                "asked for unit ({}, {}), answered for ({}, {})",
                hex32(job_id),
                hex32(unit_id),
                hex32(&unit.job_id),
                hex32(&unit.unit_id)
            ),
        ));
    }
    Ok(Some(unit))
}

/// `pool_getOffer(offer_bond_id)`.
pub fn get_offer<T: JsonRpcTransport + ?Sized>(
    transport: &T,
    offer_bond_id: &[u8; 32],
) -> Result<Option<PoolOffer>, PoolReadError> {
    let m = METHOD_GET_OFFER;
    let result = transport
        .call(m, serde_json::json!([hex32(offer_bond_id)]))
        .map_err(|e| PoolReadError::Rpc {
            method: m,
            message: e.to_string(),
        })?;
    if result.is_null() {
        return Ok(None);
    }
    let raw: RawOffer = from_value(m, result)?;
    let offer = raw.into_offer(m)?;
    if &offer.offer_bond_id != offer_bond_id {
        return Err(malformed(
            m,
            format!(
                "asked for offer {}, answered for {}",
                hex32(offer_bond_id),
                hex32(&offer.offer_bond_id)
            ),
        ));
    }
    Ok(Some(offer))
}

/// `pool_getOffer` for this node's own offer: the id is derived from `own`
/// (chain id, identity, `offer_seq`), never supplied. [`get_offer`] already
/// refuses an answer for another id; the accept path also checks the row's
/// identity and `offer_seq`.
pub fn get_own_offer<T: JsonRpcTransport + ?Sized>(
    transport: &T,
    own: &OwnOffer,
) -> Result<Option<PoolOffer>, PoolReadError> {
    get_offer(transport, &own.offer_bond_id())
}

impl<T: JsonRpcTransport> SumChainClient<T> {
    /// `pool_getUnit` through this client's transport. See [`get_unit`].
    pub fn pool_get_unit(
        &self,
        job_id: &[u8; 32],
        unit_id: &[u8; 32],
    ) -> Result<Option<PoolUnit>, PoolReadError> {
        get_unit(self.transport(), job_id, unit_id)
    }

    /// `pool_getOffer` through this client's transport. See [`get_offer`].
    pub fn pool_get_offer(
        &self,
        offer_bond_id: &[u8; 32],
    ) -> Result<Option<PoolOffer>, PoolReadError> {
        get_offer(self.transport(), offer_bond_id)
    }

    /// `pool_getOffer` for this node's own offer, by its derived id. See
    /// [`get_own_offer`].
    pub fn pool_get_own_offer(&self, own: &OwnOffer) -> Result<Option<PoolOffer>, PoolReadError> {
        get_own_offer(self.transport(), own)
    }
}

// ── Strict decoding ──────────────────────────────────────────────────────────

fn malformed(method: &'static str, reason: String) -> PoolReadError {
    PoolReadError::Malformed { method, reason }
}

fn from_value<D: serde::de::DeserializeOwned>(
    method: &'static str,
    v: serde_json::Value,
) -> Result<D, PoolReadError> {
    serde_json::from_value(v).map_err(|e| malformed(method, e.to_string()))
}

fn parse_id(method: &'static str, field: &str, s: &str) -> Result<[u8; 32], PoolReadError> {
    let digits = s
        .strip_prefix("0x")
        .ok_or_else(|| malformed(method, format!("{field}: not 0x-prefixed: {s:?}")))?;
    if digits.len() != 64 {
        return Err(malformed(
            method,
            format!("{field}: {} hex digits, not 64", digits.len()),
        ));
    }
    let mut out = [0u8; 32];
    for (i, pair) in digits.as_bytes().chunks(2).enumerate() {
        let hi = lower_hex_digit(pair[0]);
        let lo = lower_hex_digit(pair[1]);
        match (hi, lo) {
            (Some(h), Some(l)) => out[i] = (h << 4) | l,
            _ => {
                return Err(malformed(
                    method,
                    format!("{field}: not lowercase hex: {s:?}"),
                ))
            }
        }
    }
    Ok(out)
}

fn lower_hex_digit(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

/// A canonical decimal `u128`: digits only, no sign, no leading zero (except
/// `"0"` itself), at most `u128::MAX`.
fn parse_amount(method: &'static str, field: &str, s: &str) -> Result<u128, PoolReadError> {
    let canonical =
        !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()) && (s == "0" || !s.starts_with('0'));
    if !canonical {
        return Err(malformed(
            method,
            format!("{field}: not a canonical decimal amount: {s:?}"),
        ));
    }
    s.parse::<u128>()
        .map_err(|_| malformed(method, format!("{field}: above u128::MAX: {s:?}")))
}

/// A checksummed base58 address in its one canonical spelling.
fn parse_address(method: &'static str, field: &str, s: &str) -> Result<Address, PoolReadError> {
    let a = Address::from_base58(s)
        .map_err(|e| malformed(method, format!("{field}: not a base58 address: {e}")))?;
    if a.to_base58() != s {
        return Err(malformed(
            method,
            format!("{field}: not the canonical base58 spelling: {s:?}"),
        ));
    }
    Ok(a)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRequiredInput {
    predecessor: String,
    required_output_slot_id: String,
    pred_output_manifest_root: String,
    required_slot_state_object_root: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAssignment {
    winner_offer_bond_id: String,
    winner_payment_addr: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAcceptedLeaf {
    offer_bond_id: String,
    commit_bond_id: String,
    accepted_bytes: String,
}

/// Every field is required: `Option` fields still have to be present (as
/// `null` or a value), because `deny_unknown_fields` plus `default`-less
/// `Option` would otherwise let a missing field read as `null`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawUnit {
    job_id: String,
    unit_id: String,
    generation: u64,
    state: String,
    predecessors: Vec<String>,
    required_inputs: Vec<RawRequiredInput>,
    #[serde(deserialize_with = "required_nullable")]
    assignment: Option<RawAssignment>,
    #[serde(deserialize_with = "required_nullable")]
    accepted_leaf: Option<RawAcceptedLeaf>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawReservation {
    offered: String,
    reserved: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOffer {
    offer_bond_id: String,
    identity: String,
    payment_addr: String,
    offered_bytes: String,
    offer_seq: u64,
    bond_locked: String,
    active: bool,
    #[serde(deserialize_with = "required_nullable")]
    reservation: Option<RawReservation>,
}

/// An `Option` field that must be present: a field with `deserialize_with`
/// and no `default` is required by serde, so a missing key is an error while
/// an explicit `null` reads as `None`.
fn required_nullable<'de, D, V>(d: D) -> Result<Option<V>, D::Error>
where
    D: serde::Deserializer<'de>,
    V: Deserialize<'de>,
{
    Option::<V>::deserialize(d)
}

impl RawUnit {
    fn into_unit(self, m: &'static str) -> Result<PoolUnit, PoolReadError> {
        let state = UnitState::parse(&self.state)
            .ok_or_else(|| malformed(m, format!("state: unknown unit state {:?}", self.state)))?;
        let predecessors = self
            .predecessors
            .iter()
            .map(|p| parse_id(m, "predecessors[]", p))
            .collect::<Result<Vec<_>, _>>()?;
        let required_inputs = self
            .required_inputs
            .iter()
            .map(|ri| {
                Ok(RequiredInput {
                    predecessor: parse_id(m, "required_inputs[].predecessor", &ri.predecessor)?,
                    required_output_slot_id: parse_id(
                        m,
                        "required_inputs[].required_output_slot_id",
                        &ri.required_output_slot_id,
                    )?,
                    pred_output_manifest_root: parse_id(
                        m,
                        "required_inputs[].pred_output_manifest_root",
                        &ri.pred_output_manifest_root,
                    )?,
                    required_slot_state_object_root: parse_id(
                        m,
                        "required_inputs[].required_slot_state_object_root",
                        &ri.required_slot_state_object_root,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, PoolReadError>>()?;
        let assignment = self
            .assignment
            .map(|a| {
                Ok::<_, PoolReadError>(Assignment {
                    winner_offer_bond_id: parse_id(
                        m,
                        "assignment.winner_offer_bond_id",
                        &a.winner_offer_bond_id,
                    )?,
                    winner_payment_addr: parse_address(
                        m,
                        "assignment.winner_payment_addr",
                        &a.winner_payment_addr,
                    )?,
                })
            })
            .transpose()?;
        let accepted_leaf = self
            .accepted_leaf
            .map(|l| {
                Ok::<_, PoolReadError>(AcceptedLeaf {
                    offer_bond_id: parse_id(m, "accepted_leaf.offer_bond_id", &l.offer_bond_id)?,
                    commit_bond_id: parse_id(m, "accepted_leaf.commit_bond_id", &l.commit_bond_id)?,
                    accepted_bytes: parse_amount(
                        m,
                        "accepted_leaf.accepted_bytes",
                        &l.accepted_bytes,
                    )?,
                })
            })
            .transpose()?;
        Ok(PoolUnit {
            job_id: parse_id(m, "job_id", &self.job_id)?,
            unit_id: parse_id(m, "unit_id", &self.unit_id)?,
            generation: self.generation,
            state,
            predecessors,
            required_inputs,
            assignment,
            accepted_leaf,
        })
    }
}

impl RawOffer {
    fn into_offer(self, m: &'static str) -> Result<PoolOffer, PoolReadError> {
        let reservation = self
            .reservation
            .map(|r| {
                let offered = parse_amount(m, "reservation.offered", &r.offered)?;
                let reserved = parse_amount(m, "reservation.reserved", &r.reserved)?;
                // The chain's capacity invariant; its own reader refuses a row
                // that breaks it, so a response that does is not chain state.
                if reserved > offered {
                    return Err(malformed(
                        m,
                        format!("reservation.reserved {reserved} above offered {offered}"),
                    ));
                }
                Ok(Reservation { offered, reserved })
            })
            .transpose()?;
        Ok(PoolOffer {
            offer_bond_id: parse_id(m, "offer_bond_id", &self.offer_bond_id)?,
            identity: parse_address(m, "identity", &self.identity)?,
            payment_addr: parse_address(m, "payment_addr", &self.payment_addr)?,
            offered_bytes: parse_amount(m, "offered_bytes", &self.offered_bytes)?,
            offer_seq: self.offer_seq,
            bond_locked: parse_amount(m, "bond_locked", &self.bond_locked)?,
            active: self.active,
            reservation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: &str = "test";

    #[test]
    fn ids_are_exactly_0x_and_64_lowercase_hex_digits() {
        let ok = format!("0x{}", "ab".repeat(32));
        assert_eq!(parse_id(M, "f", &ok).unwrap(), [0xab; 32]);
        assert_eq!(hex32(&[0xab; 32]), ok);
        for bad in [
            "ab".repeat(32),                  // no 0x
            format!("0X{}", "ab".repeat(32)), // upper-case prefix
            format!("0x{}", "AB".repeat(32)), // upper-case digits
            format!("0x{}", "ab".repeat(31)), // short
            format!("0x{}", "ab".repeat(33)), // long
            format!("0x{}g", "a".repeat(63)), // not hex
            String::new(),
        ] {
            assert!(parse_id(M, "f", &bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn amounts_are_canonical_decimal_u128() {
        assert_eq!(parse_amount(M, "f", "0").unwrap(), 0);
        assert_eq!(
            parse_amount(M, "f", "340282366920938463463374607431768211455").unwrap(),
            u128::MAX
        );
        for bad in [
            "",
            "-1",
            "+1",
            "01",
            "1.0",
            "1e3",
            " 1",
            "340282366920938463463374607431768211456", // u128::MAX + 1
        ] {
            assert!(parse_amount(M, "f", bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn addresses_must_be_checksummed_and_canonical() {
        let a = Address::new([0x50; 20]);
        let s = a.to_base58();
        assert_eq!(parse_address(M, "f", &s).unwrap(), a);
        // A changed character breaks the checksum.
        let mut chars: Vec<char> = s.chars().collect();
        let last = chars.len() - 1;
        chars[last] = if chars[last] == '2' { '3' } else { '2' };
        let broken: String = chars.into_iter().collect();
        assert!(parse_address(M, "f", &broken).is_err());
        assert!(parse_address(M, "f", "").is_err());
        assert!(parse_address(M, "f", "0OIl").is_err());
    }

    #[test]
    fn unit_states_are_exactly_the_chains_six() {
        for (s, st) in [
            ("blocked", UnitState::Blocked),
            ("eligible", UnitState::Eligible),
            ("assigned", UnitState::Assigned),
            ("accepted", UnitState::Accepted),
            ("reassignable", UnitState::Reassignable),
            ("assignment_halted", UnitState::AssignmentHalted),
        ] {
            assert_eq!(UnitState::parse(s), Some(st));
        }
        for bad in ["Assigned", "ASSIGNED", "halted", ""] {
            assert_eq!(UnitState::parse(bad), None, "{bad:?}");
        }
    }
}
