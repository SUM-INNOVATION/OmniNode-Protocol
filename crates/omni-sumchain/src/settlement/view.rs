//! Issue #83 — normalized OmniNode-side views composed from one or
//! more [`super::wire`] raw DTOs. This is where multi-verifier
//! `Vec<_>` and consistency-group composition live; the raw wire
//! types stay a per-RPC verbatim mirror.
//!
//! ## View-level gate policy
//!
//! [`SettlementSessionView::compose`] enforces four view-level
//! preconditions before it will produce a composed view:
//!
//! 1. `session.consistency_required && !consistency_gate_active`
//!    → `ViewIncomplete { Consistency, .. }`. The chain's
//!    consistency-mode grouping is off; the view cannot be trusted.
//! 2. `session.consistency_required && consistency.is_none()`
//!    → `ViewIncomplete { Consistency, .. }`. The caller ran the RPC
//!    for a consistency-mode session but did not pass the fetched
//!    [`InferenceConsistencyRaw`]. The view falls back to attestation
//!    digests without it, which defeats the "full consistency-group
//!    read from day one" requirement.
//! 3. `session.bond_required && !bonding_gate_active`
//!    → `ViewIncomplete { Bonding, .. }`. Verifier registry state is
//!    unavailable on the chain.
//! 4. `session.bond_required && verifier_registry.is_none()`
//!    → `ViewIncomplete { Bonding, .. }`. Bonding gate is active but
//!    the caller did not fetch verifier registry entries for the
//!    verifiers seen in the session; per-verifier bond state cannot
//!    be composed.
//!
//! Callers typically pass the flags derived from `chain_getChainParams`
//! after any RPC-level gate has already passed:
//!
//! ```ignore
//! let consistency_active = params
//!     .inference_settlement_consistency_enabled_from_height
//!     .map(|n| head >= n)
//!     .unwrap_or(false);
//! ```

use std::collections::{BTreeMap, BTreeSet};

use crate::dto::InferenceAttestationInfo;

use super::dormancy::{SettlementGate, SettlementReadError};
use super::wire::{
    ConsistencyGroupRaw, DigestTupleRaw, InferenceClaimRaw, InferenceClaimsRaw,
    InferenceConsistencyRaw, InferenceDisputeRaw, InferenceDisputesRaw,
    InferenceSessionRaw, VerifierRegistryRaw,
};

// ── Enums / typed view fields ────────────────────────────────────────────────

/// Session mode flags — mirror of the two boolean fields on
/// [`super::wire::InferenceSessionRaw`], surfaced separately so a view
/// consumer can `match` on the combined mode without wire concerns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionModeFlags {
    pub consistency_required: bool,
    pub bond_required: bool,
}

/// Session lifecycle. `Unknown(_)` preserves the raw wire string if
/// the chain ever emits a value outside the documented set — the
/// operator can surface it verbatim without a hard parse failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionLifecycle {
    Active,
    Settled,
    Refunded,
    Unknown(String),
}

impl SessionLifecycle {
    pub fn from_wire(s: &str) -> Self {
        // Chain-team confirmed (sum-chain#110) capitalized status
        // values on the wire (`"Open"` / `"Refunded"` / `"Settled"`).
        // The lowercase variants remain for backward compat with
        // older mocks / fixtures that predate the chain-team
        // clarification. Any unrecognized value falls through to
        // `Unknown` so the caller can still render it verbatim.
        match s {
            // Chain-confirmed shape ─────────────────────────────
            "Open" => Self::Active,       // "Open" is the chain's active-session status
            "Settled" => Self::Settled,
            "Refunded" => Self::Refunded,
            // Legacy / pre-clarification shape ──────────────────
            "active" => Self::Active,
            "settled" => Self::Settled,
            "refunded" => Self::Refunded,
            _ => Self::Unknown(s.to_string()),
        }
    }
}

/// Full four-part digest tuple. Same shape as
/// [`super::wire::DigestTupleRaw`]; kept as a distinct view type so
/// the view layer doesn't leak wire-namespaced identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DigestTuple {
    pub model_hash: String,
    pub manifest_root: String,
    pub response_hash: String,
    pub proof_root: String,
}

impl From<DigestTupleRaw> for DigestTuple {
    fn from(r: DigestTupleRaw) -> Self {
        Self {
            model_hash: r.model_hash,
            manifest_root: r.manifest_root,
            response_hash: r.response_hash,
            proof_root: r.proof_root,
        }
    }
}

/// Claim state for a single (session, verifier) tuple. `UnknownWire`
/// preserves the raw string if the chain adds a state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimState {
    NotSubmitted,
    Pending { claimed_at_height: u64, reward_amount: u128 },
    Paid { paid_at_height: u64, reward_amount: u128 },
    Denied {
        denied_at_height: Option<u64>,
        reason: Option<String>,
    },
    UnknownWire(String),
}

/// Dispute state for a single (session, verifier) tuple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisputeState {
    None,
    Open {
        opened_at_height: u64,
    },
    ResolvedApproved {
        resolved_at_height: Option<u64>,
        approve_bps: Option<u32>,
        deny_bps: Option<u32>,
    },
    /// Chain resolved the dispute against the verifier. **Reward is
    /// denied.** Whether stake is also removed depends on the session's
    /// `bond_required` flag and is chain-side; the read layer never
    /// calls a reward-denied outcome "slashing".
    ResolvedDenied {
        resolved_at_height: Option<u64>,
        approve_bps: Option<u32>,
        deny_bps: Option<u32>,
    },
    UnknownWire(String),
}

/// Bond state — normalized from the wire `status` string
/// (`"Active"` → [`BondState::Bonded`], `"Unbonding"`, `"Withdrawn"`).
/// `UnknownWire` preserves the raw value if the chain adds a state, so
/// a read-only client never hard-fails on an unrecognized status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BondState {
    Bonded,
    Unbonding,
    Withdrawn,
    UnknownWire(String),
}

/// Per-verifier bond summary composed from
/// [`super::wire::VerifierRegistryRaw`]. Present on
/// [`PerVerifierView::bond_summary`] iff the session is bond-required
/// AND the caller supplied a registry entry for this verifier's
/// address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BondSummary {
    pub bond_amount: u128,
    pub bond_state: BondState,
    pub unbonding_since_height: Option<u64>,
    pub withdrawable_at_height: Option<u64>,
}

/// Per-verifier view for one session — always populated inside
/// [`SettlementSessionView::verifiers`], regardless of whether the
/// session has one verifier or many.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerVerifierView {
    pub verifier_address: String,
    /// The attestation record if one was included on chain.
    pub attestation: Option<AttestationSummary>,
    /// Consistency-group digest tuple for this verifier, if the
    /// consistency-gate-active read produced grouping data covering
    /// this verifier. `None` if consistency mode is off or the group
    /// data hasn't been fetched.
    pub digest_tuple: Option<DigestTuple>,
    pub claim_state: ClaimState,
    pub dispute_state: DisputeState,
    /// Bond summary composed from the verifier registry entry. `None`
    /// when the session is not bond-required, or when the caller
    /// supplied registry data but no entry covered this verifier
    /// (verifier may be pending registration or de-registered).
    /// For a bond-required session, [`SettlementSessionView::compose`]
    /// guarantees the caller PASSED registry data (else
    /// `ViewIncomplete` is returned before this field is populated).
    pub bond_summary: Option<BondSummary>,
}

/// Minimal attestation summary — full attestation details stay on
/// [`crate::dto::InferenceAttestationInfo`]; the view keeps just the
/// fields most callers need to correlate with a claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationSummary {
    pub tx_hash: String,
    pub included_at_height: u64,
    pub finalized: bool,
    pub digest_tuple: DigestTuple,
}

/// Multi-verifier settlement session view. `verifiers` is always a
/// `Vec`, populated by unioning the verifier addresses seen across
/// attestations / claims / disputes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettlementSessionView {
    pub session_id: String,
    pub mode: SessionModeFlags,
    pub max_verifiers: u32,
    pub escrow_total: u128,
    pub escrow_remaining: u128,
    pub claims_count: u32,
    pub verifiers: Vec<PerVerifierView>,
    pub lifecycle: SessionLifecycle,
    pub created_at_height: u64,
    pub settled_at_height: Option<u64>,
    pub refunded_at_height: Option<u64>,
    /// Plurality-winning digest tuple, if the caller passed a
    /// `consistency` argument to [`Self::compose`] and the chain
    /// had resolved plurality at read time.
    pub plurality_key: Option<DigestTuple>,
}

impl SettlementSessionView {
    /// Compose a normalized view from the raw wire responses that
    /// together describe one session.
    ///
    /// `consistency_gate_active` and `bonding_gate_active` are the
    /// **caller-observed** results of the two view-level gate checks —
    /// derived from `chain_getChainParams` fields against the current
    /// head.
    ///
    /// `consistency` is `Some` only when the caller has fetched
    /// [`InferenceConsistencyRaw`] separately (typically via
    /// [`crate::SumChainClient::omninode_get_inference_consistency`]).
    /// **For consistency-mode sessions, `Some(...)` is REQUIRED** —
    /// composing without it silently falls back to attestation digests
    /// and defeats the day-one plurality-group requirement.
    ///
    /// `verifier_registry` carries the entries the caller fetched via
    /// [`crate::SumChainClient::omninode_get_verifier`] for each
    /// verifier of interest. **For bond-required sessions, `Some(...)`
    /// is REQUIRED** — an empty `Some(vec![])` is acceptable if the
    /// registry has no matching entries yet, but `None` indicates the
    /// caller failed to fetch registry state and per-verifier bond
    /// summaries cannot be composed.
    ///
    /// Returns [`SettlementReadError::ViewIncomplete`] when any of the
    /// four view-level preconditions in this module's docs are not
    /// met.
    pub fn compose(
        session: InferenceSessionRaw,
        claims: InferenceClaimsRaw,
        disputes: InferenceDisputesRaw,
        attestations: Vec<InferenceAttestationInfo>,
        consistency: Option<InferenceConsistencyRaw>,
        verifier_registry: Option<Vec<VerifierRegistryRaw>>,
        consistency_gate_active: bool,
        bonding_gate_active: bool,
    ) -> Result<Self, SettlementReadError> {
        // View-level gate + input checks — RPC-level checks are the
        // adapter's job; here we ensure the session's declared mode is
        // fully supportable by the currently-active gates AND the
        // caller supplied the DTOs those gates unlock.
        if session.consistency_required && !consistency_gate_active {
            return Err(SettlementReadError::ViewIncomplete {
                missing_gate: SettlementGate::Consistency,
                session_id: session.session_id.clone(),
                reason: "session is consistency-mode; composing a plurality-aware \
                         view requires the consistency gate to be active"
                    .to_string(),
            });
        }
        if session.consistency_required && consistency.is_none() {
            return Err(SettlementReadError::ViewIncomplete {
                missing_gate: SettlementGate::Consistency,
                session_id: session.session_id.clone(),
                reason: "session is consistency-mode; composing a plurality-aware \
                         view requires the caller to pass Some(InferenceConsistencyRaw) \
                         fetched via omninode_getInferenceConsistency"
                    .to_string(),
            });
        }
        if session.bond_required && !bonding_gate_active {
            return Err(SettlementReadError::ViewIncomplete {
                missing_gate: SettlementGate::Bonding,
                session_id: session.session_id.clone(),
                reason: "session is bond-required; verifier registry state \
                         requires the bonding gate to be active"
                    .to_string(),
            });
        }
        if session.bond_required && verifier_registry.is_none() {
            return Err(SettlementReadError::ViewIncomplete {
                missing_gate: SettlementGate::Bonding,
                session_id: session.session_id.clone(),
                reason: "session is bond-required; composing per-verifier bond \
                         state requires the caller to pass Some(Vec<VerifierRegistryRaw>) \
                         fetched via omninode_getVerifier for each verifier of interest"
                    .to_string(),
            });
        }

        let escrow_total = parse_u128_decimal(&session.escrow_total, "escrow_total")?;
        let escrow_remaining =
            parse_u128_decimal(&session.escrow_remaining, "escrow_remaining")?;

        // Union verifier addresses across attestations / claims / disputes.
        // BTreeSet gives deterministic ordering — tests can pin the
        // vec's element order without extra sort logic.
        let mut verifier_set: BTreeSet<String> = BTreeSet::new();
        for a in &attestations {
            verifier_set.insert(a.verifier_address.clone());
        }
        for c in &claims.claims {
            verifier_set.insert(c.verifier_address.clone());
        }
        for d in &disputes.disputes {
            verifier_set.insert(d.verifier_address.clone());
        }

        // Index consistency groups by verifier address for lookup.
        // For consistency-mode sessions, `consistency` is guaranteed
        // `Some` by the guard above.
        let mut verifier_to_digest: BTreeMap<String, DigestTuple> = BTreeMap::new();
        let plurality_key = match &consistency {
            Some(c) => {
                for ConsistencyGroupRaw { key, members } in &c.groups {
                    let tuple: DigestTuple = key.clone().into();
                    for m in members {
                        verifier_to_digest.insert(m.clone(), tuple.clone());
                    }
                }
                c.plurality_key.clone().map(Into::into)
            }
            None => None,
        };

        // Index verifier registry entries by address for per-verifier
        // bond composition. For bond-required sessions, `verifier_registry`
        // is guaranteed `Some(_)` by the guard above; empty Vec is
        // acceptable (registry may have no entries yet). For non-bond
        // sessions, callers may pass either `None` or `Some(...)`; the
        // per-verifier bond_summary will only be populated if we find a
        // matching entry.
        let registry_by_address: BTreeMap<String, &VerifierRegistryRaw> = match &verifier_registry
        {
            Some(entries) => entries.iter().map(|e| (e.verifier.clone(), e)).collect(),
            None => BTreeMap::new(),
        };

        // Build per-verifier records. Any wire-parse failure inside a
        // claim's amount field or a bond's amount field bubbles up as
        // SettlementReadError.
        let mut verifiers: Vec<PerVerifierView> = Vec::with_capacity(verifier_set.len());
        for addr in verifier_set.into_iter() {
            let attestation = attestations
                .iter()
                .find(|a| a.verifier_address == addr)
                .map(|a| AttestationSummary {
                    tx_hash: a.tx_hash.clone(),
                    included_at_height: a.included_at_height,
                    finalized: a.finalized,
                    digest_tuple: DigestTuple {
                        model_hash: a.model_hash.clone(),
                        manifest_root: a.manifest_root.clone(),
                        response_hash: a.response_hash.clone(),
                        proof_root: a.proof_root.clone(),
                    },
                });

            // Prefer explicit consistency-group data when present;
            // otherwise fall back to the attestation's own digest.
            // (The fallback only runs on non-consistency-mode sessions
            // where `consistency` was legitimately `None`; consistency-
            // mode sessions are guarded to require `Some(...)` above.)
            let digest_tuple = verifier_to_digest
                .get(&addr)
                .cloned()
                .or_else(|| attestation.as_ref().map(|s| s.digest_tuple.clone()));

            let claim_state = match claims.claims.iter().find(|c| c.verifier_address == addr)
            {
                Some(c) => classify_claim(c)?,
                None => ClaimState::NotSubmitted,
            };

            let dispute_state = disputes
                .disputes
                .iter()
                .find(|d| d.verifier_address == addr)
                .map(classify_dispute)
                .unwrap_or(DisputeState::None);

            let bond_summary = registry_by_address.get(&addr).map(|r| parse_bond_summary(r));

            verifiers.push(PerVerifierView {
                verifier_address: addr,
                attestation,
                digest_tuple,
                claim_state,
                dispute_state,
                bond_summary,
            });
        }

        Ok(SettlementSessionView {
            session_id: session.session_id,
            mode: SessionModeFlags {
                consistency_required: session.consistency_required,
                bond_required: session.bond_required,
            },
            max_verifiers: session.max_verifiers,
            escrow_total,
            escrow_remaining,
            claims_count: session.claims_count,
            verifiers,
            lifecycle: SessionLifecycle::from_wire(&session.lifecycle),
            created_at_height: session.created_at_height,
            settled_at_height: session.settled_at_height,
            refunded_at_height: session.refunded_at_height,
            plurality_key,
        })
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn parse_u128_decimal(s: &str, field: &str) -> Result<u128, SettlementReadError> {
    s.parse::<u128>().map_err(|e| {
        SettlementReadError::WireParse(format!(
            "failed to parse {field} as u128 decimal from '{s}': {e}"
        ))
    })
}

fn classify_claim(c: &InferenceClaimRaw) -> Result<ClaimState, SettlementReadError> {
    let reward_amount = parse_u128_decimal(&c.reward_amount, "reward_amount")?;
    Ok(match c.state.as_str() {
        "pending" => ClaimState::Pending {
            claimed_at_height: c.claimed_at_height,
            reward_amount,
        },
        "paid" => ClaimState::Paid {
            paid_at_height: c.paid_at_height.unwrap_or(c.claimed_at_height),
            reward_amount,
        },
        "denied" => ClaimState::Denied {
            denied_at_height: c.denied_at_height,
            reason: c.denied_reason.clone(),
        },
        other => ClaimState::UnknownWire(other.to_string()),
    })
}

fn parse_bond_summary(raw: &VerifierRegistryRaw) -> BondSummary {
    let bond_state = match raw.status.as_str() {
        "Active" => BondState::Bonded,
        "Unbonding" => BondState::Unbonding,
        "Withdrawn" => BondState::Withdrawn,
        other => BondState::UnknownWire(other.to_string()),
    };
    BondSummary {
        bond_amount: raw.bond,
        bond_state,
        unbonding_since_height: raw.unbonding_started_height,
        withdrawable_at_height: raw.unlock_height,
    }
}

fn classify_dispute(d: &InferenceDisputeRaw) -> DisputeState {
    match d.state.as_str() {
        "open" => DisputeState::Open {
            opened_at_height: d.opened_at_height,
        },
        "resolved_approved" => DisputeState::ResolvedApproved {
            resolved_at_height: d.resolved_at_height,
            approve_bps: d.approve_bps,
            deny_bps: d.deny_bps,
        },
        "resolved_denied" => DisputeState::ResolvedDenied {
            resolved_at_height: d.resolved_at_height,
            approve_bps: d.approve_bps,
            deny_bps: d.deny_bps,
        },
        other => DisputeState::UnknownWire(other.to_string()),
    }
}
