//! O2 (#104): accepting an assigned unit, from the chain's own pool reads.
//!
//! [`build_accept`] turns a [`PoolUnit`] and this node's [`PoolOffer`] (from
//! [`super::reads`]) into the canonical `AcceptWorkUnitV1` op, and refuses
//! every case the chain's C1 model already refuses. Each rule is the chain's,
//! from sum-chain `crates/state/src/compute_pool.rs`:
//!
//! | refusal | chain rule |
//! |---|---|
//! | the unit is not held | no unit, nothing to accept (and on every network today the pool gate is closed, so no unit is held) |
//! | already accepted | the accepted leaf is write-once (`accept_leaf` → `LeafAlreadyAccepted`) |
//! | not `assigned` | `UnitState::transition` allows `Accepted` only from `Assigned` |
//! | assigned to another offer | the accept's offer is derived from the assignment, not carried (`AcceptWorkUnitV1` docs, owner ruling 2026-09-02) |
//! | the offer row is not this node's | the offer id is derived from the chain id, the sender and `offer_seq` (ratified 2026-10-16, G1-1), so the row under this node's derived id must name the same identity and `offer_seq` |
//! | over the offer's capacity | `reserve_capacity` / `Reservation::try_reserve`: `reserved + req ≤ offered`, checked, with a missing reservation starting at `reserved = 0` over the offer's `offered_bytes` |
//!
//! This node's `offer_bond_id` is never supplied: it is derived from its
//! [`OwnOffer`] (chain id, identity, `offer_seq`) exactly as the chain derives
//! it ([`super::ids`]).
//!
//! **Still caller-supplied, because nothing on the chain settles them:**
//! * `commit_bond_id`: no C1 operation mints commit bonds, and `B_commit` is
//!   an unfrozen governance value;
//! * `accepted_bytes`: no read exposes a unit's reservation requirement.
//!
//! Nothing here submits. The chain has no executor for any C1 operation, and
//! its mempool refuses ComputePool transactions while the gate is closed.

use sumchain_wire::address::Address;
use sumchain_wire::transaction::ComputePoolTxData;

use super::ids::OwnOffer;
use super::offer::{accept_unit, OfferError};
use super::reads::{hex32, PoolOffer, PoolUnit, UnitState};

/// Why an acceptance was not built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AcceptRefusal {
    #[error("the chain holds no such unit")]
    UnitNotHeld,
    #[error("the chain holds no offer {0}")]
    OfferNotHeld(String),
    #[error("the offer read is for {read}, not this node's offer {own}")]
    OfferMismatch { own: String, read: String },
    #[error("the offer row under this node's derived id names {field} {read}, not {own}")]
    OfferRowMismatch {
        field: &'static str,
        own: String,
        read: String,
    },
    #[error("the unit's current work item is already accepted (the accepted leaf is write-once)")]
    AlreadyAccepted,
    #[error("the unit is {0:?}; only an assigned unit can be accepted")]
    NotAssigned(UnitState),
    #[error("the unit is assigned but its assignment row is not held")]
    AssignmentMissing,
    #[error("the unit is assigned to offer {winner}, not this node's offer {own}")]
    AssignedToAnotherOffer { own: String, winner: String },
    #[error(
        "accepting {requested} bytes would reserve {total} of the offer's {offered} \
         (already reserved: {reserved})"
    )]
    CapacityExceeded {
        offered: u128,
        reserved: u128,
        requested: u128,
        total: u128,
    },
    #[error("reserving {requested} bytes over {reserved} overflows u128")]
    Overflow { reserved: u128, requested: u128 },
    #[error(transparent)]
    Encode(#[from] OfferError),
}

/// Build the `AcceptWorkUnitV1` op for `unit`, as this node's offer `own`,
/// or refuse with the reason the chain would.
///
/// `unit` and `own_offer` are the reads' answers (`None` = not held). The
/// offer's id is derived from `own`, never supplied. The work item, generation
/// included, comes from the unit read, so an acceptance cannot name a stale
/// generation.
pub fn build_accept(
    unit: Option<&PoolUnit>,
    own_offer: Option<&PoolOffer>,
    own: &OwnOffer,
    commit_bond_id: [u8; 32],
    accepted_bytes: u128,
) -> Result<ComputePoolTxData, AcceptRefusal> {
    let own_offer_bond_id = &own.offer_bond_id();
    let unit = unit.ok_or(AcceptRefusal::UnitNotHeld)?;
    if unit.accepted_leaf.is_some() {
        return Err(AcceptRefusal::AlreadyAccepted);
    }
    if unit.state != UnitState::Assigned {
        return Err(AcceptRefusal::NotAssigned(unit.state));
    }
    let assignment = unit
        .assignment
        .as_ref()
        .ok_or(AcceptRefusal::AssignmentMissing)?;
    if &assignment.winner_offer_bond_id != own_offer_bond_id {
        return Err(AcceptRefusal::AssignedToAnotherOffer {
            own: hex32(own_offer_bond_id),
            winner: hex32(&assignment.winner_offer_bond_id),
        });
    }
    let offer = own_offer.ok_or_else(|| AcceptRefusal::OfferNotHeld(hex32(own_offer_bond_id)))?;
    if &offer.offer_bond_id != own_offer_bond_id {
        return Err(AcceptRefusal::OfferMismatch {
            own: hex32(own_offer_bond_id),
            read: hex32(&offer.offer_bond_id),
        });
    }
    // The id binds identity and `offer_seq`; a row under it that names others
    // is not this node's offer, however it came to be stored there.
    if offer.identity != own.identity {
        return Err(AcceptRefusal::OfferRowMismatch {
            field: "identity",
            own: own.identity.to_base58(),
            read: offer.identity.to_base58(),
        });
    }
    if offer.offer_seq != own.offer_seq {
        return Err(AcceptRefusal::OfferRowMismatch {
            field: "offer_seq",
            own: own.offer_seq.to_string(),
            read: offer.offer_seq.to_string(),
        });
    }
    let (offered, reserved) = match offer.reservation {
        Some(r) => (r.offered, r.reserved),
        None => (offer.offered_bytes, 0),
    };
    let total = reserved
        .checked_add(accepted_bytes)
        .ok_or(AcceptRefusal::Overflow {
            reserved,
            requested: accepted_bytes,
        })?;
    if total > offered {
        return Err(AcceptRefusal::CapacityExceeded {
            offered,
            reserved,
            requested: accepted_bytes,
            total,
        });
    }
    Ok(accept_unit(
        unit.work_item(),
        commit_bond_id,
        accepted_bytes,
    )?)
}

/// Why a new active offer would be refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PublishRefusal {
    #[error(
        "identity already has active offer {0}; the chain allows one active offer per identity"
    )]
    ActiveOfferExists(String),
    #[error("the offer read belongs to identity {read}, not this node's {own}")]
    NotOwnOffer { own: String, read: String },
}

/// The chain allows one active offer per identity (`publish_offer` →
/// `DuplicateActiveOffer`): refuse to publish while `current`, this node's
/// most recent offer as read by id, is still active. `None` (not held) admits.
///
/// Which `offer_seq` the next offer must use is not settled on the chain (the
/// model does not check it), so it stays the caller's.
pub fn ensure_no_active_offer(
    own_identity: &Address,
    current: Option<&PoolOffer>,
) -> Result<(), PublishRefusal> {
    match current {
        None => Ok(()),
        Some(o) if &o.identity != own_identity => Err(PublishRefusal::NotOwnOffer {
            own: own_identity.to_base58(),
            read: o.identity.to_base58(),
        }),
        Some(o) if o.active => Err(PublishRefusal::ActiveOfferExists(hex32(&o.offer_bond_id))),
        Some(_) => Ok(()),
    }
}
