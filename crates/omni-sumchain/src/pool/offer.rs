//! O2 (#104): bonded offers and work-unit acceptance.
//!
//! * **The 35% cap is client-local.** #104 requires the client to refuse an
//!   offer above 35% of the *user-configured* physical capacity, in bytes. The
//!   capacity is never put on chain and never advertised as attested: the op
//!   carries `offered_bytes` only. The comparison is exact
//!   (`offered ≤ ⌊capacity × 7 / 20⌋`), with no floating point and no
//!   overflow for any `u128` capacity.
//! * **Op bytes come from the chain's own carriers.** The frozen C1
//!   `PublishBondedOfferV1` (53 bytes) and `AcceptWorkUnitV1` (129 bytes) are
//!   encoded through `ComputePoolTxData::from_operation`, the same path the
//!   chain's transaction type uses. Bond and reimbursement amounts are not
//!   operands (sum-chain #217 B6/B7), so nothing here needs an unfrozen value.
//! * **The offer's bond id is derived by the chain** from the chain id, the
//!   sender and `offer_seq`; it is not carried and not computed here.

use sumchain_wire::address::Address;
use sumchain_wire::b0::codec::DecodeError;
use sumchain_wire::compute_pool_wire::{
    AcceptWorkUnitV1, ComputePoolOperation, PublishBondedOfferV1, WorkItemRef,
};
use sumchain_wire::transaction::ComputePoolTxData;

/// Numerator of the 35% cap, as the exact ratio 7/20.
pub const OFFER_CAP_NUMERATOR: u128 = 7;
/// Denominator of the 35% cap, as the exact ratio 7/20.
pub const OFFER_CAP_DENOMINATOR: u128 = 20;

/// Why an offer or an acceptance was not built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OfferError {
    #[error(
        "offer of {offered_bytes} bytes is above 35% of the configured physical capacity \
         ({physical_capacity_bytes} bytes; at most {max_offer_bytes} bytes may be offered)"
    )]
    AboveCap {
        offered_bytes: u128,
        physical_capacity_bytes: u128,
        max_offer_bytes: u128,
    },
    #[error("compute-pool op refused by its frozen encoder: {0}")]
    Encode(DecodeError),
}

/// The node operator's configured physical capacity, in bytes.
///
/// Local configuration only: it decides which offers this client is willing to
/// build and is never encoded into any op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OfferCapacity {
    physical_capacity_bytes: u128,
}

impl OfferCapacity {
    pub fn new(physical_capacity_bytes: u128) -> Self {
        Self {
            physical_capacity_bytes,
        }
    }

    pub fn physical_capacity_bytes(&self) -> u128 {
        self.physical_capacity_bytes
    }

    /// `⌊capacity × 7 / 20⌋`, the largest offer the cap admits, computed
    /// without overflow: `capacity = 20q + r`, so the product is `7q + ⌊7r/20⌋`.
    pub fn max_offer_bytes(&self) -> u128 {
        let q = self.physical_capacity_bytes / OFFER_CAP_DENOMINATOR;
        let r = self.physical_capacity_bytes % OFFER_CAP_DENOMINATOR;
        OFFER_CAP_NUMERATOR * q + (OFFER_CAP_NUMERATOR * r) / OFFER_CAP_DENOMINATOR
    }

    /// Whether an offer of `offered_bytes` is at or below 35% of capacity.
    pub fn admits(&self, offered_bytes: u128) -> bool {
        offered_bytes <= self.max_offer_bytes()
    }
}

/// Build the canonical `PublishBondedOfferV1` op, refusing it locally when it
/// is above 35% of `capacity`.
pub fn publish_offer(
    capacity: &OfferCapacity,
    offer_seq: u64,
    offered_bytes: u128,
    payment_addr: Address,
) -> Result<ComputePoolTxData, OfferError> {
    if !capacity.admits(offered_bytes) {
        return Err(OfferError::AboveCap {
            offered_bytes,
            physical_capacity_bytes: capacity.physical_capacity_bytes(),
            max_offer_bytes: capacity.max_offer_bytes(),
        });
    }
    let op = ComputePoolOperation::PublishOffer(PublishBondedOfferV1 {
        offer_seq,
        offered_bytes,
        payment_addr,
    });
    ComputePoolTxData::from_operation(&op).map_err(OfferError::Encode)
}

/// Build the canonical `AcceptWorkUnitV1` op for the work item the chain
/// assigned. The full work-item identity (generation included) is carried, so
/// an acceptance cannot name a stale generation by accident; which offer it
/// accepts is derived by the chain from the assignment, not carried.
pub fn accept_unit(
    work_item: WorkItemRef,
    commit_bond_id: [u8; 32],
    accepted_bytes: u128,
) -> Result<ComputePoolTxData, OfferError> {
    let op = ComputePoolOperation::AcceptUnit(AcceptWorkUnitV1 {
        work_item,
        commit_bond_id,
        accepted_bytes,
    });
    ComputePoolTxData::from_operation(&op).map_err(OfferError::Encode)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u128 = 1_000_000_000;
    const GIB: u128 = 1 << 30;

    fn addr() -> Address {
        Address::new([0x5A; 20])
    }

    /// #104's unit-test fixtures: 16 / 24 / 128 GB and GiB.
    #[test]
    fn the_cap_is_exactly_35_percent_for_the_gb_and_gib_fixtures() {
        let cases = [
            (16 * GB, 5_600_000_000u128),
            (24 * GB, 8_400_000_000),
            (128 * GB, 44_800_000_000),
            // 35% of a GiB count is fractional; the floor is the largest offer.
            (16 * GIB, 6_012_954_214),
            (24 * GIB, 9_019_431_321),
            (128 * GIB, 48_103_633_715),
        ];
        for (capacity, max) in cases {
            let c = OfferCapacity::new(capacity);
            assert_eq!(c.max_offer_bytes(), max, "capacity {capacity}");
            // Cross-check against the plain product, which fits for these.
            assert_eq!(max, capacity * 7 / 20, "capacity {capacity}");
            assert!(c.admits(max));
            assert!(!c.admits(max + 1));
        }
    }

    #[test]
    fn an_offer_at_the_cap_is_built_and_one_byte_more_is_refused() {
        let c = OfferCapacity::new(24 * GIB);
        let max = c.max_offer_bytes();
        let data = publish_offer(&c, 7, max, addr()).unwrap();
        assert_eq!(data.op_bytes.len(), PublishBondedOfferV1::LEN);
        assert_eq!(&data.op_bytes[..7], &PublishBondedOfferV1::MAGIC);

        assert_eq!(
            publish_offer(&c, 7, max + 1, addr()),
            Err(OfferError::AboveCap {
                offered_bytes: max + 1,
                physical_capacity_bytes: 24 * GIB,
                max_offer_bytes: max,
            })
        );
    }

    #[test]
    fn the_capacity_never_reaches_the_op() {
        // Two very different capacities admit the same offer and produce the
        // same bytes: nothing about the physical capacity is encoded.
        let a = publish_offer(&OfferCapacity::new(16 * GB), 1, GB, addr()).unwrap();
        let b = publish_offer(&OfferCapacity::new(128 * GIB), 1, GB, addr()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn the_cap_has_no_overflow_at_the_top_of_the_range() {
        let c = OfferCapacity::new(u128::MAX);
        let max = c.max_offer_bytes();
        // u128::MAX = 20q + 15, so the cap is 7q + ⌊105/20⌋ = 7q + 5.
        assert_eq!(max, 7 * (u128::MAX / 20) + 5);
        assert!(c.admits(max));
        assert!(!c.admits(max + 1));
        assert_eq!(OfferCapacity::new(0).max_offer_bytes(), 0);
        assert_eq!(OfferCapacity::new(2).max_offer_bytes(), 0);
        assert_eq!(OfferCapacity::new(3).max_offer_bytes(), 1);
    }

    #[test]
    fn both_ops_decode_through_the_chains_dispatch() {
        let offer = publish_offer(&OfferCapacity::new(128 * GB), 42, 3 * GB, addr()).unwrap();
        assert_eq!(
            ComputePoolOperation::decode_exact(&offer.op_bytes).unwrap(),
            ComputePoolOperation::PublishOffer(PublishBondedOfferV1 {
                offer_seq: 42,
                offered_bytes: 3 * GB,
                payment_addr: addr(),
            })
        );

        let item = WorkItemRef {
            job_id: [1; 32],
            unit_id: [2; 32],
            generation: 3,
        };
        let accept = accept_unit(item, [4; 32], 5 * GB).unwrap();
        assert_eq!(accept.op_bytes.len(), AcceptWorkUnitV1::LEN);
        assert_eq!(
            ComputePoolOperation::decode_exact(&accept.op_bytes).unwrap(),
            ComputePoolOperation::AcceptUnit(AcceptWorkUnitV1 {
                work_item: item,
                commit_bond_id: [4; 32],
                accepted_bytes: 5 * GB,
            })
        );
    }
}
