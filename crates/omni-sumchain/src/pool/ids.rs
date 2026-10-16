//! O2 (#104): this node's `offer_bond_id`, derived exactly as the chain
//! derives it.
//!
//! **RATIFIED 2026-10-16** (sum-chain G1-1, `docs/design/C1-POOL-IDS.md`):
//!
//! ```text
//! offer_bond_id = BLAKE3("SUMCHAIN/POOL/OFFERBOND/v1\n" ‖ u64_le(chain_id)
//!                        ‖ sender[20] ‖ u64_le(offer_seq))      (63-byte preimage)
//! ```
//!
//! `sender` is the sender of the `PublishBondedOfferV1` transaction, which is
//! the offer's `identity`; `offer_seq` is the op's operand. The op carries no
//! id, so a node never names its offer by anything but this derivation.
//!
//! **A local copy, deliberately.** `sumchain-wire =0.4.0` (crates.io) predates
//! the ratification and has no derivation, and a path, git or patch override
//! is not allowed in this release manifest. Once a published `sumchain-wire`
//! carries `compute_pool_graph::offer_bond_id`, use that instead. The KATs
//! below are the chain's own (`crates/sumchain-wire/tests/
//! compute_pool_ids_golden.rs`), so the two cannot drift silently.
//!
//! Only `offer_bond_id` is needed client-side. `reservation_id` and
//! `entitlement_id` are chain-internal, and their dependent state (the
//! reservation lifecycle, the reassignment counter, amounts) is not ratified.

use sumchain_wire::address::Address;

/// `offer_bond_id` domain tag (27 bytes, `\n`-terminated).
pub const POOL_OFFERBOND_TAG: &[u8] = b"SUMCHAIN/POOL/OFFERBOND/v1\n";

/// `offer_bond_id(chain_id, sender, offer_seq)`, as the chain derives it.
pub fn offer_bond_id(chain_id: u64, sender: &Address, offer_seq: u64) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(POOL_OFFERBOND_TAG);
    h.update(&chain_id.to_le_bytes());
    h.update(sender.as_bytes());
    h.update(&offer_seq.to_le_bytes());
    *h.finalize().as_bytes()
}

/// This node's offer, named the way the chain names it: the chain id, the
/// publishing identity (the transaction sender) and the offer's `offer_seq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnOffer {
    pub chain_id: u64,
    pub identity: Address,
    pub offer_seq: u64,
}

impl OwnOffer {
    /// The offer's id, derived; never caller-supplied.
    pub fn offer_bond_id(&self) -> [u8; 32] {
        offer_bond_id(self.chain_id, &self.identity, self.offer_seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The chain's ratified vectors (sum-chain `compute_pool_ids_golden.rs`).
    const OFFER_BASE_PREIMAGE_HEX: &str = "53554d434841494e2f504f4f4c2f4f46464552424f4e442f76310a01000000000000000102030405060708090a0b0c0d0e0f10111213140700000000000000";
    const OFFER_BASE_HEX: &str = "98fc8ba3a47488bc5c3941691f57f4ea7915086d4057ade85aa8debea7a1a4ea";
    const OFFER_CHAIN_2_PREIMAGE_HEX: &str = "53554d434841494e2f504f4f4c2f4f46464552424f4e442f76310a02000000000000000102030405060708090a0b0c0d0e0f10111213140700000000000000";
    const OFFER_CHAIN_2_HEX: &str =
        "85181c227b1f434961fdbdb1f9a3050871b04c3592149c7c7175d09245d78851";
    const OFFER_BYTE_ORDER_PREIMAGE_HEX: &str = "53554d434841494e2f504f4f4c2f4f46464552424f4e442f76310a0102030405060708ffffffffffffffffffffffffffffffffffffffff8877665544332211";
    const OFFER_BYTE_ORDER_HEX: &str =
        "5826d4c3a89fd1374b06f69b10a9531300786cee60f07a2e73ad1e815cde071e";

    fn kat_sender() -> Address {
        let mut a = [0u8; 20];
        for (i, b) in a.iter_mut().enumerate() {
            *b = (i + 1) as u8;
        }
        Address::new(a)
    }

    fn preimage(chain_id: u64, sender: &Address, offer_seq: u64) -> Vec<u8> {
        let mut p = POOL_OFFERBOND_TAG.to_vec();
        p.extend_from_slice(&chain_id.to_le_bytes());
        p.extend_from_slice(sender.as_bytes());
        p.extend_from_slice(&offer_seq.to_le_bytes());
        p
    }

    fn pin(chain_id: u64, sender: Address, offer_seq: u64, pre_hex: &str, out_hex: &str) {
        let p = preimage(chain_id, &sender, offer_seq);
        assert_eq!(p.len(), 63);
        assert_eq!(hex(&p), pre_hex, "preimage layout");
        assert_eq!(
            hex(blake3::hash(&p).as_bytes()),
            out_hex,
            "BLAKE3 of the preimage"
        );
        assert_eq!(hex(&offer_bond_id(chain_id, &sender, offer_seq)), out_hex);
        let own = OwnOffer {
            chain_id,
            identity: sender,
            offer_seq,
        };
        assert_eq!(hex(&own.offer_bond_id()), out_hex);
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn tag_is_the_ratified_bytes() {
        assert_eq!(
            hex(POOL_OFFERBOND_TAG),
            "53554d434841494e2f504f4f4c2f4f46464552424f4e442f76310a"
        );
        assert_eq!(POOL_OFFERBOND_TAG.len(), 27);
    }

    #[test]
    fn kat_base() {
        pin(1, kat_sender(), 7, OFFER_BASE_PREIMAGE_HEX, OFFER_BASE_HEX);
    }

    #[test]
    fn kat_chain_id_2() {
        pin(
            2,
            kat_sender(),
            7,
            OFFER_CHAIN_2_PREIMAGE_HEX,
            OFFER_CHAIN_2_HEX,
        );
    }

    #[test]
    fn kat_byte_order() {
        pin(
            0x0807_0605_0403_0201,
            Address::new([0xFF; 20]),
            0x1122_3344_5566_7788,
            OFFER_BYTE_ORDER_PREIMAGE_HEX,
            OFFER_BYTE_ORDER_HEX,
        );
    }
}
