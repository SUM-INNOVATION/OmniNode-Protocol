//! Client side of the dormant ComputePool, against the frozen chain wire.
//!
//! Issues #104 (O2), #105 (O3) and #106 (O4). Only the parts whose rules are
//! already fixed by merged chain artifacts live here:
//!
//! | module | issue | what it does | fixed by |
//! |---|---|---|---|
//! | [`offer`] | O2 | the client-local 35% offer cap; `PublishBondedOfferV1` / `AcceptWorkUnitV1` op bytes | #104 scope; the C1 op carriers (`compute_pool_wire`, sum-chain #130 / #217) |
//! | [`manifest`] | O2, O4 | a layer-group unit's one output manifest; root-checked verification of retrieved slot objects | the B0 manifest / object-commitment codecs and statement binding (sum-chain #123 B0-FINAL) |
//! | [`tokens`] | O3 | the SelectToken result chain: range and EOS checks, stop rule, EOS-excluded output count, token-sequence commitments | the B0 SelectToken statement and `MAX_OUTPUT_TOKENS` |
//! | [`reads`] | O2 | typed, fail-closed `pool_getUnit` / `pool_getOffer` reads of committed C1 rows | sum-chain's read-only pool RPCs (#130, server at `90c49da1`) |
//! | [`accept`] | O2 | the accept path: from a unit read and this node's offer read to `AcceptWorkUnitV1`, refusing what the chain's C1 model refuses; the one-active-offer precheck | the C1 model's transition, write-once leaf and capacity rules |
//! | [`ids`] | O2 | this node's `offer_bond_id`, derived from the chain id, its identity and `offer_seq` | the ratified derivation (sum-chain G1-1, 2026-10-16) |
//!
//! ## Not here, and why
//!
//! * **Proving** (`PartialComputeProofV1` production): needs the RISC Zero
//!   prover selected by B0; the on-chain verifier and the
//!   `CommitPartialProofV1` op are sum-chain #131 (C3) and do not exist yet.
//! * **A live pool.** The reads return `None` on every network today: genesis
//!   refuses the pool gate and no C1 operation (ordinal 27) has an executor
//!   (sum-chain #130 / #217), so no block writes pool state. This node's
//!   `offer_bond_id` is derived ([`ids`]); its `commit_bond_id` and the
//!   `accepted_bytes` stay caller-supplied (see [`accept`]).
//! * **Observation** (`ObservedRetrievalV1` / `AcknowledgeInputRetrievedV1`)
//!   and the retention read (`pool_getRetentionState`) / output read
//!   (`pool_getJobOutput`): not on the chain (sum-chain #133 C2, #129).
//!   [`manifest::verify_retrieved_output`] is the client-side precondition
//!   those ops will need: observe only on a root match.
//! * **Economic and policy values** (`B_offer`, `C_layer`/`C_tok`/`C_sel`/
//!   `C_emit`, `D_avail`/`D_ack`/`D_final`, `max_reassignments_per_file`, …):
//!   excluded from B0-FINAL by owner ruling and not yet frozen by their owner
//!   issues. Nothing here takes or assumes a value for any of them; the op
//!   carriers carry no amounts.
//! * **Tokenizer**: no tokenizer is registered (sum-chain #128 C5), and the
//!   wire reserves and rejects the `Tokenizer` object kind. [`tokens`] works on
//!   token ids only.
//! * **Submission**: the chain's mempool refuses ComputePool transactions
//!   while the pool gate is closed, so this module only builds op bytes.
//!
//! Everything compiles against `sumchain-wire =0.4.0` from crates.io, whose
//! `b0`, `compute_pool_wire` and `ComputePoolTxData` are byte-identical to
//! sum-chain main; `tests/pool_client_chain_vectors.rs` checks this module
//! against the chain's frozen official-workload vectors. The one exception is
//! [`ids::offer_bond_id`], ratified after 0.4.0: a local copy pinned by the
//! chain's own KATs until a published `sumchain-wire` carries it.

pub mod accept;
pub mod ids;
pub mod manifest;
pub mod offer;
pub mod reads;
pub mod tokens;
