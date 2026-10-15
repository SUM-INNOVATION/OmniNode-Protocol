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
//!
//! ## Not here, and why
//!
//! * **Proving** (`PartialComputeProofV1` production): needs the RISC Zero
//!   prover selected by B0; the on-chain verifier and the
//!   `CommitPartialProofV1` op are sum-chain #131 (C3) and do not exist yet.
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
//! against the chain's frozen official-workload vectors.

pub mod manifest;
pub mod offer;
pub mod tokens;
