# `tools/halo2_production_mlp_regen` — Stage 11d.2 production-MLP fixture regen tool

Standalone Cargo package (not a workspace member) that generates / verifies the committed halo2 production-MLP proof fixtures at `crates/omni-proofs-halo2-production-mlp/fixtures/halo2/`, plus the 16-entry cross-framework corpus under `crates/omni-proofs-halo2-production-mlp/tests/fixtures/`.

## Why standalone

Mirrors the `tools/halo2_reference_regen/` and `tools/rumus_export/` pattern: this package depends on `omni-proofs-halo2-production-mlp` with the `prove` feature (which transitively pulls `halo2_proofs`), and lives outside the OmniNode workspace so `cargo build -p omni-node` cannot reach it. The root `Cargo.toml` declares `exclude = ["tools"]` to enforce this.

## Usage

```bash
# Verify the committed fixtures
cd tools/halo2_production_mlp_regen
cargo run --release verify-only

# Regenerate (overwrites committed bytes — only do this when the
# canonical spec, the circuit, halo2_proofs version, or HALO2_K
# changes; the verifier's drift check refuses to load a stale
# params.bin against new EXPECTED_* constants)
cargo run --release regen

# Regenerate the 16-entry cross-framework corpus + the four
# framework placeholder corpora (rumus / pytorch / tensorflow /
# caffe). Currently all entries are computed by the Rust canonical
# evaluator; per Stage 11d.2 plan §6 decision #8, the production-
# specific framework exporters are a planned post-11d.2 follow-up.
cargo run --release generate-corpus
```

Fixtures written by `regen`:
- `crates/omni-proofs-halo2-production-mlp/fixtures/halo2/params.bin`
- `crates/omni-proofs-halo2-production-mlp/fixtures/halo2/proof.bin`
- `crates/omni-proofs-halo2-production-mlp/fixtures/halo2/proof_artifact.json`

The artifact's metadata carries `circuit_id_hex` + `verification_key_hex` (the `mainnet_vk_hash` audit value); the verifier enforces drift against `shared::EXPECTED_CIRCUIT_ID_HEX` / `shared::EXPECTED_VK_HASH_HEX`. A `halo2_proofs` version bump or any circuit edit will shift these — `regen` prints the live values; paste them back into `crates/omni-proofs-halo2-production-mlp/src/shared.rs` and re-run.

The `params.bin` is deterministic from `HALO2_K` alone (IPA, no trusted setup). The `proof.bin` is deterministic given the RNG seed pinned in `crates/omni-proofs-halo2-production-mlp/src/prover.rs::PROVER_RNG_SEED` plus the `halo2_proofs` version.

## CI

This tool is **not** invoked by GitHub CI. The crate-side integration tests in `crates/omni-proofs-halo2-production-mlp/tests/halo2_proof_verifies.rs` (verifier construction + identity drift + committed-artifact verify) and `tests/cross_framework_corpus.rs` (16-entry corpus equivalence across the five corpus files) exercise the committed fixtures; those are enough for the verifier-only `stage11d-production-verify-build-test` CI gate.
