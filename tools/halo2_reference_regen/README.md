# `tools/halo2_reference_regen` — Stage 11b.1.b fixture regen tool

Standalone Cargo package (not a workspace member) that generates / verifies the committed halo2-reference proof fixtures at `crates/omni-proofs-halo2-reference/fixtures/halo2/`.

## Why standalone

Mirrors the `tools/rumus_export/` pattern from Stage 11b.1.a: this package depends on `omni-proofs-halo2-reference` with the `prove` feature (which transitively pulls `halo2_proofs`), and lives outside the OmniNode workspace so `cargo build -p omni-node` cannot reach it. The root `Cargo.toml` declares `exclude = ["tools"]` to enforce this.

## Usage

```bash
# Verify the committed fixtures
cd tools/halo2_reference_regen
cargo run --release verify-only

# Regenerate (overwrites committed bytes — only do this when the
# canonical spec, the circuit, or halo2_proofs version changes)
cargo run --release regen
```

Fixtures written:
- `crates/omni-proofs-halo2-reference/fixtures/halo2/params.bin`
- `crates/omni-proofs-halo2-reference/fixtures/halo2/proof.bin`
- `crates/omni-proofs-halo2-reference/fixtures/halo2/proof_artifact.json`

The `params.bin` is deterministic from `HALO2_K` alone (IPA, no trusted setup). The `proof.bin` is deterministic given the RNG seed pinned in `src/prover.rs::PROVER_RNG_SEED` plus the halo2_proofs version. A halo2_proofs version bump can shift the proof bytes; if so, document the change in the PR description and re-run `regen`.

## Determinism guard

The regen tool's `verify-only` mode is the byte-stability check: if the committed `params.bin` doesn't match what `Params::new(HALO2_K)` produces today, or if the committed `proof.bin` doesn't verify, the tool errors loudly. Run it from CI on the developer host if you want a hard determinism gate; for now it's invoked manually pre-PR.

## CI

This tool is **not** invoked by GitHub CI. The crate-side integration test in `crates/omni-proofs-halo2-reference/tests/halo2_proof_verifies.rs` exercises the verifier against the committed bytes; that's enough for the verifier-only CI gate.
