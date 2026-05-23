# `tools/rumus_export` — RUMUS canonical-spec exporter

Standalone Cargo package (not a workspace member) that produces / verifies the committed `crates/omni-proofs-halo2-reference/tests/fixtures/rumus_manifest.json` fixture by running the framework-neutral canonical spec through `rumus = "0.4.0"`'s deterministic CPU `FixedI16` path.

## Why standalone

This package lives outside the OmniNode workspace so `cargo build -p omni-node` cannot transitively pull `rumus` into the operator binary. The root `Cargo.toml` declares `exclude = ["tools"]` to enforce this.

## Crate verification (one-time)

- **Version:** `rumus = "0.4.0"`
- **License:** `MIT OR Apache-2.0` (compatible with the workspace)
- **Default features:** `[]` (CPU-only; no GPU/wgpu pulled by default)
- **Required APIs:** `rumus::fixed::FixedLinear`, `rumus::fixed::requantize`, `rumus::fixed::relu`, `rumus::tensor::Tensor::{from_i16_fixed, fixed_i16_data}`

The arithmetic contract RUMUS 0.4.0's `fixed/` module implements is **bit-identical** to the Stage 11b.1.a canonical contract: `i16 × i16` → `i64` accumulation, bias promoted to widened scale² domain via `<<scale_log2` BEFORE saturation, round-half-away-from-zero requantization, saturate to `i16`, ReLU as `max(x, 0)`.

## Usage

```bash
# Verify the committed manifest matches what RUMUS produces today.
cd tools/rumus_export
cargo run --release -- verify-only

# Regenerate the committed manifest (developer-host only).
cargo run --release -- regen
```

Both modes read `crates/omni-proofs-halo2-reference/assets/canonical_spec.json` and write/check `crates/omni-proofs-halo2-reference/tests/fixtures/rumus_manifest.json`. Pass `--spec` / `--manifest` to override paths.

## What happens on a halt-and-report

If `rumus = "0.4.0"` ever regresses such that:
- it is no longer published on crates.io with a compatible license, or
- `rumus::fixed` no longer exposes `FixedLinear`, or
- the arithmetic semantics change in a way that no longer matches the canonical spec,

then this exporter will fail (compile error or runtime mismatch). **Do not silently downgrade the RUMUS manifest to `IntendedRepresentation`** — surface the blocker and update the canonical spec or pinned RUMUS version explicitly.

## CI

This package is **not** built or run by CI. The cross-framework equivalence test in `crates/omni-proofs-halo2-reference` validates the committed manifest using pure Rust, with no `rumus` dependency.
