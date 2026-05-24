# Stage 11d.2 Benchmark Record — Production Fixed-Point MLP

**Companion to** [`stage11d.2-review-packet-production-fixedpoint-mlp.md`](stage11d.2-review-packet-production-fixedpoint-mlp.md).

This record captures the verify-time / prove-time / fixture-size numbers used to validate the Stage 11d.2 performance gates approved per Stage 11d.2 plan §11 decision #3.

## Configuration

| Item | Value |
|---|---|
| Circuit | `ProductionMlpCircuit` (`16 → 32 → 16 → 8` int16 fixed-point MLP) |
| Halo2 framework | `halo2_proofs 0.3.2` (Pasta IPA; default features off) |
| `HALO2_K` | `11` (2048 rows) — preauthorized ceiling `k ≤ 13`; halt-and-report at `k ≥ 14` (plan §13 OQ9) |
| Prover RNG seed | `*b"OmniNode/Stage11d.2/prover-rngv1"` (byte-stable across runs) |
| Build profile | `release` |
| Host | dev host (Stage 11d.2 PR author's macOS arm64 dev machine) |
| `RUST_MIN_STACK` | `134217728` (128 MB; default 2 MB causes MockProver stack overflow for the wider production circuit) |

## Approved targets

| Target | Threshold |
|---|---|
| Verifier p95 | ≤ 1000 ms |
| Prover wall time | < 60 s release-mode dev host |
| `params.bin` size | < 1 MB (informal; halt-and-report at the OQ9 ceiling) |

## Measured numbers

### Verifier (`tests/halo2_benchmark.rs`, `verify_microbench_100_iterations`, `--ignored`)

`cargo test --release --features verify --test halo2_benchmark -- --ignored --nocapture`

| Metric | Value |
|---|---|
| Iterations | 100 |
| min | 17.7 ms |
| p50 | 17.96 ms |
| p95 | **19.6 ms** |
| max | 20.5 ms |

Well under the 1000 ms ceiling — verifier headroom is ≈ 50× the target.

### Prover (end-to-end via `tools/halo2_production_mlp_regen/`)

`cd tools/halo2_production_mlp_regen && cargo run --release -- regen`

Wall time end-to-end (prove + verify roundtrip + JSON serialization + IO): **≈ 3.0 s**.

`prove_canonical` alone (in the `prove_canonical_produces_proof` unit test, including VK + PK keygen): **≈ 2.3 s**.

Both well under the 60 s ceiling.

### Fixture sizes (committed under `crates/omni-proofs-halo2-production-mlp/fixtures/halo2/`)

| File | Bytes |
|---|---|
| `params.bin` | 131,140 |
| `proof.bin` | 7,744 |
| `proof_artifact.json` | 16,396 |

`params.bin` is 12.5% of the informal 1 MB cap; far from the OQ9 halt-and-report ceiling.

## Reproducing

```bash
# Verifier microbench (100 iters; opt-in via --ignored)
RUST_MIN_STACK=134217728 cargo test --release \
    -p omni-proofs-halo2-production-mlp \
    --features verify \
    --test halo2_benchmark \
    -- --ignored --nocapture

# Prover end-to-end (regen)
cd tools/halo2_production_mlp_regen && cargo run --release -- regen

# Verifier-only test suite (mirrors CI)
RUST_MIN_STACK=67108864 cargo test \
    -p omni-proofs-halo2-production-mlp \
    --features verify
```

## Notes

- `RUST_MIN_STACK` is required because the production circuit's `ProductionMlpCircuit` witness struct is large (`[DenseUnitWitness; 32]` for Layer 1 + magnitude bit arrays for ReLU 1, etc.), and `MockProver` walks the constraint system on a 2 MB-defaulted test thread. The release builds (operator binary + the regen tool) do not need this knob because they don't run `MockProver` and the witness lives on the heap during proving.
- Prover RNG seed is the **only** intentional source of non-determinism; pinning it makes proof bytes reproducible across hosts, which the `prove_canonical_is_byte_deterministic` unit test enforces.
- Fixture bytes are byte-stable across regen runs given the same `halo2_proofs` version. A version bump may shift the bytes; that must be caught by the verifier-only CI job (`stage11d-production-verify-build-test`) and requires an explicit fixture-regen PR.
