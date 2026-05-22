# omni-proofs-halo2-reference

**Stage 11b.1.a — bounded multi-framework halo2 reference scaffold.**

This crate ships the pure-Rust **architectural foundation** for the Stage 11b.1 halo2 reference backend. It is not a production zkML system; it is a deliberately bounded scaffold whose job is to:

1. Define a **canonical quantized MLP** (4 → 8 → 4 with ReLU, int16 fixed-point) that every supported ML framework can reproduce.
2. Ship a **pure-Rust canonical evaluator** as the single source of truth for the arithmetic.
3. Ship the **`FrameworkManifest` schema** and committed fixture manifests for RUMUS / PyTorch / TensorFlow / Caffe / FrameworkAgnostic, proving cross-framework equivalence at test time without invoking any framework runtime.

**What does NOT ship in Stage 11b.1.a:**
- No halo2 circuit, no halo2 dependency, no SNARK prover.
- No `ProofBackend` / `ProofVerifier` implementations.
- No operator-binary dispatch arm.
- No framework runtime in the operator (PyTorch, TensorFlow, Caffe, RUMUS are not pulled in anywhere).

Stage 11b.1.b adds the halo2 circuit, the `Halo2ReferenceVerifier` (overriding `omni_zkml::ProofVerifier::verify_artifact`), the `omni-node` opt-in feature, the verifier-only CI gate, and the manual proof-regen workflow.

## Crate layout

```
src/
├── lib.rs       — module declarations + re-exports
├── shared.rs    — constants + EXPECTED_SPEC_HASH (build.rs-emitted)
├── canonical.rs — canonical_evaluate() — single source of truth
├── encoding.rs  — canonical byte encoding for i16 tensors
└── manifest.rs  — FrameworkManifest schema

tests/
├── cross_framework_equivalence.rs — load all 5 manifests + assert canonical match
└── fixtures/
    ├── framework_agnostic_manifest.json
    ├── pytorch_manifest.json
    ├── tensorflow_manifest.json
    ├── caffe_manifest.json
    └── rumus_manifest.json

assets/
└── canonical_spec.json — frozen weights/biases/architecture/quantization

build.rs — compute BLAKE3 of canonical_spec.json at compile time
```

## Canonical numeric contract (frozen in `assets/canonical_spec.json`)

- **Architecture:** Dense(4 → 8) → ReLU → Dense(8 → 4). No output activation.
- **Dtype:** signed 16-bit integers throughout (`i16`).
- **Scale:** `S = 256 = 2^8`. Logical real `r` maps to `q = round_half_to_even(r * 256)` in `[-32768, 32767]`.
- **Dense layer arithmetic per output `j`:**
  1. `acc_i32 = Σ_i (input[i] as i32) * (W[i][j] as i32)`
  2. `scaled_i32 = acc_i32 / 256`  *(truncate-toward-zero)*
  3. `with_bias_i32 = scaled_i32 + (bias[j] as i32)`
  4. `output[j] = saturate_to_i16(with_bias_i32)`
- **ReLU:** `max(0i16, x)`.
- **Tensor encoding:** `[i16; 4]` → 8 bytes, little-endian, no header.

The pure-Rust `canonical_evaluate(input: [i16; 4]) -> [i16; 4]` in `src/canonical.rs` is the **single source of truth**. Frameworks reproduce; framework-runtime never executes inside OmniNode.

## Frozen canonical evaluation

| Field | Value |
|---|---|
| `input` | `[-5, 10, 20, -100]` |
| `hidden (after ReLU)` | `[58, 0, 0, 0, 43, 0, 0, 7]` |
| `output` | `[32, -32, 17, 8]` |

The `canonical_invariant_holds` test asserts `canonical_evaluate(input) == output`. The `every_framework_fixture_matches_canonical_evaluator` integration test asserts the same equivalence for each of the five framework manifests.

## RUMUS status (deferred)

RUMUS does not currently expose a deterministic CPU fixed-point integer-dense path. The committed `tests/fixtures/rumus_manifest.json` carries `generation_mode: "IntendedRepresentation"` and an explanatory `notes` field. The cross-framework test validates it against the canonical evaluator the same way it validates the other frameworks — the canonical evaluator is the single source of truth, and RUMUS will be validated against it once a live runtime path exists. **Do not interpret the RUMUS manifest as evidence that RUMUS has reproduced the canonical model today; it has not.**

A future PR that upgrades RUMUS to `LiveExport` mode must wire an actual runtime regen path; the `rumus_manifest_is_marked_intended_representation_until_runtime_lands` test pins this contract.

## Mainnet posture

Stage 11b.1 artifacts (produced by Stage 11b.1.b's halo2 circuit) carry `testnet_or_dev_only: Some(true)`. Three independent `omni_zkml::check_mainnet_eligible` refusal layers fire on mainnet:

1. **Layer 1** — `testnet_or_dev_only: Some(true)` → `TestnetOrDevOnly`.
2. **Layer 3** — `proof_system: Stage11bHalo2Reference` → `BoundedReference`.
3. **Layer 6** — `MAINNET_APPROVED_PROOF_SYSTEMS` is empty → `NotInMainnetAllowlist`.

**`MAINNET_APPROVED_PROOF_SYSTEMS` remains empty at end of Stage 11b.1 (a + b).** Mainnet eligibility is a Stage 11c+ deliverable with chain-team review.

## License

MIT OR Apache-2.0 (workspace inherited).
