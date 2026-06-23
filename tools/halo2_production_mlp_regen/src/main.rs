//! Stage 11d.2 — halo2 production-MLP fixture regen tool.
//!
//! Standalone Cargo package (NOT a workspace member). Generates /
//! verifies the committed fixtures at
//! `crates/omni-proofs-halo2-production-mlp/fixtures/halo2/`:
//!
//!   - `params.bin`          — IPA params for `k = HALO2_K`
//!   - `proof.bin`           — the halo2 proof bytes
//!   - `proof_artifact.json` — full `ProofArtifactBody` JSON
//!
//! Modes:
//!   - `regen`            : run the prover and overwrite all three.
//!   - `verify-only`      : load committed bytes and verify against
//!                          the committed `proof_artifact.json`.
//!                          Used by CI on developer hosts to catch
//!                          drift.
//!   - `generate-corpus`  : emit the per-framework corpus JSONs
//!                          under `tests/fixtures/`.
//!
//! **Off-chain only.** Never touches chain wire / tx /
//! `InferenceAttestationDigest` / RPC / validator-side verification.

use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use omni_proofs_halo2_production_mlp::{
    canonical_evaluate,
    encoding::{encode_canonical_input, encode_canonical_output},
    prover::prove_canonical,
    verifier::{derive_vk_identity_from_params, Halo2ProductionMlpVerifier},
    BACKEND_ID, CANONICAL_INPUT, EXPECTED_PRODUCTION_SPEC_HASH, PRODUCTION_SPEC_NAME,
    PRODUCTION_SPEC_VERSION,
};
use omni_zkml::{
    ModelFormat, ModelFramework, ProofArtifactBody, ProofMetadata, ProofSystem, ProofVerifier,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    /// Re-run the prover and overwrite the committed fixtures.
    Regen,
    /// Load the committed fixtures and verify.
    VerifyOnly,
    /// Generate the per-framework corpus JSONs under
    /// `tests/fixtures/`.
    GenerateCorpus,
}

#[derive(Debug, Parser)]
#[command(name = "halo2_production_mlp_regen")]
struct Args {
    #[arg(value_enum, default_value_t = Mode::VerifyOnly)]
    mode: Mode,
}

fn workspace_root() -> PathBuf {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or(here)
}

fn fixtures_dir() -> PathBuf {
    workspace_root().join("crates/omni-proofs-halo2-production-mlp/fixtures/halo2")
}

fn spec_hash_hex() -> String {
    let mut s = String::with_capacity(64);
    for b in EXPECTED_PRODUCTION_SPEC_HASH {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn hex_blake3(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn build_artifact(
    input_i16: [i16; 16],
    output_i16: [i16; 8],
    proof_bytes: &[u8],
    circuit_id_hex: &str,
    vk_hash_hex: &str,
) -> ProofArtifactBody {
    let spec_hex = spec_hash_hex();
    let input_hash = hex_blake3(&encode_canonical_input(&input_i16));
    let output_hash = hex_blake3(&encode_canonical_output(&output_i16));

    let mut metadata = ProofMetadata::new_stage11a(
        BACKEND_ID.to_string(),
        spec_hex,
        input_hash,
        output_hash,
    );
    metadata.proof_system = Some(ProofSystem::Stage11dProductionFixedPointMlp);
    metadata.model_format = Some(ModelFormat::ProductionFixedPointMlp);
    metadata.model_framework = Some(ModelFramework::FrameworkAgnostic);
    // Production-shape artifact: testnet_or_dev_only is Some(false).
    // Mainnet refusal continues to come from the empty
    // MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES table at layer 6 of
    // check_mainnet_eligible (Stage 11d.1 schema). Allowlist entry
    // for this proof class is a Stage 11d.3 deliverable.
    metadata.testnet_or_dev_only = Some(false);
    // Stage 11d.1 layer-6 eligibility registry key. Pinned in
    // `shared::EXPECTED_CIRCUIT_ID_HEX`; verifier checks for drift.
    metadata.circuit_id_hex = Some(circuit_id_hex.to_string());
    // Audit field — `mainnet_vk_hash` of the canonical VK bytes
    // (criteria §1.7). Recorded on the artifact so an auditor can
    // cross-check against the eventual `AllowlistEntry.verification_key_hash_hex`
    // without re-running keygen_vk locally.
    metadata.verification_key_hex = Some(vk_hash_hex.to_string());
    metadata.public_inputs = Some(serde_json::json!({
        "input":  input_i16.to_vec(),
        "output": output_i16.to_vec(),
    }));

    ProofArtifactBody::from_components(metadata, proof_bytes)
}

fn run_regen() {
    let fixtures = fixtures_dir();
    std::fs::create_dir_all(&fixtures).expect("create fixtures dir");

    let (params_bytes, proof_bytes, output_i16) =
        prove_canonical(CANONICAL_INPUT).expect("prove_canonical");
    assert_eq!(output_i16, canonical_evaluate(CANONICAL_INPUT));

    // Derive the live VK identity hashes from the just-generated
    // params + the production circuit. Print them so the dev can
    // pin them into `crates/omni-proofs-halo2-production-mlp/src/shared.rs`
    // (`EXPECTED_CIRCUIT_ID_HEX` / `EXPECTED_VK_HASH_HEX`). Any
    // drift between the live values and the pinned constants
    // is caught by `verify_vk_identity_matches_pinned_constants`.
    let (circuit_id_hex, vk_hash_hex) =
        derive_vk_identity_from_params(&params_bytes).expect("derive VK identity");
    println!("Derived VK identity:");
    println!("  EXPECTED_CIRCUIT_ID_HEX = \"{circuit_id_hex}\"");
    println!("  EXPECTED_VK_HASH_HEX    = \"{vk_hash_hex}\"");

    let params_path = fixtures.join("params.bin");
    let proof_path = fixtures.join("proof.bin");
    let body_path = fixtures.join("proof_artifact.json");

    std::fs::write(&params_path, &params_bytes).expect("write params.bin");
    std::fs::write(&proof_path, &proof_bytes).expect("write proof.bin");

    let body = build_artifact(
        CANONICAL_INPUT,
        output_i16,
        &proof_bytes,
        &circuit_id_hex,
        &vk_hash_hex,
    );
    let body_json = serde_json::to_string_pretty(&body).expect("serialize body");
    std::fs::write(&body_path, body_json).expect("write proof_artifact.json");

    println!("halo2_production_mlp_regen regen wrote:");
    println!("  {}", params_path.display());
    println!("  {} ({} bytes)", proof_path.display(), proof_bytes.len());
    println!("  {}", body_path.display());
    println!("Sanity-checking the just-written artifact against the verifier...");
    run_verify_only();
}

fn run_verify_only() {
    let fixtures = fixtures_dir();
    let params_path = fixtures.join("params.bin");
    let body_path = fixtures.join("proof_artifact.json");

    let params_bytes = std::fs::read(&params_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", params_path.display()));
    let body_bytes = std::fs::read(&body_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", body_path.display()));
    let body: ProofArtifactBody =
        serde_json::from_slice(&body_bytes).expect("parse proof_artifact.json");

    let verifier = Halo2ProductionMlpVerifier::from_params_bytes(&params_bytes)
        .expect("Halo2ProductionMlpVerifier::from_params_bytes");

    let ok = verifier
        .verify_artifact(&body)
        .expect("verify_artifact must not error");
    assert!(ok, "committed fixture must verify");
    println!("halo2_production_mlp_regen verify-only OK");
    println!("  params.bin           ({} bytes)", params_bytes.len());
    println!(
        "  proof.bin            ({} bytes)",
        body.proof_bytes().expect("decode proof_bytes_hex").len()
    );
    println!("  proof_artifact.json  ({} bytes)", body_bytes.len());
}

fn corpus_inputs() -> Vec<(&'static str, [i16; 16], &'static str)> {
    // Stage 11d.2 — committed test corpus, 16 entries, satisfying
    // the criteria §1.2 Claim C2 floor of `≥ 16 inputs`. Spans
    // declared-domain boundaries: zero, ±extremes, single-hot,
    // alternating-sign patterns, monotonic sequences, all-ones,
    // paired-extremes alternation, and tiny perturbations of the
    // canonical input. All inputs are i16-valued by construction;
    // the chosen weight ranges (W₁/W₂/W₃ ∈ [-8, 8] / [-4, 4] /
    // [-8, 8]; biases ∈ [-64, 64] / [-32, 32] / [-32, 32]) keep
    // `|with_bias| < 2^23` for every i16 input, so no dense-layer
    // output saturates — the architectural invariant Stage 11d.2
    // depends on.
    vec![
        (
            "canonical",
            [-5, 10, 20, -100, 7, -3, 14, 25, -8, 1, 11, -22, 4, -1, 17, 9],
            "Frozen Stage 11d.2 canonical pair (from canonical_spec.json).",
        ),
        (
            "zero_input",
            [0i16; 16],
            "Bias-only path. Useful regression for the dense identity gate \
             when all input cells are zero.",
        ),
        (
            "extreme_pos_input",
            [32767i16; 16],
            "Maximum positive i16 inputs. Stresses dense-layer magnitudes; \
             |with_bias| bounded under 2^23 by the chosen weight ranges.",
        ),
        (
            "extreme_neg_input",
            [-32768i16; 16],
            "Minimum negative i16 inputs. Same bound argument as extreme_pos_input.",
        ),
        (
            "alternating_small",
            [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8],
            "Small alternating signs. Covers sign-bit gadget activations at \
             low magnitudes; no saturation, mixed ReLU branches.",
        ),
        (
            "alternating_large",
            [100, -100, 50, -50, 25, -25, 12, -12, 6, -6, 3, -3, 1, -1, 0, 0],
            "Mixed-magnitude alternating signs. Broad coverage of Layer-1 \
             requantization range.",
        ),
        (
            "single_hot",
            [0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0],
            "Single non-zero input (i=8). Exercises one-hot encoding regression.",
        ),
        (
            "increasing_sequence",
            [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "Monotonic increasing input. Diverse pre-ReLU activations.",
        ),
        (
            "decreasing_sequence",
            [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0, -10, -20, -30, -40, -50],
            "Monotonic decreasing input (mirror of increasing_sequence). \
             Catches index-orientation bugs in dense layouts.",
        ),
        (
            "all_ones",
            [1i16; 16],
            "All ones. Each dense output reduces to `Σ W[i][j] + bias·S`; \
             confirms the bias-application path without input-mixing.",
        ),
        (
            "all_neg_ones",
            [-1i16; 16],
            "All negative ones. Mirror of all_ones; flips the sign bit on \
             the input·W contribution.",
        ),
        (
            "paired_extremes_alternating",
            [32767, -32768, 32767, -32768, 32767, -32768, 32767, -32768,
             32767, -32768, 32767, -32768, 32767, -32768, 32767, -32768],
            "Alternating i16 extrema. Maximum-amplitude oscillation; \
             stresses sign-bit transitions on every input index.",
        ),
        (
            "high_pos_density",
            [16000i16; 16],
            "Half-extreme positive uniform. Confirms behavior at high but \
             not-quite-extreme magnitudes (`Σ input·W` ~ 2.05M < 2^23).",
        ),
        (
            "mixed_negative_majority",
            [-50, -25, -10, -5, -3, -1, 0, 1, 3, 5, 10, 25, 50, 100, 200, -200],
            "Mostly negative with positive tail. Heterogeneous magnitudes; \
             broad coverage of ReLU-1 sign-bit branches.",
        ),
        (
            "canonical_tiny_perturbation",
            [-5, 10, 20, -100, 7, -3, 14, 25, -8, 1, 11, -22, 4, -1, 17, 10],
            "Canonical input with the last value bumped +1. Adjacent-input \
             stability regression — confirms the circuit is sensitive to \
             every input position, not just the magnitudes.",
        ),
        (
            "small_primes",
            [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53],
            "Distinct small primes per input index. Forces every dense-1 \
             output through a unique linear combination — broad arithmetic \
             coverage with no symmetric cancellation.",
        ),
    ]
}

fn encode_input_le(t: [i16; 16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    for v in &t {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_output_le(t: [i16; 8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16);
    for v in &t {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn run_generate_corpus() {
    let spec_hash_hex = spec_hash_hex();

    let entries: Vec<serde_json::Value> = corpus_inputs()
        .into_iter()
        .map(|(label, input, notes)| {
            let output = canonical_evaluate(input);
            serde_json::json!({
                "label": label,
                "input": input.to_vec(),
                "output": output.to_vec(),
                "input_hash": hex_blake3(&encode_input_le(input)),
                "output_hash": hex_blake3(&encode_output_le(output)),
                "notes": notes,
            })
        })
        .collect();

    // Ground-truth corpus (no framework metadata).
    let corpus = serde_json::json!({
        "spec_name": PRODUCTION_SPEC_NAME,
        "spec_version": PRODUCTION_SPEC_VERSION,
        "spec_hash": spec_hash_hex,
        "tensor_encoding": "i16-little-endian",
        "description":
            "Stage 11d.2 production fixed-point MLP test corpus. Each entry's `output` is \
             the pure-Rust canonical_evaluate() applied to the entry's `input`. The \
             cross-framework corpus tests assert that all five framework corpus files \
             (rumus, pytorch, tensorflow, caffe, framework_agnostic) agree with this \
             ground-truth corpus byte-for-byte. Off-chain only — never consumed by chain \
             wire / tx / InferenceAttestationDigest / RPC.",
        "entries": entries.clone(),
    });
    write_corpus_file("corpus.json", &corpus);
    let n = corpus["entries"].as_array().unwrap().len();
    println!(
        "halo2_production_mlp_regen generate-corpus wrote corpus.json ({n} entries, spec_hash {spec_hash_hex})"
    );

    // Framework corpora. Per the four-equal-primary posture, all
    // four frameworks (RUMUS, PyTorch, TensorFlow, Caffe) are
    // committed; but **at Stage 11d.2 none of the production-specific
    // framework exporters have shipped yet** (Stage 11d.2 plan §6
    // decision #8: "prefer new production-specific exporters if
    // adapting existing tools risks touching Stage 11b/11c fixture
    // paths"). To avoid the misleading-provenance bug flagged in the
    // first code review (marking RUMUS/PyTorch/TensorFlow corpus
    // files as `LiveExport` when no framework runtime was actually
    // invoked), the corpus entries are computed by the Rust
    // canonical evaluator AND every framework file is marked
    // `IntendedRepresentation` until the corresponding
    // `tools/halo2_production_mlp_<framework>_export/` tool lands
    // (planned for a Stage 11d.2.x follow-up PR).
    //
    // The framework-agnostic corpus is the honest exception — it
    // *is* the Rust canonical evaluator's output, marked
    // `ManualFixture`. Caffe's `PureNumpyCompatibility` variant
    // would still imply a pure-NumPy exporter ran, which it didn't
    // either at Stage 11d.2; so Caffe is also `IntendedRepresentation`
    // until the production-specific Caffe exporter ships.
    let frameworks = [
        (
            "framework_agnostic_corpus.json",
            "FrameworkAgnostic",
            "canonical-evaluator-v1 (crates/omni-proofs-halo2-production-mlp)",
            "ManualFixture",
            serde_json::json!({"runtime_mode": "rust-canonical-evaluator"}),
            "Framework-neutral ground truth (Rust pure-integer arithmetic in canonical.rs).",
        ),
        (
            "rumus_corpus.json",
            "Rumus",
            "rumus 0.4.0 (crates.io; MIT OR Apache-2.0) — exporter NOT YET SHIPPED",
            "IntendedRepresentation",
            serde_json::json!({
                "runtime_mode": "placeholder-rust-canonical-evaluator-stand-in",
                "exporter_status": "not-yet-shipped",
                "follow_up": "tools/halo2_production_mlp_rumus_export/ (planned, post-Stage-11d.2)",
                "note": "Entries are the Rust canonical evaluator's output verbatim; framework runtime was NOT invoked at Stage 11d.2 fixture-regen time. Replace this file when the RUMUS exporter lands."
            }),
            "Stage 11d.2 placeholder. RUMUS exporter is a planned post-11d.2 follow-up; \
             entries here come from the Rust canonical evaluator, not RUMUS.",
        ),
        (
            "pytorch_corpus.json",
            "PyTorch",
            "torch 2.x — exporter NOT YET SHIPPED",
            "IntendedRepresentation",
            serde_json::json!({
                "runtime_mode": "placeholder-rust-canonical-evaluator-stand-in",
                "exporter_status": "not-yet-shipped",
                "follow_up": "tools/halo2_production_mlp_pytorch_export/ (planned, post-Stage-11d.2)",
                "note": "Entries are the Rust canonical evaluator's output verbatim; framework runtime was NOT invoked at Stage 11d.2 fixture-regen time. Replace this file when the PyTorch exporter lands."
            }),
            "Stage 11d.2 placeholder. PyTorch exporter is a planned post-11d.2 follow-up; \
             entries here come from the Rust canonical evaluator, not PyTorch.",
        ),
        (
            "tensorflow_corpus.json",
            "TensorFlow",
            "tensorflow 2.x — exporter NOT YET SHIPPED",
            "IntendedRepresentation",
            serde_json::json!({
                "runtime_mode": "placeholder-rust-canonical-evaluator-stand-in",
                "exporter_status": "not-yet-shipped",
                "follow_up": "tools/halo2_production_mlp_tensorflow_export/ (planned, post-Stage-11d.2)",
                "note": "Entries are the Rust canonical evaluator's output verbatim; framework runtime was NOT invoked at Stage 11d.2 fixture-regen time. Replace this file when the TensorFlow exporter lands."
            }),
            "Stage 11d.2 placeholder. TensorFlow exporter is a planned post-11d.2 follow-up; \
             entries here come from the Rust canonical evaluator, not TensorFlow.",
        ),
        (
            "caffe_corpus.json",
            "Caffe",
            "Caffe (legacy) — exporter NOT YET SHIPPED",
            "IntendedRepresentation",
            serde_json::json!({
                "runtime_mode": "placeholder-rust-canonical-evaluator-stand-in",
                "exporter_status": "not-yet-shipped",
                "caffe_runtime_present": false,
                "follow_up": "tools/halo2_production_mlp_caffe_export/ (planned, post-Stage-11d.2)",
                "note": "Entries are the Rust canonical evaluator's output verbatim; neither a Caffe runtime nor the pure-NumPy fallback was invoked at Stage 11d.2 fixture-regen time. Replace this file when the Caffe exporter lands."
            }),
            "Stage 11d.2 placeholder. Caffe exporter is a planned post-11d.2 follow-up; \
             entries here come from the Rust canonical evaluator, not Caffe.",
        ),
    ];

    for (filename, framework, version, gen_mode, gen_meta, notes) in frameworks {
        let payload = serde_json::json!({
            "framework": framework,
            "framework_version": version,
            "generation_mode": gen_mode,
            "generator_metadata": gen_meta,
            "spec_name": PRODUCTION_SPEC_NAME,
            "spec_version": PRODUCTION_SPEC_VERSION,
            "spec_hash": spec_hash_hex,
            "tensor_encoding": "i16-little-endian",
            "description": notes,
            "entries": entries.clone(),
        });
        write_corpus_file(filename, &payload);
        println!("  wrote {filename}");
    }
}

fn write_corpus_file(filename: &str, payload: &serde_json::Value) {
    let path = workspace_root()
        .join("crates/omni-proofs-halo2-production-mlp/tests/fixtures")
        .join(filename);
    std::fs::create_dir_all(path.parent().unwrap()).expect("mkdir tests/fixtures");
    std::fs::write(&path, serde_json::to_string_pretty(payload).unwrap() + "\n")
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn main() {
    let args = Args::parse();
    match args.mode {
        Mode::Regen => run_regen(),
        Mode::VerifyOnly => run_verify_only(),
        Mode::GenerateCorpus => run_generate_corpus(),
    }
}
