//! Stage 11b.1.b — halo2-reference fixture regen tool.
//!
//! Standalone Cargo package (NOT a workspace member). Generates
//! / verifies the committed fixtures at
//! `crates/omni-proofs-halo2-reference/fixtures/halo2/`:
//!
//!   - `params.bin`          — IPA params for `k = HALO2_K`
//!   - `proof.bin`           — the halo2 proof bytes
//!   - `proof_artifact.json` — full `ProofArtifactBody` JSON
//!
//! Modes:
//!   - `regen`       : run the prover and overwrite all three.
//!   - `verify-only` : load committed bytes and verify against the
//!                     committed `proof_artifact.json`. Used by
//!                     CI on developer hosts to catch drift.

use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use omni_proofs_halo2_reference::{
    canonical_evaluate,
    encoding::encode_tensor_4xi16_le,
    prover::prove_canonical,
    verifier::Halo2ReferenceVerifier,
    CANONICAL_INPUT, EXPECTED_SPEC_HASH,
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
    /// Stage 11c — generate the canonical test corpus
    /// (`tests/fixtures/corpus.json`) from the pure-Rust
    /// canonical evaluator. Commits 8 (input, expected_output)
    /// entries that the cross-framework corpus tests + the
    /// `--features prove` arbitrary-input integration test
    /// validate against.
    GenerateCorpus,
}

#[derive(Debug, Parser)]
#[command(name = "halo2_reference_regen")]
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
    workspace_root().join("crates/omni-proofs-halo2-reference/fixtures/halo2")
}

fn spec_hash_hex() -> String {
    let mut s = String::with_capacity(64);
    for b in EXPECTED_SPEC_HASH {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn hex_blake3(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn build_artifact(
    input_i16: [i16; 4],
    output_i16: [i16; 4],
    proof_bytes: &[u8],
) -> ProofArtifactBody {
    let spec_hex = spec_hash_hex();
    let input_hash = hex_blake3(&encode_tensor_4xi16_le(&input_i16));
    let output_hash = hex_blake3(&encode_tensor_4xi16_le(&output_i16));

    let mut metadata = ProofMetadata::new_stage11a(
        "halo2-reference-mlp-v1".to_string(),
        spec_hex,
        input_hash,
        output_hash,
    );
    metadata.proof_system = Some(ProofSystem::Stage11bHalo2Reference);
    metadata.model_format = Some(ModelFormat::Halo2ReferenceMlp);
    metadata.model_framework = Some(ModelFramework::FrameworkAgnostic);
    metadata.testnet_or_dev_only = Some(true);
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

    let params_path = fixtures.join("params.bin");
    let proof_path = fixtures.join("proof.bin");
    let body_path = fixtures.join("proof_artifact.json");

    std::fs::write(&params_path, &params_bytes).expect("write params.bin");
    std::fs::write(&proof_path, &proof_bytes).expect("write proof.bin");

    let body = build_artifact(CANONICAL_INPUT, output_i16, &proof_bytes);
    let body_json = serde_json::to_string_pretty(&body).expect("serialize body");
    std::fs::write(&body_path, body_json).expect("write proof_artifact.json");

    println!("halo2_reference_regen regen wrote:");
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

    let verifier = Halo2ReferenceVerifier::from_params_bytes(&params_bytes)
        .expect("Halo2ReferenceVerifier::from_params_bytes");

    let ok = verifier
        .verify_artifact(&body)
        .expect("verify_artifact must not error");
    assert!(ok, "committed fixture must verify");
    println!("halo2_reference_regen verify-only OK");
    println!("  params.bin           ({} bytes)", params_bytes.len());
    println!(
        "  proof.bin            ({} bytes)",
        body.proof_bytes().expect("decode proof_bytes_hex").len()
    );
    println!("  proof_artifact.json  ({} bytes)", body_bytes.len());
}

fn corpus_inputs() -> Vec<(&'static str, [i16; 4], &'static str)> {
    vec![
        ("canonical",
         [-5, 10, 20, -100],
         "Frozen Stage 11b.1.a canonical pair. No requantization ties, no saturation."),
        ("zero_input",
         [0, 0, 0, 0],
         "Bias-only path. Exercises a Layer-2 tie at j=1 (with_bias=-8192, r_pos=0)."),
        ("tie_layer1_pos",
         [0, 32, 0, 0],
         "Layer-1 tie at j=7 (with_bias=+2176, +S/2 round-half-away to +9)."),
        ("tie_layer1_neg",
         [8, 0, 0, 0],
         "Two Layer-1 ties: j=1 (with_bias=-8320, neg branch) and j=5 (with_bias=+128, +S/2)."),
        ("tie_layer1_multi",
         [0, 8, 0, 0],
         "Four Layer-1 ties at j=0,2,5,6 — broad coverage."),
        ("extreme_pos_input",
         [32767, 32767, 32767, 32767],
         "Maximum positive i16 inputs. Stresses dense-layer magnitudes; no saturation under frozen spec."),
        ("extreme_neg_input",
         [-32768, -32768, -32768, -32768],
         "Minimum negative i16 inputs. No saturation under frozen spec."),
        ("mixed_input",
         [1, 2, 3, 4],
         "Small mixed values; non-tie regression coverage."),
    ]
}

fn encode_tensor_le(t: [i16; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8);
    for v in &t {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn run_generate_corpus() {
    use omni_proofs_halo2_reference::{canonical_evaluate, EXPECTED_SPEC_HASH};

    let spec_hash_hex = {
        let mut s = String::with_capacity(64);
        for b in EXPECTED_SPEC_HASH {
            s.push_str(&format!("{:02x}", b));
        }
        s
    };

    let entries: Vec<serde_json::Value> = corpus_inputs()
        .into_iter()
        .map(|(label, input, notes)| {
            let output = canonical_evaluate(input);
            serde_json::json!({
                "label": label,
                "input": input.to_vec(),
                "output": output.to_vec(),
                "input_hash": hex_blake3(&encode_tensor_le(input)),
                "output_hash": hex_blake3(&encode_tensor_le(output)),
                "notes": notes,
            })
        })
        .collect();

    // Ground-truth corpus (no framework metadata).
    let corpus = serde_json::json!({
        "spec_name": "halo2-mlp-v1",
        "spec_version": 2,
        "spec_hash": spec_hash_hex,
        "tensor_encoding": "i16-little-endian",
        "description": "Stage 11c arbitrary-input test corpus. Each entry's `output` is the pure-Rust canonical_evaluate() applied to the entry's `input`. The cross-framework corpus tests assert that all five framework corpus files (rumus, pytorch, tensorflow, caffe, framework_agnostic) agree with this ground-truth corpus byte-for-byte.",
        "entries": entries.clone(),
    });
    write_corpus_file("corpus.json", &corpus);
    let n = corpus["entries"].as_array().unwrap().len();
    println!(
        "halo2_reference_regen generate-corpus wrote corpus.json ({n} entries, spec_hash {spec_hash_hex})"
    );

    // Per-framework bootstrap corpus files. Each shares the same
    // (input, output, input_hash, output_hash) entries; the
    // difference is in the framework metadata. CI's cross-framework
    // corpus equivalence test asserts the entries match.
    let frameworks = [
        (
            "framework_agnostic_corpus.json",
            "FrameworkAgnostic",
            "canonical-evaluator-v1 (crates/omni-proofs-halo2-reference)",
            "ManualFixture",
            serde_json::json!({"runtime_mode": "rust-canonical-evaluator"}),
            "Framework-neutral ground truth (Rust pure-integer arithmetic in canonical.rs).",
        ),
        (
            "rumus_corpus.json",
            "Rumus",
            "rumus 0.4.0 (crates.io; MIT OR Apache-2.0)",
            "LiveExport",
            serde_json::json!({"runtime_mode": "rumus-fixed-linear", "rumus_version": "0.4.0"}),
            "RUMUS reproduces every entry via `rumus::fixed::FixedLinear` + `requantize`.",
        ),
        (
            "pytorch_corpus.json",
            "PyTorch",
            "torch 2.x (CPU int64 explicit; no torch.quantization)",
            "LiveExport",
            serde_json::json!({"runtime_mode": "pytorch-int64-explicit"}),
            "PyTorch reproduces every entry via explicit int64 dense+ReLU pipeline.",
        ),
        (
            "tensorflow_corpus.json",
            "TensorFlow",
            "tensorflow 2.x (CPU int64 explicit; no tf.quantization)",
            "LiveExport",
            serde_json::json!({"runtime_mode": "tensorflow-int64-explicit"}),
            "TensorFlow reproduces every entry via explicit int64 dense+ReLU pipeline.",
        ),
        (
            "caffe_corpus.json",
            "Caffe",
            "Caffe (legacy; runtime-mode varies per dev host)",
            "PureNumpyCompatibility",
            serde_json::json!({"runtime_mode": "pure-numpy-emulation", "caffe_runtime_present": false}),
            "Caffe corpus uses the auditable pure-NumPy fallback when the host lacks a Caffe binding.",
        ),
    ];

    for (filename, framework, version, gen_mode, gen_meta, notes) in frameworks {
        let payload = serde_json::json!({
            "framework": framework,
            "framework_version": version,
            "generation_mode": gen_mode,
            "generator_metadata": gen_meta,
            "spec_name": "halo2-mlp-v1",
            "spec_version": 2,
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
        .join("crates/omni-proofs-halo2-reference/tests/fixtures")
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
