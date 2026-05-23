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

fn main() {
    let args = Args::parse();
    match args.mode {
        Mode::Regen => run_regen(),
        Mode::VerifyOnly => run_verify_only(),
    }
}
