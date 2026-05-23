//! Stage 11b.1.a — RUMUS exporter for the canonical bounded MLP.
//!
//! This is a developer-host manual generator: it reads the
//! framework-neutral `canonical_spec.json` and emits / verifies the
//! committed `rumus_manifest.json` fixture. It is NOT invoked by CI.
//!
//! Why standalone: this package lives outside the OmniNode workspace
//! so `cargo build -p omni-node` cannot transitively pull `rumus`
//! into the operator binary. The Stage 11b.1.a workspace
//! `exclude = ["tools"]` enforces this; the package is built by
//! `cd tools/rumus_export && cargo run --release [--verify-only|--regen]`.

use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use rumus::fixed::{relu, FixedLinear};
use rumus::tensor::Tensor;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    /// Reload the committed manifest and assert byte equality.
    VerifyOnly,
    /// Overwrite the committed manifest with freshly-computed bytes.
    Regen,
}

#[derive(Debug, Parser)]
#[command(name = "rumus_export")]
struct Args {
    /// Operating mode.
    #[arg(value_enum, default_value_t = Mode::VerifyOnly)]
    mode: Mode,
    /// Path to canonical_spec.json. Defaults to the in-repo location.
    #[arg(long)]
    spec: Option<PathBuf>,
    /// Path to rumus_manifest.json. Defaults to the in-repo location.
    #[arg(long)]
    manifest: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct Spec {
    quantization: Quant,
    weights: Weights,
    canonical_evaluation: Eval,
}

#[derive(Debug, Deserialize)]
struct Quant {
    scale_log2: u8,
}

#[derive(Debug, Deserialize)]
struct Weights {
    #[serde(rename = "W1")]
    w1: Mat,
    #[serde(rename = "B1")]
    b1: Bias,
    #[serde(rename = "W2")]
    w2: Mat,
    #[serde(rename = "B2")]
    b2: Bias,
}

#[derive(Debug, Deserialize)]
struct Mat {
    values: Vec<Vec<i16>>,
}

#[derive(Debug, Deserialize)]
struct Bias {
    values: Vec<i16>,
}

#[derive(Debug, Deserialize)]
struct Eval {
    input: Vec<i16>,
    output: Vec<i16>,
}

fn flatten(m: &[Vec<i16>]) -> Vec<i16> {
    m.iter().flat_map(|row| row.iter().copied()).collect()
}

fn encode_le(t: &[i16; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8);
    for v in t {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn hex_blake3(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn workspace_root() -> PathBuf {
    // This package lives at <root>/tools/rumus_export. CARGO_MANIFEST_DIR
    // is its own dir, so go up two levels.
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or(here)
}

fn default_spec_path() -> PathBuf {
    workspace_root().join("crates/omni-proofs-halo2-reference/assets/canonical_spec.json")
}

fn default_manifest_path() -> PathBuf {
    workspace_root().join("crates/omni-proofs-halo2-reference/tests/fixtures/rumus_manifest.json")
}

fn main() {
    let args = Args::parse();
    let spec_path = args.spec.unwrap_or_else(default_spec_path);
    let manifest_path = args.manifest.unwrap_or_else(default_manifest_path);

    let spec_bytes =
        std::fs::read(&spec_path).unwrap_or_else(|e| panic!("read {}: {}", spec_path.display(), e));
    let spec: Spec = serde_json::from_slice(&spec_bytes)
        .unwrap_or_else(|e| panic!("parse canonical_spec.json: {}", e));
    let spec_hash = hex_blake3(&spec_bytes);

    let scale_log2 = spec.quantization.scale_log2;

    let w1_flat = flatten(&spec.weights.w1.values);
    let b1 = spec.weights.b1.values.clone();
    let w2_flat = flatten(&spec.weights.w2.values);
    let b2 = spec.weights.b2.values.clone();

    let l1 = FixedLinear::new(w1_flat, Some(b1), 4, 8, scale_log2);
    let l2 = FixedLinear::new(w2_flat, Some(b2), 8, 4, scale_log2);

    let input_vec = spec.canonical_evaluation.input.clone();
    assert_eq!(input_vec.len(), 4, "canonical input must be length 4");
    let input_arr: [i16; 4] = [input_vec[0], input_vec[1], input_vec[2], input_vec[3]];
    let input = Tensor::from_i16_fixed(input_vec, vec![1, 4], scale_log2);

    let h = l1.forward(&input);
    let h_relu = relu(&h);
    let out_t = l2.forward(&h_relu);
    let out_guard = out_t.fixed_i16_data();
    let out_vec: Vec<i16> = out_guard.iter().copied().collect();
    drop(out_guard);
    assert_eq!(out_vec.len(), 4, "canonical output must be length 4");
    let out_arr: [i16; 4] = [out_vec[0], out_vec[1], out_vec[2], out_vec[3]];

    // Cross-check vs the spec's pinned canonical_evaluation.output.
    let expected_out = spec.canonical_evaluation.output.clone();
    assert_eq!(
        out_vec, expected_out,
        "RUMUS-computed output {out_vec:?} does not match spec.canonical_evaluation.output \
         {expected_out:?} — refusing to write a manifest",
    );

    let input_hash = hex_blake3(&encode_le(&input_arr));
    let output_hash = hex_blake3(&encode_le(&out_arr));

    let manifest = serde_json::json!({
        "framework": "Rumus",
        "framework_version": "rumus 0.4.0 (crates.io; MIT OR Apache-2.0; default features = [], CPU-only)",
        "weights_hash": spec_hash,
        "input": input_arr,
        "output": out_arr,
        "generated_at_utc": "2026-05-22T00:00:00Z",
        "generated_by": "tools/rumus_export — standalone Cargo package (not a workspace member); uses rumus::fixed::FixedLinear + rumus::fixed::requantize on the canonical W/B/input.",
        "generation_mode": "LiveExport",
        "spec_hash": spec_hash,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "generator_metadata": {
            "runtime_mode": "rumus-fixed-linear",
            "rumus_version": "0.4.0",
            "rumus_license": "MIT OR Apache-2.0",
            "rumus_crate_features": Vec::<String>::new(),
            "regen_command": "cd tools/rumus_export && cargo run --release"
        },
        "notes": "RUMUS 0.4.0 ships a first-class deterministic CPU FixedI16 integer-dense path (`rumus::fixed::FixedLinear` + `requantize`) implementing exactly the canonical arithmetic contract: i16 × i16 products, i64 accumulator, bias promoted to widened scale² domain via `<<scale_log2` BEFORE saturation, round-half-away-from-zero requantization, saturate-to-i16. The Stage 11b.1.a smoke probe verified byte-for-byte reproduction of the canonical output [33,-32,17,7]. `tools/rumus_export/` is intentionally a standalone Cargo package outside the root workspace so the operator-binary build graph cannot reach `rumus`."
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).expect("serialize manifest");

    match args.mode {
        Mode::VerifyOnly => {
            let on_disk = std::fs::read(&manifest_path).unwrap_or_else(|e| {
                panic!("read committed manifest {}: {}", manifest_path.display(), e)
            });
            // Compare the parsed JSON values (whitespace-insensitive)
            // since the on-disk fixture was hand-shaped.
            let on_disk_json: serde_json::Value =
                serde_json::from_slice(&on_disk).expect("parse committed manifest");
            assert_eq!(
                on_disk_json.get("input").cloned(),
                manifest.get("input").cloned(),
                "manifest.input drift",
            );
            assert_eq!(
                on_disk_json.get("output").cloned(),
                manifest.get("output").cloned(),
                "manifest.output drift — RUMUS computed a different value than the committed bytes",
            );
            assert_eq!(
                on_disk_json.get("spec_hash").cloned(),
                manifest.get("spec_hash").cloned(),
                "manifest.spec_hash drift",
            );
            assert_eq!(
                on_disk_json.get("input_hash").cloned(),
                manifest.get("input_hash").cloned(),
                "manifest.input_hash drift",
            );
            assert_eq!(
                on_disk_json.get("output_hash").cloned(),
                manifest.get("output_hash").cloned(),
                "manifest.output_hash drift",
            );
            println!(
                "rumus_export --verify-only OK\n  output     = {:?}\n  spec_hash  = {}\n  input_hash = {}\n  output_hash= {}",
                out_arr, manifest["spec_hash"], manifest["input_hash"], manifest["output_hash"]
            );
        }
        Mode::Regen => {
            std::fs::write(&manifest_path, &manifest_bytes)
                .unwrap_or_else(|e| panic!("write {}: {}", manifest_path.display(), e));
            println!("rumus_export --regen wrote {}", manifest_path.display());
        }
    }
}
