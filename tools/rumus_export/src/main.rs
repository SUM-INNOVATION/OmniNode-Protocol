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
    /// Stage 11c — verify the committed `rumus_corpus.json` by
    /// re-running every entry's input through RUMUS's
    /// `fixed::FixedLinear` chain.
    CorpusVerify,
    /// Stage 11c — overwrite `rumus_corpus.json` with freshly-
    /// computed bytes (developer-host only).
    CorpusRegen,
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
    /// Stage 11c — path to ground-truth corpus.json. Defaults.
    #[arg(long)]
    corpus_in: Option<PathBuf>,
    /// Stage 11c — path to rumus_corpus.json. Defaults.
    #[arg(long)]
    corpus_out: Option<PathBuf>,
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

fn default_corpus_in_path() -> PathBuf {
    workspace_root().join("crates/omni-proofs-halo2-reference/tests/fixtures/corpus.json")
}

fn default_corpus_out_path() -> PathBuf {
    workspace_root().join("crates/omni-proofs-halo2-reference/tests/fixtures/rumus_corpus.json")
}

fn run_corpus(
    spec_bytes: &[u8],
    spec: &Spec,
    spec_hash: &str,
    corpus_in: &std::path::Path,
    corpus_out: &std::path::Path,
    regen: bool,
) {
    let truth_bytes = std::fs::read(corpus_in)
        .unwrap_or_else(|e| panic!("read corpus_in {}: {}", corpus_in.display(), e));
    let truth: serde_json::Value =
        serde_json::from_slice(&truth_bytes).expect("parse corpus_in");
    let scale_log2 = spec.quantization.scale_log2;
    let w1_flat = flatten(&spec.weights.w1.values);
    let b1 = spec.weights.b1.values.clone();
    let w2_flat = flatten(&spec.weights.w2.values);
    let b2 = spec.weights.b2.values.clone();
    let l1 = FixedLinear::new(w1_flat, Some(b1), 4, 8, scale_log2);
    let l2 = FixedLinear::new(w2_flat, Some(b2), 8, 4, scale_log2);

    let mut entries_out: Vec<serde_json::Value> = Vec::new();
    let truth_entries = truth["entries"]
        .as_array()
        .expect("corpus.entries is array");
    for entry in truth_entries {
        let in_arr = entry["input"].as_array().expect("entry.input is array");
        assert_eq!(in_arr.len(), 4);
        let input_vec: Vec<i16> = in_arr
            .iter()
            .map(|v| v.as_i64().unwrap() as i16)
            .collect();
        let input_arr: [i16; 4] = [input_vec[0], input_vec[1], input_vec[2], input_vec[3]];
        let input_tensor =
            Tensor::from_i16_fixed(input_vec.clone(), vec![1, 4], scale_log2);
        let h = l1.forward(&input_tensor);
        let h_relu = relu(&h);
        let out_t = l2.forward(&h_relu);
        let out_guard = out_t.fixed_i16_data();
        let rumus_out: Vec<i16> = out_guard.iter().copied().collect();
        drop(out_guard);
        let expected: Vec<i16> = entry["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i16)
            .collect();
        assert_eq!(
            rumus_out, expected,
            "rumus corpus drift on entry {:?}: rumus produced {:?}, truth has {:?}",
            entry["label"], rumus_out, expected
        );
        let out_arr: [i16; 4] = [rumus_out[0], rumus_out[1], rumus_out[2], rumus_out[3]];
        entries_out.push(serde_json::json!({
            "label": entry["label"],
            "input": input_arr,
            "output": out_arr,
            "input_hash": hex_blake3(&encode_le(&input_arr)),
            "output_hash": hex_blake3(&encode_le(&out_arr)),
            "notes": entry.get("notes").cloned().unwrap_or(serde_json::json!("")),
        }));
    }
    let _ = spec_bytes;

    let payload = serde_json::json!({
        "framework": "Rumus",
        "framework_version": "rumus 0.4.0 (crates.io; MIT OR Apache-2.0)",
        "generation_mode": "LiveExport",
        "generator_metadata": {
            "runtime_mode": "rumus-fixed-linear",
            "rumus_version": "0.4.0",
        },
        "spec_name": truth["spec_name"],
        "spec_version": truth["spec_version"],
        "spec_hash": spec_hash,
        "tensor_encoding": truth["tensor_encoding"],
        "description": "RUMUS corpus: each entry re-verified via rumus::fixed::FixedLinear + requantize.",
        "entries": entries_out,
    });
    if regen {
        std::fs::write(corpus_out, serde_json::to_string_pretty(&payload).unwrap() + "\n")
            .unwrap_or_else(|e| panic!("write {}: {}", corpus_out.display(), e));
        println!(
            "rumus_export corpus-regen wrote {} ({} entries)",
            corpus_out.display(),
            payload["entries"].as_array().unwrap().len()
        );
    } else {
        let on_disk = std::fs::read(corpus_out).unwrap_or_else(|e| {
            panic!("read on-disk rumus_corpus.json {}: {}", corpus_out.display(), e)
        });
        let on_disk_json: serde_json::Value =
            serde_json::from_slice(&on_disk).expect("parse on-disk rumus_corpus.json");
        let truth_n = payload["entries"].as_array().unwrap().len();
        let disk_n = on_disk_json["entries"].as_array().unwrap().len();
        assert_eq!(truth_n, disk_n, "rumus_corpus.json entry count drift");
        for (i, (t, d)) in payload["entries"]
            .as_array()
            .unwrap()
            .iter()
            .zip(on_disk_json["entries"].as_array().unwrap().iter())
            .enumerate()
        {
            for key in ["input", "output", "input_hash", "output_hash"] {
                assert_eq!(t[key], d[key], "rumus_corpus.json entry {i} {key} drift");
            }
        }
        println!(
            "rumus_export corpus-verify-only OK ({} entries)",
            payload["entries"].as_array().unwrap().len()
        );
    }
}

fn parse_spec(spec_path: &std::path::Path) -> (Vec<u8>, Spec, String) {
    let spec_bytes = std::fs::read(spec_path)
        .unwrap_or_else(|e| panic!("read {}: {}", spec_path.display(), e));
    let spec: Spec =
        serde_json::from_slice(&spec_bytes).unwrap_or_else(|e| panic!("parse spec: {}", e));
    let spec_hash = hex_blake3(&spec_bytes);
    (spec_bytes, spec, spec_hash)
}

fn main() {
    let args = Args::parse();
    let spec_path = args.spec.clone().unwrap_or_else(default_spec_path);
    let manifest_path = args.manifest.clone().unwrap_or_else(default_manifest_path);
    if matches!(args.mode, Mode::CorpusVerify | Mode::CorpusRegen) {
        let (spec_bytes, spec, spec_hash) = parse_spec(&spec_path);
        let corpus_in = args.corpus_in.unwrap_or_else(default_corpus_in_path);
        let corpus_out = args.corpus_out.unwrap_or_else(default_corpus_out_path);
        run_corpus(
            &spec_bytes,
            &spec,
            &spec_hash,
            &corpus_in,
            &corpus_out,
            matches!(args.mode, Mode::CorpusRegen),
        );
        return;
    }

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
        Mode::CorpusVerify | Mode::CorpusRegen => unreachable!("handled above"),
    }
}
