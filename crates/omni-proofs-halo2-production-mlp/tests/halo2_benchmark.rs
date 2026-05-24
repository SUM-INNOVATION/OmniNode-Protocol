//! Stage 11d.2 — quick verify-time microbenchmark.
//!
//! Not a real criterion-style bench, but enough to populate
//! `docs/stage11d.2-benchmark-record.md` with concrete numbers
//! that exercise the same code path the operator binary calls.
//! Runs 100 verifications back-to-back; prints min/median/max.
//!
//! Marked `#[ignore]` so it does NOT run in default CI; invoke
//! locally via `cargo test --release --features verify
//! --test halo2_benchmark -- --ignored --nocapture`.

#![cfg(feature = "verify")]

use std::time::Instant;

use omni_proofs_halo2_production_mlp::Halo2ProductionMlpVerifier;
use omni_zkml::{ProofArtifactBody, ProofVerifier};

fn load_committed_artifact() -> ProofArtifactBody {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/halo2/proof_artifact.json");
    let bytes =
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("parse proof_artifact.json")
}

#[test]
#[ignore = "Stage 11d.2 microbench — opt-in via `--ignored`; not in CI"]
fn verify_microbench_100_iterations() {
    let verifier = Halo2ProductionMlpVerifier::from_embedded_fixtures()
        .expect("from_embedded_fixtures");
    let body = load_committed_artifact();

    // Warm up once.
    let _ = verifier.verify_artifact(&body).unwrap();

    const N: usize = 100;
    let mut samples_us = Vec::with_capacity(N);
    for _ in 0..N {
        let t0 = Instant::now();
        let ok = verifier.verify_artifact(&body).unwrap();
        assert!(ok);
        samples_us.push(t0.elapsed().as_micros() as u64);
    }
    samples_us.sort();
    let min = samples_us[0];
    let p50 = samples_us[N / 2];
    let p95 = samples_us[(N * 95) / 100];
    let max = samples_us[N - 1];
    eprintln!("verify N={N} min={min}us p50={p50}us p95={p95}us max={max}us");
}
