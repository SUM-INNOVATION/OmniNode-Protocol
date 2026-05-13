//! Phase 5 Stage 6 — chain-attestation test vector deliverable.
//!
//! Default mode: **verify**. Compute the three chain attestation vectors
//! via the actual `omni-zkml` implementation path and assert byte-equality
//! against the committed fixture at
//! `tests/fixtures/chain_attestation_vectors.json`. This is what runs
//! under `cargo test`; CI is in this mode.
//!
//! Regen mode: set `OMNINODE_REGEN_VECTORS=1` to overwrite the committed
//! fixture with the freshly-computed vectors. Explicit-only — a developer
//! runs it once when intentionally bumping the spec, then commits the
//! regenerated JSON.

use std::path::PathBuf;

use omni_zkml::{compute_chain_attestation_vector, ChainAttestationVector};

/// All `model_hash` / `manifest_root` / `response_hash` / `proof_root`
/// fields here are 64-char lowercase hex (= 32 bytes). The seed is also
/// 64-char lowercase hex (= 32 bytes).
const VECTORS: &[Input] = &[
    // Vector 1: byte 0x00 model, 0x11 manifest, 0x22 response, 0x33 proof,
    // seed 0x00 — repeated 32 times each.
    Input {
        session_id: "omninode-stage6-vec-1",
        model_hash:    "0000000000000000000000000000000000000000000000000000000000000000",
        manifest_root: "1111111111111111111111111111111111111111111111111111111111111111",
        response_hash: "2222222222222222222222222222222222222222222222222222222222222222",
        proof_root:    "3333333333333333333333333333333333333333333333333333333333333333",
        verifier_seed: "0000000000000000000000000000000000000000000000000000000000000000",
    },
    // Vector 2: byte 0x44 / 0x55 / 0x66 / 0x77, seed 0x01.
    Input {
        session_id: "omninode-stage6-vec-2",
        model_hash:    "4444444444444444444444444444444444444444444444444444444444444444",
        manifest_root: "5555555555555555555555555555555555555555555555555555555555555555",
        response_hash: "6666666666666666666666666666666666666666666666666666666666666666",
        proof_root:    "7777777777777777777777777777777777777777777777777777777777777777",
        verifier_seed: "0101010101010101010101010101010101010101010101010101010101010101",
    },
    // Vector 3: ASCII-only UUID-shaped session id (39 chars), byte
    // 0x88 / 0x99 / 0xaa / 0xbb, seed 0x02.
    Input {
        session_id: "omninode-stage6-vec-3-abcdef-0123456789",
        model_hash:    "8888888888888888888888888888888888888888888888888888888888888888",
        manifest_root: "9999999999999999999999999999999999999999999999999999999999999999",
        response_hash: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        proof_root:    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        verifier_seed: "0202020202020202020202020202020202020202020202020202020202020202",
    },
];

struct Input {
    session_id: &'static str,
    model_hash: &'static str,
    manifest_root: &'static str,
    response_hash: &'static str,
    proof_root: &'static str,
    verifier_seed: &'static str,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("chain_attestation_vectors.json")
}

fn compute_all() -> Vec<ChainAttestationVector> {
    VECTORS
        .iter()
        .enumerate()
        .map(|(i, v)| {
            compute_chain_attestation_vector(
                v.session_id,
                v.model_hash,
                v.manifest_root,
                v.response_hash,
                v.proof_root,
                v.verifier_seed,
            )
            .unwrap_or_else(|e| panic!("vector {i} computation failed: {e}"))
        })
        .collect()
}

#[test]
fn three_chain_attestation_vectors_match_committed_fixture() {
    let computed = compute_all();
    let path = fixture_path();

    if std::env::var("OMNINODE_REGEN_VECTORS").as_deref() == Ok("1") {
        // Regen mode: overwrite the committed fixture and stop here.
        std::fs::create_dir_all(path.parent().unwrap()).expect("create fixtures dir");
        let bytes = serde_json::to_vec_pretty(&computed).expect("pretty-print JSON");
        std::fs::write(&path, &bytes).expect("write fixture");
        eprintln!(
            "[regen] wrote {} vectors to {}",
            computed.len(),
            path.display()
        );
        return;
    }

    // Verify mode (default).
    let committed_bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read fixture at {}: {e} — run with OMNINODE_REGEN_VECTORS=1 to regenerate",
            path.display()
        )
    });
    let committed: Vec<ChainAttestationVector> = serde_json::from_slice(&committed_bytes)
        .unwrap_or_else(|e| panic!("fixture at {} is not valid JSON: {e}", path.display()));

    assert_eq!(
        committed.len(),
        computed.len(),
        "fixture vector count mismatch"
    );
    for (i, (got, want)) in computed.iter().zip(committed.iter()).enumerate() {
        assert_eq!(
            got, want,
            "chain attestation vector {i} ({}) drifted from committed fixture",
            got.session_id
        );
    }
}
