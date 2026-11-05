//! The frozen canonical model, `production-fixedpoint-mlp-v1`:
//! Dense(16 → 32), ReLU, Dense(32 → 16), ReLU, Dense(16 → 8).
//!
//! The weights are the closed-form patterns of
//! `omni-proofs-halo2-production-mlp::canonical` (copied, not shared, so this
//! crate stays dependency-free). The proof crate's tests check that they
//! equal its `W1`..`B3` and the arrays in its `assets/canonical_spec.json`,
//! and that [`CANONICAL_SPEC_BLAKE3_HEX`] equals its
//! `EXPECTED_PRODUCTION_SPEC_HASH`.

/// `spec_name` of the canonical model.
pub const CANONICAL_SPEC_NAME: &str = "production-fixedpoint-mlp-v1";

/// `spec_version` of the canonical model.
pub const CANONICAL_SPEC_VERSION: u32 = 1;

/// The canonical model's identity: `BLAKE3` of the exact bytes of
/// `canonical_spec.json`, 64 lowercase hex. A job running the canonical
/// model carries this as `model_hash`, and its manifest is that file.
pub const CANONICAL_SPEC_BLAKE3_HEX: &str =
    "1c95eea59ab7fe811f1a3c668798221577225c917846888a803b939f9cbda741";

/// Layer 1 weights, `[input=16][output=32]`:
/// `W1[i][j] = ((i * 13 + j * 7) % 17) - 8`.
pub const W1: [[i16; 32]; 16] = build_w1();
/// Layer 1 biases: `B1[j] = ((j * 11) % 129) - 64`.
pub const B1: [i16; 32] = build_b1();
/// Layer 2 weights, `[input=32][output=16]`:
/// `W2[i][j] = ((i * 5 + j * 3) % 9) - 4`.
pub const W2: [[i16; 16]; 32] = build_w2();
/// Layer 2 biases: `B2[j] = ((j * 17) % 65) - 32`.
pub const B2: [i16; 16] = build_b2();
/// Layer 3 weights, `[input=16][output=8]`:
/// `W3[i][j] = ((i * 7 + j * 11) % 17) - 8`.
pub const W3: [[i16; 8]; 16] = build_w3();
/// Layer 3 biases: `B3[j] = ((j * 19) % 65) - 32`.
pub const B3: [i16; 8] = build_b3();

const fn build_w1() -> [[i16; 32]; 16] {
    let mut out = [[0i16; 32]; 16];
    let mut i = 0;
    while i < 16 {
        let mut j = 0;
        while j < 32 {
            out[i][j] = ((i * 13 + j * 7) % 17) as i16 - 8;
            j += 1;
        }
        i += 1;
    }
    out
}

const fn build_b1() -> [i16; 32] {
    let mut out = [0i16; 32];
    let mut j = 0;
    while j < 32 {
        out[j] = ((j * 11) % 129) as i16 - 64;
        j += 1;
    }
    out
}

const fn build_w2() -> [[i16; 16]; 32] {
    let mut out = [[0i16; 16]; 32];
    let mut i = 0;
    while i < 32 {
        let mut j = 0;
        while j < 16 {
            out[i][j] = ((i * 5 + j * 3) % 9) as i16 - 4;
            j += 1;
        }
        i += 1;
    }
    out
}

const fn build_b2() -> [i16; 16] {
    let mut out = [0i16; 16];
    let mut j = 0;
    while j < 16 {
        out[j] = ((j * 17) % 65) as i16 - 32;
        j += 1;
    }
    out
}

const fn build_w3() -> [[i16; 8]; 16] {
    let mut out = [[0i16; 8]; 16];
    let mut i = 0;
    while i < 16 {
        let mut j = 0;
        while j < 8 {
            out[i][j] = ((i * 7 + j * 11) % 17) as i16 - 8;
            j += 1;
        }
        i += 1;
    }
    out
}

const fn build_b3() -> [i16; 8] {
    let mut out = [0i16; 8];
    let mut j = 0;
    while j < 8 {
        out[j] = ((j * 19) % 65) as i16 - 32;
        j += 1;
    }
    out
}
