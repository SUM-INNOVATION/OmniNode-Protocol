//! Stage 11b.1.b — developer-host halo2 prover.
//!
//! Gated by feature `prove`. NEVER reachable from `verify`-only
//! builds (and therefore never reachable from `omni-node`'s
//! `halo2-reference-verify` feature, which only enables `verify`
//! on this crate).
//!
//! The single entry point [`prove_canonical`] takes a canonical
//! `[i16; 4]` input, runs the canonical evaluator to derive the
//! full witness, generates a halo2 proof, and returns the
//! (params, proof, output) tuple for the regen tool to serialize.
//!
//! Proof determinism: the RNG is seeded with a fixed constant
//! ([`PROVER_RNG_SEED`]) so the proof bytes are reproducible
//! given the same `halo2_proofs` version + the same canonical
//! spec. A halo2_proofs version bump may shift the bytes; that
//! must be caught by the verifier's determinism guard test and
//! requires explicit fixture regen.

use halo2_proofs::{
    pasta::{EqAffine, Fp},
    plonk::{create_proof, keygen_pk, keygen_vk, Error},
    poly::commitment::Params,
    transcript::{Blake2bWrite, Challenge255},
};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::circuit::{fp_from_i64, BoundedMlpCircuit};
use crate::shared::HALO2_K;

/// Stage 11b.1.b — deterministic prover RNG seed. Pinned so
/// proof generation is byte-stable across hosts.
pub const PROVER_RNG_SEED: [u8; 32] = *b"OmniNode/Stage11b.1.b/prover-rng";

/// Produce the canonical instance column for `(input, output)`.
pub fn instance_for(input_i16: [i16; 4], output_i16: [i16; 4]) -> Vec<Fp> {
    let mut v = Vec::with_capacity(8);
    for x in &input_i16 {
        v.push(fp_from_i64(*x as i64));
    }
    for y in &output_i16 {
        v.push(fp_from_i64(*y as i64));
    }
    v
}

/// Generate a fresh `Params<EqAffine>` for the configured circuit
/// size. Deterministic: `Params::new(HALO2_K)` always produces
/// byte-identical params for a given `HALO2_K` value.
pub fn generate_params() -> Params<EqAffine> {
    Params::new(HALO2_K)
}

/// Run the full prove pipeline for a canonical input.
///
/// Returns `(params_bytes, proof_bytes, output_i16)` so the caller
/// can persist the params + proof and embed the canonical output
/// in the artifact body.
///
/// `Params::write` returns `io::Result<()>`; writing to an
/// in-memory `Vec<u8>` is infallible, so the panic on serialization
/// failure here is a programming-error signal, not a runtime path.
pub fn prove_canonical(input_i16: [i16; 4]) -> Result<(Vec<u8>, Vec<u8>, [i16; 4]), Error> {
    let circuit = BoundedMlpCircuit::from_canonical_input(input_i16);
    let output_i16 = BoundedMlpCircuit::canonical_outputs_for(input_i16);
    let instance = instance_for(input_i16, output_i16);

    let params = generate_params();
    let vk = keygen_vk(&params, &BoundedMlpCircuit::default())?;
    let pk = keygen_pk(&params, vk, &BoundedMlpCircuit::default())?;

    let mut transcript = Blake2bWrite::<Vec<u8>, EqAffine, Challenge255<_>>::init(vec![]);
    let rng = ChaCha20Rng::from_seed(PROVER_RNG_SEED);

    create_proof(
        &params,
        &pk,
        &[circuit],
        &[&[&instance]],
        rng,
        &mut transcript,
    )?;
    let proof_bytes = transcript.finalize();

    let mut params_bytes = Vec::new();
    params
        .write(&mut params_bytes)
        .expect("Params::write to Vec<u8> is infallible");

    Ok((params_bytes, proof_bytes, output_i16))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{CANONICAL_INPUT, CANONICAL_OUTPUT};

    #[test]
    fn prove_canonical_produces_proof() {
        let (params_bytes, proof_bytes, output) = prove_canonical(CANONICAL_INPUT).unwrap();
        assert!(!params_bytes.is_empty(), "params_bytes should be non-empty");
        assert!(!proof_bytes.is_empty(), "proof_bytes should be non-empty");
        assert_eq!(output, CANONICAL_OUTPUT, "prover-derived output must match canonical");
    }

    #[test]
    fn prove_canonical_is_byte_deterministic() {
        // Two runs with the same RNG seed and same params/vk derivation
        // must produce byte-identical proofs.
        let (params_a, proof_a, _) = prove_canonical(CANONICAL_INPUT).unwrap();
        let (params_b, proof_b, _) = prove_canonical(CANONICAL_INPUT).unwrap();
        assert_eq!(params_a, params_b, "params must be byte-deterministic");
        assert_eq!(proof_a, proof_b, "proof bytes must be byte-deterministic");
    }
}
