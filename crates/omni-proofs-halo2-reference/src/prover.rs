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
//!
//! ## Stage 14.1 — operator-reachable prover
//!
//! Stage 14.1 adds [`Halo2ReferenceProofBackend`], a thin adapter
//! that implements [`omni_zkml::ProofBackend`] over
//! [`prove_canonical`] so the existing prover surface is
//! reachable through `omni-node`'s new `halo2-reference-prove`
//! feature. The adapter validates `(model, input, output)` bytes
//! against the canonical spec + canonical evaluator before
//! producing a proof. Same byte-determinism guarantee as
//! [`prove_canonical`]; same `ProofSystem::Stage11bHalo2Reference`
//! variant the verifier already dispatches on; no new variants,
//! no schema growth. Mainnet refusal is unchanged (the variant is
//! not in `MAINNET_APPROVED_PROOF_SYSTEMS`).

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

// ── Stage 14.1 — ProofBackend adapter ────────────────────────────────────────

/// Stage 14.1 adapter: implements [`omni_zkml::ProofBackend`] over
/// [`prove_canonical`] so the halo2 reference prover is reachable
/// through `omni-node`'s `halo2-reference-prove` feature without
/// duplicating the prover logic.
///
/// The adapter's `prove(model, input, output)` validates the
/// `(model, input, output)` byte triple against the canonical
/// spec + canonical evaluator before invoking the halo2 prover:
///
/// 1. `BLAKE3(model) == EXPECTED_SPEC_HASH` — operator must
///    supply the bytes of `assets/canonical_spec.json` (or an
///    equivalent committed copy). Any drift fails fast before
///    any halo2 work runs.
/// 2. `input.len() == 8` — exactly four `i16` little-endian
///    values. Decoded to `[i16; 4]`.
/// 3. `output == encode_tensor_4xi16_le(canonical_evaluate(input))`
///    — the operator-claimed output must match the neutral
///    canonical evaluator. This binding is the same defense-in-
///    depth check the verifier performs (step 7.5 in
///    [`crate::verifier::Halo2ReferenceVerifier::verify_artifact`]).
///
/// On success the adapter returns the halo2 proof bytes as
/// produced by [`prove_canonical`]. Params bytes are discarded:
/// the verifier embeds its own `params.bin` at compile time and
/// re-derives the verifying key from the circuit, so params do
/// not need to flow through the artifact.
///
/// Mainnet posture unchanged — `ProofSystem::Stage11bHalo2Reference`
/// is not in `MAINNET_APPROVED_PROOF_SYSTEMS`; the operator binary
/// hard-refuses submission to `chain_id == 1` at
/// [`omni_zkml::check_mainnet_eligible`] layers 1 + 3 + 6.
#[derive(Debug, Default, Clone, Copy)]
pub struct Halo2ReferenceProofBackend;

impl Halo2ReferenceProofBackend {
    pub fn new() -> Self {
        Self
    }
}

impl omni_zkml::ProofBackend for Halo2ReferenceProofBackend {
    fn prove(
        &self,
        model: &[u8],
        input: &[u8],
        output: &[u8],
    ) -> std::result::Result<Vec<u8>, omni_zkml::ProofBackendError> {
        // 1. Pin model bytes to the canonical spec.
        let model_hash = blake3::hash(model);
        if model_hash.as_bytes() != &crate::shared::EXPECTED_SPEC_HASH {
            return Err(omni_zkml::ProofBackendError::BackendInternal(format!(
                "model bytes do not match canonical halo2-mlp-v1 spec hash \
                 (got {}, expected {})",
                model_hash.to_hex(),
                {
                    let mut s = String::with_capacity(64);
                    for b in crate::shared::EXPECTED_SPEC_HASH {
                        s.push_str(&format!("{:02x}", b));
                    }
                    s
                }
            )));
        }
        // 2. Decode input bytes to [i16; 4].
        let input_i16 = decode_4xi16_le(input).map_err(|e| {
            omni_zkml::ProofBackendError::BackendInternal(format!(
                "halo2-reference input bytes: {e}"
            ))
        })?;
        // 3. Bind claimed output to canonical_evaluate(input).
        let expected_output =
            crate::circuit::BoundedMlpCircuit::canonical_outputs_for(input_i16);
        let expected_output_bytes =
            crate::encoding::encode_tensor_4xi16_le(&expected_output);
        if output != expected_output_bytes.as_slice() {
            return Err(omni_zkml::ProofBackendError::BackendInternal(format!(
                "output bytes do not match canonical_evaluate(input): \
                 caller supplied {:?}, canonical evaluator produced {:?}",
                output, expected_output_bytes
            )));
        }
        // 4. Run the halo2 prover.
        let (_params_bytes, proof_bytes, _output_i16) =
            prove_canonical(input_i16).map_err(|e| {
                omni_zkml::ProofBackendError::BackendInternal(format!(
                    "halo2 prove_canonical: {e:?}"
                ))
            })?;
        Ok(proof_bytes)
    }

    fn backend_id(&self) -> &'static str {
        crate::shared::BACKEND_ID
    }

    fn proof_system(&self) -> omni_zkml::ProofSystem {
        omni_zkml::ProofSystem::Stage11bHalo2Reference
    }

    fn supported_formats(&self) -> &[omni_zkml::ModelFormat] {
        const FORMATS: &[omni_zkml::ModelFormat] =
            &[omni_zkml::ModelFormat::Halo2ReferenceMlp];
        FORMATS
    }

    /// The canonical-spec BLAKE3 hash uniquely identifies the
    /// circuit family for this backend (same hash the verifier
    /// pins via `EXPECTED_SPEC_HASH`).
    fn circuit_id(&self) -> Option<[u8; 32]> {
        Some(crate::shared::EXPECTED_SPEC_HASH)
    }

    /// Halo2 proving runs entirely on the operator's host with
    /// no hosted service. The mainnet refusal already fires at
    /// the proof_system layer for `Stage11bHalo2Reference`;
    /// `is_local_only=true` is the consistent declaration.
    fn is_local_only(&self) -> bool {
        true
    }
}

fn decode_4xi16_le(bytes: &[u8]) -> std::result::Result<[i16; 4], String> {
    if bytes.len() != 8 {
        return Err(format!(
            "expected 8 bytes (4 × i16 LE); got {}",
            bytes.len()
        ));
    }
    let mut out = [0i16; 4];
    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        out[i] = i16::from_le_bytes([chunk[0], chunk[1]]);
    }
    Ok(out)
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

    // ── Stage 14.1 — ProofBackend adapter ───────────────────────────────────

    fn canonical_model_bytes() -> Vec<u8> {
        // The canonical spec JSON is committed under assets/.
        // `include_bytes!` pins the bytes at compile time so the
        // BLAKE3 round-trips against EXPECTED_SPEC_HASH.
        include_bytes!("../assets/canonical_spec.json").to_vec()
    }

    fn canonical_input_bytes() -> Vec<u8> {
        crate::encoding::encode_tensor_4xi16_le(&CANONICAL_INPUT)
    }

    fn canonical_output_bytes() -> Vec<u8> {
        crate::encoding::encode_tensor_4xi16_le(&CANONICAL_OUTPUT)
    }

    #[test]
    fn adapter_backend_id_and_proof_system_match_committed_constants() {
        use omni_zkml::ProofBackend;
        let b = Halo2ReferenceProofBackend::new();
        assert_eq!(b.backend_id(), crate::shared::BACKEND_ID);
        assert_eq!(
            b.proof_system(),
            omni_zkml::ProofSystem::Stage11bHalo2Reference
        );
        assert_eq!(
            b.supported_formats(),
            &[omni_zkml::ModelFormat::Halo2ReferenceMlp]
        );
        assert_eq!(b.circuit_id(), Some(crate::shared::EXPECTED_SPEC_HASH));
        assert!(b.is_local_only());
    }

    #[test]
    fn adapter_produces_proof_for_canonical_triple() {
        use omni_zkml::ProofBackend;
        let b = Halo2ReferenceProofBackend::new();
        let proof = b
            .prove(
                &canonical_model_bytes(),
                &canonical_input_bytes(),
                &canonical_output_bytes(),
            )
            .expect("prove on canonical triple should succeed");
        assert!(!proof.is_empty());
        // Determinism cross-check: the adapter's proof equals
        // `prove_canonical(CANONICAL_INPUT)`'s proof bytes.
        let (_params, expected_proof, _out) =
            prove_canonical(CANONICAL_INPUT).unwrap();
        assert_eq!(proof, expected_proof);
    }

    #[test]
    fn adapter_refuses_when_model_bytes_do_not_match_spec_hash() {
        use omni_zkml::ProofBackend;
        let b = Halo2ReferenceProofBackend::new();
        let err = b
            .prove(
                b"definitely not the canonical spec json",
                &canonical_input_bytes(),
                &canonical_output_bytes(),
            )
            .expect_err("non-canonical model bytes must be refused");
        match err {
            omni_zkml::ProofBackendError::BackendInternal(msg) => {
                assert!(
                    msg.contains("canonical halo2-mlp-v1 spec hash"),
                    "expected canonical-spec refusal, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn adapter_refuses_when_input_byte_length_is_wrong() {
        use omni_zkml::ProofBackend;
        let b = Halo2ReferenceProofBackend::new();
        let err = b
            .prove(
                &canonical_model_bytes(),
                b"short", // 5 bytes, not 8
                &canonical_output_bytes(),
            )
            .expect_err("short input bytes must be refused");
        match err {
            omni_zkml::ProofBackendError::BackendInternal(msg) => {
                assert!(
                    msg.contains("expected 8 bytes"),
                    "expected length-refusal message, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn adapter_refuses_when_output_does_not_match_canonical_evaluator() {
        use omni_zkml::ProofBackend;
        let b = Halo2ReferenceProofBackend::new();
        let mut bogus_output = canonical_output_bytes();
        bogus_output[0] ^= 0x01;
        let err = b
            .prove(
                &canonical_model_bytes(),
                &canonical_input_bytes(),
                &bogus_output,
            )
            .expect_err("output drift must be refused before halo2 prove runs");
        match err {
            omni_zkml::ProofBackendError::BackendInternal(msg) => {
                assert!(
                    msg.contains("do not match canonical_evaluate"),
                    "expected canonical-evaluator binding refusal, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn adapter_proof_is_byte_deterministic_across_two_calls() {
        use omni_zkml::ProofBackend;
        let b = Halo2ReferenceProofBackend::new();
        let p1 = b
            .prove(
                &canonical_model_bytes(),
                &canonical_input_bytes(),
                &canonical_output_bytes(),
            )
            .unwrap();
        let p2 = b
            .prove(
                &canonical_model_bytes(),
                &canonical_input_bytes(),
                &canonical_output_bytes(),
            )
            .unwrap();
        assert_eq!(p1, p2, "adapter must preserve prove_canonical determinism");
    }
}
