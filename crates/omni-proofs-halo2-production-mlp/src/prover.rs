//! Stage 11d.2 — developer-host halo2 prover for the production
//! fixed-point MLP.
//!
//! Gated by feature `prove`. NEVER reachable from `verify`-only
//! builds (and therefore never reachable from `omni-node`'s
//! `stage11d-production-verify` feature, which only enables `verify`
//! on this crate).
//!
//! The single entry point [`prove_canonical`] takes a canonical
//! `[i16; 16]` input, runs the canonical evaluator to derive the
//! full witness, generates a halo2 proof, and returns the
//! `(params, proof, output)` tuple for the regen tool to serialize.
//!
//! Proof determinism: the RNG is seeded with a fixed constant
//! ([`PROVER_RNG_SEED`]) so the proof bytes are reproducible
//! given the same `halo2_proofs` version + the same canonical
//! spec. A `halo2_proofs` version bump may shift the bytes; that
//! must be caught by the verifier's determinism guard test and
//! requires explicit fixture regen.
//!
//! ## Stage 14.5 — operator-reachable prover
//!
//! Stage 14.5 adds [`Halo2ProductionMlpProofBackend`], a thin
//! adapter that implements [`omni_zkml::ProofBackend`] over
//! [`prove_canonical`] so the existing prover surface is reachable
//! through `omni-node`'s new `stage11d-production-prove` feature.
//! The adapter validates `(model, input, output)` bytes against
//! the canonical spec and the neutral canonical evaluator before
//! invoking the halo2 prover. Same byte-determinism guarantee as
//! [`prove_canonical`]; same
//! `ProofSystem::Stage11dProductionFixedPointMlp` variant the
//! verifier already dispatches on; no new variants, no schema
//! growth. Mainnet posture unchanged — the variant is not in
//! `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`; mainnet refusal lands
//! at `check_mainnet_eligible` layer 6 (NOT layer 1, which only
//! fires for `testnet_or_dev_only=Some(true)` artifacts — production
//! artifacts declare `Some(false)`).

use halo2_proofs::{
    pasta::{EqAffine, Fp},
    plonk::{create_proof, keygen_pk, keygen_vk, Error},
    poly::commitment::Params,
    transcript::{Blake2bWrite, Challenge255},
};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::circuit::{fp_from_i64, ProductionMlpCircuit};
use crate::shared::HALO2_K;

/// Stage 11d.2 — deterministic prover RNG seed. Pinned so proof
/// generation is byte-stable across hosts.
pub const PROVER_RNG_SEED: [u8; 32] = *b"OmniNode/Stage11d.2/prover-rngv1";

/// Produce the canonical instance column for `(input, output)`.
pub fn instance_for(input_i16: [i16; 16], output_i16: [i16; 8]) -> Vec<Fp> {
    let mut v = Vec::with_capacity(24);
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
pub fn prove_canonical(input_i16: [i16; 16]) -> Result<(Vec<u8>, Vec<u8>, [i16; 8]), Error> {
    let circuit = ProductionMlpCircuit::from_canonical_input(input_i16);
    let output_i16 = ProductionMlpCircuit::canonical_outputs_for(input_i16);
    let instance = instance_for(input_i16, output_i16);

    let params = generate_params();
    let vk = keygen_vk(&params, &ProductionMlpCircuit::default())?;
    let pk = keygen_pk(&params, vk, &ProductionMlpCircuit::default())?;

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

// ── Stage 14.5 — ProofBackend adapter ────────────────────────────────────────

/// Stage 14.5 adapter: implements [`omni_zkml::ProofBackend`]
/// over [`prove_canonical`] so the halo2 production-MLP prover
/// is reachable through `omni-node`'s `stage11d-production-prove`
/// feature without duplicating the prover logic.
///
/// The adapter's `prove(model, input, output)` validates the
/// `(model, input, output)` byte triple against the canonical
/// spec + canonical evaluator before invoking the halo2 prover:
///
/// 1. `BLAKE3(model) == EXPECTED_PRODUCTION_SPEC_HASH` — operator
///    must supply the bytes of `assets/canonical_spec.json` (or
///    an equivalent committed copy). Any drift fails fast before
///    any halo2 work runs.
/// 2. `input.len() == 32` — exactly sixteen `i16` little-endian
///    values. Decoded to `[i16; 16]`.
/// 3. `output == encode_canonical_output(canonical_outputs_for(input))`
///    — the operator-claimed output (8 i16 values, 16 bytes) must
///    match the neutral canonical evaluator. The verifier performs
///    the same check; pinning it here refuses non-canonical
///    output bytes before halo2 work runs.
///
/// On success the adapter returns the halo2 proof bytes as
/// produced by [`prove_canonical`]. Params bytes are discarded:
/// the verifier embeds its own `params.bin` at compile time and
/// re-derives the verifying key from the circuit.
///
/// Mainnet posture unchanged.
/// `ProofSystem::Stage11dProductionFixedPointMlp` is not in
/// `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`; mainnet refusal lands
/// at `check_mainnet_eligible` **layer 6 only** (the production
/// artifact declares `testnet_or_dev_only=Some(false)`, so layer
/// 1 does NOT fire — distinct from
/// `ProofSystem::Stage11bHalo2Reference` which fires both layer
/// 1 and layer 6).
#[derive(Debug, Default, Clone, Copy)]
pub struct Halo2ProductionMlpProofBackend;

impl Halo2ProductionMlpProofBackend {
    pub fn new() -> Self {
        Self
    }
}

impl omni_zkml::ProofBackend for Halo2ProductionMlpProofBackend {
    fn prove(
        &self,
        model: &[u8],
        input: &[u8],
        output: &[u8],
    ) -> std::result::Result<Vec<u8>, omni_zkml::ProofBackendError> {
        // 1. Pin model bytes to the canonical production spec.
        let model_hash = blake3::hash(model);
        if model_hash.as_bytes() != &crate::shared::EXPECTED_PRODUCTION_SPEC_HASH {
            return Err(omni_zkml::ProofBackendError::BackendInternal(format!(
                "model bytes do not match canonical production-fixedpoint-mlp-v1 spec hash \
                 (got {}, expected {})",
                model_hash.to_hex(),
                {
                    let mut s = String::with_capacity(64);
                    for b in crate::shared::EXPECTED_PRODUCTION_SPEC_HASH {
                        s.push_str(&format!("{:02x}", b));
                    }
                    s
                }
            )));
        }
        // 2. Decode input bytes to [i16; 16].
        let input_i16 = decode_16xi16_le(input).map_err(|e| {
            omni_zkml::ProofBackendError::BackendInternal(format!(
                "halo2 production-MLP input bytes: {e}"
            ))
        })?;
        // 3. Bind claimed output to canonical_outputs_for(input).
        let expected_output =
            crate::circuit::ProductionMlpCircuit::canonical_outputs_for(input_i16);
        let expected_output_bytes =
            crate::encoding::encode_canonical_output(&expected_output);
        if output != expected_output_bytes.as_slice() {
            return Err(omni_zkml::ProofBackendError::BackendInternal(format!(
                "output bytes do not match canonical_outputs_for(input): \
                 caller supplied {:?}, canonical evaluator produced {:?}",
                output, expected_output_bytes
            )));
        }
        // 4. Run the halo2 prover.
        let (_params_bytes, proof_bytes, _output_i16) =
            prove_canonical(input_i16).map_err(|e| {
                omni_zkml::ProofBackendError::BackendInternal(format!(
                    "halo2 production-MLP prove_canonical: {e:?}"
                ))
            })?;
        Ok(proof_bytes)
    }

    fn backend_id(&self) -> &'static str {
        crate::shared::BACKEND_ID
    }

    fn proof_system(&self) -> omni_zkml::ProofSystem {
        omni_zkml::ProofSystem::Stage11dProductionFixedPointMlp
    }

    fn supported_formats(&self) -> &[omni_zkml::ModelFormat] {
        const FORMATS: &[omni_zkml::ModelFormat] =
            &[omni_zkml::ModelFormat::ProductionFixedPointMlp];
        FORMATS
    }

    /// The canonical-spec BLAKE3 hash uniquely identifies the
    /// circuit family for this backend (same hash the verifier
    /// pins via `EXPECTED_PRODUCTION_SPEC_HASH`).
    fn circuit_id(&self) -> Option<[u8; 32]> {
        Some(crate::shared::EXPECTED_PRODUCTION_SPEC_HASH)
    }

    /// Halo2 proving runs entirely on the operator's host with
    /// no hosted service.
    fn is_local_only(&self) -> bool {
        true
    }
}

fn decode_16xi16_le(bytes: &[u8]) -> std::result::Result<[i16; 16], String> {
    if bytes.len() != 32 {
        return Err(format!(
            "expected 32 bytes (16 × i16 LE); got {}",
            bytes.len()
        ));
    }
    let mut out = [0i16; 16];
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
        let (params_a, proof_a, _) = prove_canonical(CANONICAL_INPUT).unwrap();
        let (params_b, proof_b, _) = prove_canonical(CANONICAL_INPUT).unwrap();
        assert_eq!(params_a, params_b, "params must be byte-deterministic");
        assert_eq!(proof_a, proof_b, "proof bytes must be byte-deterministic");
    }

    // ── Stage 14.5 — ProofBackend adapter ───────────────────────────────────

    fn canonical_model_bytes() -> Vec<u8> {
        include_bytes!("../assets/canonical_spec.json").to_vec()
    }

    fn canonical_input_bytes() -> Vec<u8> {
        crate::encoding::encode_canonical_input(&CANONICAL_INPUT)
    }

    fn canonical_output_bytes() -> Vec<u8> {
        crate::encoding::encode_canonical_output(&CANONICAL_OUTPUT)
    }

    #[test]
    fn adapter_backend_id_and_proof_system_match_committed_constants() {
        use omni_zkml::ProofBackend;
        let b = Halo2ProductionMlpProofBackend::new();
        assert_eq!(b.backend_id(), crate::shared::BACKEND_ID);
        assert_eq!(
            b.proof_system(),
            omni_zkml::ProofSystem::Stage11dProductionFixedPointMlp
        );
        assert_eq!(
            b.supported_formats(),
            &[omni_zkml::ModelFormat::ProductionFixedPointMlp]
        );
        assert_eq!(
            b.circuit_id(),
            Some(crate::shared::EXPECTED_PRODUCTION_SPEC_HASH)
        );
        assert!(b.is_local_only());
    }

    #[test]
    fn adapter_produces_proof_for_canonical_triple() {
        use omni_zkml::ProofBackend;
        let b = Halo2ProductionMlpProofBackend::new();
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
        let b = Halo2ProductionMlpProofBackend::new();
        let err = b
            .prove(
                b"definitely not the canonical production spec json",
                &canonical_input_bytes(),
                &canonical_output_bytes(),
            )
            .expect_err("non-canonical model bytes must be refused");
        match err {
            omni_zkml::ProofBackendError::BackendInternal(msg) => {
                assert!(
                    msg.contains("production-fixedpoint-mlp-v1 spec hash"),
                    "expected canonical-spec refusal, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn adapter_refuses_when_input_byte_length_is_wrong() {
        use omni_zkml::ProofBackend;
        let b = Halo2ProductionMlpProofBackend::new();
        let err = b
            .prove(
                &canonical_model_bytes(),
                b"short", // 5 bytes, not 32
                &canonical_output_bytes(),
            )
            .expect_err("short input bytes must be refused");
        match err {
            omni_zkml::ProofBackendError::BackendInternal(msg) => {
                assert!(
                    msg.contains("expected 32 bytes"),
                    "expected length-refusal message, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn adapter_refuses_when_output_does_not_match_canonical_evaluator() {
        use omni_zkml::ProofBackend;
        let b = Halo2ProductionMlpProofBackend::new();
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
                    msg.contains("do not match canonical_outputs_for"),
                    "expected canonical-evaluator binding refusal, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn adapter_proof_is_byte_deterministic_across_two_calls() {
        use omni_zkml::ProofBackend;
        let b = Halo2ProductionMlpProofBackend::new();
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
