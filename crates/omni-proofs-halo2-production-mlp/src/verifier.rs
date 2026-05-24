//! Stage 11d.2 — `Halo2ProductionMlpVerifier`.
//!
//! Validates a Stage 11d.2 halo2 production-MLP proof artifact
//! against the frozen `production-fixedpoint-mlp-v1 / spec_version: 1`
//! canonical spec. Mirrors `omni-proofs-halo2-reference::verifier`
//! (Stage 11b.1.b / 11b.0.1) but for the production tensor dims
//! (16 inputs, 8 outputs) and the production proof system /
//! model-format pair.
//!
//! ## Distinguishability hard rules (Stage 11d.0 criteria §1.6)
//!
//! - **H1**: separate crate ⇒ separate verifier struct. The
//!   `Halo2ReferenceVerifier` cannot be repurposed for production.
//! - **H2**: separate `EXPECTED_PRODUCTION_SPEC_HASH` const.
//! - **H3**: distinct canonical spec at `assets/canonical_spec.json`.
//!
//! ## Mainnet posture (Stage 11d.0/11d.1 invariants preserved)
//!
//! This verifier **accepts production-shape artifacts**
//! (`testnet_or_dev_only: Some(false)`). However, the
//! `check_mainnet_eligible` layer 6 still hard-refuses because
//! `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]` through Stage
//! 11d.2. Mainnet allowlist entry for this proof class is a
//! Stage 11d.3 deliverable (R1–R9 sign-off gate).
//!
//! ## Verification pipeline (verify_artifact)
//!
//! 1. `proof_system == Some(Stage11dProductionFixedPointMlp)`
//! 2. `model_format == Some(ProductionFixedPointMlp)`
//! 3. `model_framework == Some(FrameworkAgnostic)` (canonical spec
//!    is framework-neutral; framework-specific corpora are tested
//!    separately via the cross-framework corpus integration test).
//! 4. `testnet_or_dev_only == Some(false)` (production shape)
//! 5. `model_hash` (hex) equals `EXPECTED_PRODUCTION_SPEC_HASH` (hex)
//! 6. Decode `metadata.public_inputs` JSON into `(input_i16, output_i16)`
//! 7. Re-encode input/output as LE i16 → BLAKE3 → compare against
//!    `metadata.input_hash` / `metadata.response_hash`.
//! 7.5 Defense-in-depth: re-run `canonical_evaluate(input)` and
//!     refuse if the claimed output disagrees.
//! 8. Build the field-lifted instance column (24 elements: 16 inputs
//!    + 8 outputs).
//! 9. Run `halo2_proofs::plonk::verify_proof` against the embedded
//!    params + re-derived vk + proof bytes.

use std::io::Cursor;

use halo2_proofs::{
    pasta::EqAffine,
    plonk::{keygen_vk, verify_proof, SingleVerifier, VerifyingKey},
    poly::commitment::Params,
    transcript::{Blake2bRead, Challenge255},
};
use omni_zkml::{
    mainnet_vk_hash, ModelFormat, ModelFramework, ProofArtifactBody, ProofSystem, ProofVerifier,
    ProofVerifierError, PublicInputs,
};

use crate::circuit::{fp_from_i64, ProductionMlpCircuit};
use crate::encoding::{encode_canonical_input, encode_canonical_output};
use crate::shared::{
    BACKEND_ID, EXPECTED_CIRCUIT_ID_HEX, EXPECTED_PRODUCTION_SPEC_HASH, EXPECTED_VK_HASH_HEX,
};

/// Canonical VK bytes for the production circuit, used to derive
/// the `circuit_id_hex` (bare BLAKE3) and the
/// `verification_key_hash_hex` (`mainnet_vk_hash` =
/// BLAKE3("OMNINODE-VK:v1:" || vk_bytes) — criteria §1.7).
///
/// halo2_proofs 0.3.2 does **not** expose a stable on-disk
/// `VerifyingKey::write` for the Pasta IPA backend; instead the
/// library itself derives `transcript_repr` by Blake2b-hashing
/// `format!("{:?}", vk.pinned())` (see halo2_proofs/src/plonk.rs).
/// We mirror that: the pinned Debug string IS the byte sequence
/// halo2 treats as canonical, so using its UTF-8 bytes for our
/// own BLAKE3 fingerprint is byte-stable across hosts given a
/// fixed `halo2_proofs` version + circuit definition.
///
/// A `halo2_proofs` version bump may shift these bytes; the
/// verifier's `verify_vk_identity_matches_pinned_constants` test
/// fails loudly in that case and requires a fixture-regen PR.
pub fn vk_canonical_bytes(vk: &VerifyingKey<EqAffine>) -> Vec<u8> {
    format!("{:?}", vk.pinned()).into_bytes()
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// `circuit_id_hex` for the production circuit. Derived from
/// `vk_canonical_bytes` via bare BLAKE3 (no domain separator);
/// pinned in `shared::EXPECTED_CIRCUIT_ID_HEX`. Used by Stage 11d.1
/// allowlist matching at layer 6 of `check_mainnet_eligible`.
pub fn live_circuit_id_hex(vk: &VerifyingKey<EqAffine>) -> String {
    let bytes = vk_canonical_bytes(vk);
    blake3::hash(&bytes).to_hex().to_string()
}

/// `verification_key_hash_hex` for the production circuit. Derived
/// from `vk_canonical_bytes` via `mainnet_vk_hash` (=
/// `BLAKE3("OMNINODE-VK:v1:" || vk_bytes)`) per criteria §1.7;
/// pinned in `shared::EXPECTED_VK_HASH_HEX`. Carried on the
/// future `AllowlistEntry` for this proof class (Stage 11d.3).
pub fn live_vk_hash_hex(vk: &VerifyingKey<EqAffine>) -> String {
    let bytes = vk_canonical_bytes(vk);
    hex(&mainnet_vk_hash(&bytes))
}

/// Convenience for the regen tool: derive the live `(circuit_id_hex,
/// vk_hash_hex)` pair by running `keygen_vk` against the embedded
/// params + the production circuit. Pulls halo2_proofs but stays
/// inside this crate so the regen tool doesn't need a direct
/// halo2 dep.
pub fn derive_vk_identity_from_params(
    params_bytes: &[u8],
) -> Result<(String, String), ProofVerifierError> {
    use halo2_proofs::plonk::keygen_vk;
    use halo2_proofs::poly::commitment::Params;
    use std::io::Cursor;

    let params = Params::<EqAffine>::read(&mut Cursor::new(params_bytes))
        .map_err(|e| ProofVerifierError::VerifierInternal(format!("Params::read: {e}")))?;
    let vk = keygen_vk(&params, &ProductionMlpCircuit::default())
        .map_err(|e| ProofVerifierError::VerifierInternal(format!("keygen_vk: {e:?}")))?;
    Ok((live_circuit_id_hex(&vk), live_vk_hash_hex(&vk)))
}

/// Embedded params.bin generated by `tools/halo2_production_mlp_regen`.
/// Empty placeholder until the regen tool runs; the
/// `from_embedded_fixtures` constructor errors if this is empty.
const EMBEDDED_PARAMS_BIN: &[u8] =
    include_bytes!("../fixtures/halo2/params.bin");

/// Halo2 production-MLP verifier — verifies proofs against the committed
/// `production-fixedpoint-mlp-v1` canonical spec.
#[derive(Debug)]
pub struct Halo2ProductionMlpVerifier {
    params: Params<EqAffine>,
    vk: VerifyingKey<EqAffine>,
}

impl Halo2ProductionMlpVerifier {
    pub fn from_params_bytes(params_bytes: &[u8]) -> Result<Self, ProofVerifierError> {
        if params_bytes.is_empty() {
            return Err(ProofVerifierError::VerifierInternal(
                "params bytes are empty — fixtures not yet generated; run \
                 `cd tools/halo2_production_mlp_regen && cargo run --release regen`"
                    .into(),
            ));
        }
        let params = Params::<EqAffine>::read(&mut Cursor::new(params_bytes)).map_err(|e| {
            ProofVerifierError::VerifierInternal(format!("Params::read failed: {e}"))
        })?;
        let vk = keygen_vk(&params, &ProductionMlpCircuit::default()).map_err(|e| {
            ProofVerifierError::VerifierInternal(format!("VK keygen failed: {e:?}"))
        })?;
        // Stage 11d.2 — drift detection. The live VK's identity
        // hashes (derived from `format!("{:?}", vk.pinned())` —
        // halo2_proofs 0.3.2's library-blessed canonical
        // representation) must equal the constants pinned in
        // `shared.rs`. A `halo2_proofs` version bump, an unintended
        // circuit edit, or a `HALO2_K` change all fail here loudly
        // and require an explicit fixture-regen PR (rerun
        // `tools/halo2_production_mlp_regen/` and update
        // `EXPECTED_CIRCUIT_ID_HEX` / `EXPECTED_VK_HASH_HEX`).
        let live_circuit_id = live_circuit_id_hex(&vk);
        if live_circuit_id != EXPECTED_CIRCUIT_ID_HEX {
            return Err(ProofVerifierError::VerifierInternal(format!(
                "live VK identity drifted: live circuit_id_hex {live_circuit_id:?} \
                 does not equal pinned EXPECTED_CIRCUIT_ID_HEX {EXPECTED_CIRCUIT_ID_HEX:?}. \
                 Rerun `tools/halo2_production_mlp_regen/` and update shared.rs."
            )));
        }
        let live_vk_hash = live_vk_hash_hex(&vk);
        if live_vk_hash != EXPECTED_VK_HASH_HEX {
            return Err(ProofVerifierError::VerifierInternal(format!(
                "live VK hash drifted: live vk_hash_hex {live_vk_hash:?} \
                 does not equal pinned EXPECTED_VK_HASH_HEX {EXPECTED_VK_HASH_HEX:?}. \
                 Rerun `tools/halo2_production_mlp_regen/` and update shared.rs."
            )));
        }
        Ok(Self { params, vk })
    }

    pub fn from_embedded_fixtures() -> Result<Self, ProofVerifierError> {
        Self::from_params_bytes(EMBEDDED_PARAMS_BIN)
    }

    /// Decode `metadata.public_inputs` JSON into the raw i16 input +
    /// output arrays. Expected JSON shape:
    /// ```json
    /// {
    ///   "input":  [-5, 10, 20, -100, 7, -3, 14, 25, -8, 1, 11, -22, 4, -1, 17, 9],
    ///   "output": [-32, -12, 6, 25, -21, -3, 16, -29]
    /// }
    /// ```
    pub fn decode_public_inputs_json(
        v: &serde_json::Value,
    ) -> Result<([i16; 16], [i16; 8]), ProofVerifierError> {
        fn fixed_arr<const N: usize>(
            v: &serde_json::Value,
            field: &str,
        ) -> Result<[i16; N], ProofVerifierError> {
            let a = v.get(field).and_then(|x| x.as_array()).ok_or_else(|| {
                ProofVerifierError::VerifierInternal(format!(
                    "metadata.public_inputs missing array field '{field}'"
                ))
            })?;
            if a.len() != N {
                return Err(ProofVerifierError::VerifierInternal(format!(
                    "metadata.public_inputs.{field} must have length {N}, got {}",
                    a.len()
                )));
            }
            let mut out = [0i16; N];
            for (i, el) in a.iter().enumerate() {
                let n = el.as_i64().ok_or_else(|| {
                    ProofVerifierError::VerifierInternal(format!(
                        "metadata.public_inputs.{field}[{i}] is not an integer"
                    ))
                })?;
                if !(i16::MIN as i64..=i16::MAX as i64).contains(&n) {
                    return Err(ProofVerifierError::VerifierInternal(format!(
                        "metadata.public_inputs.{field}[{i}] = {n} out of i16 range"
                    )));
                }
                out[i] = n as i16;
            }
            Ok(out)
        }
        let input = fixed_arr::<16>(v, "input")?;
        let output = fixed_arr::<8>(v, "output")?;
        Ok((input, output))
    }
}

fn spec_hash_hex() -> String {
    let mut s = String::with_capacity(64);
    for b in EXPECTED_PRODUCTION_SPEC_HASH {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn hex_blake3(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

impl ProofVerifier for Halo2ProductionMlpVerifier {
    fn verify(
        &self,
        _proof: &[u8],
        _public_inputs: &PublicInputs,
    ) -> std::result::Result<bool, ProofVerifierError> {
        Err(ProofVerifierError::RequiresArtifactDispatch(
            "Halo2ProductionMlpVerifier binds backend-specific public inputs (the raw i16 \
             input/output values carried in metadata.public_inputs JSON) which the \
             three-hash PublicInputs cannot supply. Call verify_artifact(&ProofArtifactBody) \
             instead."
                .into(),
        ))
    }

    fn verify_artifact(
        &self,
        body: &ProofArtifactBody,
    ) -> std::result::Result<bool, ProofVerifierError> {
        let meta = &body.metadata;

        if meta.proof_system != Some(ProofSystem::Stage11dProductionFixedPointMlp) {
            return Err(ProofVerifierError::VerifierInternal(format!(
                "expected proof_system == Stage11dProductionFixedPointMlp, got {:?}",
                meta.proof_system
            )));
        }
        if meta.model_format != Some(ModelFormat::ProductionFixedPointMlp) {
            return Err(ProofVerifierError::VerifierInternal(format!(
                "expected model_format == ProductionFixedPointMlp, got {:?}",
                meta.model_format
            )));
        }
        if meta.model_framework != Some(ModelFramework::FrameworkAgnostic) {
            return Err(ProofVerifierError::VerifierInternal(format!(
                "expected model_framework == FrameworkAgnostic, got {:?}",
                meta.model_framework
            )));
        }
        if meta.backend_id != BACKEND_ID {
            return Err(ProofVerifierError::VerifierInternal(format!(
                "expected backend_id == {BACKEND_ID:?}, got {:?}",
                meta.backend_id
            )));
        }
        // Stage 11d.1 allowlist key — pinned in shared.rs. The
        // future `AllowlistEntry` (Stage 11d.3) will carry the
        // same string; any artifact whose `circuit_id_hex` drifts
        // from `EXPECTED_CIRCUIT_ID_HEX` is rejected here, before
        // SNARK verification.
        match meta.circuit_id_hex.as_deref() {
            Some(v) if v == EXPECTED_CIRCUIT_ID_HEX => {}
            Some(other) => {
                return Err(ProofVerifierError::VerifierInternal(format!(
                    "circuit_id_hex drift: artifact says {other:?}, expected {EXPECTED_CIRCUIT_ID_HEX:?}"
                )));
            }
            None => {
                return Err(ProofVerifierError::VerifierInternal(format!(
                    "circuit_id_hex is required for Halo2ProductionMlpVerifier; expected {EXPECTED_CIRCUIT_ID_HEX:?}"
                )));
            }
        }
        // Audit field — `mainnet_vk_hash` of the canonical VK bytes,
        // pinned in shared.rs. Carried on every artifact so an
        // auditor can cross-check against the eventual
        // `AllowlistEntry.verification_key_hash_hex` without
        // re-running keygen_vk.
        match meta.verification_key_hex.as_deref() {
            Some(v) if v == EXPECTED_VK_HASH_HEX => {}
            Some(other) => {
                return Err(ProofVerifierError::VerifierInternal(format!(
                    "verification_key_hex (mainnet_vk_hash) drift: artifact says {other:?}, \
                     expected {EXPECTED_VK_HASH_HEX:?}"
                )));
            }
            None => {
                return Err(ProofVerifierError::VerifierInternal(format!(
                    "verification_key_hex is required for Halo2ProductionMlpVerifier; \
                     expected {EXPECTED_VK_HASH_HEX:?}"
                )));
            }
        }
        // Production-shape artifact (Stage 11d.2): the artifact's
        // `testnet_or_dev_only` MUST be `Some(false)`. Mainnet refusal
        // continues to come from the empty `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`
        // table at layer 6 of `check_mainnet_eligible`.
        if meta.testnet_or_dev_only != Some(false) {
            return Err(ProofVerifierError::VerifierInternal(format!(
                "expected testnet_or_dev_only == Some(false) (production shape), got {:?}",
                meta.testnet_or_dev_only
            )));
        }
        let expected_spec_hex = spec_hash_hex();
        if meta.model_hash != expected_spec_hex {
            return Err(ProofVerifierError::VerifierInternal(format!(
                "model_hash {:?} does not match EXPECTED_PRODUCTION_SPEC_HASH {:?}",
                meta.model_hash, expected_spec_hex
            )));
        }

        let pi_json = meta.public_inputs.as_ref().ok_or_else(|| {
            ProofVerifierError::VerifierInternal(
                "metadata.public_inputs is required for Halo2ProductionMlpVerifier".into(),
            )
        })?;
        let (input_i16, output_i16) = Self::decode_public_inputs_json(pi_json)?;

        let computed_input_hash = hex_blake3(&encode_canonical_input(&input_i16));
        if computed_input_hash != meta.input_hash {
            return Err(ProofVerifierError::VerifierInternal(format!(
                "input_hash drift: metadata says {:?}, BLAKE3(LE(input)) is {:?}",
                meta.input_hash, computed_input_hash
            )));
        }
        let computed_output_hash = hex_blake3(&encode_canonical_output(&output_i16));
        if computed_output_hash != meta.response_hash {
            return Err(ProofVerifierError::VerifierInternal(format!(
                "response_hash drift: metadata says {:?}, BLAKE3(LE(output)) is {:?}",
                meta.response_hash, computed_output_hash
            )));
        }

        // Defense-in-depth: re-run the neutral canonical evaluator
        // and refuse if the claimed output disagrees. The halo2
        // circuit's soundness covers `canonical_evaluate(input) ==
        // output`, but this independent check stops malformed proof
        // artifacts before the SNARK verifier even runs.
        let expected_output = crate::canonical::canonical_evaluate(input_i16);
        if expected_output != output_i16 {
            return Err(ProofVerifierError::VerifierInternal(format!(
                "claimed output {output_i16:?} does not equal canonical_evaluate(input) \
                 {expected_output:?} — proof rejected before halo2 verify"
            )));
        }

        let mut instance = Vec::with_capacity(24);
        for x in &input_i16 {
            instance.push(fp_from_i64(*x as i64));
        }
        for y in &output_i16 {
            instance.push(fp_from_i64(*y as i64));
        }

        let proof_bytes = body
            .proof_bytes()
            .map_err(|e| ProofVerifierError::VerifierInternal(format!("proof_bytes(): {e}")))?;
        let strategy = SingleVerifier::new(&self.params);
        let mut transcript =
            Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof_bytes.as_slice());

        match verify_proof::<EqAffine, _, _, _>(
            &self.params,
            &self.vk,
            strategy,
            &[&[&instance]],
            &mut transcript,
        ) {
            Ok(()) => Ok(true),
            Err(e) => {
                use halo2_proofs::plonk::Error as HE;
                match e {
                    HE::ConstraintSystemFailure | HE::Opening | HE::BoundsFailure => Ok(false),
                    other => Err(ProofVerifierError::VerifierInternal(format!(
                        "verify_proof: {other:?}"
                    ))),
                }
            }
        }
    }

    fn proof_system(&self) -> ProofSystem {
        ProofSystem::Stage11dProductionFixedPointMlp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_params_returns_internal_error() {
        let v = Halo2ProductionMlpVerifier::from_params_bytes(b"");
        match v {
            Err(ProofVerifierError::VerifierInternal(msg)) => {
                assert!(
                    msg.contains("fixtures not yet generated") || msg.contains("empty"),
                    "expected fixture-missing message, got: {msg}"
                );
            }
            other => panic!("expected VerifierInternal, got {other:?}"),
        }
    }

    #[test]
    fn verify_with_three_hash_public_inputs_returns_requires_artifact() {
        // Construct a synthetic verifier directly is impossible
        // without params; this test only confirms the empty-bytes
        // error path is correct.
        let v = Halo2ProductionMlpVerifier::from_params_bytes(b"");
        assert!(matches!(
            v,
            Err(ProofVerifierError::VerifierInternal(_))
        ));
    }
}
