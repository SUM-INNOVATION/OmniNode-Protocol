//! Stage 11d.2 — `FrameworkManifest` schema for the production
//! fixed-point MLP proof class.
//!
//! Mirrors `omni-proofs-halo2-reference::manifest` (the bounded
//! reference's Stage 11b.1.a schema) but typed to the production
//! tensor dims (input `[i16; 16]`, output `[i16; 8]`). A separate
//! struct keeps the two manifest universes type-isolated:
//! a bounded-reference fixture cannot be mis-loaded into the
//! production cross-framework test, and vice-versa.
//!
//! Per Stage 11d.2 plan §6 four-equal-primaries posture, four
//! framework manifests (RUMUS, PyTorch, TensorFlow, Caffe) plus
//! one framework-agnostic schema-coverage manifest are committed
//! under `tests/fixtures/`. The cross-framework equivalence test
//! loads each manifest, asserts arithmetic equivalence against
//! `canonical_evaluate`, and validates the optional hash fields
//! when present.

use omni_zkml::ModelFramework;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Provenance mode for a [`FrameworkManifest`]. Carries the same
/// semantics as the bounded reference's `GenerationMode` — restated
/// here so the production crate has no semantic dependency on the
/// reference's enum surface and the two manifest universes can
/// evolve independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenerationMode {
    /// Manifest is the live output of running the framework via a
    /// developer-host exporter under
    /// `tools/halo2_production_mlp_<framework>_export/`.
    LiveExport,
    /// Caffe-specific: the host did NOT have a working Caffe Python
    /// binding, so the exporter fell back to a pure-NumPy emulation
    /// of the canonical contract. Arithmetic is identical to the
    /// spec; this variant exists purely to make the fallback
    /// auditable. `generator_metadata.runtime_mode` must be
    /// `"pure-numpy-emulation"` when this variant is used.
    PureNumpyCompatibility,
    /// Manifest is hand-authored to represent what the framework
    /// SHOULD produce, with no runtime currently capable of
    /// emitting it. Reserved as a placeholder; Stage 11d.2 does
    /// not commit fixtures in this mode under the four-equal-
    /// primaries posture.
    IntendedRepresentation,
    /// Manifest is curated by a developer based on real framework
    /// output but without a ready-to-run regen tool in the repo.
    /// Used by `framework_agnostic_manifest.json` only.
    ManualFixture,
}

/// On-disk shape of a production-tier framework fixture manifest.
///
/// Loaded from `crates/omni-proofs-halo2-production-mlp/tests/
/// fixtures/<framework>_corpus.json`. The shape is intentionally
/// minimal: just enough for the cross-framework equivalence test to
/// assert equivalence against the canonical evaluator.
///
/// **No framework runtime is ever invoked when parsing or asserting
/// against a manifest.** The struct is pure data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameworkManifest {
    /// Which framework this manifest comes from. Maps 1:1 to
    /// [`omni_zkml::ModelFramework`].
    pub framework: ModelFramework,
    /// Free-form version string the developer recorded.
    pub framework_version: String,
    /// Hex of BLAKE3 of `assets/canonical_spec.json` — equal to
    /// `EXPECTED_PRODUCTION_SPEC_HASH`. Required on every manifest.
    pub weights_hash: String,
    /// Canonical input tensor. MUST equal
    /// [`crate::shared::CANONICAL_INPUT`].
    pub input: [i16; 16],
    /// Output tensor reported by the framework. MUST equal
    /// `canonical_evaluate(input)` byte-for-byte.
    pub output: [i16; 8],
    /// ISO 8601 UTC timestamp when the manifest was generated.
    pub generated_at_utc: String,
    /// Free-form "who/what generated this".
    pub generated_by: String,
    /// Provenance mode for this manifest. See [`GenerationMode`].
    pub generation_mode: GenerationMode,
    /// Optional alias of `weights_hash` — hex of BLAKE3 of
    /// `canonical_spec.json`. When present, MUST equal
    /// `weights_hash`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub spec_hash: Option<String>,
    /// Optional hex of BLAKE3 of the LE-encoded input tensor
    /// (32 bytes). When present, MUST equal
    /// `blake3(encode_canonical_input(input))`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub input_hash: Option<String>,
    /// Optional hex of BLAKE3 of the LE-encoded output tensor
    /// (16 bytes). When present, MUST equal
    /// `blake3(encode_canonical_output(output))`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub output_hash: Option<String>,
    /// Optional structured generator-metadata block. Free-form
    /// JSON; recorded for audit.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub generator_metadata: Option<serde_json::Value>,
    /// Optional free-form notes.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub notes: Option<String>,
}

/// Errors loading or validating a [`FrameworkManifest`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ManifestError {
    #[error("manifest input {got:?} does not equal canonical input {canonical:?}")]
    InputMismatch { got: [i16; 16], canonical: [i16; 16] },

    #[error(
        "manifest output {got:?} does not match canonical evaluator output {canonical:?} \
         — framework reproduced inconsistent arithmetic"
    )]
    OutputMismatch { got: [i16; 8], canonical: [i16; 8] },

    #[error(
        "manifest weights_hash {got:?} does not match canonical spec hash {expected:?}"
    )]
    WeightsHashMismatch { got: String, expected: String },

    #[error("manifest framework field {got:?} does not match expected {expected:?}")]
    FrameworkMismatch {
        got: ModelFramework,
        expected: ModelFramework,
    },

    #[error(
        "manifest spec_hash {got:?} disagrees with weights_hash {weights_hash:?}; both must \
         carry the same canonical-spec BLAKE3"
    )]
    SpecHashAliasMismatch { got: String, weights_hash: String },

    #[error("manifest spec_hash {got:?} does not match canonical spec hash {expected:?}")]
    SpecHashMismatch { got: String, expected: String },

    #[error(
        "manifest input_hash {got:?} does not match BLAKE3 of LE-encoded input ({expected:?})"
    )]
    InputHashMismatch { got: String, expected: String },

    #[error(
        "manifest output_hash {got:?} does not match BLAKE3 of LE-encoded output ({expected:?})"
    )]
    OutputHashMismatch { got: String, expected: String },
}

fn hex_blake3(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

impl FrameworkManifest {
    /// Validate a manifest against the canonical spec + evaluator.
    /// This is what the cross-framework integration test calls.
    pub fn validate_against_canonical(
        &self,
        expected_framework: ModelFramework,
        expected_spec_hash_hex: &str,
    ) -> Result<(), ManifestError> {
        use crate::canonical::canonical_evaluate;
        use crate::encoding::{encode_canonical_input, encode_canonical_output};
        use crate::shared::CANONICAL_INPUT;

        if self.framework != expected_framework {
            return Err(ManifestError::FrameworkMismatch {
                got: self.framework,
                expected: expected_framework,
            });
        }
        if self.input != CANONICAL_INPUT {
            return Err(ManifestError::InputMismatch {
                got: self.input,
                canonical: CANONICAL_INPUT,
            });
        }
        let canonical_output = canonical_evaluate(self.input);
        if self.output != canonical_output {
            return Err(ManifestError::OutputMismatch {
                got: self.output,
                canonical: canonical_output,
            });
        }
        if self.weights_hash != expected_spec_hash_hex {
            return Err(ManifestError::WeightsHashMismatch {
                got: self.weights_hash.clone(),
                expected: expected_spec_hash_hex.to_string(),
            });
        }
        if let Some(spec_hash) = &self.spec_hash {
            if spec_hash != &self.weights_hash {
                return Err(ManifestError::SpecHashAliasMismatch {
                    got: spec_hash.clone(),
                    weights_hash: self.weights_hash.clone(),
                });
            }
            if spec_hash != expected_spec_hash_hex {
                return Err(ManifestError::SpecHashMismatch {
                    got: spec_hash.clone(),
                    expected: expected_spec_hash_hex.to_string(),
                });
            }
        }
        if let Some(input_hash) = &self.input_hash {
            let expected = hex_blake3(&encode_canonical_input(&self.input));
            if input_hash != &expected {
                return Err(ManifestError::InputHashMismatch {
                    got: input_hash.clone(),
                    expected,
                });
            }
        }
        if let Some(output_hash) = &self.output_hash {
            let expected = hex_blake3(&encode_canonical_output(&self.output));
            if output_hash != &expected {
                return Err(ManifestError::OutputHashMismatch {
                    got: output_hash.clone(),
                    expected,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::canonical_evaluate;
    use crate::encoding::{encode_canonical_input, encode_canonical_output};
    use crate::shared::{CANONICAL_INPUT, EXPECTED_PRODUCTION_SPEC_HASH};

    fn expected_spec_hash_hex() -> String {
        let mut s = String::with_capacity(64);
        for b in EXPECTED_PRODUCTION_SPEC_HASH {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }

    fn ok_manifest(framework: ModelFramework) -> FrameworkManifest {
        let h = expected_spec_hash_hex();
        let output = canonical_evaluate(CANONICAL_INPUT);
        FrameworkManifest {
            framework,
            framework_version: "test-only".into(),
            weights_hash: h.clone(),
            input: CANONICAL_INPUT,
            output,
            generated_at_utc: "2026-05-22T00:00:00Z".into(),
            generated_by: "test".into(),
            generation_mode: GenerationMode::LiveExport,
            spec_hash: Some(h),
            input_hash: Some(hex_blake3(&encode_canonical_input(&CANONICAL_INPUT))),
            output_hash: Some(hex_blake3(&encode_canonical_output(&output))),
            generator_metadata: None,
            notes: None,
        }
    }

    #[test]
    fn validate_accepts_correct_manifest() {
        let m = ok_manifest(ModelFramework::PyTorch);
        m.validate_against_canonical(ModelFramework::PyTorch, &expected_spec_hash_hex())
            .unwrap();
    }

    #[test]
    fn validate_accepts_manifest_without_optional_hash_fields() {
        let mut m = ok_manifest(ModelFramework::PyTorch);
        m.spec_hash = None;
        m.input_hash = None;
        m.output_hash = None;
        m.validate_against_canonical(ModelFramework::PyTorch, &expected_spec_hash_hex())
            .unwrap();
    }

    #[test]
    fn validate_rejects_wrong_framework() {
        let m = ok_manifest(ModelFramework::PyTorch);
        let err = m
            .validate_against_canonical(ModelFramework::TensorFlow, &expected_spec_hash_hex())
            .unwrap_err();
        assert!(matches!(err, ManifestError::FrameworkMismatch { .. }), "got {err:?}");
    }

    #[test]
    fn validate_rejects_wrong_input() {
        let mut m = ok_manifest(ModelFramework::PyTorch);
        m.input = [0i16; 16];
        let err = m
            .validate_against_canonical(ModelFramework::PyTorch, &expected_spec_hash_hex())
            .unwrap_err();
        assert!(matches!(err, ManifestError::InputMismatch { .. }), "got {err:?}");
    }

    #[test]
    fn validate_rejects_wrong_output() {
        let mut m = ok_manifest(ModelFramework::PyTorch);
        m.output = [0i16; 8];
        let err = m
            .validate_against_canonical(ModelFramework::PyTorch, &expected_spec_hash_hex())
            .unwrap_err();
        assert!(matches!(err, ManifestError::OutputMismatch { .. }), "got {err:?}");
    }

    #[test]
    fn validate_rejects_wrong_weights_hash() {
        let mut m = ok_manifest(ModelFramework::PyTorch);
        m.weights_hash = "0".repeat(64);
        let err = m
            .validate_against_canonical(ModelFramework::PyTorch, &expected_spec_hash_hex())
            .unwrap_err();
        assert!(matches!(err, ManifestError::WeightsHashMismatch { .. }), "got {err:?}");
    }

    #[test]
    fn validate_rejects_spec_hash_disagreeing_with_weights_hash() {
        let mut m = ok_manifest(ModelFramework::PyTorch);
        m.spec_hash = Some("f".repeat(64));
        let err = m
            .validate_against_canonical(ModelFramework::PyTorch, &expected_spec_hash_hex())
            .unwrap_err();
        assert!(matches!(err, ManifestError::SpecHashAliasMismatch { .. }), "got {err:?}");
    }

    #[test]
    fn validate_rejects_wrong_input_hash() {
        let mut m = ok_manifest(ModelFramework::PyTorch);
        m.input_hash = Some("0".repeat(64));
        let err = m
            .validate_against_canonical(ModelFramework::PyTorch, &expected_spec_hash_hex())
            .unwrap_err();
        assert!(matches!(err, ManifestError::InputHashMismatch { .. }), "got {err:?}");
    }

    #[test]
    fn validate_rejects_wrong_output_hash() {
        let mut m = ok_manifest(ModelFramework::PyTorch);
        m.output_hash = Some("0".repeat(64));
        let err = m
            .validate_against_canonical(ModelFramework::PyTorch, &expected_spec_hash_hex())
            .unwrap_err();
        assert!(matches!(err, ManifestError::OutputHashMismatch { .. }), "got {err:?}");
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let m = ok_manifest(ModelFramework::FrameworkAgnostic);
        let s = serde_json::to_string(&m).unwrap();
        let back: FrameworkManifest = serde_json::from_str(&s).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn generation_mode_variants_round_trip() {
        for mode in [
            GenerationMode::LiveExport,
            GenerationMode::PureNumpyCompatibility,
            GenerationMode::IntendedRepresentation,
            GenerationMode::ManualFixture,
        ] {
            let s = serde_json::to_string(&mode).unwrap();
            let back: GenerationMode = serde_json::from_str(&s).unwrap();
            assert_eq!(mode, back);
        }
    }
}
