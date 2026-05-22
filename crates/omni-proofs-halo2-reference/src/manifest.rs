//! Stage 11b.1.a — `FrameworkManifest` schema.
//!
//! Every framework fixture (`tests/fixtures/*.json`) deserializes
//! into this struct. The cross-framework equivalence test loads
//! each manifest, asserts `manifest.output == canonical_evaluate(
//! manifest.input)`, and asserts every manifest's `weights_hash`
//! matches the canonical spec's frozen `EXPECTED_SPEC_HASH`.
//!
//! Per Stage 11b.1.a OQ4: the manifest carries a `generation_mode`
//! field that distinguishes:
//!   * `LiveExport`  — manifest produced by running the framework
//!                     (PyTorch, TensorFlow). Reproducible by
//!                     re-running the export tool.
//!   * `IntendedRepresentation` — manifest is hand-authored to
//!                     match what the framework SHOULD produce.
//!                     RUMUS uses this until RUMUS exposes a
//!                     deterministic CPU fixed-point integer-dense
//!                     path.
//!   * `ManualFixture` — manifest is a developer-host-curated file
//!                     based on real framework output but without a
//!                     ready-to-run regen tool in repo (Caffe is
//!                     the typical case).
//!
//! No framework runtime appears at OmniNode runtime regardless of
//! the mode. These distinctions live purely in fixture provenance.

use omni_zkml::ModelFramework;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Provenance mode for a [`FrameworkManifest`]. Recorded explicitly
/// so a future contributor revisiting the fixtures can tell why a
/// particular framework's manifest was produced the way it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenerationMode {
    /// Manifest is the live output of running the framework via a
    /// developer-host script (e.g. `tools/pytorch_export.py`).
    /// Reproducible by re-running the tool.
    LiveExport,
    /// Manifest is hand-authored to represent what the framework
    /// SHOULD produce, but the framework currently lacks the
    /// runtime support needed to verify (e.g. RUMUS has no
    /// deterministic CPU integer-dense path as of Stage 11b.1.a).
    IntendedRepresentation,
    /// Manifest is curated by a developer based on real framework
    /// output but without a ready-to-run regen tool in the repo
    /// (e.g. Caffe — `.prototxt` + manual extraction).
    ManualFixture,
}

/// On-disk shape of a framework fixture manifest.
///
/// Loaded from `crates/omni-proofs-halo2-reference/tests/fixtures/
/// <framework>_manifest.json`. The shape is intentionally minimal:
/// just enough for the cross-framework equivalence test to assert
/// equivalence against the canonical evaluator.
///
/// **No framework runtime is ever invoked when parsing or asserting
/// against a manifest.** The struct is pure data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameworkManifest {
    /// Which framework this manifest comes from. Maps 1:1 to
    /// [`omni_zkml::ModelFramework`].
    pub framework: ModelFramework,
    /// Free-form version string the developer recorded (e.g.
    /// `"PyTorch 2.4.0"` or `"RUMUS 0.x (deferred — intended-only)"`).
    /// Pinned per-regen; not parsed by the equivalence test.
    pub framework_version: String,
    /// Hex of BLAKE3 of the framework's canonical-spec representation.
    /// MUST equal the `EXPECTED_SPEC_HASH` constant (`build.rs`
    /// computes that from `assets/canonical_spec.json`). The
    /// cross-framework test asserts this.
    pub weights_hash: String,
    /// Canonical input tensor. MUST equal
    /// [`crate::shared::CANONICAL_INPUT`].
    pub input: [i16; 4],
    /// Output tensor reported by the framework. MUST equal
    /// `canonical_evaluate(input)` byte-for-byte. The cross-framework
    /// test asserts this; a framework that diverges fails CI.
    pub output: [i16; 4],
    /// ISO 8601 UTC timestamp when the manifest was generated.
    /// Informational only — not asserted by the test, but recorded
    /// for fixture provenance.
    pub generated_at_utc: String,
    /// Free-form "who/what generated this" — developer-host details,
    /// script paths, machine names, etc. Recorded for provenance.
    pub generated_by: String,
    /// Provenance mode for this manifest. See [`GenerationMode`].
    pub generation_mode: GenerationMode,
    /// Optional free-form notes — e.g. RUMUS's manifest explains
    /// why it's an `IntendedRepresentation` rather than `LiveExport`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub notes: Option<String>,
}

/// Errors loading or validating a [`FrameworkManifest`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ManifestError {
    #[error("manifest input {got:?} does not equal canonical input {canonical:?}")]
    InputMismatch { got: [i16; 4], canonical: [i16; 4] },

    #[error(
        "manifest output {got:?} does not match canonical evaluator output {canonical:?} \
         — framework reproduced inconsistent arithmetic"
    )]
    OutputMismatch { got: [i16; 4], canonical: [i16; 4] },

    #[error(
        "manifest weights_hash {got:?} does not match canonical spec hash {expected:?}"
    )]
    WeightsHashMismatch { got: String, expected: String },

    #[error("manifest framework field {got:?} does not match expected {expected:?}")]
    FrameworkMismatch {
        got: ModelFramework,
        expected: ModelFramework,
    },
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::canonical_evaluate;
    use crate::shared::{CANONICAL_INPUT, CANONICAL_OUTPUT, EXPECTED_SPEC_HASH};

    fn expected_spec_hash_hex() -> String {
        let mut s = String::with_capacity(64);
        for b in EXPECTED_SPEC_HASH {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }

    fn ok_manifest(framework: ModelFramework) -> FrameworkManifest {
        FrameworkManifest {
            framework,
            framework_version: "test-only".into(),
            weights_hash: expected_spec_hash_hex(),
            input: CANONICAL_INPUT,
            output: CANONICAL_OUTPUT,
            generated_at_utc: "2026-05-22T00:00:00Z".into(),
            generated_by: "test".into(),
            generation_mode: GenerationMode::LiveExport,
            notes: None,
        }
    }

    #[test]
    fn validate_accepts_correct_manifest() {
        let m = ok_manifest(ModelFramework::PyTorch);
        m.validate_against_canonical(ModelFramework::PyTorch, &expected_spec_hash_hex()).unwrap();
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
        m.input = [0, 0, 0, 0];
        let err = m
            .validate_against_canonical(ModelFramework::PyTorch, &expected_spec_hash_hex())
            .unwrap_err();
        assert!(matches!(err, ManifestError::InputMismatch { .. }), "got {err:?}");
    }

    #[test]
    fn validate_rejects_wrong_output() {
        let mut m = ok_manifest(ModelFramework::PyTorch);
        m.output = [0, 0, 0, 0];
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
            GenerationMode::IntendedRepresentation,
            GenerationMode::ManualFixture,
        ] {
            let s = serde_json::to_string(&mode).unwrap();
            let back: GenerationMode = serde_json::from_str(&s).unwrap();
            assert_eq!(mode, back);
        }
    }

    #[test]
    fn canonical_output_matches_canonical_evaluator() {
        // Defense in depth — same content as the canonical_evaluator
        // test, but lives here so the manifest module also fails
        // loudly if the constants drift.
        assert_eq!(canonical_evaluate(CANONICAL_INPUT), CANONICAL_OUTPUT);
    }
}
