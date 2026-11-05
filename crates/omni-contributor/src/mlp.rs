//! #131 item 1(a) — the in-process fixed-point MLP runner.
//!
//! [`FixedPointMlpRunner`] evaluates a [`FixedPointMlp`] (from the
//! dependency-free `omni-fixedpoint-mlp` crate) inside this process. It
//! serves the canonical model, `production-fixedpoint-mlp-v1`; no weight-file
//! loader is wired.
//!
//! **Model binding (exact bytes).** The runner serves one model identity,
//! `BLAKE3` of the model file's exact bytes. A run refuses unless the
//! manifest it is handed hashes to that identity: the manifest bytes are
//! the model, so a job cannot point the runner at other weights. For the
//! canonical model the identity is [`CANONICAL_SPEC_BLAKE3_HEX`], and the
//! manifest is `canonical_spec.json` itself.
//!
//! **Accounting, derived from the validated model:**
//! - measured input = the model's input elements, measured output = its
//!   output elements; the orchestrator requires the input count to equal the
//!   job's `input_token_count` and the output count to be within its cap;
//! - one stage contribution, [`WorkUnitKind::FlopsEstimate`], of `2 ×` the
//!   model's multiply-accumulates.
//!
//! For the canonical model: 16 / 8 elements, 24 base units, 2,304 FLOPs.
//!
//! **Audit metadata.** An MLP has no tokenizer. An MLP job's accounting
//! carries `tokenizer_hash = model_hash` (the model file fixes the input
//! encoding) and `tokenizer_id = `[`MLP_TOKENIZER_ID`];
//! [`FixedPointMlpRunner::job_accounting`] builds it, and
//! [`FixedPointMlpRunner::check_job`] refuses a job that differs.
//!
//! Proofs exist for the canonical model only (the halo2 production proof
//! crate); this runner produces no proof.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use omni_fixedpoint_mlp::{CANONICAL_SPEC_BLAKE3_HEX, FixedPointMlp, MlpAccounting};
use thiserror::Error;

use crate::error::RunnerError;
use crate::job::{BaseUnitRewardPolicy, ContributorJob, JobAccounting};
use crate::result::{StageContribution, WorkUnitKind};
use crate::runner::{InferenceRunner, RunOutput};

/// `tokenizer_id` of an MLP job: the input and output are `i16`
/// little-endian elements, counted as elements.
pub const MLP_TOKENIZER_ID: &str = "omninode/i16-le-elements";

/// `stage_label` of the MLP runner's stage contribution.
pub const MLP_STAGE_LABEL: &str = "fixedpoint-mlp";

/// The most manifest bytes the runner reads before hashing them. The
/// canonical `canonical_spec.json` is 9,320 bytes.
pub const MAX_MLP_MANIFEST_BYTES: u64 = 1 << 20;

/// Why a job is not one the MLP runner can account for.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MlpJobError {
    #[error("job model_hash {got} is not the model this runner serves ({expected})")]
    ModelHash { expected: String, got: String },
    #[error("job tokenizer_hash {got} must equal the model hash {expected} for an MLP job")]
    TokenizerHash { expected: String, got: String },
    #[error("job tokenizer_id {got:?} must be {expected:?} for an MLP job")]
    TokenizerId { expected: &'static str, got: String },
    #[error("job input_token_count {got} must equal the model's {expected} input elements")]
    InputCount { expected: u64, got: u64 },
    #[error("job max_output_token_count {got} is below the model's {expected} output elements")]
    OutputCap { expected: u64, got: u64 },
}

/// The in-process fixed-point MLP runner. See the module docs.
#[derive(Debug, Clone)]
pub struct FixedPointMlpRunner {
    model: FixedPointMlp,
    model_hash_hex: String,
    accounting: MlpAccounting,
    contributor_pubkey_hex: String,
}

impl FixedPointMlpRunner {
    /// The runner for the canonical model, `production-fixedpoint-mlp-v1`,
    /// whose identity is [`CANONICAL_SPEC_BLAKE3_HEX`].
    /// `contributor_pubkey_hex` is attributed the stage, as for every runner.
    pub fn canonical(contributor_pubkey_hex: String) -> Self {
        let model = FixedPointMlp::canonical();
        let accounting = model
            .accounting()
            .expect("the canonical model's counts fit in u64");
        Self {
            model,
            model_hash_hex: CANONICAL_SPEC_BLAKE3_HEX.to_string(),
            accounting,
            contributor_pubkey_hex,
        }
    }

    /// The model identity this runner serves: BLAKE3 of the model file's
    /// exact bytes, 64 lowercase hex.
    pub fn model_hash_hex(&self) -> &str {
        &self.model_hash_hex
    }

    /// The model this runner evaluates.
    pub fn model(&self) -> &FixedPointMlp {
        &self.model
    }

    /// What one run reports, derived from the model.
    pub fn accounting(&self) -> MlpAccounting {
        self.accounting
    }

    /// The job accounting a dispatcher writes for this model:
    /// `tokenizer_hash = model_hash`, `tokenizer_id = MLP_TOKENIZER_ID`, the
    /// input count exactly, and the output count as the cap.
    pub fn job_accounting(&self) -> JobAccounting {
        JobAccounting {
            tokenizer_hash: self.model_hash_hex.clone(),
            tokenizer_id: MLP_TOKENIZER_ID.to_string(),
            input_token_count: self.accounting.input_elements,
            max_output_token_count: self.accounting.output_elements,
            base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
        }
    }

    /// Refuse a job this runner cannot account for: another `model_hash`,
    /// audit metadata other than [`Self::job_accounting`]'s, an input count
    /// other than the model's, or an output cap below the model's output.
    /// The run itself checks the manifest bytes.
    pub fn check_job(&self, job: &ContributorJob) -> Result<(), MlpJobError> {
        if job.model_hash != self.model_hash_hex {
            return Err(MlpJobError::ModelHash {
                expected: self.model_hash_hex.clone(),
                got: job.model_hash.clone(),
            });
        }
        let a = &job.accounting;
        if a.tokenizer_hash != self.model_hash_hex {
            return Err(MlpJobError::TokenizerHash {
                expected: self.model_hash_hex.clone(),
                got: a.tokenizer_hash.clone(),
            });
        }
        if a.tokenizer_id != MLP_TOKENIZER_ID {
            return Err(MlpJobError::TokenizerId {
                expected: MLP_TOKENIZER_ID,
                got: a.tokenizer_id.clone(),
            });
        }
        if a.input_token_count != self.accounting.input_elements {
            return Err(MlpJobError::InputCount {
                expected: self.accounting.input_elements,
                got: a.input_token_count,
            });
        }
        if a.max_output_token_count < self.accounting.output_elements {
            return Err(MlpJobError::OutputCap {
                expected: self.accounting.output_elements,
                got: a.max_output_token_count,
            });
        }
        Ok(())
    }

    /// Read at most [`MAX_MLP_MANIFEST_BYTES`] of the manifest.
    fn read_manifest(path: &Path) -> Result<Vec<u8>, RunnerError> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len > MAX_MLP_MANIFEST_BYTES {
            return Err(RunnerError::ModelManifestTooLarge {
                len,
                limit: MAX_MLP_MANIFEST_BYTES,
            });
        }
        let mut bytes = Vec::with_capacity(len as usize);
        file.take(MAX_MLP_MANIFEST_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_MLP_MANIFEST_BYTES {
            return Err(RunnerError::ModelManifestTooLarge {
                len: bytes.len() as u64,
                limit: MAX_MLP_MANIFEST_BYTES,
            });
        }
        Ok(bytes)
    }
}

impl InferenceRunner for FixedPointMlpRunner {
    fn run(&self, manifest_path: &Path, input_bytes: &[u8]) -> Result<RunOutput, RunnerError> {
        let manifest = Self::read_manifest(manifest_path)?;
        let got = blake3::hash(&manifest).to_hex().to_string();
        if got != self.model_hash_hex {
            return Err(RunnerError::ModelManifestMismatch {
                expected: self.model_hash_hex.clone(),
                got,
            });
        }
        let response_bytes = self
            .model
            .evaluate_bytes(input_bytes)
            .map_err(|e| RunnerError::ModelInput(e.to_string()))?;
        Ok(RunOutput {
            response_bytes,
            measured_input_tokens: self.accounting.input_elements,
            measured_output_tokens: self.accounting.output_elements,
            stage_contributions: vec![StageContribution {
                contributor_pubkey_hex: self.contributor_pubkey_hex.clone(),
                stage_label: MLP_STAGE_LABEL.to_string(),
                work_unit_kind: WorkUnitKind::FlopsEstimate,
                work_units: self.accounting.flops_estimate,
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const PUBKEY: &str = "ab";

    /// The canonical spec file, from the proof crate that owns it.
    fn canonical_spec() -> Vec<u8> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../omni-proofs-halo2-production-mlp/assets/canonical_spec.json");
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    }

    fn manifest_file(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    fn le(values: &[i16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    const CANONICAL_INPUT: [i16; 16] = [
        -5, 10, 20, -100, 7, -3, 14, 25, -8, 1, 11, -22, 4, -1, 17, 9,
    ];
    const CANONICAL_OUTPUT: [i16; 8] = [-32, -12, 6, 25, -21, -3, 16, -29];

    #[test]
    fn the_canonical_runner_serves_blake3_of_the_spec_file() {
        let r = FixedPointMlpRunner::canonical(PUBKEY.into());
        assert_eq!(
            r.model_hash_hex(),
            blake3::hash(&canonical_spec()).to_hex().as_str()
        );
    }

    /// The canonical model: 16 / 8 elements, 24 base units, FlopsEstimate 2,304.
    #[test]
    fn a_run_reports_the_canonical_accounting() {
        let r = FixedPointMlpRunner::canonical(PUBKEY.into());
        let spec = manifest_file(&canonical_spec());
        let out = r.run(spec.path(), &le(&CANONICAL_INPUT)).unwrap();
        assert_eq!(out.response_bytes, le(&CANONICAL_OUTPUT));
        assert_eq!(out.measured_input_tokens, 16);
        assert_eq!(out.measured_output_tokens, 8);
        assert_eq!(
            out.measured_input_tokens + out.measured_output_tokens,
            24,
            "base units"
        );
        assert_eq!(
            out.stage_contributions,
            vec![StageContribution {
                contributor_pubkey_hex: PUBKEY.into(),
                stage_label: MLP_STAGE_LABEL.into(),
                work_unit_kind: WorkUnitKind::FlopsEstimate,
                work_units: 2_304,
            }]
        );
        assert_eq!(r.accounting().base_units, 24);
    }

    #[test]
    fn the_job_accounting_carries_the_audit_metadata() {
        let r = FixedPointMlpRunner::canonical(PUBKEY.into());
        let a = r.job_accounting();
        assert_eq!(a.tokenizer_hash, CANONICAL_SPEC_BLAKE3_HEX);
        assert_eq!(a.tokenizer_id, "omninode/i16-le-elements");
        assert_eq!((a.input_token_count, a.max_output_token_count), (16, 8));
        assert_eq!(a.base_unit_reward_policy, BaseUnitRewardPolicy::Unspecified);
    }

    /// The manifest's exact bytes are the model: one changed bit, a missing
    /// byte, an extra byte or another file is refused, before any evaluation.
    #[test]
    fn a_manifest_that_is_not_the_served_model_is_refused() {
        let r = FixedPointMlpRunner::canonical(PUBKEY.into());
        let spec = canonical_spec();
        let mut flipped = spec.clone();
        flipped[100] ^= 1;
        let mut longer = spec.clone();
        longer.push(b'\n');
        for (name, bytes) in [
            ("one bit changed", flipped),
            ("truncated", spec[..spec.len() - 1].to_vec()),
            ("one byte added", longer),
            ("another file", b"{\"spec_name\":\"other\"}".to_vec()),
            ("empty", Vec::new()),
        ] {
            let f = manifest_file(&bytes);
            match r.run(f.path(), &le(&CANONICAL_INPUT)) {
                Err(RunnerError::ModelManifestMismatch { expected, got }) => {
                    assert_eq!(expected, CANONICAL_SPEC_BLAKE3_HEX, "{name}");
                    assert_eq!(got, blake3::hash(&bytes).to_hex().as_str(), "{name}");
                }
                other => panic!("{name}: {other:?}"),
            }
        }
    }

    #[test]
    fn a_manifest_over_the_read_limit_is_refused_unhashed() {
        let r = FixedPointMlpRunner::canonical(PUBKEY.into());
        let f = manifest_file(&vec![b' '; (MAX_MLP_MANIFEST_BYTES + 1) as usize]);
        assert!(matches!(
            r.run(f.path(), &le(&CANONICAL_INPUT)),
            Err(RunnerError::ModelManifestTooLarge { len, limit })
                if len == MAX_MLP_MANIFEST_BYTES + 1 && limit == MAX_MLP_MANIFEST_BYTES
        ));
    }

    #[test]
    fn an_input_of_the_wrong_size_is_refused() {
        let r = FixedPointMlpRunner::canonical(PUBKEY.into());
        let spec = manifest_file(&canonical_spec());
        for len in [0, 31, 33] {
            match r.run(spec.path(), &vec![0; len]) {
                Err(RunnerError::ModelInput(msg)) => {
                    assert!(msg.contains("expected 32 input bytes"), "{msg}")
                }
                other => panic!("len {len}: {other:?}"),
            }
        }
    }

    fn job_with(accounting: JobAccounting, model_hash: &str) -> ContributorJob {
        ContributorJob {
            schema_version: 1,
            job_id: String::new(),
            model_hash: model_hash.to_string(),
            manifest_snip_root: String::new(),
            input_snip_root: String::new(),
            input_hash: String::new(),
            verification_requirement: crate::job::VerificationRequirement::AttestationOnly,
            accounting,
            dispatched_at_utc: "2026-11-06T00:00:00Z".to_string(),
            expires_at_utc: None,
            dispatcher_pubkey_hex: None,
            dispatcher_signature_hex: None,
            notes: None,
        }
    }

    #[test]
    fn check_job_accepts_the_job_accounting_and_refuses_each_difference() {
        let r = FixedPointMlpRunner::canonical(PUBKEY.into());
        let good = r.job_accounting();
        assert_eq!(
            r.check_job(&job_with(good.clone(), CANONICAL_SPEC_BLAKE3_HEX)),
            Ok(())
        );
        // A larger output cap is fine: the output stays within it.
        let mut roomy = good.clone();
        roomy.max_output_token_count = 100;
        assert_eq!(
            r.check_job(&job_with(roomy, CANONICAL_SPEC_BLAKE3_HEX)),
            Ok(())
        );

        let other = "00".repeat(32);
        assert!(matches!(
            r.check_job(&job_with(good.clone(), &other)),
            Err(MlpJobError::ModelHash { .. })
        ));
        let mut a = good.clone();
        a.tokenizer_hash = other.clone();
        assert!(matches!(
            r.check_job(&job_with(a, CANONICAL_SPEC_BLAKE3_HEX)),
            Err(MlpJobError::TokenizerHash { .. })
        ));
        let mut a = good.clone();
        a.tokenizer_id = "tiktoken/cl100k_base".into();
        assert!(matches!(
            r.check_job(&job_with(a, CANONICAL_SPEC_BLAKE3_HEX)),
            Err(MlpJobError::TokenizerId { .. })
        ));
        let mut a = good.clone();
        a.input_token_count = 32;
        assert_eq!(
            r.check_job(&job_with(a, CANONICAL_SPEC_BLAKE3_HEX)),
            Err(MlpJobError::InputCount {
                expected: 16,
                got: 32
            })
        );
        let mut a = good;
        a.max_output_token_count = 7;
        assert_eq!(
            r.check_job(&job_with(a, CANONICAL_SPEC_BLAKE3_HEX)),
            Err(MlpJobError::OutputCap {
                expected: 8,
                got: 7
            })
        );
    }
}
