//! Stage 12.0 — `InferenceRunner` trait + concrete implementations.
//!
//! The runner is the **only** compute boundary in the contributor
//! crate. Everything else (SNIP fetch/publish, signing, schema
//! validation, accounting) is orchestration. Runners report measured
//! token counts and per-stage work contributions; the orchestrator
//! copies those numbers verbatim into the `ContributorResult` envelope.
//!
//! Stage 12.0 ships:
//!
//! - [`StubRunner`] — deterministic test/fixture-generation runner.
//! - [`ExternalCommandRunner`] — shells out to a configured external
//!   command, parses the documented JSON envelope from stdout.
//!
//! Stage 12.0 does NOT depend on `omni-pipeline` — single-runner
//! manual flow only.

use std::path::Path;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use base64::Engine;
use serde::Deserialize;

use crate::error::RunnerError;
use crate::result::{StageContribution, WorkUnitKind};

/// What an inference runner produces. The orchestrator copies token
/// counts verbatim into `ContributorResult.measured_accounting`.
#[derive(Debug, Clone)]
pub struct RunOutput {
    pub response_bytes: Vec<u8>,
    pub measured_input_tokens: u64,
    pub measured_output_tokens: u64,
    pub stage_contributions: Vec<StageContribution>,
}

/// Stage 12.0 inference-runner abstraction. Implementations must be
/// deterministic given `(manifest_path, input_bytes)` for the
/// byte-stability of the contributor's signed result to hold.
pub trait InferenceRunner {
    fn run(
        &self,
        manifest_path: &Path,
        input_bytes: &[u8],
    ) -> Result<RunOutput, RunnerError>;

    /// Stage 12.4 — pooled-stage variant. Runs one stage of a
    /// pipelined inference with an optional **upstream activation**
    /// (bytes produced by the previous contributor's stage) and an
    /// option to emit an **output activation** for the next
    /// contributor.
    ///
    /// Default impl: ignores upstream activation, runs the standard
    /// 12.0/12.1/12.2 path, returns no output activation. This keeps
    /// every existing runner (StubRunner, ExternalCommandRunner,
    /// AnyRunner shims) usable in 12.4 without code changes —
    /// callers that want live handoff opt in by overriding.
    fn run_with_activations(
        &self,
        manifest_path: &Path,
        input_bytes: &[u8],
        _upstream_activation_bytes: Option<&[u8]>,
    ) -> Result<RunOutputWithActivation, RunnerError> {
        let run_output = self.run(manifest_path, input_bytes)?;
        Ok(RunOutputWithActivation {
            run_output,
            output_activation: None,
        })
    }
}

/// Stage 12.4 — typed activation produced by a runner for the next
/// stage. `bytes` is opaque to this crate; `dtype` + `shape` carry
/// the structural metadata that the CLI's `live_send_activation`
/// stamps onto the signed `ActivationHandoff` envelope.
///
/// **`activation-out-mode=live|both` REQUIRES this to be `Some`.**
/// The hard-coded `F16 + [byte_len/2]` fallback was removed in the
/// Stage 12.4 review; the live path now refuses to invent dtype /
/// shape on behalf of a runner that didn't declare them.
#[derive(Debug, Clone)]
pub struct RunnerOutputActivation {
    pub bytes: Vec<u8>,
    pub dtype: crate::handoff::TensorDtype,
    pub shape: Vec<u64>,
}

/// Stage 12.4 — extension of [`RunOutput`] carrying an optional
/// typed output activation. Runners that produce one for the next
/// stage populate `output_activation: Some(...)`; runners that don't
/// (or are the last stage) leave it `None`. Runners producing
/// activation bytes WITHOUT dtype + shape declared via this struct
/// will be refused when the caller's `activation-out-mode` requires
/// a live handoff.
#[derive(Debug, Clone)]
pub struct RunOutputWithActivation {
    pub run_output: RunOutput,
    pub output_activation: Option<RunnerOutputActivation>,
}

// ── StubRunner ────────────────────────────────────────────────────────────

/// Deterministic, configurable runner used in tests + fixture
/// generation + CLI smoke runs. Returns the configured response /
/// token counts regardless of input.
#[derive(Debug, Clone)]
pub struct StubRunner {
    pub fixed_response: Vec<u8>,
    pub measured_input_tokens: u64,
    pub measured_output_tokens: u64,
    /// Pubkey hex to attribute the single stage to. Must match the
    /// top-level `contributor_pubkey_hex` for the orchestrator's
    /// schema validation to pass.
    pub contributor_pubkey_hex: String,
    /// Stage label to emit. Free-form; default `"stub-runner"`.
    pub stage_label: String,
    /// Work-unit kind to emit. Default `WorkUnitKind::Tokens`.
    pub work_unit_kind: WorkUnitKind,
}

impl StubRunner {
    /// Convenience constructor with sensible defaults.
    pub fn new(
        contributor_pubkey_hex: String,
        fixed_response: Vec<u8>,
        measured_input_tokens: u64,
        measured_output_tokens: u64,
    ) -> Self {
        Self {
            fixed_response,
            measured_input_tokens,
            measured_output_tokens,
            contributor_pubkey_hex,
            stage_label: "stub-runner".to_string(),
            work_unit_kind: WorkUnitKind::Tokens,
        }
    }
}

impl InferenceRunner for StubRunner {
    fn run(
        &self,
        _manifest_path: &Path,
        _input_bytes: &[u8],
    ) -> Result<RunOutput, RunnerError> {
        // `work_units` is an audit field, not a protocol-bound value.
        // Use saturating_add so a stub configured with `u64::MAX` for
        // both counts (the overflow-orchestrator-refusal test) still
        // returns a RunOutput; the orchestrator's
        // `validate_runner_output_against_job` then refuses the
        // overall result with a typed AccountingTotalOverflow error.
        let total_units = self
            .measured_input_tokens
            .saturating_add(self.measured_output_tokens);
        Ok(RunOutput {
            response_bytes: self.fixed_response.clone(),
            measured_input_tokens: self.measured_input_tokens,
            measured_output_tokens: self.measured_output_tokens,
            stage_contributions: vec![StageContribution {
                contributor_pubkey_hex: self.contributor_pubkey_hex.clone(),
                stage_label: self.stage_label.clone(),
                work_unit_kind: self.work_unit_kind,
                work_units: total_units,
            }],
        })
    }
}

// ── ExternalCommandRunner ─────────────────────────────────────────────────

/// Shells out to a configured external command. The command receives
/// the manifest path + input path as positional arguments (preceded
/// by any `extra_args`) and writes a JSON envelope to stdout (see
/// [`ExternalRunnerEnvelope`]).
///
/// Stage 12.0 contract — the command's invocation shape:
///
/// ```text
/// <command> [<extra_args>...] --manifest <manifest_path> --input <input_path>
/// ```
///
/// The command's stdout MUST be a single JSON object parsing into
/// [`ExternalRunnerEnvelope`]. stderr is unrestricted (used for
/// diagnostics / progress logs); stdout outside the JSON envelope is
/// rejected (`deny_unknown_fields`).
///
/// No framework runtime ever enters the `omni-node` compile graph —
/// the command is loaded at run time as a binary path; its dependency
/// tree is invisible to cargo.
pub struct ExternalCommandRunner {
    pub command: PathBuf,
    pub extra_args: Vec<String>,
    pub work_dir: Option<PathBuf>,
    pub env_allowlist: Vec<String>,
    // Stage 12.0 intentionally does NOT expose a `timeout` field. A
    // half-implemented timeout that we silently never enforce would
    // be a worse safety control than no timeout at all (operators
    // would assume protection that doesn't exist). Operators rely on
    // the external runtime's own timeout mechanism + the surrounding
    // process supervisor (systemd, k8s probe, etc.). A real
    // wall-clock kill switch is a Stage 12.x hardening item that
    // requires a watcher thread (or platform-specific waitid timeout)
    // — out of scope here.
}

impl ExternalCommandRunner {
    pub fn new(command: PathBuf) -> Self {
        Self {
            command,
            extra_args: Vec::new(),
            work_dir: None,
            env_allowlist: Vec::new(),
        }
    }
}

impl InferenceRunner for ExternalCommandRunner {
    fn run(
        &self,
        manifest_path: &Path,
        input_bytes: &[u8],
    ) -> Result<RunOutput, RunnerError> {
        // Stage 12.0 contract: invoke `<command> [extra_args...]
        // --manifest <m> --input <i>`. No `--activation-in` /
        // `--activation-out` flags — preserving the old contract so
        // existing pre-12.4 external runners continue to work when
        // the operator is not using live handoff. (Review #1 closed
        // the regression where every run() call was injecting
        // `--activation-out`.)
        let (envelope, _activation_out_path) =
            self.spawn_and_read_envelope(manifest_path, input_bytes, None, false)?;
        envelope_to_run_output(envelope)
    }

    fn run_with_activations(
        &self,
        manifest_path: &Path,
        input_bytes: &[u8],
        upstream_activation_bytes: Option<&[u8]>,
    ) -> Result<RunOutputWithActivation, RunnerError> {
        // Stage 12.4 path: ALWAYS pass `--activation-out` and
        // optionally `--activation-in`. Callers must wire an
        // external runner that recognizes these flags. The Stage
        // 12.0 `run()` path does not change.
        let (envelope, activation_out_path) = self.spawn_and_read_envelope(
            manifest_path,
            input_bytes,
            upstream_activation_bytes,
            /* pass_activation_out = */ true,
        )?;
        let run_output = envelope_to_run_output(envelope.clone())?;

        let activation_out_path = activation_out_path
            .expect("pass_activation_out=true allocates the path");
        let act_bytes = std::fs::read(&activation_out_path)?;
        let output_activation = if act_bytes.is_empty() {
            None
        } else {
            match (
                envelope.output_activation_dtype,
                envelope.output_activation_shape,
            ) {
                (Some(dtype), Some(shape)) => Some(RunnerOutputActivation {
                    bytes: act_bytes,
                    dtype,
                    shape,
                }),
                _ => {
                    return Err(RunnerError::ExternalEnvelopeMalformed(
                        "external runner wrote --activation-out bytes but the stdout \
                         envelope did not declare `output_activation_dtype` + \
                         `output_activation_shape`. Stage 12.4 live handoff requires \
                         both."
                            .to_string(),
                    ));
                }
            }
        };

        Ok(RunOutputWithActivation {
            run_output,
            output_activation,
        })
    }
}

impl ExternalCommandRunner {
    /// Shared subprocess driver for both [`InferenceRunner::run`]
    /// (Stage 12.0 contract: no activation flags) and
    /// [`InferenceRunner::run_with_activations`] (Stage 12.4
    /// contract: `--activation-in` + `--activation-out`).
    ///
    /// `pass_activation_out=true` allocates an activation-out
    /// tempfile and passes its path on the command line; the
    /// caller can read it back after `output.status.success()`.
    /// `pass_activation_out=false` leaves the command line at the
    /// frozen 12.0 shape so pre-12.4 external runners keep working.
    fn spawn_and_read_envelope(
        &self,
        manifest_path: &Path,
        input_bytes: &[u8],
        upstream_activation_bytes: Option<&[u8]>,
        pass_activation_out: bool,
    ) -> Result<(ExternalRunnerEnvelope, Option<PathBuf>), RunnerError> {
        use std::io::Write;

        let mut input_tmp = tempfile::Builder::new()
            .prefix("omni-contributor-input-")
            .tempfile()?;
        input_tmp.write_all(input_bytes)?;
        input_tmp.flush()?;
        let input_path = input_tmp.path().to_path_buf();

        // Stage 12.4 optional upstream-activation tempfile. Only
        // allocated if the caller supplied upstream bytes. Pre-12.4
        // `run()` callers never reach this branch.
        let activation_in_tmp = match upstream_activation_bytes {
            Some(bytes) => {
                let mut tmp = tempfile::Builder::new()
                    .prefix("omni-contributor-activation-in-")
                    .tempfile()?;
                tmp.write_all(bytes)?;
                tmp.flush()?;
                Some(tmp)
            }
            None => None,
        };

        // Stage 12.4 output-activation tempfile is only allocated
        // (and only flagged on the command line) when
        // `pass_activation_out=true`. Pre-12.4 runners never see
        // `--activation-out`.
        let (activation_out_tmp, activation_out_path) = if pass_activation_out {
            let tmp = tempfile::Builder::new()
                .prefix("omni-contributor-activation-out-")
                .tempfile()?;
            let p = tmp.path().to_path_buf();
            (Some(tmp), Some(p))
        } else {
            (None, None)
        };

        let mut cmd = Command::new(&self.command);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        cmd.env_clear();
        for key in &self.env_allowlist {
            if let Ok(v) = std::env::var(key) {
                cmd.env(key, v);
            }
        }
        if let Some(wd) = &self.work_dir {
            cmd.current_dir(wd);
        }
        for a in &self.extra_args {
            cmd.arg(a);
        }
        cmd.arg("--manifest").arg(manifest_path);
        cmd.arg("--input").arg(&input_path);
        if let Some(tmp) = &activation_in_tmp {
            cmd.arg("--activation-in").arg(tmp.path());
        }
        if let Some(path) = &activation_out_path {
            cmd.arg("--activation-out").arg(path);
        }

        let output = cmd.output()?;
        if !output.status.success() {
            let code = output.status.code().unwrap_or(-1);
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            return Err(RunnerError::ExternalCommandFailure { code, stderr });
        }
        let stdout = std::str::from_utf8(&output.stdout)
            .map_err(|e| RunnerError::ExternalEnvelopeMalformed(format!("non-UTF8 stdout: {e}")))?;
        let envelope: ExternalRunnerEnvelope =
            serde_json::from_str(stdout.trim()).map_err(|e| {
                RunnerError::ExternalEnvelopeMalformed(format!(
                    "stdout did not parse as ExternalRunnerEnvelope: {e}"
                ))
            })?;
        // Hold tempfiles alive until after stdout has been parsed.
        let _ = (activation_in_tmp, activation_out_tmp);
        Ok((envelope, activation_out_path))
    }
}

fn envelope_to_run_output(envelope: ExternalRunnerEnvelope) -> Result<RunOutput, RunnerError> {
    let response_bytes = base64::engine::general_purpose::STANDARD
        .decode(&envelope.response_b64)
        .map_err(|e| RunnerError::ExternalResponseDecode(e.to_string()))?;
    let stage_contributions = envelope
        .stage_contributions
        .into_iter()
        .map(|e| StageContribution {
            contributor_pubkey_hex: e.contributor_pubkey_hex,
            stage_label: e.stage_label,
            work_unit_kind: e.work_unit_kind,
            work_units: e.work_units,
        })
        .collect();
    Ok(RunOutput {
        response_bytes,
        measured_input_tokens: envelope.measured_input_tokens,
        measured_output_tokens: envelope.measured_output_tokens,
        stage_contributions,
    })
}

/// Documented stdout envelope shape for `ExternalCommandRunner`.
/// `deny_unknown_fields` is intentional: forward compatibility lives
/// in `schema_version`, not in tolerant deserializers.
///
/// Stage 12.4 — two optional fields added:
/// `output_activation_dtype` + `output_activation_shape`. These are
/// REQUIRED when the runner writes non-empty bytes to the
/// `--activation-out` path. Pre-12.4 runners that don't write
/// activation bytes are unaffected (both fields default to `None`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalRunnerEnvelope {
    pub response_b64: String,
    pub measured_input_tokens: u64,
    pub measured_output_tokens: u64,
    pub stage_contributions: Vec<ExternalRunnerStageContribution>,
    /// Required iff the runner wrote any bytes to `--activation-out`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_activation_dtype: Option<crate::handoff::TensorDtype>,
    /// Required iff the runner wrote any bytes to `--activation-out`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_activation_shape: Option<Vec<u64>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalRunnerStageContribution {
    pub contributor_pubkey_hex: String,
    pub stage_label: String,
    pub work_unit_kind: WorkUnitKind,
    pub work_units: u64,
}
