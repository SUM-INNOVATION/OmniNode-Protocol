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
        // Write input bytes to a tempfile so the external command can
        // open them.
        let mut input_tmp = tempfile::Builder::new()
            .prefix("omni-contributor-input-")
            .tempfile()?;
        use std::io::Write;
        input_tmp.write_all(input_bytes)?;
        input_tmp.flush()?;
        let input_path = input_tmp.path().to_path_buf();

        // Build the subprocess. We pass --manifest / --input by name
        // to be unambiguous and to allow operators to wrap arbitrary
        // existing runtimes.
        let mut cmd = Command::new(&self.command);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Environment scoping: clear all env vars by default, then
        // restore only those on `env_allowlist`. PATH is implicitly
        // useful for `command` resolution but is supplied by the
        // operator's shell at omni-node-invocation time, not by us.
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

        // (No wall-clock timeout enforcement in Stage 12.0; see the
        // struct's doc note above.)

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
}

/// Documented stdout envelope shape for `ExternalCommandRunner`.
/// `deny_unknown_fields` is intentional: forward compatibility lives
/// in `schema_version`, not in tolerant deserializers.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalRunnerEnvelope {
    pub response_b64: String,
    pub measured_input_tokens: u64,
    pub measured_output_tokens: u64,
    pub stage_contributions: Vec<ExternalRunnerStageContribution>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalRunnerStageContribution {
    pub contributor_pubkey_hex: String,
    pub stage_label: String,
    pub work_unit_kind: WorkUnitKind,
    pub work_units: u64,
}
