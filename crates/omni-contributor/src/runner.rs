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
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

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
    /// Issue #130: wall-clock limit on one run of the command, from spawn
    /// until its exit status and both output streams are in. `None` (the
    /// default): no limit, as before. Unix only.
    ///
    /// A value the runner cannot enforce (zero, or one whose deadline
    /// overflows the clock) fails the run with
    /// [`RunnerError::ExternalTimeoutUnsupported`] before anything is
    /// spawned; [`check_external_timeout`] checks a value up front.
    ///
    /// When the limit passes, the run fails with
    /// [`RunnerError::ExternalTimeout`] and nothing the command wrote is used.
    /// What is stopped, and what is not:
    /// - **The command's own process** (the one this runner spawned) is
    ///   killed with `SIGKILL` and reaped, if it is still running.
    /// - **Processes it started** are not signalled. They stay in
    ///   `omni-node`'s process group and session (the command is not moved
    ///   into a group of its own), so a supervisor that stops `omni-node`'s
    ///   process group or cgroup still stops them.
    /// - **Its stdout and stderr pipes** are closed on `omni-node`'s side when
    ///   the run returns, by any path. Nothing keeps reading them: no thread
    ///   or buffer of the run outlives it, and a descendant that still writes
    ///   to them gets `SIGPIPE` / `EPIPE`.
    ///
    /// The output is read on the calling thread, with the child polled every
    /// [`TIMEOUT_POLL`], so the kill comes within that interval of the limit.
    pub timeout: Option<Duration>,
}

/// How often a run with [`ExternalCommandRunner::timeout`] set checks
/// whether the command has exited.
pub const TIMEOUT_POLL: Duration = Duration::from_millis(20);

/// Issue #130: whether [`ExternalCommandRunner::timeout`] can be set to
/// `timeout`. Refused, with [`RunnerError::ExternalTimeoutUnsupported`]:
/// zero; a duration whose deadline (now + `timeout`) overflows the clock;
/// any duration on a platform other than Unix. A run checks the same before
/// it spawns anything; callers can check once at startup instead of failing
/// every run.
pub fn check_external_timeout(timeout: Duration) -> Result<(), RunnerError> {
    deadline_after(timeout).map(|_| ())
}

fn deadline_after(timeout: Duration) -> Result<Instant, RunnerError> {
    let unsupported = |reason| RunnerError::ExternalTimeoutUnsupported { timeout, reason };
    if timeout.is_zero() {
        return Err(unsupported("zero"));
    }
    if cfg!(not(unix)) {
        return Err(unsupported("a bounded run needs Unix"));
    }
    Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| unsupported("its deadline overflows the clock"))
}

impl ExternalCommandRunner {
    pub fn new(command: PathBuf) -> Self {
        Self {
            command,
            extra_args: Vec::new(),
            work_dir: None,
            env_allowlist: Vec::new(),
            timeout: None,
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

        let output = output_within(cmd, self.timeout)?;
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

/// `cmd.output()`, bounded by `timeout` (issue #130).
///
/// Without a timeout this is exactly `cmd.output()`. With one, an
/// unsupported value is refused before spawning; otherwise see
/// [`bounded::output_until`].
fn output_within(mut cmd: Command, timeout: Option<Duration>) -> Result<Output, RunnerError> {
    let Some(timeout) = timeout else {
        return Ok(cmd.output()?);
    };
    let deadline = deadline_after(timeout)?;
    #[cfg(unix)]
    {
        bounded::output_until(cmd, deadline, timeout)
    }
    #[cfg(not(unix))]
    {
        // `deadline_after` refuses every timeout off Unix.
        let _ = deadline;
        unreachable!("a bounded run needs Unix")
    }
}

#[cfg(unix)]
mod bounded {
    use std::io::{self, Read};
    use std::os::unix::io::AsRawFd;
    use std::process::{Child, Command, ExitStatus, Output};
    use std::time::{Duration, Instant};

    use super::TIMEOUT_POLL;
    use crate::error::RunnerError;

    /// One output pipe of the child, read without blocking on this thread.
    /// `src` becomes `None` (closing the reading end) at end of file.
    struct Pipe<R> {
        src: Option<R>,
        buf: Vec<u8>,
    }

    impl<R: Read + AsRawFd> Pipe<R> {
        fn new(src: Option<R>) -> io::Result<Self> {
            if let Some(r) = &src {
                set_nonblocking(r.as_raw_fd())?;
            }
            Ok(Self {
                src,
                buf: Vec::new(),
            })
        }

        fn fd(&self) -> Option<i32> {
            self.src.as_ref().map(AsRawFd::as_raw_fd)
        }

        /// Reads everything available now.
        fn pump(&mut self) -> io::Result<()> {
            let mut chunk = [0u8; 8192];
            while let Some(r) = self.src.as_mut() {
                match r.read(&mut chunk) {
                    Ok(0) => self.src = None,
                    Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        }
    }

    fn set_nonblocking(fd: i32) -> io::Result<()> {
        // SAFETY: `fd` is the reading end of a pipe owned by a live
        // `ChildStdout` / `ChildStderr`; F_GETFL / F_SETFL only change its
        // status flags (the child's writing end is a separate description).
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFL);
            if flags < 0 || libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }

    /// Waits until one of `fds` is readable or `wait` has passed.
    fn wait_readable(fds: &[i32], wait: Duration) -> io::Result<()> {
        if fds.is_empty() {
            std::thread::sleep(wait);
            return Ok(());
        }
        let mut pfds: Vec<libc::pollfd> = fds
            .iter()
            .map(|&fd| libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            })
            .collect();
        // `wait` is at most TIMEOUT_POLL, so this cannot overflow.
        let ms = wait.as_millis().clamp(1, TIMEOUT_POLL.as_millis()) as libc::c_int;
        // SAFETY: `pfds` is a valid, exclusively borrowed array of
        // `pfds.len()` pollfd entries for the duration of the call.
        let rc = unsafe { libc::poll(pfds.as_mut_ptr(), pfds.len() as libc::nfds_t, ms) };
        if rc < 0 {
            let e = io::Error::last_os_error();
            if e.kind() != io::ErrorKind::Interrupted {
                return Err(e);
            }
        }
        Ok(())
    }

    enum Stop {
        Deadline,
        Io(io::Error),
    }

    /// Spawns `cmd` and collects its exit status and both output streams by
    /// `deadline`. Both pipes are read on this thread only, so when this
    /// returns, by any path, they are closed and their buffers freed: no
    /// reader thread exists to outlive the run. If the deadline passes first
    /// (the command is still running, or something it left behind still
    /// holds a pipe open), the command's own process is killed (`SIGKILL`)
    /// and reaped if still running, and the run fails with
    /// [`RunnerError::ExternalTimeout`]. Processes it started are not
    /// signalled (see `ExternalCommandRunner::timeout`).
    pub(super) fn output_until(
        mut cmd: Command,
        deadline: Instant,
        timeout: Duration,
    ) -> Result<Output, RunnerError> {
        let mut child = cmd.spawn()?;
        let res = Pipe::new(child.stdout.take())
            .and_then(|out| Ok((out, Pipe::new(child.stderr.take())?)))
            .map_err(Stop::Io)
            .and_then(|(mut out, mut err)| {
                let status = collect(&mut child, &mut out, &mut err, deadline)?;
                Ok(Output {
                    status,
                    stdout: out.buf,
                    stderr: err.buf,
                })
            });
        match res {
            Ok(output) => Ok(output),
            Err(stop) => {
                // `kill` does nothing for a child already reaped by `try_wait`.
                let _ = child.kill();
                let _ = child.wait();
                Err(match stop {
                    Stop::Deadline => RunnerError::ExternalTimeout(timeout),
                    Stop::Io(e) => RunnerError::Io(e),
                })
            }
        }
    }

    fn collect<O: Read + AsRawFd, E: Read + AsRawFd>(
        child: &mut Child,
        out: &mut Pipe<O>,
        err: &mut Pipe<E>,
        deadline: Instant,
    ) -> Result<ExitStatus, Stop> {
        let mut status = None;
        loop {
            if status.is_none() {
                status = child.try_wait().map_err(Stop::Io)?;
            }
            if let Some(s) = status {
                if out.src.is_none() && err.src.is_none() {
                    return Ok(s);
                }
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(Stop::Deadline);
            }
            let fds: Vec<i32> = out.fd().into_iter().chain(err.fd()).collect();
            wait_readable(&fds, TIMEOUT_POLL.min(deadline - now)).map_err(Stop::Io)?;
            out.pump().map_err(Stop::Io)?;
            err.pump().map_err(Stop::Io)?;
        }
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
