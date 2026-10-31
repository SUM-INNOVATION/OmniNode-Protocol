//! Issue #130: `ExternalCommandRunner::timeout`.
//!
//! Each test runs a real external command (a `/bin/sh` script) through the
//! production `ExternalCommandRunner`. Without a timeout a run waits for the
//! command however long it takes, as before. With one, a command still
//! running at the deadline is killed and the run fails with
//! `RunnerError::ExternalTimeout`, and a command that answers in time is
//! unaffected.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use omni_contributor::{
    check_external_timeout, ExternalCommandRunner, InferenceRunner, RunnerError,
};

/// `{"response_b64":"b2s="...}`: the response bytes are `ok`.
const ENVELOPE: &str = r#"{"response_b64":"b2s=","measured_input_tokens":1,"measured_output_tokens":2,"stage_contributions":[{"contributor_pubkey_hex":"00","stage_label":"t","work_unit_kind":"tokens","work_units":3}]}"#;

/// The limit for tests whose command must get past its first line before the
/// deadline (to record a pid, a path or its descendants). On a loaded macOS
/// host, launching `/bin/sh` can take over a second (seen: the shell still in
/// dyld start-up at a 1 s deadline while the other tests here ran), and a
/// command killed before its first line leaves nothing to check.
const LAUNCH_TOLERANT: Duration = Duration::from_secs(5);

fn script(dir: &Path, body: &str) -> PathBuf {
    let p = dir.join("runner.sh");
    std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

fn manifest(dir: &Path) -> PathBuf {
    let p = dir.join("manifest.json");
    std::fs::write(&p, b"{}").unwrap();
    p
}

fn runner(command: PathBuf, timeout: Option<Duration>) -> ExternalCommandRunner {
    let mut r = ExternalCommandRunner::new(command);
    r.timeout = timeout;
    r
}

/// Whether process `pid` still exists (`kill -0`).
fn alive(pid: &str) -> bool {
    Command::new("kill")
        .args(["-0", pid])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Whether process `pid` is still running: it exists and is not a zombie
/// (an orphan that exited may wait a moment for its new parent to reap it).
fn running(pid: &str) -> bool {
    Command::new("ps")
        .args(["-o", "stat=", "-p", pid])
        .output()
        .map(|o| {
            let stat = String::from_utf8_lossy(&o.stdout);
            let stat = stat.trim();
            !stat.is_empty() && !stat.starts_with('Z')
        })
        .unwrap_or(false)
}

/// The pids listed in `path`, one per line.
fn read_pids(path: &Path, expect: usize) -> Vec<String> {
    let t = Instant::now();
    loop {
        let pids: Vec<String> = std::fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        if pids.len() >= expect {
            return pids;
        }
        assert!(
            t.elapsed() < Duration::from_secs(10),
            "{} of {expect} pids in {path:?}",
            pids.len()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The pids among `pids` still running after waiting up to `within`.
fn survivors(pids: &[String], within: Duration) -> Vec<String> {
    let t = Instant::now();
    loop {
        let left: Vec<String> = pids.iter().filter(|p| running(p)).cloned().collect();
        if left.is_empty() || t.elapsed() >= within {
            return left;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Kills the listed processes when dropped. Every test asserts what it
/// measures BEFORE this runs, so the cleanup cannot hide a leak: a reader
/// that kept a descendant's pipe open is detected while the descendant is
/// still alive, and only then is the descendant killed.
struct KillOnDrop(Vec<String>);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        for pid in &self.0 {
            let _ = Command::new("kill").args(["-9", pid]).status();
        }
    }
}

/// A runner script that starts two descendants holding its stdout and its
/// stderr and writing to them every 50 ms (each records its pid in `pids`),
/// then hangs. A descendant ends at its first write after the reading end of
/// its pipe is closed (SIGPIPE, or the failed write).
fn writers_then_hang(dir: &Path, pids: &Path) -> PathBuf {
    script(
        dir,
        &format!(
            "( while :; do echo tick || exit 0; sleep 0.05; done ) &\n\
             echo $! >> '{p}'\n\
             ( while :; do echo tick >&2 || exit 0; sleep 0.05; done ) &\n\
             echo $! >> '{p}'\n\
             exec sleep 600",
            p = pids.display()
        ),
    )
}

fn read_pid(path: &Path) -> String {
    let t = Instant::now();
    loop {
        if let Ok(s) = std::fs::read_to_string(path) {
            if s.ends_with('\n') {
                return s.trim().to_string();
            }
        }
        assert!(t.elapsed() < Duration::from_secs(10), "no pid in {path:?}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_hung_runner_is_killed_at_its_timeout() {
    let dir = tempfile::tempdir().unwrap();
    let pid_file = dir.path().join("pid");
    // `exec`: the process OmniNode started is the one that hangs.
    let cmd = script(
        dir.path(),
        &format!("echo $$ > '{}'\nexec sleep 600", pid_file.display()),
    );
    let started = Instant::now();
    let err = runner(cmd, Some(LAUNCH_TOLERANT))
        .run(&manifest(dir.path()), b"input")
        .unwrap_err();
    let took = started.elapsed();
    assert!(
        matches!(err, RunnerError::ExternalTimeout(d) if d == LAUNCH_TOLERANT),
        "{err}"
    );
    assert_eq!(err.to_string(), "external runner timed out after 5s");
    assert!(
        took >= LAUNCH_TOLERANT && took < LAUNCH_TOLERANT + Duration::from_secs(10),
        "returned after {took:?}"
    );
    let pid = read_pid(&pid_file);
    assert!(
        !alive(&pid),
        "the timed-out runner (pid {pid}) is still running"
    );
}

#[test]
fn a_runner_that_answers_in_time_is_unaffected() {
    let dir = tempfile::tempdir().unwrap();
    let cmd = script(dir.path(), &format!("sleep 0.2\necho '{ENVELOPE}'"));
    for timeout in [None, Some(Duration::from_secs(30))] {
        let out = runner(cmd.clone(), timeout)
            .run(&manifest(dir.path()), b"input")
            .unwrap();
        assert_eq!(out.response_bytes, b"ok");
        assert_eq!(
            (out.measured_input_tokens, out.measured_output_tokens),
            (1, 2)
        );
    }
}

#[test]
fn without_a_timeout_a_slow_runner_is_waited_for() {
    let dir = tempfile::tempdir().unwrap();
    let cmd = script(dir.path(), &format!("sleep 2\necho '{ENVELOPE}'"));
    let out = runner(cmd.clone(), None)
        .run(&manifest(dir.path()), b"input")
        .unwrap();
    assert_eq!(out.response_bytes, b"ok");
    // The same command with a shorter limit is stopped.
    assert!(matches!(
        runner(cmd, Some(Duration::from_secs(1))).run(&manifest(dir.path()), b"input"),
        Err(RunnerError::ExternalTimeout(_))
    ));
}

/// A command that writes more than a pipe holds before exiting must not
/// stall on its own output while a timeout is enforced.
#[test]
fn a_large_output_does_not_stall_a_bounded_run() {
    let dir = tempfile::tempdir().unwrap();
    let cmd = script(
        dir.path(),
        &format!("head -c 1048576 /dev/zero >&2\necho '{ENVELOPE}'"),
    );
    let out = runner(cmd, Some(Duration::from_secs(30)))
        .run(&manifest(dir.path()), b"input")
        .unwrap();
    assert_eq!(out.response_bytes, b"ok");
}

/// Exit status and failure output are reported as without a timeout.
#[test]
fn a_failing_runner_still_reports_its_exit_code() {
    let dir = tempfile::tempdir().unwrap();
    let cmd = script(dir.path(), "echo boom >&2\nexit 3");
    let err = runner(cmd, Some(Duration::from_secs(30)))
        .run(&manifest(dir.path()), b"input")
        .unwrap_err();
    match err {
        RunnerError::ExternalCommandFailure { code, stderr } => {
            assert_eq!(code, 3);
            assert!(stderr.contains("boom"), "{stderr}");
        }
        other => panic!("{other}"),
    }
}

/// The deadline also bounds collecting the output: a process the command
/// left behind holding its stdout open ends the run at the deadline instead
/// of holding it until that process exits.
#[test]
fn a_left_behind_process_holding_stdout_ends_the_run_at_the_deadline() {
    let dir = tempfile::tempdir().unwrap();
    let pid_file = dir.path().join("pid");
    let cmd = script(
        dir.path(),
        &format!(
            "sleep 600 &\necho $! > '{}'\necho '{ENVELOPE}'\nexit 0",
            pid_file.display()
        ),
    );
    let started = Instant::now();
    let res = runner(cmd, Some(LAUNCH_TOLERANT)).run(&manifest(dir.path()), b"input");
    let took = started.elapsed();
    let pid = read_pid(&pid_file);
    let _ = Command::new("kill").args(["-9", &pid]).status();
    assert!(
        matches!(res, Err(RunnerError::ExternalTimeout(_))),
        "{res:?}"
    );
    assert!(
        took < LAUNCH_TOLERANT + Duration::from_secs(10),
        "returned after {took:?}"
    );
}

/// A timed-out run must not leave anything reading the pipes that the
/// command's descendants still hold: once `run` returns, the reading ends
/// are closed, so a descendant still writing there ends (SIGPIPE) instead
/// of being drained into a buffer nobody will read.
#[test]
fn a_timed_out_run_closes_the_pipes_its_descendants_still_hold() {
    let dir = tempfile::tempdir().unwrap();
    let pids_file = dir.path().join("pids");
    let cmd = writers_then_hang(dir.path(), &pids_file);
    let started = Instant::now();
    let res = runner(cmd, Some(LAUNCH_TOLERANT)).run(&manifest(dir.path()), b"input");
    let took = started.elapsed();
    let pids = read_pids(&pids_file, 2);
    let _cleanup = KillOnDrop(pids.clone());
    assert!(
        matches!(res, Err(RunnerError::ExternalTimeout(_))),
        "{res:?}"
    );
    assert!(
        took < LAUNCH_TOLERANT + Duration::from_secs(10),
        "returned after {took:?}"
    );
    let left = survivors(&pids, Duration::from_secs(5));
    assert!(
        left.is_empty(),
        "descendants {left:?} still write to pipes the timed-out run should have closed"
    );
}

/// After a timeout, the run's temporary input file is gone, as after any
/// other outcome.
#[test]
fn a_timed_out_run_removes_its_temp_input() {
    let dir = tempfile::tempdir().unwrap();
    let input_file = dir.path().join("input-path");
    // Arguments: --manifest <m> --input <i>.
    let cmd = script(
        dir.path(),
        &format!("echo \"$4\" > '{}'\nexec sleep 600", input_file.display()),
    );
    let res = runner(cmd, Some(LAUNCH_TOLERANT)).run(&manifest(dir.path()), b"input");
    assert!(
        matches!(res, Err(RunnerError::ExternalTimeout(_))),
        "{res:?}"
    );
    let input = PathBuf::from(read_pid(&input_file));
    assert!(input.is_absolute(), "{input:?}");
    assert!(
        !input.exists(),
        "temp input {input:?} survived the timed-out run"
    );
}

/// A timeout the runner cannot enforce is refused with a typed error before
/// anything is spawned: zero, and durations whose deadline overflows the
/// clock (this used to panic in `Instant + Duration`).
#[test]
fn unsupported_timeouts_are_refused_before_spawning() {
    for timeout in [Duration::ZERO, Duration::MAX, Duration::from_secs(u64::MAX)] {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("ran");
        let cmd = script(
            dir.path(),
            &format!("touch '{}'\necho '{ENVELOPE}'", marker.display()),
        );
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runner(cmd, Some(timeout)).run(&manifest(dir.path()), b"input")
        }))
        .unwrap_or_else(|_| panic!("timeout {timeout:?} panicked"));
        assert!(
            matches!(res, Err(RunnerError::ExternalTimeoutUnsupported { timeout: t, .. }) if t == timeout),
            "timeout {timeout:?}: {res:?}"
        );
        assert!(
            !marker.exists(),
            "timeout {timeout:?}: the command was spawned"
        );
        assert!(
            matches!(
                check_external_timeout(timeout),
                Err(RunnerError::ExternalTimeoutUnsupported { timeout: t, .. }) if t == timeout
            ),
            "check_external_timeout({timeout:?})"
        );
    }
    // The smallest and a very large enforceable value pass the check.
    for timeout in [
        Duration::from_nanos(1),
        Duration::from_secs(10 * 365 * 24 * 3600),
    ] {
        check_external_timeout(timeout).unwrap();
    }
}
