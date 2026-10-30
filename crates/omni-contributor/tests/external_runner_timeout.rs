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

use omni_contributor::{ExternalCommandRunner, InferenceRunner, RunnerError};

/// `{"response_b64":"b2s="...}`: the response bytes are `ok`.
const ENVELOPE: &str = r#"{"response_b64":"b2s=","measured_input_tokens":1,"measured_output_tokens":2,"stage_contributions":[{"contributor_pubkey_hex":"00","stage_label":"t","work_unit_kind":"tokens","work_units":3}]}"#;

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
    let err = runner(cmd, Some(Duration::from_secs(1)))
        .run(&manifest(dir.path()), b"input")
        .unwrap_err();
    let took = started.elapsed();
    assert!(
        matches!(err, RunnerError::ExternalTimeout(d) if d == Duration::from_secs(1)),
        "{err}"
    );
    assert_eq!(err.to_string(), "external runner timed out after 1s");
    assert!(
        took >= Duration::from_secs(1) && took < Duration::from_secs(10),
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
    let res = runner(cmd, Some(Duration::from_secs(2))).run(&manifest(dir.path()), b"input");
    let took = started.elapsed();
    let pid = read_pid(&pid_file);
    let _ = Command::new("kill").args(["-9", &pid]).status();
    assert!(
        matches!(res, Err(RunnerError::ExternalTimeout(_))),
        "{res:?}"
    );
    assert!(took < Duration::from_secs(10), "returned after {took:?}");
}
