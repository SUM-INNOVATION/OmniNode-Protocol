//! Issue #130: repeated timeouts of `ExternalCommandRunner` must not
//! accumulate anything.
//!
//! Each run's command leaves two descendants holding its stdout and stderr
//! and writing to them, then hangs until the timeout kills it. A run that
//! returned while something of it still read those pipes would leave a
//! thread (and a growing buffer) per pipe, and keep the descendants alive.
//! This file holds a single test, so the process's thread count is not
//! disturbed by other tests running in parallel.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use omni_contributor::{ExternalCommandRunner, InferenceRunner, RunnerError};

const RUNS: usize = 6;

/// Long enough for each run's command to start its two descendants before the
/// deadline, even when launching `/bin/sh` is slow (over a second on a loaded
/// macOS host).
const TIMEOUT: Duration = Duration::from_secs(2);

fn script(dir: &Path, body: &str) -> PathBuf {
    let p = dir.join("runner.sh");
    std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

/// Threads of this process.
fn threads() -> usize {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_dir("/proc/self/task").unwrap().count()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let out = Command::new("ps")
            .args(["-M", "-p", &std::process::id().to_string()])
            .output()
            .unwrap();
        // One header line, then one line per thread.
        String::from_utf8_lossy(&out.stdout).lines().count() - 1
    }
}

/// Running and not a zombie.
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

/// Kills the listed processes when dropped, i.e. only after the assertions.
struct KillOnDrop(Vec<String>);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        for pid in &self.0 {
            let _ = Command::new("kill").args(["-9", pid]).status();
        }
    }
}

#[test]
fn repeated_timeouts_leave_no_reader_buffer_or_descendant_pipe_behind() {
    let dir = tempfile::tempdir().unwrap();
    let pids_file = dir.path().join("pids");
    let manifest = dir.path().join("manifest.json");
    std::fs::write(&manifest, b"{}").unwrap();
    let cmd = script(
        dir.path(),
        &format!(
            "( while :; do echo tick || exit 0; sleep 0.05; done ) &\n\
             echo $! >> '{p}'\n\
             ( while :; do echo tick >&2 || exit 0; sleep 0.05; done ) &\n\
             echo $! >> '{p}'\n\
             exec sleep 600",
            p = pids_file.display()
        ),
    );
    let mut runner = ExternalCommandRunner::new(cmd);
    runner.timeout = Some(TIMEOUT);

    let before = threads();
    for i in 0..RUNS {
        let started = Instant::now();
        let res = runner.run(&manifest, b"input");
        let took = started.elapsed();
        assert!(
            matches!(res, Err(RunnerError::ExternalTimeout(_))),
            "run {i}: {res:?}"
        );
        assert!(
            took < TIMEOUT + Duration::from_secs(5),
            "run {i} returned after {took:?}"
        );
    }
    let after = threads();

    let t = Instant::now();
    let pids = loop {
        let pids: Vec<String> = std::fs::read_to_string(&pids_file)
            .unwrap_or_default()
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        if pids.len() >= 2 * RUNS || t.elapsed() > Duration::from_secs(10) {
            break pids;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let _cleanup = KillOnDrop(pids.clone());
    assert_eq!(pids.len(), 2 * RUNS, "descendant pids recorded");

    // No thread outlives its run: a leaked reader per pipe would add 2 per run.
    assert!(
        after <= before,
        "threads grew from {before} to {after} over {RUNS} timed-out runs"
    );
    // Nothing still reads the descendants' pipes, so every writer has ended.
    let t = Instant::now();
    let left = loop {
        let left: Vec<&String> = pids.iter().filter(|p| running(p)).collect();
        if left.is_empty() || t.elapsed() > Duration::from_secs(5) {
            break left;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(
        left.is_empty(),
        "descendants {left:?} still write to pipes of timed-out runs"
    );
}
