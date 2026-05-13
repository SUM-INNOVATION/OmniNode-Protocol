//! SNIP V2 CLI adapter — wraps the `sum-node` ingest/download commands.
//!
//! Stage-1 scope of OmniNode Phase 5. This module deliberately speaks only to
//! the **Public V2** surface of `sum-node` and only via the documented CLI
//! contract:
//!
//! - `sum-node ingest-v2 <path> --visibility public`
//! - `sum-node download <merkle_root_hex> --output <path>`
//!
//! It does not bind to any unstable `ObjectStore` Rust trait and does not
//! attempt range reads, Private V2, or V1 fallback.
//!
//! Layout: a [`SnipV2CliConfig`] plain config, a set of **pure** parser
//! functions ([`parse_ingest_stdout`], [`check_lifecycle`]) that never touch
//! the OS, and a thin [`SnipV2Cli`] wrapper that performs the actual process
//! invocation. Parser tests cover all parse paths without ever spawning
//! `sum-node`.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;

use omni_types::phase5::{
    SnipV2Lifecycle, SnipV2ObjectId, SnipV2ObjectRef, SnipV2ParseError,
};

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for invoking the `sum-node` CLI.
#[derive(Debug, Clone)]
pub struct SnipV2CliConfig {
    /// Path to (or resolvable name of) the `sum-node` binary.
    pub binary_path: PathBuf,
    /// Optional seed file passed as `--seed <path>` when present.
    pub seed_path: Option<PathBuf>,
    /// Extra trailing arguments appended verbatim. Default empty.
    pub extra_args: Vec<String>,
    /// Whether `Pending` / `Abandoned` lifecycles are tolerated.
    /// Default `false` → those lifecycles cause a typed error.
    pub allow_non_active: bool,
}

impl Default for SnipV2CliConfig {
    fn default() -> Self {
        Self {
            binary_path: PathBuf::from("sum-node"),
            seed_path: None,
            extra_args: Vec::new(),
            allow_non_active: false,
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

/// All failure modes of the SNIP V2 CLI adapter, typed.
#[derive(Debug, thiserror::Error)]
pub enum SnipV2Error {
    #[error("failed to spawn SNIP V2 binary: {0}")]
    CommandSpawn(#[from] io::Error),

    #[error("SNIP V2 ingest-v2 exited non-zero (code={code}): {stderr}")]
    NonZeroExit { code: i32, stderr: String },

    #[error("local input not found or not a regular file: {}", path.display())]
    InputNotFound { path: PathBuf },

    #[error("SNIP V2 ingest stdout is missing required `merkle_root:` line")]
    MissingMerkleRootLine,

    #[error("SNIP V2 ingest stdout is missing required `lifecycle:` line")]
    MissingLifecycleLine,

    #[error("SNIP V2 ingest stdout has invalid merkle root: {0}")]
    InvalidMerkleRoot(#[from] SnipV2ParseError),

    #[error("SNIP V2 ingest stdout has unknown lifecycle token: {0}")]
    UnknownLifecycle(String),

    #[error("SNIP V2 lifecycle '{0}' is not allowed (allow_non_active=false)")]
    UnsupportedLifecycle(SnipV2Lifecycle),

    #[error("SNIP V2 stdout parse failure: {reason}")]
    ParseFailure { reason: String },

    #[error("SNIP V2 download exited non-zero (code={code}): {stderr}")]
    DownloadFailed { code: i32, stderr: String },
}

// ── Pure parsers ──────────────────────────────────────────────────────────────

/// Parse the documented `sum-node ingest-v2` stdout. Pure function — never
/// touches the OS — so the unit tests below cover every code path without
/// spawning the binary.
///
/// Recognised lines:
/// - `merkle_root: 0x<64 hex>`
/// - `lifecycle: Active|Pending|Abandoned`
///
/// Other lines (banners, blank lines, future diagnostic output) are ignored.
/// If a key appears more than once, the first occurrence wins.
///
/// The `lifecycle` value is returned verbatim; gating Pending/Abandoned is
/// the job of [`check_lifecycle`] so callers can opt-in to non-Active
/// lifecycles when they need to.
pub fn parse_ingest_stdout(stdout: &str) -> Result<SnipV2ObjectRef, SnipV2Error> {
    let mut merkle: Option<SnipV2ObjectId> = None;
    let mut lifecycle: Option<SnipV2Lifecycle> = None;

    for raw in stdout.lines() {
        let line = raw.trim();
        if let Some(value) = strip_key(line, "merkle_root") {
            if merkle.is_some() {
                continue;
            }
            merkle = Some(SnipV2ObjectId::from_str(value)?);
        } else if let Some(value) = strip_key(line, "lifecycle") {
            if lifecycle.is_some() {
                continue;
            }
            lifecycle = Some(
                SnipV2Lifecycle::from_str(value)
                    .map_err(|e| SnipV2Error::UnknownLifecycle(e.0))?,
            );
        }
    }

    let merkle_root = merkle.ok_or(SnipV2Error::MissingMerkleRootLine)?;
    let lifecycle = lifecycle.ok_or(SnipV2Error::MissingLifecycleLine)?;

    Ok(SnipV2ObjectRef {
        merkle_root,
        lifecycle,
        plaintext_size_bytes: None,
    })
}

fn strip_key<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let (k, v) = line.split_once(':')?;
    if k.trim() == key {
        Some(v.trim())
    } else {
        None
    }
}

/// Reject `Pending` and `Abandoned` unless the caller explicitly opted in.
pub fn check_lifecycle(
    lc: &SnipV2Lifecycle,
    allow_non_active: bool,
) -> Result<(), SnipV2Error> {
    match lc {
        SnipV2Lifecycle::Active => Ok(()),
        SnipV2Lifecycle::Pending | SnipV2Lifecycle::Abandoned => {
            if allow_non_active {
                Ok(())
            } else {
                Err(SnipV2Error::UnsupportedLifecycle(*lc))
            }
        }
    }
}

// ── Adapter trait ─────────────────────────────────────────────────────────────

/// The minimal `sum-node` surface OmniNode depends on. Implemented for
/// [`SnipV2Cli`] (the real CLI wrapper) and any test fake.
///
/// Stage-2 orchestration in [`crate::snip_v2_artifacts`] takes an
/// `&impl SnipV2Adapter`, so unit tests can substitute an in-memory fake
/// without spawning the real binary.
pub trait SnipV2Adapter {
    fn ingest_public(&self, path: &Path) -> Result<SnipV2ObjectRef, SnipV2Error>;
    fn download_public(
        &self,
        root: &SnipV2ObjectId,
        output_path: &Path,
    ) -> Result<(), SnipV2Error>;
}

// ── Process wrapper ───────────────────────────────────────────────────────────

/// Thin wrapper around the `sum-node` CLI. Owns a [`SnipV2CliConfig`] and
/// is the only place in this crate that calls [`std::process::Command`].
pub struct SnipV2Cli {
    config: SnipV2CliConfig,
}

impl SnipV2Adapter for SnipV2Cli {
    fn ingest_public(&self, path: &Path) -> Result<SnipV2ObjectRef, SnipV2Error> {
        SnipV2Cli::ingest_public(self, path)
    }

    fn download_public(
        &self,
        root: &SnipV2ObjectId,
        output_path: &Path,
    ) -> Result<(), SnipV2Error> {
        SnipV2Cli::download_public(self, root, output_path)
    }
}

impl SnipV2Cli {
    pub fn new(config: SnipV2CliConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &SnipV2CliConfig {
        &self.config
    }

    /// Run `sum-node ingest-v2 <path> --visibility public` and parse the
    /// resulting stdout into a [`SnipV2ObjectRef`].
    pub fn ingest_public(&self, path: &Path) -> Result<SnipV2ObjectRef, SnipV2Error> {
        if !path.is_file() {
            return Err(SnipV2Error::InputNotFound {
                path: path.to_path_buf(),
            });
        }

        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("ingest-v2")
            .arg(path)
            .arg("--visibility")
            .arg("public");
        if let Some(seed) = &self.config.seed_path {
            cmd.arg("--seed").arg(seed);
        }
        for a in &self.config.extra_args {
            cmd.arg(a);
        }

        let output = cmd.output()?;
        if !output.status.success() {
            return Err(SnipV2Error::NonZeroExit {
                code: output.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        let stdout = std::str::from_utf8(&output.stdout).map_err(|_| {
            SnipV2Error::ParseFailure {
                reason: "non-utf8 stdout".into(),
            }
        })?;
        let object_ref = parse_ingest_stdout(stdout)?;
        check_lifecycle(&object_ref.lifecycle, self.config.allow_non_active)?;

        tracing::info!(
            merkle = %object_ref.merkle_root,
            lifecycle = %object_ref.lifecycle,
            "SNIP V2 ingest succeeded"
        );
        Ok(object_ref)
    }

    /// Run `sum-node download <root> --output <path>`.
    ///
    /// Returns `Ok(())` on a clean exit. No post-exit assertion is made on
    /// the size or contents of `output_path` — zero-byte objects are valid
    /// unless SNIP documents otherwise, and content verification against
    /// `root` is deferred to a later stage.
    pub fn download_public(
        &self,
        root: &SnipV2ObjectId,
        output_path: &Path,
    ) -> Result<(), SnipV2Error> {
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("download")
            .arg(root.to_hex())
            .arg("--output")
            .arg(output_path);
        if let Some(seed) = &self.config.seed_path {
            cmd.arg("--seed").arg(seed);
        }
        for a in &self.config.extra_args {
            cmd.arg(a);
        }

        let output = cmd.output()?;
        if !output.status.success() {
            return Err(SnipV2Error::DownloadFailed {
                code: output.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        tracing::info!(
            merkle = %root,
            output = %output_path.display(),
            "SNIP V2 download succeeded"
        );
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_HEX: &str =
        "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn parse_ingest_stdout_success_minimal() {
        let s = format!("merkle_root: {VALID_HEX}\nlifecycle: Active\n");
        let r = parse_ingest_stdout(&s).unwrap();
        assert_eq!(r.lifecycle, SnipV2Lifecycle::Active);
        assert_eq!(r.merkle_root.to_hex(), VALID_HEX);
        assert_eq!(r.plaintext_size_bytes, None);
    }

    #[test]
    fn parse_ingest_stdout_success_with_extra_lines() {
        let s = format!(
            "[sum-node v0.4.0-rc3] starting ingest\n\
             working...\n\
             \n\
             merkle_root: {VALID_HEX}\n\
             lifecycle: Active\n\
             done.\n"
        );
        let r = parse_ingest_stdout(&s).unwrap();
        assert_eq!(r.lifecycle, SnipV2Lifecycle::Active);
        assert_eq!(r.merkle_root.to_hex(), VALID_HEX);
    }

    #[test]
    fn parse_ingest_stdout_lifecycle_pending_parsed_without_error() {
        let s = format!("merkle_root: {VALID_HEX}\nlifecycle: Pending\n");
        let r = parse_ingest_stdout(&s).unwrap();
        assert_eq!(r.lifecycle, SnipV2Lifecycle::Pending);
    }

    #[test]
    fn parse_ingest_stdout_lifecycle_abandoned_parsed_without_error() {
        let s = format!("merkle_root: {VALID_HEX}\nlifecycle: Abandoned\n");
        let r = parse_ingest_stdout(&s).unwrap();
        assert_eq!(r.lifecycle, SnipV2Lifecycle::Abandoned);
    }

    #[test]
    fn parse_ingest_stdout_missing_merkle_root() {
        let s = "lifecycle: Active\n";
        assert!(matches!(
            parse_ingest_stdout(s),
            Err(SnipV2Error::MissingMerkleRootLine)
        ));
    }

    #[test]
    fn parse_ingest_stdout_missing_lifecycle() {
        let s = format!("merkle_root: {VALID_HEX}\n");
        assert!(matches!(
            parse_ingest_stdout(&s),
            Err(SnipV2Error::MissingLifecycleLine)
        ));
    }

    #[test]
    fn parse_ingest_stdout_malformed_merkle_root_no_prefix() {
        let s = "merkle_root: abcd\nlifecycle: Active\n";
        let err = parse_ingest_stdout(s).unwrap_err();
        assert!(matches!(
            err,
            SnipV2Error::InvalidMerkleRoot(SnipV2ParseError::MissingHexPrefix)
        ));
    }

    #[test]
    fn parse_ingest_stdout_malformed_merkle_root_uppercase() {
        let bad = format!("0x{}", "A".repeat(64));
        let s = format!("merkle_root: {bad}\nlifecycle: Active\n");
        let err = parse_ingest_stdout(&s).unwrap_err();
        assert!(matches!(
            err,
            SnipV2Error::InvalidMerkleRoot(SnipV2ParseError::NonLowercaseHex { .. })
        ));
    }

    #[test]
    fn parse_ingest_stdout_malformed_merkle_root_short() {
        let bad = format!("0x{}", "a".repeat(63));
        let s = format!("merkle_root: {bad}\nlifecycle: Active\n");
        let err = parse_ingest_stdout(&s).unwrap_err();
        assert!(matches!(
            err,
            SnipV2Error::InvalidMerkleRoot(SnipV2ParseError::WrongLength { got: 65 })
        ));
    }

    #[test]
    fn parse_ingest_stdout_unknown_lifecycle_token() {
        let s = format!("merkle_root: {VALID_HEX}\nlifecycle: WeirdValue\n");
        match parse_ingest_stdout(&s) {
            Err(SnipV2Error::UnknownLifecycle(token)) => {
                assert_eq!(token, "WeirdValue");
            }
            other => panic!("expected UnknownLifecycle, got {other:?}"),
        }
    }

    #[test]
    fn check_lifecycle_strict_rejects_pending() {
        let err = check_lifecycle(&SnipV2Lifecycle::Pending, false).unwrap_err();
        assert!(matches!(
            err,
            SnipV2Error::UnsupportedLifecycle(SnipV2Lifecycle::Pending)
        ));
    }

    #[test]
    fn check_lifecycle_strict_rejects_abandoned() {
        let err = check_lifecycle(&SnipV2Lifecycle::Abandoned, false).unwrap_err();
        assert!(matches!(
            err,
            SnipV2Error::UnsupportedLifecycle(SnipV2Lifecycle::Abandoned)
        ));
    }

    #[test]
    fn check_lifecycle_permissive_allows_pending() {
        assert!(check_lifecycle(&SnipV2Lifecycle::Pending, true).is_ok());
        assert!(check_lifecycle(&SnipV2Lifecycle::Abandoned, true).is_ok());
        assert!(check_lifecycle(&SnipV2Lifecycle::Active, false).is_ok());
    }

    #[test]
    fn cli_config_default_uses_sum_node() {
        let c = SnipV2CliConfig::default();
        assert_eq!(c.binary_path, PathBuf::from("sum-node"));
        assert!(c.seed_path.is_none());
        assert!(c.extra_args.is_empty());
        assert!(!c.allow_non_active);
    }
}
