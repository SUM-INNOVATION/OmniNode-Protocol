//! SNIP V2 CLI adapter — wraps the `sum-node` ingest/download commands.
//!
//! Stage-1 scope of OmniNode Phase 5. This module deliberately speaks only to
//! the **Public V2** surface of `sum-node` and only via the documented CLI
//! contract:
//!
//! - `sum-node ingest-v2 <path> --visibility public`
//! - `sum-node download <merkle_root_hex> --output <path>`
//! - `sum-node resume <merkle_root_hex> <path>` (issue #149)
//!
//! plus one read-only chain call, `storage_getFileInfoV2`, against the RPC
//! endpoint `sum-node` itself uses (issue #149; see
//! [`crate::snip_v2_publish`]).
//!
//! It does not bind to any unstable `ObjectStore` Rust trait and does not
//! attempt range reads, Private V2, or V1 fallback.
//!
//! `sum-node` reads its RPC endpoint and signing key from `SUM_RPC_URL` and
//! `SUM_KEY_FILE` (or the matching global flags, which this adapter does not
//! pass). The adapter resolves both the same way — its config, else the
//! environment, else `sum-node`'s default endpoint — and, when its config
//! sets them, passes them to `sum-node` through those variables, so the
//! chain it reads and the account it compares owners against are the ones
//! `sum-node` registers with.
//!
//! Layout: a [`SnipV2CliConfig`] plain config, a set of **pure** parser
//! functions ([`parse_ingest_stdout`], [`check_lifecycle`]) that never touch
//! the OS, and a thin [`SnipV2Cli`] wrapper that performs the actual process
//! invocation. Parser tests cover all parse paths without ever spawning
//! `sum-node`.

use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::str::FromStr;
use std::time::Duration;

use omni_types::phase5::{
    SnipV2Lifecycle, SnipV2ObjectId, SnipV2ObjectRef, SnipV2ParseError,
};

use crate::snip_v2_publish::{SnipV2FileInfo, SnipV2ResumeOutcome};

/// `sum-node`'s `--rpc-url` default (env `SUM_RPC_URL`).
pub const SUM_NODE_DEFAULT_RPC_URL: &str = "http://127.0.0.1:9944";
/// Environment variable `sum-node` reads its RPC endpoint from.
pub const SUM_RPC_URL_ENV: &str = "SUM_RPC_URL";
/// Environment variable `sum-node` reads its key file path from.
pub const SUM_KEY_FILE_ENV: &str = "SUM_KEY_FILE";

/// Exit status of `sum-node resume` (SNIP `ingest_v2::resume_exit`; `2` is
/// the argument parser's usage error).
pub mod resume_exit {
    pub const ACTIVE: i32 = 0;
    pub const ERROR: i32 = 1;
    pub const USAGE: i32 = 2;
    pub const PENDING: i32 = 3;
    pub const ABANDONED: i32 = 4;
    pub const ROOT_MISMATCH: i32 = 5;
}

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
    /// Chain RPC endpoint. `None`: `SUM_RPC_URL`, else `sum-node`'s default.
    /// When set, `sum-node` is given it through `SUM_RPC_URL`.
    pub rpc_url: Option<String>,
    /// Key file `sum-node` signs with (64 hex characters, `sum-node
    /// --key-file`). `None`: `SUM_KEY_FILE`. When set, `sum-node` is given it
    /// through `SUM_KEY_FILE`.
    pub key_file: Option<PathBuf>,
}

impl Default for SnipV2CliConfig {
    fn default() -> Self {
        Self {
            binary_path: PathBuf::from("sum-node"),
            seed_path: None,
            extra_args: Vec::new(),
            allow_non_active: false,
            rpc_url: None,
            key_file: None,
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

    // ── Issue #149: publication decided from the chain's record ─────────────
    /// `sum-node` exited 0 without the outcome lines it prints for every
    /// outcome that has a root (`ingest-v2` prints none when registration
    /// fails or is refused). `diagnostics` is what it reported instead.
    #[error("sum-node {command} exited 0 without its outcome lines ({cause}); sum-node reported: {diagnostics}")]
    OutcomeMissing {
        command: &'static str,
        cause: String,
        diagnostics: String,
    },

    #[error("sum-node resume failed ({status}): {diagnostics}")]
    ResumeFailed { status: String, diagnostics: String },

    #[error("SNIP V2 chain lookup (storage_getFileInfoV2 at {url}) failed for root {root}: {detail}")]
    ChainLookup {
        url: String,
        root: String,
        detail: String,
    },

    #[error("SNIP V2 chain record for root {root} is malformed: {detail}")]
    ChainRecordMalformed { root: String, detail: String },

    #[error("SNIP V2 root {root} is registered on chain with {field} = {chain}, but these bytes need {expected}; refusing to use or resume it")]
    ChainRecordMismatch {
        root: String,
        field: &'static str,
        chain: String,
        expected: String,
    },

    #[error("SNIP V2 root {root} is Pending on chain, registered by {owner}, not by the configured account {account}; refusing to resume or register it")]
    PendingOwnedByOther {
        root: String,
        owner: String,
        account: String,
    },

    #[error("SNIP V2 root {root} is Abandoned on chain (registered by {owner}); it cannot be activated or registered again ({detail})")]
    AbandonedOnChain {
        root: String,
        owner: String,
        detail: String,
    },

    #[error("SNIP V2 root {root} is still Pending on chain after one resume: {detail}")]
    StillPending { root: String, detail: String },

    #[error("SNIP V2 root mismatch: expected {expected}, got {reported}: {detail}")]
    RootMismatch {
        expected: String,
        reported: String,
        detail: String,
    },

    #[error("SNIP V2 root {root} has no chain record after: {attempt}")]
    NotRegistered { root: String, attempt: String },

    #[error("SNIP V2 publication of root {root} is not confirmed: {attempt}; re-reading the chain record then failed: {recheck}")]
    PublicationUnconfirmed {
        root: String,
        attempt: String,
        recheck: String,
    },

    #[error("cannot determine the account sum-node signs with: {0}")]
    PublisherAccount(String),

    /// The adapter does not implement a capability idempotent publication
    /// needs (a [`SnipV2Adapter`] default method).
    #[error("this SNIP adapter does not provide {capability}; idempotent publication needs it")]
    Unsupported { capability: &'static str },

    /// The chain records the root Active and matching, but the object could
    /// not be fetched back, or what came back is not the published bytes.
    #[error("SNIP V2 root {root} is Active on chain but its content is not confirmed: {detail}")]
    ContentUnverified { root: String, detail: String },
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
///
/// Issue #149 adds the three methods idempotent publication
/// ([`crate::snip_v2_publish::publish_public_idempotent`]) decides from.
/// They have defaults that return [`SnipV2Error::Unsupported`], so an
/// existing implementation still compiles unchanged; publishing through
/// it then refuses clearly instead of registering blindly.
pub trait SnipV2Adapter {
    fn ingest_public(&self, path: &Path) -> Result<SnipV2ObjectRef, SnipV2Error>;
    fn download_public(
        &self,
        root: &SnipV2ObjectId,
        output_path: &Path,
    ) -> Result<(), SnipV2Error>;

    /// The chain's `storage_getFileInfoV2` record of `root`. `Ok(None)` when
    /// the chain has none; an unreachable RPC or an error answer is `Err`.
    fn file_info_v2(&self, root: &SnipV2ObjectId) -> Result<Option<SnipV2FileInfo>, SnipV2Error> {
        let _ = root;
        Err(SnipV2Error::Unsupported {
            capability: "storage_getFileInfoV2 (the chain record of a root)",
        })
    }

    /// The account (base58) [`Self::ingest_public`] and
    /// [`Self::resume_public`] sign with: the owner of what they register.
    fn publisher_account(&self) -> Result<String, SnipV2Error> {
        Err(SnipV2Error::Unsupported {
            capability: "the publishing account",
        })
    }

    /// `sum-node resume <root> <path>`: the post-registration stages (push,
    /// manifest push, coverage, activation) for a Pending root. Never
    /// registers.
    fn resume_public(
        &self,
        root: &SnipV2ObjectId,
        path: &Path,
    ) -> Result<SnipV2ResumeOutcome, SnipV2Error> {
        let _ = (root, path);
        Err(SnipV2Error::Unsupported {
            capability: "sum-node resume",
        })
    }
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

    fn file_info_v2(&self, root: &SnipV2ObjectId) -> Result<Option<SnipV2FileInfo>, SnipV2Error> {
        SnipV2Cli::file_info_v2(self, root)
    }

    fn publisher_account(&self) -> Result<String, SnipV2Error> {
        let path = self.key_file().ok_or_else(|| {
            SnipV2Error::PublisherAccount(format!(
                "no key file: neither the adapter config nor {SUM_KEY_FILE_ENV} names the key \
                 sum-node signs with"
            ))
        })?;
        snip_account_from_key_file(&path)
    }

    fn resume_public(
        &self,
        root: &SnipV2ObjectId,
        path: &Path,
    ) -> Result<SnipV2ResumeOutcome, SnipV2Error> {
        SnipV2Cli::resume_public(self, root, path)
    }
}

impl SnipV2Cli {
    pub fn new(config: SnipV2CliConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &SnipV2CliConfig {
        &self.config
    }

    /// The chain RPC endpoint `sum-node` uses: the config's, else
    /// `SUM_RPC_URL`, else `sum-node`'s default.
    pub fn rpc_url(&self) -> String {
        self.config
            .rpc_url
            .clone()
            .or_else(|| std::env::var(SUM_RPC_URL_ENV).ok().filter(|v| !v.is_empty()))
            .unwrap_or_else(|| SUM_NODE_DEFAULT_RPC_URL.to_string())
    }

    /// The key file `sum-node` signs with: the config's, else `SUM_KEY_FILE`.
    pub fn key_file(&self) -> Option<PathBuf> {
        self.config.key_file.clone().or_else(|| {
            std::env::var_os(SUM_KEY_FILE_ENV)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        })
    }

    /// `sum-node <subcommand>`, with the configured endpoint and key passed
    /// through `sum-node`'s own environment variables.
    fn command(&self, subcommand: &str) -> Command {
        let mut cmd = Command::new(&self.config.binary_path);
        if let Some(url) = &self.config.rpc_url {
            cmd.env(SUM_RPC_URL_ENV, url);
        }
        if let Some(key) = &self.config.key_file {
            cmd.env(SUM_KEY_FILE_ENV, key);
        }
        cmd.arg(subcommand);
        cmd
    }

    fn push_trailing_args(&self, cmd: &mut Command) {
        if let Some(seed) = &self.config.seed_path {
            cmd.arg("--seed").arg(seed);
        }
        for a in &self.config.extra_args {
            cmd.arg(a);
        }
    }

    /// Run `sum-node ingest-v2 <path> --visibility public` and parse the
    /// resulting stdout into a [`SnipV2ObjectRef`].
    ///
    /// `sum-node` exits 0 for every ingest outcome, and prints its
    /// `merkle_root:` / `lifecycle:` lines only for those that have a root:
    /// a refused or failed registration prints neither and logs why. A
    /// missing line is therefore reported as [`SnipV2Error::OutcomeMissing`]
    /// with what `sum-node` logged, and a non-zero exit with its stderr and
    /// log, so the real reason is never replaced by a parse error.
    pub fn ingest_public(&self, path: &Path) -> Result<SnipV2ObjectRef, SnipV2Error> {
        if !path.is_file() {
            return Err(SnipV2Error::InputNotFound {
                path: path.to_path_buf(),
            });
        }

        let mut cmd = self.command("ingest-v2");
        cmd.arg(path).arg("--visibility").arg("public");
        self.push_trailing_args(&mut cmd);

        let output = cmd.output()?;
        if !output.status.success() {
            return Err(SnipV2Error::NonZeroExit {
                code: output.status.code().unwrap_or(-1),
                stderr: format!("{}; {}", output.status, output_diagnostics(&output)),
            });
        }

        let stdout = std::str::from_utf8(&output.stdout).map_err(|_| {
            SnipV2Error::ParseFailure {
                reason: "non-utf8 stdout".into(),
            }
        })?;
        let object_ref = parse_ingest_stdout(stdout).map_err(|e| match e {
            SnipV2Error::MissingMerkleRootLine | SnipV2Error::MissingLifecycleLine => {
                SnipV2Error::OutcomeMissing {
                    command: "ingest-v2",
                    cause: e.to_string(),
                    diagnostics: output_diagnostics(&output),
                }
            }
            other => other,
        })?;
        check_lifecycle(&object_ref.lifecycle, self.config.allow_non_active)?;

        tracing::info!(
            merkle = %object_ref.merkle_root,
            lifecycle = %object_ref.lifecycle,
            "SNIP V2 ingest succeeded"
        );
        Ok(object_ref)
    }

    /// Run `sum-node resume <root> <path>` and map its exit status
    /// ([`resume_exit`]). `0` counts as Active only with the matching
    /// `merkle_root:` and `lifecycle: Active` lines.
    pub fn resume_public(
        &self,
        root: &SnipV2ObjectId,
        path: &Path,
    ) -> Result<SnipV2ResumeOutcome, SnipV2Error> {
        if !path.is_file() {
            return Err(SnipV2Error::InputNotFound {
                path: path.to_path_buf(),
            });
        }
        let mut cmd = self.command("resume");
        cmd.arg(root.to_hex()).arg(path);
        self.push_trailing_args(&mut cmd);

        let output = cmd.output()?;
        let diagnostics = output_diagnostics(&output);
        let outcome = match output.status.code() {
            Some(resume_exit::ACTIVE) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let reported = parse_ingest_stdout(&stdout).map_err(|e| {
                    SnipV2Error::OutcomeMissing {
                        command: "resume",
                        cause: e.to_string(),
                        diagnostics: diagnostics.clone(),
                    }
                })?;
                if reported.merkle_root != *root {
                    return Err(SnipV2Error::RootMismatch {
                        expected: root.to_hex(),
                        reported: reported.merkle_root.to_hex(),
                        detail: "sum-node resume reported a different root".into(),
                    });
                }
                if reported.lifecycle != SnipV2Lifecycle::Active {
                    return Err(SnipV2Error::ResumeFailed {
                        status: output.status.to_string(),
                        diagnostics: format!(
                            "exit 0 with `lifecycle: {}`; {diagnostics}",
                            reported.lifecycle
                        ),
                    });
                }
                SnipV2ResumeOutcome::Active
            }
            Some(resume_exit::PENDING) => SnipV2ResumeOutcome::Pending { diagnostics },
            Some(resume_exit::ABANDONED) => SnipV2ResumeOutcome::Abandoned { diagnostics },
            Some(resume_exit::ROOT_MISMATCH) => SnipV2ResumeOutcome::RootMismatch { diagnostics },
            _ => {
                return Err(SnipV2Error::ResumeFailed {
                    status: output.status.to_string(),
                    diagnostics,
                });
            }
        };
        tracing::info!(merkle = %root, ?outcome, "SNIP V2 resume finished");
        Ok(outcome)
    }

    /// The chain's `storage_getFileInfoV2` record of `root`, read from
    /// [`Self::rpc_url`]. `Ok(None)` when the chain answers `null`.
    pub fn file_info_v2(
        &self,
        root: &SnipV2ObjectId,
    ) -> Result<Option<SnipV2FileInfo>, SnipV2Error> {
        let url = self.rpc_url();
        let root_hex = root.to_hex();
        let result = json_rpc_call(
            &url,
            "storage_getFileInfoV2",
            serde_json::json!([root_hex, null, null]),
        )
        .map_err(|detail| SnipV2Error::ChainLookup {
            url: url.clone(),
            root: root_hex.clone(),
            detail,
        })?;
        file_info_from_result(&root_hex, result)
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
        let mut cmd = self.command("download");
        cmd.arg(root.to_hex()).arg("--output").arg(output_path);
        self.push_trailing_args(&mut cmd);

        let output = cmd.output()?;
        if !output.status.success() {
            return Err(SnipV2Error::DownloadFailed {
                code: output.status.code().unwrap_or(-1),
                stderr: format!("{}; {}", output.status, output_diagnostics(&output)),
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

// ── Chain lookup ──────────────────────────────────────────────────────────────

/// Whole-request timeout of the `storage_getFileInfoV2` lookup.
const RPC_TIMEOUT: Duration = Duration::from_secs(30);
/// Largest JSON-RPC answer read.
const MAX_RPC_RESPONSE_BYTES: u64 = 4 * 1024 * 1024;

/// One JSON-RPC 2.0 call; the `result` member, or why there is none.
fn json_rpc_call(
    url: &str,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    // Proxy settings in the environment are not read; no redirects.
    let agent = ureq::Agent::config_builder()
        .proxy(None)
        .timeout_global(Some(RPC_TIMEOUT))
        .max_redirects(0)
        .build()
        .new_agent();
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    })
    .to_string();
    let mut resp = agent
        .post(url)
        .header("content-type", "application/json")
        .send(&body)
        .map_err(|e| format!("{method}: {e}"))?;
    let text = resp
        .body_mut()
        .with_config()
        .limit(MAX_RPC_RESPONSE_BYTES)
        .read_to_string()
        .map_err(|e| format!("{method}: reading the answer: {e}"))?;
    json_rpc_result(method, &text)
}

/// The `result` of a JSON-RPC answer body; an `error` member is an error.
fn json_rpc_result(method: &str, text: &str) -> Result<serde_json::Value, String> {
    let envelope: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| format!("{method}: the answer is not JSON: {e}"))?;
    if let Some(err) = envelope.get("error").filter(|e| !e.is_null()) {
        return Err(format!("{method}: JSON-RPC error {err}"));
    }
    envelope
        .get("result")
        .cloned()
        .ok_or_else(|| format!("{method}: the answer has no `result`"))
}

/// `null` is "no record"; anything else must be a whole record.
fn file_info_from_result(
    root_hex: &str,
    result: serde_json::Value,
) -> Result<Option<SnipV2FileInfo>, SnipV2Error> {
    if result.is_null() {
        return Ok(None);
    }
    serde_json::from_value(result)
        .map(Some)
        .map_err(|e| SnipV2Error::ChainRecordMalformed {
            root: root_hex.to_string(),
            detail: e.to_string(),
        })
}

// ── Account ───────────────────────────────────────────────────────────────────

/// The chain account (base58) of the key in a `sum-node --key-file`: 64 hex
/// characters of an Ed25519 seed, surrounding whitespace ignored.
pub fn snip_account_from_key_file(path: &Path) -> Result<String, SnipV2Error> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        SnipV2Error::PublisherAccount(format!("read key file {}: {e}", path.display()))
    })?;
    let hex = text.trim();
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(SnipV2Error::PublisherAccount(format!(
            "key file {} does not hold exactly 64 hex characters",
            path.display()
        )));
    }
    let mut seed = [0u8; 32];
    for (i, pair) in hex.as_bytes().chunks(2).enumerate() {
        let pair = std::str::from_utf8(pair).unwrap_or("00");
        seed[i] = u8::from_str_radix(pair, 16).unwrap_or(0);
    }
    snip_account_from_seed(&seed)
}

/// The chain account (base58) of an Ed25519 seed, as SNIP and the chain
/// derive it ([`account_base58_from_public_key`]).
pub fn snip_account_from_seed(seed: &[u8; 32]) -> Result<String, SnipV2Error> {
    let mut secret = *seed;
    let secret = libp2p::identity::ed25519::SecretKey::try_from_bytes(&mut secret)
        .map_err(|e| SnipV2Error::PublisherAccount(format!("invalid Ed25519 seed: {e}")))?;
    let public = libp2p::identity::ed25519::Keypair::from(secret).public().to_bytes();
    Ok(account_base58_from_public_key(&public))
}

/// The chain's `Address::from_public_key(pk).to_base58()`: the last 20
/// bytes of `blake3(pk)`, then base58 of those bytes followed by the first
/// 4 bytes of `blake3(blake3(address))` (SNIP `sum-net` `identity.rs`).
pub fn account_base58_from_public_key(public_key: &[u8; 32]) -> String {
    let hash = blake3::hash(public_key);
    let mut with_checksum = Vec::with_capacity(24);
    with_checksum.extend_from_slice(&hash.as_bytes()[12..32]);
    let check = blake3::hash(blake3::hash(&with_checksum).as_bytes());
    with_checksum.extend_from_slice(&check.as_bytes()[..4]);
    bs58::encode(with_checksum).into_string()
}

// ── Diagnostics ───────────────────────────────────────────────────────────────

/// Log lines kept, the last ones.
const DIAG_LOG_LINES: usize = 12;
/// Characters kept of one log line.
const DIAG_LINE_CHARS: usize = 600;
/// Characters kept of stderr, the last ones.
const DIAG_STDERR_CHARS: usize = 2000;

fn output_diagnostics(output: &Output) -> String {
    sum_node_diagnostics(&output.stdout, &output.stderr)
}

/// What `sum-node` reported besides its outcome lines: the end of stderr
/// (`main`'s `Error: …`, a usage error, a panic) and its WARN / ERROR log
/// lines. `sum-node` logs through `tracing` to **stdout**, coloured, so the
/// reason for a refused registration is only there.
pub fn sum_node_diagnostics(stdout: &[u8], stderr: &[u8]) -> String {
    let out = strip_ansi(&String::from_utf8_lossy(stdout));
    let log: Vec<String> = out
        .lines()
        .filter(|l| is_warn_or_error_line(l))
        .map(|l| head_chars(l.trim(), DIAG_LINE_CHARS))
        .collect();
    let log = &log[log.len().saturating_sub(DIAG_LOG_LINES)..];
    let err = strip_ansi(&String::from_utf8_lossy(stderr));
    let err = tail_chars(err.trim(), DIAG_STDERR_CHARS).replace('\n', " / ");

    let mut parts = Vec::new();
    if !err.is_empty() {
        parts.push(format!("stderr: {err}"));
    }
    if !log.is_empty() {
        parts.push(format!("log: {}", log.join(" | ")));
    }
    if parts.is_empty() {
        "no stderr and no WARN/ERROR log line".into()
    } else {
        parts.join("; ")
    }
}

/// A `tracing` fmt line at WARN or ERROR: `<timestamp> WARN <target>: …`.
fn is_warn_or_error_line(line: &str) -> bool {
    line.split_whitespace()
        .take(3)
        .any(|w| w == "WARN" || w == "ERROR")
}

/// Remove ANSI CSI sequences (`ESC [ … final`).
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for n in chars.by_ref() {
                    if ('@'..='~').contains(&n) {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

fn head_chars(s: &str, n: usize) -> String {
    match s.char_indices().nth(n) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}

fn tail_chars(s: &str, n: usize) -> String {
    let count = s.chars().count();
    if count <= n {
        return s.to_string();
    }
    let skip = count - n;
    let i = s.char_indices().nth(skip).map(|(i, _)| i).unwrap_or(0);
    format!("…{}", &s[i..])
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
        assert!(c.rpc_url.is_none());
        assert!(c.key_file.is_none());
    }

    // ── Issue #149: diagnostics ─────────────────────────────────────────────

    /// `ingest-v2`'s stdout when the chain refuses the registration, as
    /// `tracing`'s coloured fmt layer writes it (sum-node 0.4.0-rc4).
    const REFUSED_STDOUT: &str = "\u{1b}[2m2026-10-28T01:41:57.000000Z\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m \u{1b}[2msum_node\u{1b}[0m\u{1b}[2m:\u{1b}[0m loaded L1 keypair\n\
         \u{1b}[2m2026-10-28T01:41:59.000000Z\u{1b}[0m \u{1b}[33m WARN\u{1b}[0m \u{1b}[2msum_node\u{1b}[0m\u{1b}[2m:\u{1b}[0m V2 ingest FAILED before chain registration \u{1b}[3mstage\u{1b}[0m\u{1b}[2m=\u{1b}[0mRegister \u{1b}[3msource\u{1b}[0m\u{1b}[2m=\u{1b}[0mRegisterFilePendingV2 failed: merkle_root already registered as V2 file\n";

    #[test]
    fn diagnostics_keep_snips_log_reason_and_stderr() {
        let d = sum_node_diagnostics(REFUSED_STDOUT.as_bytes(), b"");
        assert!(d.contains("merkle_root already registered as V2 file"), "{d}");
        assert!(d.contains("WARN"), "{d}");
        assert!(!d.contains("loaded L1 keypair"), "INFO is not a diagnostic: {d}");
        assert!(!d.contains('\u{1b}'), "ANSI stripped: {d:?}");

        let d = sum_node_diagnostics(b"", b"Error: ingest-v2 requires --key-file\n");
        assert_eq!(d, "stderr: Error: ingest-v2 requires --key-file");

        assert_eq!(
            sum_node_diagnostics(b"merkle_root: 0x00\n", b""),
            "no stderr and no WARN/ERROR log line"
        );
    }

    #[test]
    fn diagnostics_are_bounded() {
        let mut out = String::new();
        for i in 0..100 {
            out.push_str(&format!("2026-10-28T00:00:00Z  WARN x: line {i} {}\n", "y".repeat(5000)));
        }
        let d = sum_node_diagnostics(out.as_bytes(), "e".repeat(10_000).as_bytes());
        assert!(d.contains("line 99"), "the last lines are kept");
        assert!(!d.contains("line 87 "), "only the last 12 lines");
        assert!(d.len() < 20_000, "{}", d.len());
    }

    #[test]
    fn strip_ansi_removes_only_escape_sequences() {
        assert_eq!(strip_ansi("\u{1b}[1;33mWARN\u{1b}[0m x"), "WARN x");
        assert_eq!(strip_ansi("plain [text]"), "plain [text]");
    }

    // ── Issue #149: the configured account ──────────────────────────────────

    /// Vectors from SNIP's own derivation (`e2e-helper l1-address
    /// --seed-hex`, sum-node 0.4.0-rc4 at SNIP 4cdba14): a constant seed
    /// and RFC 8032 test 1's secret key.
    #[test]
    fn account_derivation_matches_snip() {
        assert_eq!(
            snip_account_from_seed(&[7u8; 32]).unwrap(),
            "JXNSrKvo6w4CMWTiRRsypDz9i2XD2srDA"
        );
        let rfc8032_1: [u8; 32] = [
            0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec,
            0x2c, 0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03,
            0x1c, 0xae, 0x7f, 0x60,
        ];
        assert_eq!(
            snip_account_from_seed(&rfc8032_1).unwrap(),
            "M3Bdd1Hmimp3FPLDpvH4jimtdYAFG34k9"
        );
    }

    #[test]
    fn account_derivation_equals_the_chain_crates() {
        for seed in [[0u8; 32], [7u8; 32], [0xffu8; 32], *b"0123456789abcdef0123456789abcdef"] {
            let mut s = seed;
            let secret = libp2p::identity::ed25519::SecretKey::try_from_bytes(&mut s).unwrap();
            let pk = libp2p::identity::ed25519::Keypair::from(secret).public().to_bytes();
            assert_eq!(
                account_base58_from_public_key(&pk),
                sumchain_wire::Address::from_public_key(&pk).to_base58()
            );
            assert_eq!(
                snip_account_from_seed(&seed).unwrap(),
                sumchain_wire::Address::from_public_key(&pk).to_base58()
            );
        }
    }

    #[test]
    fn account_from_key_file_reads_sum_nodes_format() {
        let dir = tempfile::tempdir().unwrap();
        let ok = dir.path().join("k.hex");
        std::fs::write(&ok, format!("{}\n", "07".repeat(32))).unwrap();
        assert_eq!(
            snip_account_from_key_file(&ok).unwrap(),
            "JXNSrKvo6w4CMWTiRRsypDz9i2XD2srDA"
        );
        let upper = dir.path().join("u.hex");
        std::fs::write(&upper, "0A".repeat(32)).unwrap();
        assert!(snip_account_from_key_file(&upper).is_ok());
        for bad in ["07".repeat(31), format!("0x{}", "07".repeat(31)), "zz".repeat(32)] {
            let p = dir.path().join("bad.hex");
            std::fs::write(&p, bad).unwrap();
            assert!(matches!(
                snip_account_from_key_file(&p),
                Err(SnipV2Error::PublisherAccount(_))
            ));
        }
        assert!(matches!(
            snip_account_from_key_file(&dir.path().join("missing")),
            Err(SnipV2Error::PublisherAccount(_))
        ));
    }

    #[test]
    fn publisher_account_uses_the_configured_key_file() {
        let dir = tempfile::tempdir().unwrap();
        let k = dir.path().join("k.hex");
        std::fs::write(&k, "07".repeat(32)).unwrap();
        let cli = SnipV2Cli::new(SnipV2CliConfig {
            key_file: Some(k),
            ..SnipV2CliConfig::default()
        });
        assert_eq!(
            cli.publisher_account().unwrap(),
            "JXNSrKvo6w4CMWTiRRsypDz9i2XD2srDA"
        );
    }

    // ── Issue #149: chain lookup ────────────────────────────────────────────

    #[test]
    fn json_rpc_result_shapes() {
        assert_eq!(
            json_rpc_result("m", r#"{"jsonrpc":"2.0","id":1,"result":null}"#).unwrap(),
            serde_json::Value::Null
        );
        let err = json_rpc_result("m", r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"no"}}"#)
            .unwrap_err();
        assert!(err.contains("-32601"), "{err}");
        assert!(json_rpc_result("m", r#"{"jsonrpc":"2.0","id":1}"#).is_err());
        assert!(json_rpc_result("m", "<html>").is_err());
        // `"error": null` beside a result is not an error.
        assert_eq!(
            json_rpc_result("m", r#"{"result":5,"error":null}"#).unwrap(),
            serde_json::json!(5)
        );
    }

    #[test]
    fn file_info_from_result_null_object_and_malformed() {
        let root = "0x".to_string() + &"ab".repeat(32);
        assert_eq!(file_info_from_result(&root, serde_json::Value::Null).unwrap(), None);
        let rec = serde_json::json!({
            "merkle_root": root, "owner": "o", "plaintext_size_bytes": 1,
            "stored_size_bytes": 1, "chunk_count": 1, "created_at": 3,
            "activated_at_height": null, "visibility": 0, "lifecycle": 0
        });
        assert!(file_info_from_result(&root, rec).unwrap().is_some());
        assert!(matches!(
            file_info_from_result(&root, serde_json::json!({"merkle_root": root})),
            Err(SnipV2Error::ChainRecordMalformed { .. })
        ));
        assert!(matches!(
            file_info_from_result(&root, serde_json::json!("Active")),
            Err(SnipV2Error::ChainRecordMalformed { .. })
        ));
    }

    /// One-shot HTTP server on loopback answering `body` to one request;
    /// returns its URL and the request it received.
    fn serve_once(
        body: &'static str,
    ) -> (String, std::thread::JoinHandle<String>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let h = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut req = Vec::new();
            let mut buf = [0u8; 4096];
            // Headers, then the Content-Length body.
            loop {
                let n = sock.read(&mut buf).unwrap();
                req.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&req).to_string();
                if let Some(end) = text.find("\r\n\r\n") {
                    let len = text
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if req.len() >= end + 4 + len {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            sock.write_all(resp.as_bytes()).unwrap();
            String::from_utf8_lossy(&req).to_string()
        });
        (url, h)
    }

    fn cli_at(url: &str) -> SnipV2Cli {
        SnipV2Cli::new(SnipV2CliConfig {
            rpc_url: Some(url.to_string()),
            ..SnipV2CliConfig::default()
        })
    }

    #[test]
    fn file_info_v2_asks_the_chain_and_reads_null_as_absent() {
        let root = SnipV2ObjectId::from_bytes([0xab; 32]);
        let (url, h) = serve_once(r#"{"jsonrpc":"2.0","id":1,"result":null}"#);
        assert_eq!(cli_at(&url).file_info_v2(&root).unwrap(), None);
        let req = h.join().unwrap();
        assert!(req.starts_with("POST "), "{req}");
        assert!(req.contains(r#""method":"storage_getFileInfoV2""#), "{req}");
        assert!(req.contains(&format!(r#""params":["{}",null,null]"#, root.to_hex())), "{req}");
    }

    #[test]
    fn file_info_v2_parses_a_record() {
        let root = SnipV2ObjectId::from_bytes([0xab; 32]);
        let (url, h) = serve_once(
            r#"{"jsonrpc":"2.0","id":1,"result":{"merkle_root":"0xabababababababababababababababababababababababababababababababab","owner":"o","plaintext_size_bytes":16,"stored_size_bytes":16,"chunk_count":1,"fee_pool":0,"created_at":7,"activated_at_height":9,"abandoned_at_height":null,"assignment_height":7,"visibility":0,"lifecycle":1,"access_list":[],"access_total":0,"access_offset":0,"predecessor_root":null}}"#,
        );
        let info = cli_at(&url).file_info_v2(&root).unwrap().unwrap();
        h.join().unwrap();
        assert_eq!(info.lifecycle, 1);
        assert_eq!(info.activated_at_height, Some(9));
    }

    #[test]
    fn file_info_v2_rpc_error_and_unreachable_are_lookup_errors() {
        let root = SnipV2ObjectId::from_bytes([0xab; 32]);
        let (url, h) =
            serve_once(r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"boom"}}"#);
        let err = cli_at(&url).file_info_v2(&root).unwrap_err();
        h.join().unwrap();
        assert!(matches!(err, SnipV2Error::ChainLookup { .. }), "{err}");
        assert!(err.to_string().contains("boom"), "{err}");

        // A port nothing listens on.
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let err = cli_at(&format!("http://127.0.0.1:{port}"))
            .file_info_v2(&root)
            .unwrap_err();
        assert!(matches!(err, SnipV2Error::ChainLookup { .. }), "{err}");
    }

    // ── Issue #149: sum-node exit statuses, against a scripted sum-node ─────

    #[cfg(unix)]
    fn fake_sum_node(dir: &Path, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join("sum-node");
        std::fs::write(&p, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    #[cfg(unix)]
    fn fake_cli(dir: &Path, script: &str) -> SnipV2Cli {
        SnipV2Cli::new(SnipV2CliConfig {
            binary_path: fake_sum_node(dir, script),
            rpc_url: Some("http://127.0.0.1:1".into()),
            key_file: Some(dir.join("key.hex")),
            ..SnipV2CliConfig::default()
        })
    }

    #[cfg(unix)]
    fn input(dir: &Path) -> PathBuf {
        let p = dir.join("in.bin");
        std::fs::write(&p, b"x").unwrap();
        p
    }

    #[cfg(unix)]
    #[test]
    fn ingest_exit_zero_without_outcome_reports_snips_reason() {
        let dir = tempfile::tempdir().unwrap();
        let cli = fake_cli(
            dir.path(),
            "printf '%s\n' '2026-10-28T01:41:59Z  WARN sum_node: V2 ingest FAILED before chain registration stage=Register source=RegisterFilePendingV2 failed: merkle_root already registered as V2 file'",
        );
        let err = cli.ingest_public(&input(dir.path())).unwrap_err();
        match &err {
            SnipV2Error::OutcomeMissing { command, cause, diagnostics } => {
                assert_eq!(*command, "ingest-v2");
                assert!(cause.contains("merkle_root"), "{cause}");
                assert!(diagnostics.contains("already registered"), "{diagnostics}");
            }
            other => panic!("{other}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn ingest_passes_the_configured_endpoint_and_key_to_sum_node() {
        let dir = tempfile::tempdir().unwrap();
        let root = format!("0x{}", "ab".repeat(32));
        let cli = fake_cli(
            dir.path(),
            &format!(
                "[ \"$1\" = ingest-v2 ] && [ \"$3 $4\" = '--visibility public' ] || exit 9\n\
                 [ \"$SUM_RPC_URL\" = http://127.0.0.1:1 ] || exit 8\n\
                 [ \"$SUM_KEY_FILE\" = '{}' ] || exit 7\n\
                 echo 'merkle_root: {root}'; echo 'lifecycle: Active'",
                dir.path().join("key.hex").display()
            ),
        );
        let r = cli.ingest_public(&input(dir.path())).unwrap();
        assert_eq!(r.merkle_root.to_hex(), root);
    }

    #[cfg(unix)]
    #[test]
    fn ingest_nonzero_exit_keeps_stderr_and_log() {
        let dir = tempfile::tempdir().unwrap();
        let cli = fake_cli(
            dir.path(),
            "echo '2026-10-28T00:00:00Z ERROR sum_node: rpc down'; echo 'Error: boom' >&2; exit 1",
        );
        let err = cli.ingest_public(&input(dir.path())).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, SnipV2Error::NonZeroExit { code: 1, .. }), "{msg}");
        assert!(msg.contains("Error: boom") && msg.contains("rpc down"), "{msg}");
    }

    #[cfg(unix)]
    #[test]
    fn resume_exit_statuses_map_to_outcomes() {
        let root = SnipV2ObjectId::from_bytes([0xab; 32]);
        let hex = root.to_hex();
        let dir = tempfile::tempdir().unwrap();
        let path = input(dir.path());
        let run = |script: String| {
            let d = tempfile::tempdir().unwrap();
            let cli = fake_cli(d.path(), &script);
            cli.resume_public(&root, &path)
        };

        let active = format!(
            "[ \"$1 $2 $3\" = 'resume {hex} {}' ] || exit 9\n\
             echo 'merkle_root: {hex}'; echo 'lifecycle: Active'",
            path.display()
        );
        assert_eq!(run(active).unwrap(), SnipV2ResumeOutcome::Active);

        let pending = format!(
            "echo '2026-10-28T00:00:00Z  WARN sum_node: RESUME: still PENDING'; \
             echo 'merkle_root: {hex}'; echo 'lifecycle: Pending'; exit 3"
        );
        match run(pending).unwrap() {
            SnipV2ResumeOutcome::Pending { diagnostics } => {
                assert!(diagnostics.contains("still PENDING"), "{diagnostics}")
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            run(format!("echo 'merkle_root: {hex}'; echo 'lifecycle: Abandoned'; exit 4")).unwrap(),
            SnipV2ResumeOutcome::Abandoned { .. }
        ));
        assert!(matches!(
            run("exit 5".into()).unwrap(),
            SnipV2ResumeOutcome::RootMismatch { .. }
        ));
        for code in [1, 2, 6, 101] {
            let err = run(format!("echo 'Error: code {code}' >&2; exit {code}")).unwrap_err();
            match &err {
                SnipV2Error::ResumeFailed { status, diagnostics } => {
                    assert!(status.contains(&code.to_string()), "{status}");
                    assert!(diagnostics.contains(&format!("code {code}")), "{diagnostics}");
                }
                other => panic!("{code}: {other}"),
            }
        }
        // Exit 0 must come with the matching Active lines.
        assert!(matches!(
            run("exit 0".into()).unwrap_err(),
            SnipV2Error::OutcomeMissing { command: "resume", .. }
        ));
        let other_root = format!("0x{}", "cd".repeat(32));
        assert!(matches!(
            run(format!("echo 'merkle_root: {other_root}'; echo 'lifecycle: Active'")).unwrap_err(),
            SnipV2Error::RootMismatch { .. }
        ));
        assert!(matches!(
            run(format!("echo 'merkle_root: {hex}'; echo 'lifecycle: Pending'")).unwrap_err(),
            SnipV2Error::ResumeFailed { .. }
        ));
        // Killed by a signal.
        assert!(matches!(
            run("kill -9 $$".into()).unwrap_err(),
            SnipV2Error::ResumeFailed { .. }
        ));
    }
}
