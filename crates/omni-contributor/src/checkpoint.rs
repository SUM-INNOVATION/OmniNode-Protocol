//! Issue #149 — durable run checkpoints, so a restarted `run-job` resumes
//! instead of starting again.
//!
//! Without one, every attempt re-invokes the runner (which need not be
//! deterministic), publishes from a temporary copy of the response, and
//! writes the signed result only at the very end. A crash after the
//! response was registered would then make the next attempt compute — and
//! possibly publish and account for — a different response.
//!
//! A checkpoint is written **before the first registration** and records
//! everything the signed result needs, so a restart never runs the job
//! again:
//!
//! ```text
//! <store>/<job_id>/<contributor_pubkey_hex>/
//!     checkpoint.json            RunCheckpoint (below)
//!     response-<hash16>.bin      the original response bytes
//! ```
//!
//! [`RunCheckpoint`] holds the job identity (`job_id`, `job_hash`), the
//! contributor identity (`contributor_pubkey_hex`), the response identity
//! (file name, BLAKE3 hash, size, chunk count, canonical SNIP root), the
//! measured accounting, the result fields fixed at preparation
//! (`produced_at_utc`, `notes`, `job_snip_root`), and the phase. It holds no
//! key, seed or other credential.
//!
//! Phases only move forward:
//!
//! | Phase | Meaning | What a restart does |
//! |---|---|---|
//! | `prepared` | runner output validated and durable; nothing sent | publish from the stored response |
//! | `publishing` | a publication attempt started; the chain may hold a Pending registration | publish from the stored response (the chain record decides: ingest, resume or reuse) |
//! | `active` | the chain records the response Active, matching, fetched back and checked | sign the result from the checkpoint and write it; no SNIP call |
//! | `result_written` | the signed result is durably written | nothing is recomputed; the same result is returned (and rewritten byte for byte if its file is missing) |
//!
//! The signed result is a pure function of the checkpoint and the
//! contributor key (Ed25519 signatures are deterministic), so every attempt
//! produces the same result bytes: one response, one registration, one
//! accounting per (job, contributor).
//!
//! Every file is written to a temporary name in its directory, flushed with
//! `fsync`, renamed into place, and the directory is `fsync`ed. The first
//! `checkpoint.json` is created with a hard link, which fails if another
//! process prepared the same job first; the later preparer then adopts the
//! earlier checkpoint and discards its own runner output.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::canonical::hex_lower;
use crate::result::MeasuredAccounting;

/// `RunCheckpoint.schema`.
pub const RUN_CHECKPOINT_SCHEMA: &str = "omni-contributor/run-checkpoint/v1";
/// Directory name of the default store, beside a run's result file.
pub const DEFAULT_CHECKPOINT_DIR: &str = ".omni-run-job";
const CHECKPOINT_FILE: &str = "checkpoint.json";

/// Where a run is. Ordered: a checkpoint only moves forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunPhase {
    Prepared,
    Publishing,
    Active,
    ResultWritten,
}

impl RunPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prepared => "prepared",
            Self::Publishing => "publishing",
            Self::Active => "active",
            Self::ResultWritten => "result_written",
        }
    }
}

impl std::fmt::Display for RunPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How the response became Active, as recorded when it did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointPublication {
    /// `ingested`, `reused_active` or `resumed`.
    pub action: String,
    /// Owner the chain recorded.
    pub owner: String,
}

/// `checkpoint.json`. No credential of any kind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunCheckpoint {
    pub schema: String,
    pub phase: RunPhase,
    // Job and contributor identity.
    pub job_id: String,
    pub job_hash: String,
    pub contributor_pubkey_hex: String,
    // Response identity.
    /// File name in the checkpoint directory.
    pub response_file: String,
    /// BLAKE3, lowercase hex.
    pub response_hash: String,
    pub response_size_bytes: u64,
    pub response_chunk_count: u64,
    /// Canonical SNIP V2 root, `0x` hex.
    pub response_snip_root: String,
    // What the signed result needs besides the response.
    pub measured_accounting: MeasuredAccounting,
    pub produced_at_utc: String,
    pub notes: Option<String>,
    pub job_snip_root: Option<String>,
    // Progress.
    #[serde(default)]
    pub publication: Option<CheckpointPublication>,
    /// BLAKE3 of the result file's bytes, once written.
    #[serde(default)]
    pub result_hash: Option<String>,
    pub prepared_at_utc: String,
    pub updated_at_utc: String,
}

/// Checkpoint failures. Each one leaves the checkpoint as it was.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("checkpoint io at {}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("checkpoint {} is not a valid run checkpoint: {detail}", path.display())]
    Malformed { path: PathBuf, detail: String },

    #[error("checkpoint {} belongs to a different run: {detail}", path.display())]
    Mismatch { path: PathBuf, detail: String },

    #[error("the response stored in checkpoint {} is damaged: {detail}", path.display())]
    ResponseDamaged { path: PathBuf, detail: String },

    #[error(
        "{} already holds a different result than checkpoint {} produces; \
         refusing to overwrite it",
        out.display(),
        checkpoint.display()
    )]
    ResultConflict { out: PathBuf, checkpoint: PathBuf },

    #[error("not a safe checkpoint path component: {0:?}")]
    BadComponent(String),
}

fn io_err(path: &Path) -> impl FnOnce(io::Error) -> CheckpointError + '_ {
    move |source| CheckpointError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// A checkpoint as read from disk, with its directory.
#[derive(Debug, Clone)]
pub struct LoadedCheckpoint {
    pub dir: PathBuf,
    pub checkpoint: RunCheckpoint,
}

impl LoadedCheckpoint {
    pub fn path(&self) -> PathBuf {
        self.dir.join(CHECKPOINT_FILE)
    }

    pub fn response_path(&self) -> PathBuf {
        self.dir.join(&self.checkpoint.response_file)
    }

    /// The stored response, after checking it against the checkpoint's
    /// hash, size, chunk count and root.
    pub fn verified_response(&self) -> Result<Vec<u8>, CheckpointError> {
        let path = self.response_path();
        let bytes = fs::read(&path).map_err(|e| CheckpointError::ResponseDamaged {
            path: self.path(),
            detail: format!("read {}: {e}", path.display()),
        })?;
        let c = &self.checkpoint;
        let expected = omni_store::snip_v2_publish::ExpectedPublicObject::from_bytes(&bytes);
        let hash = hex_lower(blake3::hash(&bytes).as_bytes());
        let problems: Vec<String> = [
            (hash != c.response_hash).then(|| format!("hash {hash} != {}", c.response_hash)),
            (expected.size_bytes != c.response_size_bytes)
                .then(|| format!("size {} != {}", expected.size_bytes, c.response_size_bytes)),
            (expected.chunk_count != c.response_chunk_count).then(|| {
                format!(
                    "chunks {} != {}",
                    expected.chunk_count, c.response_chunk_count
                )
            }),
            (expected.root.to_hex() != c.response_snip_root)
                .then(|| format!("root {} != {}", expected.root, c.response_snip_root)),
        ]
        .into_iter()
        .flatten()
        .collect();
        if problems.is_empty() {
            Ok(bytes)
        } else {
            Err(CheckpointError::ResponseDamaged {
                path: self.path(),
                detail: problems.join("; "),
            })
        }
    }
}

/// What [`RunCheckpointStore::prepare`] records.
pub struct Preparation<'a> {
    pub job_id: &'a str,
    pub job_hash: &'a str,
    pub contributor_pubkey_hex: &'a str,
    pub response: &'a [u8],
    pub measured_accounting: &'a MeasuredAccounting,
    pub produced_at_utc: &'a str,
    pub notes: Option<&'a str>,
    pub job_snip_root: Option<&'a str>,
}

/// A directory of run checkpoints, one per (job, contributor).
#[derive(Debug, Clone)]
pub struct RunCheckpointStore {
    root: PathBuf,
}

impl RunCheckpointStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The default store for a run whose result goes to `out`:
    /// `<directory of out>/.omni-run-job`.
    pub fn beside(out: &Path) -> Self {
        let parent = out
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        Self::new(parent.join(DEFAULT_CHECKPOINT_DIR))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `<root>/<job_id>/<contributor_pubkey_hex>`.
    pub fn dir_for(
        &self,
        job_id: &str,
        contributor_pubkey_hex: &str,
    ) -> Result<PathBuf, CheckpointError> {
        Ok(self
            .root
            .join(safe_component(job_id)?)
            .join(safe_component(contributor_pubkey_hex)?))
    }

    /// The checkpoint of (job, contributor), if one exists.
    pub fn load(
        &self,
        job_id: &str,
        contributor_pubkey_hex: &str,
    ) -> Result<Option<LoadedCheckpoint>, CheckpointError> {
        let dir = self.dir_for(job_id, contributor_pubkey_hex)?;
        let path = dir.join(CHECKPOINT_FILE);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io_err(&path)(e)),
        };
        let checkpoint: RunCheckpoint =
            serde_json::from_slice(&bytes).map_err(|e| CheckpointError::Malformed {
                path: path.clone(),
                detail: e.to_string(),
            })?;
        if checkpoint.schema != RUN_CHECKPOINT_SCHEMA {
            return Err(CheckpointError::Malformed {
                path,
                detail: format!(
                    "schema {:?}, expected {RUN_CHECKPOINT_SCHEMA:?}",
                    checkpoint.schema
                ),
            });
        }
        if checkpoint.job_id != job_id
            || checkpoint.contributor_pubkey_hex != contributor_pubkey_hex
        {
            return Err(CheckpointError::Mismatch {
                path,
                detail: format!(
                    "records job {} / contributor {}",
                    checkpoint.job_id, checkpoint.contributor_pubkey_hex
                ),
            });
        }
        Ok(Some(LoadedCheckpoint { dir, checkpoint }))
    }

    /// Durably record a validated runner output, before anything is
    /// registered. If another process prepared the same (job, contributor)
    /// first, its checkpoint is returned and `p.response` is discarded.
    pub fn prepare(&self, p: &Preparation<'_>) -> Result<LoadedCheckpoint, CheckpointError> {
        let dir = self.dir_for(p.job_id, p.contributor_pubkey_hex)?;
        create_dir_durable(&dir)?;

        let hash = hex_lower(blake3::hash(p.response).as_bytes());
        let expected = omni_store::snip_v2_publish::ExpectedPublicObject::from_bytes(p.response);
        let response_file = format!("response-{}.bin", &hash[..16]);
        // Content-named: replacing a file of the same name changes nothing.
        write_durable(&dir, &response_file, p.response, Create::Replace)?;

        let now = now_utc();
        let checkpoint = RunCheckpoint {
            schema: RUN_CHECKPOINT_SCHEMA.into(),
            phase: RunPhase::Prepared,
            job_id: p.job_id.into(),
            job_hash: p.job_hash.into(),
            contributor_pubkey_hex: p.contributor_pubkey_hex.into(),
            response_file: response_file.clone(),
            response_hash: hash,
            response_size_bytes: expected.size_bytes,
            response_chunk_count: expected.chunk_count,
            response_snip_root: expected.root.to_hex(),
            measured_accounting: p.measured_accounting.clone(),
            produced_at_utc: p.produced_at_utc.into(),
            notes: p.notes.map(str::to_string),
            job_snip_root: p.job_snip_root.map(str::to_string),
            publication: None,
            result_hash: None,
            prepared_at_utc: now.clone(),
            updated_at_utc: now,
        };
        let json =
            serde_json::to_vec_pretty(&checkpoint).map_err(|e| CheckpointError::Malformed {
                path: dir.join(CHECKPOINT_FILE),
                detail: e.to_string(),
            })?;
        if write_durable(&dir, CHECKPOINT_FILE, &json, Create::New)? {
            return Ok(LoadedCheckpoint { dir, checkpoint });
        }
        // Lost the race: adopt the earlier preparation.
        let earlier = self
            .load(p.job_id, p.contributor_pubkey_hex)?
            .ok_or_else(|| CheckpointError::Malformed {
                path: dir.join(CHECKPOINT_FILE),
                detail: "exists but cannot be read".into(),
            })?;
        if earlier.checkpoint.response_file != response_file {
            let _ = fs::remove_file(dir.join(&response_file));
        }
        Ok(earlier)
    }

    /// Move `loaded` forward to `phase`, recording `publication` /
    /// `result_hash` when given. Never moves a checkpoint back.
    pub fn advance(
        &self,
        loaded: &mut LoadedCheckpoint,
        phase: RunPhase,
        publication: Option<CheckpointPublication>,
        result_hash: Option<String>,
    ) -> Result<(), CheckpointError> {
        let c = &mut loaded.checkpoint;
        if phase > c.phase {
            c.phase = phase;
        }
        if publication.is_some() {
            c.publication = publication;
        }
        if result_hash.is_some() {
            c.result_hash = result_hash;
        }
        c.updated_at_utc = now_utc();
        let json = serde_json::to_vec_pretty(c).map_err(|e| CheckpointError::Malformed {
            path: loaded.dir.join(CHECKPOINT_FILE),
            detail: e.to_string(),
        })?;
        write_durable(&loaded.dir, CHECKPOINT_FILE, &json, Create::Replace)?;
        Ok(())
    }

    /// Durably write the signed result `result_json` to `out` and mark the
    /// checkpoint `result_written`. Idempotent: an `out` that already holds
    /// exactly these bytes is left as it is; one holding other bytes is
    /// refused.
    pub fn write_result(
        &self,
        loaded: &mut LoadedCheckpoint,
        out: &Path,
        result_json: &[u8],
    ) -> Result<(), CheckpointError> {
        match fs::read(out) {
            Ok(existing) if existing == result_json => {}
            Ok(_) => {
                return Err(CheckpointError::ResultConflict {
                    out: out.to_path_buf(),
                    checkpoint: loaded.path(),
                });
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                let dir = out
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                let name = out
                    .file_name()
                    .ok_or_else(|| CheckpointError::BadComponent(out.display().to_string()))?
                    .to_string_lossy()
                    .into_owned();
                write_durable(dir, &name, result_json, Create::Replace)?;
            }
            Err(e) => return Err(io_err(out)(e)),
        }
        let hash = hex_lower(blake3::hash(result_json).as_bytes());
        self.advance(loaded, RunPhase::ResultWritten, None, Some(hash))
    }
}

fn now_utc() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Job ids and public keys are hex; anything else never becomes a path.
fn safe_component(s: &str) -> Result<&str, CheckpointError> {
    if !s.is_empty() && s.len() <= 128 && s.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(s)
    } else {
        Err(CheckpointError::BadComponent(s.to_string()))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Create {
    /// Fail (return `false`) if the file exists.
    New,
    /// Atomically replace any existing file.
    Replace,
}

/// Write `bytes` to `dir/name` durably: temporary file in `dir`, `fsync`,
/// rename (or hard link for [`Create::New`]), `fsync` of `dir`. Returns
/// `false` only for [`Create::New`] when the file already existed.
fn write_durable(
    dir: &Path,
    name: &str,
    bytes: &[u8],
    create: Create,
) -> Result<bool, CheckpointError> {
    let target = dir.join(name);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = dir.join(format!(".{name}.{}.{nanos}.tmp", std::process::id()));
    let result = (|| {
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(io_err(&tmp))?;
        f.write_all(bytes).map_err(io_err(&tmp))?;
        f.sync_all().map_err(io_err(&tmp))?;
        drop(f);
        match create {
            Create::Replace => fs::rename(&tmp, &target).map_err(io_err(&target))?,
            Create::New => match fs::hard_link(&tmp, &target) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => return Ok(false),
                Err(e) => return Err(io_err(&target)(e)),
            },
        }
        sync_dir(dir)?;
        Ok(true)
    })();
    // Gone after a rename; the link source after a hard link; the partial
    // file after a failure.
    let _ = fs::remove_file(&tmp);
    result
}

fn sync_dir(dir: &Path) -> Result<(), CheckpointError> {
    File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(io_err(dir))
}

/// `create_dir_all`, then `fsync` the directory and its parents up to the
/// store's parent, so the new directory entries survive a crash too.
fn create_dir_durable(dir: &Path) -> Result<(), CheckpointError> {
    fs::create_dir_all(dir).map_err(io_err(dir))?;
    let mut d = Some(dir);
    for _ in 0..4 {
        let Some(cur) = d else { break };
        sync_dir(cur)?;
        d = cur.parent().filter(|p| !p.as_os_str().is_empty());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{StageContribution, WorkUnitKind};

    fn accounting() -> MeasuredAccounting {
        MeasuredAccounting {
            tokenizer_hash: "ab".repeat(32),
            input_token_count: 16,
            output_token_count: 8,
            total_base_units: 24,
            stage_contributions: vec![StageContribution {
                contributor_pubkey_hex: "cd".repeat(32),
                stage_label: "s".into(),
                work_unit_kind: WorkUnitKind::Tokens,
                work_units: 24,
            }],
        }
    }

    fn prep<'a>(response: &'a [u8], acct: &'a MeasuredAccounting) -> Preparation<'a> {
        Preparation {
            job_id: "0a1b",
            job_hash: "0a1b",
            contributor_pubkey_hex: "ff00",
            response,
            measured_accounting: acct,
            produced_at_utc: "2026-10-28T00:00:00Z",
            notes: Some("n"),
            job_snip_root: None,
        }
    }

    #[test]
    fn prepare_then_load_round_trips_and_checks_the_response() {
        let dir = tempfile::tempdir().unwrap();
        let store = RunCheckpointStore::new(dir.path().join("cp"));
        let acct = accounting();
        let loaded = store.prepare(&prep(b"response bytes", &acct)).unwrap();
        assert_eq!(loaded.checkpoint.phase, RunPhase::Prepared);
        let again = store.load("0a1b", "ff00").unwrap().unwrap();
        assert_eq!(again.checkpoint, loaded.checkpoint);
        assert_eq!(again.verified_response().unwrap(), b"response bytes");
        assert!(store.load("0a1c", "ff00").unwrap().is_none());
        // No temporary file is left behind.
        let leftovers: Vec<_> = fs::read_dir(&loaded.dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn the_first_preparation_wins() {
        let dir = tempfile::tempdir().unwrap();
        let store = RunCheckpointStore::new(dir.path());
        let acct = accounting();
        let first = store.prepare(&prep(b"first", &acct)).unwrap();
        let second = store.prepare(&prep(b"second, different", &acct)).unwrap();
        assert_eq!(second.checkpoint, first.checkpoint);
        assert_eq!(second.verified_response().unwrap(), b"first");
        // The loser's response file was removed.
        let bins = fs::read_dir(&first.dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".bin")
            })
            .count();
        assert_eq!(bins, 1);
    }

    #[test]
    fn phases_only_move_forward() {
        let dir = tempfile::tempdir().unwrap();
        let store = RunCheckpointStore::new(dir.path());
        let acct = accounting();
        let mut cp = store.prepare(&prep(b"x", &acct)).unwrap();
        store
            .advance(&mut cp, RunPhase::Active, None, None)
            .unwrap();
        store
            .advance(&mut cp, RunPhase::Publishing, None, None)
            .unwrap();
        assert_eq!(
            store
                .load("0a1b", "ff00")
                .unwrap()
                .unwrap()
                .checkpoint
                .phase,
            RunPhase::Active
        );
    }

    #[test]
    fn a_damaged_response_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let store = RunCheckpointStore::new(dir.path());
        let acct = accounting();
        let cp = store.prepare(&prep(b"original", &acct)).unwrap();
        fs::write(cp.response_path(), b"tampered").unwrap();
        let err = cp.verified_response().unwrap_err();
        assert!(
            matches!(err, CheckpointError::ResponseDamaged { .. }),
            "{err}"
        );
    }

    #[test]
    fn the_result_write_is_idempotent_and_refuses_other_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let store = RunCheckpointStore::new(dir.path().join("cp"));
        let acct = accounting();
        let mut cp = store.prepare(&prep(b"x", &acct)).unwrap();
        let out = dir.path().join("result.json");
        store.write_result(&mut cp, &out, b"{\"r\":1}").unwrap();
        assert_eq!(cp.checkpoint.phase, RunPhase::ResultWritten);
        store.write_result(&mut cp, &out, b"{\"r\":1}").unwrap();
        let err = store.write_result(&mut cp, &out, b"{\"r\":2}").unwrap_err();
        assert!(
            matches!(err, CheckpointError::ResultConflict { .. }),
            "{err}"
        );
        assert_eq!(fs::read(&out).unwrap(), b"{\"r\":1}");
    }

    #[test]
    fn unsafe_components_never_become_paths() {
        let store = RunCheckpointStore::new("/tmp/x");
        for bad in ["", "../etc", "a/b", "ab\u{0}", "zz"] {
            assert!(store.dir_for(bad, "ff").is_err(), "{bad:?}");
            assert!(store.dir_for("ff", bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_checkpoint_holds_no_credential_fields() {
        let dir = tempfile::tempdir().unwrap();
        let store = RunCheckpointStore::new(dir.path());
        let acct = accounting();
        let cp = store.prepare(&prep(b"x", &acct)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&fs::read(cp.path()).unwrap()).unwrap();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        for k in &keys {
            assert!(
                !["seed", "secret", "private", "key_file", "signature"]
                    .iter()
                    .any(|bad| k.contains(bad)),
                "{k}"
            );
        }
    }

    #[test]
    fn beside_uses_the_result_directory() {
        assert_eq!(
            RunCheckpointStore::beside(Path::new("/a/b/result.json")).root(),
            Path::new("/a/b/.omni-run-job")
        );
        assert_eq!(
            RunCheckpointStore::beside(Path::new("result.json")).root(),
            Path::new("./.omni-run-job")
        );
    }
}
