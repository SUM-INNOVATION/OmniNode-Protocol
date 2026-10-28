//! Issue #149 — idempotent, retry-safe Public V2 publication.
//!
//! `sum-node ingest-v2` registers a file's root on chain
//! (`RegisterFilePendingV2`) before it pushes any chunk, and the chain refuses
//! to register a root it already holds. Running `ingest-v2` again is therefore
//! not a retry: it fails for every root the chain already has, whether the
//! same bytes were published by an earlier run or by another job, or a run
//! was interrupted between registration and activation and left the root
//! `Pending`.
//!
//! [`publish_public_idempotent`] decides from the chain's record of the root
//! (`storage_getFileInfoV2`), never from `sum-node`'s stdout alone. The root is
//! the canonical SNIP V2 identity of the bytes ([`public_object_root`]);
//! nothing is added to the bytes or the root to get past a registered root.
//!
//! | Chain record of the expected root | Action | Outcome |
//! |---|---|---|
//! | the lookup fails | none | refused; nothing registered |
//! | none | `sum-node ingest-v2` | [`PublicationAction::Ingested`] once the record reads Active |
//! | root, visibility, size or chunk count differs | none | refused: [`SnipV2Error::ChainRecordMismatch`] |
//! | Active | none (reuse) | [`PublicationAction::ReusedActive`] |
//! | Pending, owned by the configured account | `sum-node resume <root> <file>` | [`PublicationAction::Resumed`] once the record reads Active |
//! | Pending, owned by another account | none | refused: [`SnipV2Error::PendingOwnedByOther`] |
//! | Abandoned | none | refused: [`SnipV2Error::AbandonedOnChain`] |
//!
//! After `ingest-v2` or `resume`, whatever it reported, the record is read
//! again and the same table applies to it. That covers a registration that
//! landed although `sum-node` could not confirm it, and another publisher of
//! the same bytes registering between the first read and ours. `ingest-v2`
//! runs only while the chain has no record of the root and `resume` only for
//! the configured account's own Pending record, each at most once per call:
//! the root is never registered twice and a failed registration is never
//! retried blindly.
//!
//! Success is only a record that reads `Active` and matches the bytes. A root
//! left Pending or Abandoned, an RPC failure and a root mismatch are errors
//! that carry what `sum-node` and the chain reported.

use std::io::Read;
use std::path::Path;

use omni_types::phase5::SnipV2ObjectId;
use serde::Deserialize;
use sumchain_wire::Hash;
use sumchain_wire::storage_metadata::CHUNK_SIZE;

use crate::snip_v2::{SnipV2Adapter, SnipV2Error};

// ── Canonical Public V2 identity ──────────────────────────────────────────────

/// SNIP V2 chunk size in bytes: the chain's `storage_metadata::CHUNK_SIZE`
/// (1 MiB), which SNIP's chunker uses.
pub const SNIP_V2_CHUNK_SIZE: u64 = CHUNK_SIZE;

/// `StorageFileInfoV2.visibility` of a Public file.
pub const VISIBILITY_PUBLIC: u8 = 0;

/// The Public V2 root of `bytes`: the chain's `Hash::merkle_root` over the
/// BLAKE3 hash of each [`SNIP_V2_CHUNK_SIZE`] chunk, the last one short.
/// This is what SNIP's `BinaryChunker::chunk_file` and `MerkleTree` compute
/// (an odd node is paired with itself; a pair hashes as
/// `blake3(left || right)`; one chunk is its own root; no bytes give the zero
/// root), so it equals the `merkle_root:` `sum-node ingest-v2` reports for a
/// Public file holding exactly these bytes.
pub fn public_object_root(bytes: &[u8]) -> SnipV2ObjectId {
    let leaves: Vec<Hash> = bytes
        .chunks(SNIP_V2_CHUNK_SIZE as usize)
        .map(Hash::hash)
        .collect();
    SnipV2ObjectId::from_bytes(*Hash::merkle_root(&leaves).as_bytes())
}

/// Number of Public V2 chunks of a `len`-byte file.
pub fn public_chunk_count(len: u64) -> u64 {
    len.div_ceil(SNIP_V2_CHUNK_SIZE)
}

/// What the chain must record for a Public file holding the bytes being
/// published.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedPublicObject {
    pub root: SnipV2ObjectId,
    /// `plaintext_size_bytes`, and `stored_size_bytes` (equal for Public).
    pub size_bytes: u64,
    pub chunk_count: u64,
}

impl ExpectedPublicObject {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            root: public_object_root(bytes),
            size_bytes: bytes.len() as u64,
            chunk_count: public_chunk_count(bytes.len() as u64),
        }
    }

    /// Same as [`Self::from_bytes`] over the file's contents, read one chunk
    /// at a time.
    pub fn from_file(path: &Path) -> Result<Self, SnipV2Error> {
        if !path.is_file() {
            return Err(SnipV2Error::InputNotFound {
                path: path.to_path_buf(),
            });
        }
        let mut file = std::fs::File::open(path)?;
        let mut leaves = Vec::new();
        let mut size_bytes = 0u64;
        let mut buf = vec![0u8; SNIP_V2_CHUNK_SIZE as usize];
        loop {
            let mut filled = 0;
            while filled < buf.len() {
                let n = file.read(&mut buf[filled..])?;
                if n == 0 {
                    break;
                }
                filled += n;
            }
            if filled == 0 {
                break;
            }
            leaves.push(Hash::hash(&buf[..filled]));
            size_bytes += filled as u64;
            if filled < buf.len() {
                break;
            }
        }
        Ok(Self {
            root: SnipV2ObjectId::from_bytes(*Hash::merkle_root(&leaves).as_bytes()),
            size_bytes,
            chunk_count: leaves.len() as u64,
        })
    }
}

// ── The chain's record ────────────────────────────────────────────────────────

/// Lifecycle of a V2 file as the chain records it
/// (`StorageFileInfoV2.lifecycle`: 0 Pending, 1 Active, 2 Abandoned).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnipV2ChainLifecycle {
    Pending,
    Active,
    Abandoned,
}

impl SnipV2ChainLifecycle {
    pub fn from_wire(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Pending),
            1 => Some(Self::Active),
            2 => Some(Self::Abandoned),
            _ => None,
        }
    }
}

/// The fields of the chain's `storage_getFileInfoV2` answer that publication
/// depends on (sum-chain `StorageFileInfoV2`; other fields are ignored).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SnipV2FileInfo {
    /// `0x`-prefixed lowercase hex.
    pub merkle_root: String,
    /// Registering account, base58.
    pub owner: String,
    pub plaintext_size_bytes: u64,
    pub stored_size_bytes: u64,
    pub chunk_count: u64,
    /// Height `RegisterFilePendingV2` finalized at.
    pub created_at: u64,
    pub activated_at_height: Option<u64>,
    #[serde(default)]
    pub abandoned_at_height: Option<u64>,
    pub visibility: u8,
    pub lifecycle: u8,
}

/// Outcome of `sum-node resume <root> <file>`, by its exit status
/// (`0` Active, `3` Pending, `4` Abandoned, `5` root mismatch). Exit `1`
/// (error) and `2` (usage) are [`SnipV2Error::ResumeFailed`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnipV2ResumeOutcome {
    Active,
    Pending { diagnostics: String },
    Abandoned { diagnostics: String },
    RootMismatch { diagnostics: String },
}

/// The publication surface on top of [`SnipV2Adapter`]: the chain's record
/// of a root, the account publication signs with, and `sum-node resume`.
pub trait SnipV2Publisher: SnipV2Adapter {
    /// The chain's `storage_getFileInfoV2` record of `root`. `Ok(None)` when
    /// the chain has none; an unreachable RPC or an error answer is `Err`.
    fn file_info_v2(&self, root: &SnipV2ObjectId) -> Result<Option<SnipV2FileInfo>, SnipV2Error>;

    /// The account (base58) [`SnipV2Adapter::ingest_public`] and
    /// [`Self::resume_public`] sign with: the owner of what they register.
    fn publisher_account(&self) -> Result<String, SnipV2Error>;

    /// `sum-node resume <root> <path>`: the post-registration stages
    /// (push, manifest push, coverage, activation) for a Pending root. Never
    /// registers.
    fn resume_public(
        &self,
        root: &SnipV2ObjectId,
        path: &Path,
    ) -> Result<SnipV2ResumeOutcome, SnipV2Error>;
}

// ── Idempotent publication ────────────────────────────────────────────────────

/// How a published object came to be Active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicationAction {
    /// The chain had no record; this call's `ingest-v2` registered it and
    /// reported success.
    Ingested,
    /// The chain holds it Active although this call neither ingested it
    /// successfully nor resumed it: published earlier (by this account or
    /// another), or registered by another publisher during this call.
    ReusedActive,
    /// The chain held it Pending, owned by the configured account; this
    /// call resumed and activated it.
    Resumed,
}

impl PublicationAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ingested => "ingested",
            Self::ReusedActive => "reused_active",
            Self::Resumed => "resumed",
        }
    }
}

impl std::fmt::Display for PublicationAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A Public object the chain records Active, matching the published bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedPublicObject {
    pub root: SnipV2ObjectId,
    pub action: PublicationAction,
    /// Owner the chain records (not necessarily the configured account for
    /// a reused Active object).
    pub owner: String,
    pub activated_at_height: u64,
}

/// Publish the file at `path` as a Public V2 object, or confirm it is
/// already, following the decision table in the [module docs](self).
/// Returns only once the chain records the file's root `Active` with the
/// file's size, chunk count and Public visibility.
pub fn publish_public_idempotent<A: SnipV2Publisher + ?Sized>(
    adapter: &A,
    path: &Path,
) -> Result<PublishedPublicObject, SnipV2Error> {
    let expected = ExpectedPublicObject::from_file(path)?;
    let root = expected.root.to_hex();
    let mut ingested = false;
    let mut ingest_reported_ok = false;
    let mut resumed = false;
    // What the last `ingest-v2` / `resume` reported, kept for the error when
    // the record that follows it is not Active.
    let mut last_attempt: Option<String> = None;

    loop {
        let info = match adapter.file_info_v2(&expected.root) {
            Ok(info) => info,
            Err(e) => {
                return Err(match last_attempt {
                    None => e,
                    Some(attempt) => SnipV2Error::PublicationUnconfirmed {
                        root,
                        attempt,
                        recheck: e.to_string(),
                    },
                });
            }
        };

        let Some(info) = info else {
            if ingested {
                return Err(SnipV2Error::NotRegistered {
                    root,
                    attempt: last_attempt.unwrap_or_default(),
                });
            }
            tracing::info!(%root, "SNIP V2 publication: no chain record of the root; ingesting");
            ingested = true;
            last_attempt = Some(match adapter.ingest_public(path) {
                Ok(obj) if obj.merkle_root != expected.root => {
                    return Err(SnipV2Error::RootMismatch {
                        expected: root,
                        reported: obj.merkle_root.to_hex(),
                        detail: "sum-node ingest-v2 registered a different root for these bytes"
                            .into(),
                    });
                }
                Ok(obj) => {
                    ingest_reported_ok = true;
                    format!("sum-node ingest-v2 reported lifecycle {}", obj.lifecycle)
                }
                Err(e) => format!("sum-node ingest-v2 failed: {e}"),
            });
            continue;
        };

        let lifecycle = match check_record(&expected, &info) {
            Ok(lc) => lc,
            Err(e) => {
                if let Some(attempt) = &last_attempt {
                    tracing::warn!(%root, %attempt, "SNIP V2 publication: refused after an attempt");
                }
                return Err(e);
            }
        };
        match lifecycle {
            SnipV2ChainLifecycle::Active => {
                let action = if resumed {
                    PublicationAction::Resumed
                } else if ingest_reported_ok {
                    PublicationAction::Ingested
                } else {
                    PublicationAction::ReusedActive
                };
                tracing::info!(
                    %root,
                    %action,
                    owner = %info.owner,
                    "SNIP V2 publication: chain record is Active and matches the bytes"
                );
                return Ok(PublishedPublicObject {
                    root: expected.root,
                    action,
                    owner: info.owner,
                    // `check_record` refuses Active without it.
                    activated_at_height: info.activated_at_height.unwrap_or_default(),
                });
            }
            SnipV2ChainLifecycle::Abandoned => {
                return Err(SnipV2Error::AbandonedOnChain {
                    root,
                    owner: info.owner,
                    detail: last_attempt.unwrap_or_else(|| "no attempt in this run".into()),
                });
            }
            SnipV2ChainLifecycle::Pending => {
                let account = adapter.publisher_account()?;
                if info.owner != account {
                    return Err(SnipV2Error::PendingOwnedByOther {
                        root,
                        owner: info.owner,
                        account,
                    });
                }
                if resumed {
                    return Err(SnipV2Error::StillPending {
                        root,
                        detail: last_attempt.unwrap_or_default(),
                    });
                }
                tracing::info!(
                    %root,
                    created_at = info.created_at,
                    "SNIP V2 publication: Pending registration of this account; resuming"
                );
                resumed = true;
                last_attempt = Some(match adapter.resume_public(&expected.root, path) {
                    Ok(SnipV2ResumeOutcome::RootMismatch { diagnostics }) => {
                        return Err(SnipV2Error::RootMismatch {
                            expected: root,
                            reported: "(sum-node resume exit 5)".into(),
                            detail: diagnostics,
                        });
                    }
                    Ok(SnipV2ResumeOutcome::Active) => {
                        "sum-node resume reported lifecycle Active".to_string()
                    }
                    Ok(SnipV2ResumeOutcome::Pending { diagnostics }) => {
                        format!("sum-node resume left the root Pending (exit 3): {diagnostics}")
                    }
                    Ok(SnipV2ResumeOutcome::Abandoned { diagnostics }) => {
                        format!("sum-node resume found the root Abandoned (exit 4): {diagnostics}")
                    }
                    Err(e) => format!("sum-node resume failed: {e}"),
                });
            }
        }
    }
}

/// Refuse a record that is not the one these bytes would produce, and return
/// its lifecycle.
fn check_record(
    expected: &ExpectedPublicObject,
    info: &SnipV2FileInfo,
) -> Result<SnipV2ChainLifecycle, SnipV2Error> {
    let root = expected.root.to_hex();
    let mismatch =
        |field: &'static str, chain: String, want: String| SnipV2Error::ChainRecordMismatch {
            root: root.clone(),
            field,
            chain,
            expected: want,
        };

    if SnipV2ObjectId::from_hex(&info.merkle_root).ok() != Some(expected.root) {
        return Err(mismatch(
            "merkle_root",
            info.merkle_root.clone(),
            root.clone(),
        ));
    }
    if info.visibility != VISIBILITY_PUBLIC {
        return Err(mismatch(
            "visibility",
            info.visibility.to_string(),
            format!("{VISIBILITY_PUBLIC} (Public)"),
        ));
    }
    if info.plaintext_size_bytes != expected.size_bytes {
        return Err(mismatch(
            "plaintext_size_bytes",
            info.plaintext_size_bytes.to_string(),
            expected.size_bytes.to_string(),
        ));
    }
    if info.stored_size_bytes != expected.size_bytes {
        return Err(mismatch(
            "stored_size_bytes",
            info.stored_size_bytes.to_string(),
            expected.size_bytes.to_string(),
        ));
    }
    if info.chunk_count != expected.chunk_count {
        return Err(mismatch(
            "chunk_count",
            info.chunk_count.to_string(),
            expected.chunk_count.to_string(),
        ));
    }
    let lifecycle = SnipV2ChainLifecycle::from_wire(info.lifecycle).ok_or_else(|| {
        SnipV2Error::ChainRecordMalformed {
            root: root.clone(),
            detail: format!("unknown lifecycle {}", info.lifecycle),
        }
    })?;
    if lifecycle == SnipV2ChainLifecycle::Active && info.activated_at_height.is_none() {
        return Err(SnipV2Error::ChainRecordMalformed {
            root,
            detail: "lifecycle is Active but activated_at_height is null".into(),
        });
    }
    Ok(lifecycle)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::io::Write;
    use std::path::PathBuf;

    use omni_types::phase5::{SnipV2Lifecycle, SnipV2ObjectRef};

    use super::*;

    const ME: &str = "ConfiguredAccount111111111111111";
    const OTHER: &str = "SomeoneElse11111111111111111111";

    // ── Root derivation ─────────────────────────────────────────────────────

    /// SNIP's `MerkleTree::build` (sum-store `merkle.rs`), written out
    /// independently of `sumchain_wire::Hash::merkle_root`.
    fn snip_merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
        if leaves.is_empty() {
            return [0u8; 32];
        }
        let mut level = leaves.to_vec();
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|p| {
                    let right = p.get(1).unwrap_or(&p[0]);
                    let mut h = blake3::Hasher::new();
                    h.update(&p[0]);
                    h.update(right);
                    *h.finalize().as_bytes()
                })
                .collect();
        }
        level[0]
    }

    fn snip_root_of(bytes: &[u8]) -> [u8; 32] {
        let leaves: Vec<[u8; 32]> = bytes
            .chunks(1_048_576)
            .map(|c| *blake3::hash(c).as_bytes())
            .collect();
        snip_merkle_root(&leaves)
    }

    fn patterned(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i.wrapping_mul(31) ^ (i >> 7)) as u8)
            .collect()
    }

    #[test]
    fn chunk_size_is_snips_one_mebibyte() {
        assert_eq!(SNIP_V2_CHUNK_SIZE, 1_048_576);
    }

    #[test]
    fn root_matches_snip_merkle_tree_across_chunk_boundaries() {
        const C: usize = 1_048_576;
        for len in [0, 1, 16, C - 1, C, C + 1, 2 * C, 3 * C + 5, 5 * C - 1] {
            let bytes = patterned(len);
            let ours = public_object_root(&bytes);
            assert_eq!(*ours.as_bytes(), snip_root_of(&bytes), "len {len}");
            assert_eq!(public_chunk_count(len as u64), len.div_ceil(C) as u64);
        }
    }

    #[test]
    fn one_chunk_root_is_blake3_of_the_bytes() {
        let bytes = b"sixteen bytes!!!";
        assert_eq!(
            *public_object_root(bytes).as_bytes(),
            *blake3::hash(bytes).as_bytes()
        );
    }

    #[test]
    fn no_bytes_give_the_zero_root_and_no_chunks() {
        let e = ExpectedPublicObject::from_bytes(&[]);
        assert_eq!(*e.root.as_bytes(), [0u8; 32]);
        assert_eq!(e.chunk_count, 0);
        assert_eq!(e.size_bytes, 0);
    }

    #[test]
    fn from_file_equals_from_bytes() {
        const C: usize = 1_048_576;
        for len in [0, 7, C, C + 3, 2 * C + 1] {
            let bytes = patterned(len);
            let mut f = tempfile::NamedTempFile::new().unwrap();
            f.write_all(&bytes).unwrap();
            f.flush().unwrap();
            assert_eq!(
                ExpectedPublicObject::from_file(f.path()).unwrap(),
                ExpectedPublicObject::from_bytes(&bytes),
                "len {len}"
            );
        }
    }

    #[test]
    fn from_file_refuses_a_missing_file() {
        let err = ExpectedPublicObject::from_file(Path::new("/nonexistent/omni-149")).unwrap_err();
        assert!(matches!(err, SnipV2Error::InputNotFound { .. }), "{err}");
    }

    #[test]
    fn file_info_parses_the_chain_wire_shape() {
        let v = serde_json::json!({
            "merkle_root": format!("0x{}", "ab".repeat(32)),
            "owner": ME,
            "plaintext_size_bytes": 16,
            "stored_size_bytes": 16,
            "chunk_count": 1,
            "fee_pool": 0,
            "created_at": 10,
            "activated_at_height": null,
            "abandoned_at_height": null,
            "assignment_height": 10,
            "visibility": 0,
            "lifecycle": 0,
            "access_list": [],
            "access_total": 0,
            "access_offset": 0,
            "predecessor_root": null
        });
        let info: SnipV2FileInfo = serde_json::from_value(v).unwrap();
        assert_eq!(info.owner, ME);
        assert_eq!(info.lifecycle, 0);
        assert_eq!(info.activated_at_height, None);
        // `abandoned_at_height` is optional on older chains.
        let mut v = serde_json::to_value(serde_json::json!({
            "merkle_root": format!("0x{}", "ab".repeat(32)),
            "owner": ME, "plaintext_size_bytes": 1, "stored_size_bytes": 1,
            "chunk_count": 1, "created_at": 1, "activated_at_height": 2,
            "visibility": 0, "lifecycle": 1
        }))
        .unwrap();
        let info: SnipV2FileInfo = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(info.abandoned_at_height, None);
        // A missing required field is refused, not defaulted.
        v.as_object_mut().unwrap().remove("owner");
        assert!(serde_json::from_value::<SnipV2FileInfo>(v).is_err());
    }

    // ── Scripted publisher ──────────────────────────────────────────────────

    enum Ingest {
        /// Register, then leave the record in this lifecycle; report `report`.
        Register {
            lifecycle: u8,
            report: Result<SnipV2Lifecycle, String>,
        },
        /// Register nothing; fail.
        Fail(String),
        /// Report a different root.
        WrongRoot,
    }

    enum Resume {
        /// Set the record to this lifecycle and report `outcome`.
        To {
            lifecycle: u8,
            outcome: Result<SnipV2ResumeOutcome, String>,
        },
    }

    #[derive(Default)]
    struct Calls {
        lookups: usize,
        ingests: usize,
        resumes: usize,
        account: usize,
    }

    struct Scripted {
        record: RefCell<Option<SnipV2FileInfo>>,
        /// Lookups fail while this is > 0 (decremented per lookup), or from
        /// lookup number `fail_from` on.
        fail_lookups: RefCell<usize>,
        fail_from: Option<usize>,
        /// Another publisher registers `record_on_race` right before our
        /// ingest runs.
        record_on_race: RefCell<Option<SnipV2FileInfo>>,
        ingest: RefCell<VecDeque<Ingest>>,
        resume: RefCell<VecDeque<Resume>>,
        account: Result<String, String>,
        expected: ExpectedPublicObject,
        calls: RefCell<Calls>,
    }

    impl Scripted {
        fn new(expected: &ExpectedPublicObject) -> Self {
            Self {
                record: RefCell::new(None),
                fail_lookups: RefCell::new(0),
                fail_from: None,
                record_on_race: RefCell::new(None),
                ingest: RefCell::new(VecDeque::new()),
                resume: RefCell::new(VecDeque::new()),
                account: Ok(ME.into()),
                expected: expected.clone(),
                calls: RefCell::new(Calls::default()),
            }
        }

        fn record(&self, owner: &str, lifecycle: u8) -> SnipV2FileInfo {
            SnipV2FileInfo {
                merkle_root: self.expected.root.to_hex(),
                owner: owner.into(),
                plaintext_size_bytes: self.expected.size_bytes,
                stored_size_bytes: self.expected.size_bytes,
                chunk_count: self.expected.chunk_count,
                created_at: 10,
                activated_at_height: (lifecycle == 1).then_some(12),
                abandoned_at_height: (lifecycle == 2).then_some(70),
                visibility: VISIBILITY_PUBLIC,
                lifecycle,
            }
        }

        fn with_record(self, info: SnipV2FileInfo) -> Self {
            *self.record.borrow_mut() = Some(info);
            self
        }
    }

    impl SnipV2Adapter for Scripted {
        fn ingest_public(&self, _path: &Path) -> Result<SnipV2ObjectRef, SnipV2Error> {
            self.calls.borrow_mut().ingests += 1;
            if let Some(raced) = self.record_on_race.borrow_mut().take() {
                *self.record.borrow_mut() = Some(raced);
                return Err(SnipV2Error::OutcomeMissing {
                    command: "ingest-v2",
                    cause: "no `merkle_root:` line".into(),
                    diagnostics: "WARN V2 ingest FAILED before chain registration \
                                  source=RegisterFilePendingV2 failed: merkle_root \
                                  already registered as V2 file"
                        .into(),
                });
            }
            match self
                .ingest
                .borrow_mut()
                .pop_front()
                .expect("unexpected ingest")
            {
                Ingest::Register { lifecycle, report } => {
                    if self.record.borrow().is_some() {
                        panic!("duplicate registration of a registered root");
                    }
                    *self.record.borrow_mut() = Some(self.record(ME, lifecycle));
                    report
                        .map(|lc| SnipV2ObjectRef {
                            merkle_root: self.expected.root,
                            lifecycle: lc,
                            plaintext_size_bytes: None,
                        })
                        .map_err(|stderr| SnipV2Error::NonZeroExit { code: 1, stderr })
                }
                Ingest::Fail(msg) => Err(SnipV2Error::NonZeroExit {
                    code: 1,
                    stderr: msg,
                }),
                Ingest::WrongRoot => Ok(SnipV2ObjectRef {
                    merkle_root: SnipV2ObjectId::from_bytes([7u8; 32]),
                    lifecycle: SnipV2Lifecycle::Active,
                    plaintext_size_bytes: None,
                }),
            }
        }

        fn download_public(&self, _: &SnipV2ObjectId, _: &Path) -> Result<(), SnipV2Error> {
            unreachable!("publication never downloads")
        }
    }

    impl SnipV2Publisher for Scripted {
        fn file_info_v2(
            &self,
            root: &SnipV2ObjectId,
        ) -> Result<Option<SnipV2FileInfo>, SnipV2Error> {
            assert_eq!(*root, self.expected.root);
            let n = {
                let mut c = self.calls.borrow_mut();
                c.lookups += 1;
                c.lookups
            };
            let mut fail = self.fail_lookups.borrow_mut();
            if *fail > 0 || self.fail_from.is_some_and(|k| n >= k) {
                *fail = fail.saturating_sub(1);
                return Err(SnipV2Error::ChainLookup {
                    url: "http://127.0.0.1:1".into(),
                    root: root.to_hex(),
                    detail: "connection refused".into(),
                });
            }
            Ok(self.record.borrow().clone())
        }

        fn publisher_account(&self) -> Result<String, SnipV2Error> {
            self.calls.borrow_mut().account += 1;
            self.account.clone().map_err(SnipV2Error::PublisherAccount)
        }

        fn resume_public(
            &self,
            root: &SnipV2ObjectId,
            _path: &Path,
        ) -> Result<SnipV2ResumeOutcome, SnipV2Error> {
            assert_eq!(*root, self.expected.root);
            self.calls.borrow_mut().resumes += 1;
            let Resume::To { lifecycle, outcome } = self
                .resume
                .borrow_mut()
                .pop_front()
                .expect("unexpected resume");
            if let Some(r) = self.record.borrow_mut().as_mut() {
                r.lifecycle = lifecycle;
                if lifecycle == 1 {
                    r.activated_at_height = Some(40);
                }
            }
            outcome.map_err(|diagnostics| SnipV2Error::ResumeFailed {
                status: "exit status: 1".into(),
                diagnostics,
            })
        }
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        path: PathBuf,
        expected: ExpectedPublicObject,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("response.bin");
        let bytes = b"sixteen bytes!!!";
        std::fs::write(&path, bytes).unwrap();
        Fixture {
            _dir: dir,
            path,
            expected: ExpectedPublicObject::from_bytes(bytes),
        }
    }

    fn counts(s: &Scripted) -> (usize, usize, usize) {
        let c = s.calls.borrow();
        (c.lookups, c.ingests, c.resumes)
    }

    // ── Decision table ──────────────────────────────────────────────────────

    #[test]
    fn absent_root_is_ingested_and_confirmed_active() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        s.ingest.borrow_mut().push_back(Ingest::Register {
            lifecycle: 1,
            report: Ok(SnipV2Lifecycle::Active),
        });
        let p = publish_public_idempotent(&s, &f.path).unwrap();
        assert_eq!(p.action, PublicationAction::Ingested);
        assert_eq!(p.root, f.expected.root);
        assert_eq!(p.owner, ME);
        assert_eq!(p.activated_at_height, 12);
        assert_eq!(counts(&s), (2, 1, 0), "lookup, ingest, confirming lookup");
    }

    #[test]
    fn active_matching_root_is_reused_without_any_registration() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        let s = {
            let r = s.record(OTHER, 1);
            s.with_record(r)
        };
        let p = publish_public_idempotent(&s, &f.path).unwrap();
        assert_eq!(p.action, PublicationAction::ReusedActive);
        // Another account's Active public object is reused: the root binds
        // the bytes. Its owner is reported.
        assert_eq!(p.owner, OTHER);
        assert_eq!(counts(&s), (1, 0, 0));
        assert_eq!(
            s.calls.borrow().account,
            0,
            "ownership is irrelevant for Active"
        );
    }

    #[test]
    fn running_twice_registers_once() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        s.ingest.borrow_mut().push_back(Ingest::Register {
            lifecycle: 1,
            report: Ok(SnipV2Lifecycle::Active),
        });
        let first = publish_public_idempotent(&s, &f.path).unwrap();
        let second = publish_public_idempotent(&s, &f.path).unwrap();
        assert_eq!(first.action, PublicationAction::Ingested);
        assert_eq!(second.action, PublicationAction::ReusedActive);
        assert_eq!(first.root, second.root);
        assert_eq!(s.calls.borrow().ingests, 1);
    }

    #[test]
    fn own_pending_root_is_resumed_never_registered() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        let s = {
            let r = s.record(ME, 0);
            s.with_record(r)
        };
        s.resume.borrow_mut().push_back(Resume::To {
            lifecycle: 1,
            outcome: Ok(SnipV2ResumeOutcome::Active),
        });
        let p = publish_public_idempotent(&s, &f.path).unwrap();
        assert_eq!(p.action, PublicationAction::Resumed);
        assert_eq!(p.activated_at_height, 40);
        assert_eq!(counts(&s), (2, 0, 1), "lookup, resume, confirming lookup");
    }

    #[test]
    fn resume_reporting_active_is_not_trusted_without_the_chain() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        let s = {
            let r = s.record(ME, 0);
            s.with_record(r)
        };
        // resume says Active but the chain still reads Pending.
        s.resume.borrow_mut().push_back(Resume::To {
            lifecycle: 0,
            outcome: Ok(SnipV2ResumeOutcome::Active),
        });
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(matches!(err, SnipV2Error::StillPending { .. }), "{err}");
        assert_eq!(counts(&s), (2, 0, 1), "one resume only");
    }

    #[test]
    fn pending_owned_by_another_account_is_refused_untouched() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        let s = {
            let r = s.record(OTHER, 0);
            s.with_record(r)
        };
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        match &err {
            SnipV2Error::PendingOwnedByOther { owner, account, .. } => {
                assert_eq!(owner, OTHER);
                assert_eq!(account, ME);
            }
            other => panic!("{other}"),
        }
        assert!(err.to_string().contains(OTHER), "{err}");
        assert_eq!(counts(&s), (1, 0, 0));
    }

    #[test]
    fn pending_with_unknown_configured_account_is_refused() {
        let f = fixture();
        let mut s = Scripted::new(&f.expected);
        s.account = Err("no key file: SUM_KEY_FILE is not set".into());
        let s = {
            let r = s.record(ME, 0);
            s.with_record(r)
        };
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(matches!(err, SnipV2Error::PublisherAccount(_)), "{err}");
        assert_eq!(counts(&s), (1, 0, 0));
    }

    #[test]
    fn abandoned_root_is_refused() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        let s = {
            let r = s.record(ME, 2);
            s.with_record(r)
        };
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(matches!(err, SnipV2Error::AbandonedOnChain { .. }), "{err}");
        assert_eq!(counts(&s), (1, 0, 0));
    }

    #[test]
    fn every_conflicting_field_is_refused_in_every_lifecycle() {
        let f = fixture();
        type Mutate = fn(&mut SnipV2FileInfo);
        let cases: [(&str, Mutate); 6] = [
            ("merkle_root", |r| {
                r.merkle_root = format!("0x{}", "cd".repeat(32))
            }),
            ("merkle_root", |r| r.merkle_root = "not-hex".into()),
            ("visibility", |r| r.visibility = 1),
            ("plaintext_size_bytes", |r| r.plaintext_size_bytes += 1),
            ("stored_size_bytes", |r| r.stored_size_bytes += 16),
            ("chunk_count", |r| r.chunk_count += 1),
        ];
        for lifecycle in [0u8, 1, 2] {
            for owner in [ME, OTHER] {
                for (field, mutate) in cases {
                    let s = Scripted::new(&f.expected);
                    let mut r = s.record(owner, lifecycle);
                    mutate(&mut r);
                    let s = s.with_record(r);
                    let err = publish_public_idempotent(&s, &f.path).unwrap_err();
                    match &err {
                        SnipV2Error::ChainRecordMismatch { field: got, .. } => {
                            assert_eq!(*got, field, "lifecycle {lifecycle}")
                        }
                        other => panic!("lifecycle {lifecycle} {field}: {other}"),
                    }
                    assert_eq!(counts(&s), (1, 0, 0), "{field}: nothing sent");
                }
            }
        }
    }

    #[test]
    fn malformed_records_are_refused() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        let s = {
            let mut r = s.record(ME, 1);
            r.lifecycle = 9;
            s.with_record(r)
        };
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(
            matches!(err, SnipV2Error::ChainRecordMalformed { .. }),
            "{err}"
        );

        let s = Scripted::new(&f.expected);
        let s = {
            let mut r = s.record(ME, 1);
            r.activated_at_height = None;
            s.with_record(r)
        };
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(
            matches!(err, SnipV2Error::ChainRecordMalformed { .. }),
            "{err}"
        );
        assert_eq!(counts(&s), (1, 0, 0));
    }

    #[test]
    fn unreachable_rpc_refuses_before_registering() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        *s.fail_lookups.borrow_mut() = 1;
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(matches!(err, SnipV2Error::ChainLookup { .. }), "{err}");
        assert!(err.to_string().contains("connection refused"), "{err}");
        assert_eq!(counts(&s), (1, 0, 0), "no ingest without a chain read");
    }

    #[test]
    fn ingest_failure_with_no_record_keeps_snips_diagnostics() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        s.ingest.borrow_mut().push_back(Ingest::Fail(
            "Error: ingest-v2: production profile requires chain_getChainParams".into(),
        ));
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(matches!(err, SnipV2Error::NotRegistered { .. }), "{err}");
        assert!(err.to_string().contains("chain_getChainParams"), "{err}");
        assert_eq!(
            counts(&s),
            (2, 1, 0),
            "one ingest, then a re-read; no retry"
        );
    }

    #[test]
    fn registration_landed_but_unconfirmed_then_rpc_down_is_unconfirmed() {
        // Issue #149's first attempt: the registration landed, sum-node lost
        // the RPC and printed no outcome, and the re-read fails too.
        let f = fixture();
        let mut s = Scripted::new(&f.expected);
        s.fail_from = Some(2);
        s.ingest.borrow_mut().push_back(Ingest::Register {
            lifecycle: 0,
            report: Err("RegisterFilePendingV2: error sending request".into()),
        });
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        match &err {
            SnipV2Error::PublicationUnconfirmed {
                attempt, recheck, ..
            } => {
                assert!(attempt.contains("error sending request"), "{attempt}");
                assert!(recheck.contains("connection refused"), "{recheck}");
            }
            other => panic!("{other}"),
        }
        assert_eq!(counts(&s), (2, 1, 0));
    }

    #[test]
    fn retry_after_registration_without_activation_resumes() {
        // Issue #149's second attempt, after this change: the run that
        // follows the interrupted one finds its own Pending record.
        let f = fixture();
        let mut s = Scripted::new(&f.expected);
        s.fail_from = Some(2);
        s.ingest.borrow_mut().push_back(Ingest::Register {
            lifecycle: 0,
            report: Err("RegisterFilePendingV2: error sending request".into()),
        });
        assert!(publish_public_idempotent(&s, &f.path).is_err());
        s.fail_from = None;
        s.resume.borrow_mut().push_back(Resume::To {
            lifecycle: 1,
            outcome: Ok(SnipV2ResumeOutcome::Active),
        });
        let p = publish_public_idempotent(&s, &f.path).unwrap();
        assert_eq!(p.action, PublicationAction::Resumed);
        assert_eq!(s.calls.borrow().ingests, 1, "never registered again");
    }

    #[test]
    fn own_ingest_left_pending_is_resumed_in_the_same_call() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        s.ingest.borrow_mut().push_back(Ingest::Register {
            lifecycle: 0,
            report: Err("lifecycle 'Pending' is not allowed".into()),
        });
        s.resume.borrow_mut().push_back(Resume::To {
            lifecycle: 1,
            outcome: Ok(SnipV2ResumeOutcome::Active),
        });
        let p = publish_public_idempotent(&s, &f.path).unwrap();
        assert_eq!(p.action, PublicationAction::Resumed);
        assert_eq!(counts(&s), (3, 1, 1));
    }

    #[test]
    fn resume_left_pending_is_a_failure_with_its_diagnostics() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        let s = {
            let r = s.record(ME, 0);
            s.with_record(r)
        };
        s.resume.borrow_mut().push_back(Resume::To {
            lifecycle: 0,
            outcome: Ok(SnipV2ResumeOutcome::Pending {
                diagnostics: "WARN RESUME: still PENDING failed_stage=Coverage".into(),
            }),
        });
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(matches!(err, SnipV2Error::StillPending { .. }), "{err}");
        assert!(err.to_string().contains("failed_stage=Coverage"), "{err}");
        assert_eq!(counts(&s), (2, 0, 1));
    }

    #[test]
    fn resume_error_then_still_pending_is_a_failure_with_the_error() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        let s = {
            let r = s.record(ME, 0);
            s.with_record(r)
        };
        s.resume.borrow_mut().push_back(Resume::To {
            lifecycle: 0,
            outcome: Err("Error: resume: production profile requires chain_getChainParams".into()),
        });
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(matches!(err, SnipV2Error::StillPending { .. }), "{err}");
        assert!(err.to_string().contains("chain_getChainParams"), "{err}");
    }

    #[test]
    fn resume_abandoned_and_root_mismatch_are_failures() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        let s = {
            let r = s.record(ME, 0);
            s.with_record(r)
        };
        s.resume.borrow_mut().push_back(Resume::To {
            lifecycle: 2,
            outcome: Ok(SnipV2ResumeOutcome::Abandoned {
                diagnostics: String::new(),
            }),
        });
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(matches!(err, SnipV2Error::AbandonedOnChain { .. }), "{err}");

        let s = Scripted::new(&f.expected);
        let s = {
            let r = s.record(ME, 0);
            s.with_record(r)
        };
        s.resume.borrow_mut().push_back(Resume::To {
            lifecycle: 0,
            outcome: Ok(SnipV2ResumeOutcome::RootMismatch {
                diagnostics: "WARN RESUME: root mismatch".into(),
            }),
        });
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(matches!(err, SnipV2Error::RootMismatch { .. }), "{err}");
        assert_eq!(counts(&s), (1, 0, 1), "no re-read after a root mismatch");
    }

    #[test]
    fn ingest_reporting_a_different_root_is_refused() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        s.ingest.borrow_mut().push_back(Ingest::WrongRoot);
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(matches!(err, SnipV2Error::RootMismatch { .. }), "{err}");
    }

    #[test]
    fn race_with_another_publishers_active_registration_is_reused() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        *s.record_on_race.borrow_mut() = Some(s.record(OTHER, 1));
        let p = publish_public_idempotent(&s, &f.path).unwrap();
        // Our ingest ran (and failed: already registered); the re-read found
        // the other publisher's Active record.
        assert_eq!(p.action, PublicationAction::ReusedActive);
        assert_eq!(p.owner, OTHER);
        assert_eq!(counts(&s), (2, 1, 0));
    }

    #[test]
    fn race_with_another_publishers_pending_registration_is_refused() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        *s.record_on_race.borrow_mut() = Some(s.record(OTHER, 0));
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(
            matches!(err, SnipV2Error::PendingOwnedByOther { .. }),
            "{err}"
        );
        assert_eq!(counts(&s), (2, 1, 0), "no second ingest, no resume");
    }

    #[test]
    fn race_with_our_own_pending_registration_is_resumed() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        *s.record_on_race.borrow_mut() = Some(s.record(ME, 0));
        s.resume.borrow_mut().push_back(Resume::To {
            lifecycle: 1,
            outcome: Ok(SnipV2ResumeOutcome::Active),
        });
        let p = publish_public_idempotent(&s, &f.path).unwrap();
        assert_eq!(p.action, PublicationAction::Resumed);
        assert_eq!(counts(&s), (3, 1, 1));
    }

    #[test]
    fn race_with_a_conflicting_registration_is_refused() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        let mut r = s.record(OTHER, 0);
        r.plaintext_size_bytes = 17;
        *s.record_on_race.borrow_mut() = Some(r);
        let err = publish_public_idempotent(&s, &f.path).unwrap_err();
        assert!(
            matches!(err, SnipV2Error::ChainRecordMismatch { .. }),
            "{err}"
        );
    }

    #[test]
    fn ingest_reporting_active_without_a_chain_record_is_not_success() {
        let f = fixture();
        let s = Scripted::new(&f.expected);
        // Reports Active, but leaves nothing on chain: simulate by an ingest
        // that registers and a record that disappears.
        s.ingest.borrow_mut().push_back(Ingest::Register {
            lifecycle: 1,
            report: Ok(SnipV2Lifecycle::Active),
        });
        struct Forgetful<'a>(&'a Scripted);
        impl SnipV2Adapter for Forgetful<'_> {
            fn ingest_public(&self, p: &Path) -> Result<SnipV2ObjectRef, SnipV2Error> {
                let r = self.0.ingest_public(p);
                *self.0.record.borrow_mut() = None;
                r
            }
            fn download_public(&self, r: &SnipV2ObjectId, p: &Path) -> Result<(), SnipV2Error> {
                self.0.download_public(r, p)
            }
        }
        impl SnipV2Publisher for Forgetful<'_> {
            fn file_info_v2(
                &self,
                r: &SnipV2ObjectId,
            ) -> Result<Option<SnipV2FileInfo>, SnipV2Error> {
                self.0.file_info_v2(r)
            }
            fn publisher_account(&self) -> Result<String, SnipV2Error> {
                self.0.publisher_account()
            }
            fn resume_public(
                &self,
                r: &SnipV2ObjectId,
                p: &Path,
            ) -> Result<SnipV2ResumeOutcome, SnipV2Error> {
                self.0.resume_public(r, p)
            }
        }
        let err = publish_public_idempotent(&Forgetful(&s), &f.path).unwrap_err();
        assert!(matches!(err, SnipV2Error::NotRegistered { .. }), "{err}");
        assert_eq!(counts(&s), (2, 1, 0));
    }
}
