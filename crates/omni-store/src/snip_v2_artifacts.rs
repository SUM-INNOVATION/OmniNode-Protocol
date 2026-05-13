//! Phase 5 Stage 2 — SNIP V2 model-artifact publish and restore.
//!
//! Higher-level orchestration on top of the Stage 1 CLI adapter
//! ([`crate::snip_v2`]) and the existing [`crate::store::ShardStore`].
//! Stage 2 is intentionally additive:
//!
//! - `omni-store`'s CID / BLAKE3 identity is unchanged and remains
//!   authoritative for cache resolution.
//! - libp2p shard transfer (`omni-net`) is unchanged.
//! - `manifest::build_manifest` is unchanged — SNIP V2 publishing is an
//!   explicit, separate call.
//!
//! All public functions are generic over [`crate::snip_v2::SnipV2Adapter`]
//! so unit tests can substitute a fake adapter without spawning `sum-node`.

use std::path::Path;

use omni_types::model::ModelManifest;
use omni_types::phase5::{SnipV2ObjectId, SnipV2ObjectRef};

use crate::error::{Result, StoreError};
use crate::manifest;
use crate::mmap;
use crate::snip_v2::SnipV2Adapter;
use crate::store::ShardStore;
use crate::verify;

// ── Reports ───────────────────────────────────────────────────────────────────

/// Returned by [`publish_to_snip`] on full success. A `Result::Err` produces
/// no report; callers detect resume cases by re-calling and inspecting the
/// second-call counters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PublishReport {
    pub shards_published: usize,
    pub shards_skipped_already_populated: usize,
    pub manifest_published: bool,
    pub manifest_skipped_already_populated: bool,
}

/// Returned by [`restore_from_snip`] on full success.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RestoreReport {
    pub shards_downloaded: usize,
    pub shards_skipped_already_cached: usize,
}

// ── Publish ───────────────────────────────────────────────────────────────────

/// Publish a model's shard files and manifest to SNIP V2 Public, populating
/// the optional `snip_v2` refs in `manifest` as ingest calls succeed.
///
/// Resumability: the manifest file at `manifest_path` is rewritten **after
/// every successful shard ingest**, so a mid-loop failure leaves
/// already-published shard refs on disk. Re-running with the same arguments
/// after fixing the underlying issue skips populated shards and resumes
/// from the failure point.
///
/// On-disk vs. SNIP-stored bytes — intentional divergence: the manifest
/// bytes ingested into SNIP are the *shards-populated, top-level-`None`*
/// form. After SNIP returns the root, this function writes a final on-disk
/// copy that additionally carries the top-level `snip_v2` ref as a local
/// "self-pointer" annotation. The SNIP-stored bytes therefore do **not**
/// match the final on-disk bytes byte-for-byte; this is documented behavior
/// so that the SNIP root is a stable identifier of the canonical manifest
/// independent of the local-view annotation.
///
/// Preconditions:
/// - `manifest_path` already exists on disk and its current contents agree
///   with the in-memory `manifest` (the caller, e.g. `OmniStore::ingest_model`,
///   has produced it).
/// - Every shard with `snip_v2: None` has its `<cid>.shard` file present in
///   `store`; otherwise [`StoreError::ShardFileMissing`] is returned.
pub fn publish_to_snip<A: SnipV2Adapter>(
    adapter: &A,
    store: &ShardStore,
    manifest: &mut ModelManifest,
    manifest_path: &Path,
) -> Result<PublishReport> {
    let mut report = PublishReport::default();

    // 1. Per-shard pass. Persist the manifest after every successful ingest
    //    so a mid-loop failure preserves prior refs on disk. Indexed
    //    iteration so the mid-loop `write_manifest(manifest, ...)` borrow
    //    of the whole manifest is allowed.
    for i in 0..manifest.shards.len() {
        if manifest.shards[i].snip_v2.is_some() {
            report.shards_skipped_already_populated += 1;
            continue;
        }

        let shard_path = store.shard_path(&manifest.shards[i].cid);
        if !shard_path.is_file() {
            return Err(StoreError::ShardFileMissing {
                cid: manifest.shards[i].cid.clone(),
                path: shard_path,
            });
        }

        let object_ref = adapter.ingest_public(&shard_path)?;
        let size_bytes = manifest.shards[i].size_bytes;
        let merkle_root = object_ref.merkle_root;
        manifest.shards[i].snip_v2 = Some(SnipV2ObjectRef {
            merkle_root,
            lifecycle: object_ref.lifecycle,
            plaintext_size_bytes: Some(size_bytes),
        });

        manifest::write_manifest(manifest, manifest_path)?;
        tracing::info!(
            cid = %manifest.shards[i].cid,
            merkle = %merkle_root,
            "published shard to SNIP V2"
        );
        report.shards_published += 1;
    }

    // 2. Top-level manifest pass. The bytes ingested here include every
    //    shard's `snip_v2` ref but NOT a top-level `snip_v2` (which is still
    //    `None` on disk at this point). That is exactly the form a future
    //    `restore_manifest_from_snip` will reconstruct.
    if manifest.snip_v2.is_none() {
        let object_ref = adapter.ingest_public(manifest_path)?;
        let file_len = std::fs::metadata(manifest_path)?.len();
        manifest.snip_v2 = Some(SnipV2ObjectRef {
            merkle_root: object_ref.merkle_root,
            lifecycle: object_ref.lifecycle,
            plaintext_size_bytes: Some(file_len),
        });
        // Persist a local-view file that includes the self-pointer. The SNIP
        // root remains stable because it was computed before this rewrite.
        manifest::write_manifest(manifest, manifest_path)?;
        tracing::info!(
            merkle = %object_ref.merkle_root,
            "published manifest to SNIP V2"
        );
        report.manifest_published = true;
    } else {
        report.manifest_skipped_already_populated = true;
    }

    Ok(report)
}

// ── Restore: manifest by root ─────────────────────────────────────────────────

/// Download a SNIP V2-stored manifest into `dest_manifest_path` and parse it.
///
/// The returned `ModelManifest` reflects the canonical SNIP bytes, which by
/// construction do **not** include a top-level `snip_v2` ref — see
/// [`publish_to_snip`]. Callers that want a self-pointing local annotation
/// may set `manifest.snip_v2 = Some(...)` after this function returns and
/// persist with [`manifest::write_manifest`]; this function deliberately
/// does not do that on behalf of the caller.
///
/// No external content check is performed beyond CBOR parse: SNIP's own
/// Merkle invariant guarantees the downloaded bytes hash to `root`.
pub fn restore_manifest_from_snip<A: SnipV2Adapter>(
    adapter: &A,
    root: &SnipV2ObjectId,
    dest_manifest_path: &Path,
) -> Result<ModelManifest> {
    adapter.download_public(root, dest_manifest_path)?;
    manifest::read_manifest(dest_manifest_path)
}

// ── Restore: shards from manifest ─────────────────────────────────────────────

/// Download every shard in `manifest` whose `<cid>.shard` is not already in
/// `store`, using each shard's `snip_v2.merkle_root` as the SNIP V2
/// identifier.
///
/// For each missing shard the download lands at `<store>/<cid>.shard.partial`
/// and is then verified against the manifest's authoritative BLAKE3 hash and
/// CID via the existing [`crate::verify`] primitives. Only after both
/// checks pass is the file atomically renamed to `<store>/<cid>.shard`. On
/// any verification failure the partial is removed and
/// [`StoreError::IntegrityMismatch`] is returned; the cache state is
/// otherwise unchanged.
///
/// If any shard in `manifest` lacks a `snip_v2` ref, [`StoreError::ShardLacksSnipRef`]
/// is returned for that shard.
pub fn restore_from_snip<A: SnipV2Adapter>(
    adapter: &A,
    store: &ShardStore,
    manifest: &ModelManifest,
) -> Result<RestoreReport> {
    let mut report = RestoreReport::default();

    for shard in &manifest.shards {
        if store.has(&shard.cid) {
            report.shards_skipped_already_cached += 1;
            continue;
        }

        let snip_ref = shard
            .snip_v2
            .as_ref()
            .ok_or_else(|| StoreError::ShardLacksSnipRef {
                cid: shard.cid.clone(),
            })?;

        let tmp = store.temp_path_for(&shard.cid);
        adapter.download_public(&snip_ref.merkle_root, &tmp)?;

        // Verify against authoritative CID + BLAKE3 from the manifest. The
        // mmap is dropped at the end of the closure so the atomic rename
        // below can proceed without holding a mapping.
        let verify_outcome: Result<()> = (|| -> Result<()> {
            let mapped = mmap::mmap_file(&tmp)?;
            verify::verify_blake3(&mapped, &shard.blake3_hash)?;
            verify::verify_cid(&mapped, &shard.cid)?;
            Ok(())
        })();

        if let Err(e) = verify_outcome {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }

        std::fs::rename(&tmp, store.shard_path(&shard.cid))?;
        tracing::info!(
            cid = %shard.cid,
            merkle = %snip_ref.merkle_root,
            "restored shard from SNIP V2"
        );
        report.shards_downloaded += 1;
    }

    Ok(report)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::collections::{HashMap, HashSet};
    use std::path::PathBuf;

    use omni_types::model::{LayerRange, ShardDescriptor};
    use omni_types::phase5::{SnipV2Lifecycle, SnipV2ObjectRef};

    use crate::content_id;
    use crate::snip_v2::SnipV2Error;

    // ── Fake adapter ─────────────────────────────────────────────────────

    /// Content-addressed in-memory SNIP V2 fake. `ingest_public` reads the
    /// real file from disk, hashes it with BLAKE3 to produce the SNIP root,
    /// and stores the bytes; `download_public` retrieves those bytes (or a
    /// per-test override) and writes them to the requested path. This
    /// mirrors the real SNIP invariant "bytes hash to root" without
    /// spawning `sum-node`.
    #[derive(Default)]
    struct FakeState {
        ingest_calls:        Vec<PathBuf>,
        ingest_responses:    HashMap<PathBuf, SnipV2ObjectRef>,
        ingest_fail_paths:   HashSet<PathBuf>,
        objects:             HashMap<SnipV2ObjectId, Vec<u8>>,
        download_overrides:  HashMap<SnipV2ObjectId, Vec<u8>>,
        download_fail_roots: HashSet<SnipV2ObjectId>,
        download_calls:      Vec<(SnipV2ObjectId, PathBuf)>,
    }

    struct FakeSnipV2Adapter {
        state: RefCell<FakeState>,
    }

    impl FakeSnipV2Adapter {
        fn new() -> Self {
            Self { state: RefCell::new(FakeState::default()) }
        }
        fn fail_on_ingest(&self, path: PathBuf) {
            self.state.borrow_mut().ingest_fail_paths.insert(path);
        }
        fn clear_ingest_failures(&self) {
            self.state.borrow_mut().ingest_fail_paths.clear();
        }
        fn set_download_override(&self, root: SnipV2ObjectId, bytes: Vec<u8>) {
            self.state.borrow_mut().download_overrides.insert(root, bytes);
        }
        fn ingest_calls(&self) -> Vec<PathBuf> {
            self.state.borrow().ingest_calls.clone()
        }
        fn download_calls(&self) -> Vec<(SnipV2ObjectId, PathBuf)> {
            self.state.borrow().download_calls.clone()
        }
    }

    impl SnipV2Adapter for FakeSnipV2Adapter {
        fn ingest_public(&self, path: &Path) -> std::result::Result<SnipV2ObjectRef, SnipV2Error> {
            let pbuf = path.to_path_buf();
            {
                let mut s = self.state.borrow_mut();
                s.ingest_calls.push(pbuf.clone());
                if s.ingest_fail_paths.contains(&pbuf) {
                    return Err(SnipV2Error::NonZeroExit {
                        code: 1,
                        stderr: "fake forced ingest failure".into(),
                    });
                }
            }
            let bytes = std::fs::read(path).map_err(SnipV2Error::CommandSpawn)?;
            let mut s = self.state.borrow_mut();
            let r = match s.ingest_responses.get(&pbuf).cloned() {
                Some(r) => r,
                None => {
                    let hash = blake3::hash(&bytes);
                    let mut id_bytes = [0u8; 32];
                    id_bytes.copy_from_slice(hash.as_bytes());
                    SnipV2ObjectRef {
                        merkle_root: SnipV2ObjectId::from_bytes(id_bytes),
                        lifecycle: SnipV2Lifecycle::Active,
                        plaintext_size_bytes: None,
                    }
                }
            };
            s.objects.insert(r.merkle_root, bytes);
            Ok(r)
        }

        fn download_public(
            &self,
            root: &SnipV2ObjectId,
            output_path: &Path,
        ) -> std::result::Result<(), SnipV2Error> {
            let mut s = self.state.borrow_mut();
            s.download_calls.push((*root, output_path.to_path_buf()));
            if s.download_fail_roots.contains(root) {
                return Err(SnipV2Error::DownloadFailed {
                    code: 1,
                    stderr: "fake forced download failure".into(),
                });
            }
            let bytes = s.download_overrides
                .get(root)
                .cloned()
                .or_else(|| s.objects.get(root).cloned())
                .ok_or_else(|| SnipV2Error::DownloadFailed {
                    code: 1,
                    stderr: format!("fake: no bytes mapped for root {root}"),
                })?;
            std::fs::write(output_path, &bytes).map_err(SnipV2Error::CommandSpawn)?;
            Ok(())
        }
    }

    // ── Test helpers ─────────────────────────────────────────────────────

    /// Build a real shard byte stream and the matching `ShardDescriptor`.
    fn real_shard(index: u32, payload: &[u8], layers: (u32, u32)) -> (Vec<u8>, ShardDescriptor) {
        let cid = content_id::cid_from_data(payload);
        let blake3_hash = blake3::hash(payload).to_hex().to_string();
        let sd = ShardDescriptor {
            shard_index: index,
            cid,
            layer_range: LayerRange { start: layers.0, end: layers.1 },
            includes_embedding: index == 0,
            includes_output_head: false,
            size_bytes: payload.len() as u64,
            blake3_hash,
            snip_v2: None,
        };
        (payload.to_vec(), sd)
    }

    fn empty_manifest(shards: Vec<ShardDescriptor>) -> ModelManifest {
        ModelManifest {
            model_name: "test-model".into(),
            model_hash: "f".repeat(64),
            architecture: "llama".into(),
            total_layers: 4,
            quantization: "F16".into(),
            total_size_bytes: shards.iter().map(|s| s.size_bytes).sum(),
            gguf_version: 3,
            shards,
            snip_v2: None,
        }
    }

    /// Write real payloads into a fresh ShardStore, then assemble manifest
    /// + manifest_path on disk.
    fn setup_publish_fixture(
        payloads: &[(&[u8], (u32, u32))],
    ) -> (tempfile::TempDir, ShardStore, ModelManifest, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let store = ShardStore::new(dir.path().join("shards")).unwrap();
        let mut shards = Vec::new();
        for (i, (bytes, layers)) in payloads.iter().enumerate() {
            let (data, sd) = real_shard(i as u32, bytes, *layers);
            store.put(&sd.cid, &data).unwrap();
            shards.push(sd);
        }
        let manifest = empty_manifest(shards);
        let manifest_path = dir.path().join("manifest.cbor");
        manifest::write_manifest(&manifest, &manifest_path).unwrap();
        (dir, store, manifest, manifest_path)
    }

    // ── 1. publish populates all refs ────────────────────────────────────

    #[test]
    fn publish_populates_all_refs() {
        let (_dir, store, mut manifest, manifest_path) = setup_publish_fixture(&[
            (b"shard 0 bytes payload", (0, 3)),
            (b"shard 1 bytes payload", (4, 7)),
        ]);
        let fake = FakeSnipV2Adapter::new();

        let report = publish_to_snip(&fake, &store, &mut manifest, &manifest_path).unwrap();

        assert_eq!(report.shards_published, 2);
        assert_eq!(report.shards_skipped_already_populated, 0);
        assert!(report.manifest_published);
        assert!(!report.manifest_skipped_already_populated);

        for shard in &manifest.shards {
            assert!(shard.snip_v2.is_some());
            assert_eq!(
                shard.snip_v2.as_ref().unwrap().plaintext_size_bytes,
                Some(shard.size_bytes)
            );
        }
        assert!(manifest.snip_v2.is_some());

        // On-disk file must reflect the same refs (final write includes the
        // top-level self-pointer).
        let reloaded = manifest::read_manifest(&manifest_path).unwrap();
        assert!(reloaded.snip_v2.is_some());
        assert_eq!(
            reloaded.snip_v2.as_ref().unwrap().merkle_root,
            manifest.snip_v2.as_ref().unwrap().merkle_root
        );
        for shard in &reloaded.shards {
            assert!(shard.snip_v2.is_some());
        }
    }

    // ── 2. publish skips already-populated shards ────────────────────────

    #[test]
    fn publish_skips_already_populated_shards() {
        let (_dir, store, mut manifest, manifest_path) = setup_publish_fixture(&[
            (b"shard zero", (0, 3)),
            (b"shard one", (4, 7)),
        ]);
        // Pre-populate shard 0.
        let mut bytes = [0u8; 32];
        bytes.fill(0xAA);
        manifest.shards[0].snip_v2 = Some(SnipV2ObjectRef {
            merkle_root: SnipV2ObjectId::from_bytes(bytes),
            lifecycle: SnipV2Lifecycle::Active,
            plaintext_size_bytes: Some(manifest.shards[0].size_bytes),
        });
        manifest::write_manifest(&manifest, &manifest_path).unwrap();

        let fake = FakeSnipV2Adapter::new();
        let report = publish_to_snip(&fake, &store, &mut manifest, &manifest_path).unwrap();

        assert_eq!(report.shards_published, 1);
        assert_eq!(report.shards_skipped_already_populated, 1);

        // FakeAdapter must NOT have been called with shard 0's path.
        let shard0_path = store.shard_path(&manifest.shards[0].cid);
        assert!(!fake.ingest_calls().contains(&shard0_path));
        // It must have been called with shard 1's path.
        let shard1_path = store.shard_path(&manifest.shards[1].cid);
        assert!(fake.ingest_calls().contains(&shard1_path));
    }

    // ── 3. publish skips already-populated manifest ──────────────────────

    #[test]
    fn publish_skips_already_populated_manifest() {
        let (_dir, store, mut manifest, manifest_path) = setup_publish_fixture(&[
            (b"only shard", (0, 3)),
        ]);
        let mut bytes = [0u8; 32];
        bytes.fill(0xBB);
        manifest.snip_v2 = Some(SnipV2ObjectRef {
            merkle_root: SnipV2ObjectId::from_bytes(bytes),
            lifecycle: SnipV2Lifecycle::Active,
            plaintext_size_bytes: None,
        });
        manifest::write_manifest(&manifest, &manifest_path).unwrap();

        let fake = FakeSnipV2Adapter::new();
        let report = publish_to_snip(&fake, &store, &mut manifest, &manifest_path).unwrap();

        assert!(report.manifest_skipped_already_populated);
        assert!(!report.manifest_published);
        // Manifest path must NOT have been ingested.
        assert!(!fake.ingest_calls().contains(&manifest_path));
    }

    // ── 4. publish fails when local shard file is missing ───────────────

    #[test]
    fn publish_fails_when_local_shard_missing() {
        let dir = tempfile::tempdir().unwrap();
        let store = ShardStore::new(dir.path().join("shards")).unwrap();
        // Build a shard descriptor whose CID has no file on disk.
        let bogus_cid = "bafknonexistent".to_string();
        let manifest = empty_manifest(vec![ShardDescriptor {
            shard_index: 0,
            cid: bogus_cid.clone(),
            layer_range: LayerRange { start: 0, end: 3 },
            includes_embedding: true,
            includes_output_head: false,
            size_bytes: 0,
            blake3_hash: "0".repeat(64),
            snip_v2: None,
        }]);
        let manifest_path = dir.path().join("manifest.cbor");
        manifest::write_manifest(&manifest, &manifest_path).unwrap();
        let mut manifest = manifest;

        let fake = FakeSnipV2Adapter::new();
        let err = publish_to_snip(&fake, &store, &mut manifest, &manifest_path).unwrap_err();
        match err {
            StoreError::ShardFileMissing { cid, .. } => assert_eq!(cid, bogus_cid),
            other => panic!("expected ShardFileMissing, got {other:?}"),
        }
    }

    // ── 5. partial failure: shard 0 ref persisted, resume works ─────────

    #[test]
    fn publish_persists_shard_refs_after_partial_failure() {
        let (_dir, store, mut manifest, manifest_path) = setup_publish_fixture(&[
            (b"first", (0, 3)),
            (b"second", (4, 7)),
        ]);
        let shard0_path = store.shard_path(&manifest.shards[0].cid);
        let shard1_path = store.shard_path(&manifest.shards[1].cid);

        let fake = FakeSnipV2Adapter::new();
        fake.fail_on_ingest(shard1_path.clone());

        let err = publish_to_snip(&fake, &store, &mut manifest, &manifest_path).unwrap_err();
        assert!(matches!(err, StoreError::SnipV2(_)));

        // On-disk: shard 0 populated, shard 1 not, top-level not.
        let reloaded = manifest::read_manifest(&manifest_path).unwrap();
        assert!(reloaded.shards[0].snip_v2.is_some());
        assert!(reloaded.shards[1].snip_v2.is_none());
        assert!(reloaded.snip_v2.is_none());
        // FakeAdapter was called for both shard paths (shard 1 is the one
        // that failed; shard 0 succeeded before that).
        assert!(fake.ingest_calls().contains(&shard0_path));
        assert!(fake.ingest_calls().contains(&shard1_path));

        // Resume: clear the failure, re-call. The loaded-from-disk manifest
        // is our resume point.
        let mut resumed = reloaded;
        fake.clear_ingest_failures();
        let report =
            publish_to_snip(&fake, &store, &mut resumed, &manifest_path).unwrap();
        assert_eq!(report.shards_skipped_already_populated, 1);
        assert_eq!(report.shards_published, 1);
        assert!(report.manifest_published);
        assert!(resumed.snip_v2.is_some());
        assert!(resumed.shards[0].snip_v2.is_some());
        assert!(resumed.shards[1].snip_v2.is_some());
    }

    // ── 6. restore downloads missing shards ─────────────────────────────

    #[test]
    fn restore_downloads_missing_shards() {
        // Stage A: publish into a fresh store/adapter to seed FakeAdapter's
        // content-addressed object map.
        let (dir_a, store_a, mut manifest, manifest_path_a) = setup_publish_fixture(&[
            (b"alpha shard contents", (0, 3)),
            (b"beta shard contents", (4, 7)),
        ]);
        let fake = FakeSnipV2Adapter::new();
        publish_to_snip(&fake, &store_a, &mut manifest, &manifest_path_a).unwrap();
        drop(dir_a);

        // Stage B: a completely fresh, empty store. Restore from the
        // already-populated manifest.
        let dir_b = tempfile::tempdir().unwrap();
        let store_b = ShardStore::new(dir_b.path().join("shards")).unwrap();
        for shard in &manifest.shards {
            assert!(!store_b.has(&shard.cid));
        }

        let report = restore_from_snip(&fake, &store_b, &manifest).unwrap();
        assert_eq!(report.shards_downloaded, 2);
        assert_eq!(report.shards_skipped_already_cached, 0);
        for shard in &manifest.shards {
            assert!(store_b.has(&shard.cid));
            // No stray .partial files left behind.
            assert!(!store_b.temp_path_for(&shard.cid).exists());
        }
    }

    // ── 7. restore skips already-cached shards ──────────────────────────

    #[test]
    fn restore_skips_already_cached_shards() {
        let (_dir_a, store_a, mut manifest, manifest_path_a) = setup_publish_fixture(&[
            (b"cached shard", (0, 3)),
            (b"to be downloaded", (4, 7)),
        ]);
        let fake = FakeSnipV2Adapter::new();
        publish_to_snip(&fake, &store_a, &mut manifest, &manifest_path_a).unwrap();

        let dir_b = tempfile::tempdir().unwrap();
        let store_b = ShardStore::new(dir_b.path().join("shards")).unwrap();
        // Pre-cache shard 0.
        store_b.put(&manifest.shards[0].cid, b"cached shard").unwrap();
        let cached_root = manifest.shards[0].snip_v2.as_ref().unwrap().merkle_root;
        let other_root = manifest.shards[1].snip_v2.as_ref().unwrap().merkle_root;

        let report = restore_from_snip(&fake, &store_b, &manifest).unwrap();
        assert_eq!(report.shards_downloaded, 1);
        assert_eq!(report.shards_skipped_already_cached, 1);

        let download_roots: Vec<SnipV2ObjectId> =
            fake.download_calls().into_iter().map(|(r, _)| r).collect();
        assert!(!download_roots.contains(&cached_root));
        assert!(download_roots.contains(&other_root));
    }

    // ── 8. restore rejects corrupted bytes (BLAKE3 mismatch) ────────────

    #[test]
    fn restore_rejects_corrupted_shard_bytes() {
        let (_dir_a, store_a, mut manifest, manifest_path_a) = setup_publish_fixture(&[
            (b"original good bytes", (0, 3)),
        ]);
        let fake = FakeSnipV2Adapter::new();
        publish_to_snip(&fake, &store_a, &mut manifest, &manifest_path_a).unwrap();
        // Override the SNIP-stored bytes with deliberately wrong content.
        let shard_root = manifest.shards[0].snip_v2.as_ref().unwrap().merkle_root;
        fake.set_download_override(shard_root, b"DIFFERENT bytes".to_vec());

        let dir_b = tempfile::tempdir().unwrap();
        let store_b = ShardStore::new(dir_b.path().join("shards")).unwrap();
        let err = restore_from_snip(&fake, &store_b, &manifest).unwrap_err();
        assert!(matches!(err, StoreError::IntegrityMismatch { .. }));
        assert!(!store_b.has(&manifest.shards[0].cid));
        assert!(!store_b.temp_path_for(&manifest.shards[0].cid).exists());
    }

    // ── 9. restore rejects mismatched CID (BLAKE3 passes, CID does not) ─

    #[test]
    fn restore_rejects_corrupted_cid() {
        let payload = b"cid test payload bytes".to_vec();
        let real_blake3 = blake3::hash(&payload).to_hex().to_string();
        let real_cid = content_id::cid_from_data(&payload);

        // Construct a shard whose blake3_hash is correct but whose CID is
        // mismatched. BLAKE3 verification will pass; CID will fail.
        let fake_cid = "bafkrlies".to_string();
        let mut bytes = [0u8; 32];
        bytes.fill(0x77);
        let shard = ShardDescriptor {
            shard_index: 0,
            cid: fake_cid.clone(),
            layer_range: LayerRange { start: 0, end: 3 },
            includes_embedding: true,
            includes_output_head: false,
            size_bytes: payload.len() as u64,
            blake3_hash: real_blake3,
            snip_v2: Some(SnipV2ObjectRef {
                merkle_root: SnipV2ObjectId::from_bytes(bytes),
                lifecycle: SnipV2Lifecycle::Active,
                plaintext_size_bytes: Some(payload.len() as u64),
            }),
        };
        let manifest = empty_manifest(vec![shard]);

        // Seed the fake adapter so the configured root returns the real
        // payload (which has the right BLAKE3 but the wrong CID for the
        // descriptor's `fake_cid`).
        let fake = FakeSnipV2Adapter::new();
        fake.set_download_override(
            manifest.shards[0].snip_v2.as_ref().unwrap().merkle_root,
            payload,
        );

        // Sanity check that real_cid != fake_cid (otherwise the test is moot).
        assert_ne!(real_cid, fake_cid);

        let dir = tempfile::tempdir().unwrap();
        let store = ShardStore::new(dir.path().join("shards")).unwrap();
        let err = restore_from_snip(&fake, &store, &manifest).unwrap_err();
        assert!(matches!(err, StoreError::IntegrityMismatch { .. }));
        assert!(!store.has(&fake_cid));
        assert!(!store.temp_path_for(&fake_cid).exists());
    }

    // ── 10. restore errors when shard lacks snip_v2 ref ─────────────────

    #[test]
    fn restore_errors_when_shard_lacks_snip_ref() {
        let payload = b"any bytes".to_vec();
        let (_, sd) = real_shard(0, &payload, (0, 3));
        let manifest = empty_manifest(vec![sd]); // snip_v2 is None
        let dir = tempfile::tempdir().unwrap();
        let store = ShardStore::new(dir.path().join("shards")).unwrap();
        let fake = FakeSnipV2Adapter::new();

        let err = restore_from_snip(&fake, &store, &manifest).unwrap_err();
        match err {
            StoreError::ShardLacksSnipRef { cid } => {
                assert_eq!(cid, manifest.shards[0].cid)
            }
            other => panic!("expected ShardLacksSnipRef, got {other:?}"),
        }
    }

    // ── 11. restore_manifest_from_root writes, parses, top-level None ───

    #[test]
    fn restore_manifest_from_root_writes_and_parses() {
        // Publish into a fake adapter so we have a known manifest root.
        let (_dir_a, store_a, mut manifest, manifest_path_a) = setup_publish_fixture(&[
            (b"first", (0, 3)),
        ]);
        let fake = FakeSnipV2Adapter::new();
        publish_to_snip(&fake, &store_a, &mut manifest, &manifest_path_a).unwrap();
        let manifest_root = manifest.snip_v2.as_ref().unwrap().merkle_root;

        // Restore into a fresh path.
        let dir_b = tempfile::tempdir().unwrap();
        let dest = dir_b.path().join("restored_manifest.cbor");
        let restored = restore_manifest_from_snip(&fake, &manifest_root, &dest).unwrap();

        // Top-level contract: restored manifest's snip_v2 is None (the SNIP
        // bytes do not embed the self-pointer).
        assert!(restored.snip_v2.is_none());
        // Shard refs survive intact.
        assert_eq!(restored.shards.len(), 1);
        assert!(restored.shards[0].snip_v2.is_some());
        assert_eq!(
            restored.shards[0].snip_v2.as_ref().unwrap().merkle_root,
            manifest.shards[0].snip_v2.as_ref().unwrap().merkle_root
        );
        // Dest file exists and decodes equivalently.
        let reloaded = manifest::read_manifest(&dest).unwrap();
        assert!(reloaded.snip_v2.is_none());
        assert_eq!(reloaded.shards[0].cid, manifest.shards[0].cid);
    }

    // ── 12. restore_manifest_from_root rejects unparseable bytes ────────

    #[test]
    fn restore_manifest_from_root_rejects_unparseable_bytes() {
        let mut id_bytes = [0u8; 32];
        id_bytes.fill(0xCC);
        let root = SnipV2ObjectId::from_bytes(id_bytes);

        let fake = FakeSnipV2Adapter::new();
        fake.set_download_override(root, b"this is not valid CBOR".to_vec());

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("garbage.cbor");
        let err = restore_manifest_from_snip(&fake, &root, &dest).unwrap_err();
        assert!(matches!(err, StoreError::Other(_)));
    }

    // ── 13. round trip publish → restore preserves bytes ────────────────

    #[test]
    fn backward_compat_publish_then_restore_round_trip() {
        // Legacy manifest: no `snip_v2` anywhere.
        let payloads: &[(&[u8], (u32, u32))] = &[
            (b"shard A original bytes", (0, 3)),
            (b"shard B original bytes", (4, 7)),
            (b"shard C original bytes", (8, 11)),
        ];
        let original_bytes: Vec<Vec<u8>> =
            payloads.iter().map(|(b, _)| b.to_vec()).collect();
        let (_dir_a, store_a, mut manifest, manifest_path_a) =
            setup_publish_fixture(payloads);
        for shard in &manifest.shards {
            assert!(shard.snip_v2.is_none());
        }
        assert!(manifest.snip_v2.is_none());

        let fake = FakeSnipV2Adapter::new();
        publish_to_snip(&fake, &store_a, &mut manifest, &manifest_path_a).unwrap();

        // Fresh empty store. Restore.
        let dir_b = tempfile::tempdir().unwrap();
        let store_b = ShardStore::new(dir_b.path().join("shards")).unwrap();
        let report = restore_from_snip(&fake, &store_b, &manifest).unwrap();
        assert_eq!(report.shards_downloaded, 3);

        // Every restored byte stream matches the original.
        for (shard, expected) in manifest.shards.iter().zip(original_bytes.iter()) {
            let got = store_b.get(&shard.cid).unwrap();
            assert_eq!(&got, expected);
        }
    }

    // ── 14. empty manifest publish: noop shards, manifest still published ─

    #[test]
    fn empty_manifest_publish_is_noop_except_for_top_level() {
        let dir = tempfile::tempdir().unwrap();
        let store = ShardStore::new(dir.path().join("shards")).unwrap();
        let mut manifest = empty_manifest(vec![]);
        let manifest_path = dir.path().join("manifest.cbor");
        manifest::write_manifest(&manifest, &manifest_path).unwrap();

        let fake = FakeSnipV2Adapter::new();
        let report = publish_to_snip(&fake, &store, &mut manifest, &manifest_path).unwrap();

        assert_eq!(report.shards_published, 0);
        assert_eq!(report.shards_skipped_already_populated, 0);
        assert!(report.manifest_published);
        assert!(manifest.snip_v2.is_some());
        // The only ingest call was for the manifest path itself.
        assert_eq!(fake.ingest_calls(), vec![manifest_path.clone()]);
    }

    // ── 15. ingest call order matches manifest.shards order ─────────────

    #[test]
    fn fake_adapter_ingest_call_order_is_stable() {
        let (_dir, store, mut manifest, manifest_path) = setup_publish_fixture(&[
            (b"zero", (0, 3)),
            (b"one", (4, 7)),
            (b"two", (8, 11)),
        ]);
        let expected: Vec<PathBuf> = manifest
            .shards
            .iter()
            .map(|s| store.shard_path(&s.cid))
            .chain(std::iter::once(manifest_path.clone()))
            .collect();

        let fake = FakeSnipV2Adapter::new();
        publish_to_snip(&fake, &store, &mut manifest, &manifest_path).unwrap();
        assert_eq!(fake.ingest_calls(), expected);
    }
}
