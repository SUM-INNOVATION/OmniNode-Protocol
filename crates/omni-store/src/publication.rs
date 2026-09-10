//! Immutable publication into a cooperating, exclusively managed store root.
//!
//! Final names never open for writing. A successful put also synchronizes a
//! previously published duplicate. This does not protect mappings from arbitrary
//! external writers, writable hard-link aliases, or replacement of the root.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::content_id;
use crate::error::{Result, StoreError};

pub(crate) struct PublicationStore {
    root: PathBuf,
    directory: File,
    suffix: &'static str,
    #[cfg(test)]
    fault: std::sync::Mutex<Option<&'static str>>,
}

macro_rules! checkpoint {
    ($store:expr, $name:literal) => {
        #[cfg(test)]
        $store.checkpoint($name)?;
    };
}

impl PublicationStore {
    pub(crate) fn new(root: PathBuf, suffix: &'static str) -> Result<Self> {
        // Fail closed BEFORE any filesystem mutation. Rejecting at construction
        // (rather than at publish time) also keeps the confined path/read
        // operations below unreachable on targets where their confinement is
        // not established.
        #[cfg(not(unix))]
        {
            drop((root, suffix));
            Err(StoreError::UnsupportedPlatform)
        }
        #[cfg(unix)]
        {
            let root = create_root(root)?;
            let directory = open_directory(&root)?;
            Ok(Self {
                root,
                directory,
                suffix,
                #[cfg(test)]
                fault: std::sync::Mutex::new(None),
            })
        }
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    fn name(&self, key: &str) -> Result<String> {
        let mut components = Path::new(key).components();
        let single_component = matches!(components.next(), Some(std::path::Component::Normal(_)))
            && components.next().is_none();
        if key.is_empty()
            || key.len() > 249
            || key == "."
            || key == ".."
            || key.contains(['/', '\\', '\0'])
            || !single_component
        {
            return Err(StoreError::InvalidKey(key.to_owned()));
        }
        Ok(format!("{key}{}", self.suffix))
    }

    pub(crate) fn path(&self, key: &str) -> Result<PathBuf> {
        Ok(self.root.join(self.name(key)?))
    }

    pub(crate) fn open(&self, key: &str) -> Result<Option<File>> {
        let name = self.name(key)?;
        match open_regular(&self.directory, &self.root, &name) {
            Ok(file) => Ok(Some(file)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    pub(crate) fn has(&self, key: &str) -> bool {
        self.open(key).ok().flatten().is_some()
    }

    pub(crate) fn get(&self, key: &str) -> Result<Vec<u8>> {
        let mut file = self
            .open(key)?
            .ok_or_else(|| StoreError::NotFound(key.to_owned()))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    pub(crate) fn put(&self, cid: &str, bytes: &[u8]) -> Result<()> {
        self.canonical_cid(cid)?;
        crate::verify::verify_cid(bytes, cid)?;
        self.publish_with(cid, None, |path| {
            let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
            checkpoint!(self, "write");
            file.write_all(bytes)?;
            Ok(())
        })?;
        Ok(())
    }

    /// The callback must finish all writes before returning. It receives an
    /// attempt-owned path, never a final name or a shared download filename.
    pub(crate) fn publish_with<F>(
        &self,
        cid: &str,
        expected_blake3: Option<&str>,
        write: F,
    ) -> Result<bool>
    where
        F: FnOnce(&Path) -> Result<()>,
    {
        self.canonical_cid(cid)?;
        let staging = tempfile::Builder::new()
            .prefix(".publish-")
            .tempdir_in(&self.root)?;
        let result = (|| {
            let stage_directory = open_directory(staging.path())?;
            write(&staging.path().join("payload"))?;
            let mut payload = open_regular(&stage_directory, staging.path(), "payload")?;
            verify_file(&mut payload, cid, expected_blake3)?;
            checkpoint!(self, "file_sync");
            payload.sync_all()?;
            drop(payload);
            checkpoint!(self, "link");
            let name = self.name(cid)?;
            let inserted = match link_payload(
                &stage_directory,
                staging.path(),
                &self.directory,
                &self.root,
                &name,
            ) {
                Ok(()) => true,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    // EEXIST is not evidence of content identity or durability.
                    // Disappearance here is retryable by the caller.
                    if !self.verify_existing(cid, expected_blake3)? {
                        return Err(StoreError::Io(io::Error::new(
                            io::ErrorKind::NotFound,
                            "publication destination disappeared; retry",
                        )));
                    }
                    false
                }
                Err(error) => return Err(error.into()),
            };
            checkpoint!(self, "after_link");
            if inserted {
                self.sync_directory(cid)?;
            }
            Ok(inserted)
        })();
        // Cleanup only this attempt; never remove the final name as rollback.
        // A process crash leaves an identifiable directory for offline cleanup.
        #[cfg(test)]
        if self.checkpoint("cleanup").is_err() {
            let residue = staging.keep();
            tracing::warn!(path = %residue.display(), "publication staging cleanup failed");
            return result;
        }
        if let Err(error) = staging.close() {
            tracing::warn!(%error, "publication staging cleanup failed");
        }
        result
    }

    pub(crate) fn verify_existing(&self, cid: &str, expected_blake3: Option<&str>) -> Result<bool> {
        self.canonical_cid(cid)?;
        let Some(mut file) = self.open(cid)? else {
            return Ok(false);
        };
        verify_file(&mut file, cid, expected_blake3)?;
        checkpoint!(self, "file_sync");
        file.sync_all()?;
        self.sync_directory(cid)?;
        Ok(true)
    }

    fn sync_directory(&self, cid: &str) -> Result<()> {
        let result: Result<()> = (|| {
            checkpoint!(self, "directory_sync");
            self.directory.sync_all()?;
            Ok(())
        })();
        result.map_err(|error| StoreError::DurabilityUnconfirmed {
            cid: cid.to_owned(),
            detail: error.to_string(),
        })
    }

    fn canonical_cid(&self, cid: &str) -> Result<()> {
        self.name(cid)?;
        let parsed = cid::Cid::try_from(cid).map_err(|_| StoreError::InvalidCid(cid.to_owned()))?;
        if parsed.version() != cid::Version::V1
            || parsed.codec() != 0x55
            || parsed.hash().code() != 0x1e
            || parsed.hash().digest().len() != 32
            || parsed.to_string() != cid
        {
            return Err(StoreError::InvalidCid(cid.to_owned()));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn delete(&self, key: &str) -> Result<bool> {
        let name = self.name(key)?;
        if self.open(key)?.is_none() {
            return Ok(false);
        }
        #[cfg(unix)]
        let result =
            rustix::fs::unlinkat(&self.directory, name.as_str(), rustix::fs::AtFlags::empty())
                .map_err(io::Error::from);
        #[cfg(not(unix))]
        let result: io::Result<()> = {
            drop(name);
            Err(unsupported_platform())
        };
        match result {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    #[cfg(test)]
    fn checkpoint(&self, name: &str) -> Result<()> {
        if std::env::var("S3P_CRASH_AT").as_deref() == Ok(name) {
            std::process::exit(73);
        }
        if self
            .fault
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|point| *point == name)
        {
            return Err(io::Error::other(format!("injected {name} failure")).into());
        }
        Ok(())
    }
}

fn verify_file(file: &mut File, cid: &str, expected_blake3: Option<&str>) -> Result<()> {
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = match file.read(&mut buffer) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let hash = hasher.finalize();
    if let Some(expected) = expected_blake3 {
        let actual = hash.to_hex().to_string();
        if actual != expected {
            return Err(StoreError::IntegrityMismatch {
                expected: expected.to_owned(),
                actual,
            });
        }
    }
    let actual = content_id::cid_from_blake3_hash(&hash);
    if actual != cid {
        return Err(StoreError::IntegrityMismatch {
            expected: cid.to_owned(),
            actual,
        });
    }
    Ok(())
}

// A newly created root is itself a directory entry. Synchronize every newly
// created ancestor plus its existing parent before promising durable files
// inside it; syncing only the final store directory would miss that entry.
fn create_root(root: PathBuf) -> io::Result<PathBuf> {
    let absolute = if root.is_absolute() {
        root
    } else {
        std::env::current_dir()?.join(root)
    };
    let mut missing = Vec::new();
    let mut ancestor = absolute.as_path();
    loop {
        match fs::symlink_metadata(ancestor) {
            Ok(_) => break,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                missing.push(ancestor.to_owned());
                ancestor = ancestor.parent().ok_or(error)?;
            }
            Err(error) => return Err(error),
        }
    }
    fs::create_dir_all(&absolute)?;
    if !missing.is_empty() {
        for created in &missing {
            open_directory(&fs::canonicalize(created)?)?.sync_all()?;
        }
        open_directory(&fs::canonicalize(ancestor)?)?.sync_all()?;
    }
    fs::canonicalize(absolute)
}

#[cfg(unix)]
fn open_directory(path: &Path) -> io::Result<File> {
    use rustix::fs::{Mode, OFlags, open};
    Ok(open(
        path,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )?
    .into())
}

#[cfg(unix)]
fn open_regular(directory: &File, _root: &Path, name: &str) -> io::Result<File> {
    use rustix::fs::{Mode, OFlags, openat};
    let file: File = openat(
        directory,
        name,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )?
    .into();
    require_regular(file)
}

#[cfg(unix)]
fn link_payload(
    stage: &File,
    _stage_path: &Path,
    root: &File,
    _root_path: &Path,
    name: &str,
) -> io::Result<()> {
    rustix::fs::linkat(stage, "payload", root, name, rustix::fs::AtFlags::empty())
        .map_err(Into::into)
}

// Native Windows and every other non-Unix target are FAIL-CLOSED for
// publication. `PublicationStore::new` refuses before touching the filesystem,
// so the shims below are unreachable; they exist only so the crate keeps
// compiling and deliberately provide no partial implementation. The previous
// Windows helpers were removed rather than left in place: read-only confined
// handles cannot satisfy FlushFileBuffers, so neither file nor directory
// durability could be claimed for them, and a hard-link fallback gives no
// no-clobber guarantee. Equivalent native support is tracked in Omni #119.

#[cfg(not(unix))]
fn unsupported_platform() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "native Windows store publication is not supported; run under WSL2 \
         or another Unix target",
    )
}

#[cfg(not(unix))]
fn open_directory(_path: &Path) -> io::Result<File> {
    Err(unsupported_platform())
}

#[cfg(not(unix))]
fn open_regular(_directory: &File, _root: &Path, _name: &str) -> io::Result<File> {
    Err(unsupported_platform())
}

#[cfg(not(unix))]
fn link_payload(
    _stage: &File,
    _stage_path: &Path,
    _root: &File,
    _root_path: &Path,
    _name: &str,
) -> io::Result<()> {
    Err(unsupported_platform())
}

fn require_regular(file: File) -> io::Result<File> {
    if !file.metadata()?.is_file() {
        return Err(io::Error::other("not a regular store file"));
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Child, Command};

    fn store(root: &Path) -> PublicationStore {
        PublicationStore::new(root.to_owned(), ".chunk").unwrap()
    }

    fn residue_count(root: &Path) -> usize {
        fs::read_dir(root)
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".publish-")
            })
            .count()
    }

    fn child(root: &Path, mode: &str, crash: Option<&str>) -> Child {
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", "publication::tests::child_worker", "--nocapture"])
            .env("S3P_CHILD_ROOT", root)
            .env("S3P_CHILD_MODE", mode)
            .env_remove("S3P_CRASH_AT");
        if let Some(point) = crash {
            cmd.env("S3P_CRASH_AT", point);
        }
        cmd.spawn().unwrap()
    }

    #[test]
    fn child_worker() {
        let Some(root) = std::env::var_os("S3P_CHILD_ROOT") else {
            return;
        };
        let root = PathBuf::from(root);
        let store = store(&root);
        let bytes = vec![0xAB; 128 * 1024];
        let cid = content_id::cid_from_data(&bytes);
        match std::env::var("S3P_CHILD_MODE").unwrap().as_str() {
            "download" => {
                store
                    .publish_with(&cid, Some(&blake3::hash(&bytes).to_hex()), |path| {
                        fs::write(path, &bytes)?;
                        Ok(())
                    })
                    .unwrap();
            }
            "map" => {
                store.put(&cid, &bytes).unwrap();
                let file = store.open(&cid).unwrap().unwrap();
                // This child is the only inode writer; publication must not
                // modify it, including through a second store instance.
                let mapped = unsafe { memmap2::Mmap::map(&file).unwrap() };
                let writer_root = root.clone();
                let writer_bytes = bytes.clone();
                let writer_cid = cid.clone();
                let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
                let writer_barrier = barrier.clone();
                let writer = std::thread::spawn(move || {
                    let second = PublicationStore::new(writer_root, ".chunk").unwrap();
                    writer_barrier.wait();
                    for _ in 0..32 {
                        second.put(&writer_cid, &writer_bytes).unwrap();
                    }
                });
                barrier.wait();
                while !writer.is_finished() {
                    assert_eq!(&*mapped, bytes);
                }
                writer.join().unwrap();
                let second = PublicationStore::new(root, ".chunk").unwrap();
                assert_eq!(&*mapped, bytes);
                assert!(second.delete(&cid).unwrap());
                second.put(&cid, &bytes).unwrap();
                assert_eq!(&*mapped, bytes);
                assert_eq!(second.get(&cid).unwrap(), bytes);
            }
            _ => store.put(&cid, &bytes).unwrap(),
        }
    }

    #[test]
    fn concurrent_process_puts_and_downloads_publish_one_complete_inode() {
        let dir = tempfile::tempdir().unwrap();
        let mut children: Vec<_> = (0..8)
            .map(|index| {
                child(
                    dir.path(),
                    if index % 2 == 0 { "put" } else { "download" },
                    None,
                )
            })
            .collect();
        for child in &mut children {
            assert!(child.wait().unwrap().success());
        }
        let bytes = vec![0xAB; 128 * 1024];
        let cid = content_id::cid_from_data(&bytes);
        assert_eq!(store(dir.path()).get(&cid).unwrap(), bytes);
        assert_eq!(residue_count(dir.path()), 0);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn mapped_inode_survives_duplicate_and_delete_republication_in_child() {
        let dir = tempfile::tempdir().unwrap();
        assert!(child(dir.path(), "map", None).wait().unwrap().success());
    }

    #[test]
    fn process_crashes_leave_only_complete_final_files_and_owned_residue() {
        for (point, visible) in [("link", false), ("after_link", true)] {
            let dir = tempfile::tempdir().unwrap();
            assert_eq!(
                child(dir.path(), "put", Some(point)).wait().unwrap().code(),
                Some(73)
            );
            let bytes = vec![0xAB; 128 * 1024];
            let cid = content_id::cid_from_data(&bytes);
            let store = store(dir.path());
            assert_eq!(store.has(&cid), visible);
            if visible {
                assert_eq!(store.get(&cid).unwrap(), bytes);
            }
            assert_eq!(residue_count(dir.path()), 1);
            store.put(&cid, &bytes).unwrap();
            assert!(store.verify_existing(&cid, None).unwrap());
            // The new attempt must never clean another attempt's residue.
            assert_eq!(residue_count(dir.path()), 1);
        }
    }

    #[test]
    fn write_file_sync_and_link_failures_leave_no_final_or_live_temp() {
        for point in ["write", "file_sync", "link"] {
            let dir = tempfile::tempdir().unwrap();
            let store = store(dir.path());
            *store.fault.lock().unwrap() = Some(point);
            let cid = content_id::cid_from_data(b"content");
            assert!(store.put(&cid, b"content").is_err(), "{point}");
            assert!(!store.has(&cid));
            assert_eq!(residue_count(dir.path()), 0);
        }
    }

    #[test]
    fn duplicate_retries_establish_file_and_directory_durability() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let cid = content_id::cid_from_data(b"content");
        *store.fault.lock().unwrap() = Some("directory_sync");
        assert!(matches!(
            store.put(&cid, b"content"),
            Err(StoreError::DurabilityUnconfirmed { .. })
        ));
        assert_eq!(store.get(&cid).unwrap(), b"content");
        assert!(matches!(
            store.put(&cid, b"content"),
            Err(StoreError::DurabilityUnconfirmed { .. })
        ));
        *store.fault.lock().unwrap() = Some("file_sync");
        assert!(store.verify_existing(&cid, None).is_err());
        *store.fault.lock().unwrap() = None;
        store.put(&cid, b"content").unwrap();
        assert!(store.verify_existing(&cid, None).unwrap());
        assert_eq!(residue_count(dir.path()), 0);
    }

    #[test]
    fn cleanup_failure_preserves_published_inode_and_owned_residue() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let cid = content_id::cid_from_data(b"content");
        *store.fault.lock().unwrap() = Some("cleanup");
        store.put(&cid, b"content").unwrap();
        assert_eq!(store.get(&cid).unwrap(), b"content");
        assert_eq!(residue_count(dir.path()), 1);
    }

    #[test]
    fn failed_partial_download_cleans_only_current_attempt() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let other = dir.path().join(".publish-other-active-attempt");
        fs::create_dir(&other).unwrap();
        fs::write(other.join("payload"), b"other").unwrap();
        let cid = content_id::cid_from_data(b"whole");
        let result = store.publish_with(&cid, None, |path| {
            fs::write(path, b"part")?;
            Err(io::Error::other("download interrupted").into())
        });
        assert!(result.is_err());
        assert!(!store.has(&cid));
        assert_eq!(residue_count(dir.path()), 1);
        assert_eq!(fs::read(other.join("payload")).unwrap(), b"other");
    }

    #[test]
    fn wrong_incoming_or_existing_content_never_replaces_final_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let cid = content_id::cid_from_data(b"right");
        assert!(matches!(
            store.put(&cid, b"wrong"),
            Err(StoreError::IntegrityMismatch { .. })
        ));
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
        fs::write(store.path(&cid).unwrap(), b"corrupt existing").unwrap();
        assert!(matches!(
            store.put(&cid, b"right"),
            Err(StoreError::IntegrityMismatch { .. })
        ));
        assert_eq!(store.get(&cid).unwrap(), b"corrupt existing");
        assert_eq!(residue_count(dir.path()), 0);
    }

    #[test]
    fn downloaded_bytes_must_match_both_hash_and_cid() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let cid = content_id::cid_from_data(b"right");
        let right_hash = blake3::hash(b"right").to_hex().to_string();
        for (payload, hash) in [
            (b"wrong".as_slice(), right_hash.as_str()),
            (b"right".as_slice(), "wrong hash"),
        ] {
            let result = store.publish_with(&cid, Some(hash), |path| {
                fs::write(path, payload)?;
                Ok(())
            });
            assert!(matches!(result, Err(StoreError::IntegrityMismatch { .. })));
            assert!(!store.has(&cid));
            assert_eq!(residue_count(dir.path()), 0);
        }
    }

    #[test]
    fn confinement_and_legacy_reads_are_separate_from_new_cid_policy() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        for key in [
            "",
            ".",
            "..",
            "../escape",
            "a/b",
            "a\\b",
            "bad\0key",
            &"a".repeat(250),
        ] {
            assert!(store.path(key).is_err());
            assert!(store.open(key).is_err());
            assert!(store.get(key).is_err());
            assert!(store.delete(key).is_err());
            assert!(store.put(key, b"payload").is_err());
            assert!(!store.has(key));
        }
        for key in [
            "legacy_alias",
            "legacy..dots",
            "unicode-\u{00e9}",
            &"a".repeat(249),
        ] {
            fs::write(store.path(key).unwrap(), b"legacy").unwrap();
            assert_eq!(store.get(key).unwrap(), b"legacy");
            assert!(store.put(key, b"new").is_err());
            assert!(store.delete(key).unwrap());
        }
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn empty_content_and_generated_cids_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        for bytes in [b"".as_slice(), b"small", &[0; 128 * 1024]] {
            let cid = content_id::cid_from_data(bytes);
            store.put(&cid, bytes).unwrap();
            store.put(&cid, bytes).unwrap();
            assert_eq!(store.get(&cid).unwrap(), bytes);
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_directories_and_fifos_are_not_store_files() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let outside = tempfile::NamedTempFile::new().unwrap();
        fs::write(outside.path(), b"untouched").unwrap();
        let cid = content_id::cid_from_data(b"right");
        let destination = store.path(&cid).unwrap();
        symlink(outside.path(), &destination).unwrap();
        assert!(store.open(&cid).is_err());
        assert!(store.put(&cid, b"right").is_err());
        assert!(store.delete(&cid).is_err());
        assert_eq!(fs::read(outside.path()).unwrap(), b"untouched");
        fs::remove_file(&destination).unwrap();
        fs::create_dir(&destination).unwrap();
        assert!(store.open(&cid).is_err());
        assert!(store.put(&cid, b"right").is_err());
        fs::remove_dir(&destination).unwrap();
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&destination)
                .status()
                .unwrap()
                .success()
        );
        assert!(store.open(&cid).is_err());
        assert!(store.put(&cid, b"right").is_err());
    }
}
