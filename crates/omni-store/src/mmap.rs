//! Thin wrappers around `memmap2` for memory-mapped file I/O.
//!
//! On Apple Silicon the mmap path is particularly efficient because the
//! unified memory architecture avoids an extra copy between GPU and CPU
//! address spaces.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::error::Result;

/// Memory-map an existing file for read-only access.
///
/// # Safety contract
///
/// The caller must ensure that no other process truncates or overwrites the
/// file while the returned [`Mmap`] is alive.  The CID-based naming scheme
/// used by [`crate::store::ShardStore`] guarantees this in practice because
/// shard files are write-once / content-addressed.
pub fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = File::open(path)?;
    // SAFETY: shard files are write-once; see doc-comment above.
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(mmap)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn mmap_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");

        let data = b"hello mmap";
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(data).unwrap();
        }

        let mapped = mmap_file(&path).unwrap();
        assert_eq!(&*mapped, data);
    }
}
