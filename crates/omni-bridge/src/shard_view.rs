//! Zero-copy shard access via the Python buffer protocol.
//!
//! `PyShardView` wraps a `memmap2::Mmap` and implements `__getbuffer__` so
//! Python gets zero-copy access via `memoryview()` or `numpy.frombuffer()`.
//!
//! The `shape` and `strides` arrays are owned by the struct (not by the
//! caller-allocated `Py_buffer`) to guarantee pointer stability.

use std::os::raw::{c_int, c_void};

use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;

#[pyclass(name = "ShardView")]
pub struct PyShardView {
    mmap: memmap2::Mmap,
    cid: String,
    /// Owned backing for `Py_buffer.shape` — must outlive any memoryview.
    shape: [isize; 1],
    /// Owned backing for `Py_buffer.strides` — must outlive any memoryview.
    strides: [isize; 1],
}

impl PyShardView {
    /// Construct from a freshly obtained mmap.
    pub fn new(mmap: memmap2::Mmap, cid: String) -> Self {
        let len = mmap.len() as isize;
        Self {
            mmap,
            cid,
            shape: [len],
            strides: [1],
        }
    }
}

#[pymethods]
impl PyShardView {
    /// Length in bytes.
    fn __len__(&self) -> usize {
        self.mmap.len()
    }

    /// CID of the shard.
    #[getter]
    fn cid(&self) -> &str {
        &self.cid
    }

    /// Size in bytes.
    #[getter]
    fn size(&self) -> usize {
        self.mmap.len()
    }

    /// Python buffer protocol — zero-copy read-only access.
    ///
    /// # Safety
    ///
    /// `view` must be a valid pointer to a `Py_buffer` struct allocated by the
    /// Python runtime. The `obj` field is set to a new reference to `self`,
    /// preventing GC while any `memoryview` is outstanding.
    unsafe fn __getbuffer__(
        slf: PyRefMut<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("null Py_buffer pointer"));
        }

        // Reject writable requests — mmap is read-only.
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("ShardView is read-only"));
        }

        let bytes = &slf.mmap[..];

        unsafe {
            (*view).buf = bytes.as_ptr() as *mut c_void;
            (*view).len = bytes.len() as isize;
            (*view).readonly = 1;
            (*view).itemsize = 1;
            // "B" = unsigned byte (C `unsigned char`)
            (*view).format = c"B".as_ptr() as *mut _;
            (*view).ndim = 1;
            // Point to struct-owned arrays — stable across Py_buffer moves.
            (*view).shape = slf.shape.as_ptr() as *mut isize;
            (*view).strides = slf.strides.as_ptr() as *mut isize;
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
            // Prevent GC of our PyShardView while memoryview is alive.
            (*view).obj = ffi::Py_NewRef(slf.as_ptr());
        }

        Ok(())
    }

    /// Release buffer (no-op — the mmap stays alive until the PyShardView is dropped).
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}

    fn __repr__(&self) -> String {
        format!("ShardView(cid='{}', size={})", self.cid, self.mmap.len())
    }
}
#[cfg(test)]
mod tests {
    //! The view's Rust side: its length, identity and the bytes it maps. The
    //! buffer protocol itself (`memoryview`, read-only refusal, lifetime) needs
    //! an interpreter and is covered by `tests/python/`.

    use super::*;

    #[test]
    fn a_view_reports_the_mapped_file_and_holds_its_bytes() {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("omni-bridge-view-{}-{n}", std::process::id()));
        let content: Vec<u8> = (0u8..=255).collect();
        std::fs::write(&path, &content).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        // SAFETY: the file is private to this test and not modified while mapped.
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let view = PyShardView::new(mmap, "bafy-test".into());
        assert_eq!(
            (view.__len__(), view.size(), view.cid()),
            (256, 256, "bafy-test")
        );
        assert_eq!(&view.mmap[..], &content[..]);
        assert_eq!((view.shape, view.strides), ([256], [1]));
        assert_eq!(view.__repr__(), "ShardView(cid='bafy-test', size=256)");
        drop(view);
        std::fs::remove_file(&path).unwrap();
    }
}
