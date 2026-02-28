//! Zero-copy GGUF v2/v3 parser.
//!
//! The parser works on an `&[u8]` slice (typically backed by an mmap).
//! It reads the file header, all metadata key-value pairs, and tensor info
//! entries **without ever touching the raw tensor data bytes**.  This means
//! a multi-gigabyte GGUF file can be fully parsed with only a few kilobytes
//! of heap allocation (for the metadata strings and tensor names).
//!
//! # Wire format (little-endian throughout)
//!
//! ```text
//! [4B magic "GGUF"]
//! [4B version u32]
//! [8B tensor_count u64]
//! [8B metadata_kv_count u64]
//! [metadata_kv_count × MetadataKV]
//! [tensor_count × TensorInfo]
//! [alignment padding to `general.alignment` (default 32)]
//! [tensor data ...]
//! ```

use crate::error::{Result, StoreError};

// ── Public types ──────────────────────────────────────────────────────────────

/// Parsed GGUF file header.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

/// A single metadata key-value entry.
#[derive(Debug, Clone)]
pub struct MetadataKv {
    pub key: String,
    pub value: MetadataValue,
}

/// Metadata value — mirrors the GGUF spec's 13 value types.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            _ => None,
        }
    }
}

/// Parsed tensor info entry (no tensor data is read).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dimensions: u32,
    pub dimensions: Vec<u64>,
    /// Raw GGML type discriminant (see [`omni_types::model::GgmlType`]).
    pub ggml_type: u32,
    /// Byte offset of this tensor's data **relative to the start of the
    /// tensor data section** (i.e. relative to [`GgufFile::tensor_data_offset`]).
    pub offset: u64,
}

/// Complete parsed representation of a GGUF file.
///
/// Holds only the header, metadata, and tensor descriptors — the raw tensor
/// bytes remain untouched in the underlying mmap / buffer.
#[derive(Debug)]
pub struct GgufFile {
    pub header: GgufHeader,
    pub metadata: Vec<MetadataKv>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute byte offset where tensor data begins in the file.
    pub tensor_data_offset: u64,
}

impl GgufFile {
    /// Look up a metadata value by key.
    pub fn metadata_value(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata
            .iter()
            .find(|kv| kv.key == key)
            .map(|kv| &kv.value)
    }

    /// `general.architecture` (e.g. `"llama"`).
    pub fn architecture(&self) -> Option<&str> {
        self.metadata_value("general.architecture")?.as_string()
    }

    /// `general.name` (e.g. `"LLaMA v2"`).
    pub fn model_name(&self) -> Option<&str> {
        self.metadata_value("general.name")?.as_string()
    }

    /// `general.file_type` — GGML file-type code indicating quantization.
    pub fn file_type(&self) -> Option<u32> {
        self.metadata_value("general.file_type")?.as_u32()
    }

    /// Number of transformer blocks, read from `{arch}.block_count`.
    pub fn block_count(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.metadata_value(&format!("{arch}.block_count"))?.as_u32()
    }
}

// ── Top-level parse function ──────────────────────────────────────────────────

/// Parse a GGUF file from a byte slice (typically an mmap).
///
/// This is the only entry point.  It validates the magic number and version,
/// then reads all metadata and tensor info entries.  **No tensor data bytes
/// are accessed.**
pub fn parse_gguf(data: &[u8]) -> Result<GgufFile> {
    let mut cursor: usize = 0;

    // ── Magic ────────────────────────────────────────────────────────────
    ensure(data, cursor, 4)?;
    if &data[cursor..cursor + 4] != b"GGUF" {
        return Err(StoreError::GgufParse("invalid GGUF magic".into()));
    }
    cursor += 4;

    // ── Header ───────────────────────────────────────────────────────────
    let version = read_u32_le(data, &mut cursor)?;
    if !(2..=3).contains(&version) {
        return Err(StoreError::GgufParse(format!(
            "unsupported GGUF version {version} (expected 2 or 3)"
        )));
    }
    let tensor_count = read_u64_le(data, &mut cursor)?;
    let metadata_kv_count = read_u64_le(data, &mut cursor)?;

    // ── Metadata KV pairs ────────────────────────────────────────────────
    let mut metadata = Vec::with_capacity(metadata_kv_count as usize);
    for _ in 0..metadata_kv_count {
        metadata.push(read_metadata_kv(data, &mut cursor)?);
    }

    // ── Tensor info entries ──────────────────────────────────────────────
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        tensors.push(read_tensor_info(data, &mut cursor)?);
    }

    // ── Alignment padding ────────────────────────────────────────────────
    let alignment = find_alignment(&metadata);
    let tensor_data_offset = align_offset(cursor, alignment);

    Ok(GgufFile {
        header: GgufHeader {
            version,
            tensor_count,
            metadata_kv_count,
        },
        metadata,
        tensors,
        tensor_data_offset: tensor_data_offset as u64,
    })
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Bounds check: ensure at least `n` bytes remain from `cursor`.
fn ensure(data: &[u8], cursor: usize, n: usize) -> Result<()> {
    if cursor + n > data.len() {
        Err(StoreError::GgufParse(format!(
            "unexpected EOF at offset {cursor} (need {n} bytes, have {})",
            data.len().saturating_sub(cursor)
        )))
    } else {
        Ok(())
    }
}

fn read_u8(data: &[u8], cursor: &mut usize) -> Result<u8> {
    ensure(data, *cursor, 1)?;
    let v = data[*cursor];
    *cursor += 1;
    Ok(v)
}

fn read_i8(data: &[u8], cursor: &mut usize) -> Result<i8> {
    Ok(read_u8(data, cursor)? as i8)
}

fn read_u16_le(data: &[u8], cursor: &mut usize) -> Result<u16> {
    ensure(data, *cursor, 2)?;
    let v = u16::from_le_bytes(data[*cursor..*cursor + 2].try_into().unwrap());
    *cursor += 2;
    Ok(v)
}

fn read_i16_le(data: &[u8], cursor: &mut usize) -> Result<i16> {
    ensure(data, *cursor, 2)?;
    let v = i16::from_le_bytes(data[*cursor..*cursor + 2].try_into().unwrap());
    *cursor += 2;
    Ok(v)
}

fn read_u32_le(data: &[u8], cursor: &mut usize) -> Result<u32> {
    ensure(data, *cursor, 4)?;
    let v = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

fn read_i32_le(data: &[u8], cursor: &mut usize) -> Result<i32> {
    ensure(data, *cursor, 4)?;
    let v = i32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

fn read_u64_le(data: &[u8], cursor: &mut usize) -> Result<u64> {
    ensure(data, *cursor, 8)?;
    let v = u64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(v)
}

fn read_i64_le(data: &[u8], cursor: &mut usize) -> Result<i64> {
    ensure(data, *cursor, 8)?;
    let v = i64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(v)
}

fn read_f32_le(data: &[u8], cursor: &mut usize) -> Result<f32> {
    ensure(data, *cursor, 4)?;
    let v = f32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

fn read_f64_le(data: &[u8], cursor: &mut usize) -> Result<f64> {
    ensure(data, *cursor, 8)?;
    let v = f64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(v)
}

/// Read a GGUF string: `[u64 LE length][UTF-8 bytes]`.
fn read_gguf_string(data: &[u8], cursor: &mut usize) -> Result<String> {
    let len = read_u64_le(data, cursor)? as usize;
    ensure(data, *cursor, len)?;
    let s = std::str::from_utf8(&data[*cursor..*cursor + len])
        .map_err(|e| StoreError::GgufParse(format!("invalid UTF-8: {e}")))?;
    *cursor += len;
    Ok(s.to_string())
}

/// Read one metadata key-value pair.
fn read_metadata_kv(data: &[u8], cursor: &mut usize) -> Result<MetadataKv> {
    let key = read_gguf_string(data, cursor)?;
    let value_type = read_u32_le(data, cursor)?;
    let value = read_metadata_value(data, cursor, value_type)?;
    Ok(MetadataKv { key, value })
}

/// Read a typed metadata value.
fn read_metadata_value(
    data: &[u8],
    cursor: &mut usize,
    value_type: u32,
) -> Result<MetadataValue> {
    match value_type {
        0  => Ok(MetadataValue::Uint8(read_u8(data, cursor)?)),
        1  => Ok(MetadataValue::Int8(read_i8(data, cursor)?)),
        2  => Ok(MetadataValue::Uint16(read_u16_le(data, cursor)?)),
        3  => Ok(MetadataValue::Int16(read_i16_le(data, cursor)?)),
        4  => Ok(MetadataValue::Uint32(read_u32_le(data, cursor)?)),
        5  => Ok(MetadataValue::Int32(read_i32_le(data, cursor)?)),
        6  => Ok(MetadataValue::Float32(read_f32_le(data, cursor)?)),
        7  => Ok(MetadataValue::Bool(read_u8(data, cursor)? != 0)),
        8  => Ok(MetadataValue::String(read_gguf_string(data, cursor)?)),
        9  => {
            // Array: [u32 element_type][u64 count][elements...]
            let elem_type = read_u32_le(data, cursor)?;
            let count = read_u64_le(data, cursor)?;
            let mut elems = Vec::with_capacity(count as usize);
            for _ in 0..count {
                elems.push(read_metadata_value(data, cursor, elem_type)?);
            }
            Ok(MetadataValue::Array(elems))
        }
        10 => Ok(MetadataValue::Uint64(read_u64_le(data, cursor)?)),
        11 => Ok(MetadataValue::Int64(read_i64_le(data, cursor)?)),
        12 => Ok(MetadataValue::Float64(read_f64_le(data, cursor)?)),
        _  => Err(StoreError::GgufParse(format!(
            "unknown metadata value type: {value_type}"
        ))),
    }
}

/// Read one tensor info entry.
fn read_tensor_info(data: &[u8], cursor: &mut usize) -> Result<TensorInfo> {
    let name = read_gguf_string(data, cursor)?;
    let n_dimensions = read_u32_le(data, cursor)?;
    let mut dimensions = Vec::with_capacity(n_dimensions as usize);
    for _ in 0..n_dimensions {
        dimensions.push(read_u64_le(data, cursor)?);
    }
    let ggml_type = read_u32_le(data, cursor)?;
    let offset = read_u64_le(data, cursor)?;
    Ok(TensorInfo {
        name,
        n_dimensions,
        dimensions,
        ggml_type,
        offset,
    })
}

/// Extract `general.alignment` from metadata, defaulting to 32.
fn find_alignment(metadata: &[MetadataKv]) -> usize {
    for kv in metadata {
        if kv.key == "general.alignment" {
            if let MetadataValue::Uint32(v) = &kv.value {
                return *v as usize;
            }
        }
    }
    32
}

/// Round `offset` up to the next multiple of `alignment`.
fn align_offset(offset: usize, alignment: usize) -> usize {
    let rem = offset % alignment;
    if rem == 0 {
        offset
    } else {
        offset + alignment - rem
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: write a GGUF-format string into a buffer.
    fn write_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Build a minimal valid GGUF v3 file in memory.
    fn make_test_gguf() -> Vec<u8> {
        let mut buf = Vec::new();

        // ── Header ───────────────────────────────────────────────────────
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());  // version
        buf.extend_from_slice(&1u64.to_le_bytes());  // tensor_count
        buf.extend_from_slice(&3u64.to_le_bytes());  // metadata_kv_count

        // ── Metadata KV 1: general.architecture = "llama" ────────────────
        write_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes());   // STRING
        write_str(&mut buf, "llama");

        // ── Metadata KV 2: general.name = "test-model" ──────────────────
        write_str(&mut buf, "general.name");
        buf.extend_from_slice(&8u32.to_le_bytes());   // STRING
        write_str(&mut buf, "test-model");

        // ── Metadata KV 3: llama.block_count = 32 ───────────────────────
        write_str(&mut buf, "llama.block_count");
        buf.extend_from_slice(&4u32.to_le_bytes());   // UINT32
        buf.extend_from_slice(&32u32.to_le_bytes());

        // ── Tensor 1: "blk.0.attn_q.weight" 2D [4096, 4096] F16 ────────
        write_str(&mut buf, "blk.0.attn_q.weight");
        buf.extend_from_slice(&2u32.to_le_bytes());   // n_dimensions
        buf.extend_from_slice(&4096u64.to_le_bytes());
        buf.extend_from_slice(&4096u64.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());   // F16
        buf.extend_from_slice(&0u64.to_le_bytes());   // offset

        // ── Alignment padding (to 32 bytes) ──────────────────────────────
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        // ── Fake tensor data ─────────────────────────────────────────────
        buf.extend_from_slice(&[0xAB; 128]);

        buf
    }

    #[test]
    fn parse_minimal_gguf() {
        let data = make_test_gguf();
        let gguf = parse_gguf(&data).unwrap();

        // Header
        assert_eq!(gguf.header.version, 3);
        assert_eq!(gguf.header.tensor_count, 1);
        assert_eq!(gguf.header.metadata_kv_count, 3);

        // Metadata
        assert_eq!(gguf.architecture(), Some("llama"));
        assert_eq!(gguf.model_name(), Some("test-model"));
        assert_eq!(gguf.block_count(), Some(32));

        // Tensor info
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensors[0].name, "blk.0.attn_q.weight");
        assert_eq!(gguf.tensors[0].n_dimensions, 2);
        assert_eq!(gguf.tensors[0].dimensions, vec![4096, 4096]);
        assert_eq!(gguf.tensors[0].ggml_type, 1); // F16

        // Tensor data offset must be 32-byte aligned
        assert_eq!(gguf.tensor_data_offset % 32, 0);
    }

    #[test]
    fn rejects_bad_magic() {
        let data = b"NOT_A_GGUF_FILE_AT_ALL";
        assert!(parse_gguf(data).is_err());
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&99u32.to_le_bytes());
        data.extend_from_slice(&[0u8; 16]); // tensor_count + metadata_kv_count
        assert!(parse_gguf(&data).is_err());
    }

    #[test]
    fn align_offset_works() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(31, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }
}
