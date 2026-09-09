//! Zero-copy GGUF v2/v3 parser.
//!
//! The parser works on an `&[u8]` slice (typically backed by an mmap).
//! It reads the file header, all metadata key-value pairs, and tensor info
//! entries **without ever touching the raw tensor data bytes**. Heap use grows
//! with decoded metadata and tensor descriptors, including container capacity;
//! untrusted declared counts do not determine up-front allocations.
//!
//! Nested-array depth and tensor payload layout are not bounded/validated here.
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
        self.metadata_value(&format!("{arch}.block_count"))?
            .as_u32()
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
    // Empty key length (8), value tag (4), and smallest scalar payload (1).
    let metadata_count = checked_count(data, cursor, metadata_kv_count, 8 + 4 + 1)?;
    let mut metadata = Vec::new();
    for _ in 0..metadata_count {
        let kv = read_metadata_kv(data, &mut cursor)?;
        push_fallible(&mut metadata, kv)?;
    }

    // ── Tensor info entries ──────────────────────────────────────────────
    // Empty name length (8), dimension count (4), type (4), and offset (8).
    let tensors_count = checked_count(data, cursor, tensor_count, 8 + 4 + 4 + 8)?;
    let mut tensors = Vec::new();
    for _ in 0..tensors_count {
        let tensor = read_tensor_info(data, &mut cursor)?;
        push_fallible(&mut tensors, tensor)?;
    }

    // ── Alignment padding ────────────────────────────────────────────────
    let alignment = find_alignment(&metadata);
    let tensor_data_offset = align_offset(cursor, alignment)?;

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
    if cursor.checked_add(n).is_none_or(|end| end > data.len()) {
        Err(StoreError::GgufParse(format!(
            "unexpected EOF at offset {cursor} (need {n} bytes, have {})",
            data.len().saturating_sub(cursor)
        )))
    } else {
        Ok(())
    }
}

/// Validate a wire count before narrowing, using a positive lower bound on
/// the bytes consumed by each encoded item. This does not reserve that count:
/// containers grow only after an item has actually been decoded.
fn checked_count(data: &[u8], cursor: usize, count: u64, min_size: usize) -> Result<usize> {
    ensure(data, cursor, 0)?;
    let max_count = (data.len() - cursor) / min_size;
    if count > max_count as u64 {
        return Err(StoreError::GgufParse(format!(
            "count {count} exceeds remaining input at offset {cursor}"
        )));
    }
    usize::try_from(count).map_err(|_| StoreError::GgufParse("count does not fit in usize".into()))
}

fn reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<()> {
    values
        .try_reserve(additional)
        .map_err(|e| StoreError::GgufParse(format!("cannot allocate parsed values: {e}")))
}

fn push_fallible<T>(values: &mut Vec<T>, value: T) -> Result<()> {
    reserve(values, 1)?;
    // try_reserve guarantees capacity for this push without another allocation.
    values.push(value);
    Ok(())
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
    let wire_len = read_u64_le(data, cursor)?;
    let len = checked_count(data, *cursor, wire_len, 1)?;
    let s = std::str::from_utf8(&data[*cursor..*cursor + len])
        .map_err(|e| StoreError::GgufParse(format!("invalid UTF-8: {e}")))?;
    let mut owned = String::new();
    owned
        .try_reserve_exact(len)
        .map_err(|e| StoreError::GgufParse(format!("cannot allocate parsed string: {e}")))?;
    owned.push_str(s);
    *cursor += len;
    Ok(owned)
}

/// Read one metadata key-value pair.
fn read_metadata_kv(data: &[u8], cursor: &mut usize) -> Result<MetadataKv> {
    let key = read_gguf_string(data, cursor)?;
    let value_type = read_u32_le(data, cursor)?;
    let value = read_metadata_value(data, cursor, value_type)?;
    Ok(MetadataKv { key, value })
}

/// Read a typed metadata value.
fn read_metadata_value(data: &[u8], cursor: &mut usize, value_type: u32) -> Result<MetadataValue> {
    match value_type {
        0 => Ok(MetadataValue::Uint8(read_u8(data, cursor)?)),
        1 => Ok(MetadataValue::Int8(read_i8(data, cursor)?)),
        2 => Ok(MetadataValue::Uint16(read_u16_le(data, cursor)?)),
        3 => Ok(MetadataValue::Int16(read_i16_le(data, cursor)?)),
        4 => Ok(MetadataValue::Uint32(read_u32_le(data, cursor)?)),
        5 => Ok(MetadataValue::Int32(read_i32_le(data, cursor)?)),
        6 => Ok(MetadataValue::Float32(read_f32_le(data, cursor)?)),
        7 => Ok(MetadataValue::Bool(read_u8(data, cursor)? != 0)),
        8 => Ok(MetadataValue::String(read_gguf_string(data, cursor)?)),
        9 => {
            // Array: [u32 element_type][u64 count][elements...]
            let elem_type = read_u32_le(data, cursor)?;
            let count = read_u64_le(data, cursor)?;
            // Empty arrays have no encoded element to validate; preserve the
            // existing acceptance of any element tag when count is zero.
            let min_size = match elem_type {
                0 | 1 | 7 => 1,
                2 | 3 => 2,
                4..=6 => 4,
                8 | 10 | 11 | 12 => 8,
                9 => 12,
                _ if count == 0 => 1,
                _ => {
                    return Err(StoreError::GgufParse(format!(
                        "unknown metadata value type: {elem_type}"
                    )))
                }
            };
            let count = checked_count(data, *cursor, count, min_size)?;
            let mut elems = Vec::new();
            for _ in 0..count {
                let value = read_metadata_value(data, cursor, elem_type)?;
                push_fallible(&mut elems, value)?;
            }
            Ok(MetadataValue::Array(elems))
        }
        10 => Ok(MetadataValue::Uint64(read_u64_le(data, cursor)?)),
        11 => Ok(MetadataValue::Int64(read_i64_le(data, cursor)?)),
        12 => Ok(MetadataValue::Float64(read_f64_le(data, cursor)?)),
        _ => Err(StoreError::GgufParse(format!(
            "unknown metadata value type: {value_type}"
        ))),
    }
}

/// Read one tensor info entry.
fn read_tensor_info(data: &[u8], cursor: &mut usize) -> Result<TensorInfo> {
    let name = read_gguf_string(data, cursor)?;
    let n_dimensions = read_u32_le(data, cursor)?;
    let count = checked_count(data, *cursor, u64::from(n_dimensions), 8)?;
    let mut dimensions = Vec::new();
    for _ in 0..count {
        let dimension = read_u64_le(data, cursor)?;
        push_fallible(&mut dimensions, dimension)?;
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
fn align_offset(offset: usize, alignment: usize) -> Result<usize> {
    if alignment == 0 {
        return Err(StoreError::GgufParse("alignment must be positive".into()));
    }
    let rem = offset % alignment;
    let padding = if rem == 0 { 0 } else { alignment - rem };
    offset
        .checked_add(padding)
        .ok_or_else(|| StoreError::GgufParse("aligned offset overflows usize".into()))
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
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&3u64.to_le_bytes()); // metadata_kv_count

        // ── Metadata KV 1: general.architecture = "llama" ────────────────
        write_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        write_str(&mut buf, "llama");

        // ── Metadata KV 2: general.name = "test-model" ──────────────────
        write_str(&mut buf, "general.name");
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        write_str(&mut buf, "test-model");

        // ── Metadata KV 3: llama.block_count = 32 ───────────────────────
        write_str(&mut buf, "llama.block_count");
        buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
        buf.extend_from_slice(&32u32.to_le_bytes());

        // ── Tensor 1: "blk.0.attn_q.weight" 2D [4096, 4096] F16 ────────
        write_str(&mut buf, "blk.0.attn_q.weight");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dimensions
        buf.extend_from_slice(&4096u64.to_le_bytes());
        buf.extend_from_slice(&4096u64.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // F16
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

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
        assert_eq!(align_offset(0, 32).unwrap(), 0);
        assert_eq!(align_offset(1, 32).unwrap(), 32);
        assert_eq!(align_offset(31, 32).unwrap(), 32);
        assert_eq!(align_offset(32, 32).unwrap(), 32);
        assert_eq!(align_offset(33, 32).unwrap(), 64);
    }

    fn header(version: u32, tensor_count: u64, metadata_count: u64) -> Vec<u8> {
        let mut raw = b"GGUF".to_vec();
        raw.extend_from_slice(&version.to_le_bytes());
        raw.extend_from_slice(&tensor_count.to_le_bytes());
        raw.extend_from_slice(&metadata_count.to_le_bytes());
        raw
    }

    fn metadata_entry(raw: &mut Vec<u8>, key: &str, tag: u32, body: &[u8]) {
        write_str(raw, key);
        raw.extend_from_slice(&tag.to_le_bytes());
        raw.extend_from_slice(body);
    }

    fn one_value(tag: u32, body: &[u8]) -> Vec<u8> {
        let mut raw = header(3, 0, 1);
        metadata_entry(&mut raw, "value", tag, body);
        raw
    }

    fn array_body(tag: u32, count: u64, body: &[u8]) -> Vec<u8> {
        let mut raw = tag.to_le_bytes().to_vec();
        raw.extend_from_slice(&count.to_le_bytes());
        raw.extend_from_slice(body);
        raw
    }

    fn assert_parse_error<T: std::fmt::Debug>(result: Result<T>) {
        assert!(
            matches!(result, Err(StoreError::GgufParse(_))),
            "{result:?}"
        );
    }

    fn assert_same_value(actual: &MetadataValue, expected: &MetadataValue) {
        use MetadataValue::*;
        match (actual, expected) {
            (Uint8(a), Uint8(b)) => assert_eq!(a, b),
            (Int8(a), Int8(b)) => assert_eq!(a, b),
            (Uint16(a), Uint16(b)) => assert_eq!(a, b),
            (Int16(a), Int16(b)) => assert_eq!(a, b),
            (Uint32(a), Uint32(b)) => assert_eq!(a, b),
            (Int32(a), Int32(b)) => assert_eq!(a, b),
            (Float32(a), Float32(b)) => assert_eq!(a.to_bits(), b.to_bits()),
            (Bool(a), Bool(b)) => assert_eq!(a, b),
            (String(a), String(b)) => assert_eq!(a, b),
            (Array(a), Array(b)) => {
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(b) {
                    assert_same_value(a, b);
                }
            }
            (Uint64(a), Uint64(b)) => assert_eq!(a, b),
            (Int64(a), Int64(b)) => assert_eq!(a, b),
            (Float64(a), Float64(b)) => assert_eq!(a.to_bits(), b.to_bits()),
            _ => panic!("value variants differ: {actual:?}, {expected:?}"),
        }
    }

    #[test]
    fn scalar_and_array_values_preserve_bits_and_order() {
        let mut string = Vec::new();
        write_str(&mut string, "λ\0value");
        let cases = vec![
            (0, vec![255], MetadataValue::Uint8(255)),
            (1, vec![128], MetadataValue::Int8(-128)),
            (
                2,
                u16::MAX.to_le_bytes().to_vec(),
                MetadataValue::Uint16(u16::MAX),
            ),
            (
                3,
                i16::MIN.to_le_bytes().to_vec(),
                MetadataValue::Int16(i16::MIN),
            ),
            (
                4,
                u32::MAX.to_le_bytes().to_vec(),
                MetadataValue::Uint32(u32::MAX),
            ),
            (
                5,
                i32::MIN.to_le_bytes().to_vec(),
                MetadataValue::Int32(i32::MIN),
            ),
            (
                6,
                0x7fc0_0123u32.to_le_bytes().to_vec(),
                MetadataValue::Float32(f32::from_bits(0x7fc0_0123)),
            ),
            (7, vec![255], MetadataValue::Bool(true)),
            (8, string, MetadataValue::String("λ\0value".into())),
            (
                10,
                u64::MAX.to_le_bytes().to_vec(),
                MetadataValue::Uint64(u64::MAX),
            ),
            (
                11,
                i64::MIN.to_le_bytes().to_vec(),
                MetadataValue::Int64(i64::MIN),
            ),
            (
                12,
                0x8000_0000_0000_0000u64.to_le_bytes().to_vec(),
                MetadataValue::Float64(-0.0),
            ),
        ];
        for version in [2, 3] {
            let mut raw = header(version, 0, cases.len() as u64 * 2);
            for (tag, body, _) in &cases {
                metadata_entry(&mut raw, &format!("scalar.{tag}"), *tag, body);
                let mut two = body.clone();
                two.extend_from_slice(body);
                metadata_entry(
                    &mut raw,
                    &format!("array.{tag}"),
                    9,
                    &array_body(*tag, 2, &two),
                );
            }
            let parsed = parse_gguf(&raw).unwrap();
            assert_eq!(parsed.header.version, version);
            assert_eq!(parsed.header.tensor_count, 0);
            assert_eq!(parsed.header.metadata_kv_count, cases.len() as u64 * 2);
            assert_eq!(parsed.metadata.len(), cases.len() * 2);
            for (i, (tag, _, value)) in cases.iter().enumerate() {
                assert_eq!(parsed.metadata[i * 2].key, format!("scalar.{tag}"));
                assert_same_value(&parsed.metadata[i * 2].value, value);
                assert_eq!(parsed.metadata[i * 2 + 1].key, format!("array.{tag}"));
                assert_same_value(
                    &parsed.metadata[i * 2 + 1].value,
                    &MetadataValue::Array(vec![value.clone(), value.clone()]),
                );
            }
        }
    }

    #[test]
    fn empty_values_and_existing_permissive_cases_are_preserved() {
        for version in [2, 3] {
            let parsed = parse_gguf(&header(version, 0, 0)).unwrap();
            assert!(parsed.metadata.is_empty());
            assert!(parsed.tensors.is_empty());
            // Checking physical tensor-layout/padding validity is a separate change.
            assert_eq!(parsed.tensor_data_offset, 32);
        }
        let parsed = parse_gguf(&one_value(8, &0u64.to_le_bytes())).unwrap();
        assert_same_value(
            &parsed.metadata[0].value,
            &MetadataValue::String(String::new()),
        );
        for tag in (0..=12).chain([u32::MAX]) {
            let parsed = parse_gguf(&one_value(9, &array_body(tag, 0, &[]))).unwrap();
            assert_same_value(&parsed.metadata[0].value, &MetadataValue::Array(vec![]));
        }
        let mut raw = header(3, 1, 1);
        metadata_entry(&mut raw, "", 7, &[0]);
        write_str(&mut raw, "");
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&u32::MAX.to_le_bytes());
        raw.extend_from_slice(&u64::MAX.to_le_bytes());
        let parsed = parse_gguf(&raw).unwrap();
        assert_eq!(parsed.metadata[0].key, "");
        assert_same_value(&parsed.metadata[0].value, &MetadataValue::Bool(false));
        assert_eq!(parsed.tensors[0].name, "");
        assert_eq!(parsed.tensors[0].n_dimensions, 0);
        assert!(parsed.tensors[0].dimensions.is_empty());
        assert_eq!(parsed.tensors[0].ggml_type, u32::MAX);
        assert_eq!(parsed.tensors[0].offset, u64::MAX);
    }

    #[test]
    fn shallow_nested_arrays_remain_supported() {
        let inner = array_body(0, 3, &[7, 2, 9]);
        let outer = array_body(9, 2, &[inner.clone(), inner].concat());
        let parsed = parse_gguf(&one_value(9, &outer)).unwrap();
        let expected = MetadataValue::Array(vec![
            MetadataValue::Uint8(7),
            MetadataValue::Uint8(2),
            MetadataValue::Uint8(9),
        ]);
        assert_same_value(
            &parsed.metadata[0].value,
            &MetadataValue::Array(vec![expected.clone(), expected]),
        );
    }

    #[test]
    fn max_counts_in_minimal_headers_return_errors() {
        assert_parse_error(parse_gguf(&header(3, 0, u64::MAX)));
        assert_parse_error(parse_gguf(&header(3, u64::MAX, 0)));
    }

    #[test]
    fn oversized_top_level_counts_are_errors() {
        for count in [1, u32::MAX as u64, u32::MAX as u64 + 2, u64::MAX] {
            assert_parse_error(parse_gguf(&header(3, 0, count)));
            assert_parse_error(parse_gguf(&header(3, count, 0)));
        }
    }

    #[test]
    fn max_array_count_returns_error() {
        assert_parse_error(parse_gguf(&one_value(9, &array_body(0, u64::MAX, &[]))));
    }

    #[test]
    fn oversized_array_counts_are_errors_for_every_element_type() {
        for tag in 0..=12 {
            for count in [1, u32::MAX as u64, u32::MAX as u64 + 2, u64::MAX] {
                assert_parse_error(parse_gguf(&one_value(9, &array_body(tag, count, &[]))));
            }
        }
    }

    #[test]
    fn oversized_strings_and_dimensions_are_errors() {
        for count in [u32::MAX as u64, u32::MAX as u64 + 2, u64::MAX] {
            let mut body = count.to_le_bytes().to_vec();
            body.extend_from_slice(b"small");
            assert_parse_error(parse_gguf(&one_value(8, &body)));
            let mut key = header(3, 0, 1);
            key.extend_from_slice(&body);
            key.extend_from_slice(&[0; 16]);
            assert_parse_error(parse_gguf(&key));
            let mut tensor = header(3, 1, 0);
            tensor.extend_from_slice(&body);
            tensor.extend_from_slice(&[0; 24]);
            assert_parse_error(parse_gguf(&tensor));
        }
        let mut raw = header(3, 1, 0);
        write_str(&mut raw, "tensor");
        raw.extend_from_slice(&u32::MAX.to_le_bytes());
        raw.extend_from_slice(&[0; 12]);
        assert_parse_error(parse_gguf(&raw));
    }

    #[test]
    fn unknown_types_and_invalid_utf8_are_errors() {
        assert_parse_error(parse_gguf(&one_value(u32::MAX, &[0])));
        assert_parse_error(parse_gguf(&one_value(9, &array_body(u32::MAX, 1, &[0]))));
        let mut bad = 1u64.to_le_bytes().to_vec();
        bad.push(0xff);
        assert_parse_error(parse_gguf(&one_value(8, &bad)));
        assert_parse_error(parse_gguf(&one_value(9, &array_body(8, 1, &bad))));
        let mut raw = header(3, 0, 1);
        raw.extend_from_slice(&bad);
        raw.extend_from_slice(&[0; 5]);
        assert_parse_error(parse_gguf(&raw));
    }

    #[test]
    fn cursor_bounds_do_not_wrap() {
        assert!(ensure(&[], 0, 0).is_ok());
        assert!(ensure(&[0; 3], 3, 0).is_ok());
        assert_parse_error(ensure(&[0; 3], 4, 0));
        assert_parse_error(ensure(&[0; 3], usize::MAX, 2));
        assert_parse_error(ensure(&[0; 3], 2, usize::MAX));
        assert_parse_error(checked_count(&[0; 3], usize::MAX, 0, 1));
    }

    #[test]
    fn wire_count_bounds_are_checked_before_narrowing() {
        for min_size in [1, 2, 4, 8, 12, 13, 24] {
            let data = vec![0; min_size * 3 + 1];
            assert_eq!(checked_count(&data, 1, 3, min_size).unwrap(), 3);
            assert_parse_error(checked_count(&data, 2, 3, min_size));
            assert_parse_error(checked_count(&data, 1, 4, min_size));
            for count in [u32::MAX as u64, u32::MAX as u64 + 2, u64::MAX] {
                assert_parse_error(checked_count(&data, 0, count, min_size));
            }
        }
    }

    #[test]
    fn capacity_failure_returns_error_without_mutating_values() {
        let mut values = vec![17u64, 42];
        assert_parse_error(reserve(&mut values, usize::MAX));
        assert_eq!(values, [17, 42]);
        push_fallible(&mut values, 99).unwrap();
        assert_eq!(values, [17, 42, 99]);
    }

    #[test]
    fn zero_alignment_returns_error() {
        let mut raw = header(3, 0, 1);
        metadata_entry(&mut raw, "general.alignment", 4, &0u32.to_le_bytes());
        assert_parse_error(parse_gguf(&raw));
        assert_parse_error(align_offset(0, 0));
    }

    #[test]
    fn positive_alignment_policy_is_unchanged() {
        for alignment in [1, 3, 8, 24, 32, u32::MAX] {
            let mut raw = header(3, 0, 1);
            metadata_entry(&mut raw, "general.alignment", 4, &alignment.to_le_bytes());
            let parsed = parse_gguf(&raw).unwrap();
            let alignment = u64::from(alignment);
            let expected = (raw.len() as u64).div_ceil(alignment) * alignment;
            assert_eq!(parsed.tensor_data_offset, expected);
        }
    }

    #[test]
    fn alignment_rounding_does_not_overflow() {
        assert_eq!(align_offset(usize::MAX, 1).unwrap(), usize::MAX);
        assert_eq!(align_offset(usize::MAX - 1, 2).unwrap(), usize::MAX - 1);
        assert_parse_error(align_offset(usize::MAX, 2));
        assert_parse_error(align_offset(usize::MAX - 1, 4));
    }

    // Four tensors, each 16 F16 values = 32 bytes, with aligned contiguous data.
    fn ingestion_fixture(version: u32) -> (Vec<u8>, usize) {
        let mut raw = header(version, 4, 4);
        let mut arch = Vec::new();
        write_str(&mut arch, "llama");
        metadata_entry(&mut raw, "general.architecture", 8, &arch);
        let mut name = Vec::new();
        write_str(&mut name, "compat-model");
        metadata_entry(&mut raw, "general.name", 8, &name);
        metadata_entry(&mut raw, "llama.block_count", 4, &2u32.to_le_bytes());
        metadata_entry(&mut raw, "general.file_type", 4, &1u32.to_le_bytes());
        for (i, name) in [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.1.attn_q.weight",
            "output.weight",
        ]
        .iter()
        .enumerate()
        {
            write_str(&mut raw, name);
            raw.extend_from_slice(&1u32.to_le_bytes());
            raw.extend_from_slice(&16u64.to_le_bytes());
            raw.extend_from_slice(&1u32.to_le_bytes());
            raw.extend_from_slice(&(i as u64 * 32).to_le_bytes());
        }
        let descriptor_end = raw.len();
        while raw.len() % 32 != 0 {
            raw.push(0);
        }
        raw.extend(0..128u8);
        (raw, descriptor_end)
    }

    #[test]
    fn truncation_is_rejected_at_every_descriptor_byte() {
        let (raw, end) = ingestion_fixture(3);
        for cut in 0..end {
            assert_parse_error(parse_gguf(&raw[..cut]));
        }
        // Tensor bytes and physical padding are not validated by this increment.
        for cut in end..=raw.len() {
            assert!(parse_gguf(&raw[..cut]).is_ok(), "cut {cut}");
        }
    }

    #[test]
    fn valid_ingestion_preserves_chunk_plans_and_manifest_bytes() {
        for version in [2, 3] {
            let (raw, _) = ingestion_fixture(version);
            let parsed = parse_gguf(&raw).unwrap();
            let plans = crate::chunker::plan_chunks(&parsed, 1).unwrap();
            assert_eq!(parsed.header.version, version);
            assert_eq!(parsed.header.tensor_count, 4);
            assert_eq!(parsed.header.metadata_kv_count, 4);
            assert_eq!(parsed.architecture(), Some("llama"));
            assert_eq!(parsed.model_name(), Some("compat-model"));
            assert_eq!(parsed.block_count(), Some(2));
            assert_eq!(parsed.file_type(), Some(1));
            for (i, tensor) in parsed.tensors.iter().enumerate() {
                assert_eq!(tensor.n_dimensions, 1);
                assert_eq!(tensor.dimensions, [16]);
                assert_eq!(tensor.ggml_type, 1);
                assert_eq!(tensor.offset, i as u64 * 32);
            }
            assert_eq!(plans.len(), 2);
            assert_eq!(plans[0].shard_index, 0);
            assert_eq!(plans[0].layer_range.start, 0);
            assert_eq!(plans[0].layer_range.end, 0);
            assert!(plans[0].includes_embedding);
            assert!(!plans[0].includes_output_head);
            assert_eq!(plans[0].tensor_indices, [0, 1]);
            assert_eq!(plans[1].shard_index, 1);
            assert_eq!(plans[1].layer_range.start, 1);
            assert_eq!(plans[1].layer_range.end, 1);
            assert!(!plans[1].includes_embedding);
            assert!(plans[1].includes_output_head);
            assert_eq!(plans[1].tensor_indices, [2, 3]);
            let manifest = crate::manifest::build_manifest(&parsed, &raw, &plans).unwrap();
            let mut cbor = Vec::new();
            ciborium::ser::into_writer(&manifest, &mut cbor).unwrap();
            assert_eq!(parsed.tensor_data_offset, 384);
            assert_eq!(
                manifest.shards[0].cid,
                "bafkr4ico5vyud2skltklpcdanpjd6rxcckxzzlhlvtoh2h2mnxd7eui3ta"
            );
            assert_eq!(
                manifest.shards[1].cid,
                "bafkr4ibzpxps4ro7qoupep6rlktrr35qxn6j3ivs4qygtk452atqyn6j5i"
            );
            // Complete serialized bytes captured with the unmodified base parser.
            let expected_hex = match version {
                2 => concat!(
                    "a86a6d6f64656c5f6e616d656c636f6d7061742d6d6f64656c6a6d6f64656c5f68617368784039666161386364333865",
                    "376336303031303132663061636536343762373035353065323361613134316434363264363065616439656432313037",
                    "3932373661316c617263686974656374757265656c6c616d616c746f74616c5f6c6179657273026c7175616e74697a61",
                    "74696f6e6b66696c655f747970655f3170746f74616c5f73697a655f62797465731902006c676775665f76657273696f",
                    "6e026673686172647382a76b73686172645f696e6465780063636964783b6261666b723469636f357679756432736b6c",
                    "746b6c706364616e706a6436727863636b787a7a6c686c76746f683268326d6e7864376575693374616b6c617965725f",
                    "72616e6765a26573746172740063656e640072696e636c756465735f656d62656464696e67f574696e636c756465735f",
                    "6f75747075745f68656164f46a73697a655f627974657318406b626c616b65335f686173687840346565643731343165",
                    "613461356364346237383836303662643233663436653231326166396361636562616364633764316634633664633766",
                    "32353131623938a76b73686172645f696e6465780163636964783b6261666b723469627a7078707334726f37716f7570",
                    "657036726c6b747272333571786e366a3369767334717967746b343532617471796e366a35696b6c617965725f72616e",
                    "6765a26573746172740163656e640172696e636c756465735f656d62656464696e67f474696e636c756465735f6f7574",
                    "7075745f68656164f56a73697a655f627974657318406b626c616b65335f686173687840333937646466326534356466",
                    "383361386632336664313561613731386566623062623763396461326232653433303639616239646430323730633337",
                    "63396561",
                ),
                3 => concat!(
                    "a86a6d6f64656c5f6e616d656c636f6d7061742d6d6f64656c6a6d6f64656c5f68617368784037363263363336316661",
                    "303233613739306433626133323562303933323764623864356633323238343636356635656137323839323965646631",
                    "6135626131366c617263686974656374757265656c6c616d616c746f74616c5f6c6179657273026c7175616e74697a61",
                    "74696f6e6b66696c655f747970655f3170746f74616c5f73697a655f62797465731902006c676775665f76657273696f",
                    "6e036673686172647382a76b73686172645f696e6465780063636964783b6261666b723469636f357679756432736b6c",
                    "746b6c706364616e706a6436727863636b787a7a6c686c76746f683268326d6e7864376575693374616b6c617965725f",
                    "72616e6765a26573746172740063656e640072696e636c756465735f656d62656464696e67f574696e636c756465735f",
                    "6f75747075745f68656164f46a73697a655f627974657318406b626c616b65335f686173687840346565643731343165",
                    "613461356364346237383836303662643233663436653231326166396361636562616364633764316634633664633766",
                    "32353131623938a76b73686172645f696e6465780163636964783b6261666b723469627a7078707334726f37716f7570",
                    "657036726c6b747272333571786e366a3369767334717967746b343532617471796e366a35696b6c617965725f72616e",
                    "6765a26573746172740163656e640172696e636c756465735f656d62656464696e67f474696e636c756465735f6f7574",
                    "7075745f68656164f56a73697a655f627974657318406b626c616b65335f686173687840333937646466326534356466",
                    "383361386632336664313561613731386566623062623763396461326232653433303639616239646430323730633337",
                    "63396561",
                ),
                _ => unreachable!(),
            };
            let expected: Vec<u8> = expected_hex
                .as_bytes()
                .chunks_exact(2)
                .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
                .collect();
            assert_eq!(cbor, expected);
        }
    }
}
