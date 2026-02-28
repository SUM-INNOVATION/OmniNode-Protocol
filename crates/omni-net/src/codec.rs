// omni-net::codec — ShardCodec for the `/omni/shard-xfer/1` request-response
// protocol. Transfers model shard data between peers over the existing libp2p
// QUIC transport.
//
// Wire format: [u32 big-endian length][bincode payload]
// Bincode is used because ShardResponse carries raw `Vec<u8>` weight data;
// bincode writes this as length + raw bytes (zero overhead), whereas JSON
// would base64-encode it (+33%) and CBOR adds tagging overhead.

use std::io;

use async_trait::async_trait;
use futures::prelude::*;
use serde::{Deserialize, Serialize};

/// Protocol identifier negotiated via ALPN during substream opening.
pub const SHARD_XFER_PROTOCOL: &str = "/omni/shard-xfer/1";

/// Safety limit: reject any single message larger than 256 MiB.
/// The actual chunk size is controlled by `StoreConfig::max_shard_msg_bytes`
/// (default 64 MiB); this codec limit is intentionally higher to allow for
/// bincode framing overhead.
const MAX_MSG_BYTES: usize = 256 * 1024 * 1024;

// ── Message Types ─────────────────────────────────────────────────────────────

/// Request for a shard (or sub-chunk thereof), identified by CID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardRequest {
    /// CIDv1 string identifying the desired shard.
    pub cid: String,
    /// Byte offset within the shard to start reading from.
    /// `None` or `Some(0)` means from the beginning.
    pub offset: Option<u64>,
    /// Maximum bytes to return. `None` means the entire shard.
    /// Used for windowed streaming of large shards.
    pub max_bytes: Option<u64>,
}

/// Response carrying shard data (or an error).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardResponse {
    /// The CID this response corresponds to.
    pub cid: String,
    /// Byte offset this chunk starts at within the full shard.
    pub offset: u64,
    /// Total shard size in bytes (so the requester knows when it has
    /// received everything and how many sub-chunk requests remain).
    pub total_bytes: u64,
    /// The shard payload (may be a sub-chunk).
    pub data: Vec<u8>,
    /// If present, the request failed — this contains the error message.
    /// When set, `data` is empty.
    pub error: Option<String>,
}

// ── Codec ─────────────────────────────────────────────────────────────────────

/// Codec for the `/omni/shard-xfer/1` request-response protocol.
#[derive(Debug, Clone)]
pub struct ShardCodec {
    max_msg_bytes: usize,
}

impl Default for ShardCodec {
    fn default() -> Self {
        Self {
            max_msg_bytes: MAX_MSG_BYTES,
        }
    }
}

#[async_trait]
impl libp2p::request_response::Codec for ShardCodec {
    type Protocol = String;
    type Request = ShardRequest;
    type Response = ShardResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let buf = read_length_prefixed(io, self.max_msg_bytes).await?;
        let (req, _) =
            bincode::serde::decode_from_slice(&buf, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Ok(req)
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let buf = read_length_prefixed(io, self.max_msg_bytes).await?;
        let (resp, _) =
            bincode::serde::decode_from_slice(&buf, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Ok(resp)
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let buf = bincode::serde::encode_to_vec(&req, bincode::config::standard())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        write_length_prefixed(io, &buf).await
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        resp: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let buf = bincode::serde::encode_to_vec(&resp, bincode::config::standard())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        write_length_prefixed(io, &buf).await
    }
}

// ── Wire Helpers ──────────────────────────────────────────────────────────────

/// Read a `[u32 BE length][payload]` frame.
async fn read_length_prefixed<T>(io: &mut T, max_bytes: usize) -> io::Result<Vec<u8>>
where
    T: AsyncRead + Unpin + Send,
{
    let mut len_buf = [0u8; 4];
    io.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("message too large: {len} bytes (max {max_bytes})"),
        ));
    }
    let mut buf = vec![0u8; len];
    io.read_exact(&mut buf).await?;
    Ok(buf)
}

/// Write a `[u32 BE length][payload]` frame.
async fn write_length_prefixed<T>(io: &mut T, data: &[u8]) -> io::Result<()>
where
    T: AsyncWrite + Unpin + Send,
{
    let len = u32::try_from(data.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("payload exceeds u32::MAX: {} bytes", data.len()),
        )
    })?;
    io.write_all(&len.to_be_bytes()).await?;
    io.write_all(data).await?;
    io.flush().await?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use futures::io::Cursor;
    use libp2p::request_response::Codec;

    #[tokio::test]
    async fn request_round_trip() {
        let mut codec = ShardCodec::default();
        let req = ShardRequest {
            cid: "bafkr4itest".into(),
            offset: Some(1024),
            max_bytes: Some(65536),
        };

        let mut buf = Vec::new();
        codec
            .write_request(&String::new(), &mut Cursor::new(&mut buf), req.clone())
            .await
            .unwrap();

        let decoded = codec
            .read_request(&String::new(), &mut Cursor::new(&buf))
            .await
            .unwrap();

        assert_eq!(decoded.cid, req.cid);
        assert_eq!(decoded.offset, req.offset);
        assert_eq!(decoded.max_bytes, req.max_bytes);
    }

    #[tokio::test]
    async fn response_round_trip() {
        let mut codec = ShardCodec::default();
        let resp = ShardResponse {
            cid: "bafkr4itest".into(),
            offset: 0,
            total_bytes: 1_000_000,
            data: vec![0xAB; 4096],
            error: None,
        };

        let mut buf = Vec::new();
        codec
            .write_response(&String::new(), &mut Cursor::new(&mut buf), resp.clone())
            .await
            .unwrap();

        let decoded = codec
            .read_response(&String::new(), &mut Cursor::new(&buf))
            .await
            .unwrap();

        assert_eq!(decoded.cid, resp.cid);
        assert_eq!(decoded.offset, resp.offset);
        assert_eq!(decoded.total_bytes, resp.total_bytes);
        assert_eq!(decoded.data.len(), 4096);
        assert_eq!(decoded.data[0], 0xAB);
        assert!(decoded.error.is_none());
    }

    #[tokio::test]
    async fn error_response_round_trip() {
        let mut codec = ShardCodec::default();
        let resp = ShardResponse {
            cid: "bafkr4imissing".into(),
            offset: 0,
            total_bytes: 0,
            data: Vec::new(),
            error: Some("shard not found".into()),
        };

        let mut buf = Vec::new();
        codec
            .write_response(&String::new(), &mut Cursor::new(&mut buf), resp.clone())
            .await
            .unwrap();

        let decoded = codec
            .read_response(&String::new(), &mut Cursor::new(&buf))
            .await
            .unwrap();

        assert_eq!(decoded.error.as_deref(), Some("shard not found"));
        assert!(decoded.data.is_empty());
    }

    #[tokio::test]
    async fn rejects_oversized_message() {
        let mut codec = ShardCodec {
            max_msg_bytes: 16, // tiny limit for test
        };

        // Fabricate a frame claiming 1000 bytes.
        let mut buf = Vec::new();
        buf.extend_from_slice(&1000u32.to_be_bytes());
        buf.extend_from_slice(&[0u8; 1000]);

        let result = codec
            .read_request(&String::new(), &mut Cursor::new(&buf))
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("message too large"));
    }
}
