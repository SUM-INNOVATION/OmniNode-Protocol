// omni-net::tensor_codec — TensorCodec for the `/omni/tensor-xfer/1`
// request-response protocol. Transports hidden-state activation tensors
// between pipeline stages over the existing libp2p QUIC transport.
//
// Wire format: [u32 big-endian length][bincode payload]
// Identical framing to ShardCodec — activation data is in TensorRequest.data.

use std::io;

use async_trait::async_trait;
use futures::prelude::*;
use serde::{Deserialize, Serialize};

/// Protocol identifier negotiated via ALPN during substream opening.
pub const TENSOR_XFER_PROTOCOL: &str = "/omni/tensor-xfer/1";

/// Safety limit: reject any single message larger than 128 MiB.
/// Actual activation sizes: 7B ≈ 4 MB, 13B ≈ 5 MB, 70B ≈ 32 MB (f16).
const MAX_MSG_BYTES: usize = 128 * 1024 * 1024;

// ── Message Types ─────────────────────────────────────────────────────────────

/// Request carrying a hidden-state activation tensor to the next pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRequest {
    /// Pipeline session UUID.
    pub session_id: String,
    /// Micro-batch index within the GPipe schedule.
    pub micro_batch_index: u32,
    /// Stage that produced this activation.
    pub from_stage: u32,
    /// Stage that should consume this activation.
    pub to_stage: u32,
    /// Sequence length dimension.
    pub seq_len: u32,
    /// Hidden dimension.
    pub hidden_dim: u32,
    /// Dtype discriminant: 0 = F16, 1 = BF16, 2 = F32.
    pub dtype: u8,
    /// Raw activation bytes (seq_len × hidden_dim × dtype_bytes).
    pub data: Vec<u8>,
}

/// Acknowledgment response from the receiving pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorResponse {
    /// Pipeline session UUID.
    pub session_id: String,
    /// Micro-batch index being acknowledged.
    pub micro_batch_index: u32,
    /// Stage that is acknowledging receipt.
    pub stage_index: u32,
    /// Whether the tensor was accepted.
    pub accepted: bool,
    /// Error message if not accepted.
    pub error: Option<String>,
}

// ── Codec ─────────────────────────────────────────────────────────────────────

/// Codec for the `/omni/tensor-xfer/1` request-response protocol.
#[derive(Debug, Clone)]
pub struct TensorCodec {
    max_msg_bytes: usize,
}

impl Default for TensorCodec {
    fn default() -> Self {
        Self {
            max_msg_bytes: MAX_MSG_BYTES,
        }
    }
}

#[async_trait]
impl libp2p::request_response::Codec for TensorCodec {
    type Protocol = String;
    type Request = TensorRequest;
    type Response = TensorResponse;

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
        let mut codec = TensorCodec::default();
        let req = TensorRequest {
            session_id: "sess-001".into(),
            micro_batch_index: 0,
            from_stage: 0,
            to_stage: 1,
            seq_len: 32,
            hidden_dim: 64,
            dtype: 0, // F16
            data: vec![0xAB; 32 * 64 * 2], // 4096 bytes
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

        assert_eq!(decoded.session_id, "sess-001");
        assert_eq!(decoded.from_stage, 0);
        assert_eq!(decoded.to_stage, 1);
        assert_eq!(decoded.seq_len, 32);
        assert_eq!(decoded.hidden_dim, 64);
        assert_eq!(decoded.dtype, 0);
        assert_eq!(decoded.data.len(), 32 * 64 * 2);
        assert_eq!(decoded.data[0], 0xAB);
    }

    #[tokio::test]
    async fn response_round_trip() {
        let mut codec = TensorCodec::default();
        let resp = TensorResponse {
            session_id: "sess-001".into(),
            micro_batch_index: 0,
            stage_index: 1,
            accepted: true,
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

        assert_eq!(decoded.session_id, "sess-001");
        assert_eq!(decoded.micro_batch_index, 0);
        assert_eq!(decoded.stage_index, 1);
        assert!(decoded.accepted);
        assert!(decoded.error.is_none());
    }

    #[tokio::test]
    async fn error_response_round_trip() {
        let mut codec = TensorCodec::default();
        let resp = TensorResponse {
            session_id: "sess-001".into(),
            micro_batch_index: 0,
            stage_index: 1,
            accepted: false,
            error: Some("stage not ready".into()),
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

        assert!(!decoded.accepted);
        assert_eq!(decoded.error.as_deref(), Some("stage not ready"));
    }

    #[tokio::test]
    async fn rejects_oversized_message() {
        let mut codec = TensorCodec {
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
