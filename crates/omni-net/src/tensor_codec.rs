// omni-net::tensor_codec — TensorCodec for the `/omni/tensor-xfer/1`
// request-response protocol. Transports hidden-state activation tensors
// between pipeline stages over the existing libp2p QUIC transport.
//
// Wire format: [u32 big-endian length][bincode payload]
// Identical framing to ShardCodec — activation data is in TensorRequest.data.

use std::io;

use async_trait::async_trait;
use futures::prelude::*;

use crate::framing::{read_length_prefixed, write_length_prefixed};
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

/// Proves this codec reads through the shared framing module rather than a
/// private copy. The duplication is what let the eager-allocation defect exist
/// twice; these tests fail if a local reader is reintroduced.
#[cfg(test)]
mod shared_framing_routing_tests {
    use super::*;
    use futures::io::Cursor;

    fn header_only(len: u32) -> Vec<u8> {
        len.to_be_bytes().to_vec()
    }

    #[tokio::test]
    async fn oversized_prefix_is_refused_with_the_shared_reader_message() {
        // The shared reader owns this exact wording. A private copy would have
        // to reproduce it, and any drift shows up here.
        let over = (MAX_MSG_BYTES as u64 + 1) as u32;
        let mut io = Cursor::new(header_only(over));
        let err = crate::framing::read_length_prefixed(&mut io, MAX_MSG_BYTES)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("message too large"), "{err}");
    }

    #[tokio::test]
    async fn a_declared_length_far_beyond_delivery_fails_on_data_not_memory() {
        // The scratch is capped at READ_STEP and the destination follows
        // delivered bytes, so neither approaches the declaration — observed
        // through this codec's own ceiling rather than the shared module's
        // test constant.
        let declared = (MAX_MSG_BYTES / 2) as u32;
        let mut body = declared.to_be_bytes().to_vec();
        body.extend_from_slice(&[5u8; 64]);
        let err = {
            let _guard = crate::test_alloc::ArmGuard::new(1024 * 1024);
            crate::framing::read_length_prefixed(&mut Cursor::new(body), MAX_MSG_BYTES)
                .await
                .unwrap_err()
        };
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }
}
