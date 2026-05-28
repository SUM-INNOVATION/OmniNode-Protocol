//! Stage 12.4 — tensor transport abstraction.
//!
//! Two impls, mirroring 12.2's `ContributorRelay` shape:
//!
//!   - [`InMemoryTensorTransport`] — `VecDeque`-backed, no networking.
//!     Used by tests + by single-process pipelines where the sender
//!     and receiver share an address space.
//!   - [`OmniNetTensorTransport`] — feature-gated (`network`).
//!     Bridges to `omni-net`'s existing request/response tensor
//!     codec by carrying our signed `ActivationHandoff` inside
//!     `TensorRequest.data`. The outer `TensorRequest` fields are
//!     populated from the inner envelope at send time but treated
//!     as **advisory routing hints only** at receive time — the
//!     signed inner envelope is the source of truth.
//!
//! Receiver-side flow (both impls):
//!
//!   1. `poll_handoffs` drains incoming envelopes.
//!   2. Caller runs [`verify_activation_handoff`](crate::handoff_verify::verify_activation_handoff)
//!      to check schema, sender signature, session/assignment binding.
//!   3. Caller feeds each verified envelope into a [`HandoffReceiver`](crate::handoff_verify::HandoffReceiver),
//!      which reassembles chunks and emits `Complete { tensor_bytes }`
//!      ONLY after the reconstructed BLAKE3 equals the signed
//!      `tensor_hash` and total length equals `byte_len`.
//!
//! Bincode 1.3 is the wire format for the inner envelope (same as
//! every other canonical body in this crate). The outer `omni-net`
//! codec uses its own bincode 2.0 framing — two distinct layers.

use std::collections::VecDeque;

use crate::handoff::ActivationHandoff;

/// Transport errors. Distinct from `RelayError` so callers can
/// distinguish "tensor pipe broken" from "session gossip broken".
#[derive(Debug, thiserror::Error)]
pub enum TensorTransportError {
    #[error("tensor transport send failed: {0}")]
    Send(String),

    #[error("tensor transport poll failed: {0}")]
    Poll(String),

    #[error("tensor envelope encode failed: {0}")]
    Encode(String),

    #[error("tensor envelope decode failed: {0}")]
    Decode(String),

    #[error("malformed target peer hint: {0}")]
    BadPeerHint(String),
}

/// Sync, transport-agnostic interface for sending + draining
/// `ActivationHandoff` envelopes.
pub trait TensorTransport {
    /// Send one handoff to a target peer. `target_peer_hint` is an
    /// operator-supplied address string (libp2p multiaddr or PeerId
    /// base58); in-memory impls may ignore it.
    fn send_handoff(
        &mut self,
        target_peer_hint: Option<&str>,
        handoff: &ActivationHandoff,
    ) -> Result<(), TensorTransportError>;

    /// Drain all pending handoffs received since the last poll.
    /// Non-blocking; returns empty if nothing is pending.
    fn poll_handoffs(
        &mut self,
    ) -> Result<Vec<ActivationHandoff>, TensorTransportError>;
}

// ── InMemoryTensorTransport ───────────────────────────────────────────────

/// Test-only transport. One FIFO queue, no transport. `send_handoff`
/// pushes; `poll_handoffs` drains. Sender and receiver share the
/// same instance.
pub struct InMemoryTensorTransport {
    queue: VecDeque<ActivationHandoff>,
}

impl InMemoryTensorTransport {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    pub fn pending(&self) -> usize {
        self.queue.len()
    }
}

impl Default for InMemoryTensorTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorTransport for InMemoryTensorTransport {
    fn send_handoff(
        &mut self,
        _target_peer_hint: Option<&str>,
        handoff: &ActivationHandoff,
    ) -> Result<(), TensorTransportError> {
        self.queue.push_back(handoff.clone());
        Ok(())
    }

    fn poll_handoffs(
        &mut self,
    ) -> Result<Vec<ActivationHandoff>, TensorTransportError> {
        Ok(self.queue.drain(..).collect())
    }
}

// ── OmniNetTensorTransport (feature = "network") ──────────────────────────

#[cfg(feature = "network")]
pub use omni_net_tensor_transport::OmniNetTensorTransport;

#[cfg(feature = "network")]
mod omni_net_tensor_transport {
    use super::*;

    use std::sync::{Arc, Mutex as StdMutex};

    use libp2p::{multiaddr::Protocol, Multiaddr, PeerId};
    use omni_net::{OmniNet, OmniNetEvent, TensorRequest, TensorResponse};
    use tokio::runtime::Handle;
    use tokio::sync::Mutex as AsyncMutex;

    /// Production tensor transport over `omni_net::OmniNet`'s
    /// request/response tensor codec. Holds an
    /// `Arc<tokio::sync::Mutex<OmniNet>>` so it can share the mesh
    /// connection with the Stage 12.2 `OmniNetRelay` that the watch
    /// loop already opens.
    ///
    /// Outer `TensorRequest` fields are populated from the inner
    /// `ActivationHandoff` at send time but are NOT trusted at
    /// receive time — the receiver decodes `TensorRequest.data` as
    /// an `ActivationHandoff` and runs the verifier on that. The
    /// `micro_batch_index` field is overloaded for `chunk_index`
    /// routing; documented as Stage 12.4 transport overlay.
    #[derive(Clone)]
    pub struct OmniNetTensorTransport {
        net: Arc<AsyncMutex<OmniNet>>,
        handle: Handle,
        pending: Arc<StdMutex<VecDeque<ActivationHandoff>>>,
    }

    impl OmniNetTensorTransport {
        pub fn new(net: Arc<AsyncMutex<OmniNet>>, handle: Handle) -> Self {
            Self {
                net,
                handle,
                pending: Arc::new(StdMutex::new(VecDeque::new())),
            }
        }

        fn parse_peer_id(hint: &str) -> Result<PeerId, TensorTransportError> {
            use std::str::FromStr;
            // Accept either a bare PeerId base58 or a multiaddr
            // whose final component is /p2p/<peer-id>.
            if let Ok(p) = PeerId::from_str(hint) {
                return Ok(p);
            }
            if let Ok(maddr) = Multiaddr::from_str(hint) {
                // libp2p's Multiaddr Iter is forward-only; collect and
                // scan from the end since the /p2p/<peer-id> protocol
                // component conventionally sits last.
                let protos: Vec<_> = maddr.iter().collect();
                for proto in protos.into_iter().rev() {
                    if let Protocol::P2p(peer_id) = proto {
                        return Ok(peer_id);
                    }
                }
            }
            Err(TensorTransportError::BadPeerHint(hint.to_string()))
        }

        fn drain_events(&self) {
            let mut net = tokio::task::block_in_place(|| {
                self.handle.block_on(self.net.lock())
            });
            let mut pending = self.pending.lock().expect("pending poisoned");
            // ACK semantics on `/omni/tensor-xfer/1`:
            //   - `accepted: true`  → we decoded a Stage 12.4
            //     ActivationHandoff out of `request.data`. The
            //     handoff is now queued for the caller's verifier;
            //     we do NOT trust it at this layer.
            //   - `accepted: false` → bincode decode of
            //     `request.data` as `ActivationHandoff` failed. The
            //     request is unprocessable at the transport layer,
            //     so we tell the sender immediately rather than
            //     silently swallow a malformed payload while
            //     ACKing success (review #3 closed that hole).
            // Verifier-layer rejections (signature, drift, hash
            // mismatch, etc.) are per-envelope outcomes the caller
            // consumes via `poll_handoffs`; they do NOT generate
            // negative ACKs here.
            while let Some(ev) = net.try_next_event() {
                if let OmniNetEvent::TensorReceived {
                    request,
                    channel_id,
                    ..
                } = ev
                {
                    let (accepted, error) =
                        match bincode1::deserialize::<ActivationHandoff>(
                            &request.data,
                        ) {
                            Ok(h) => {
                                pending.push_back(h);
                                (true, None)
                            }
                            Err(e) => (
                                false,
                                Some(format!(
                                    "tensor-xfer payload did not bincode-decode \
                                     as Stage 12.4 ActivationHandoff: {e}"
                                )),
                            ),
                        };
                    let resp = TensorResponse {
                        session_id: request.session_id.clone(),
                        micro_batch_index: request.micro_batch_index,
                        stage_index: request.to_stage,
                        accepted,
                        error,
                    };
                    let _ = tokio::task::block_in_place(|| {
                        self.handle
                            .block_on(net.respond_tensor(channel_id, resp))
                    });
                }
            }
        }
    }

    impl TensorTransport for OmniNetTensorTransport {
        fn send_handoff(
            &mut self,
            target_peer_hint: Option<&str>,
            handoff: &ActivationHandoff,
        ) -> Result<(), TensorTransportError> {
            let hint = target_peer_hint.ok_or_else(|| {
                TensorTransportError::BadPeerHint(
                    "OmniNetTensorTransport requires a non-empty target peer hint".into(),
                )
            })?;
            let peer_id = Self::parse_peer_id(hint)?;
            let data = bincode1::serialize(handoff)
                .map_err(|e| TensorTransportError::Encode(e.to_string()))?;
            // Map our signed envelope's identity onto the outer
            // TensorRequest for routing. Receiver does NOT trust
            // these outer fields — see module-level docs.
            let dtype_u8 = match handoff.dtype {
                crate::handoff::TensorDtype::F16 => 0u8,
                crate::handoff::TensorDtype::Bf16 => 1u8,
                crate::handoff::TensorDtype::F32 => 2u8,
            };
            // Outer TensorRequest stage hints are intentionally
            // unset at Stage 12.4 (left at 0). They predate Stage
            // 12.3's signed-session model and are not trusted by
            // any 12.4 verifier — the signed inner ActivationHandoff
            // is the source of truth for stage identity. A future
            // Stage 12.5+ "signed contributor peer-advertisement"
            // PR can populate them as receiver-side pre-filter
            // hints; doing so today would suggest a trust gradient
            // these fields do not carry.
            let from_stage = 0u32;
            let to_stage = 0u32;
            let seq_len = handoff.shape.first().copied().unwrap_or(0) as u32;
            let hidden_dim = handoff.shape.get(1).copied().unwrap_or(1) as u32;
            let req = TensorRequest {
                session_id: handoff.session_id.clone(),
                // chunk_index reuse: documented as Stage 12.4
                // transport overlay; receiver does not trust this.
                micro_batch_index: handoff.chunk_index,
                from_stage,
                to_stage,
                seq_len,
                hidden_dim,
                dtype: dtype_u8,
                data,
            };
            let result = tokio::task::block_in_place(|| {
                self.handle.block_on(async {
                    let g = self.net.lock().await;
                    g.request_tensor(peer_id, req).await
                })
            });
            result.map_err(|e| TensorTransportError::Send(e.to_string()))?;
            Ok(())
        }

        fn poll_handoffs(
            &mut self,
        ) -> Result<Vec<ActivationHandoff>, TensorTransportError> {
            self.drain_events();
            let mut q = self.pending.lock().expect("pending poisoned");
            Ok(q.drain(..).collect())
        }
    }
}
