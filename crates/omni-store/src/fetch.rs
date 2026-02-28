//! Outbound shard fetch orchestration.
//!
//! [`FetchManager`] tracks in-progress fetches.  Each fetch proceeds as a
//! sequence of 64 MiB windowed request-response round-trips, reassembles the
//! chunks, verifies the CID, and stores the shard to disk.

use std::collections::HashMap;

use libp2p::PeerId;
use omni_net::{OmniNet, ShardResponse};
use tracing::{info, warn};

use crate::error::{Result, StoreError};
use crate::store::ShardStore;
use crate::verify;

/// Tracks in-flight fetches keyed by CID.
pub struct FetchManager {
    /// Maximum bytes per request-response chunk (default 64 MiB).
    chunk_size: u64,
    /// In-progress fetches: CID → state.
    active: HashMap<String, FetchState>,
}

/// State for a single in-progress shard fetch.
struct FetchState {
    peer_id: PeerId,
    /// Total shard size (learned from first response).
    total_bytes: Option<u64>,
    /// Byte offset of the next chunk to request.
    next_offset: u64,
    /// Accumulated data chunks.
    buffer: Vec<u8>,
}

/// Outcome of processing a received chunk.
pub enum FetchOutcome {
    /// More chunks are needed — the next request has been sent.
    InProgress,
    /// The entire shard has been received, verified, and stored.
    Complete { cid: String, size: u64 },
    /// The fetch failed.
    Failed { cid: String, error: String },
}

impl FetchManager {
    /// Create a new fetch manager with the given chunk size.
    pub fn new(max_shard_msg_bytes: usize) -> Self {
        Self {
            chunk_size: max_shard_msg_bytes as u64,
            active: HashMap::new(),
        }
    }

    /// Start fetching a shard from a remote peer.
    ///
    /// Sends the first chunk request.  Subsequent chunks are requested
    /// automatically when [`Self::on_chunk_received`] is called.
    pub async fn start_fetch(
        &mut self,
        net: &OmniNet,
        peer_id: PeerId,
        cid: String,
    ) -> Result<()> {
        if self.active.contains_key(&cid) {
            return Err(StoreError::Other(format!("fetch already in progress: {cid}")));
        }

        info!(%cid, %peer_id, "starting shard fetch");

        net.request_shard_chunk(peer_id, cid.clone(), Some(0), Some(self.chunk_size))
            .await
            .map_err(|e| StoreError::Other(e.to_string()))?;

        self.active.insert(cid, FetchState {
            peer_id,
            total_bytes: None,
            next_offset: 0,
            buffer: Vec::new(),
        });

        Ok(())
    }

    /// Process a received shard chunk.
    ///
    /// Returns a [`FetchOutcome`] indicating whether the fetch is complete,
    /// still in progress, or has failed.
    pub async fn on_chunk_received(
        &mut self,
        net: &OmniNet,
        store: &ShardStore,
        response: &ShardResponse,
    ) -> FetchOutcome {
        // Check for server-side error.
        if let Some(ref err) = response.error {
            self.active.remove(&response.cid);
            return FetchOutcome::Failed {
                cid: response.cid.clone(),
                error: err.clone(),
            };
        }

        let state = match self.active.get_mut(&response.cid) {
            Some(s) => s,
            None => {
                warn!(cid = %response.cid, "received chunk for unknown fetch — ignoring");
                return FetchOutcome::Failed {
                    cid: response.cid.clone(),
                    error: "no active fetch for this CID".into(),
                };
            }
        };

        // Record total size from first response.
        if state.total_bytes.is_none() {
            state.total_bytes = Some(response.total_bytes);
            state.buffer.reserve(response.total_bytes as usize);
        }

        // Append chunk data.
        state.buffer.extend_from_slice(&response.data);
        state.next_offset = response.offset + response.data.len() as u64;

        let total = state.total_bytes.unwrap();
        info!(
            cid = %response.cid,
            received = state.next_offset,
            total,
            "chunk received"
        );

        // Check if we need more chunks.
        if state.next_offset < total {
            let peer_id = state.peer_id;
            let cid = response.cid.clone();
            let offset = state.next_offset;
            let chunk_size = self.chunk_size;

            if let Err(e) = net.request_shard_chunk(
                peer_id,
                cid.clone(),
                Some(offset),
                Some(chunk_size),
            ).await {
                self.active.remove(&cid);
                return FetchOutcome::Failed {
                    cid,
                    error: format!("failed to request next chunk: {e}"),
                };
            }

            return FetchOutcome::InProgress;
        }

        // All chunks received — verify and store.
        let cid = response.cid.clone();
        let state = self.active.remove(&cid).unwrap();

        if let Err(e) = verify::verify_cid(&state.buffer, &cid) {
            return FetchOutcome::Failed {
                cid,
                error: format!("integrity check failed: {e}"),
            };
        }

        let size = state.buffer.len() as u64;
        if let Err(e) = store.put(&cid, &state.buffer) {
            return FetchOutcome::Failed {
                cid,
                error: format!("failed to write shard: {e}"),
            };
        }

        info!(%cid, size, "shard fetch complete and verified");
        FetchOutcome::Complete { cid, size }
    }

    /// Check if a fetch is currently active for the given CID.
    pub fn is_active(&self, cid: &str) -> bool {
        self.active.contains_key(cid)
    }

    /// Number of in-progress fetches.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
}
