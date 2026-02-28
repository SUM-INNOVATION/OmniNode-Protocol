//! Inbound shard request handler.
//!
//! When a remote peer requests a shard (or sub-chunk), this module reads the
//! shard from the local [`ShardStore`] via mmap, slices the requested byte
//! window, and sends the response through [`OmniNet::respond_shard`].

use omni_net::{OmniNet, ShardRequest, ShardResponse};
use tracing::{info, warn};

use crate::store::ShardStore;

/// Handle an inbound shard request.
///
/// - Reads the shard from `store` via mmap
/// - Slices `[offset .. offset + max_bytes]` (clamped to shard size)
/// - Sends the response via `net.respond_shard(channel_id, ...)`
pub async fn handle_request(
    net: &OmniNet,
    store: &ShardStore,
    request: &ShardRequest,
    channel_id: u64,
) {
    let cid = &request.cid;

    // Check if we have the shard locally.
    if !store.has(cid) {
        warn!(%cid, channel_id, "requested shard not found locally");
        let resp = ShardResponse {
            cid: cid.clone(),
            offset: 0,
            total_bytes: 0,
            data: Vec::new(),
            error: Some(format!("shard not found: {cid}")),
        };
        if let Err(e) = net.respond_shard(channel_id, resp).await {
            warn!(%cid, %e, "failed to send error response");
        }
        return;
    }

    // Memory-map the shard.
    let mapped = match store.mmap(cid) {
        Ok(m) => m,
        Err(e) => {
            warn!(%cid, %e, "failed to mmap shard");
            let resp = ShardResponse {
                cid: cid.clone(),
                offset: 0,
                total_bytes: 0,
                data: Vec::new(),
                error: Some(format!("mmap error: {e}")),
            };
            let _ = net.respond_shard(channel_id, resp).await;
            return;
        }
    };

    let total = mapped.len() as u64;
    let offset = request.offset.unwrap_or(0).min(total);
    let max_bytes = request.max_bytes.unwrap_or(total);
    let end = (offset + max_bytes).min(total);
    let chunk = &mapped[offset as usize..end as usize];

    info!(
        %cid,
        offset,
        chunk_len = chunk.len(),
        total,
        channel_id,
        "serving shard chunk"
    );

    let resp = ShardResponse {
        cid: cid.clone(),
        offset,
        total_bytes: total,
        data: chunk.to_vec(),
        error: None,
    };

    if let Err(e) = net.respond_shard(channel_id, resp).await {
        warn!(%cid, %e, "failed to send shard response");
    }
}
