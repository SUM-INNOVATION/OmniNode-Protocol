//! Inbound shard request handler.
//!
//! When a remote peer requests a shard (or sub-chunk), this module reads the
//! shard from the local [`ShardStore`] via mmap, slices the requested byte
//! window, and sends the response through [`OmniNet::respond_shard`].

use std::sync::Arc;

use omni_net::{OmniNet, ShardRequest, ShardResponse};
use tracing::{info, warn};

use crate::store::ShardStore;

/// Outbound-response abstraction over [`OmniNet::respond_shard`].
///
/// Exists so [`handle_request`] can be driven from tests with a capturing
/// implementation, without standing up a real libp2p swarm. Production
/// keeps passing `OmniNet` exactly as before.
#[async_trait::async_trait]
pub trait RespondShard: Send + Sync {
    async fn respond_shard(&self, channel_id: u64, response: ShardResponse) -> anyhow::Result<()>;
}

#[async_trait::async_trait]
impl RespondShard for OmniNet {
    async fn respond_shard(&self, channel_id: u64, response: ShardResponse) -> anyhow::Result<()> {
        OmniNet::respond_shard(self, channel_id, response).await
    }
}

#[async_trait::async_trait]
impl<T: RespondShard + ?Sized> RespondShard for Arc<T> {
    async fn respond_shard(&self, channel_id: u64, response: ShardResponse) -> anyhow::Result<()> {
        (**self).respond_shard(channel_id, response).await
    }
}

/// Resolve the byte window a shard request selects, as `(start, end)`
/// indices into a `total`-byte shard.
///
/// Preserves this endpoint's established policy exactly: an `offset` past
/// the end of the shard is CLAMPED to the end (yielding an empty, but
/// successful, response), and an absent `max_bytes` means "the rest of the
/// shard". Only the arithmetic changes.
///
/// The window is resolved entirely in `u64` and converted only once it is
/// known to be in range, so peer-supplied `offset`/`max_bytes` values can
/// neither overflow nor truncate. The previous `offset + max_bytes`
/// panicked on overflow under debug assertions and, with them off, wrapped
/// to an inverted range that panicked at the slice instead — either way
/// terminating the node's request loop.
///
/// The returned pair always satisfies `start <= end <= total`.
fn resolve_shard_window(
    total: usize,
    offset: Option<u64>,
    max_bytes: Option<u64>,
) -> (usize, usize) {
    // `usize` -> `u64` is lossless on every supported target.
    let total_u64 = total as u64;
    let start = offset.unwrap_or(0).min(total_u64);
    let len = max_bytes.unwrap_or(total_u64);
    let end = start.saturating_add(len).min(total_u64);
    debug_assert!(start <= end && end <= total_u64);
    // Both are <= `total_u64`, which came from a `usize`, so neither
    // conversion can truncate.
    (start as usize, end as usize)
}

/// Handle an inbound shard request.
///
/// - Reads the shard from `store` via mmap
/// - Slices `[offset .. offset + max_bytes]` (clamped to shard size)
/// - Sends the response via `net.respond_shard(channel_id, ...)`
pub async fn handle_request<N: RespondShard + ?Sized>(
    net: &N,
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
    let (start, end) = resolve_shard_window(mapped.len(), request.offset, request.max_bytes);
    let offset = start as u64;
    let chunk = &mapped[start..end];

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // ── WP-S1: shard-range regression fixtures ──────────────────────────
    //
    // NOTE ON REACHABILITY: unlike SNIP, the Omni shard-serve path has NO
    // access-control gate — `omni-node/src/main.rs` dispatches every
    // `ShardRequested` event straight to `handle_request`. There is no ACL
    // fixture to write here because there is no ACL; any peer that knows a
    // CID this node holds reaches the code below. That absence is recorded
    // as a finding, not fixed here (out of this increment's scope).

    /// Capturing [`RespondShard`] so the real handler can be driven
    /// without a libp2p swarm.
    #[derive(Default)]
    struct RecorderNet {
        sent: Mutex<Vec<(u64, ShardResponse)>>,
    }

    #[async_trait::async_trait]
    impl RespondShard for RecorderNet {
        async fn respond_shard(
            &self,
            channel_id: u64,
            response: ShardResponse,
        ) -> anyhow::Result<()> {
            self.sent.lock().unwrap().push((channel_id, response));
            Ok(())
        }
    }

    impl RecorderNet {
        fn take(&self) -> Vec<(u64, ShardResponse)> {
            std::mem::take(&mut *self.sent.lock().unwrap())
        }
    }

    fn store_with(bytes: &[u8]) -> (tempfile::TempDir, ShardStore, String) {
        let dir = tempfile::tempdir().unwrap();
        let store = ShardStore::new(dir.path().join("shards")).unwrap();
        let cid = crate::content_id::cid_from_data(bytes);
        store.put(&cid, bytes).unwrap();
        (dir, store, cid)
    }

    fn pull(cid: &str, offset: Option<u64>, max_bytes: Option<u64>) -> ShardRequest {
        ShardRequest {
            cid: cid.to_string(),
            offset,
            max_bytes,
        }
    }

    /// Round-trip through the REAL wire codec so the handler is driven with
    /// a request that actually survived encode/decode.
    async fn through_codec(req: ShardRequest) -> ShardRequest {
        use futures::io::Cursor;
        use libp2p::request_response::Codec;
        let mut codec = omni_net::ShardCodec::default();
        let mut buf = Vec::new();
        codec
            .write_request(&String::new(), &mut Cursor::new(&mut buf), req)
            .await
            .unwrap();
        codec
            .read_request(&String::new(), &mut Cursor::new(&buf))
            .await
            .unwrap()
    }

    // ── resolve_shard_window: boundary matrix ───────────────────────────
    //
    // Omni policy under test: offsets past EOF CLAMP (empty success),
    // absent `max_bytes` means "rest of the shard".

    #[test]
    fn window_offset_boundaries_nonempty() {
        assert_eq!(resolve_shard_window(4, Some(0), None), (0, 4));
        assert_eq!(resolve_shard_window(4, Some(1), None), (1, 4));
        assert_eq!(resolve_shard_window(4, Some(4), None), (4, 4));
        assert_eq!(resolve_shard_window(4, Some(5), None), (4, 4));
        assert_eq!(resolve_shard_window(4, Some(u64::MAX), None), (4, 4));
    }

    #[test]
    fn window_max_bytes_boundaries_nonempty() {
        assert_eq!(resolve_shard_window(4, Some(1), Some(0)), (1, 1));
        assert_eq!(resolve_shard_window(4, Some(1), Some(2)), (1, 3));
        assert_eq!(resolve_shard_window(4, Some(1), Some(3)), (1, 4));
        assert_eq!(resolve_shard_window(4, Some(1), Some(4)), (1, 4));
        assert_eq!(resolve_shard_window(4, Some(1), Some(u64::MAX)), (1, 4));
        assert_eq!(resolve_shard_window(4, Some(0), Some(u64::MAX)), (0, 4));
    }

    #[test]
    fn window_absent_option_fields() {
        assert_eq!(resolve_shard_window(4, None, None), (0, 4));
        assert_eq!(resolve_shard_window(4, None, Some(u64::MAX)), (0, 4));
        assert_eq!(resolve_shard_window(4, Some(u64::MAX), None), (4, 4));
    }

    #[test]
    fn window_empty_buffer_never_indexes_out_of_range() {
        for offset in [None, Some(0), Some(1), Some(u64::MAX)] {
            for max_bytes in [None, Some(0), Some(1), Some(u64::MAX)] {
                assert_eq!(
                    resolve_shard_window(0, offset, max_bytes),
                    (0, 0),
                    "empty shard, offset={offset:?} max_bytes={max_bytes:?}"
                );
            }
        }
    }

    /// Synthetic: `total` values are asserted against, not allocated.
    #[test]
    fn window_invariant_holds_over_extremes() {
        let totals = [0usize, 1, 2, 4, 4096];
        let extremes = [
            None,
            Some(0u64),
            Some(1),
            Some(2),
            Some(4095),
            Some(4096),
            // Narrowing-specific: truncates to 1 under a 32-bit `as usize`,
            // i.e. lands inside a small buffer. u64::MAX truncates to
            // u32::MAX, which stays past the end and distinguishes nothing.
            Some(u32::MAX as u64),
            Some(u32::MAX as u64 + 2),
            Some(u64::MAX - 1),
            Some(u64::MAX),
        ];
        for total in totals {
            for offset in extremes {
                for max_bytes in extremes {
                    let (start, end) = resolve_shard_window(total, offset, max_bytes);
                    assert!(
                        start <= end && end <= total,
                        "total={total} offset={offset:?} max_bytes={max_bytes:?} \
                         -> ({start}, {end})"
                    );
                }
            }
        }
    }

    // ── the real handler ────────────────────────────────────────────────

    #[tokio::test]
    async fn malicious_range_is_answered_then_handler_still_serves() {
        let (_dir, store, cid) = store_with(b"abcd");
        let net = RecorderNet::default();

        let bad = through_codec(pull(&cid, Some(1), Some(u64::MAX))).await;
        handle_request(&net, &store, &bad, 7).await;
        let sent = net.take();
        assert_eq!(sent.len(), 1);
        let (channel, resp) = &sent[0];
        assert_eq!(*channel, 7);
        assert_eq!(resp.cid, cid);
        assert_eq!(resp.offset, 1);
        assert_eq!(resp.total_bytes, 4);
        assert_eq!(resp.data, b"bcd");
        assert!(resp.error.is_none());

        let good = through_codec(pull(&cid, Some(0), Some(2))).await;
        handle_request(&net, &store, &good, 8).await;
        let sent = net.take();
        assert_eq!(sent.len(), 1);
        let (channel, resp) = &sent[0];
        assert_eq!(*channel, 8);
        assert_eq!(resp.offset, 0);
        assert_eq!(resp.total_bytes, 4);
        assert_eq!(resp.data, b"ab");
        assert!(resp.error.is_none());
    }

    #[tokio::test]
    async fn handler_survives_the_full_extreme_grid() {
        let (_dir, store, cid) = store_with(b"abcd");
        let net = RecorderNet::default();
        let extremes = [None, Some(0u64), Some(1), Some(4), Some(5), Some(u64::MAX)];
        let mut answered = 0;
        for offset in extremes {
            for max_bytes in extremes {
                let req = through_codec(pull(&cid, offset, max_bytes)).await;
                handle_request(&net, &store, &req, 1).await;
                let sent = net.take();
                assert_eq!(sent.len(), 1, "offset={offset:?} max_bytes={max_bytes:?}");
                let resp = &sent[0].1;
                assert!(resp.error.is_none());
                assert_eq!(resp.total_bytes, 4);
                assert!(resp.offset <= 4);
                assert!(resp.offset as usize + resp.data.len() <= 4);
                answered += 1;
            }
        }
        assert_eq!(answered, 36);
    }

    #[tokio::test]
    async fn handler_survives_extremes_on_empty_shard() {
        let (_dir, store, cid) = store_with(b"");
        let net = RecorderNet::default();
        for offset in [None, Some(0u64), Some(1), Some(u64::MAX)] {
            for max_bytes in [None, Some(0u64), Some(1), Some(u64::MAX)] {
                let req = through_codec(pull(&cid, offset, max_bytes)).await;
                handle_request(&net, &store, &req, 1).await;
                let sent = net.take();
                assert_eq!(sent.len(), 1);
                let resp = &sent[0].1;
                assert_eq!(resp.offset, 0);
                assert_eq!(resp.total_bytes, 0);
                assert!(resp.data.is_empty());
                assert!(resp.error.is_none());
            }
        }
    }
}
