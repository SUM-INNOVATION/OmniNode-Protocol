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

/// How the fetcher asks a peer for the next piece.
///
/// Exists so the handler can be driven in tests without binding a socket:
/// `OmniNet` is one implementation, a recording fake is another. Without it
/// the handler tests need a live swarm, which fails wherever socket binding
/// is prohibited.
#[allow(async_fn_in_trait)]
pub(crate) trait ChunkRequester {
    async fn request_chunk(
        &self,
        peer_id: PeerId,
        cid: String,
        offset: u64,
        len: u64,
    ) -> std::result::Result<(), String>;
}

/// The production implementation: a real libp2p request.
pub(crate) struct NetRequester<'a>(pub(crate) &'a OmniNet);

impl ChunkRequester for NetRequester<'_> {
    async fn request_chunk(
        &self,
        peer_id: PeerId,
        cid: String,
        offset: u64,
        len: u64,
    ) -> std::result::Result<(), String> {
        self.0
            .request_shard_chunk(peer_id, cid, Some(offset), Some(len))
            .await
            .map_err(|e| e.to_string())
    }
}
use crate::verify;

/// Tracks in-flight fetches keyed by CID.
pub struct FetchManager {
    /// Maximum bytes per request-response chunk (default 64 MiB).
    chunk_size: u64,
    /// In-progress fetches: CID → state.
    active: HashMap<String, FetchState>,
    /// Hard ceiling on total accumulated bytes for one shard.
    max_shard_total_bytes: u64,
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
    /// Shard size from the gossip announcement, when one was seen.
    expected_size: Option<u64>,
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
    ///
    /// The total-bytes ceiling defaults to
    /// [`omni_types::config::DEFAULT_MAX_SHARD_TOTAL_BYTES`]. Use
    /// [`FetchManager::with_limits`] to set it explicitly.
    pub fn new(max_shard_msg_bytes: usize) -> Self {
        use omni_types::config::{DEFAULT_MAX_SHARD_MSG_BYTES, DEFAULT_MAX_SHARD_TOTAL_BYTES};
        Self::with_limits(max_shard_msg_bytes, DEFAULT_MAX_SHARD_TOTAL_BYTES).unwrap_or_else(|e| {
            // Signature preserved, so this cannot return an error. Rather than
            // keep an invalid limit, fall back to the built-in defaults and say
            // so loudly. Callers wanting the error should use `with_limits`.
            warn!(
                error = %e,
                "invalid fetch limits; falling back to built-in defaults"
            );
            Self::with_limits(DEFAULT_MAX_SHARD_MSG_BYTES, DEFAULT_MAX_SHARD_TOTAL_BYTES)
                .expect("built-in defaults are valid")
        })
    }

    /// Create a fetch manager with both limits stated explicitly.
    ///
    /// `max_shard_msg_bytes` sizes each outbound range request.
    /// `max_shard_total_bytes` is the hard ceiling on the total bytes this
    /// fetcher will accumulate for one shard, whatever a peer claims.
    /// Rejects limits a [`omni_types::config::StoreConfig`] would reject,
    /// through the same rules, so a directly-constructed fetcher cannot hold
    /// a configuration the store would refuse.
    pub fn with_limits(max_shard_msg_bytes: usize, max_shard_total_bytes: u64) -> Result<Self> {
        omni_types::config::validate_shard_limits(max_shard_msg_bytes, max_shard_total_bytes)
            .map_err(StoreError::Other)?;
        Ok(Self {
            chunk_size: max_shard_msg_bytes as u64,
            active: HashMap::new(),
            max_shard_total_bytes,
        })
    }

    /// Start fetching a shard from a remote peer.
    ///
    /// Sends the first chunk request.  Subsequent chunks are requested
    /// automatically when [`Self::on_chunk_received`] is called.
    /// Start fetching a shard from a remote peer.
    pub async fn start_fetch(&mut self, net: &OmniNet, peer_id: PeerId, cid: String) -> Result<()> {
        self.start_fetch_announced(net, peer_id, cid, None).await
    }

    /// Start a fetch, pinning the size the gossip announcement advertised.
    ///
    /// Supplying `announced_size` lets the fetcher require an exact match
    /// instead of only bounding the total, so a peer can neither grow nor
    /// shrink the shard.
    pub async fn start_fetch_announced(
        &mut self,
        net: &OmniNet,
        peer_id: PeerId,
        cid: String,
        announced_size: Option<u64>,
    ) -> Result<()> {
        self.begin(&NetRequester(net), peer_id, cid, announced_size)
            .await
    }

    async fn begin<R: ChunkRequester>(
        &mut self,
        req: &R,
        peer_id: PeerId,
        cid: String,
        expected_size: Option<u64>,
    ) -> Result<()> {
        if self.active.contains_key(&cid) {
            return Err(StoreError::Other(format!(
                "fetch already in progress: {cid}"
            )));
        }

        info!(%cid, %peer_id, "starting shard fetch");

        req.request_chunk(peer_id, cid.clone(), 0, self.chunk_size)
            .await
            .map_err(StoreError::Other)?;

        self.active.insert(
            cid,
            FetchState {
                peer_id,
                total_bytes: None,
                next_offset: 0,
                buffer: Vec::new(),
                expected_size,
            },
        );

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
        self.process(&NetRequester(net), store, response).await
    }

    /// The whole handler, independent of how the next request is sent.
    async fn process<R: ChunkRequester>(
        &mut self,
        req: &R,
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

        // ── Validate metadata BEFORE any allocation or buffer mutation ──
        // `total_bytes` is peer-supplied. Reserving from it unchecked lets a
        // single response abort the process on a capacity overflow, so nothing
        // is reserved until the value has been bounded.
        let next_pos = match validate_response_metadata(
            response,
            state.next_offset,
            state.total_bytes,
            state.expected_size,
            self.max_shard_total_bytes,
        ) {
            Ok(next_pos) => next_pos,
            Err(msg) => {
                warn!(cid = %response.cid, error = %msg, "rejecting ShardResponse — invalid metadata");
                return self.reject(&response.cid, msg);
            }
        };

        if state.total_bytes.is_none() {
            state.total_bytes = Some(response.total_bytes);
        }

        // Grow only by what actually arrived, and fallibly.
        //
        // Reserving `total_bytes` up front would commit the whole declared
        // size on the first piece — bounded by the cap, but still a single
        // allocation an attacker chooses the size of. `try_reserve` grows by
        // the received length and returns an error instead of aborting when
        // the allocator refuses.
        // `data.len()` is already a `usize`; no conversion is needed or wanted.
        if let Err(e) = state.buffer.try_reserve(response.data.len()) {
            let msg = format!(
                "allocation failed reserving {} bytes: {e}",
                response.data.len()
            );
            warn!(cid = %response.cid, error = %msg, "rejecting ShardResponse");
            return self.reject(&response.cid, msg);
        }

        // Append chunk data.
        state.buffer.extend_from_slice(&response.data);
        // Reuse the position validation already computed with checked
        // arithmetic, rather than redoing `offset + len` unchecked.
        state.next_offset = next_pos;

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

            if let Err(e) = req
                .request_chunk(peer_id, cid.clone(), offset, chunk_size)
                .await
            {
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
    /// Abandon an in-progress fetch and report it failed.
    ///
    /// Dropping the state is what makes recovery possible: the accumulated
    /// buffer is released, and a later `start_fetch` for the same CID — from
    /// a different peer — is accepted rather than refused as already active.
    fn reject(&mut self, cid: &str, error: String) -> FetchOutcome {
        self.active.remove(cid);
        FetchOutcome::Failed {
            cid: cid.to_string(),
            error,
        }
    }

    pub fn is_active(&self, cid: &str) -> bool {
        self.active.contains_key(cid)
    }

    /// Number of in-progress fetches.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
}

/// Validate a [`ShardResponse`]'s metadata **before** any allocation.
///
/// Ported from SNIP's `sum-store::fetch::validate_response_metadata`, which
/// closed the equivalent defect there (SNIP issue #3). Omni never received
/// that fix: `total_bytes` is peer-supplied and was passed straight to
/// `Vec::reserve`, so one response could abort the process on a capacity
/// overflow or allocation failure.
///
/// `declared_total` is `None` on the first piece of a fetch and `Some(t)`
/// thereafter, where `t` is what the peer reported first.
///
/// This is a pure function — no I/O, no async — so every rule below is
/// exhaustively unit-testable without a swarm or a mock network.
pub(crate) fn validate_response_metadata(
    response: &ShardResponse,
    next_offset: u64,
    declared_total: Option<u64>,
    expected_size: Option<u64>,
    max_shard_total_bytes: u64,
) -> std::result::Result<u64, String> {
    let total = response.total_bytes;
    let data_len = u64::try_from(response.data.len())
        .map_err(|_| "ShardResponse.data.len() is not representable as u64".to_string())?;

    // (0) The one canonical empty completion.
    //
    //     An empty shard is publishable (`publication.rs` round-trips b"")
    //     and servable (`serve.rs` answers total_bytes = 0, error = None), so
    //     refusing every zero total made a legitimately empty shard
    //     unfetchable — the fetcher rejected an honest response.
    //
    //     Exactly one shape is accepted, and every clause is load-bearing:
    //     the fetch must not have started (`next_offset == 0`, so no prior
    //     non-empty response), the peer must place it at the beginning
    //     (`offset == 0`), it must carry no bytes, the total must be zero,
    //     **no total may have been declared yet** (`declared_total.is_none()`),
    //     and an announced size must be absent or zero. Anything else falls
    //     through to the ordinary rules below.
    //
    //     `declared_total.is_none()` is the strict initial-state condition and
    //     is deliberately narrower than "declared zero". A `Some(0)` means a
    //     total was already accepted for this fetch — but the contract is one
    //     initial response followed by immediate removal, so a fetch that has
    //     accepted a zero total cannot still be active. Admitting `Some(0)`
    //     would keep alive a duplicate/replayed completion state that
    //     production should never retain.
    //
    //     `Ok(0)` makes the handler complete immediately: `next_offset` stays
    //     0, `0 < 0` is false, so no follow-up piece is requested and the
    //     empty buffer goes straight to CID verification.
    if total == 0
        && data_len == 0
        && response.offset == 0
        && next_offset == 0
        && declared_total.is_none()
        && expected_size.map(|s| s == 0).unwrap_or(true)
    {
        return Ok(0);
    }

    // (1) Otherwise zero is not a valid total. Peers signal "nothing"
    //     through `error`, never through `total_bytes = 0`.
    if total == 0 {
        return Err("ShardResponse.total_bytes is zero".into());
    }

    // (2) The core defence: a peer cannot claim more than the configured
    //     ceiling. This is what bounds the subsequent `reserve`.
    if total > max_shard_total_bytes {
        return Err(format!(
            "ShardResponse.total_bytes = {total} exceeds safety bound {max_shard_total_bytes}"
        ));
    }

    // (3) When an announcement gave us the shard's size, `total_bytes` must
    //     match it exactly — the peer can neither grow nor shrink the shard.
    if let Some(expected) = expected_size {
        if total != expected {
            return Err(format!(
                "ShardResponse.total_bytes = {total} does not match announced size {expected}"
            ));
        }
    }

    // (4) The first response fixes `total_bytes`; later ones must agree. A
    //     peer that changes the value mid-stream is malicious.
    if let Some(prev_total) = declared_total {
        if total != prev_total {
            return Err(format!(
                "ShardResponse.total_bytes = {total} differs from previously declared {prev_total}"
            ));
        }
    }

    // (5) One piece cannot be larger than the whole shard.
    if data_len > total {
        return Err(format!(
            "ShardResponse.data.len() = {data_len} exceeds total_bytes = {total}"
        ));
    }

    // (6) Pieces must arrive in order.
    if response.offset != next_offset {
        return Err(format!(
            "ShardResponse.offset = {} does not match expected next offset {next_offset}",
            response.offset
        ));
    }

    // (7) Cumulative guard, with checked addition so a crafted offset cannot
    //     wrap u64 and appear to fit.
    let next_pos = next_offset
        .checked_add(data_len)
        .ok_or_else(|| "ShardResponse.offset + data.len() overflows u64".to_string())?;
    if next_pos > total {
        return Err(format!(
            "cumulative bytes {next_pos} would exceed total_bytes = {total}"
        ));
    }

    // (8) A successful zero-length piece is always rejected.
    //
    //     Before completion it makes no progress — `next_offset` does not
    //     advance, so the fetcher re-requests the same offset forever. At
    //     completion it is unreachable: a fetch that reaches
    //     `next_offset == total` is removed immediately, so no active fetch
    //     can be sitting at the end waiting for more. Peers report "nothing
    //     to send" through `error`, never through an empty successful piece.
    //
    //     The sole exception is the canonical empty completion handled in
    //     (0), which returns before reaching here.
    if data_len == 0 {
        return Err(format!(
            "zero-length piece at offset {next_offset} with {total} total bytes makes no progress"
        ));
    }

    Ok(next_pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content_id::cid_from_data;
    use std::cell::RefCell;

    const MAX: u64 = 1024 * 1024 * 1024;

    /// Records requests instead of sending them. No socket, no swarm, so the
    /// handler tests run anywhere — including where binding is prohibited.
    #[derive(Default)]
    pub(super) struct FakeRequester {
        pub(super) sent: RefCell<Vec<(String, u64, u64)>>,
        pub(super) fail_with: Option<String>,
    }

    impl FakeRequester {
        pub(super) fn failing(msg: &str) -> Self {
            Self {
                sent: RefCell::new(Vec::new()),
                fail_with: Some(msg.to_string()),
            }
        }
        pub(super) fn count(&self) -> usize {
            self.sent.borrow().len()
        }
    }

    impl ChunkRequester for FakeRequester {
        async fn request_chunk(
            &self,
            _peer_id: PeerId,
            cid: String,
            offset: u64,
            len: u64,
        ) -> std::result::Result<(), String> {
            if let Some(e) = &self.fail_with {
                return Err(e.clone());
            }
            self.sent.borrow_mut().push((cid, offset, len));
            Ok(())
        }
    }

    pub(super) fn store() -> (tempfile::TempDir, ShardStore) {
        let dir = tempfile::tempdir().unwrap();
        let s = ShardStore::new(dir.path().to_path_buf()).unwrap();
        (dir, s)
    }

    pub(super) fn resp(cid: &str, total: u64, offset: u64, data: Vec<u8>) -> ShardResponse {
        ShardResponse {
            cid: cid.to_string(),
            data,
            offset,
            total_bytes: total,
            error: None,
        }
    }

    pub(super) fn manager() -> FetchManager {
        FetchManager::with_limits(64 * 1024, MAX).unwrap()
    }

    /// Seed an active fetch through the ordinary code path — no test-only
    /// public API, so nothing here can leak into a production build.
    pub(super) async fn seeded(
        m: &mut FetchManager,
        req: &FakeRequester,
        cid: &str,
        announced: Option<u64>,
    ) {
        m.begin(req, PeerId::random(), cid.to_string(), announced)
            .await
            .unwrap();
    }

    // ── configuration ───────────────────────────────────────────────────

    #[test]
    fn the_default_total_ceiling_is_one_gib() {
        use omni_types::config::{StoreConfig, DEFAULT_MAX_SHARD_TOTAL_BYTES};
        assert_eq!(DEFAULT_MAX_SHARD_TOTAL_BYTES, 1024 * 1024 * 1024);
        assert_eq!(
            StoreConfig::default().max_shard_total_bytes,
            1024 * 1024 * 1024
        );
    }

    #[test]
    fn the_preserved_constructor_uses_the_default_ceiling() {
        use omni_types::config::DEFAULT_MAX_SHARD_TOTAL_BYTES;
        let m = FetchManager::new(64 * 1024);
        assert_eq!(m.max_shard_total_bytes, DEFAULT_MAX_SHARD_TOTAL_BYTES);
    }

    #[test]
    fn with_limits_rejects_zero_limits() {
        assert!(FetchManager::with_limits(0, MAX).is_err());
        assert!(FetchManager::with_limits(64 * 1024, 0).is_err());
        assert!(FetchManager::with_limits(64 * 1024, MAX).is_ok());
        // A total below the message size stays valid.
        assert!(FetchManager::with_limits(64 * 1024 * 1024, 1024).is_ok());
    }

    #[test]
    fn new_stays_infallible_and_never_holds_invalid_limits() {
        use omni_types::config::{DEFAULT_MAX_SHARD_MSG_BYTES, DEFAULT_MAX_SHARD_TOTAL_BYTES};
        let good = FetchManager::new(64 * 1024);
        assert_eq!(good.chunk_size, 64 * 1024);
        assert_eq!(good.max_shard_total_bytes, DEFAULT_MAX_SHARD_TOTAL_BYTES);

        // A zero message size would be invalid, so it falls back to defaults
        // rather than constructing an unusable fetcher.
        let fallback = FetchManager::new(0);
        assert_eq!(fallback.chunk_size, DEFAULT_MAX_SHARD_MSG_BYTES as u64);
        assert_eq!(
            fallback.max_shard_total_bytes,
            DEFAULT_MAX_SHARD_TOTAL_BYTES
        );
    }

    #[test]
    fn omni_store_validates_before_creating_the_store_directory() {
        use omni_types::config::StoreConfig;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("should-not-exist");

        let mut cfg = StoreConfig {
            store_dir: dir.clone(),
            ..StoreConfig::default()
        };
        cfg.max_shard_total_bytes = 0;

        let err = crate::OmniStore::new(cfg)
            .err()
            .expect("invalid config must fail");
        assert!(
            format!("{err}").contains("max_shard_total_bytes"),
            "unexpected error: {err}"
        );
        assert!(
            !dir.exists(),
            "an invalid configuration must not create the store directory"
        );

        // The same directory is created once the configuration is valid, so
        // the assertion above is about validation and not about the path.
        let ok_cfg = StoreConfig {
            store_dir: dir.clone(),
            ..StoreConfig::default()
        };
        assert!(crate::OmniStore::new(ok_cfg).is_ok());
        assert!(dir.exists());
    }

    #[test]
    fn config_validation_rejects_zero_limits_and_allows_a_small_total() {
        use omni_types::config::StoreConfig;
        let mut c = StoreConfig::default();
        assert!(c.validate().is_ok());

        c.max_shard_msg_bytes = 0;
        assert!(c.validate().is_err());

        c = StoreConfig::default();
        c.max_shard_total_bytes = 0;
        assert!(c.validate().is_err());

        // A total below the message size is unusual, not invalid: the message
        // size is the window a fetcher asks for, and a shard may be shorter.
        c = StoreConfig::default();
        c.max_shard_msg_bytes = 64 * 1024 * 1024;
        c.max_shard_total_bytes = 1024;
        assert!(
            c.validate().is_ok(),
            "a small total must not be rejected merely for being below the message size"
        );
    }

    // ── metadata rules ──────────────────────────────────────────────────

    fn r(total: u64, offset: u64, len: usize) -> ShardResponse {
        resp("cid", total, offset, vec![0u8; len])
    }

    #[test]
    fn zero_total_is_rejected_except_for_the_canonical_empty_shard() {
        // This test previously asserted that EVERY zero total is invalid.
        // That made a legitimately empty shard unfetchable, which rule (0)
        // now fixes, so the assertion is retargeted rather than dropped: the
        // canonical shape is accepted and every other zero total is still
        // refused.
        assert!(
            validate_response_metadata(&r(0, 0, 0), 0, None, None, MAX).is_ok(),
            "the canonical empty response is now the one accepted zero total"
        );
        // Zero total carrying data.
        assert!(validate_response_metadata(&r(0, 0, 8), 0, None, None, MAX).is_err());
        // Zero total at a non-zero offset.
        assert!(validate_response_metadata(&r(0, 8, 0), 0, None, None, MAX).is_err());
        // Zero total after the fetch advanced.
        assert!(validate_response_metadata(&r(0, 0, 0), 8, Some(0), None, MAX).is_err());
        // Zero total already declared — a replayed completion.
        assert!(validate_response_metadata(&r(0, 0, 0), 0, Some(0), None, MAX).is_err());
        // Zero total against a non-zero announced size.
        assert!(validate_response_metadata(&r(0, 0, 0), 0, None, Some(64), MAX).is_err());
    }

    #[test]
    fn a_malicious_total_is_rejected() {
        let err = validate_response_metadata(&r(u64::MAX, 0, 8), 0, None, None, MAX).unwrap_err();
        assert!(err.contains("exceeds safety bound"), "{err}");
    }

    #[test]
    fn the_bound_is_inclusive() {
        assert!(validate_response_metadata(&r(MAX, 0, 8), 0, None, None, MAX).is_ok());
        assert!(validate_response_metadata(&r(MAX + 1, 0, 8), 0, None, None, MAX).is_err());
    }

    #[test]
    fn an_announced_size_must_match_exactly() {
        assert!(validate_response_metadata(&r(2048, 0, 8), 0, None, Some(1024), MAX).is_err());
        assert!(validate_response_metadata(&r(512, 0, 8), 0, None, Some(1024), MAX).is_err());
        assert!(validate_response_metadata(&r(1024, 0, 8), 0, None, Some(1024), MAX).is_ok());
    }

    #[test]
    fn a_total_change_mid_stream_is_rejected() {
        let err =
            validate_response_metadata(&r(2048, 100, 8), 100, Some(1024), None, MAX).unwrap_err();
        assert!(err.contains("differs from previously declared"), "{err}");
    }

    #[test]
    fn a_piece_larger_than_the_shard_is_rejected() {
        assert!(validate_response_metadata(&r(100, 0, 200), 0, None, None, MAX).is_err());
    }

    #[test]
    fn out_of_order_and_replayed_pieces_are_rejected() {
        assert!(validate_response_metadata(&r(1024, 512, 8), 100, Some(1024), None, MAX).is_err());
        assert!(validate_response_metadata(&r(1024, 0, 512), 512, Some(1024), None, MAX).is_err());
    }

    #[test]
    fn cumulative_overrun_and_u64_overflow_are_rejected() {
        assert!(validate_response_metadata(&r(100, 80, 40), 80, Some(100), None, MAX).is_err());
        let err = validate_response_metadata(
            &r(100, u64::MAX - 1, 8),
            u64::MAX - 1,
            Some(100),
            None,
            MAX,
        )
        .unwrap_err();
        assert!(
            err.contains("overflows u64") || err.contains("would exceed"),
            "{err}"
        );
    }

    #[test]
    fn every_empty_piece_is_rejected() {
        // Before completion it loops the exchange; at completion it cannot
        // occur, because a finished fetch is removed immediately.
        assert!(validate_response_metadata(&r(1024, 512, 0), 512, Some(1024), None, MAX).is_err());
        assert!(
            validate_response_metadata(&r(1024, 1024, 0), 1024, Some(1024), None, MAX).is_err()
        );
    }

    #[test]
    fn valid_pieces_return_the_checked_next_position() {
        assert_eq!(
            validate_response_metadata(&r(1024, 0, 512), 0, None, None, MAX).unwrap(),
            512
        );
        assert_eq!(
            validate_response_metadata(&r(1024, 512, 512), 512, Some(1024), None, MAX).unwrap(),
            1024
        );
    }

    // ── the handler itself, socket-free ─────────────────────────────────

    #[tokio::test]
    async fn handler_rejects_a_malicious_total_and_clears_the_fetch() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();
        seeded(&mut m, &req, "cid-a", None).await;

        let bad = resp("cid-a", u64::MAX, 0, vec![0u8; 8]);
        match m.process(&req, &st, &bad).await {
            FetchOutcome::Failed { cid, error } => {
                assert_eq!(cid, "cid-a");
                assert!(error.contains("exceeds safety bound"), "{error}");
            }
            _ => panic!("expected Failed"),
        }
        assert!(!m.is_active("cid-a"));
    }

    // ── the canonical empty completion (rule 0) ─────────────────────────

    #[test]
    fn the_canonical_empty_response_is_accepted() {
        // total 0, offset 0, no data, nothing declared, nothing announced.
        let empty = resp("cid", 0, 0, Vec::new());
        assert_eq!(
            validate_response_metadata(&empty, 0, None, None, MAX).unwrap(),
            0,
            "next position must stay 0 so the handler completes immediately"
        );
        // An announced size of exactly zero is equally canonical.
        assert_eq!(
            validate_response_metadata(&empty, 0, None, Some(0), MAX).unwrap(),
            0
        );
        // But a previously declared total is NOT canonical, even if zero: the
        // contract is one initial response then immediate removal, so an
        // active fetch cannot already have accepted a total.
        assert!(
            validate_response_metadata(&empty, 0, Some(0), Some(0), MAX).is_err(),
            "declared_total must be None for the canonical empty completion"
        );
    }

    #[test]
    fn every_non_canonical_empty_response_is_still_rejected() {
        // Each case flips exactly one clause of rule (0).
        /// (label, response, next_offset, declared_total, expected_size)
        type Case = (&'static str, ShardResponse, u64, Option<u64>, Option<u64>);
        let cases: Vec<Case> = vec![
            // A replayed empty response AFTER the fetch advanced. Response
            // offset is 0, so only the `next_offset == 0` clause of rule (0)
            // rejects it — this case isolates that clause specifically.
            (
                "replayed after progress",
                resp("cid", 0, 0, Vec::new()),
                8,
                Some(0),
                None,
            ),
            // zero total but the fetch already advanced, at a later offset
            ("mid-fetch", resp("cid", 0, 8, Vec::new()), 8, Some(0), None),
            // zero total at a non-zero response offset
            (
                "nonzero offset",
                resp("cid", 0, 8, Vec::new()),
                0,
                None,
                None,
            ),
            // zero-length piece against a non-zero total
            (
                "nonzero total",
                resp("cid", 1024, 0, Vec::new()),
                0,
                None,
                None,
            ),
            // announced size disagrees
            (
                "announced nonzero",
                resp("cid", 0, 0, Vec::new()),
                0,
                None,
                Some(1024),
            ),
            // the peer previously declared a different total
            (
                "total changed",
                resp("cid", 0, 0, Vec::new()),
                0,
                Some(1024),
                None,
            ),
        ];
        for (label, r, next_offset, declared, expected) in cases {
            assert!(
                validate_response_metadata(&r, next_offset, declared, expected, MAX).is_err(),
                "case '{label}' must remain invalid"
            );
        }
    }

    #[tokio::test]
    async fn handler_completes_an_empty_fetch_without_requesting_more() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();

        let cid = cid_from_data(b"");
        seeded(&mut m, &req, &cid, Some(0)).await;
        let issued_by_begin = req.count();

        let empty = resp(&cid, 0, 0, Vec::new());
        match m.process(&req, &st, &empty).await {
            FetchOutcome::Complete { cid: c, size } => {
                assert_eq!(c, cid);
                assert_eq!(size, 0);
            }
            other => panic!(
                "expected Complete, got failed={}",
                matches!(other, FetchOutcome::Failed { .. })
            ),
        }
        assert!(st.has(&cid), "the empty shard must be published");
        assert!(!m.is_active(&cid));
        assert_eq!(
            req.count(),
            issued_by_begin,
            "no follow-up piece may be requested for an empty shard"
        );
    }

    #[tokio::test]
    async fn handler_rejects_an_empty_response_whose_cid_is_wrong() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();

        // A CID that is NOT the CID of empty content.
        let wrong = cid_from_data(b"not empty");
        seeded(&mut m, &req, &wrong, Some(0)).await;

        let empty = resp(&wrong, 0, 0, Vec::new());
        match m.process(&req, &st, &empty).await {
            FetchOutcome::Failed { error, .. } => {
                assert!(error.contains("integrity"), "{error}")
            }
            _ => panic!("an empty body must still be CID-verified before publication"),
        }
        assert!(
            !st.has(&wrong),
            "nothing may be published on a failed verification"
        );
    }

    #[tokio::test]
    async fn handler_rejects_an_empty_piece() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();
        seeded(&mut m, &req, "cid-a", None).await;

        let empty = resp("cid-a", 1024, 0, Vec::new());
        match m.process(&req, &st, &empty).await {
            FetchOutcome::Failed { error, .. } => {
                assert!(error.contains("makes no progress"), "{error}")
            }
            _ => panic!("expected Failed"),
        }
    }

    #[tokio::test]
    async fn handler_enforces_the_announced_size() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();
        seeded(&mut m, &req, "cid-a", Some(1024)).await;

        let wrong = resp("cid-a", 4096, 0, vec![0u8; 8]);
        match m.process(&req, &st, &wrong).await {
            FetchOutcome::Failed { error, .. } => {
                assert!(
                    error.contains("does not match announced size 1024"),
                    "{error}"
                )
            }
            _ => panic!("expected Failed"),
        }
    }

    #[tokio::test]
    async fn handler_advances_and_requests_the_next_piece() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();
        seeded(&mut m, &req, "cid-a", None).await;
        assert_eq!(req.count(), 1, "begin() issues the first request");

        let first = resp("cid-a", 1024, 0, vec![7u8; 512]);
        assert!(matches!(
            m.process(&req, &st, &first).await,
            FetchOutcome::InProgress
        ));

        let sent = req.sent.borrow();
        assert_eq!(sent.len(), 2, "a follow-up request must be issued");
        assert_eq!(
            sent[1].1, 512,
            "and it must start at the checked next position"
        );
    }

    #[tokio::test]
    async fn handler_completes_and_persists_a_valid_fetch() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();

        let data = b"a whole small shard".to_vec();
        let cid = cid_from_data(&data);
        seeded(&mut m, &req, &cid, Some(data.len() as u64)).await;

        let whole = resp(&cid, data.len() as u64, 0, data.clone());
        match m.process(&req, &st, &whole).await {
            FetchOutcome::Complete { cid: c, size } => {
                assert_eq!(c, cid);
                assert_eq!(size, data.len() as u64);
            }
            _ => panic!("expected Complete"),
        }
        assert!(st.has(&cid));
        assert!(!m.is_active(&cid));
    }

    #[tokio::test]
    async fn handler_rejects_only_the_affected_fetch() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();
        seeded(&mut m, &req, "cid-bad", None).await;
        seeded(&mut m, &req, "cid-good", None).await;
        assert_eq!(m.active_count(), 2);

        let bad = resp("cid-bad", u64::MAX, 0, vec![0u8; 8]);
        assert!(matches!(
            m.process(&req, &st, &bad).await,
            FetchOutcome::Failed { .. }
        ));
        assert!(!m.is_active("cid-bad"));
        assert!(m.is_active("cid-good"), "an unrelated fetch must survive");
    }

    #[tokio::test]
    async fn handler_surfaces_a_server_side_error() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();
        seeded(&mut m, &req, "cid-a", None).await;

        let mut err_resp = resp("cid-a", 0, 0, Vec::new());
        err_resp.error = Some("no such shard".into());
        match m.process(&req, &st, &err_resp).await {
            FetchOutcome::Failed { error, .. } => assert_eq!(error, "no such shard"),
            _ => panic!("expected Failed"),
        }
        assert!(!m.is_active("cid-a"));
    }

    #[tokio::test]
    async fn handler_fails_the_fetch_when_the_next_request_cannot_be_sent() {
        let (_d, st) = store();
        let ok = FakeRequester::default();
        let mut m = manager();
        seeded(&mut m, &ok, "cid-a", None).await;

        let broken = FakeRequester::failing("transport down");
        let first = resp("cid-a", 1024, 0, vec![7u8; 512]);
        match m.process(&broken, &st, &first).await {
            FetchOutcome::Failed { error, .. } => assert!(error.contains("transport down")),
            _ => panic!("expected Failed"),
        }
        assert!(!m.is_active("cid-a"));
    }

    #[tokio::test]
    async fn an_unparented_response_is_reported_not_panicked() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();
        let orphan = resp("never-started", 1024, 0, vec![0u8; 8]);
        assert!(matches!(
            m.process(&req, &st, &orphan).await,
            FetchOutcome::Failed { .. }
        ));
        assert_eq!(m.active_count(), 0);
    }
}

/// Allocation-failure coverage, running under an ordinary `cargo test`.
///
/// Installs a global allocator that refuses large allocations **only while
/// armed on the current thread**, so the other tests in this binary are
/// unaffected. `try_reserve` then returns `Err` where a plain `reserve` would
/// call `handle_alloc_error` and abort the process — which is the difference
/// this change is about.
#[cfg(test)]
mod allocation_failure_tests {
    use super::tests::*;
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static FAIL_AT_OR_ABOVE: Cell<usize> = const { Cell::new(usize::MAX) };
    }

    pub(super) struct Failing;

    unsafe impl GlobalAlloc for Failing {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let armed = FAIL_AT_OR_ABOVE.try_with(|c| c.get()).unwrap_or(usize::MAX);
            if layout.size() >= armed {
                return std::ptr::null_mut();
            }
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let armed = FAIL_AT_OR_ABOVE.try_with(|c| c.get()).unwrap_or(usize::MAX);
            if new_size >= armed {
                return std::ptr::null_mut();
            }
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    /// Arms the failing allocator for as long as it is held.
    ///
    /// An RAII guard rather than manual arm/disarm: a panic between the two
    /// would otherwise leave allocation failure enabled for every later test
    /// on this thread, turning one failure into a cascade of unrelated ones.
    struct ArmGuard(usize);

    impl ArmGuard {
        fn new(threshold: usize) -> Self {
            let previous = FAIL_AT_OR_ABOVE.with(|c| c.replace(threshold));
            ArmGuard(previous)
        }
    }

    impl Drop for ArmGuard {
        fn drop(&mut self) {
            // Restores the previous threshold, so nesting is safe too.
            FAIL_AT_OR_ABOVE.with(|c| c.set(self.0));
        }
    }

    fn armed<T>(threshold: usize, f: impl FnOnce() -> T) -> T {
        let _guard = ArmGuard::new(threshold);
        f()
    }

    #[test]
    fn the_harness_fails_only_large_allocations() {
        // Without this the test below could pass vacuously.
        let refused = armed(1 << 20, || Vec::<u8>::new().try_reserve(4 << 20).is_err());
        let allowed = armed(1 << 20, || Vec::<u8>::new().try_reserve(1024).is_ok());
        assert!(refused, "a large reservation must be refused while armed");
        assert!(allowed, "a small one must still succeed");
        assert!(
            Vec::<u8>::new().try_reserve(4 << 20).is_ok(),
            "and everything must succeed once disarmed"
        );
    }

    #[test]
    fn the_guard_restores_the_threshold_even_on_panic() {
        // Without this, a panicking test would leave allocation failure armed
        // and every later test on this thread would fail for the wrong reason.
        let before = FAIL_AT_OR_ABOVE.with(|c| c.get());
        let r = std::panic::catch_unwind(|| {
            let _guard = ArmGuard::new(1024);
            panic!("boom");
        });
        assert!(r.is_err());
        assert_eq!(
            FAIL_AT_OR_ABOVE.with(|c| c.get()),
            before,
            "the guard must disarm on unwind"
        );
        assert!(Vec::<u8>::new().try_reserve(4 << 20).is_ok());
    }

    #[tokio::test]
    async fn allocation_failure_fails_only_the_affected_fetch() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();
        seeded(&mut m, &req, "cid-bad", None).await;
        seeded(&mut m, &req, "cid-good", None).await;

        // Built before arming, so its own allocation succeeds.
        let piece = resp("cid-bad", 8 * 1024 * 1024, 0, vec![0u8; 2 * 1024 * 1024]);

        let outcome = {
            let _guard = ArmGuard::new(1024 * 1024);
            m.process(&req, &st, &piece).await
        };

        match outcome {
            FetchOutcome::Failed { cid, error } => {
                assert_eq!(cid, "cid-bad");
                assert!(error.contains("allocation failed"), "{error}");
            }
            _ => panic!("expected Failed — the reservation could not have succeeded"),
        }
        // Still alive (a plain `reserve` would have aborted here), the bad
        // fetch is cleared, and the unrelated one is untouched.
        assert!(!m.is_active("cid-bad"));
        assert!(m.is_active("cid-good"), "an unrelated fetch must survive");
        assert_eq!(m.active_count(), 1);
    }
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOC: allocation_failure_tests::Failing = allocation_failure_tests::Failing;
