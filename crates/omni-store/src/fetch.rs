//! Outbound shard fetch orchestration.
//!
//! [`FetchManager`] tracks in-progress fetches.  Each fetch proceeds as a
//! sequence of 64 MiB windowed request-response round-trips. Every piece is
//! written to an attempt-owned staging file under the store root and hashed
//! as it arrives (issue #132), so a fetch never holds the shard in memory.
//! When the last byte is in, the running BLAKE3 must equal the CID; only then
//! is the staged file published, through the store's verifying, no-clobber
//! publication path.
//!
//! At most `max_concurrent_fetches` fetches are in flight at once (issue
//! #132; [`omni_types::config::StoreConfig::max_concurrent_fetches`], default
//! 4). A fetch started while every slot is held waits in a first-in,
//! first-out queue, and is started (its first request sent) as soon as a
//! slot frees. A fetch frees its slot when it completes, fails, is cancelled
//! ([`FetchManager::cancel`], [`FetchManager::on_request_failed`]) or is
//! dropped with the manager. A store therefore stages at most
//! `max_concurrent_fetches × max_shard_total_bytes` for this path.

use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

use libp2p::PeerId;
use omni_net::{OmniNet, ShardResponse};
use omni_types::config::StoreConfig;
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
use crate::content_id;

/// Tracks in-flight fetches keyed by CID, and the fetches waiting for a slot.
pub struct FetchManager {
    /// Maximum bytes per request-response chunk (default 64 MiB).
    chunk_size: u64,
    /// In-progress fetches: CID → state. Never more than
    /// `max_concurrent_fetches` entries.
    active: HashMap<String, FetchState>,
    /// Fetches waiting for a slot, oldest first. Nothing is requested or
    /// staged for them yet.
    queued: VecDeque<Queued>,
    /// Queued fetches whose first request could not be sent when a slot
    /// freed, as `FetchOutcome::Failed`, until the caller takes them.
    failed_starts: Vec<FetchOutcome>,
    /// Hard ceiling on total accumulated bytes for one shard.
    max_shard_total_bytes: u64,
    /// How many fetches may be in flight at once (issue #132).
    max_concurrent_fetches: usize,
}

/// A fetch waiting for a slot.
struct Queued {
    peer_id: PeerId,
    cid: String,
    expected_size: Option<u64>,
}

/// State for a single in-progress shard fetch.
struct FetchState {
    peer_id: PeerId,
    /// Total shard size (learned from first response).
    total_bytes: Option<u64>,
    /// Byte offset of the next chunk to request.
    next_offset: u64,
    /// The bytes received so far, on disk; created with the first accepted
    /// piece, when the store (and so its root) is known.
    staged: Option<Staged>,
    /// Shard size from the gossip announcement, when one was seen.
    expected_size: Option<u64>,
}

/// One fetch's partial shard: an attempt-owned directory under the store
/// root holding a `partial` file, and the running BLAKE3 of its contents.
/// Dropping it (a rejected, failed or abandoned fetch) removes the directory.
struct Staged {
    dir: tempfile::TempDir,
    file: File,
    hasher: blake3::Hasher,
}

impl Staged {
    fn new(root: &Path) -> io::Result<Self> {
        let dir = tempfile::Builder::new()
            .prefix(".fetch-")
            .tempdir_in(root)?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.path().join("partial"))?;
        Ok(Self {
            dir,
            file,
            hasher: blake3::Hasher::new(),
        })
    }

    fn append(&mut self, data: &[u8]) -> io::Result<()> {
        self.file.write_all(data)?;
        self.hasher.update(data);
        Ok(())
    }
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
    /// [`omni_types::config::DEFAULT_MAX_SHARD_TOTAL_BYTES`] and the
    /// concurrency bound to
    /// [`omni_types::config::DEFAULT_MAX_CONCURRENT_FETCHES`]. Use
    /// [`FetchManager::with_limits`] or [`FetchManager::from_config`] to set
    /// them explicitly.
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
    /// a configuration the store would refuse. At most
    /// [`omni_types::config::DEFAULT_MAX_CONCURRENT_FETCHES`] fetches run at
    /// once.
    pub fn with_limits(max_shard_msg_bytes: usize, max_shard_total_bytes: u64) -> Result<Self> {
        use omni_types::config::DEFAULT_MAX_CONCURRENT_FETCHES;
        Self::build(
            max_shard_msg_bytes,
            max_shard_total_bytes,
            DEFAULT_MAX_CONCURRENT_FETCHES,
        )
    }

    /// Create a fetch manager with every limit taken from `config`, which is
    /// validated first.
    pub fn from_config(config: &StoreConfig) -> Result<Self> {
        config.validate().map_err(StoreError::Other)?;
        Self::build(
            config.max_shard_msg_bytes,
            config.max_shard_total_bytes,
            config.max_concurrent_fetches,
        )
    }

    fn build(
        max_shard_msg_bytes: usize,
        max_shard_total_bytes: u64,
        max_concurrent_fetches: usize,
    ) -> Result<Self> {
        omni_types::config::validate_shard_limits(max_shard_msg_bytes, max_shard_total_bytes)
            .map_err(StoreError::Other)?;
        omni_types::config::validate_max_concurrent_fetches(max_concurrent_fetches)
            .map_err(StoreError::Other)?;
        Ok(Self {
            chunk_size: max_shard_msg_bytes as u64,
            active: HashMap::new(),
            queued: VecDeque::new(),
            failed_starts: Vec::new(),
            max_shard_total_bytes,
            max_concurrent_fetches,
        })
    }

    /// Start fetching a shard from a remote peer.
    ///
    /// If a slot is free (and no earlier fetch is waiting), the first chunk
    /// request is sent now; subsequent chunks are requested automatically
    /// when [`Self::on_chunk_received`] is called. Otherwise the fetch waits
    /// in line ([`Self::is_queued`]) and is started when a slot frees. An
    /// error means the fetch is neither in flight nor queued: the CID is
    /// already being fetched, or its first request could not be sent.
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
        if self.active.contains_key(&cid) || self.is_queued(&cid) {
            return Err(StoreError::Other(format!(
                "fetch already in progress: {cid}"
            )));
        }

        // Every fetch enters the line; one with a free slot leaves it at once.
        self.queued.push_back(Queued {
            peer_id,
            cid: cid.clone(),
            expected_size,
        });
        self.fill_slots(req).await;

        if self.is_queued(&cid) {
            info!(
                %cid,
                %peer_id,
                in_flight = self.active.len(),
                queued = self.queued.len(),
                "shard fetch queued: every fetch slot is in use"
            );
            return Ok(());
        }
        // Not active and not queued: its own first request failed. That is
        // this call's error, not a later promotion's.
        if let Some(i) = self
            .failed_starts
            .iter()
            .rposition(|o| matches!(o, FetchOutcome::Failed { cid: c, .. } if *c == cid))
        {
            if let FetchOutcome::Failed { error, .. } = self.failed_starts.remove(i) {
                return Err(StoreError::Other(error));
            }
        }
        Ok(())
    }

    /// Start queued fetches, oldest first, while slots are free.
    ///
    /// A queued fetch leaves the line only once its first request is sent
    /// (it becomes active) or has failed (it is recorded in
    /// `failed_starts`). If this future is dropped while a request is being
    /// sent, that fetch stays first in line and holds no slot.
    async fn fill_slots<R: ChunkRequester>(&mut self, req: &R) {
        while self.active.len() < self.max_concurrent_fetches {
            let Some(next) = self.queued.front() else {
                break;
            };
            let (peer_id, cid) = (next.peer_id, next.cid.clone());
            let sent = req
                .request_chunk(peer_id, cid.clone(), 0, self.chunk_size)
                .await;
            let next = self.queued.pop_front().expect("still first in line");
            match sent {
                Ok(()) => {
                    info!(%cid, %peer_id, "starting shard fetch");
                    self.active.insert(
                        cid,
                        FetchState {
                            peer_id,
                            total_bytes: None,
                            next_offset: 0,
                            staged: None,
                            expected_size: next.expected_size,
                        },
                    );
                }
                Err(e) => {
                    warn!(%cid, %peer_id, error = %e, "shard fetch could not start");
                    self.failed_starts
                        .push(FetchOutcome::Failed { cid, error: e });
                }
            }
        }
    }

    /// Process a received shard chunk.
    ///
    /// Returns a [`FetchOutcome`] indicating whether the fetch is complete,
    /// still in progress, or has failed. A fetch that ends frees its slot,
    /// and the oldest queued fetch is started in it before this returns; if
    /// that start fails, see [`Self::take_failed_starts`].
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
        let outcome = self.process_response(req, store, response).await;
        // A fetch that ended has left `active`: its slot goes to the queue.
        self.fill_slots(req).await;
        outcome
    }

    async fn process_response<R: ChunkRequester>(
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

        // Write the piece to the staging file and hash it (issue #132). No
        // allocation grows with the shard: memory holds one message at a
        // time, and the validated total bounds what a fetch may stage on disk.
        if state.staged.is_none() {
            match Staged::new(store.root()) {
                Ok(staged) => state.staged = Some(staged),
                Err(e) => {
                    let msg = format!("cannot stage the shard: {e}");
                    warn!(cid = %response.cid, error = %msg, "rejecting ShardResponse");
                    return self.reject(&response.cid, msg);
                }
            }
        }
        let staged = state.staged.as_mut().expect("created above");
        if let Err(e) = staged.append(&response.data) {
            let msg = format!(
                "cannot stage {} bytes at offset {}: {e}",
                response.data.len(),
                state.next_offset
            );
            warn!(cid = %response.cid, error = %msg, "rejecting ShardResponse");
            return self.reject(&response.cid, msg);
        }
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

        // All chunks received: the running hash must be the CID, then the
        // staged file is published. Every early return drops `staged`, which
        // removes the staging directory.
        let cid = response.cid.clone();
        let state = self.active.remove(&cid).unwrap();
        let size = state.next_offset;
        let Staged { dir, file, hasher } = state.staged.expect("every accepted piece is staged");
        drop(file);
        let partial = dir.path().join("partial");
        let hash = hasher.finalize();

        let actual = content_id::cid_from_blake3_hash(&hash);
        if actual != cid {
            let e = StoreError::IntegrityMismatch {
                expected: cid.clone(),
                actual,
            };
            return FetchOutcome::Failed {
                cid,
                error: format!("integrity check failed: {e}"),
            };
        }

        // The store re-verifies the file from disk, syncs it and links it
        // under its final name without replacing an existing one.
        let hash_hex = hash.to_hex().to_string();
        if let Err(e) = store.publish_download(&cid, &hash_hex, |path| {
            std::fs::rename(&partial, path)?;
            Ok(())
        }) {
            return FetchOutcome::Failed {
                cid,
                error: format!("failed to write shard: {e}"),
            };
        }
        drop(dir);

        info!(%cid, size, "shard fetch complete and verified");
        FetchOutcome::Complete { cid, size }
    }

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

    /// Cancel the fetch of `cid`, in flight or queued. An in-flight fetch
    /// loses its staging and frees its slot, which goes to the oldest queued
    /// fetch. Returns whether there was such a fetch.
    pub async fn cancel(&mut self, net: &OmniNet, cid: &str) -> bool {
        self.cancel_with(&NetRequester(net), cid).await
    }

    async fn cancel_with<R: ChunkRequester>(&mut self, req: &R, cid: &str) -> bool {
        let found = if self.active.remove(cid).is_some() {
            info!(%cid, "shard fetch cancelled");
            true
        } else if let Some(i) = self.queued.iter().position(|q| q.cid == cid) {
            self.queued.remove(i);
            info!(%cid, "queued shard fetch cancelled");
            true
        } else {
            false
        };
        self.fill_slots(req).await;
        found
    }

    /// A request to `peer_id` failed in the transport
    /// (`OmniNetEvent::ShardRequestFailed`, which names no CID). Every fetch
    /// in flight from that peer fails, freeing its slot for the queue; they
    /// can be started again, from this peer or another. Returns their
    /// outcomes; queued fetches from that peer stay in line.
    pub async fn on_request_failed(
        &mut self,
        net: &OmniNet,
        peer_id: PeerId,
        error: &str,
    ) -> Vec<FetchOutcome> {
        self.request_failed_with(&NetRequester(net), peer_id, error)
            .await
    }

    async fn request_failed_with<R: ChunkRequester>(
        &mut self,
        req: &R,
        peer_id: PeerId,
        error: &str,
    ) -> Vec<FetchOutcome> {
        let mut cids: Vec<String> = self
            .active
            .iter()
            .filter(|(_, state)| state.peer_id == peer_id)
            .map(|(cid, _)| cid.clone())
            .collect();
        cids.sort();
        let failed = cids
            .into_iter()
            .map(|cid| {
                warn!(%cid, %peer_id, %error, "shard request failed");
                self.reject(&cid, format!("shard request to {peer_id} failed: {error}"))
            })
            .collect();
        self.fill_slots(req).await;
        failed
    }

    /// Take the queued fetches that could not start when a slot freed (their
    /// first request could not be sent), as `FetchOutcome::Failed`. They are
    /// neither in flight nor queued any more.
    pub fn take_failed_starts(&mut self) -> Vec<FetchOutcome> {
        std::mem::take(&mut self.failed_starts)
    }

    /// Check if a fetch is currently in flight for the given CID.
    pub fn is_active(&self, cid: &str) -> bool {
        self.active.contains_key(cid)
    }

    /// Whether a fetch for `cid` is waiting for a slot.
    pub fn is_queued(&self, cid: &str) -> bool {
        self.queued.iter().any(|q| q.cid == cid)
    }

    /// Number of in-progress fetches. Never more than
    /// [`Self::max_concurrent_fetches`].
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Number of fetches waiting for a slot.
    pub fn queued_count(&self) -> usize {
        self.queued.len()
    }

    /// How many fetches may be in flight at once.
    pub fn max_concurrent_fetches(&self) -> usize {
        self.max_concurrent_fetches
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

    /// The `.fetch-*` staging directories under the store root.
    pub(super) fn staging_dirs(st: &ShardStore) -> Vec<std::path::PathBuf> {
        std::fs::read_dir(st.root())
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with(".fetch-")
            })
            .collect()
    }

    /// Issue #132: what an in-progress fetch has received is on disk, in an
    /// attempt-owned staging directory under the store root, not in memory;
    /// nothing is published until the whole shard hashes to its CID.
    #[tokio::test]
    async fn an_in_progress_fetch_holds_its_bytes_on_disk() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();
        let data: Vec<u8> = (0..1024u32).map(|i| i as u8).collect();
        let cid = cid_from_data(&data);
        seeded(&mut m, &req, &cid, Some(1024)).await;

        let first = resp(&cid, 1024, 0, data[..512].to_vec());
        assert!(matches!(
            m.process(&req, &st, &first).await,
            FetchOutcome::InProgress
        ));
        let dirs = staging_dirs(&st);
        assert_eq!(dirs.len(), 1, "one staging directory per fetch: {dirs:?}");
        assert_eq!(
            std::fs::read(dirs[0].join("partial")).unwrap(),
            &data[..512]
        );
        assert!(!st.has(&cid), "nothing is published before verification");

        let rest = resp(&cid, 1024, 512, data[512..].to_vec());
        match m.process(&req, &st, &rest).await {
            FetchOutcome::Complete { size, .. } => assert_eq!(size, 1024),
            _ => panic!("expected Complete"),
        }
        assert_eq!(st.get(&cid).unwrap(), data);
        assert!(
            staging_dirs(&st).is_empty(),
            "staging is removed once published"
        );
    }

    /// A shard whose bytes do not hash to its CID is refused at its last byte:
    /// nothing is published and no staging is left behind.
    #[tokio::test]
    async fn a_corrupted_multi_piece_shard_publishes_nothing() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();
        let cid = cid_from_data(&[1u8; 1024]);
        seeded(&mut m, &req, &cid, Some(1024)).await;

        let good = resp(&cid, 1024, 0, vec![1u8; 512]);
        assert!(matches!(
            m.process(&req, &st, &good).await,
            FetchOutcome::InProgress
        ));
        let corrupt = resp(&cid, 1024, 512, vec![2u8; 512]);
        match m.process(&req, &st, &corrupt).await {
            FetchOutcome::Failed { error, .. } => assert!(error.contains("integrity"), "{error}"),
            _ => panic!("expected Failed"),
        }
        assert!(!st.has(&cid));
        assert!(!m.is_active(&cid));
        assert!(staging_dirs(&st).is_empty());
    }

    /// A fetch that ends early — rejected mid-stream, unable to send its next
    /// request, or dropped with its manager — removes its staging at once.
    #[tokio::test]
    async fn an_ended_fetch_leaves_no_staging_behind() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();

        seeded(&mut m, &req, "cid-a", None).await;
        let first = resp("cid-a", 1024, 0, vec![7u8; 512]);
        assert!(matches!(
            m.process(&req, &st, &first).await,
            FetchOutcome::InProgress
        ));
        assert_eq!(staging_dirs(&st).len(), 1);
        let changed_total = resp("cid-a", 2048, 512, vec![7u8; 512]);
        assert!(matches!(
            m.process(&req, &st, &changed_total).await,
            FetchOutcome::Failed { .. }
        ));
        assert!(staging_dirs(&st).is_empty(), "rejected mid-stream");

        seeded(&mut m, &req, "cid-b", None).await;
        let broken = FakeRequester::failing("transport down");
        let first_b = resp("cid-b", 1024, 0, vec![7u8; 512]);
        assert!(matches!(
            m.process(&broken, &st, &first_b).await,
            FetchOutcome::Failed { .. }
        ));
        assert!(staging_dirs(&st).is_empty(), "next request not sent");

        seeded(&mut m, &req, "cid-c", None).await;
        let first_c = resp("cid-c", 1024, 0, vec![7u8; 512]);
        assert!(matches!(
            m.process(&req, &st, &first_c).await,
            FetchOutcome::InProgress
        ));
        assert_eq!(staging_dirs(&st).len(), 1);
        drop(m);
        assert!(staging_dirs(&st).is_empty(), "manager dropped mid-fetch");
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

    // ── the fetch-concurrency bound (issue #132) ────────────────────────

    /// A manager that runs at most `n` fetches at once.
    fn capped(n: usize) -> FetchManager {
        use omni_types::config::StoreConfig;
        FetchManager::from_config(&StoreConfig {
            max_shard_msg_bytes: 64 * 1024,
            max_shard_total_bytes: MAX,
            max_concurrent_fetches: n,
            ..StoreConfig::default()
        })
        .unwrap()
    }

    /// `count` distinct 1 KiB shards, as (CID, bytes), in start order.
    fn shards(count: u8) -> Vec<(String, Vec<u8>)> {
        (0..count)
            .map(|i| {
                let data: Vec<u8> = (0..1024u32).map(|j| (j as u8) ^ i).collect();
                (cid_from_data(&data), data)
            })
            .collect()
    }

    /// The CIDs of the first requests (offset 0) sent so far, in order.
    fn first_requests(req: &FakeRequester) -> Vec<String> {
        req.sent
            .borrow()
            .iter()
            .filter(|(_, offset, _)| *offset == 0)
            .map(|(cid, _, _)| cid.clone())
            .collect()
    }

    /// The honest answer to request `index`: at most 512 bytes from the
    /// requested offset, so every 1 KiB shard takes two pieces.
    fn answer(req: &FakeRequester, shards: &[(String, Vec<u8>)], index: usize) -> ShardResponse {
        let (cid, offset, _) = req.sent.borrow()[index].clone();
        let data = &shards.iter().find(|(c, _)| *c == cid).unwrap().1;
        let start = offset as usize;
        let end = (start + 512).min(data.len());
        resp(&cid, data.len() as u64, offset, data[start..end].to_vec())
    }

    /// Answer every request from index `from` on, oldest first, until none
    /// is left unanswered. After each piece, check that no more than `cap`
    /// fetches are in flight or staged. Returns the CIDs in completion order
    /// and the peak number of staging directories seen.
    async fn serve_all(
        m: &mut FetchManager,
        req: &FakeRequester,
        st: &ShardStore,
        shards: &[(String, Vec<u8>)],
        cap: usize,
        from: usize,
    ) -> (Vec<String>, usize) {
        let (mut served, mut peak) = (from, 0);
        let mut completed = Vec::new();
        while served < req.count() {
            let piece = answer(req, shards, served);
            served += 1;
            match m.process(req, st, &piece).await {
                FetchOutcome::InProgress => {}
                FetchOutcome::Complete { cid, .. } => completed.push(cid),
                FetchOutcome::Failed { cid, error } => panic!("{cid} failed: {error}"),
            }
            let staged = staging_dirs(st).len();
            peak = peak.max(staged);
            assert!(
                staged <= cap,
                "{staged} fetches staged at once with a bound of {cap}"
            );
            assert!(m.active_count() <= cap, "{} in flight", m.active_count());
        }
        (completed, peak)
    }

    /// The key test. With a bound of 2, five fetches started at once: two are
    /// in flight, three wait without sending anything, and at no point are
    /// more than two staged on disk. The waiting ones start in the order they
    /// were started and all complete. Without the bound, the third piece
    /// served creates a third staging directory.
    #[tokio::test]
    async fn never_more_than_the_bound_are_staged_and_queued_fetches_complete() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = capped(2);
        let set = shards(5);
        for (cid, data) in &set {
            m.begin(&req, PeerId::random(), cid.clone(), Some(data.len() as u64))
                .await
                .expect("an excess fetch queues, it is not refused");
        }
        assert_eq!((m.active_count(), m.queued_count()), (2, 3));
        assert_eq!(req.count(), 2, "a queued fetch sends nothing");
        assert!(m.is_queued(&set[2].0) && !m.is_active(&set[2].0));

        let (completed, peak) = serve_all(&mut m, &req, &st, &set, 2, 0).await;

        let order: Vec<String> = set.iter().map(|(cid, _)| cid.clone()).collect();
        assert_eq!(
            first_requests(&req),
            order,
            "queued fetches start first in, first out"
        );
        assert_eq!(completed, order, "every queued fetch completes");
        assert_eq!(peak, 2, "the bound was reached, so it was exercised");
        for (cid, data) in &set {
            assert_eq!(&st.get(cid).unwrap(), data);
        }
        assert_eq!((m.active_count(), m.queued_count()), (0, 0));
        assert!(staging_dirs(&st).is_empty());
    }

    /// A bound of 1 runs the fetches strictly one after another: the next one
    /// sends its first request only after the previous one has completed.
    #[tokio::test]
    async fn a_bound_of_one_serializes_fetches() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = capped(1);
        let set = shards(3);
        for (cid, _) in &set {
            m.begin(&req, PeerId::random(), cid.clone(), None)
                .await
                .unwrap();
        }
        assert_eq!((m.active_count(), m.queued_count()), (1, 2));

        let (completed, peak) = serve_all(&mut m, &req, &st, &set, 1, 0).await;

        let sequence: Vec<String> = req
            .sent
            .borrow()
            .iter()
            .map(|(c, _, _)| c.clone())
            .collect();
        let expected: Vec<String> = set
            .iter()
            .flat_map(|(cid, _)| [cid.clone(), cid.clone()])
            .collect();
        assert_eq!(
            sequence, expected,
            "each fetch's two pieces, one fetch at a time"
        );
        assert_eq!(completed.len(), 3);
        assert_eq!(peak, 1);
    }

    /// A fetch that fails, whichever way, frees its slot, and the oldest
    /// waiting fetch is started in it before the failure is returned.
    #[tokio::test]
    async fn a_failed_fetch_frees_its_slot_for_the_next_in_line() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = capped(1);
        let set = shards(4);
        for (cid, _) in &set {
            m.begin(&req, PeerId::random(), cid.clone(), None)
                .await
                .unwrap();
        }
        let failed = |o: FetchOutcome, want: &str| match o {
            FetchOutcome::Failed { cid, .. } => assert_eq!(cid, want),
            _ => panic!("expected {want} to fail"),
        };

        // 1. The serving peer reports an error.
        let mut refused = resp(&set[0].0, 0, 0, Vec::new());
        refused.error = Some("no such shard".into());
        failed(m.process(&req, &st, &refused).await, &set[0].0);
        assert!(m.is_active(&set[1].0), "the next in line started");

        // 2. Rejected mid-stream: a changed total after one staged piece.
        let first = answer(&req, &set, req.count() - 1);
        assert!(matches!(
            m.process(&req, &st, &first).await,
            FetchOutcome::InProgress
        ));
        assert_eq!(staging_dirs(&st).len(), 1);
        let changed = resp(&set[1].0, 2048, 512, vec![0u8; 512]);
        failed(m.process(&req, &st, &changed).await, &set[1].0);
        assert!(m.is_active(&set[2].0));
        assert!(staging_dirs(&st).is_empty());

        // 3. Integrity: the whole shard arrives but does not hash to its CID.
        let wrong = resp(&set[2].0, 1024, 0, vec![0xEE; 1024]);
        failed(m.process(&req, &st, &wrong).await, &set[2].0);
        assert!(!st.has(&set[2].0));
        assert!(m.is_active(&set[3].0));

        // 4. The next request cannot be sent.
        let broken = FakeRequester::failing("transport down");
        let first = answer(&req, &set, req.count() - 1);
        failed(m.process(&broken, &st, &first).await, &set[3].0);
        assert_eq!((m.active_count(), m.queued_count()), (0, 0));
        assert!(staging_dirs(&st).is_empty());
    }

    /// A queued fetch whose first request cannot be sent when its turn comes
    /// is reported through `take_failed_starts`, not lost; the one after it
    /// still gets the slot.
    #[tokio::test]
    async fn a_queued_fetch_that_cannot_start_is_reported_and_the_next_one_starts() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = capped(1);
        let set = shards(2);
        for (cid, _) in &set {
            m.begin(&req, PeerId::random(), cid.clone(), None)
                .await
                .unwrap();
        }
        let broken = FakeRequester::failing("transport down");
        let mut refused = resp(&set[0].0, 0, 0, Vec::new());
        refused.error = Some("gone".into());
        // The slot frees while the transport is down: the waiting fetch
        // cannot start.
        assert!(matches!(
            m.process(&broken, &st, &refused).await,
            FetchOutcome::Failed { .. }
        ));
        let reported = m.take_failed_starts();
        assert_eq!(reported.len(), 1);
        assert!(matches!(&reported[0], FetchOutcome::Failed { cid, error }
            if *cid == set[1].0 && error.contains("transport down")));
        assert_eq!((m.active_count(), m.queued_count()), (0, 0));
        assert!(m.take_failed_starts().is_empty(), "reported once");

        // A start that fails for want of a transport is still this call's
        // error when a slot is free, as before the bound.
        let err = m
            .begin(&broken, PeerId::random(), set[1].0.clone(), None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("transport down"), "{err}");
        assert!(m.take_failed_starts().is_empty());
        assert!(!m.is_active(&set[1].0) && !m.is_queued(&set[1].0));
    }

    /// A cancelled fetch frees its slot: in flight, its staging is removed
    /// and the oldest waiting fetch starts; queued, it simply leaves the line.
    #[tokio::test]
    async fn a_cancelled_fetch_frees_its_slot() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = capped(1);
        let set = shards(3);
        for (cid, _) in &set {
            m.begin(&req, PeerId::random(), cid.clone(), None)
                .await
                .unwrap();
        }
        let first = answer(&req, &set, 0);
        assert!(matches!(
            m.process(&req, &st, &first).await,
            FetchOutcome::InProgress
        ));
        assert_eq!(staging_dirs(&st).len(), 1);

        assert!(m.cancel_with(&req, &set[2].0).await, "a queued fetch");
        assert!(!m.is_queued(&set[2].0));
        assert!(
            m.is_active(&set[0].0),
            "cancelling a queued fetch frees nothing"
        );

        assert!(m.cancel_with(&req, &set[0].0).await, "an in-flight fetch");
        assert!(staging_dirs(&st).is_empty(), "its staging is removed");
        assert!(m.is_active(&set[1].0), "and the next in line starts");
        assert_eq!(
            first_requests(&req),
            vec![set[0].0.clone(), set[1].0.clone()]
        );

        assert!(!m.cancel_with(&req, "never-started").await);

        // Only set[1]'s first request is still unanswered.
        let (completed, _) = serve_all(&mut m, &req, &st, &set, 1, req.count() - 1).await;
        assert_eq!(completed, vec![set[1].0.clone()]);
        assert!(!st.has(&set[0].0) && !st.has(&set[2].0));
    }

    /// `ShardRequestFailed` names a peer, not a CID: every fetch in flight
    /// from that peer fails and frees its slot; the others are untouched.
    #[tokio::test]
    async fn a_transport_failure_fails_that_peers_fetches_and_frees_their_slots() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = capped(2);
        let set = shards(4);
        let (p, q) = (PeerId::random(), PeerId::random());
        for (i, (cid, _)) in set.iter().enumerate() {
            let peer = if i == 1 { q } else { p };
            m.begin(&req, peer, cid.clone(), None).await.unwrap();
        }
        // set[0] (p) and set[1] (q) are in flight; set[2], set[3] (p) wait.
        let first = answer(&req, &set, 0);
        assert!(matches!(
            m.process(&req, &st, &first).await,
            FetchOutcome::InProgress
        ));

        let failed = m.request_failed_with(&req, p, "timeout").await;
        assert_eq!(failed.len(), 1);
        assert!(matches!(&failed[0], FetchOutcome::Failed { cid, error }
            if *cid == set[0].0 && error.contains("timeout")));
        assert!(staging_dirs(&st).is_empty());
        assert!(m.is_active(&set[1].0), "another peer's fetch is untouched");
        assert!(
            m.is_active(&set[2].0),
            "the freed slot went to the next in line"
        );
        assert!(m.is_queued(&set[3].0));
    }

    /// Answers like `FakeRequester`, except that the first request for
    /// `stall` never completes, as a send to a full channel would not.
    struct StallingRequester<'a> {
        inner: &'a FakeRequester,
        stall: String,
    }

    impl ChunkRequester for StallingRequester<'_> {
        async fn request_chunk(
            &self,
            peer_id: PeerId,
            cid: String,
            offset: u64,
            len: u64,
        ) -> std::result::Result<(), String> {
            if cid == self.stall && offset == 0 {
                std::future::pending::<()>().await;
            }
            self.inner.request_chunk(peer_id, cid, offset, len).await
        }
    }

    /// A dropped fetch frees its slot. Here the future that was starting a
    /// queued fetch is dropped mid-send: the fetch holds no slot and keeps
    /// its place, first in line. Dropping the manager then removes the
    /// staging of everything it held.
    #[tokio::test]
    async fn a_dropped_fetch_holds_no_slot() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = capped(1);
        let set = shards(3);
        for (cid, _) in &set[..2] {
            m.begin(&req, PeerId::random(), cid.clone(), None)
                .await
                .unwrap();
        }
        let stalling = StallingRequester {
            inner: &req,
            stall: set[1].0.clone(),
        };
        let first = answer(&req, &set, 0);
        assert!(matches!(
            m.process(&req, &st, &first).await,
            FetchOutcome::InProgress
        ));
        let last = answer(&req, &set, 1);
        let dropped = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            m.process(&stalling, &st, &last),
        )
        .await;
        assert!(
            dropped.is_err(),
            "the start of the queued fetch never completes"
        );
        assert!(st.has(&set[0].0), "the finished fetch was published");
        assert_eq!(m.active_count(), 0, "the dropped start holds no slot");
        assert!(m.is_queued(&set[1].0), "and keeps its place in line");

        // The next call fills the free slot, oldest first.
        m.begin(&req, PeerId::random(), set[2].0.clone(), None)
            .await
            .unwrap();
        assert!(m.is_active(&set[1].0) && m.is_queued(&set[2].0));
        let piece = answer(&req, &set, req.count() - 1);
        assert!(matches!(
            m.process(&req, &st, &piece).await,
            FetchOutcome::InProgress
        ));
        assert_eq!(staging_dirs(&st).len(), 1);
        drop(m);
        assert!(
            staging_dirs(&st).is_empty(),
            "the dropped manager left nothing"
        );
    }

    /// One fetch per CID, whether it is in flight or waiting.
    #[tokio::test]
    async fn a_cid_already_in_flight_or_queued_is_refused() {
        let req = FakeRequester::default();
        let mut m = capped(1);
        let set = shards(2);
        for (cid, _) in &set {
            m.begin(&req, PeerId::random(), cid.clone(), None)
                .await
                .unwrap();
        }
        for (cid, _) in &set {
            let err = m
                .begin(&req, PeerId::random(), cid.clone(), None)
                .await
                .unwrap_err();
            assert!(err.to_string().contains("already in progress"), "{err}");
        }
        assert_eq!((m.active_count(), m.queued_count()), (1, 1));
    }

    #[test]
    fn every_constructor_takes_the_default_bound_of_four_unless_configured() {
        use omni_types::config::{StoreConfig, DEFAULT_MAX_CONCURRENT_FETCHES};
        assert_eq!(DEFAULT_MAX_CONCURRENT_FETCHES, 4);
        assert_eq!(FetchManager::new(64 * 1024).max_concurrent_fetches(), 4);
        assert_eq!(
            FetchManager::with_limits(64 * 1024, MAX)
                .unwrap()
                .max_concurrent_fetches(),
            4
        );
        assert_eq!(
            FetchManager::from_config(&StoreConfig::default())
                .unwrap()
                .max_concurrent_fetches(),
            4
        );
        assert_eq!(capped(7).max_concurrent_fetches(), 7);
        let zero = StoreConfig {
            max_concurrent_fetches: 0,
            ..StoreConfig::default()
        };
        assert!(FetchManager::from_config(&zero).is_err());
    }

    #[test]
    fn omni_store_applies_the_configured_bound_to_both_download_paths() {
        use omni_types::config::StoreConfig;
        let tmp = tempfile::tempdir().unwrap();

        let dir = tmp.path().join("refused");
        let zero = StoreConfig {
            store_dir: dir.clone(),
            max_concurrent_fetches: 0,
            ..StoreConfig::default()
        };
        let err = crate::OmniStore::new(zero)
            .err()
            .expect("zero must be refused");
        assert!(format!("{err}").contains("max_concurrent_fetches"), "{err}");
        assert!(
            !dir.exists(),
            "refused before the store directory is created"
        );

        let three = StoreConfig {
            store_dir: tmp.path().join("store"),
            max_concurrent_fetches: 3,
            ..StoreConfig::default()
        };
        let s = crate::OmniStore::new(three).unwrap();
        assert_eq!(s.fetcher.max_concurrent_fetches(), 3);
        assert_eq!(s.local.max_concurrent_fetches(), 3);
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
    use crate::content_id::cid_from_data;
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

    /// Issue #132: no allocation grows with the shard. With every allocation
    /// of 1 MiB or more refused, a 4 MiB shard sent in 512 KiB pieces still
    /// completes, because each piece goes to disk as it arrives. (The former
    /// in-memory reassembly buffer could not grow past 1 MiB here and failed
    /// the fetch with "allocation failed".)
    #[tokio::test]
    async fn a_shard_larger_than_any_allowed_allocation_streams_to_disk() {
        let (_d, st) = store();
        let req = FakeRequester::default();
        let mut m = manager();
        let data: Vec<u8> = (0..4 * 1024 * 1024u32).map(|i| (i % 251) as u8).collect();
        let cid = cid_from_data(&data);
        seeded(&mut m, &req, &cid, Some(data.len() as u64)).await;

        // Built before arming, so their own allocations succeed.
        let piece_len = 512 * 1024;
        let pieces: Vec<ShardResponse> = data
            .chunks(piece_len)
            .enumerate()
            .map(|(i, c)| resp(&cid, data.len() as u64, (i * piece_len) as u64, c.to_vec()))
            .collect();

        let mut outcomes = Vec::new();
        {
            let _guard = ArmGuard::new(1024 * 1024);
            for piece in &pieces {
                outcomes.push(m.process(&req, &st, piece).await);
            }
        }

        for (i, outcome) in outcomes.iter().enumerate() {
            match outcome {
                FetchOutcome::Failed { error, .. } => panic!("piece {i} failed: {error}"),
                FetchOutcome::InProgress if i + 1 < pieces.len() => {}
                FetchOutcome::Complete { size, .. } if i + 1 == pieces.len() => {
                    assert_eq!(*size, data.len() as u64)
                }
                _ => panic!("piece {i}: unexpected outcome"),
            }
        }
        assert_eq!(st.get(&cid).unwrap(), data);
        assert!(staging_dirs(&st).is_empty());
    }
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOC: allocation_failure_tests::Failing = allocation_failure_tests::Failing;
