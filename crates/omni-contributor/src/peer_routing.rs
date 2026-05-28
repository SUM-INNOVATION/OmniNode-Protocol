//! Stage 12.5 — local routing cache + announcement processor for
//! [`ContributorPeerAdvertisement`](crate::peer_advert::ContributorPeerAdvertisement).
//!
//! Two pieces:
//!
//!   - [`process_peer_advertisement_announcement`] — receive-side
//!     pipeline that mirrors the Stage 12.3 `process_*_announcement`
//!     helpers: announcement schema → announcer signature → SNIP
//!     fetch → body schema → contributor signature → drift checks
//!     → matching `ContributorJoin` check → expiry check. Returns a
//!     typed outcome the CLI / cache consume.
//!
//!   - [`PeerRoutingCache`] — in-process cache keyed by
//!     `(session_id, contributor_pubkey_hex)`. Holds the **newest
//!     non-expired** verified advertisement per key. `resolve` is
//!     local policy: chunk cap is `min(local_chunk_cap,
//!     advertised.max_handoff_chunk_bytes)`; dtype mismatch is a
//!     hard refusal; expired advertisements never resolve.
//!
//! Stage 12.5 is **local routing data**, not a marketplace, registry,
//! or chain authority. Two coordinators running parallel sessions
//! for the same posted_id will see their own peer advertisements
//! cached independently.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use omni_store::SnipV2Adapter;
use omni_types::phase5::SnipV2ObjectId;

use crate::canonical::{
    advertisement_id_hex, net_peer_advert_signing_input, peer_advertisement_signing_input,
};
use crate::handoff::TensorDtype;
use crate::net::NetworkPeerAdvertisementAnnouncement;
use crate::peer_advert::ContributorPeerAdvertisement;
use crate::session::ContributorJoin;
use crate::signing::verify_signature_hex;

/// Parse an RFC 3339 timestamp (with `Z` or `+00:00`, optional
/// fractional seconds) into a `chrono::DateTime<Utc>`. Used so
/// expiry / freshness comparisons never fall back to fragile
/// string-compare semantics — a fractional-second variant with
/// the same prefix would otherwise sort incorrectly against the
/// canonical `SecondsFormat::Secs` form.
fn parse_rfc3339_utc(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s).ok().map(|d| d.with_timezone(&Utc))
}

/// Per-announcement processing outcome. Parallel to Stage 12.3's
/// `AnnouncementOutcome<T>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerAdvertisementOutcome {
    /// All checks passed; `body` is the parsed inner advertisement.
    Verified { body: Box<ContributorPeerAdvertisement> },
    AnnouncementSchemaMalformed(String),
    AnnouncerSignatureFailed,
    SnipFetchFailed(String),
    BodyParseFailed(String),
    BodySchemaInvalid(String),
    /// `advertisement_id` stored on the body does NOT equal the
    /// BLAKE3 of the canonical signing body. Indicates a relayer
    /// or local-file tampered the id field after signing (the
    /// signature excludes `advertisement_id`).
    AdvertisementIdMismatch { stored: String, derived: String },
    /// Inner contributor signature did not verify against
    /// `contributor_pubkey_hex`.
    ContributorSignatureFailed,
    /// A drift-guard field on the announcement disagreed with the
    /// fetched body.
    DriftMismatch { field: &'static str },
    /// The advertisement's `(session_id, contributor_pubkey_hex)`
    /// did not match any of the supplied verified joins.
    NoMatchingJoin,
    /// `expires_at_utc <= now_utc`. Forensic re-runs can pass a
    /// `now_utc` of `None` to skip this check.
    Expired { now: String, expires_at: String },
}

impl PeerAdvertisementOutcome {
    pub fn is_verified(&self) -> bool {
        matches!(self, PeerAdvertisementOutcome::Verified { .. })
    }
}

/// Process a `NetworkPeerAdvertisementAnnouncement`. Mirrors the
/// shape of Stage 12.3's `process_*_announcement` helpers but
/// returns a peer-advert-specific outcome.
///
/// `now_utc` is `Some` for routine `watch-peer-adverts` use (rejects
/// expired adverts) and `None` for forensic re-runs.
pub fn process_peer_advertisement_announcement<A: SnipV2Adapter + ?Sized>(
    ann: &NetworkPeerAdvertisementAnnouncement,
    adapter: &A,
    verified_joins: &[ContributorJoin],
    now_utc: Option<&str>,
) -> PeerAdvertisementOutcome {
    if let Err(e) = ann.validate_schema() {
        return PeerAdvertisementOutcome::AnnouncementSchemaMalformed(e.to_string());
    }
    let ann_signing_input = match net_peer_advert_signing_input(ann) {
        Ok(b) => b,
        Err(e) => {
            return PeerAdvertisementOutcome::AnnouncementSchemaMalformed(e.to_string());
        }
    };
    let ann_ok = verify_signature_hex(
        &ann.announcer_pubkey_hex,
        &ann_signing_input,
        &ann.announcer_signature_hex,
    )
    .unwrap_or(false);
    if !ann_ok {
        return PeerAdvertisementOutcome::AnnouncerSignatureFailed;
    }

    // Fetch + parse + schema-validate the inner advertisement.
    let snip_root = match SnipV2ObjectId::from_hex(&ann.peer_advertisement_snip_root) {
        Ok(r) => r,
        Err(e) => {
            return PeerAdvertisementOutcome::SnipFetchFailed(format!(
                "bad snip root: {e:?}"
            ));
        }
    };
    let bytes = match crate::snip::fetch_bytes(adapter, &snip_root) {
        Ok(b) => b,
        Err(e) => {
            return PeerAdvertisementOutcome::SnipFetchFailed(e.to_string());
        }
    };
    let advert: ContributorPeerAdvertisement = match serde_json::from_slice(&bytes) {
        Ok(a) => a,
        Err(e) => return PeerAdvertisementOutcome::BodyParseFailed(e.to_string()),
    };

    // Body-level verification (schema + advertisement_id recompute
    // + contributor signature + matching join + expiry).
    let body_outcome = verify_peer_advertisement_body(&advert, verified_joins, now_utc);
    if !matches!(body_outcome, PeerAdvertisementOutcome::Verified { .. }) {
        return body_outcome;
    }

    // Drift checks between announcement and body. Each drift check
    // is meaningful regardless of body verification because the
    // announcer can lie about which body they're pointing at.
    if ann.advertisement_id != advert.advertisement_id {
        return PeerAdvertisementOutcome::DriftMismatch { field: "advertisement_id" };
    }
    if ann.session_id != advert.session_id {
        return PeerAdvertisementOutcome::DriftMismatch { field: "session_id" };
    }
    if ann.contributor_pubkey_hex != advert.contributor_pubkey_hex {
        return PeerAdvertisementOutcome::DriftMismatch {
            field: "contributor_pubkey_hex",
        };
    }

    PeerAdvertisementOutcome::Verified { body: Box::new(advert) }
}

/// Body-only verification for a `ContributorPeerAdvertisement`.
/// Mirrors the body-side of
/// [`process_peer_advertisement_announcement`] so callers that load
/// an advertisement from disk (or any non-mesh source) can run the
/// same checks before trusting it.
///
/// Checks (in order):
///   1. Schema valid.
///   2. `advertisement_id` recomputes from canonical body bytes
///      (the signature EXCLUDES this field, so a tampered id would
///      pass signature verify but must not be trusted by routing).
///   3. Contributor signature verifies against
///      `contributor_pubkey_hex` over the canonical signing input.
///   4. `(session_id, contributor_pubkey_hex)` matches a supplied
///      verified `ContributorJoin`.
///   5. `now_utc < expires_at_utc` (parsed as RFC 3339, NOT string
///      compared). `now_utc = None` skips this for forensic re-runs.
///
/// Returns a `PeerAdvertisementOutcome` whose `Verified` variant
/// boxes the body (same shape as the network processor for
/// uniform consumer code).
pub fn verify_peer_advertisement_body(
    advert: &ContributorPeerAdvertisement,
    verified_joins: &[ContributorJoin],
    now_utc: Option<&str>,
) -> PeerAdvertisementOutcome {
    if let Err(e) = advert.validate_schema() {
        return PeerAdvertisementOutcome::BodySchemaInvalid(e.to_string());
    }
    // advertisement_id recompute. Signature excludes it, so this
    // must be checked separately or a relayer can swap ids freely.
    let derived = match advertisement_id_hex(advert) {
        Ok(s) => s,
        Err(e) => {
            return PeerAdvertisementOutcome::BodySchemaInvalid(e.to_string());
        }
    };
    if derived != advert.advertisement_id {
        return PeerAdvertisementOutcome::AdvertisementIdMismatch {
            stored: advert.advertisement_id.clone(),
            derived,
        };
    }
    // Contributor signature over canonical body.
    let body_signing_input = match peer_advertisement_signing_input(advert) {
        Ok(b) => b,
        Err(e) => {
            return PeerAdvertisementOutcome::BodySchemaInvalid(e.to_string());
        }
    };
    let body_ok = verify_signature_hex(
        &advert.contributor_pubkey_hex,
        &body_signing_input,
        &advert.contributor_signature_hex,
    )
    .unwrap_or(false);
    if !body_ok {
        return PeerAdvertisementOutcome::ContributorSignatureFailed;
    }
    let has_join = verified_joins.iter().any(|j| {
        j.session_id == advert.session_id
            && j.contributor_pubkey_hex == advert.contributor_pubkey_hex
    });
    if !has_join {
        return PeerAdvertisementOutcome::NoMatchingJoin;
    }
    if let Some(now_s) = now_utc {
        // Parse both timestamps. Bad parses (which `validate_schema`
        // would already have caught for `expires_at_utc`) fall back
        // to a permissive "not expired" answer rather than crashing
        // — the caller already knows the schema was valid.
        if let (Some(now), Some(expires)) =
            (parse_rfc3339_utc(now_s), parse_rfc3339_utc(&advert.expires_at_utc))
        {
            if now >= expires {
                return PeerAdvertisementOutcome::Expired {
                    now: now_s.to_string(),
                    expires_at: advert.expires_at_utc.clone(),
                };
            }
        }
    }
    PeerAdvertisementOutcome::Verified {
        body: Box::new(advert.clone()),
    }
}

// ── Routing cache ─────────────────────────────────────────────────────────

/// Resolved route information returned by
/// [`PeerRoutingCache::resolve`]. The caller (Stage 12.4
/// `live_send_activation`) uses `peer_id` to address the transport
/// and the negotiated `chunk_cap` to bound outgoing chunks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPeerRoute {
    /// Base58 libp2p PeerId of the advertised peer.
    pub peer_id: String,
    /// Optional reachable multiaddrs from the advertisement.
    pub multiaddrs: Vec<String>,
    /// Negotiated maximum chunk size — `min(local_chunk_cap,
    /// advertised.max_handoff_chunk_bytes)`.
    pub max_handoff_chunk_bytes: u64,
    /// Dtype the routing call requested; included for symmetry with
    /// the advertisement.
    pub negotiated_dtype: TensorDtype,
}

/// Outcome of a [`PeerRoutingCache::resolve`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteResolution {
    /// A non-expired advertisement matched; the route is ready.
    Found(ResolvedPeerRoute),
    /// No advertisement found for this `(session_id,
    /// contributor_pubkey_hex)`.
    NoAdvertisement,
    /// An advertisement exists but its `expires_at_utc <= now_utc`.
    /// Treated identically to `NoAdvertisement` by callers that
    /// just need a route; surfaced separately so operators can
    /// distinguish "never seen" from "seen but stale".
    AllExpired { newest_expires_at: String },
    /// An advertisement exists but it doesn't support live handoff.
    LiveHandoffNotSupported,
    /// The requested dtype is not in the advertisement's supported
    /// list.
    DtypeNotSupported {
        requested: TensorDtype,
        supported: Vec<TensorDtype>,
    },
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CacheKey {
    session_id: String,
    contributor_pubkey_hex: String,
}

#[derive(Debug, Default)]
pub struct PeerRoutingCache {
    entries: HashMap<CacheKey, ContributorPeerAdvertisement>,
}

impl PeerRoutingCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entries currently cached. Used by tests.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert a verified advertisement into the cache. Caller is
    /// expected to have run
    /// [`process_peer_advertisement_announcement`] (or equivalent
    /// schema + signature + binding checks) first.
    ///
    /// Replacement policy: newer `advertised_at_utc` wins. If the
    /// candidate is older than the cached entry for the same
    /// `(session_id, contributor_pubkey_hex)`, the candidate is
    /// dropped on the floor — RFC 3339 Z timestamps lex-compare
    /// correctly given consistent precision.
    pub fn insert_verified(&mut self, advert: ContributorPeerAdvertisement) {
        let key = CacheKey {
            session_id: advert.session_id.clone(),
            contributor_pubkey_hex: advert.contributor_pubkey_hex.clone(),
        };
        // Newest non-expired wins; ordering uses parsed chrono
        // timestamps so a fractional-second variant cannot subvert
        // string-compare ordering. Bad parses on either side fall
        // back to "candidate wins" — `validate_schema` would have
        // caught a bad timestamp upstream.
        let cand_at = parse_rfc3339_utc(&advert.advertised_at_utc);
        let keep_existing = match (self.entries.get(&key), cand_at) {
            (Some(existing), Some(c)) => {
                match parse_rfc3339_utc(&existing.advertised_at_utc) {
                    Some(e) => e >= c,
                    None => false,
                }
            }
            _ => false,
        };
        if !keep_existing {
            self.entries.insert(key, advert);
        }
    }

    /// Resolve a route for a given (session, contributor, dtype),
    /// applying the local chunk cap as the upper bound of the
    /// negotiated `max_handoff_chunk_bytes`.
    ///
    /// `now_utc` is required because expired advertisements are
    /// never resolved by Stage 12.5 (operators that need to inspect
    /// stale entries should do so via the underlying cache iter
    /// helpers in a future PR; v1 keeps the resolver strict).
    pub fn resolve(
        &self,
        session_id: &str,
        contributor_pubkey_hex: &str,
        dtype: TensorDtype,
        local_chunk_cap: u64,
        now_utc: &str,
    ) -> RouteResolution {
        let key = CacheKey {
            session_id: session_id.to_string(),
            contributor_pubkey_hex: contributor_pubkey_hex.to_string(),
        };
        let advert = match self.entries.get(&key) {
            Some(a) => a,
            None => return RouteResolution::NoAdvertisement,
        };
        // Expiry check uses parsed chrono timestamps; never raw
        // string compare. A non-parseable `expires_at_utc` (which
        // `validate_schema` would have rejected upstream) is
        // treated conservatively as expired.
        let expired = match (
            parse_rfc3339_utc(now_utc),
            parse_rfc3339_utc(&advert.expires_at_utc),
        ) {
            (Some(now), Some(expires)) => now >= expires,
            _ => true,
        };
        if expired {
            return RouteResolution::AllExpired {
                newest_expires_at: advert.expires_at_utc.clone(),
            };
        }
        if !advert.capabilities.supports_live_handoff {
            return RouteResolution::LiveHandoffNotSupported;
        }
        if !advert.capabilities.supported_dtypes.contains(&dtype) {
            return RouteResolution::DtypeNotSupported {
                requested: dtype,
                supported: advert.capabilities.supported_dtypes.clone(),
            };
        }
        let negotiated = local_chunk_cap.min(advert.capabilities.max_handoff_chunk_bytes);
        RouteResolution::Found(ResolvedPeerRoute {
            peer_id: advert.libp2p_peer_id.clone(),
            multiaddrs: advert.listen_multiaddrs.clone(),
            max_handoff_chunk_bytes: negotiated,
            negotiated_dtype: dtype,
        })
    }
}
