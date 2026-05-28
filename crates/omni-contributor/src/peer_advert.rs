//! Stage 12.5 — signed, per-session contributor peer advertisement.
//!
//! Lets a contributor publish a signed mapping
//! `(session_id, contributor_pubkey_hex) → libp2p PeerId + addrs +
//! capabilities`. Receivers verify the advertisement and use it as a
//! **local routing hint** for Stage 12.4 `ActivationHandoff` sends.
//!
//! Stage 12.5 is **not** a marketplace, registry, permanent
//! identity record, or chain authority. The advertisement is:
//!
//!   - **per-session** — bound to one `ExecutionSession.session_id`,
//!   - **short-lived** — `expires_at_utc ≤ advertised_at_utc + 24h`
//!     by schema validation (12.4-style bound),
//!   - **operator-trust-local** — verification + acceptance is a
//!     receiver-side policy decision, no chain authority involved.
//!
//! ## OmniNet identity is per-process
//!
//! `omni-net` builds its swarm with
//! `SwarmBuilder::with_new_identity()`, so the libp2p PeerId is
//! regenerated on every `OmniNet::new`. Restart of `omni-node` →
//! new PeerId → any advertisement from before the restart is dead.
//! That's a feature, not a bug: Stage 12.5 advertisements are
//! short-lived routing hints, not permanent identity records.
//! Persistent libp2p identity is a separate, deferred concern.
//!
//! Posture preserved: no chain wire / Stage 7b tx / SUM Chain RPC /
//! payment / marketplace / proof mode change. SNIP only for the
//! body's durable storage; mesh announcements are pointer-only.

use std::str::FromStr;

use libp2p_identity::PeerId;
use multiaddr::{Multiaddr, Protocol};
use serde::{Deserialize, Serialize};

use crate::error::SchemaError;
use crate::handoff::{TensorDtype, HANDOFF_CHUNK_MAX_BYTES};
use crate::job::{check_blake3_hex, check_iso_8601, check_pubkey_hex, check_signature_hex};

/// Pinned at 1. Closed-enum extensions or field reorders are
/// `schema_version: 2` migrations.
pub const PEER_ADVERTISEMENT_SCHEMA_VERSION: u32 = 1;

/// Maximum freshness window between `advertised_at_utc` and
/// `expires_at_utc`. 24 hours, in seconds. Enforced in
/// `validate_schema` so stale routes cannot live forever.
pub const PEER_ADVERTISEMENT_MAX_LIFETIME_SECS: i64 = 24 * 60 * 60;

/// Capability hints carried inside [`ContributorPeerAdvertisement`].
/// Receivers use these to negotiate transport parameters with the
/// advertised peer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeerCapabilities {
    /// Must be `true` for routes that [`crate::peer_routing::PeerRoutingCache::resolve`]
    /// is allowed to return. Forward-compat flag for future
    /// "discovery-only" advertisements.
    pub supports_live_handoff: bool,

    /// Operator-declared upper bound on individual chunk bytes the
    /// advertised peer accepts. Receivers MUST send at
    /// `min(local_chunk_cap, max_handoff_chunk_bytes)`. Bounded by
    /// [`HANDOFF_CHUNK_MAX_BYTES`].
    pub max_handoff_chunk_bytes: u64,

    /// Closed-enum dtypes the advertised peer can decode. Non-empty.
    pub supported_dtypes: Vec<TensorDtype>,
}

impl PeerCapabilities {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.max_handoff_chunk_bytes == 0 {
            return Err(SchemaError::PeerAdvertChunkCapZero);
        }
        if self.max_handoff_chunk_bytes > HANDOFF_CHUNK_MAX_BYTES {
            return Err(SchemaError::PeerAdvertChunkCapTooLarge {
                got: self.max_handoff_chunk_bytes,
                max: HANDOFF_CHUNK_MAX_BYTES,
            });
        }
        if self.supported_dtypes.is_empty() {
            return Err(SchemaError::PeerAdvertSupportedDtypesEmpty);
        }
        Ok(())
    }
}

/// Signed peer advertisement envelope. Bound to one
/// `ExecutionSession.session_id` and one contributor pubkey;
/// short-lived (≤ 24h) by schema validation.
///
/// `advertisement_id` is the lowercase-hex BLAKE3 of the canonical
/// signing body, mirroring the 12.3 `session_id` / `assignment_id`
/// derivation. The body excludes both `advertisement_id` itself and
/// `contributor_signature_hex`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContributorPeerAdvertisement {
    pub schema_version: u32,

    /// 64-char lowercase hex — derived from `advertisement_id_hex(self)`.
    pub advertisement_id: String,

    /// 64-char lowercase hex — copy of `ExecutionSession.session_id`.
    pub session_id: String,

    /// 64-char lowercase hex Ed25519 public key.
    pub contributor_pubkey_hex: String,

    /// Base58 libp2p `PeerId` of the advertising node, derived from
    /// `OmniNet::local_peer_id()` at advertisement-build time.
    pub libp2p_peer_id: String,

    /// Optional reachable multiaddrs (may be empty). When a multiaddr
    /// contains a trailing `/p2p/<peer>` protocol, it MUST match
    /// `libp2p_peer_id`.
    pub listen_multiaddrs: Vec<String>,

    pub capabilities: PeerCapabilities,

    /// RFC 3339 UTC (`Z` suffix).
    pub advertised_at_utc: String,

    /// RFC 3339 UTC (`Z` suffix). `expires_at_utc > advertised_at_utc`
    /// AND `expires_at_utc ≤ advertised_at_utc + 24h` (enforced in
    /// [`Self::validate_schema`]).
    pub expires_at_utc: String,

    /// 128-char lowercase hex Ed25519 signature over the canonical
    /// signing body (which excludes `advertisement_id` and this field).
    pub contributor_signature_hex: String,
}

impl ContributorPeerAdvertisement {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != PEER_ADVERTISEMENT_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_blake3_hex("advertisement_id", &self.advertisement_id)?;
        check_blake3_hex("session_id", &self.session_id)?;
        check_pubkey_hex("contributor_pubkey_hex", &self.contributor_pubkey_hex)?;

        // PeerId base58 parse.
        if PeerId::from_str(&self.libp2p_peer_id).is_err() {
            return Err(SchemaError::PeerAdvertLibp2pPeerIdMalformed {
                got: self.libp2p_peer_id.clone(),
            });
        }

        // Multiaddrs: each must parse; any /p2p/<peer> tail must
        // match libp2p_peer_id. Empty list is allowed.
        let expected_peer = PeerId::from_str(&self.libp2p_peer_id)
            .expect("validated above by PeerId::from_str check");
        for (i, addr_s) in self.listen_multiaddrs.iter().enumerate() {
            let maddr = Multiaddr::from_str(addr_s).map_err(|_| {
                SchemaError::PeerAdvertMultiaddrMalformed {
                    index: i,
                    got: addr_s.clone(),
                }
            })?;
            // Walk protocols; if any /p2p, it must match.
            for proto in maddr.iter() {
                if let Protocol::P2p(p) = proto {
                    if p != expected_peer {
                        return Err(SchemaError::PeerAdvertMultiaddrP2pMismatch {
                            index: i,
                            multiaddr_peer: p.to_base58(),
                            advertised_peer: self.libp2p_peer_id.clone(),
                        });
                    }
                }
            }
        }

        self.capabilities.validate_schema()?;

        check_iso_8601("advertised_at_utc", &self.advertised_at_utc)?;
        check_iso_8601("expires_at_utc", &self.expires_at_utc)?;

        // Parse both timestamps and enforce ordering + 24h bound.
        let advertised = chrono::DateTime::parse_from_rfc3339(&self.advertised_at_utc)
            .map_err(|_| SchemaError::MalformedTimestamp {
                field: "advertised_at_utc",
                got: self.advertised_at_utc.clone(),
            })?;
        let expires = chrono::DateTime::parse_from_rfc3339(&self.expires_at_utc).map_err(
            |_| SchemaError::MalformedTimestamp {
                field: "expires_at_utc",
                got: self.expires_at_utc.clone(),
            },
        )?;
        if expires <= advertised {
            return Err(SchemaError::PeerAdvertExpiryNotAfterAdvertised {
                advertised_at: self.advertised_at_utc.clone(),
                expires_at: self.expires_at_utc.clone(),
            });
        }
        let lifetime = expires.signed_duration_since(advertised);
        if lifetime.num_seconds() > PEER_ADVERTISEMENT_MAX_LIFETIME_SECS {
            return Err(SchemaError::PeerAdvertExpiryTooFar {
                advertised_at: self.advertised_at_utc.clone(),
                expires_at: self.expires_at_utc.clone(),
                max_lifetime_secs: PEER_ADVERTISEMENT_MAX_LIFETIME_SECS,
            });
        }

        check_signature_hex("contributor_signature_hex", &self.contributor_signature_hex)?;
        Ok(())
    }

    /// Returns the parsed `libp2p_peer_id`. Panics if `validate_schema`
    /// has not been run; callers must validate first. Convenience for
    /// the routing cache.
    pub fn parsed_peer_id(&self) -> PeerId {
        PeerId::from_str(&self.libp2p_peer_id)
            .expect("validate_schema parses libp2p_peer_id")
    }
}
