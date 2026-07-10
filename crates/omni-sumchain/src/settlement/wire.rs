//! Issue #83 — raw JSON-RPC response DTOs for the InferenceSettlement
//! subprotocol. Each type mirrors one chain response verbatim; no
//! cross-RPC composition happens here. Composition lives in
//! [`crate::settlement::view`].
//!
//! ## Wire-shape notes
//!
//! - Field names are `snake_case`, matching the existing
//!   `chain_getChainParams` / `sum_getInferenceAttestation` DTOs in
//!   `crate::dto` (chain emits snake_case JSON keys).
//! - Hash / address fields are `0x`-prefixed lowercase hex strings
//!   (32-byte values → 66 chars incl. prefix; 20-byte addresses may
//!   use a chain-native encoding — Stage 6 uses base58 chain
//!   addresses in `verifier_address`).
//! - `Balance` fields (`escrow_total`, `reward_amount`, `bond_amount`,
//!   slash amounts) are emitted as decimal strings on the wire
//!   because a `u128` exceeds JSON's safe integer range. The view
//!   layer parses them into `u128`.
//! - Optional fields carry `#[serde(default)]` so partial responses
//!   (older chain patches, empty-state variants) parse cleanly.
//!
//! ## Verification note
//!
//! These field names reflect the chain-team response summary for
//! `sum-chain#76 + #86`. Before this crate is used against a live
//! chain build, the DTOs here MUST be diffed against the actual
//! sum-chain RPC handlers. If the chain names or shapes differ,
//! adjust `#[serde(rename)]` decorators here — the view layer is
//! insulated from wire renames.

use serde::{Deserialize, Serialize};

// ── Sessions ──────────────────────────────────────────────────────────────────

/// Response shape for `omninode_getInferenceSession(session_id)`.
/// Returns `null` when the chain has no session record under `session_id`.
///
/// **The session record itself does NOT enumerate verifiers.** Multi-
/// verifier composition happens in the view layer by unioning
/// verifier addresses across attestation / claim / dispute reads.
///
/// `Default` is derived so test fixtures + hermetic tests can
/// spread-init only the fields under test; production callers always
/// receive a fully-populated instance from serde deserialisation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceSessionRaw {
    pub session_id: String,

    /// True iff the session was created in consistency / plurality
    /// mode. Composing a claimability view over this session requires
    /// the consistency gate to be active (else the view layer returns
    /// `SettlementViewIncomplete`).
    pub consistency_required: bool,

    /// True iff the session was created with a verifier bond
    /// requirement. Requires the bonding gate to be active for the
    /// view layer to resolve verifier registry state.
    pub bond_required: bool,

    /// Chain-configured cap on the number of verifiers allowed to
    /// claim rewards for this session.
    pub max_verifiers: u32,

    /// Total escrow locked at session creation. Decimal `u128` string.
    pub escrow_total: String,

    /// Escrow remaining after any claim payouts. Decimal `u128` string.
    pub escrow_remaining: String,

    /// Number of claims already paid or in-flight against this
    /// session's escrow. Used with `max_verifiers` to determine cap
    /// availability during claimable-reward composition.
    pub claims_count: u32,

    /// Session status. Chain-team confirmed (sum-chain#110) that
    /// `omninode_getInferenceSession` returns this field as
    /// `"status": "Open"` / `"Refunded"` / `"Settled"` on the wire.
    /// `#[serde(alias = "status")]` lets the DTO accept both the
    /// canonical chain field name AND the legacy `"lifecycle"` name
    /// still used by older mocks; either is deserialised into this
    /// single Rust field. Anything the value-mapper doesn't
    /// recognise ends up as `SessionLifecycle::Unknown(...)` in the
    /// view layer.
    #[serde(alias = "status")]
    pub lifecycle: String,

    /// Height at which the session record was created.
    pub created_at_height: u64,

    #[serde(default)]
    pub settled_at_height: Option<u64>,

    #[serde(default)]
    pub refunded_at_height: Option<u64>,

    // ── Issue #81 additions ──────────────────────────────────────────
    //
    // Chain confirmed via sum-chain#110 that the session response
    // carries the funder base58 address and the dispute-window block
    // count, both consumed by the OpenDispute local prechecks (funder
    // authority + maturity). Both fields are `#[serde(default)]` so
    // any older/mocked chain response missing them still parses; the
    // consuming code refuses when the value is missing / zero.
    /// Session funder base58 address. Sole legal `OpenDispute` signer.
    #[serde(default)]
    pub funder: String,

    /// Block window between claim maturity ready-block and the last
    /// height at which `OpenDispute` is still accepted. Combined with
    /// `chain_getChainParams.finality_depth` and
    /// `sum_getInferenceAttestation.included_at_height` to compute
    /// the local dispute-window closure height.
    #[serde(default)]
    pub dispute_window_blocks: u64,
}

// ── Claims ────────────────────────────────────────────────────────────────────

/// Response shape for `omninode_getInferenceClaims(session_id)`. The
/// chain returns a single container carrying every claim already
/// submitted against the session (may be empty when active with no
/// claims yet).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceClaimsRaw {
    pub session_id: String,

    #[serde(default)]
    pub claims: Vec<InferenceClaimRaw>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceClaimRaw {
    pub verifier_address: String,
    pub claimed_at_height: u64,
    /// Decimal `u128` string.
    pub reward_amount: String,
    /// `"pending"` | `"paid"` | `"denied"`. Anything else maps to
    /// `ClaimState::UnknownWire(...)` in the view layer.
    pub state: String,
    #[serde(default)]
    pub paid_at_height: Option<u64>,
    #[serde(default)]
    pub denied_at_height: Option<u64>,
    #[serde(default)]
    pub denied_reason: Option<String>,
}

// ── Disputes ──────────────────────────────────────────────────────────────────

/// Response shape for `omninode_getInferenceDisputes(session_id)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceDisputesRaw {
    pub session_id: String,

    #[serde(default)]
    pub disputes: Vec<InferenceDisputeRaw>,
}

/// A single dispute record — one per (session, target-verifier) tuple.
/// The chain-side dispute lifecycle is:
///
/// 1. **The session funder opens the dispute.** In v1 the `OpenDispute`
///    signer is the funder only — validators, verifiers, and third-
///    party challengers cannot open disputes at the RPC level.
/// 2. Validator quorum votes on the dispute; non-signing validators
///    are abstentions (see sum-chain#86, terminology preserved in
///    `docs/inference-settlement-v1-evidence.md`).
/// 3. Chain resolves to approved (claim proceeds) or denied (claim
///    refused; slashing may follow ONLY for bond-required sessions
///    where the chain owns the slashing logic — reward denial alone
///    is NOT slashing).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceDisputeRaw {
    /// Chain address of the verifier this dispute targets.
    pub verifier_address: String,
    pub opened_at_height: u64,
    /// `"open"` | `"resolved_approved"` | `"resolved_denied"`.
    pub state: String,
    #[serde(default)]
    pub resolved_at_height: Option<u64>,
    /// Approve votes, in basis points of the active validator set.
    /// Only meaningful once resolved.
    #[serde(default)]
    pub approve_bps: Option<u32>,
    /// Deny votes, in basis points of the active validator set.
    /// Only meaningful once resolved.
    #[serde(default)]
    pub deny_bps: Option<u32>,
}

// ── Claimable reward pre-check ────────────────────────────────────────────────

/// Response shape for
/// `omninode_getClaimableReward(session_id, verifier_address)`. Chain-
/// side pre-computed answer to "can this verifier claim right now?"
/// broken out into orthogonal preconditions so the caller can render
/// each precisely.
///
/// **The `claimable_now` flag is chain-computed against RPC-level
/// preconditions only.** If the target session is consistency-mode
/// or bond-required, a fully-normalized view further requires the
/// relevant gate to be locally active — see [`crate::settlement::view`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimableRewardRaw {
    pub session_id: String,
    pub verifier_address: String,

    /// Maturity: `finalized_at_height + dispute_window_blocks <= head`.
    /// Chain applies the formula from
    /// `docs/inference-settlement-v1-evidence.md` (never double-counts
    /// finality).
    pub mature: bool,

    /// The absolute chain height at which the reward becomes claimable.
    pub claim_ready_block: u64,

    /// Blocks remaining until `claim_ready_block`. Zero when mature.
    pub blocks_until_ready: u64,

    /// True iff `escrow_remaining >= reward_amount`.
    pub escrow_available: bool,

    /// True iff `claims_count < max_verifiers`.
    pub cap_available: bool,

    /// True iff no open dispute and no dispute-resolved-denied against
    /// this (session, verifier) tuple.
    pub dispute_clear: bool,

    /// AND of `mature`, `escrow_available`, `cap_available`,
    /// `dispute_clear`, and (chain-side) not-already-claimed.
    pub claimable_now: bool,

    /// Decimal `u128` string.
    pub reward_amount: String,
}

// ── Consistency / plurality groups ────────────────────────────────────────────

/// Response shape for `omninode_getInferenceConsistency(session_id)`.
/// Only meaningful when consistency mode is active on the session AND
/// the chain has begun grouping attestations by their full digest
/// tuple (`model_hash`, `manifest_root`, `response_hash`, `proof_root`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceConsistencyRaw {
    pub session_id: String,

    #[serde(default)]
    pub groups: Vec<ConsistencyGroupRaw>,

    /// The plurality-winning group's key, once chain has resolved
    /// plurality. `None` while still resolving or when consistency
    /// mode is off.
    #[serde(default)]
    pub plurality_key: Option<DigestTupleRaw>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsistencyGroupRaw {
    pub key: DigestTupleRaw,

    /// Verifier chain addresses whose attestations map to this digest
    /// tuple.
    #[serde(default)]
    pub members: Vec<String>,
}

/// Full four-part digest tuple used as the consistency-group key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DigestTupleRaw {
    /// `0x`-prefixed lowercase hex, 32 bytes.
    pub model_hash: String,
    /// `0x`-prefixed lowercase hex, 32 bytes.
    pub manifest_root: String,
    /// `0x`-prefixed lowercase hex, 32 bytes.
    pub response_hash: String,
    /// `0x`-prefixed lowercase hex, 32 bytes.
    pub proof_root: String,
}

// ── Verifier registry ────────────────────────────────────────────────────────

/// Response shape for `omninode_getVerifier(address)`. Returns `null`
/// when the address has no verifier registry record. Requires the
/// bonding gate to be RPC-level active.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierRegistryRaw {
    pub address: String,

    /// Currently bonded amount. Decimal `u128` string.
    pub bond_amount: String,

    /// `"bonded"` | `"unbonding"` | `"withdrawn"`. Anything else
    /// maps to `BondState::UnknownWire(...)` in the view layer.
    pub bond_state: String,

    #[serde(default)]
    pub unbonding_since_height: Option<u64>,

    #[serde(default)]
    pub withdrawable_at_height: Option<u64>,

    /// Slash history is chain-append-only. **Reward denial is NOT
    /// recorded here** — this vec only carries actual slashing events
    /// against bond-required sessions where validator-quorum-denied
    /// dispute triggered chain-side stake removal.
    #[serde(default)]
    pub slash_history: Vec<SlashRecordRaw>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlashRecordRaw {
    pub slashed_at_height: u64,
    /// Decimal `u128` string.
    pub amount: String,
    #[serde(default)]
    pub reason: Option<String>,
    /// Session that triggered the slash, if the chain records it.
    #[serde(default)]
    pub session_id: Option<String>,
}
