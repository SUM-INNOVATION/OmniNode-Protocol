//! Local error type for `omni-zkml` proof / response artifact flow.
//!
//! Kept local to this crate — not bridged into `omni_store::StoreError` or
//! `omni_types::OmniError`. `omni-zkml` is allowed to depend on `omni-store`
//! and surface its errors upward via the `SnipV2` variant, but the reverse
//! is never true.

use std::io;
use std::path::PathBuf;

use omni_store::SnipV2Error;

#[derive(Debug, thiserror::Error)]
pub enum ProofArtifactError {
    #[error("local response file not found: {}", path.display())]
    ResponseFileNotFound { path: PathBuf },

    #[error("local proof file not found: {}", path.display())]
    ProofFileNotFound { path: PathBuf },

    #[error("session_id is empty")]
    EmptySessionId,

    #[error("manifest has no top-level SNIP V2 ref; cannot build commitment")]
    ManifestLacksSnipRoot,

    #[error("response artifact has no BLAKE3 hash; cannot build commitment")]
    ResponseLacksHash,

    #[error("proof artifact has no SNIP V2 ref; cannot build commitment")]
    ProofLacksSnipRoot,

    #[error("SNIP V2 error: {0}")]
    SnipV2(#[from] SnipV2Error),

    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

pub type Result<T> = std::result::Result<T, ProofArtifactError>;

// ── Stage 4: signer + attestation envelope ────────────────────────────────────

/// Failure reported by a [`crate::attestation::Signer`] implementation.
///
/// `Clone` is derived so test fixtures can store a canned
/// `Result<String, SignerError>` and `.clone()` it on every call to
/// `sign(...)`. The single-variant `Failed(String)` preserves the
/// implementation's diagnostic message verbatim; a future stage may
/// upgrade to a structured `#[source]` when concrete signer
/// implementations introduce their own error types.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SignerError {
    #[error("signer failure: {0}")]
    Failed(String),
}

/// Failure produced by [`crate::attestation::build_attestation`] and the
/// related canonical-bytes / digest helpers. Intentionally **not** `Clone`
/// — Stage 4 callers consume the error directly; fixtures only need
/// `SignerError` to be cloneable.
#[derive(Debug, thiserror::Error)]
pub enum AttestationError {
    #[error("canonical-bytes serialization failed: {0}")]
    Serialization(String),

    #[error("commitment has empty session_id")]
    EmptySessionId,

    #[error("commitment has empty model_hash")]
    EmptyModelHash,

    #[error("commitment has empty response_hash")]
    EmptyResponseHash,

    #[error("signer returned empty verifier_address")]
    EmptyVerifierAddress,

    #[error("signer returned empty signature")]
    EmptySignature,

    #[error("signer failure: {0}")]
    Signer(#[from] SignerError),
}

/// Stage 4 result alias. Deliberately separate from [`Result`] (Stage 3, =
/// `std::result::Result<T, ProofArtifactError>`) so callers in
/// `attestation.rs` never route attestation errors through the Stage-3
/// error domain.
pub type AttestationResult<T> = std::result::Result<T, AttestationError>;

// ── Stage 5: chain client + offline attestation registry ──────────────────────

/// Failure reported by a [`crate::chain::ChainClient`] implementation.
///
/// `Clone` is derived so test fixtures (and a future real chain adapter's
/// own error machinery) can store a canned outcome and reuse it. The
/// single-variant `Other(String)` shape preserves the implementation's
/// diagnostic message verbatim; a future real chain adapter may upgrade
/// to structured variants.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ChainClientError {
    #[error("chain client failure: {0}")]
    Other(String),
}

/// Failure produced by the [`crate::registry::AttestationRegistry`] and
/// its workflow free functions.
///
/// Intentionally **not** `Clone` (`io::Error` is not `Clone`).
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("registry serialization failure: {0}")]
    Serialization(String),

    #[error("registry I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("attestation record not found: {0}")]
    RecordNotFound(crate::registry::AttestationId),

    /// Returned by `insert` when an existing record under the same
    /// `(session_id, verifier_address)` carries a byte-different
    /// `InferenceAttestation`. The existing record on disk is **not**
    /// overwritten; the caller can `load(&id)` to inspect the stored
    /// value.
    #[error(
        "conflicting attestation under existing id {id}: session_id and \
         verifier_address match but the stored attestation differs from \
         the one being inserted"
    )]
    ConflictingAttestation { id: crate::registry::AttestationId },

    #[error("invalid status transition for {id}: cannot go from {from:?} to {to}")]
    InvalidStatusTransition {
        id: crate::registry::AttestationId,
        from: crate::registry::LocalAttestationStatus,
        to: &'static str,
    },

    /// Stage 5.1 integrity defense: a record in a queryable local status
    /// (`Submitted` or `Included`) had `receipt: None`. `mark_submitted`
    /// always sets the receipt, so this state can only arise from
    /// hand-edited or corrupted JSON in the registry directory. Returned
    /// by [`crate::registry::query_attestation_workflow`] instead of
    /// silently no-op'ing, so the corruption is visible to the caller.
    #[error("queryable record {id} is missing its submission receipt")]
    SubmittedRecordMissingReceipt { id: crate::registry::AttestationId },

    #[error("chain client failure: {0}")]
    ChainClient(#[from] ChainClientError),
}

/// Stage 5 result alias. Distinct from [`Result`] (Stage 3) and
/// [`AttestationResult`] (Stage 4) so each domain's errors stay typed at
/// every call site.
pub type RegistryResult<T> = std::result::Result<T, RegistryError>;

// ── Stage 6: chain wire fixture & signing-spec deliverables ────────────────────

/// Failure produced by [`crate::chain_wire`] — the chain-wire conversion,
/// canonical-bytes, signing, and address-derivation surface.
///
/// Intentionally **not** `Clone` (no consumer needs to clone a chain-wire
/// error; Stage 4's `SignerError` was `Clone` only because the fake fixture
/// stored a canned outcome).
#[derive(Debug, thiserror::Error)]
pub enum ChainWireError {
    #[error("invalid hex for field {field}: {reason}")]
    InvalidHex { field: &'static str, reason: String },

    #[error("session_id is {got} bytes; max allowed is {max}")]
    SessionIdTooLong { got: usize, max: usize },

    #[error("Ed25519 signing failure: {0}")]
    Signing(String),

    #[error("chain-wire serialization failure: {0}")]
    Serialization(String),
}

/// Stage 6 result alias. Distinct from the Stage 3/4/5 aliases so the
/// chain-wire surface stays in its own typed lane.
pub type ChainWireResult<T> = std::result::Result<T, ChainWireError>;

// ── Stage 11a: proof generation / verification ──────────────────────────────────

/// Failure returned by a [`crate::proof::ProofBackend`] implementation when
/// proof generation cannot complete. Stage 11a's [`crate::proof::MockProofBackend`]
/// is infallible and never produces this; the variant exists for the real
/// backends Stage 11b will plug in (ezkl / risc0 / sp1 / …).
///
/// Intentionally `Clone` so the operator binary can record the same backend
/// failure in both the registry's `error_message` field and the tracing
/// event without re-running the prover.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProofBackendError {
    /// Generic catch-all for backend-internal failures (e.g. the prover
    /// crashed, ran out of memory, timed out, produced ill-formed output).
    /// The string is the backend's own diagnostic, captured verbatim.
    #[error("proof backend failure: {0}")]
    BackendInternal(String),
}

/// Failure returned by a [`crate::proof::ProofVerifier`] implementation.
/// Distinct from "verification returned false" — that's a normal
/// `Ok(false)`, not an error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProofVerifierError {
    /// The verifier hit an internal failure (input parse error, malformed
    /// proof structure, runtime panic caught and translated). A successful
    /// "the proof does not satisfy the public inputs" result is `Ok(false)`,
    /// **not** this variant.
    #[error("proof verifier failure: {0}")]
    VerifierInternal(String),

    /// Stage 11b.1.b: the caller invoked [`crate::proof::ProofVerifier::verify`]
    /// on a backend that needs the full [`crate::proof::ProofArtifactBody`]
    /// to bind backend-specific public inputs (e.g. raw i16 tensors carried
    /// in `metadata.public_inputs`). Such backends override `verify_artifact`
    /// and their `verify` returns this error to make the contract explicit
    /// — `verify(&[u8], &PublicInputs)` alone is not enough.
    ///
    /// Callers should switch to
    /// [`crate::proof::ProofVerifier::verify_artifact`] (the operator
    /// dispatch entry point).
    #[error(
        "this proof verifier requires the full ProofArtifactBody — call verify_artifact() instead of verify(): {0}"
    )]
    RequiresArtifactDispatch(String),
}

/// Composite failure returned by [`crate::proof::produce_proof_artifact`]
/// — the Stage 11a orchestrator that wraps backend invocation, metadata
/// composition, file write, and SNIP V2 publish.
///
/// Wraps the underlying typed errors so the operator binary can pattern-
/// match on the failure stage without parsing strings.
#[derive(Debug, thiserror::Error)]
pub enum ProofPipelineError {
    #[error("proof backend failed: {0}")]
    Backend(#[from] ProofBackendError),

    #[error("proof artifact filesystem I/O failure: {0}")]
    Io(#[from] std::io::Error),

    #[error("proof artifact JSON envelope serialization failure: {0}")]
    Serialize(#[from] serde_json::Error),

    #[error("proof artifact publish failure: {0}")]
    Artifact(#[from] ProofArtifactError),
}

// ── Stage 13.0: integrity-evidence chain anchoring ────────────────────────────

/// Failure produced by [`crate::evidence_anchor`] — the chain-
/// anchoring surface for Stage 12.25 signed-chain-report
/// artifacts.
///
/// Closed reason-tag set; the
/// [`crate::evidence_anchor::evidence_anchor_reason_tag`] mapper
/// is the single source of truth for the stable strings that
/// flow into `event=...` lines.
///
/// Intentionally **not** `Clone` (`io::Error` is not `Clone`).
/// Mirrors the Stage 12.20+ closed-error posture.
#[derive(Debug, thiserror::Error)]
pub enum EvidenceAnchorError {
    /// Wire payload's `anchor_schema_version` is not the locked
    /// v1 constant. Surfaced before any crypto burn.
    #[error("unsupported anchor schema version: got {got}, expected {expected}")]
    UnsupportedAnchorSchemaVersion { got: u32, expected: u32 },

    /// Wrapper's `artifact_schema_version` is outside the
    /// supported set for the declared `artifact_kind`.
    #[error("unsupported artifact schema version for {kind}: got {got}, expected {expected}")]
    UnsupportedArtifactSchemaVersion {
        kind: &'static str,
        got: u32,
        expected: u32,
    },

    /// Wrapper's `artifact_kind` is not in the closed set this
    /// build understands. Defense-in-depth; serde already
    /// refuses unknown variants at parse time.
    #[error("unsupported artifact kind: {kind}")]
    UnsupportedArtifactKind { kind: String },

    /// Stage 12.25 wrapper's own Ed25519 signature did not
    /// verify under its embedded `signer_pubkey_hex`. We do not
    /// anchor unverifiable artifacts.
    #[error("wrapper signature invalid (Stage 12.25 wrapper failed verify before anchoring)")]
    WrapperSignatureInvalid,

    /// Cheap pre-check: `--submitter-seed`-derived pubkey does
    /// not equal the wrapper's `signer_pubkey_hex`. Same-key-
    /// submitter rule (Stage 13.0). Surfaced before crypto.
    #[error(
        "submitter pubkey mismatch: seed derives {derived_pubkey_hex}, wrapper signed by {wrapper_pubkey_hex}"
    )]
    SubmitterPubkeyMismatch {
        derived_pubkey_hex: String,
        wrapper_pubkey_hex: String,
    },

    /// Anchor `submitter_signature` did not verify under
    /// `digest.signer_pubkey`.
    #[error("submitter signature invalid")]
    SubmitterSignatureInvalid,

    /// Recomputed `blake3(raw_bytes)` did not match the
    /// anchor's `digest.artifact_hash`. Returned by the verify
    /// commands when the artifact bytes have diverged from what
    /// was anchored.
    #[error("artifact hash mismatch: recomputed {recomputed_hex}, anchored {anchored_hex}")]
    ArtifactHashMismatch {
        recomputed_hex: String,
        anchored_hex: String,
    },

    /// Same-key submitter rule enforced at verify time: the
    /// stored / supplied anchor's `digest.signer_pubkey` does
    /// not equal the parsed Stage 12.25 wrapper's
    /// `signer_pubkey_hex`. Defends against a hand-edited
    /// registry record or a tampered standalone anchor that
    /// re-uses the artifact hash but swaps in a different
    /// signer pubkey (with a valid signature by that other
    /// key). The wrapper's pubkey is the source of truth — the
    /// chain anchor MUST be authored by the same key that
    /// signed the artifact.
    #[error(
        "anchored signer pubkey mismatch: wrapper signed by {wrapper_pubkey_hex}, \
         anchor records {anchored_pubkey_hex}"
    )]
    AnchoredSignerPubkeyMismatch {
        wrapper_pubkey_hex: String,
        anchored_pubkey_hex: String,
    },

    /// Registry lookup miss (no record for the supplied
    /// `--artifact-hash-hex` / `--tx-id`).
    #[error("anchor not found in registry: {selector}")]
    AnchorNotFound { selector: String },

    /// Failed to read the submitter seed file or it was not
    /// exactly 32 bytes long. Mirrors Stage 12 seed-handling
    /// refusal posture.
    #[error("malformed submitter seed file at {path}: {reason}")]
    MalformedSeedFile {
        path: std::path::PathBuf,
        reason: String,
    },

    /// JSON parse failure on the Stage 12.25 wrapper file or a
    /// free-floating anchor file.
    #[error("malformed JSON at {path}: {source}")]
    MalformedJson {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// Time string in the wrapper could not be parsed as RFC
    /// 3339 / could not be converted to a Unix timestamp.
    #[error("malformed signed_at_utc {raw:?}: {reason}")]
    MalformedSignedAtUtc { raw: String, reason: String },

    /// Canonical bytes serialization failed (bincode-1).
    /// Mirrors the Stage 6 error shape.
    #[error("anchor canonical-bytes serialization failed: {0}")]
    CanonicalSerialization(String),

    /// Ed25519 signing / pubkey decode failure.
    #[error("anchor Ed25519 signing failure: {0}")]
    Signing(String),

    /// Chain client failure during submit / query.
    #[error("chain client failure: {0}")]
    ChainClient(#[from] ChainClientError),

    /// FS IO failure. Path-attached for clean operator messages.
    #[error("anchor I/O error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    // ── Stage 13.2 additions — chain-touching CLI preflight refusals ──
    //
    // These six variants surface ONLY when an anchor command is
    // invoked with the chain-touching flags (`--rpc-url` /
    // `--expect-chain-id` / `--allow-submit` /
    // `--allow-mainnet-submit`). Stage 13.0 stub-only flows never
    // produce them. The reason-tag mapper in
    // [`crate::evidence_anchor::evidence_anchor_reason_tag`]
    // routes each one-to-one to the documented closed-set string.
    /// `--expect-chain-id` did not match the chain's reported
    /// `params.chain_id`. CLI preflight gate; submit / query /
    /// reconcile all refuse here before any anchor RPC fires.
    #[error(
        "chain_id mismatch: --expect-chain-id {expected}, chain reports {actual}"
    )]
    ChainIdMismatch { expected: u64, actual: u64 },

    /// Anchor RPC activation not reached. Fires on non-mainnet
    /// dormant / scheduled-but-not-yet-reached, and on mainnet
    /// scheduled-but-not-yet-reached. (Mainnet + dormant
    /// `None` fires [`Self::MainnetPolicyUnresolved`] instead;
    /// see Stage 13.1 R-packet for the locked semantic split.)
    #[error(
        "integrity_evidence_anchor not activated on chain (chain_id {chain_id}): \
         {activation_status}"
    )]
    NotActivated {
        chain_id: u64,
        /// Human-readable activation status string. Examples:
        /// "dormant (no activation height set)" or
        /// "scheduled at height H, chain head at K".
        activation_status: String,
    },

    /// Mainnet (`chain_id == 1`) + governance has not set
    /// activation (`integrity_evidence_anchor_enabled_from_height == None`).
    /// Stage 13.1 R-packet reserved this reason tag specifically
    /// to capture "mainnet anchors are not yet permitted by
    /// chain governance" without implying a permanent refusal.
    /// When mainnet sets `Some(h)`, this becomes
    /// [`Self::NotActivated`] until head reaches `h`.
    #[error(
        "mainnet anchor policy is unresolved \
         (integrity_evidence_anchor_enabled_from_height == None on chain_id 1); \
         awaiting chain governance to set an activation height"
    )]
    MainnetPolicyUnresolved,

    /// Transport-layer JSON-RPC failure produced by
    /// `omni-sumchain` (HTTP, body read, missing `result` field,
    /// non-JSON response). Distinguished from
    /// [`Self::ChainSubmitRefused`] by
    /// [`omni_sumchain::classify_chain_client_error`].
    #[error("chain RPC failure: {0}")]
    ChainRpc(String),

    /// Chain returned a JSON-RPC error object — the chain
    /// refused the call at the application layer. The chain's
    /// failure text is surfaced verbatim (chain-side failure
    /// codes 60-63 are private detail; OmniNode does not decode
    /// them).
    #[error("chain refused the call: {0}")]
    ChainSubmitRefused(String),

    /// Chain returned success (no JSON-RPC error) but the
    /// response shape could not be parsed into the expected
    /// DTO, OR the deserialized DTO carried an unrecognized
    /// enum string (e.g. `status: "foo"`). Distinguished from
    /// [`Self::ChainRpc`] / [`Self::ChainSubmitRefused`] by
    /// [`omni_sumchain::classify_chain_client_error`].
    #[error("chain response malformed: {0}")]
    ChainResponseMalformed(String),

    // ── Stage 13.4 additions — local anchor-registry cleanup ─────────
    //
    // Five new variants, mapping one-to-one to closed-set
    // `reason=<tag>` strings on Stage 13.4 cleanup event lines.
    // Four reuse tag strings from the Stage 12.17/12.18 cleanup
    // taxonomy (`gate_required`, `cleanup_plan_hash_mismatch`,
    // `quarantine_blake3_mismatch`, `restore_target_exists`) so
    // operators reading logs across stages don't have to learn
    // two names for the same condition. The fifth tag string
    // (`cleanup_drift`) is the only new tag introduced by
    // Stage 13.4.
    /// Apply: the anchor registry's state differs from the
    /// `registry_state_hash` recorded in the plan. The operator
    /// must re-plan against the current registry state and
    /// re-apply. Stage 13.4 introduces this tag string; no
    /// Stage 12 equivalent.
    #[error(
        "anchor registry drifted since plan was generated: \
         computed {computed}, plan expected {expected}"
    )]
    CleanupDrift { computed: String, expected: String },

    /// Apply: the plan's `cleanup_plan_hash` does not match the
    /// recomputed canonical-hash of its remaining fields. The
    /// plan was hand-edited (or corrupted on disk). Reuses the
    /// Stage 12.17 tag string `cleanup_plan_hash_mismatch`.
    #[error(
        "anchor cleanup plan hash mismatch: computed {computed}, plan declares {expected}"
    )]
    CleanupPlanHashMismatch { computed: String, expected: String },

    /// Apply: a gated action (e.g. `QuarantineStaleOpenRecord`)
    /// is in the plan but the operator did not supply the
    /// required apply-time flag (e.g. `--allow-stale-quarantine`).
    /// Reuses the Stage 12.17 tag string `gate_required`.
    #[error(
        "anchor cleanup gate required: action {action_kind} requires {gate_flag}"
    )]
    CleanupGateRequired {
        action_kind: &'static str,
        gate_flag: &'static str,
    },

    /// Restore: the quarantined bytes' BLAKE3 hash does not
    /// match the manifest's recorded hash. Either the
    /// quarantine subtree was hand-edited or the manifest was
    /// regenerated against different bytes. Refuses with no FS
    /// mutation. Reuses the Stage 12.18 tag string
    /// `quarantine_blake3_mismatch`.
    #[error(
        "quarantine BLAKE3 mismatch for {source_relative}: \
         file hashes to {computed}, manifest declares {expected}"
    )]
    QuarantineBlake3Mismatch {
        source_relative: String,
        computed: String,
        expected: String,
    },

    /// Restore: the restore target path is already populated.
    /// No clobber — operator must rename/delete the conflict
    /// before retrying. Reuses the Stage 12.18 tag string
    /// `restore_target_exists`.
    #[error("restore target already exists: {target_path}")]
    RestoreTargetExists { target_path: std::path::PathBuf },

    /// Apply / Restore: an action's `source_relative` (or a
    /// quarantine-manifest entry's `quarantine_relative`)
    /// fails per-kind path-shape validation. Defends against
    /// path traversal (`..`), absolute paths, or unexpected
    /// separators in operator-supplied JSON. Refuses BEFORE
    /// any FS mutation. Stage 13.4 introduces this tag string.
    #[error(
        "anchor cleanup invalid path for action {action_kind}: \
         {source_relative:?} ({reason})"
    )]
    CleanupInvalidPath {
        action_kind: &'static str,
        source_relative: String,
        reason: &'static str,
    },

    /// Apply: the plan's `schema_version` is not the locked
    /// `ANCHOR_CLEANUP_PLAN_SCHEMA_VERSION`. Future-schema
    /// plans with re-computed hashes would otherwise apply as
    /// v1; refusing here is the schema-version-as-boundary
    /// contract. Stage 13.4 introduces this tag string.
    #[error(
        "unsupported anchor cleanup plan schema version: got {got}, expected {expected}"
    )]
    CleanupPlanSchemaUnsupported { got: u32, expected: u32 },

    // ── Stage 13.5 additions — local export / import verification ─
    //
    // Six new closed reason-tag strings, paralleling the Stage 13.4
    // shape but on a distinct export-side concept: a portable
    // handoff manifest plus copied bytes. Refusals fire in the
    // verify command's fixed preflight order (schema → manifest
    // hash → per-entry path → file presence → BLAKE3 → per-record
    // checks → strict-mode gate).
    /// Verify: the export manifest's `schema_version` is not the
    /// locked `EVIDENCE_ANCHOR_EXPORT_MANIFEST_SCHEMA_VERSION`.
    /// Refused FIRST — before the manifest-hash check — so a
    /// future-schema manifest with a re-computed hash cannot
    /// smuggle through as v1.
    #[error(
        "unsupported anchor export manifest schema version: got {got}, expected {expected}"
    )]
    ExportManifestSchemaUnsupported { got: u32, expected: u32 },

    /// Verify: the recomputed canonical-bytes BLAKE3 of the
    /// manifest (with `export_manifest_hash` blanked) does not
    /// match the manifest's declared `export_manifest_hash`.
    /// Indicates the manifest was hand-edited or transport
    /// corrupted.
    #[error(
        "anchor export manifest hash mismatch: computed {computed}, manifest declares {expected}"
    )]
    ExportManifestHashMismatch { computed: String, expected: String },

    /// Apply / Verify: a manifest entry's `relative_path` is
    /// invalid — absolute, contains `..`, contains backslash,
    /// or violates the per-kind shape (anchors/<64hex>.json,
    /// artifacts/<64hex>, signed_chain_reports/<safe-basename>).
    /// Refuses BEFORE any FS mutation / read.
    #[error(
        "anchor export invalid path for {entry_kind}: {relative_path:?} ({reason})"
    )]
    ExportInvalidPath {
        entry_kind: &'static str,
        relative_path: String,
        reason: &'static str,
    },

    /// Verify: a copied file's BLAKE3-of-bytes (or its byte
    /// length) does not match the manifest's declared
    /// `blake3_hex` / `bytes`. Indicates the export tree was
    /// hand-edited or transport corrupted.
    #[error(
        "anchor export BLAKE3 mismatch for {relative_path}: computed {computed}, manifest declares {expected}"
    )]
    ExportBlake3Mismatch {
        relative_path: String,
        computed: String,
        expected: String,
    },

    /// Verify: a manifest entry's per-record metadata (artifact
    /// hash, tx_id, status) does not match the record file's
    /// own fields, OR a paired `artifact_bytes` entry's
    /// declared `artifact_hash_hex` does not match the
    /// `anchor_record`'s `tx_data.digest.artifact_hash`
    /// (artifact-hash binding refusal). Single closed tag for
    /// every "manifest claim does not match the underlying
    /// byte fact" case (Q9 fold).
    #[error(
        "anchor export entry metadata mismatch for {relative_path} (field {field}): \
         computed {computed}, manifest declares {manifest}"
    )]
    ExportEntryMetadataMismatch {
        relative_path: String,
        field: &'static str,
        computed: String,
        manifest: String,
    },

    /// Verify, `--strict` only: an `anchor_record` entry does
    /// not have a matching `artifact_bytes` entry for the same
    /// `artifact_hash_hex`. Routes through the closed taxonomy
    /// (not a clap-level usage error) because the missing
    /// pairing is only knowable after parsing the manifest.
    #[error(
        "anchor export strict mode: anchor_record {anchor_record_relative_path} \
         has no paired artifact_bytes entry for artifact_hash_hex {artifact_hash_hex}"
    )]
    ExportStrictModeArtifactBytesMissing {
        anchor_record_relative_path: String,
        artifact_hash_hex: String,
    },

    // ── Stage 13.6 additions — local export-import / registry restore ─
    //
    // One new variant, one new closed `reason=` tag string.
    // Disambiguates the artifact-hash vs tx_id conflict case via
    // a closed `field` discriminator — single variant per the
    // Stage 13.6 reviewer-locked decision.
    /// Import: the target registry already carries a different
    /// anchor under the same key.
    ///
    /// `field = "artifact_hash"` — there is a file at
    /// `<registry>/<artifact_hash_hex>.json` whose BLAKE3 differs
    /// from the manifest's recorded `blake3_hex` (a different
    /// record under the same hash).
    ///
    /// `field = "tx_id"` — `tx_index.json` already maps the
    /// import's `tx_id` to a different `artifact_hash_hex` than
    /// the manifest declares.
    ///
    /// Byte-equal records under the same hash + same tx_id are
    /// idempotent (`skipped_already_imported`) — not refused.
    #[error(
        "anchor import target exists (field {field}): \
         artifact_hash_hex {artifact_hash_hex}, tx_id {tx_id}"
    )]
    ImportTargetExists {
        field: &'static str,
        artifact_hash_hex: String,
        tx_id: String,
    },
}

/// Stage 13.0 result alias. Distinct from earlier-stage aliases
/// so the evidence-anchor surface stays in its own typed lane.
pub type EvidenceAnchorResult<T> = std::result::Result<T, EvidenceAnchorError>;
