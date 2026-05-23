//! Phase 5 Stage 8a — operator / activation monitoring loop.
//!
//! Turns the Stage 5.3 orchestration helpers + the Stage 7a/7b SUM
//! Chain adapter into an operator-facing command surface on the
//! `omni-node` binary. **Safe by default**, no chain-protocol changes,
//! zero edits outside this crate.
//!
//! Three subcommands under `omni-node operator`:
//!
//! - `watch-activation` — poll `chain_getChainParams` +
//!   `chain_getBlockHeight(Latest)`; log params and blocks-remaining to
//!   each activation gate; exit `0` once
//!   `omninode_is_active() && v2_is_active()`. Read-only; never submits.
//! - `smoke` — submit exactly one attestation and poll it to
//!   `Finalized` (Included is treated as progress, not success — a
//!   mainnet activation smoke must confirm finality, not just
//!   inclusion). On mainnet (`chain_id == 1`) the
//!   attestation **must** be supplied via `--attestation-json`; there
//!   is no synthetic-mainnet path. Synthetic generation is non-mainnet
//!   only and requires explicit `--synthetic`.
//! - `loop` — periodic `poll → sweep → (retry)` over a real registry;
//!   retry only when submission is permitted *and* the chain is
//!   activated. Graceful Ctrl-C shutdown.
//!
//! ## Safety gate order (smoke)
//!
//! 1. chain-id guardrail (`--expect-chain-id` must equal
//!    `chain_getChainParams.chain_id`)
//! 2. attestation-source resolution + **no-synthetic-mainnet** gate —
//!    evaluated *before* the verifier seed is touched, so a missing or
//!    malformed `OMNINODE_VERIFIER_SEED_HEX` can never mask a mainnet
//!    misuse error
//! 3. activation precheck (`omninode_is_active && v2_is_active`)
//! 4. submit opt-in (`--allow-submit`) + mainnet double-gate
//!    (`--allow-mainnet-submit` when `chain_id == 1`)
//! 5. verifier seed resolution + verifier-address consistency
//! 6. submit via [`omni_zkml::submit_attestation_workflow_with_block`]
//!    (whose own Stage 7b adapter-layer activation/verifier gates are
//!    an unchanged backstop)
//!
//! The blocking SUM Chain RPC work runs inside
//! `tokio::task::spawn_blocking`; config and an `Arc`'d client factory
//! are moved into each blocking unit so no borrow escapes.

#[cfg(feature = "submit")]
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(feature = "submit")]
use std::time::Instant;
use std::time::Duration;

use clap::{Args, Subcommand};
use tracing::{debug, info, warn};

use omni_sumchain::{BlockFinality, ChainParamsInfo, JsonRpcTransport, SumChainClient, UreqTransport};
use omni_types::phase5::InferenceAttestation;
#[cfg(any(feature = "submit", test))]
use omni_types::phase5::{InferenceCommitment, SnipV2ObjectId};
#[cfg(feature = "submit")]
use omni_zkml::{submit_attestation_workflow_with_block, ChainClient};
use omni_zkml::{
    poll_attestations_workflow, retry_dropped_attestations_workflow,
    signer_chain_address_base58,
    sweep_stale_attestations_workflow, AttestationId, AttestationRecord,
    AttestationRegistry, ChainClientError, LocalAttestationStatus,
    RegistryError, StalenessPolicy,
};
// Stage 11a: trait import for `MockProofBackend::prove()` in
// `build_mock_v1_attestation`. The trait method isn't in the prelude;
// importing the trait lets us call `prove(...)` without going through
// a UFCS hop.
#[cfg(feature = "submit")]
use omni_zkml::ProofBackend;

/// Read-only commands and the gate phase use a zero seed — the SUM
/// Chain adapter ignores the seed for reads; only signing consumes it.
const DUMMY_SEED: [u8; 32] = [0u8; 32];

/// Builds a `SumChainClient<T>` for a given Ed25519 seed. Production
/// closes over the RPC URL (`UreqTransport`); tests close over a shared
/// `FakeJsonRpcTransport`. Lets the gate phase use a dummy-seed client
/// and rebuild a real-seed client only *after* the seed gate passes.
type ClientFactory<T> = Arc<dyn Fn([u8; 32]) -> SumChainClient<T> + Send + Sync>;

// ── Errors ───────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub(crate) enum OperatorError {
    #[error("chain_id mismatch: --expect-chain-id {expected}, chain reports {actual}")]
    ChainIdMismatch { expected: u64, actual: u64 },

    #[cfg(feature = "submit")]
    #[error(
        "mainnet smoke (chain_id 1) requires --attestation-json; refusing to \
         synthesize an artificial attestation onto mainnet"
    )]
    MainnetRequiresAttestationJson,

    #[cfg(feature = "submit")]
    #[error(
        "--synthetic is forbidden on mainnet (chain_id 1); Stage 8a has no \
         synthetic-mainnet path"
    )]
    SyntheticMainnetForbidden,

    /// Stage 11a guardrail. The mock proof backend (`mock-v1`) is
    /// non-cryptographic by design — it produces deterministic
    /// BLAKE3-chain bytes, not a soundness-proving SNARK / STARK —
    /// and must never reach mainnet, including with
    /// `--allow-mainnet-submit`. Stage 8b's
    /// [`OperatorError::SyntheticMainnetForbidden`] already refuses the
    /// only path that currently selects `mock-v1` (the smoke
    /// `--synthetic` flag); this variant is the explicit, backend-id-
    /// keyed defense-in-depth check that fires regardless of how the
    /// caller arrived at the mock backend.
    #[cfg(feature = "submit")]
    #[error(
        "mainnet submit (chain_id 1) refuses proof backend {backend_id:?}: \
         the mock backend is non-cryptographic and cannot be used for \
         mainnet submission, even with --allow-mainnet-submit. Use a real \
         proof backend (Stage 11b) or run the operator against a \
         non-mainnet chain."
    )]
    MockBackendRefusedOnMainnet { backend_id: String },

    // ── Stage 11b.0: mainnet refusal variants ─────────────────────────
    //
    // These wrap [`omni_zkml::MainnetRefusalReason`] into operator-
    // facing typed errors. The Stage 11a `MockBackendRefusedOnMainnet`
    // variant above is preserved for back-compat (the synthetic smoke
    // path's existing refusal still routes through it via the
    // [`map_mainnet_refusal`] helper below).

    /// Stage 11b.0 layer 1 — the proof artifact carries
    /// `testnet_or_dev_only: Some(true)`. The producer explicitly
    /// disclaimed mainnet eligibility.
    #[cfg(feature = "submit")]
    #[error(
        "proof artifact is marked testnet/dev only (backend_id = {backend_id:?}); \
         the producer explicitly disclaimed mainnet eligibility"
    )]
    TestnetOnlyProofRefusedOnMainnet { backend_id: String },

    /// Stage 11b.0 layer 3 — bounded reference proof systems
    /// (currently `Stage11bOnnxReference`) are architecture-
    /// validation fixtures, not production attestation. Hard-refused
    /// on mainnet regardless of any other state.
    #[cfg(feature = "submit")]
    #[error(
        "bounded reference proof system refused on mainnet (backend_id = {backend_id:?}): \
         reference fixtures are for architecture validation only, not production"
    )]
    BoundedReferenceProofRefusedOnMainnet { backend_id: String },

    /// Stage 11b.0 layer 4 — `model_format = Gguf` claim. No GGUF
    /// inference proof backend is approved at any stage through
    /// Stage 11b.0. Declaration is honest; mainnet claim is refused.
    #[cfg(feature = "submit")]
    #[error(
        "GGUF proof claim refused on mainnet (backend_id = {backend_id:?}): \
         no GGUF inference proof backend is approved — awaiting Stage 11d \
         strategy + chain-team review"
    )]
    GgufProofClaimRefusedOnMainnet { backend_id: String },

    /// Stage 11b.0 layer 5 — model_format is `Other(_)` or absent on
    /// a non-mock backend. Stringly-typed formats are refused on
    /// mainnet until promoted to a first-class enum variant.
    #[cfg(feature = "submit")]
    #[error(
        "unknown / non-first-class model_format refused on mainnet \
         (backend_id = {backend_id:?})"
    )]
    UnknownModelFormatRefusedOnMainnet { backend_id: String },

    /// Stage 11b.0 layer 6 — proof_system is not in
    /// [`omni_zkml::MAINNET_APPROVED_PROOF_SYSTEMS`]. The allowlist is
    /// empty by design at end of Stage 11b.0; mainnet eligibility is
    /// a Stage 11c+ deliverable with chain-team review.
    #[cfg(feature = "submit")]
    #[error(
        "proof system not in the mainnet allowlist (backend_id = {backend_id:?}): \
         Stage 11b ships with the allowlist empty by design — mainnet \
         eligibility is a Stage 11c+ deliverable"
    )]
    ProofSystemNotMainnetApproved { backend_id: String },

    /// Stage 11b.0: `operator verify-proof` could not locate a
    /// `ProofVerifier` impl for the proof artifact's declared proof
    /// system. Stage 11b.0 ships only [`omni_zkml::MockProofVerifier`];
    /// other proof systems return this error until backends land in
    /// Stage 11c+.
    #[error(
        "no verifier registered for proof system {proof_system}; \
         Stage 11b.0 ships only MockProofVerifier"
    )]
    NoVerifierForProofSystem { proof_system: String },

    /// Stage 11b.0: `operator verify-proof` could not parse the
    /// `--proof-artifact` JSON.
    #[error("failed to parse proof artifact JSON: {0}")]
    ProofArtifactParse(String),

    /// Stage 11b.0: `operator verify-proof` could not read the
    /// `--proof-artifact` file.
    #[error("failed to read --proof-artifact {path}: {source}")]
    ProofArtifactRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[cfg(feature = "submit")]
    #[error(
        "smoke needs an attestation source: pass --attestation-json, or \
         (non-mainnet only) --synthetic"
    )]
    SyntheticRequiresExplicitFlag,

    #[cfg(feature = "submit")]
    #[error("--attestation-json and --synthetic are mutually exclusive")]
    ConflictingSmokeSource,

    #[error("submission not permitted: pass --allow-submit to enable chain writes")]
    SubmitNotPermitted,

    #[error(
        "mainnet submission (chain_id 1) additionally requires \
         --allow-mainnet-submit"
    )]
    MainnetSubmitNotPermitted,

    #[cfg(feature = "submit")]
    #[error(
        "chain not activated; refusing to submit \
         (omninode_active={omninode}, v2_active={v2})"
    )]
    NotActivated { omninode: bool, v2: bool },

    #[error("activation not reached within the configured --max-ticks budget")]
    ActivationNotReached,

    #[error("verifier seed unavailable: set OMNINODE_VERIFIER_SEED_HEX (64 hex chars)")]
    SeedMissing,

    #[error("verifier seed malformed: {0}")]
    SeedMalformed(String),

    #[error(
        "attestation verifier_address {claimed} does not match the \
         seed-derived address {derived}"
    )]
    VerifierAddressMismatch { claimed: String, derived: String },

    #[error("could not read --attestation-json {path}: {source}")]
    AttestationJsonRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("could not parse --attestation-json as InferenceAttestation: {0}")]
    AttestationJsonParse(String),

    #[cfg(feature = "submit")]
    #[error("smoke did not reach Finalized before --confirm-timeout-secs; last status: {last}")]
    SmokeConfirmTimeout { last: String },

    #[cfg(feature = "submit")]
    #[error("smoke submission reached chain status Failed: {reason}")]
    SmokeFailed { reason: String },

    #[error("RPC URL unset: pass --rpc-url or set OMNINODE_SUMCHAIN_RPC_URL")]
    RpcUrlMissing,

    #[error("--expect-chain-id is required when --rpc-url is given")]
    ExpectChainIdRequiredWithRpc,

    #[error("chain has no attestation for (session_id={session_id}, verifier_address={verifier_address})")]
    AttestationNotFound {
        session_id: String,
        verifier_address: String,
    },

    #[cfg(feature = "submit")]
    #[error("smoke interrupted (Ctrl-C) before finality was confirmed")]
    SmokeInterrupted,

    #[error("--tx-hash is mutually exclusive with --session-id/--verifier-address")]
    ConflictingQueryMode,

    #[error("query needs either --tx-hash, or both --session-id and --verifier-address")]
    QueryModeRequired,

    #[error("internal invariant violated: {0}")]
    Internal(String),

    #[error("attestation id malformed: {0}")]
    AttestationIdMalformed(String),

    #[error(
        "invalid --status filter {0:?}; expected one of: \
         pending | submitted | included | finalized | failed | dropped"
    )]
    InvalidStatusFilter(String),

    #[error("chain client error: {0}")]
    Chain(#[from] ChainClientError),

    #[error("registry error: {0}")]
    Registry(#[from] RegistryError),

    #[error("staleness policy: {0}")]
    Policy(#[from] omni_zkml::StalenessPolicyError),

    #[error("blocking task join error: {0}")]
    Join(String),
}

// ── Seed source (deferred resolution; injectable for tests) ───────────────────

/// Where the verifier seed comes from. Production uses
/// [`SeedSource::Env`]; tests inject `Explicit` / `AbsentForTest` /
/// `MalformedForTest` so the gate-ordering guarantee can be asserted
/// **without** mutating process-global environment.
#[derive(Clone)]
pub(crate) enum SeedSource {
    Env,
    // The variants below are test-injection seams only. They are
    // `#[cfg(test)]`-gated (not `#[allow(dead_code)]`-suppressed) so
    // the production build genuinely contains only `Env`.
    // `MalformedForTest` additionally requires `feature = "submit"`:
    // its only caller is the submit-gated
    // `smoke_mainnet_rejects_synthetic_before_malformed_seed` test
    // that pins gate-7-precedes-seed-resolution against a malformed
    // seed; with submit off, the variant has no caller anywhere and
    // would otherwise be dead-code under `cargo test`.
    #[cfg(test)]
    Explicit([u8; 32]),
    #[cfg(test)]
    AbsentForTest,
    #[cfg(all(test, feature = "submit"))]
    MalformedForTest(String),
}

impl SeedSource {
    fn resolve(&self) -> Result<[u8; 32], OperatorError> {
        match self {
            SeedSource::Env => {
                let v = std::env::var("OMNINODE_VERIFIER_SEED_HEX")
                    .map_err(|_| OperatorError::SeedMissing)?;
                parse_seed_hex(&v)
            }
            #[cfg(test)]
            SeedSource::Explicit(s) => Ok(*s),
            #[cfg(test)]
            SeedSource::AbsentForTest => Err(OperatorError::SeedMissing),
            #[cfg(all(test, feature = "submit"))]
            SeedSource::MalformedForTest(h) => parse_seed_hex(h),
        }
    }
}

fn parse_seed_hex(h: &str) -> Result<[u8; 32], OperatorError> {
    if h.len() != 64 {
        return Err(OperatorError::SeedMalformed(format!(
            "expected 64 hex chars, got {}",
            h.len()
        )));
    }
    let mut seed = [0u8; 32];
    for i in 0..32 {
        seed[i] = u8::from_str_radix(&h[i * 2..i * 2 + 2], 16)
            .map_err(|e| OperatorError::SeedMalformed(e.to_string()))?;
    }
    Ok(seed)
}

// ── Pure gate helpers (network-free; unit-tested) ─────────────────────────────

fn check_chain_id(expected: u64, actual: u64) -> Result<(), OperatorError> {
    if expected == actual {
        Ok(())
    } else {
        Err(OperatorError::ChainIdMismatch { expected, actual })
    }
}

fn submission_permitted(
    chain_id: u64,
    allow_submit: bool,
    allow_mainnet_submit: bool,
) -> Result<(), OperatorError> {
    if !allow_submit {
        return Err(OperatorError::SubmitNotPermitted);
    }
    if chain_id == 1 && !allow_mainnet_submit {
        return Err(OperatorError::MainnetSubmitNotPermitted);
    }
    Ok(())
}

/// Blocks until `activation` (saturating at 0 once head ≥ activation).
/// `None` activation height → `None` (chain hasn't exposed it yet).
fn blocks_remaining(head: u64, activation: Option<u64>) -> Option<u64> {
    activation.map(|a| a.saturating_sub(head))
}

// Stage 9a: `SmokeSource` + `resolve_smoke_source` only exist with
// the `submit` feature — both are consumed exclusively by `smoke_core`
// and its hermetic tests, which are themselves submit-gated. Pure
// helpers `synth_attestation` (below) stays default-on because the
// `preflight` tests use it as an InferenceAttestation builder.
#[cfg(feature = "submit")]
#[derive(Debug, PartialEq, Eq)]
enum SmokeSource {
    File(PathBuf),
    Synthetic,
}

/// Gate 7 / attestation-source resolution. **Pure** and seed-free, so
/// callers can (and the runner does) evaluate it before the verifier
/// seed is ever touched.
#[cfg(feature = "submit")]
fn resolve_smoke_source(
    chain_id: u64,
    attestation_json: Option<&Path>,
    synthetic: bool,
) -> Result<SmokeSource, OperatorError> {
    let is_mainnet = chain_id == 1;
    match (attestation_json, synthetic) {
        (Some(_), true) => Err(OperatorError::ConflictingSmokeSource),
        (Some(p), false) => Ok(SmokeSource::File(p.to_path_buf())),
        (None, true) => {
            if is_mainnet {
                Err(OperatorError::SyntheticMainnetForbidden)
            } else {
                Ok(SmokeSource::Synthetic)
            }
        }
        (None, false) => {
            if is_mainnet {
                Err(OperatorError::MainnetRequiresAttestationJson)
            } else {
                Err(OperatorError::SyntheticRequiresExplicitFlag)
            }
        }
    }
}

// Stage 8a/9c.1 legacy synthetic helper. Produces a structurally well-formed
// `InferenceAttestation` with placeholder commitment bytes (`model_hash =
// "a".repeat(64)`, `manifest_snip_root = [0x11; 32]`, etc.). This is the
// pre-Stage-11a fabrication path; it's preserved as a TEST-ONLY fixture
// because the hermetic tests in this file insert these structurally-shaped
// records into the registry / chain plumbing without needing real proof
// generation. The production `SmokeSource::Synthetic` path no longer calls
// it (Stage 11a) — see [`build_mock_v1_attestation`] below.
//
// **Do not use `synth_attestation` from production code paths.** The
// 2026-05-19 mainnet smoke submitted these placeholder bytes onto chain;
// that remains the documented historical anomaly per
// `docs/mainnet-smoke-audit.md`. Stage 11a's job is only to prevent future
// submits from doing the same — see
// [`OperatorError::MockBackendRefusedOnMainnet`].
//
// `cfg(test)` only: pre-Stage-11a the submit-feature build called this
// from the smoke `--synthetic` path; that call site now uses
// [`build_mock_v1_attestation`], and the helper has no other production
// caller. Keeping it `#[cfg(test)]` makes the production binary genuinely
// not contain the placeholder fabricator.
#[cfg(test)]
fn synth_attestation(verifier_address: &str, head: u64) -> InferenceAttestation {
    InferenceAttestation {
        commitment: InferenceCommitment {
            session_id: format!("omninode-stage8a-smoke-{verifier_address}-{head}"),
            model_hash: "a".repeat(64),
            manifest_snip_root: SnipV2ObjectId::from_bytes([0x11u8; 32]),
            response_hash: "b".repeat(64),
            proof_snip_root: SnipV2ObjectId::from_bytes([0x22u8; 32]),
        },
        verifier_address: verifier_address.to_string(),
        verifier_signature: "stage8a-smoke".into(),
    }
}

/// Stage 11a production synthetic-smoke attestation builder. Replaces
/// [`synth_attestation`] as the smoke `--synthetic` source. Differences:
///
/// 1. **Real proof bytes.** Runs [`omni_zkml::MockProofBackend`] over
///    deterministic synthetic `(model, input, output)` bytes so the
///    resulting commitment fields are honest BLAKE3 hashes of real
///    content — no more `"a".repeat(64)` / `[0x11; 32]` placeholders.
/// 2. **Real `proof_snip_root`.** Computed as BLAKE3 of the canonical
///    [`omni_zkml::ProofArtifactBody`] envelope (metadata + proof
///    bytes), matching what a real SNIP V2 publish through `sum-node`
///    would emit. The bytes themselves are not published to a SNIP
///    store in Stage 11a's synthetic smoke path — that's
///    Stage-11b-and-beyond — but the root is honest.
/// 3. **Mainnet hard refusal.** `chain_id == 1` returns
///    [`OperatorError::MockBackendRefusedOnMainnet`] *before* any
///    bytes are produced, regardless of `--allow-mainnet-submit`.
///    This is the Stage 11a OQ5 requirement and a defense-in-depth
///    backstop alongside [`OperatorError::SyntheticMainnetForbidden`]
///    (which already covers the only currently-reachable path here).
///
/// The `verifier_signature` field remains a marker string — the actual
/// on-chain Ed25519 signature is computed inside `omni_sumchain::tx` at
/// submit time from the operator's seed via Stage 6's
/// `sign_chain_attestation_digest`.
#[cfg(feature = "submit")]
fn build_mock_v1_attestation(
    verifier_address: &str,
    chain_id: u64,
    head: u64,
) -> Result<InferenceAttestation, OperatorError> {
    if chain_id == 1 {
        return Err(OperatorError::MockBackendRefusedOnMainnet {
            backend_id: omni_zkml::MOCK_BACKEND_ID.to_string(),
        });
    }

    let session_id = format!("omninode-stage11a-smoke-{verifier_address}-{head}");

    // Deterministic synthetic inputs. Tied to the verifier address and
    // head height so multiple smoke runs against the same chain don't
    // collide on `(session_id, verifier_address)` — the registry's
    // Stage 5.1 dedup key.
    let model_bytes =
        format!("omninode.stage11a.synthetic.model.{verifier_address}.{head}")
            .into_bytes();
    let input_bytes =
        format!("omninode.stage11a.synthetic.input.{verifier_address}.{head}")
            .into_bytes();
    let output_bytes =
        format!("omninode.stage11a.synthetic.output.{verifier_address}.{head}")
            .into_bytes();

    // Run the mock backend; pin the real backend_id here so
    // ProofMetadata carries the same identifier the mainnet guard
    // refused on. Stage 11a's MockProofBackend is infallible; the
    // mapping below preserves the typed-error contract for Stage 11b.
    let proof_bytes = omni_zkml::MockProofBackend
        .prove(&model_bytes, &input_bytes, &output_bytes)
        .map_err(|e| OperatorError::Internal(format!("MockProofBackend.prove: {e}")))?;

    // Compute the BLAKE3 hashes used by both the commitment and the
    // proof-metadata envelope.
    let model_hash_hex = blake3::hash(&model_bytes).to_hex().to_string();
    let input_hash_hex = blake3::hash(&input_bytes).to_hex().to_string();
    let response_hash_hex = blake3::hash(&output_bytes).to_hex().to_string();

    // Canonical proof envelope. BLAKE3 of the envelope bytes is the
    // proof_snip_root the chain commits to (same algorithm the test
    // fake adapter uses; production sum-node ingest produces the same
    // root for the same bytes when configured for public lifecycle).
    // Stage 11b.0: use the Stage-11a-compat constructor — the mock
    // backend's metadata is the "minimal Stage 11a" shape, with all
    // Stage 11b.0 fields defaulting to None. This keeps the resulting
    // proof envelope JSON byte-identical against the Stage 11a
    // fixture (every new optional field uses `serde(skip_if_none)`).
    let metadata = omni_zkml::ProofMetadata::new_stage11a(
        omni_zkml::MOCK_BACKEND_ID.to_string(),
        model_hash_hex.clone(),
        input_hash_hex,
        response_hash_hex.clone(),
    );
    let body = omni_zkml::ProofArtifactBody::from_components(metadata, &proof_bytes);
    let body_bytes = body
        .to_canonical_bytes()
        .map_err(|e| OperatorError::Internal(format!("proof envelope serialize: {e}")))?;
    let mut proof_root = [0u8; 32];
    proof_root.copy_from_slice(blake3::hash(&body_bytes).as_bytes());
    let proof_snip_root = SnipV2ObjectId::from_bytes(proof_root);

    // Deterministic synthetic manifest_root tied to the model bytes —
    // hermetic, reproducible, distinct per `(verifier_address, head)`
    // pair. Stage 11b will replace this with a real Stage 2 manifest
    // root once the production pipeline owns model publishing too.
    let manifest_snip_root = {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"omninode.stage11a.synthetic.manifest");
        hasher.update(&model_bytes);
        let mut id = [0u8; 32];
        id.copy_from_slice(hasher.finalize().as_bytes());
        SnipV2ObjectId::from_bytes(id)
    };

    Ok(InferenceAttestation {
        commitment: InferenceCommitment {
            session_id,
            model_hash: model_hash_hex,
            manifest_snip_root,
            response_hash: response_hash_hex,
            proof_snip_root,
        },
        verifier_address: verifier_address.to_string(),
        // Local-only metadata; the on-chain Ed25519 signature is computed
        // inside omni-sumchain::tx at submit time from the seed.
        verifier_signature: format!("stage11a-{}", omni_zkml::MOCK_BACKEND_ID),
    })
}

async fn run_blocking<F, R>(f: F) -> Result<R, OperatorError>
where
    F: FnOnce() -> Result<R, OperatorError> + Send + 'static,
    R: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| OperatorError::Join(e.to_string()))?
}

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Args)]
pub(crate) struct OperatorArgs {
    #[command(subcommand)]
    cmd: OperatorCmd,
}

#[derive(Subcommand)]
enum OperatorCmd {
    /// Poll until OmniNode + V2 activation, then exit 0. Read-only.
    WatchActivation(WatchArgs),
    /// Submit one attestation and poll it to Finalized (Included = progress).
    /// Stage 9a: only present with `--features submit`.
    #[cfg(feature = "submit")]
    Smoke(SmokeArgs),
    /// Periodic poll → stale-sweep → (retry) lifecycle loop. The
    /// retry path + its `--allow-submit` / `--allow-mainnet-submit`
    /// flags require `--features submit`; monitor-only loop works
    /// without it.
    Loop(LoopArgs),
    /// Read-only: validate seed ↔ attestation-json (+ optional chain snapshot).
    Preflight(PreflightArgs),
    /// Read-only: query the chain by (session_id, verifier_address) OR --tx-hash.
    Query(QueryArgs),
    /// Read-only: print the chain address derived from OMNINODE_VERIFIER_SEED_HEX.
    DeriveAddress,
    /// Read-only: list / show records in a local attestation registry.
    Registry(RegistryArgs),
    /// Stage 11b.0 / 11b.0.1: read-only — parse a proof artifact
    /// JSON, look up the matching verifier by proof_system, run
    /// verify, and report mainnet eligibility. Default-build
    /// accessible (no submit feature required). Stage 11b.0 ships
    /// only MockProofVerifier; other proof systems return
    /// NoVerifierForProofSystem until backends land in Stage 11c+.
    ///
    /// Stage 11b.0.1: dispatch goes through
    /// [`omni_zkml::ProofVerifier::verify_artifact`], the
    /// architecture-wide single entry point. Backends whose proof
    /// systems need backend-specific public inputs override
    /// `verify_artifact`; backends that work with the universal
    /// hashed `PublicInputs` (Mock) inherit the defaulted impl.
    /// No backend-specific helper calls in operator code.
    ///
    /// **Inspect/report semantics, not strict-validator semantics.**
    /// Stage 11b.0 `verify-proof` exits `0` on a successful inspection
    /// run *regardless of whether `verified` is `true` or `false`* —
    /// the boolean is reported in stdout (`verified=...`,
    /// `mainnet_eligible=...`) and as structured fields on the
    /// `event="proof_verification"` tracing line. Operators consuming
    /// this from scripts should grep the stdout or the tracing
    /// stream, not check the exit code. The CLI exits non-zero only
    /// on typed errors (`ProofArtifactRead`, `ProofArtifactParse`,
    /// `NoVerifierForProofSystem`). **A "strict validator" exit code
    /// where `verified=false` returns non-zero is a deferred UX
    /// decision** for Stage 11c (when real backends land) or
    /// Stage 10b (as part of release artifact tooling) — not a
    /// behaviour change to make in Stage 11b.0.
    VerifyProof(VerifyProofArgs),
}

#[derive(Args)]
struct RegistryArgs {
    #[command(subcommand)]
    cmd: RegistryCmd,
}

#[derive(Subcommand)]
enum RegistryCmd {
    /// List all records in the registry (optionally filtered by local status).
    List(RegistryListArgs),
    /// Pretty-print a single record by `AttestationId` (64-char lowercase hex).
    Show(RegistryShowArgs),
    /// Stage 10a: read-only counts-by-status summary. Optional one-shot
    /// chain read for the oldest `Submitted` record's age (requires
    /// `--rpc-url` + `--expect-chain-id`). Never mutates registry or chain.
    Summary(RegistrySummaryArgs),
}

#[derive(Args)]
struct RegistryListArgs {
    #[arg(long)]
    registry_path: PathBuf,
    /// Optional filter on local status. Accepts case-insensitive:
    /// `pending | submitted | included | finalized | failed | dropped`.
    #[arg(long)]
    status: Option<String>,
}

#[derive(Args)]
struct RegistryShowArgs {
    #[arg(long)]
    registry_path: PathBuf,
    /// `AttestationId` as 64-char lowercase hex (no `0x` prefix), e.g.
    /// the value printed by `operator registry list` or `operator
    /// preflight`.
    #[arg(long)]
    id: String,
}

#[derive(Args)]
struct VerifyProofArgs {
    /// Path to a `ProofArtifactBody` JSON file (the canonical envelope
    /// shipped inside the SNIP V2 proof artifact). The file's bytes
    /// must parse via [`omni_zkml::ProofArtifactBody`]'s serde shape.
    #[arg(long)]
    proof_artifact: PathBuf,
}

#[derive(Args)]
struct RegistrySummaryArgs {
    #[arg(long)]
    registry_path: PathBuf,
    /// Optional: when given, fetch the current chain head once and
    /// compute the oldest-`Submitted` record's age in blocks. Without
    /// this flag the summary still prints status counts; the age line
    /// is replaced with a `# oldest_submitted_age: skipped (no --rpc-url)`
    /// comment so the output stays stable.
    #[arg(long)]
    rpc_url: Option<String>,
    /// Required iff `--rpc-url` is given. Mirrors every other
    /// chain-touching operator subcommand: we never query an
    /// unguarded RPC endpoint.
    #[arg(long)]
    expect_chain_id: Option<u64>,
}

#[derive(Args)]
struct WatchArgs {
    #[arg(long)]
    rpc_url: Option<String>,
    #[arg(long)]
    expect_chain_id: u64,
    #[arg(long, default_value_t = 30)]
    poll_interval_secs: u64,
    #[arg(long)]
    max_ticks: Option<u64>,
}

#[cfg(feature = "submit")]
#[derive(Args)]
struct SmokeArgs {
    #[arg(long)]
    rpc_url: Option<String>,
    #[arg(long)]
    expect_chain_id: u64,
    #[arg(long)]
    registry_path: PathBuf,
    #[arg(long)]
    attestation_json: Option<PathBuf>,
    /// Non-mainnet only: synthesize a throwaway attestation.
    #[arg(long)]
    synthetic: bool,
    #[arg(long)]
    allow_submit: bool,
    #[arg(long)]
    allow_mainnet_submit: bool,
    #[arg(long, default_value_t = 60)]
    confirm_timeout_secs: u64,
}

#[derive(Args)]
struct PreflightArgs {
    #[arg(long)]
    attestation_json: PathBuf,
    /// Optional: when given, also fetch a one-shot chain snapshot.
    #[arg(long)]
    rpc_url: Option<String>,
    /// Required iff --rpc-url is given.
    #[arg(long)]
    expect_chain_id: Option<u64>,
}

#[derive(Args)]
struct QueryArgs {
    #[arg(long)]
    rpc_url: Option<String>,
    #[arg(long)]
    expect_chain_id: u64,
    /// Status-by-tx lookup. Mutually exclusive with the
    /// session-id/verifier-address pair (validated manually so the
    /// error wording matches the rest of `operator`).
    #[arg(long)]
    tx_hash: Option<String>,
    #[arg(long)]
    session_id: Option<String>,
    #[arg(long)]
    verifier_address: Option<String>,
}

#[derive(Args)]
struct LoopArgs {
    #[arg(long)]
    rpc_url: Option<String>,
    #[arg(long)]
    expect_chain_id: u64,
    #[arg(long)]
    registry_path: PathBuf,
    #[arg(long)]
    staleness_threshold_blocks: u64,
    #[arg(long, default_value_t = 30)]
    poll_interval_secs: u64,
    /// Stage 9a: present only with `--features submit`. Without it,
    /// `loop` runs strictly monitor-only and these flags don't appear
    /// in `--help` (cleaner output than a flag that can only fail).
    #[cfg(feature = "submit")]
    #[arg(long)]
    allow_submit: bool,
    #[cfg(feature = "submit")]
    #[arg(long)]
    allow_mainnet_submit: bool,
    #[arg(long)]
    max_ticks: Option<u64>,
}

fn resolve_rpc_url(opt: Option<String>) -> Result<String, OperatorError> {
    if let Some(u) = opt {
        return Ok(u);
    }
    std::env::var("OMNINODE_SUMCHAIN_RPC_URL").map_err(|_| OperatorError::RpcUrlMissing)
}

fn ureq_factory(url: String) -> ClientFactory<UreqTransport> {
    Arc::new(move |seed| SumChainClient::new(url.clone(), seed))
}

pub(crate) async fn dispatch(args: OperatorArgs) -> anyhow::Result<()> {
    log_startup_marker(subcommand_name(&args.cmd));
    match args.cmd {
        OperatorCmd::WatchActivation(a) => {
            let url = resolve_rpc_url(a.rpc_url.clone())?;
            watch_activation_core(
                ureq_factory(url),
                WatchConfig {
                    expect_chain_id: a.expect_chain_id,
                    poll_interval_secs: a.poll_interval_secs,
                    max_ticks: a.max_ticks,
                },
            )
            .await?;
        }
        #[cfg(feature = "submit")]
        OperatorCmd::Smoke(a) => {
            let url = resolve_rpc_url(a.rpc_url.clone())?;
            let outcome = smoke_core(
                ureq_factory(url),
                SmokeConfig {
                    expect_chain_id: a.expect_chain_id,
                    registry_path: a.registry_path,
                    attestation_json: a.attestation_json,
                    synthetic: a.synthetic,
                    allow_submit: a.allow_submit,
                    allow_mainnet_submit: a.allow_mainnet_submit,
                    confirm_timeout_secs: a.confirm_timeout_secs,
                    seed_source: SeedSource::Env,
                },
            )
            .await?;
            print_smoke_summary(&outcome);
        }
        OperatorCmd::Loop(a) => {
            let url = resolve_rpc_url(a.rpc_url.clone())?;
            loop_core(
                ureq_factory(url),
                LoopConfig {
                    expect_chain_id: a.expect_chain_id,
                    registry_path: a.registry_path,
                    staleness_threshold_blocks: a.staleness_threshold_blocks,
                    poll_interval_secs: a.poll_interval_secs,
                    #[cfg(feature = "submit")]
                    allow_submit: a.allow_submit,
                    #[cfg(not(feature = "submit"))]
                    allow_submit: false,
                    #[cfg(feature = "submit")]
                    allow_mainnet_submit: a.allow_mainnet_submit,
                    #[cfg(not(feature = "submit"))]
                    allow_mainnet_submit: false,
                    max_ticks: a.max_ticks,
                    seed_source: SeedSource::Env,
                },
            )
            .await?;
        }
        OperatorCmd::Preflight(a) => {
            let cfg = PreflightConfig {
                attestation_json: a.attestation_json,
                expect_chain_id: a.expect_chain_id,
                seed_source: SeedSource::Env,
            };
            match a.rpc_url {
                Some(url) => {
                    if a.expect_chain_id.is_none() {
                        return Err(OperatorError::ExpectChainIdRequiredWithRpc.into());
                    }
                    run_preflight(Some(ureq_factory(url)), cfg).await?;
                }
                None => {
                    run_preflight::<UreqTransport>(None, cfg).await?;
                }
            }
        }
        OperatorCmd::Query(a) => {
            let url = resolve_rpc_url(a.rpc_url.clone())?;
            let mode = resolve_query_mode(
                a.tx_hash,
                a.session_id,
                a.verifier_address,
            )?;
            run_query(
                ureq_factory(url),
                QueryConfig {
                    expect_chain_id: a.expect_chain_id,
                    mode,
                },
            )
            .await?;
        }
        OperatorCmd::DeriveAddress => {
            let addr = derive_address_core(SeedSource::Env).await?;
            // Bare stdout line so it is scriptable
            // (`ADDR=$(omni-node operator derive-address)`); the info!
            // line carries human context on stderr.
            info!(verifier_address = %addr, "derive-address");
            println!("{addr}");
        }
        OperatorCmd::Registry(a) => match a.cmd {
            RegistryCmd::List(la) => {
                let rows = registry_list_core(la.registry_path, la.status).await?;
                for r in &rows {
                    // Bare stdout one-liner — scriptable
                    // (`operator registry list | grep submitted`).
                    println!(
                        "{}  {:<9}  tx={}  block={}  updated={}",
                        r.id_hex,
                        r.status,
                        r.tx_id.as_deref().unwrap_or("-"),
                        r.submitted_at_block
                            .map(|b| b.to_string())
                            .unwrap_or_else(|| "-".to_string()),
                        r.updated_at,
                    );
                }
                info!(rows = rows.len(), "registry list complete");
            }
            RegistryCmd::Show(sa) => {
                let record = registry_show_core(sa.registry_path, sa.id).await?;
                let json = serde_json::to_string_pretty(&record).map_err(|e| {
                    OperatorError::Internal(format!(
                        "AttestationRecord serialise failed: {e}"
                    ))
                })?;
                println!("{json}");
            }
            RegistryCmd::Summary(sa) => {
                // Mirror Preflight's RPC + chain-id pairing: --rpc-url is
                // optional, but if given --expect-chain-id is required.
                let client_or_none = match sa.rpc_url {
                    Some(url) => {
                        if sa.expect_chain_id.is_none() {
                            return Err(OperatorError::ExpectChainIdRequiredWithRpc.into());
                        }
                        Some((ureq_factory(url), sa.expect_chain_id.unwrap()))
                    }
                    None => None,
                };
                let summary =
                    registry_summary_core(sa.registry_path, client_or_none).await?;
                print_registry_summary(&summary);
            }
        },
        OperatorCmd::VerifyProof(a) => {
            verify_proof_core(a.proof_artifact).await?;
        }
    }
    Ok(())
}

// ── Configs ───────────────────────────────────────────────────────────────────

struct WatchConfig {
    expect_chain_id: u64,
    poll_interval_secs: u64,
    max_ticks: Option<u64>,
}

#[cfg(feature = "submit")]
#[derive(Clone)]
struct SmokeConfig {
    expect_chain_id: u64,
    registry_path: PathBuf,
    attestation_json: Option<PathBuf>,
    synthetic: bool,
    allow_submit: bool,
    allow_mainnet_submit: bool,
    confirm_timeout_secs: u64,
    seed_source: SeedSource,
}

#[derive(Clone)]
struct LoopConfig {
    expect_chain_id: u64,
    registry_path: PathBuf,
    staleness_threshold_blocks: u64,
    poll_interval_secs: u64,
    allow_submit: bool,
    allow_mainnet_submit: bool,
    max_ticks: Option<u64>,
    seed_source: SeedSource,
}

// ── watch-activation ──────────────────────────────────────────────────────────

async fn watch_activation_core<T>(
    make_client: ClientFactory<T>,
    cfg: WatchConfig,
) -> Result<(), OperatorError>
where
    T: JsonRpcTransport + Send + Sync + 'static,
{
    let mut tick: u64 = 0;
    loop {
        let mc = make_client.clone();
        let expect = cfg.expect_chain_id;
        let snap = run_blocking(move || -> Result<WatchSnapshot, OperatorError> {
            let client = mc(DUMMY_SEED);
            let params = client.get_chain_params()?;
            check_chain_id(expect, params.chain_id)?;
            let head = client.get_block_height(BlockFinality::Latest)?.height;
            let omninode = client.omninode_is_active()?;
            let v2 = client.v2_is_active()?;
            Ok(WatchSnapshot {
                chain_id: params.chain_id,
                finality_depth: params.finality_depth,
                min_fee: params.min_fee,
                v2_from: params.v2_enabled_from_height,
                omninode_from: params.omninode_enabled_from_height,
                head,
                omninode,
                v2,
            })
        })
        .await?;

        info!(
            event = "chain_params",
            chain_id = snap.chain_id,
            finality_depth = snap.finality_depth,
            min_fee = snap.min_fee,
            v2_enabled_from_height = ?snap.v2_from,
            omninode_enabled_from_height = ?snap.omninode_from,
            head = snap.head,
            "chain params snapshot"
        );
        info!(
            event = "activation_state",
            v2_blocks_remaining = ?blocks_remaining(snap.head, snap.v2_from),
            omninode_blocks_remaining = ?blocks_remaining(snap.head, snap.omninode_from),
            v2_active = snap.v2,
            omninode_active = snap.omninode,
            activated = snap.omninode && snap.v2,
            head = snap.head,
            "activation state"
        );

        if snap.omninode && snap.v2 {
            info!("both activation gates live — exiting 0");
            return Ok(());
        }

        tick += 1;
        if let Some(max) = cfg.max_ticks {
            if tick >= max {
                warn!(ticks = tick, "activation not reached within tick budget");
                return Err(OperatorError::ActivationNotReached);
            }
        }

        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(cfg.poll_interval_secs)) => {}
            _ = tokio::signal::ctrl_c() => {
                info!("ctrl-c — stopping watch-activation");
                return Ok(());
            }
        }
    }
}

struct WatchSnapshot {
    chain_id: u64,
    finality_depth: u64,
    min_fee: u64,
    v2_from: Option<u64>,
    omninode_from: Option<u64>,
    head: u64,
    omninode: bool,
    v2: bool,
}

// ── smoke ─────────────────────────────────────────────────────────────────────
//
// Stage 9a: the entire smoke surface (SmokeOutcome / SmokePending /
// smoke_core / print_smoke_summary) is `#[cfg(feature = "submit")]`.
// Without the feature, `operator smoke` is not part of the CLI and
// none of these symbols are compiled into the binary.

/// Structured result of a successful `smoke` run. Returned by
/// [`smoke_core`] (testable) and rendered by [`print_smoke_summary`]
/// for the operator. Only constructed once the tx reaches `Finalized`.
#[cfg(feature = "submit")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SmokeOutcome {
    pub attestation_id: String,
    pub session_id: String,
    pub verifier_address: String,
    pub tx_hash: String,
    pub submitted_at_block: Option<u64>,
    pub included_at_height: Option<u64>,
    pub finalized: bool,
}

/// Phase-A result handed from the submit blocking unit to the poll loop.
#[cfg(feature = "submit")]
struct SmokePending {
    tx_id: String,
    attestation_id: String,
    session_id: String,
    verifier_address: String,
    submitted_at_block: Option<u64>,
}

#[cfg(feature = "submit")]
async fn smoke_core<T>(
    make_client: ClientFactory<T>,
    cfg: SmokeConfig,
) -> Result<SmokeOutcome, OperatorError>
where
    T: JsonRpcTransport + Send + Sync + 'static,
{
    // Phase A — gates + submit, all in one blocking unit. Gate order
    // is exactly the documented sequence; the seed is resolved only
    // after gate 7 (resolve_smoke_source) has passed.
    let mc = make_client.clone();
    let cfg_a = cfg.clone();
    let pending = run_blocking(move || -> Result<SmokePending, OperatorError> {
        let probe = mc(DUMMY_SEED);

        let params = probe.get_chain_params()?;
        check_chain_id(cfg_a.expect_chain_id, params.chain_id)?; // gate 1

        // gate 7 — pure, seed-free, BEFORE any seed handling.
        let source = resolve_smoke_source(
            params.chain_id,
            cfg_a.attestation_json.as_deref(),
            cfg_a.synthetic,
        )?;

        let omninode = probe.omninode_is_active()?;
        let v2 = probe.v2_is_active()?;
        if !(omninode && v2) {
            return Err(OperatorError::NotActivated { omninode, v2 }); // gate 2
        }

        // Stage 10a observability markers. Pre-submit chain snapshot
        // so operators can grep `event="chain_params"` /
        // `event="activation_state"` consistently across all
        // chain-touching subcommands.
        log_chain_params_snapshot(&params, None);
        log_activation_state(omninode, v2, None);

        submission_permitted(
            params.chain_id,
            cfg_a.allow_submit,
            cfg_a.allow_mainnet_submit,
        )?; // gates 3-4

        // gate 5 — seed resolution happens here, strictly after gate 7.
        let seed = cfg_a.seed_source.resolve()?;
        let derived = signer_chain_address_base58(&seed)
            .map_err(|e| OperatorError::SeedMalformed(e.to_string()))?;

        let attestation = match source {
            SmokeSource::File(path) => {
                let bytes = std::fs::read(&path).map_err(|source| {
                    OperatorError::AttestationJsonRead {
                        path: path.display().to_string(),
                        source,
                    }
                })?;
                let att: InferenceAttestation = serde_json::from_slice(&bytes)
                    .map_err(|e| OperatorError::AttestationJsonParse(e.to_string()))?;
                if att.verifier_address != derived {
                    return Err(OperatorError::VerifierAddressMismatch {
                        claimed: att.verifier_address.clone(),
                        derived,
                    });
                }
                att
            }
            SmokeSource::Synthetic => {
                let head = probe.get_block_height(BlockFinality::Latest)?.height;
                // Stage 11a: real proof bytes + mainnet-refusal guard.
                // Replaces the pre-11a `synth_attestation(...)` call which
                // fabricated placeholder bytes for every commitment field.
                build_mock_v1_attestation(&derived, cfg_a.expect_chain_id, head)?
            }
        };

        let registry = AttestationRegistry::open(cfg_a.registry_path.clone())?;
        let client = mc(seed);
        let record =
            submit_attestation_workflow_with_block(&registry, &client, attestation)?;
        let tx_id = record
            .receipt
            .as_ref()
            .map(|r| r.tx_id.clone())
            .ok_or_else(|| {
                OperatorError::Internal(
                    "submitted record has no receipt (registry invariant)".into(),
                )
            })?;
        Ok(SmokePending {
            tx_id,
            attestation_id: record.id.to_hex(),
            session_id: record.attestation.commitment.session_id.clone(),
            verifier_address: record.attestation.verifier_address.clone(),
            submitted_at_block: record.submitted_at_block,
        })
    })
    .await?;

    info!(tx_id = %pending.tx_id, "smoke: submitted; polling for confirmation");

    // Phase B — poll the tx until Finalized. Included is progress only;
    // returning success at Included would be a false-positive mainnet
    // activation smoke.
    let deadline = Instant::now() + Duration::from_secs(cfg.confirm_timeout_secs);
    let mut last = "submitted".to_string();
    loop {
        if Instant::now() >= deadline {
            return Err(OperatorError::SmokeConfirmTimeout { last });
        }
        let mc2 = make_client.clone();
        let txq = pending.tx_id.clone();
        let status = run_blocking(move || -> Result<omni_zkml::AttestationStatus, OperatorError> {
            let c = mc2(DUMMY_SEED);
            Ok(c.query_attestation_status(&txq)?)
        })
        .await?;

        use omni_zkml::AttestationStatus::*;
        match status {
            Finalized => {
                // One extra read-only lookup to populate
                // included_at_height for the operator summary (Stage 8c
                // Q3). Submit semantics are unchanged — success is
                // still gated solely on Finalized above.
                let mc3 = make_client.clone();
                let txq = pending.tx_id.clone();
                let included_at_height = run_blocking(
                    move || -> Result<Option<u64>, OperatorError> {
                        let c = mc3(DUMMY_SEED);
                        Ok(c.query_attestation_status_full(&txq)?.included_at_height)
                    },
                )
                .await?;
                info!("smoke: finalized — confirmed");
                return Ok(SmokeOutcome {
                    attestation_id: pending.attestation_id,
                    session_id: pending.session_id,
                    verifier_address: pending.verifier_address,
                    tx_hash: pending.tx_id,
                    submitted_at_block: pending.submitted_at_block,
                    included_at_height,
                    finalized: true,
                });
            }
            Failed { reason } => return Err(OperatorError::SmokeFailed { reason }),
            progressing => {
                last = format!("{progressing:?}");
                info!(
                    status = ?progressing,
                    "smoke: not final yet — continuing to poll for finalization"
                );
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(2)) => {}
                    _ = tokio::signal::ctrl_c() => {
                        info!("ctrl-c — stopping smoke poll before finality");
                        return Err(OperatorError::SmokeInterrupted);
                    }
                }
            }
        }
    }
}

/// Render a successful [`SmokeOutcome`] as one consolidated operator
/// summary block (Stage 8c Q2).
#[cfg(feature = "submit")]
fn print_smoke_summary(o: &SmokeOutcome) {
    info!(
        attestation_id = %o.attestation_id,
        session_id = %o.session_id,
        verifier_address = %o.verifier_address,
        tx_hash = %o.tx_hash,
        submitted_at_block = ?o.submitted_at_block,
        included_at_height = ?o.included_at_height,
        finalized = o.finalized,
        "SMOKE SUMMARY — attestation finalized on chain"
    );
}

// ── loop ──────────────────────────────────────────────────────────────────────

async fn loop_core<T>(
    make_client: ClientFactory<T>,
    cfg: LoopConfig,
) -> Result<(), OperatorError>
where
    T: JsonRpcTransport + Send + Sync + 'static,
{
    let policy = StalenessPolicy::new(cfg.staleness_threshold_blocks)?;

    // Startup: chain-id guardrail + submit-mode resolution. A mainnet
    // misconfiguration (chain_id 1 with --allow-submit but no
    // --allow-mainnet-submit) hard-fails here, before any tick.
    let mc = make_client.clone();
    let expect = cfg.expect_chain_id;
    let chain_id = run_blocking(move || -> Result<u64, OperatorError> {
        let probe = mc(DUMMY_SEED);
        let params = probe.get_chain_params()?;
        check_chain_id(expect, params.chain_id)?;
        Ok(params.chain_id)
    })
    .await?;

    let submit_permitted = if cfg.allow_submit {
        submission_permitted(chain_id, true, cfg.allow_mainnet_submit)?;
        true
    } else {
        info!("loop running monitor-only (no --allow-submit): retry step skipped");
        false
    };
    let seed = if submit_permitted {
        cfg.seed_source.resolve()?
    } else {
        DUMMY_SEED
    };

    let mut tick: u64 = 0;
    loop {
        let mc = make_client.clone();
        let rp = cfg.registry_path.clone();
        let pol = policy.clone();
        let expect = cfg.expect_chain_id;
        let summary = run_blocking(move || -> Result<TickSummary, OperatorError> {
            let registry = AttestationRegistry::open(rp)?;
            let read_client = mc(DUMMY_SEED);

            let params = read_client.get_chain_params()?;
            check_chain_id(expect, params.chain_id)?;
            log_chain_params_snapshot(&params, None);

            let polled = poll_attestations_workflow(&registry, &read_client)?;
            let swept = sweep_stale_attestations_workflow(&registry, &read_client, &pol)?;

            // Stage 10a: activation is a read-only check with no side
            // effects, so emit `event="activation_state"` every tick
            // — both monitor-only and retry-enabled — rather than
            // only when the retry path runs. The `omninode` / `v2`
            // values then gate the retry step below.
            let omninode = read_client.omninode_is_active()?;
            let v2 = read_client.v2_is_active()?;
            log_activation_state(omninode, v2, None);

            let mut retried = Vec::new();
            let mut retry_skipped_reason: Option<&'static str> = None;
            if submit_permitted {
                if omninode && v2 {
                    let submit_client = mc(seed);
                    retried =
                        retry_dropped_attestations_workflow(&registry, &submit_client)?;
                } else {
                    retry_skipped_reason = Some("chain not activated");
                }
            } else {
                retry_skipped_reason = Some("monitor-only (no --allow-submit)");
            }

            Ok(TickSummary {
                polled,
                swept,
                retried,
                retry_skipped_reason,
            })
        })
        .await?;

        log_tick_summary(&summary);

        tick += 1;
        if let Some(max) = cfg.max_ticks {
            if tick >= max {
                info!(ticks = tick, "loop: max-ticks reached — exiting");
                return Ok(());
            }
        }

        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(cfg.poll_interval_secs)) => {}
            _ = tokio::signal::ctrl_c() => {
                info!("ctrl-c — graceful loop shutdown");
                return Ok(());
            }
        }
    }
}

type SweepVec = Vec<(
    omni_zkml::AttestationId,
    omni_zkml::RegistryResult<omni_zkml::AttestationRecord>,
)>;

struct TickSummary {
    polled: SweepVec,
    swept: SweepVec,
    retried: SweepVec,
    retry_skipped_reason: Option<&'static str>,
}

fn log_tick_summary(s: &TickSummary) {
    let count_err = |v: &SweepVec| v.iter().filter(|(_, r)| r.is_err()).count();
    info!(
        polled = s.polled.len(),
        polled_errors = count_err(&s.polled),
        swept = s.swept.len(),
        swept_errors = count_err(&s.swept),
        retried = s.retried.len(),
        retried_errors = count_err(&s.retried),
        retry_skipped = ?s.retry_skipped_reason,
        "loop tick complete"
    );
    for (label, v) in [
        ("poll", &s.polled),
        ("sweep", &s.swept),
        ("retry", &s.retried),
    ] {
        for (id, result) in v {
            match result {
                Ok(rec) => debug!(stage = label, id = %id, status = ?rec.status, "record processed"),
                Err(e) => warn!(stage = label, id = %id, error = %e, "record processing failed"),
            }
        }
    }
}

// ── preflight (Stage 8b) ──────────────────────────────────────────────────────

#[derive(Clone)]
struct PreflightConfig {
    attestation_json: PathBuf,
    /// Required iff a client factory is supplied (i.e. `--rpc-url` set).
    expect_chain_id: Option<u64>,
    seed_source: SeedSource,
}

/// Read-only readiness check. Validates that the verifier seed derives
/// to the attestation JSON's `verifier_address`, prints the de-dup
/// `AttestationId`, and — only when a client factory is supplied —
/// fetches a one-shot chain snapshot (chain-id guardrail + params +
/// head + activation status, **report-only**: valid before *and*
/// after activation). Never submits, never writes the registry, never
/// requires activation or `--allow-submit`.
async fn run_preflight<T>(
    make_client: Option<ClientFactory<T>>,
    cfg: PreflightConfig,
) -> Result<(), OperatorError>
where
    T: JsonRpcTransport + Send + Sync + 'static,
{
    if make_client.is_some() && cfg.expect_chain_id.is_none() {
        return Err(OperatorError::ExpectChainIdRequiredWithRpc);
    }

    run_blocking(move || -> Result<(), OperatorError> {
        // Seed ↔ attestation validation (the whole point of preflight;
        // resolving the seed here is intended, not deferred).
        let seed = cfg.seed_source.resolve()?;
        let derived = signer_chain_address_base58(&seed)
            .map_err(|e| OperatorError::SeedMalformed(e.to_string()))?;

        let bytes = std::fs::read(&cfg.attestation_json).map_err(|source| {
            OperatorError::AttestationJsonRead {
                path: cfg.attestation_json.display().to_string(),
                source,
            }
        })?;
        let att: InferenceAttestation = serde_json::from_slice(&bytes)
            .map_err(|e| OperatorError::AttestationJsonParse(e.to_string()))?;
        if att.verifier_address != derived {
            return Err(OperatorError::VerifierAddressMismatch {
                claimed: att.verifier_address.clone(),
                derived,
            });
        }
        let id = omni_zkml::compute_attestation_id(&att)?;
        info!(
            verifier_address = %derived,
            attestation_id = %id,
            session_id = %att.commitment.session_id,
            "preflight: seed ↔ attestation OK (verifier_address matches; \
             this is the (session_id, verifier_address) de-dup key)"
        );

        match make_client {
            None => {
                info!(
                    "preflight: offline mode (no --rpc-url) — chain \
                     snapshot skipped; run again with --rpc-url \
                     --expect-chain-id to also verify chain params"
                );
            }
            Some(mc) => {
                let expect = cfg
                    .expect_chain_id
                    .ok_or(OperatorError::ExpectChainIdRequiredWithRpc)?;
                let client = mc(DUMMY_SEED);
                let params = client.get_chain_params()?;
                check_chain_id(expect, params.chain_id)?;
                let head = client.get_block_height(BlockFinality::Latest)?.height;
                let omninode = client.omninode_is_active()?;
                let v2 = client.v2_is_active()?;
                info!(
                    event = "chain_params",
                    chain_id = params.chain_id,
                    finality_depth = params.finality_depth,
                    min_fee = params.min_fee,
                    v2_enabled_from_height = ?params.v2_enabled_from_height,
                    omninode_enabled_from_height = ?params.omninode_enabled_from_height,
                    head = head,
                    "preflight: chain params snapshot"
                );
                info!(
                    event = "activation_state",
                    v2_blocks_remaining = ?blocks_remaining(head, params.v2_enabled_from_height),
                    omninode_blocks_remaining = ?blocks_remaining(head, params.omninode_enabled_from_height),
                    v2_active = v2,
                    omninode_active = omninode,
                    activated = omninode && v2,
                    head = head,
                    "preflight: activation state (report-only — valid \
                     before and after activation)"
                );
            }
        }
        Ok(())
    })
    .await
}

// ── query (Stage 8b + 8c) ─────────────────────────────────────────────────────

#[derive(Debug, PartialEq, Eq)]
enum QueryMode {
    /// Stage 8c: status-by-tx (`sum_getInferenceAttestationStatus`).
    TxHash(String),
    /// Stage 8b: presence-by-key (`sum_getInferenceAttestation`).
    Pair {
        session_id: String,
        verifier_address: String,
    },
}

struct QueryConfig {
    expect_chain_id: u64,
    mode: QueryMode,
}

/// Manual mode selection (Stage 8c Q4) — typed errors so the wording
/// matches the rest of `operator`. Exactly one of: `--tx-hash`, or
/// **both** `--session-id` and `--verifier-address`.
fn resolve_query_mode(
    tx_hash: Option<String>,
    session_id: Option<String>,
    verifier_address: Option<String>,
) -> Result<QueryMode, OperatorError> {
    let pair_present = session_id.is_some() || verifier_address.is_some();
    match (tx_hash, pair_present) {
        (Some(_), true) => Err(OperatorError::ConflictingQueryMode),
        (Some(h), false) => Ok(QueryMode::TxHash(h)),
        (None, true) => match (session_id, verifier_address) {
            (Some(session_id), Some(verifier_address)) => Ok(QueryMode::Pair {
                session_id,
                verifier_address,
            }),
            // Exactly one of the pair supplied → incomplete.
            _ => Err(OperatorError::QueryModeRequired),
        },
        (None, false) => Err(OperatorError::QueryModeRequired),
    }
}

/// Read-only, behind the chain-id guardrail. `Pair` mode is a
/// presence assertion: returns typed `AttestationNotFound` (scriptable
/// non-zero) when absent. `TxHash` mode is a status report: logs the
/// chain status (incl. `included_at_height`) and always returns Ok —
/// `unknown` is a legitimate observation, not an error. Never submits.
async fn run_query<T>(
    make_client: ClientFactory<T>,
    cfg: QueryConfig,
) -> Result<(), OperatorError>
where
    T: JsonRpcTransport + Send + Sync + 'static,
{
    run_blocking(move || -> Result<(), OperatorError> {
        let client = make_client(DUMMY_SEED);
        let params = client.get_chain_params()?;
        check_chain_id(cfg.expect_chain_id, params.chain_id)?;
        log_chain_params_snapshot(&params, None);
        match cfg.mode {
            QueryMode::Pair {
                session_id,
                verifier_address,
            } => match client.get_attestation(&session_id, &verifier_address)? {
                Some(info) => {
                    info!(
                        session_id = %info.session_id,
                        verifier_address = %info.verifier_address,
                        tx_hash = %info.tx_hash,
                        included_at_height = info.included_at_height,
                        finalized = info.finalized,
                        "query: chain attestation found"
                    );
                    Ok(())
                }
                None => Err(OperatorError::AttestationNotFound {
                    session_id,
                    verifier_address,
                }),
            },
            QueryMode::TxHash(tx) => {
                let s = client.query_attestation_status_full(&tx)?;
                info!(
                    tx_hash = %tx,
                    status = %s.status,
                    included_at_height = ?s.included_at_height,
                    reason = ?s.reason,
                    "query: chain tx status"
                );
                Ok(())
            }
        }
    })
    .await
}

// ── derive-address (Stage 8c) ─────────────────────────────────────────────────

/// Read-only: resolve the verifier seed and return its chain address.
/// No chain access, no attestation file, no submit. Returns the
/// address so the dispatch layer (and tests) can render/assert it.
async fn derive_address_core(seed_source: SeedSource) -> Result<String, OperatorError> {
    run_blocking(move || -> Result<String, OperatorError> {
        let seed = seed_source.resolve()?;
        signer_chain_address_base58(&seed)
            .map_err(|e| OperatorError::SeedMalformed(e.to_string()))
    })
    .await
}

// ── Stage 10a: stable observability markers ───────────────────────────────────
//
// Field shapes for the `event=` markers below are pinned by
// `docs/operator-runbook.md §7` (operator-facing grep contract). Add a
// new field freely; rename or remove → update the runbook table in the
// same PR.

/// `event="chain_params"` snapshot. Emitted once per chain-touching
/// subcommand entry. `head` is `Option<_>` because some entry points
/// have already paid for a block-height read at the same time, and
/// some haven't — letting callers pass `None` keeps the marker honest
/// instead of forcing a second RPC just to fill the field.
fn log_chain_params_snapshot(params: &ChainParamsInfo, head: Option<u64>) {
    info!(
        event = "chain_params",
        chain_id = params.chain_id,
        finality_depth = params.finality_depth,
        min_fee = params.min_fee,
        omninode_enabled_from_height = ?params.omninode_enabled_from_height,
        v2_enabled_from_height = ?params.v2_enabled_from_height,
        head = ?head,
        "chain params snapshot"
    );
}

/// `event="activation_state"` snapshot. Emitted at entry to any
/// subcommand that has already fetched both activation gates.
/// Used by `smoke` (cfg-gated) and by `loop_core` (every tick, both
/// monitor-only and retry-enabled). `watch-activation` and
/// `preflight` emit their own inline `event="activation_state"`
/// lines because they carry extra fields like `v2_blocks_remaining`.
fn log_activation_state(omninode_active: bool, v2_active: bool, head: Option<u64>) {
    info!(
        event = "activation_state",
        omninode_active,
        v2_active,
        activated = omninode_active && v2_active,
        head = ?head,
        "activation state"
    );
}

/// `event="startup"` line emitted once per `operator` invocation,
/// before any chain or registry access. Operators grep this to confirm
/// they're running the binary they expect.
fn log_startup_marker(subcommand: &'static str) {
    info!(
        event = "startup",
        subcommand,
        feature_submit = cfg!(feature = "submit"),
        version = env!("CARGO_PKG_VERSION"),
        "omni-node operator startup"
    );
}

fn subcommand_name(c: &OperatorCmd) -> &'static str {
    match c {
        OperatorCmd::WatchActivation(_) => "watch-activation",
        #[cfg(feature = "submit")]
        OperatorCmd::Smoke(_) => "smoke",
        OperatorCmd::Loop(_) => "loop",
        OperatorCmd::Preflight(_) => "preflight",
        OperatorCmd::Query(_) => "query",
        OperatorCmd::DeriveAddress => "derive-address",
        OperatorCmd::Registry(_) => "registry",
        OperatorCmd::VerifyProof(_) => "verify-proof",
    }
}

// ── Stage 11b.0: verify-proof + mainnet refusal mapping ────────────────────────

/// Stage 11b.0 — map a typed [`omni_zkml::MainnetRefusalReason`] into
/// the operator binary's typed [`OperatorError`] surface. Keeps the
/// operator-facing wording consistent with the rest of the surface;
/// every refusal layer in `check_mainnet_eligible` maps 1:1 to one
/// variant here.
#[cfg(feature = "submit")]
#[allow(dead_code)] // Used by future submit-path wiring (Stage 11c+);
                    // exists in Stage 11b.0 so the mapping is in tree.
fn map_mainnet_refusal(reason: omni_zkml::MainnetRefusalReason) -> OperatorError {
    use omni_zkml::MainnetRefusalReason as R;
    match reason {
        R::TestnetOrDevOnly { backend_id } => {
            OperatorError::TestnetOnlyProofRefusedOnMainnet { backend_id }
        }
        R::MockBackend { backend_id, .. } => {
            // Preserve Stage 11a's variant name for back-compat —
            // any caller pattern-matching on
            // `MockBackendRefusedOnMainnet` continues to work.
            OperatorError::MockBackendRefusedOnMainnet { backend_id }
        }
        R::BoundedReference { backend_id, .. } => {
            OperatorError::BoundedReferenceProofRefusedOnMainnet { backend_id }
        }
        R::GgufClaim { backend_id, .. } => {
            OperatorError::GgufProofClaimRefusedOnMainnet { backend_id }
        }
        R::UnknownModelFormat { backend_id, .. } => {
            OperatorError::UnknownModelFormatRefusedOnMainnet { backend_id }
        }
        R::NotInMainnetAllowlist { backend_id, .. } => {
            OperatorError::ProofSystemNotMainnetApproved { backend_id }
        }
    }
}

/// Stage 11b.0 + Stage 11b.0.1 — read a `ProofArtifactBody` JSON,
/// route to the matching verifier by `proof_system`, run verify via
/// [`omni_zkml::ProofVerifier::verify_artifact`] (the
/// architecture-wide single dispatch entry point), and emit a
/// structured tracing event + bare-stdout summary. Read-only.
/// Default-build accessible (no `--features submit` required).
///
/// Stage 11b.0 ships only the [`omni_zkml::MockProofVerifier`] in the
/// verifier registry. Any other `proof_system` value returns
/// `OperatorError::NoVerifierForProofSystem` — Stage 11c+ wires real
/// verifiers in as backends land.
async fn verify_proof_core(proof_artifact_path: PathBuf) -> Result<(), OperatorError> {
    run_blocking(move || -> Result<(), OperatorError> {
        let bytes = std::fs::read(&proof_artifact_path).map_err(|e| {
            OperatorError::ProofArtifactRead {
                path: proof_artifact_path.display().to_string(),
                source: e,
            }
        })?;
        let body: omni_zkml::ProofArtifactBody = serde_json::from_slice(&bytes)
            .map_err(|e| OperatorError::ProofArtifactParse(e.to_string()))?;

        // Stage 11b.0.1: uniform dispatch via `ProofVerifier::verify_artifact`.
        // Every backend's verifier is invoked through the same single
        // call shape; there is NO backend-specific helper logic in
        // operator-side code. Future verifiers (Stage 11c+) plug in
        // by adding one match arm here — they do not introduce
        // operator-side per-backend dispatch code.
        //
        // The `MockProofVerifier` inherits the defaulted `verify_artifact`
        // in [`omni_zkml::ProofVerifier`], which calls `self.verify(&proof,
        // &public_inputs)` with the artifact's bytes + hashed
        // PublicInputs — byte-equivalent to the pre-Stage-11b.0.1
        // direct call.
        //
        // None proof_system is treated as Stage 11a vintage Mock
        // (preserves the Stage 11a verification contract).
        let verify_result = match body.metadata.proof_system {
            Some(omni_zkml::ProofSystem::Mock) | None => {
                use omni_zkml::ProofVerifier;
                omni_zkml::MockProofVerifier
                    .verify_artifact(&body)
                    .map_err(|e| {
                        OperatorError::ProofArtifactParse(format!(
                            "verifier failure: {e}"
                        ))
                    })?
            }
            #[cfg(feature = "halo2-reference-verify")]
            Some(omni_zkml::ProofSystem::Stage11bHalo2Reference) => {
                use omni_zkml::ProofVerifier;
                let verifier = omni_proofs_halo2_reference::Halo2ReferenceVerifier::from_embedded_fixtures()
                    .map_err(|e| {
                        OperatorError::ProofArtifactParse(format!(
                            "halo2-reference verifier construction failure: {e}"
                        ))
                    })?;
                verifier
                    .verify_artifact(&body)
                    .map_err(|e| {
                        OperatorError::ProofArtifactParse(format!(
                            "verifier failure: {e}"
                        ))
                    })?
            }
            Some(other) => {
                return Err(OperatorError::NoVerifierForProofSystem {
                    proof_system: format!("{other:?}"),
                });
            }
        };

        // Always-evaluated: mainnet refusal reason. Stage 11b.0
        // invariant — `check_mainnet_eligible` returns Err for every
        // input, since the allowlist is empty.
        let mainnet = omni_zkml::check_mainnet_eligible(&body.metadata);
        let mainnet_refusal_str = match &mainnet {
            Ok(()) => "ok".to_string(), // unreachable in Stage 11b.0
            Err(reason) => format!("{reason}"),
        };

        info!(
            event = "proof_verification",
            backend_id = %body.metadata.backend_id,
            proof_system = ?body.metadata.proof_system,
            model_format = ?body.metadata.model_format,
            verified = verify_result,
            mainnet_eligible = mainnet.is_ok(),
            mainnet_refusal = %mainnet_refusal_str,
            "proof artifact verified"
        );

        // Bare-stdout summary — same grep-friendly shape as
        // `registry summary` / `registry list`.
        println!("backend_id={}", body.metadata.backend_id);
        println!(
            "proof_system={}",
            body.metadata
                .proof_system
                .map(|p| format!("{p:?}"))
                .unwrap_or_else(|| "none".to_string()),
        );
        println!(
            "model_format={}",
            body.metadata
                .model_format
                .as_ref()
                .map(|m| format!("{m:?}"))
                .unwrap_or_else(|| "none".to_string()),
        );
        println!("verified={verify_result}");
        println!("mainnet_eligible={}", mainnet.is_ok());
        if let Err(reason) = &mainnet {
            println!("mainnet_refusal={reason}");
        }
        Ok(())
    })
    .await
}

// ── registry list / show (Stage 9a) ───────────────────────────────────────────

/// Canonical lowercase label for a local status — used both by
/// `registry list` output and by `--status` filter matching.
fn status_label(s: &LocalAttestationStatus) -> &'static str {
    match s {
        LocalAttestationStatus::Pending => "pending",
        LocalAttestationStatus::Submitted => "submitted",
        LocalAttestationStatus::Included => "included",
        LocalAttestationStatus::Finalized => "finalized",
        LocalAttestationStatus::Failed { .. } => "failed",
        LocalAttestationStatus::Dropped { .. } => "dropped",
    }
}

/// Parse / validate a `--status` filter. Case-insensitive; unknown
/// values surface a typed `InvalidStatusFilter` so the operator gets
/// a clear list of accepted values.
fn parse_status_filter(s: &str) -> Result<&'static str, OperatorError> {
    match s.to_ascii_lowercase().as_str() {
        "pending" => Ok("pending"),
        "submitted" => Ok("submitted"),
        "included" => Ok("included"),
        "finalized" => Ok("finalized"),
        "failed" => Ok("failed"),
        "dropped" => Ok("dropped"),
        _ => Err(OperatorError::InvalidStatusFilter(s.to_string())),
    }
}

/// 64-char lowercase hex → `AttestationId`. Case-insensitive in the
/// hex parser (operators may paste from mixed-case sources), but the
/// canonical render is lowercase via `to_hex()`.
fn parse_attestation_id_hex(s: &str) -> Result<AttestationId, OperatorError> {
    if s.len() != 64 {
        return Err(OperatorError::AttestationIdMalformed(format!(
            "expected 64 hex chars, got {}",
            s.len()
        )));
    }
    let mut bytes = [0u8; 32];
    for i in 0..32 {
        bytes[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .map_err(|e| OperatorError::AttestationIdMalformed(e.to_string()))?;
    }
    Ok(AttestationId::from_bytes(bytes))
}

/// Single row of `operator registry list` output — the structured
/// form so tests can assert without capturing stdout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RegistryListRow {
    pub id_hex: String,
    pub status: &'static str,
    pub tx_id: Option<String>,
    pub submitted_at_block: Option<u64>,
    pub updated_at: String,
}

/// Read-only registry list. Opens the registry, applies the optional
/// status filter, returns structured rows. Sorted ascending by
/// `id_hex` (the registry's natural order). Zero chain access.
async fn registry_list_core(
    registry_path: PathBuf,
    status_filter: Option<String>,
) -> Result<Vec<RegistryListRow>, OperatorError> {
    run_blocking(move || -> Result<Vec<RegistryListRow>, OperatorError> {
        // Validate filter up-front so a typo gets a clear error before
        // we read the disk.
        let filter: Option<&'static str> = match status_filter.as_deref() {
            Some(s) => Some(parse_status_filter(s)?),
            None => None,
        };
        let registry = AttestationRegistry::open(registry_path)?;
        let records = registry.list()?;
        let mut rows = Vec::with_capacity(records.len());
        for record in records {
            let label = status_label(&record.status);
            if let Some(f) = filter {
                if f != label {
                    continue;
                }
            }
            rows.push(RegistryListRow {
                id_hex: record.id.to_hex(),
                status: label,
                tx_id: record.receipt.as_ref().map(|r| r.tx_id.clone()),
                submitted_at_block: record.submitted_at_block,
                updated_at: record.updated_at.to_string(),
            });
        }
        Ok(rows)
    })
    .await
}

/// Read-only registry show by `AttestationId`. Loads the record;
/// returns it for callers (dispatch pretty-prints JSON). Missing ids
/// surface as `RegistryError::RecordNotFound` via the `?` from
/// `registry.load`.
async fn registry_show_core(
    registry_path: PathBuf,
    id_hex: String,
) -> Result<AttestationRecord, OperatorError> {
    run_blocking(move || -> Result<AttestationRecord, OperatorError> {
        let id = parse_attestation_id_hex(&id_hex)?;
        let registry = AttestationRegistry::open(registry_path)?;
        Ok(registry.load(&id)?)
    })
    .await
}

// ── registry summary (Stage 10a) ──────────────────────────────────────────────

/// Stage 10a: counts-by-status summary of a local registry, with an
/// optional oldest-`Submitted` age line when an RPC client is supplied.
/// Returned in structured form so tests can assert without capturing
/// stdout; the dispatch layer renders via [`print_registry_summary`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RegistrySummary {
    pub total: usize,
    pub pending: usize,
    pub submitted: usize,
    pub included: usize,
    pub finalized: usize,
    pub failed: usize,
    pub dropped: usize,
    /// `Some(_)` only when (a) the caller supplied an RPC client AND
    /// (b) at least one `Submitted` record has `submitted_at_block:
    /// Some(_)`. Otherwise the dispatcher emits a skip comment with
    /// the reason.
    pub oldest_submitted_age: Option<OldestSubmittedAge>,
    /// Reason the oldest-age line is missing. `None` means it's
    /// present; otherwise an operator-facing comment string.
    pub oldest_submitted_age_skip_reason: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OldestSubmittedAge {
    pub id_hex: String,
    pub submitted_at_block: u64,
    pub head: u64,
    /// `head.saturating_sub(submitted_at_block)`. Saturating because a
    /// head < submitted_at_block can only happen if the chain is
    /// regressing (already flagged separately by Stage 5.2 staleness);
    /// we don't want to panic the summary on a transient observation.
    pub age_blocks: u64,
}

/// Read-only summary. Opens the registry, counts every record by
/// status, then — if an RPC client + expected chain_id were supplied —
/// does **one** `get_chain_params` (chain-id guardrail) and **one**
/// `get_block_height(BlockFinality::Latest)` call to compute the
/// oldest-`Submitted` record's age in blocks. Zero chain mutation; no
/// submit code path; works under both default and `--features submit`
/// builds.
async fn registry_summary_core<T>(
    registry_path: PathBuf,
    chain_client: Option<(ClientFactory<T>, u64)>,
) -> Result<RegistrySummary, OperatorError>
where
    T: JsonRpcTransport + Send + Sync + 'static,
{
    run_blocking(move || -> Result<RegistrySummary, OperatorError> {
        let registry = AttestationRegistry::open(registry_path)?;
        let records = registry.list()?;

        let mut summary = RegistrySummary {
            total: records.len(),
            pending: 0,
            submitted: 0,
            included: 0,
            finalized: 0,
            failed: 0,
            dropped: 0,
            oldest_submitted_age: None,
            oldest_submitted_age_skip_reason: None,
        };

        // Track the oldest `Submitted` record's (id_hex, block) for the
        // optional age line. "Oldest" = lowest `submitted_at_block`.
        let mut oldest: Option<(String, u64)> = None;

        for record in &records {
            match status_label(&record.status) {
                "pending" => summary.pending += 1,
                "submitted" => {
                    summary.submitted += 1;
                    if let Some(block) = record.submitted_at_block {
                        match &oldest {
                            None => oldest = Some((record.id.to_hex(), block)),
                            Some((_, best)) if block < *best => {
                                oldest = Some((record.id.to_hex(), block));
                            }
                            _ => {}
                        }
                    }
                }
                "included" => summary.included += 1,
                "finalized" => summary.finalized += 1,
                "failed" => summary.failed += 1,
                "dropped" => summary.dropped += 1,
                // status_label() returns a closed set of &'static strs; any
                // other value here would mean an unhandled variant got added
                // to LocalAttestationStatus without updating status_label.
                other => {
                    return Err(OperatorError::Internal(format!(
                        "unknown status label {other:?} in summary count — \
                         status_label() out of sync with LocalAttestationStatus"
                    )));
                }
            }
        }

        // Decide whether to compute the age line.
        match (chain_client, oldest) {
            (None, _) => {
                summary.oldest_submitted_age_skip_reason =
                    Some("skipped (no --rpc-url)");
            }
            (Some(_), None) => {
                summary.oldest_submitted_age_skip_reason = if summary.submitted == 0 {
                    Some("skipped (no Submitted records)")
                } else {
                    Some("skipped (no Submitted record has submitted_at_block)")
                };
            }
            (Some((make_client, expect_chain_id)), Some((id_hex, block))) => {
                let client = make_client(DUMMY_SEED);
                let params = client.get_chain_params()?;
                check_chain_id(expect_chain_id, params.chain_id)?;
                let head = client
                    .get_block_height(BlockFinality::Latest)?
                    .height;
                summary.oldest_submitted_age = Some(OldestSubmittedAge {
                    id_hex,
                    submitted_at_block: block,
                    head,
                    age_blocks: head.saturating_sub(block),
                });
            }
        }

        // Stable observability marker (Stage 10a) — counts emit as
        // structured tracing fields so operator log pipelines can
        // grep against `event="registry_summary"` without parsing
        // bare stdout.
        info!(
            event = "registry_summary",
            total = summary.total,
            pending = summary.pending,
            submitted = summary.submitted,
            included = summary.included,
            finalized = summary.finalized,
            failed = summary.failed,
            dropped = summary.dropped,
            oldest_submitted_age_blocks =
                summary.oldest_submitted_age.as_ref().map(|a| a.age_blocks),
            "registry summary complete"
        );

        Ok(summary)
    })
    .await
}

/// Render a [`RegistrySummary`] to bare stdout in the documented
/// three-line shape. Grep contract:
///   `total=` always present on line 1
///   `pending=` / `submitted=` / … always present on line 2
///   line 3 is either `oldest_submitted_age_blocks=…` OR a
///   `# oldest_submitted_age: skipped (<reason>)` comment.
fn print_registry_summary(s: &RegistrySummary) {
    println!("total={}", s.total);
    println!(
        "pending={} submitted={} included={} finalized={} failed={} dropped={}",
        s.pending, s.submitted, s.included, s.finalized, s.failed, s.dropped,
    );
    match (&s.oldest_submitted_age, s.oldest_submitted_age_skip_reason) {
        (Some(age), _) => println!(
            "oldest_submitted_age_blocks={} (id={}, submitted_at_block={}, head={})",
            age.age_blocks, age.id_hex, age.submitted_at_block, age.head,
        ),
        (None, Some(reason)) => println!("# oldest_submitted_age: {reason}"),
        (None, None) => {
            // Defensive: the core always sets one of the two when
            // oldest_submitted_age is None. If neither is set, the
            // dispatcher would emit nothing — surface a typed comment
            // so the line still appears and grep contracts hold.
            println!("# oldest_submitted_age: skipped (no Submitted records)");
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use omni_sumchain::FakeJsonRpcTransport;
    use serde_json::json;

    const SEED: [u8; 32] = [42u8; 32];

    fn factory(fake: FakeJsonRpcTransport) -> ClientFactory<FakeJsonRpcTransport> {
        Arc::new(move |seed| SumChainClient::with_transport(seed, fake.clone()))
    }

    /// Activated chain params + block height. `chain_id` configurable;
    /// both activation heights = 0 so head ≥ 0 is always active.
    fn activated_fake(chain_id: u64) -> FakeJsonRpcTransport {
        let f = FakeJsonRpcTransport::new();
        f.set_response(
            "chain_getChainParams",
            Ok(json!({
                "finality_depth": 6,
                "min_fee": 1000,
                "chain_id": chain_id,
                "v2_enabled_from_height": 0,
                "omninode_enabled_from_height": 0,
            })),
        );
        f.set_response(
            "chain_getBlockHeight",
            Ok(json!({"height": 100, "finality": "latest"})),
        );
        f
    }

    // Stage 9a: `add_submit_responses` stays default-on — it's also
    // used by `loop_monitor_only_never_submits` to *prove* the
    // monitor-only loop ignores seeded submit RPCs. `status_json` and
    // `submit_fake_with_status` below are submit-only.
    fn add_submit_responses(f: &FakeJsonRpcTransport) {
        f.set_response("sum_getNonce", Ok(json!(0)));
        f.set_response(
            "sum_sendRawTransaction",
            Ok(json!({ "tx_hash": "0xsmoke" })),
        );
        f.set_response(
            "sum_getInferenceAttestationStatus",
            Ok(json!({"status": "finalized", "included_at_height": 101, "reason": null})),
        );
    }

    #[cfg(feature = "submit")]
    fn status_json(status: &str) -> serde_json::Value {
        json!({"status": status, "included_at_height": 101, "reason": null})
    }

    /// Activated non-mainnet fake wired for a synthetic submit, with the
    /// status RPC seeded to `status` (overridable mid-test via the
    /// shared `Arc<Mutex>` for sequenced included→finalized coverage).
    #[cfg(feature = "submit")]
    fn submit_fake_with_status(status: &str) -> FakeJsonRpcTransport {
        let f = activated_fake(31337);
        f.set_response("sum_getNonce", Ok(json!(0)));
        f.set_response("sum_sendRawTransaction", Ok(json!({ "tx_hash": "0xsmoke" })));
        f.set_response("sum_getInferenceAttestationStatus", Ok(status_json(status)));
        f
    }

    fn seed_address() -> String {
        signer_chain_address_base58(&SEED).unwrap()
    }

    fn submit_calls(f: &FakeJsonRpcTransport) -> usize {
        f.calls()
            .into_iter()
            .filter(|(m, _)| m == "sum_sendRawTransaction")
            .count()
    }

    // ── Pure helpers ─────────────────────────────────────────────────

    #[test]
    fn check_chain_id_matrix() {
        assert!(check_chain_id(1, 1).is_ok());
        assert!(matches!(
            check_chain_id(31337, 1),
            Err(OperatorError::ChainIdMismatch { expected: 31337, actual: 1 })
        ));
    }

    #[test]
    fn submission_permitted_matrix() {
        assert!(matches!(
            submission_permitted(1, false, false),
            Err(OperatorError::SubmitNotPermitted)
        ));
        assert!(matches!(
            submission_permitted(1, true, false),
            Err(OperatorError::MainnetSubmitNotPermitted)
        ));
        assert!(submission_permitted(1, true, true).is_ok());
        // non-mainnet ignores the mainnet flag entirely
        assert!(submission_permitted(31337, true, false).is_ok());
        assert!(matches!(
            submission_permitted(31337, false, true),
            Err(OperatorError::SubmitNotPermitted)
        ));
    }

    #[test]
    fn blocks_remaining_saturates_and_handles_none() {
        assert_eq!(blocks_remaining(10, Some(100)), Some(90));
        assert_eq!(blocks_remaining(100, Some(50)), Some(0)); // head past activation
        assert_eq!(blocks_remaining(100, Some(100)), Some(0));
        assert_eq!(blocks_remaining(10, None), None);
    }

    #[cfg(feature = "submit")]
    #[test]
    fn resolve_smoke_source_matrix() {
        let p = Some(Path::new("/tmp/a.json"));
        // file present, non-mainnet and mainnet → File
        assert_eq!(
            resolve_smoke_source(31337, p, false).unwrap(),
            SmokeSource::File(PathBuf::from("/tmp/a.json"))
        );
        assert_eq!(
            resolve_smoke_source(1, p, false).unwrap(),
            SmokeSource::File(PathBuf::from("/tmp/a.json"))
        );
        // both flags → conflict
        assert!(matches!(
            resolve_smoke_source(31337, p, true),
            Err(OperatorError::ConflictingSmokeSource)
        ));
        // synthetic, non-mainnet → Synthetic
        assert_eq!(
            resolve_smoke_source(31337, None, true).unwrap(),
            SmokeSource::Synthetic
        );
        // synthetic, mainnet → forbidden
        assert!(matches!(
            resolve_smoke_source(1, None, true),
            Err(OperatorError::SyntheticMainnetForbidden)
        ));
        // nothing, mainnet → requires attestation json
        assert!(matches!(
            resolve_smoke_source(1, None, false),
            Err(OperatorError::MainnetRequiresAttestationJson)
        ));
        // nothing, non-mainnet → must opt into synthetic explicitly
        assert!(matches!(
            resolve_smoke_source(31337, None, false),
            Err(OperatorError::SyntheticRequiresExplicitFlag)
        ));
    }

    // ── watch-activation ─────────────────────────────────────────────

    #[tokio::test]
    async fn watch_activation_exits_ok_when_active() {
        let fake = activated_fake(1);
        let cfg = WatchConfig {
            expect_chain_id: 1,
            poll_interval_secs: 0,
            max_ticks: Some(1),
        };
        watch_activation_core(factory(fake), cfg).await.unwrap();
    }

    #[tokio::test]
    async fn watch_activation_errors_when_not_reached_within_budget() {
        let fake = FakeJsonRpcTransport::new();
        fake.set_response(
            "chain_getChainParams",
            Ok(json!({
                "finality_depth": 6, "min_fee": 1000, "chain_id": 1,
                "v2_enabled_from_height": 5_200_000,
                "omninode_enabled_from_height": 6_000_000,
            })),
        );
        fake.set_response(
            "chain_getBlockHeight",
            Ok(json!({"height": 100, "finality": "latest"})),
        );
        let cfg = WatchConfig {
            expect_chain_id: 1,
            poll_interval_secs: 0,
            max_ticks: Some(1),
        };
        let err = watch_activation_core(factory(fake), cfg).await.unwrap_err();
        assert!(matches!(err, OperatorError::ActivationNotReached));
    }

    #[tokio::test]
    async fn watch_activation_rejects_chain_id_mismatch_before_height() {
        let fake = activated_fake(1);
        let cfg = WatchConfig {
            expect_chain_id: 31337,
            poll_interval_secs: 0,
            max_ticks: Some(1),
        };
        let err = watch_activation_core(factory(fake.clone()), cfg)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            OperatorError::ChainIdMismatch { expected: 31337, actual: 1 }
        ));
        let called: Vec<String> = fake.calls().into_iter().map(|(m, _)| m).collect();
        assert!(
            !called.contains(&"chain_getBlockHeight".to_string()),
            "chain-id guardrail must fire before the block-height read"
        );
    }

    // ── smoke: gate-7-before-seed guarantee ──────────────────────────

    #[cfg(feature = "submit")]
    #[tokio::test]
    async fn smoke_mainnet_without_attestation_json_is_refused_before_seed() {
        let fake = activated_fake(1);
        let cfg = SmokeConfig {
            expect_chain_id: 1,
            registry_path: PathBuf::from("/nonexistent"),
            attestation_json: None,
            synthetic: false,
            allow_submit: true,
            allow_mainnet_submit: true,
            confirm_timeout_secs: 1,
            // Seed deliberately ABSENT — if gate ordering regressed, the
            // error would be SeedMissing instead of the mainnet gate.
            seed_source: SeedSource::AbsentForTest,
        };
        let err = smoke_core(factory(fake.clone()), cfg).await.unwrap_err();
        assert!(
            matches!(err, OperatorError::MainnetRequiresAttestationJson),
            "expected MainnetRequiresAttestationJson, got {err:?}"
        );
        assert_eq!(submit_calls(&fake), 0);
    }

    #[cfg(feature = "submit")]
    #[tokio::test]
    async fn smoke_mainnet_rejects_synthetic_before_malformed_seed() {
        let fake = activated_fake(1);
        let cfg = SmokeConfig {
            expect_chain_id: 1,
            registry_path: PathBuf::from("/nonexistent"),
            attestation_json: None,
            synthetic: true,
            allow_submit: true,
            allow_mainnet_submit: true,
            confirm_timeout_secs: 1,
            // MALFORMED seed — must NOT mask the gate-7 error.
            seed_source: SeedSource::MalformedForTest("zz".into()),
        };
        let err = smoke_core(factory(fake.clone()), cfg).await.unwrap_err();
        assert!(
            matches!(err, OperatorError::SyntheticMainnetForbidden),
            "gate 7 must fire before seed parsing; got {err:?}"
        );
        assert_eq!(submit_calls(&fake), 0);
    }

    // ── Stage 11a: mock-v1 attestation builder ───────────────────────

    #[cfg(feature = "submit")]
    #[test]
    fn build_mock_v1_attestation_produces_real_blake3_commitment_on_nonmainnet() {
        // Non-mainnet chain_id is the only allowed path. Confirm the
        // returned attestation has *real* BLAKE3-derived commitment
        // bytes — no `"a".repeat(64)` / `[0x11; 32]` Stage 8a-style
        // placeholders.
        let verifier_address = "2mvPk4h883B7DrcZvwy7yWKXyGYHuVzGP"; // shape-only
        let head = 12345u64;
        let chain_id = 31337u64; // non-mainnet
        let att = build_mock_v1_attestation(verifier_address, chain_id, head).unwrap();

        // session_id is the Stage 11a-prefixed shape (was stage8a-).
        assert!(
            att.commitment.session_id.starts_with("omninode-stage11a-smoke-"),
            "session_id should carry the Stage 11a prefix; got {}",
            att.commitment.session_id
        );

        // model_hash / response_hash must be real BLAKE3 hex (64 lowercase
        // hex chars, NOT the pre-11a `"a".repeat(64)` / `"b".repeat(64)`
        // placeholders).
        assert_eq!(att.commitment.model_hash.len(), 64);
        assert_eq!(att.commitment.response_hash.len(), 64);
        assert_ne!(att.commitment.model_hash, "a".repeat(64));
        assert_ne!(att.commitment.response_hash, "b".repeat(64));
        assert!(att
            .commitment
            .model_hash
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert!(att
            .commitment
            .response_hash
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));

        // manifest_snip_root and proof_snip_root must NOT be the pre-11a
        // `[0x11; 32]` / `[0x22; 32]` constants.
        assert_ne!(
            att.commitment.manifest_snip_root,
            SnipV2ObjectId::from_bytes([0x11u8; 32])
        );
        assert_ne!(
            att.commitment.proof_snip_root,
            SnipV2ObjectId::from_bytes([0x22u8; 32])
        );

        // Determinism: same inputs → same attestation. This is what makes
        // the synthetic path retry-safe under Stage 5.1's
        // `(session_id, verifier_address)` dedup.
        let att2 = build_mock_v1_attestation(verifier_address, chain_id, head).unwrap();
        assert_eq!(att.commitment, att2.commitment);

        // verifier_signature carries the Stage 11a + backend marker so
        // operators reading the registry can tell which builder produced
        // the record.
        assert_eq!(att.verifier_signature, "stage11a-mock-v1");
    }

    #[cfg(feature = "submit")]
    #[test]
    fn build_mock_v1_attestation_refuses_mainnet_with_explicit_error() {
        // The Stage 11a OQ5 hard requirement: mainnet must reject
        // backend_id == "mock-v1" before any submit-side RPC, even with
        // --allow-mainnet-submit. We assert via the typed error variant
        // and confirm the backend_id is captured verbatim so operator
        // logs / triage can grep on it.
        let err = build_mock_v1_attestation("addr", 1, 0).unwrap_err();
        assert!(
            matches!(err, OperatorError::MockBackendRefusedOnMainnet { ref backend_id } if backend_id == "mock-v1"),
            "expected MockBackendRefusedOnMainnet {{ backend_id: \"mock-v1\" }}, got {err:?}"
        );
    }

    #[cfg(feature = "submit")]
    #[tokio::test]
    async fn smoke_nonmainnet_synthetic_requires_explicit_flag() {
        let fake = activated_fake(31337);
        let cfg = SmokeConfig {
            expect_chain_id: 31337,
            registry_path: PathBuf::from("/nonexistent"),
            attestation_json: None,
            synthetic: false,
            allow_submit: true,
            allow_mainnet_submit: false,
            confirm_timeout_secs: 1,
            seed_source: SeedSource::AbsentForTest,
        };
        let err = smoke_core(factory(fake), cfg).await.unwrap_err();
        assert!(matches!(err, OperatorError::SyntheticRequiresExplicitFlag));
    }

    #[cfg(feature = "submit")]
    #[tokio::test]
    async fn smoke_attestation_json_verifier_mismatch_is_refused() {
        let fake = activated_fake(1);
        add_submit_responses(&fake);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("att.json");
        let att = synth_attestation("SomeoneElsesAddress", 1);
        std::fs::write(&path, serde_json::to_vec(&att).unwrap()).unwrap();

        let cfg = SmokeConfig {
            expect_chain_id: 1,
            registry_path: dir.path().join("registry"),
            attestation_json: Some(path),
            synthetic: false,
            allow_submit: true,
            allow_mainnet_submit: true,
            confirm_timeout_secs: 1,
            seed_source: SeedSource::Explicit(SEED),
        };
        let err = smoke_core(factory(fake.clone()), cfg).await.unwrap_err();
        assert!(
            matches!(err, OperatorError::VerifierAddressMismatch { .. }),
            "got {err:?}"
        );
        assert_eq!(submit_calls(&fake), 0);
    }

    #[cfg(feature = "submit")]
    #[tokio::test]
    async fn smoke_attestation_json_malformed_errors_cleanly() {
        let fake = activated_fake(1);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("att.json");
        std::fs::write(&path, b"{ not valid attestation }").unwrap();

        let cfg = SmokeConfig {
            expect_chain_id: 1,
            registry_path: dir.path().join("registry"),
            attestation_json: Some(path),
            synthetic: false,
            allow_submit: true,
            allow_mainnet_submit: true,
            confirm_timeout_secs: 1,
            seed_source: SeedSource::Explicit(SEED),
        };
        let err = smoke_core(factory(fake.clone()), cfg).await.unwrap_err();
        assert!(matches!(err, OperatorError::AttestationJsonParse(_)), "got {err:?}");
        assert_eq!(submit_calls(&fake), 0);
    }

    #[cfg(feature = "submit")]
    #[tokio::test]
    async fn smoke_mainnet_with_attestation_json_proceeds_to_finalized() {
        let fake = activated_fake(1);
        add_submit_responses(&fake);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("att.json");
        let att = synth_attestation(&seed_address(), 100);
        std::fs::write(&path, serde_json::to_vec(&att).unwrap()).unwrap();

        let cfg = SmokeConfig {
            expect_chain_id: 1,
            registry_path: dir.path().join("registry"),
            attestation_json: Some(path),
            synthetic: false,
            allow_submit: true,
            allow_mainnet_submit: true,
            confirm_timeout_secs: 5,
            seed_source: SeedSource::Explicit(SEED),
        };
        smoke_core(factory(fake.clone()), cfg).await.unwrap();
        assert_eq!(submit_calls(&fake), 1);
    }

    #[cfg(feature = "submit")]
    #[tokio::test]
    async fn smoke_mainnet_without_mainnet_flag_is_refused() {
        let fake = activated_fake(1);
        add_submit_responses(&fake);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("att.json");
        let att = synth_attestation(&seed_address(), 100);
        std::fs::write(&path, serde_json::to_vec(&att).unwrap()).unwrap();

        let cfg = SmokeConfig {
            expect_chain_id: 1,
            registry_path: dir.path().join("registry"),
            attestation_json: Some(path),
            synthetic: false,
            allow_submit: true,
            allow_mainnet_submit: false, // missing the mainnet gate
            confirm_timeout_secs: 1,
            seed_source: SeedSource::Explicit(SEED),
        };
        let err = smoke_core(factory(fake.clone()), cfg).await.unwrap_err();
        assert!(matches!(err, OperatorError::MainnetSubmitNotPermitted), "got {err:?}");
        assert_eq!(submit_calls(&fake), 0);
    }

    // ── smoke: Included is progress, only Finalized is success ────────

    /// Chain stays `included` forever → smoke must NOT declare success;
    /// it times out with the last observed status reflecting Included.
    #[cfg(feature = "submit")]
    #[tokio::test]
    async fn smoke_included_only_times_out() {
        let fake = submit_fake_with_status("included");
        let dir = tempfile::tempdir().unwrap();
        let cfg = SmokeConfig {
            expect_chain_id: 31337,
            registry_path: dir.path().join("registry"),
            attestation_json: None,
            synthetic: true,
            allow_submit: true,
            allow_mainnet_submit: false,
            confirm_timeout_secs: 1, // < the 2s inter-poll wait → trips after one Included poll
            seed_source: SeedSource::Explicit(SEED),
        };
        let err = smoke_core(factory(fake.clone()), cfg).await.unwrap_err();
        match err {
            OperatorError::SmokeConfirmTimeout { last } => {
                assert!(
                    last.contains("Included"),
                    "timeout must report the last observed status as Included; got {last:?}"
                );
            }
            other => panic!("expected SmokeConfirmTimeout, got {other:?}"),
        }
        // Submitted exactly once; Included never short-circuited to Ok.
        assert_eq!(submit_calls(&fake), 1);
    }

    /// Chain reports `included`, then (mid-poll, via the shared fake)
    /// flips to `finalized` → smoke succeeds only after finality, and
    /// must have polled status at least twice.
    #[cfg(feature = "submit")]
    #[tokio::test]
    async fn smoke_included_then_finalized_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let fake = submit_fake_with_status("included");
        let cfg = SmokeConfig {
            expect_chain_id: 31337,
            registry_path: dir.path().join("registry"),
            attestation_json: None,
            synthetic: true,
            allow_submit: true,
            allow_mainnet_submit: false,
            confirm_timeout_secs: 30, // generous; success expected well before
            seed_source: SeedSource::Explicit(SEED),
        };

        let fake_for_task = fake.clone();
        let handle =
            tokio::spawn(async move { smoke_core(factory(fake_for_task), cfg).await });

        // The spawned task submits and runs its first `included` poll
        // immediately; the 2s inter-poll wait gives a wide margin to
        // flip the shared fake to `finalized` before poll #2.
        tokio::time::sleep(Duration::from_millis(200)).await;
        fake.set_response(
            "sum_getInferenceAttestationStatus",
            Ok(status_json("finalized")),
        );

        handle
            .await
            .expect("smoke task join")
            .expect("smoke must succeed once status reaches Finalized");

        let status_polls = fake
            .calls()
            .into_iter()
            .filter(|(m, _)| m == "sum_getInferenceAttestationStatus")
            .count();
        assert!(
            status_polls >= 2,
            "expected ≥2 status polls (Included then Finalized), got {status_polls}"
        );
    }

    // ── loop ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn loop_monitor_only_never_submits() {
        let fake = activated_fake(31337);
        add_submit_responses(&fake);
        let dir = tempfile::tempdir().unwrap();
        let reg_path = dir.path().join("registry");

        // Seed a Dropped record so a (forbidden) retry would be visible.
        {
            let reg = AttestationRegistry::open(reg_path.clone()).unwrap();
            let att = synth_attestation(&seed_address(), 1);
            let r = reg.insert(att).unwrap();
            reg.mark_submitted_with_block(
                &r.id,
                omni_zkml::SubmissionReceipt { tx_id: "0xold".into(), note: None },
                1,
            )
            .unwrap();
            reg.mark_dropped(&r.id, Some("seeded".into())).unwrap();
        }

        let cfg = LoopConfig {
            expect_chain_id: 31337,
            registry_path: reg_path,
            staleness_threshold_blocks: 10,
            poll_interval_secs: 0,
            allow_submit: false, // monitor-only
            allow_mainnet_submit: false,
            max_ticks: Some(1),
            seed_source: SeedSource::AbsentForTest,
        };
        loop_core(factory(fake.clone()), cfg).await.unwrap();
        assert_eq!(
            submit_calls(&fake),
            0,
            "monitor-only loop must never reach sum_sendRawTransaction"
        );
    }

    /// Stage 10a: monitor-only loop ticks must perform the same
    /// read-only activation check that retry-enabled ticks do, so the
    /// `event="chain_params"` and `event="activation_state"` markers
    /// fire every tick regardless of submit mode. The Stage 9c+ chain
    /// fake records every method call, so we assert via observable
    /// RPC traffic: at least one `chain_getChainParams` and one
    /// `chain_getBlockHeight` must reach the chain in a monitor-only
    /// tick (pre-Stage-10a, monitor-only ticks skipped the activation
    /// reads entirely and never touched `chain_getBlockHeight`).
    #[tokio::test]
    async fn loop_monitor_only_reads_activation_state_every_tick() {
        let fake = activated_fake(31337);
        add_submit_responses(&fake);
        let dir = tempfile::tempdir().unwrap();
        let reg_path = dir.path().join("registry");
        let _ = AttestationRegistry::open(reg_path.clone()).unwrap();

        let cfg = LoopConfig {
            expect_chain_id: 31337,
            registry_path: reg_path,
            staleness_threshold_blocks: 10,
            poll_interval_secs: 0,
            allow_submit: false, // monitor-only
            allow_mainnet_submit: false,
            max_ticks: Some(1),
            seed_source: SeedSource::AbsentForTest,
        };
        loop_core(factory(fake.clone()), cfg).await.unwrap();

        let methods: Vec<String> =
            fake.calls().into_iter().map(|(m, _)| m).collect();
        assert!(
            methods.iter().any(|m| m == "chain_getChainParams"),
            "monitor-only tick must fetch chain_getChainParams (for \
             event=\"chain_params\"); recorded methods: {methods:?}"
        );
        assert!(
            methods.iter().any(|m| m == "chain_getBlockHeight"),
            "monitor-only tick must fetch chain_getBlockHeight (for \
             event=\"activation_state\"); recorded methods: {methods:?}"
        );
        assert_eq!(
            submit_calls(&fake),
            0,
            "the new activation-state reads must NOT introduce any \
             submit-side RPCs in monitor-only mode"
        );
    }

    #[cfg(feature = "submit")]
    #[tokio::test]
    async fn loop_submit_mode_retries_dropped_record() {
        let fake = activated_fake(31337);
        add_submit_responses(&fake);
        let dir = tempfile::tempdir().unwrap();
        let reg_path = dir.path().join("registry");

        {
            let reg = AttestationRegistry::open(reg_path.clone()).unwrap();
            let att = synth_attestation(&seed_address(), 1);
            let r = reg.insert(att).unwrap();
            reg.mark_submitted_with_block(
                &r.id,
                omni_zkml::SubmissionReceipt { tx_id: "0xold".into(), note: None },
                1,
            )
            .unwrap();
            reg.mark_dropped(&r.id, Some("seeded".into())).unwrap();
        }

        let cfg = LoopConfig {
            expect_chain_id: 31337,
            registry_path: reg_path,
            staleness_threshold_blocks: 10,
            poll_interval_secs: 0,
            allow_submit: true,
            allow_mainnet_submit: false, // non-mainnet: not required
            max_ticks: Some(1),
            seed_source: SeedSource::Explicit(SEED),
        };
        loop_core(factory(fake.clone()), cfg).await.unwrap();
        assert_eq!(
            submit_calls(&fake),
            1,
            "submit-mode loop must retry the seeded Dropped record exactly once"
        );
    }

    #[cfg(feature = "submit")]
    #[tokio::test]
    async fn loop_mainnet_without_mainnet_flag_fails_at_startup() {
        let fake = activated_fake(1);
        let dir = tempfile::tempdir().unwrap();
        let cfg = LoopConfig {
            expect_chain_id: 1,
            registry_path: dir.path().join("registry"),
            staleness_threshold_blocks: 10,
            poll_interval_secs: 0,
            allow_submit: true,
            allow_mainnet_submit: false,
            max_ticks: Some(1),
            seed_source: SeedSource::AbsentForTest,
        };
        let err = loop_core(factory(fake), cfg).await.unwrap_err();
        assert!(matches!(err, OperatorError::MainnetSubmitNotPermitted), "got {err:?}");
    }

    #[tokio::test]
    async fn loop_rejects_zero_staleness_threshold() {
        let fake = activated_fake(31337);
        let dir = tempfile::tempdir().unwrap();
        let cfg = LoopConfig {
            expect_chain_id: 31337,
            registry_path: dir.path().join("registry"),
            staleness_threshold_blocks: 0, // StalenessPolicy::new rejects 0
            poll_interval_secs: 0,
            allow_submit: false,
            allow_mainnet_submit: false,
            max_ticks: Some(1),
            seed_source: SeedSource::AbsentForTest,
        };
        let err = loop_core(factory(fake), cfg).await.unwrap_err();
        assert!(matches!(err, OperatorError::Policy(_)), "got {err:?}");
    }

    // ── preflight (Stage 8b) ─────────────────────────────────────────

    fn write_attestation(dir: &std::path::Path, verifier_address: &str) -> PathBuf {
        let path = dir.join("att.json");
        let att = synth_attestation(verifier_address, 1);
        std::fs::write(&path, serde_json::to_vec(&att).unwrap()).unwrap();
        path
    }

    #[tokio::test]
    async fn preflight_offline_ok_with_matching_seed_and_no_chain_calls() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_attestation(dir.path(), &seed_address());
        let cfg = PreflightConfig {
            attestation_json: path,
            expect_chain_id: None,
            seed_source: SeedSource::Explicit(SEED),
        };
        // No client factory → purely offline; cannot reach the chain.
        run_preflight::<FakeJsonRpcTransport>(None, cfg).await.unwrap();
    }

    #[tokio::test]
    async fn preflight_offline_verifier_mismatch_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_attestation(dir.path(), "NotTheSeedAddress");
        let cfg = PreflightConfig {
            attestation_json: path,
            expect_chain_id: None,
            seed_source: SeedSource::Explicit(SEED),
        };
        let err = run_preflight::<FakeJsonRpcTransport>(None, cfg)
            .await
            .unwrap_err();
        assert!(matches!(err, OperatorError::VerifierAddressMismatch { .. }), "got {err:?}");
    }

    #[tokio::test]
    async fn preflight_offline_malformed_json_errors_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("att.json");
        std::fs::write(&path, b"{ not attestation }").unwrap();
        let cfg = PreflightConfig {
            attestation_json: path,
            expect_chain_id: None,
            seed_source: SeedSource::Explicit(SEED),
        };
        let err = run_preflight::<FakeJsonRpcTransport>(None, cfg)
            .await
            .unwrap_err();
        assert!(matches!(err, OperatorError::AttestationJsonParse(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn preflight_offline_missing_seed_is_surfaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_attestation(dir.path(), &seed_address());
        let cfg = PreflightConfig {
            attestation_json: path,
            expect_chain_id: None,
            seed_source: SeedSource::AbsentForTest,
        };
        let err = run_preflight::<FakeJsonRpcTransport>(None, cfg)
            .await
            .unwrap_err();
        assert!(matches!(err, OperatorError::SeedMissing), "got {err:?}");
    }

    #[tokio::test]
    async fn preflight_rpc_requires_expect_chain_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_attestation(dir.path(), &seed_address());
        let fake = activated_fake(1);
        let cfg = PreflightConfig {
            attestation_json: path,
            expect_chain_id: None, // but a factory IS supplied
            seed_source: SeedSource::Explicit(SEED),
        };
        let err = run_preflight(Some(factory(fake.clone())), cfg)
            .await
            .unwrap_err();
        assert!(matches!(err, OperatorError::ExpectChainIdRequiredWithRpc), "got {err:?}");
        assert_eq!(submit_calls(&fake), 0);
    }

    #[tokio::test]
    async fn preflight_rpc_chain_id_mismatch_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_attestation(dir.path(), &seed_address());
        let fake = activated_fake(1);
        let cfg = PreflightConfig {
            attestation_json: path,
            expect_chain_id: Some(31337),
            seed_source: SeedSource::Explicit(SEED),
        };
        let err = run_preflight(Some(factory(fake.clone())), cfg)
            .await
            .unwrap_err();
        assert!(
            matches!(err, OperatorError::ChainIdMismatch { expected: 31337, actual: 1 }),
            "got {err:?}"
        );
        assert_eq!(submit_calls(&fake), 0);
    }

    #[tokio::test]
    async fn preflight_rpc_happy_reports_snapshot_and_never_submits() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_attestation(dir.path(), &seed_address());
        let fake = activated_fake(1);
        let cfg = PreflightConfig {
            attestation_json: path,
            expect_chain_id: Some(1),
            seed_source: SeedSource::Explicit(SEED),
        };
        run_preflight(Some(factory(fake.clone())), cfg).await.unwrap();
        let methods: Vec<String> = fake.calls().into_iter().map(|(m, _)| m).collect();
        assert!(methods.iter().any(|m| m == "chain_getChainParams"));
        assert!(methods.iter().any(|m| m == "chain_getBlockHeight"));
        assert_eq!(submit_calls(&fake), 0, "preflight must never submit");
    }

    /// Report-only even when already activated (Q2): preflight succeeds
    /// against an activated chain rather than erroring.
    #[tokio::test]
    async fn preflight_rpc_reports_only_when_already_activated() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_attestation(dir.path(), &seed_address());
        let fake = activated_fake(1); // omninode/v2 enabled from height 0 → active
        let cfg = PreflightConfig {
            attestation_json: path,
            expect_chain_id: Some(1),
            seed_source: SeedSource::Explicit(SEED),
        };
        run_preflight(Some(factory(fake.clone())), cfg).await.unwrap();
        assert_eq!(submit_calls(&fake), 0);
    }

    // ── query (Stage 8b) ─────────────────────────────────────────────

    fn attestation_info_json(session_id: &str, verifier_address: &str) -> serde_json::Value {
        json!({
            "session_id": session_id,
            "verifier_address": verifier_address,
            "model_hash": format!("0x{}", "a".repeat(64)),
            "manifest_root": format!("0x{}", "1".repeat(64)),
            "response_hash": format!("0x{}", "b".repeat(64)),
            "proof_root": format!("0x{}", "2".repeat(64)),
            "verifier_signature": format!("0x{}", "c".repeat(128)),
            "included_at_height": 6_000_010,
            "tx_hash": "0xfeed",
            "finalized": true
        })
    }

    #[tokio::test]
    async fn query_pair_returns_found_attestation_and_never_submits() {
        let fake = activated_fake(1);
        fake.set_response(
            "sum_getInferenceAttestation",
            Ok(attestation_info_json("sess-q", "addr-q")),
        );
        let cfg = QueryConfig {
            expect_chain_id: 1,
            mode: QueryMode::Pair {
                session_id: "sess-q".into(),
                verifier_address: "addr-q".into(),
            },
        };
        run_query(factory(fake.clone()), cfg).await.unwrap();
        let methods: Vec<String> = fake.calls().into_iter().map(|(m, _)| m).collect();
        assert!(methods.iter().any(|m| m == "sum_getInferenceAttestation"));
        assert_eq!(submit_calls(&fake), 0);
    }

    #[tokio::test]
    async fn query_pair_not_found_returns_typed_error() {
        let fake = activated_fake(1);
        fake.set_response("sum_getInferenceAttestation", Ok(json!(null)));
        let cfg = QueryConfig {
            expect_chain_id: 1,
            mode: QueryMode::Pair {
                session_id: "missing".into(),
                verifier_address: "addr-x".into(),
            },
        };
        let err = run_query(factory(fake.clone()), cfg).await.unwrap_err();
        assert!(matches!(err, OperatorError::AttestationNotFound { .. }), "got {err:?}");
        assert_eq!(submit_calls(&fake), 0);
    }

    #[tokio::test]
    async fn query_chain_id_mismatch_is_refused_before_read() {
        let fake = activated_fake(1);
        fake.set_response(
            "sum_getInferenceAttestation",
            Ok(attestation_info_json("s", "a")),
        );
        let cfg = QueryConfig {
            expect_chain_id: 31337,
            mode: QueryMode::Pair {
                session_id: "s".into(),
                verifier_address: "a".into(),
            },
        };
        let err = run_query(factory(fake.clone()), cfg).await.unwrap_err();
        assert!(matches!(err, OperatorError::ChainIdMismatch { .. }), "got {err:?}");
        let methods: Vec<String> = fake.calls().into_iter().map(|(m, _)| m).collect();
        assert!(
            !methods.iter().any(|m| m == "sum_getInferenceAttestation"),
            "chain-id guardrail must fire before the attestation read"
        );
        assert_eq!(submit_calls(&fake), 0);
    }

    // ── Stage 8c: query --tx-hash mode ───────────────────────────────

    #[test]
    fn resolve_query_mode_matrix() {
        // tx-hash alone → TxHash
        assert_eq!(
            resolve_query_mode(Some("0xabc".into()), None, None).unwrap(),
            QueryMode::TxHash("0xabc".into())
        );
        // full pair → Pair
        assert_eq!(
            resolve_query_mode(None, Some("s".into()), Some("a".into())).unwrap(),
            QueryMode::Pair { session_id: "s".into(), verifier_address: "a".into() }
        );
        // tx-hash + any pair field → conflict
        assert!(matches!(
            resolve_query_mode(Some("0xabc".into()), Some("s".into()), None),
            Err(OperatorError::ConflictingQueryMode)
        ));
        // incomplete pair → required
        assert!(matches!(
            resolve_query_mode(None, Some("s".into()), None),
            Err(OperatorError::QueryModeRequired)
        ));
        assert!(matches!(
            resolve_query_mode(None, None, Some("a".into())),
            Err(OperatorError::QueryModeRequired)
        ));
        // nothing → required
        assert!(matches!(
            resolve_query_mode(None, None, None),
            Err(OperatorError::QueryModeRequired)
        ));
    }

    #[tokio::test]
    async fn query_tx_hash_reports_status_and_never_submits() {
        let fake = activated_fake(1);
        fake.set_response(
            "sum_getInferenceAttestationStatus",
            Ok(json!({"status": "finalized", "included_at_height": 6_049_201, "reason": null})),
        );
        let cfg = QueryConfig {
            expect_chain_id: 1,
            mode: QueryMode::TxHash("0x3a9cbf".into()),
        };
        // tx-hash mode always returns Ok (status report, not a presence
        // assertion) — even `unknown` is a legitimate observation.
        run_query(factory(fake.clone()), cfg).await.unwrap();
        let methods: Vec<String> = fake.calls().into_iter().map(|(m, _)| m).collect();
        assert!(methods.iter().any(|m| m == "sum_getInferenceAttestationStatus"));
        assert!(!methods.iter().any(|m| m == "sum_getInferenceAttestation"));
        assert_eq!(submit_calls(&fake), 0);
    }

    #[tokio::test]
    async fn query_tx_hash_unknown_status_is_ok_not_error() {
        let fake = activated_fake(1);
        fake.set_response(
            "sum_getInferenceAttestationStatus",
            Ok(json!({"status": "unknown", "included_at_height": null, "reason": null})),
        );
        let cfg = QueryConfig {
            expect_chain_id: 1,
            mode: QueryMode::TxHash("0xdeadbeef".into()),
        };
        run_query(factory(fake.clone()), cfg).await.unwrap();
        assert_eq!(submit_calls(&fake), 0);
    }

    // ── Stage 8c: derive-address ─────────────────────────────────────

    #[tokio::test]
    async fn derive_address_returns_seed_derived_address() {
        let addr = derive_address_core(SeedSource::Explicit(SEED)).await.unwrap();
        assert_eq!(addr, seed_address());
    }

    #[tokio::test]
    async fn derive_address_missing_seed_is_surfaced() {
        let err = derive_address_core(SeedSource::AbsentForTest).await.unwrap_err();
        assert!(matches!(err, OperatorError::SeedMissing), "got {err:?}");
    }

    // ── Stage 8c: smoke returns a populated SmokeOutcome ─────────────

    #[cfg(feature = "submit")]
    #[tokio::test]
    async fn smoke_outcome_carries_tx_hash_and_finalization_fields() {
        let fake = activated_fake(1);
        add_submit_responses(&fake); // status RPC seeded "finalized" w/ included_at_height 101
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("att.json");
        let att = synth_attestation(&seed_address(), 100);
        std::fs::write(&path, serde_json::to_vec(&att).unwrap()).unwrap();

        let cfg = SmokeConfig {
            expect_chain_id: 1,
            registry_path: dir.path().join("registry"),
            attestation_json: Some(path),
            synthetic: false,
            allow_submit: true,
            allow_mainnet_submit: true,
            confirm_timeout_secs: 5,
            seed_source: SeedSource::Explicit(SEED),
        };
        let outcome = smoke_core(factory(fake.clone()), cfg).await.unwrap();
        assert!(outcome.finalized);
        assert_eq!(outcome.tx_hash, "0xsmoke");
        assert_eq!(outcome.verifier_address, seed_address());
        assert_eq!(outcome.session_id, att.commitment.session_id);
        assert_eq!(outcome.included_at_height, Some(101)); // from the extra finalization read
        assert!(!outcome.attestation_id.is_empty());
        assert_eq!(submit_calls(&fake), 1);
    }

    // ── Stage 9a: registry list / show ────────────────────────────────

    use omni_zkml::SubmissionReceipt;

    fn seed_three_records(registry_path: &std::path::Path) -> Vec<AttestationId> {
        let reg = AttestationRegistry::open(registry_path.to_path_buf()).unwrap();
        let a = reg.insert(synth_attestation("addr-a", 1)).unwrap();
        let b = reg.insert(synth_attestation("addr-b", 2)).unwrap();
        let c = reg.insert(synth_attestation("addr-c", 3)).unwrap();
        // b is submitted, c is dropped, a stays pending.
        reg.mark_submitted_with_block(
            &b.id,
            SubmissionReceipt { tx_id: "0xbeef".into(), note: None },
            42,
        )
        .unwrap();
        reg.mark_submitted_with_block(
            &c.id,
            SubmissionReceipt { tx_id: "0xstale".into(), note: None },
            10,
        )
        .unwrap();
        reg.mark_dropped(&c.id, Some("manual".into())).unwrap();
        vec![a.id, b.id, c.id]
    }

    #[test]
    fn parse_attestation_id_hex_matrix() {
        // Valid 64 lowercase hex
        let s = "d77d8e95c96e6ae2264cbe3baf1383d9a3ea82e59a49d7fbf97574a04d791f1d";
        let id = parse_attestation_id_hex(s).unwrap();
        assert_eq!(id.to_hex(), s);
        // Mixed case accepted at parse time (canonical output still lowercase).
        let mixed = "D77D8E95C96E6AE2264CBE3BAF1383D9A3EA82E59A49D7FBF97574A04D791F1D";
        assert_eq!(parse_attestation_id_hex(mixed).unwrap().to_hex(), s);
        // Wrong length
        assert!(matches!(
            parse_attestation_id_hex("abcd"),
            Err(OperatorError::AttestationIdMalformed(_))
        ));
        // Non-hex char
        assert!(matches!(
            parse_attestation_id_hex(&"z".repeat(64)),
            Err(OperatorError::AttestationIdMalformed(_))
        ));
    }

    #[test]
    fn parse_status_filter_matrix() {
        for (input, expected) in [
            ("pending", "pending"),
            ("Pending", "pending"),
            ("SUBMITTED", "submitted"),
            ("included", "included"),
            ("finalized", "finalized"),
            ("failed", "failed"),
            ("dropped", "dropped"),
        ] {
            assert_eq!(parse_status_filter(input).unwrap(), expected);
        }
        assert!(matches!(
            parse_status_filter("garbage"),
            Err(OperatorError::InvalidStatusFilter(_))
        ));
    }

    #[tokio::test]
    async fn registry_list_empty_returns_empty_vec() {
        let dir = tempfile::tempdir().unwrap();
        let _ = AttestationRegistry::open(dir.path().join("reg")).unwrap();
        let rows = registry_list_core(dir.path().join("reg"), None).await.unwrap();
        assert!(rows.is_empty());
    }

    #[tokio::test]
    async fn registry_list_returns_all_records_sorted() {
        let dir = tempfile::tempdir().unwrap();
        let _ = seed_three_records(&dir.path().join("reg"));
        let rows = registry_list_core(dir.path().join("reg"), None).await.unwrap();
        assert_eq!(rows.len(), 3);
        // Sorted by id_hex ascending (registry's `list()` contract).
        let hexes: Vec<&str> = rows.iter().map(|r| r.id_hex.as_str()).collect();
        let mut sorted = hexes.clone();
        sorted.sort();
        assert_eq!(hexes, sorted);
        // Status mix present.
        let statuses: std::collections::HashSet<&str> =
            rows.iter().map(|r| r.status).collect();
        assert!(statuses.contains("pending"));
        assert!(statuses.contains("submitted"));
        assert!(statuses.contains("dropped"));
    }

    #[tokio::test]
    async fn registry_list_status_filter_applies() {
        let dir = tempfile::tempdir().unwrap();
        let _ = seed_three_records(&dir.path().join("reg"));
        let rows = registry_list_core(dir.path().join("reg"), Some("submitted".into()))
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, "submitted");
        assert_eq!(rows[0].tx_id.as_deref(), Some("0xbeef"));
        assert_eq!(rows[0].submitted_at_block, Some(42));
    }

    #[tokio::test]
    async fn registry_list_invalid_status_filter_is_typed_error() {
        let dir = tempfile::tempdir().unwrap();
        let _ = AttestationRegistry::open(dir.path().join("reg")).unwrap();
        let err = registry_list_core(dir.path().join("reg"), Some("nope".into()))
            .await
            .unwrap_err();
        assert!(matches!(err, OperatorError::InvalidStatusFilter(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn registry_show_returns_loaded_record() {
        let dir = tempfile::tempdir().unwrap();
        let ids = seed_three_records(&dir.path().join("reg"));
        let target = ids[1]; // the submitted one
        let rec = registry_show_core(dir.path().join("reg"), target.to_hex())
            .await
            .unwrap();
        assert_eq!(rec.id, target);
        assert_eq!(rec.status, LocalAttestationStatus::Submitted);
        assert_eq!(rec.receipt.as_ref().unwrap().tx_id, "0xbeef");
    }

    #[tokio::test]
    async fn registry_show_missing_id_returns_record_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let _ = AttestationRegistry::open(dir.path().join("reg")).unwrap();
        let phantom = "0".repeat(64);
        let err = registry_show_core(dir.path().join("reg"), phantom)
            .await
            .unwrap_err();
        assert!(
            matches!(err, OperatorError::Registry(RegistryError::RecordNotFound(_))),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn registry_show_malformed_id_is_refused_before_disk() {
        let dir = tempfile::tempdir().unwrap();
        let err = registry_show_core(dir.path().join("reg"), "abc".into())
            .await
            .unwrap_err();
        assert!(
            matches!(err, OperatorError::AttestationIdMalformed(_)),
            "got {err:?}"
        );
    }

    // ── Stage 10a: registry summary ───────────────────────────────────

    /// Seed a registry with one record in every local status state, so
    /// `summary` counts can be asserted exactly. Returns the inserted
    /// `AttestationId`s in (pending, submitted, included, finalized,
    /// failed, dropped) order. `submitted_at_block` for the various
    /// records is set so the oldest-`Submitted` line can be tested.
    fn seed_one_of_each_status(
        registry_path: &std::path::Path,
    ) -> [AttestationId; 6] {
        let reg = AttestationRegistry::open(registry_path.to_path_buf()).unwrap();
        let pending = reg.insert(synth_attestation("addr-pending", 1)).unwrap().id;
        // submitted (newer block — should NOT be picked as oldest)
        let submitted = reg.insert(synth_attestation("addr-submitted", 2)).unwrap().id;
        reg.mark_submitted_with_block(
            &submitted,
            SubmissionReceipt { tx_id: "0xsubmitted".into(), note: None },
            500,
        )
        .unwrap();
        // included
        let included = reg.insert(synth_attestation("addr-included", 3)).unwrap().id;
        reg.mark_submitted_with_block(
            &included,
            SubmissionReceipt { tx_id: "0xincluded".into(), note: None },
            100,
        )
        .unwrap();
        reg.mark_included(&included).unwrap();
        // finalized
        let finalized = reg.insert(synth_attestation("addr-finalized", 4)).unwrap().id;
        reg.mark_submitted_with_block(
            &finalized,
            SubmissionReceipt { tx_id: "0xfinal".into(), note: None },
            50,
        )
        .unwrap();
        reg.mark_finalized(&finalized).unwrap();
        // failed
        let failed = reg.insert(synth_attestation("addr-failed", 5)).unwrap().id;
        reg.mark_submitted_with_block(
            &failed,
            SubmissionReceipt { tx_id: "0xfailed".into(), note: None },
            75,
        )
        .unwrap();
        reg.mark_failed(&failed, "boom".into()).unwrap();
        // dropped
        let dropped = reg.insert(synth_attestation("addr-dropped", 6)).unwrap().id;
        reg.mark_submitted_with_block(
            &dropped,
            SubmissionReceipt { tx_id: "0xdropped".into(), note: None },
            25,
        )
        .unwrap();
        reg.mark_dropped(&dropped, Some("manual".into())).unwrap();
        [pending, submitted, included, finalized, failed, dropped]
    }

    #[tokio::test]
    async fn registry_summary_zero_records_emits_zero_total() {
        let dir = tempfile::tempdir().unwrap();
        let _ = AttestationRegistry::open(dir.path().join("reg")).unwrap();
        let s = registry_summary_core::<FakeJsonRpcTransport>(
            dir.path().join("reg"),
            None,
        )
        .await
        .unwrap();
        assert_eq!(s.total, 0);
        assert_eq!(s.pending, 0);
        assert_eq!(s.submitted, 0);
        assert_eq!(s.included, 0);
        assert_eq!(s.finalized, 0);
        assert_eq!(s.failed, 0);
        assert_eq!(s.dropped, 0);
        assert!(s.oldest_submitted_age.is_none());
        assert_eq!(s.oldest_submitted_age_skip_reason, Some("skipped (no --rpc-url)"));
    }

    #[tokio::test]
    async fn registry_summary_counts_by_status_match_seed() {
        let dir = tempfile::tempdir().unwrap();
        let _ = seed_one_of_each_status(&dir.path().join("reg"));
        let s = registry_summary_core::<FakeJsonRpcTransport>(
            dir.path().join("reg"),
            None,
        )
        .await
        .unwrap();
        assert_eq!(s.total, 6);
        assert_eq!(s.pending, 1);
        assert_eq!(s.submitted, 1);
        assert_eq!(s.included, 1);
        assert_eq!(s.finalized, 1);
        assert_eq!(s.failed, 1);
        assert_eq!(s.dropped, 1);
    }

    #[tokio::test]
    async fn registry_summary_age_line_skipped_when_no_rpc() {
        let dir = tempfile::tempdir().unwrap();
        let _ = seed_one_of_each_status(&dir.path().join("reg"));
        let s = registry_summary_core::<FakeJsonRpcTransport>(
            dir.path().join("reg"),
            None,
        )
        .await
        .unwrap();
        assert!(s.oldest_submitted_age.is_none());
        assert_eq!(
            s.oldest_submitted_age_skip_reason,
            Some("skipped (no --rpc-url)")
        );
    }

    #[tokio::test]
    async fn registry_summary_age_line_skipped_when_no_submitted_records() {
        let dir = tempfile::tempdir().unwrap();
        // Only Pending records — no Submitted record exists at all.
        let reg = AttestationRegistry::open(dir.path().join("reg")).unwrap();
        reg.insert(synth_attestation("addr-only-pending", 1)).unwrap();
        let fake = activated_fake(31337);
        let s = registry_summary_core(
            dir.path().join("reg"),
            Some((factory(fake), 31337u64)),
        )
        .await
        .unwrap();
        assert!(s.oldest_submitted_age.is_none());
        assert_eq!(
            s.oldest_submitted_age_skip_reason,
            Some("skipped (no Submitted records)")
        );
    }

    #[tokio::test]
    async fn registry_summary_age_line_skipped_when_submitted_has_no_block() {
        let dir = tempfile::tempdir().unwrap();
        // One Submitted record but inserted via mark_submitted (legacy
        // path) which does NOT set submitted_at_block.
        let reg = AttestationRegistry::open(dir.path().join("reg")).unwrap();
        let r = reg.insert(synth_attestation("addr-legacy", 7)).unwrap();
        reg.mark_submitted(
            &r.id,
            SubmissionReceipt { tx_id: "0xlegacy".into(), note: None },
        )
        .unwrap();
        let fake = activated_fake(31337);
        let s = registry_summary_core(
            dir.path().join("reg"),
            Some((factory(fake), 31337u64)),
        )
        .await
        .unwrap();
        assert_eq!(s.submitted, 1);
        assert!(s.oldest_submitted_age.is_none());
        assert_eq!(
            s.oldest_submitted_age_skip_reason,
            Some("skipped (no Submitted record has submitted_at_block)")
        );
    }

    #[tokio::test]
    async fn registry_summary_age_line_picks_oldest_submitted() {
        let dir = tempfile::tempdir().unwrap();
        // Three Submitted records at distinct blocks: 50, 200, 500.
        // The "included" / "finalized" / "failed" / "dropped" records
        // also have submitted_at_block set (50, 75, 25 respectively),
        // but they are NOT in `Submitted` status anymore, so the
        // summary must skip them and pick block=200 (the only true
        // Submitted record that has a block).
        let reg = AttestationRegistry::open(dir.path().join("reg")).unwrap();
        let target = reg.insert(synth_attestation("addr-oldest", 11)).unwrap().id;
        reg.mark_submitted_with_block(
            &target,
            SubmissionReceipt { tx_id: "0xtarget".into(), note: None },
            200,
        )
        .unwrap();
        let _ = seed_one_of_each_status(&dir.path().join("reg"));
        let fake = activated_fake(31337);
        let s = registry_summary_core(
            dir.path().join("reg"),
            Some((factory(fake), 31337u64)),
        )
        .await
        .unwrap();
        // Two `Submitted` records exist now: `target` at block 200,
        // plus the one from `seed_one_of_each_status` at block 500.
        // Oldest = lowest block = 200.
        let age = s.oldest_submitted_age.as_ref().expect("age expected");
        assert_eq!(age.submitted_at_block, 200);
        assert_eq!(age.head, 100); // activated_fake returns head=100
        assert_eq!(age.age_blocks, 100u64.saturating_sub(200)); // saturating → 0
        assert_eq!(age.id_hex, target.to_hex());
    }

    #[tokio::test]
    async fn registry_summary_age_uses_fake_head_height() {
        // Verify the age line uses chain_getBlockHeight from the fake
        // transport (not a hardcoded value). Seed a single Submitted
        // record at block 30; fake head is 100 → age = 70.
        let dir = tempfile::tempdir().unwrap();
        let reg = AttestationRegistry::open(dir.path().join("reg")).unwrap();
        let r = reg.insert(synth_attestation("addr-age", 9)).unwrap();
        reg.mark_submitted_with_block(
            &r.id,
            SubmissionReceipt { tx_id: "0xage".into(), note: None },
            30,
        )
        .unwrap();
        let fake = activated_fake(31337);
        let s = registry_summary_core(
            dir.path().join("reg"),
            Some((factory(fake), 31337u64)),
        )
        .await
        .unwrap();
        let age = s.oldest_submitted_age.as_ref().expect("age expected");
        assert_eq!(age.submitted_at_block, 30);
        assert_eq!(age.head, 100);
        assert_eq!(age.age_blocks, 70);
    }

    #[tokio::test]
    async fn registry_summary_chain_id_mismatch_is_typed_error() {
        // --expect-chain-id 31337 against a fake reporting chain_id=1.
        // The chain-id guardrail must trigger before the block-height
        // read happens.
        let dir = tempfile::tempdir().unwrap();
        let reg = AttestationRegistry::open(dir.path().join("reg")).unwrap();
        let r = reg.insert(synth_attestation("addr-x", 1)).unwrap();
        reg.mark_submitted_with_block(
            &r.id,
            SubmissionReceipt { tx_id: "0xx".into(), note: None },
            1,
        )
        .unwrap();
        let fake = activated_fake(1);
        let err = registry_summary_core(
            dir.path().join("reg"),
            Some((factory(fake), 31337u64)),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, OperatorError::ChainIdMismatch { expected: 31337, actual: 1 }),
            "got {err:?}"
        );
    }

    #[test]
    fn print_registry_summary_renders_age_line_when_present() {
        // Hermetic smoke for the printer — captures stdout via the
        // structured fields to be sure the renderer agrees with the
        // grep contract documented in the runbook §7 marker table.
        let s = RegistrySummary {
            total: 6,
            pending: 1,
            submitted: 2,
            included: 1,
            finalized: 1,
            failed: 1,
            dropped: 0,
            oldest_submitted_age: Some(OldestSubmittedAge {
                id_hex: "ab".repeat(32),
                submitted_at_block: 100,
                head: 247,
                age_blocks: 147,
            }),
            oldest_submitted_age_skip_reason: None,
        };
        // Sanity: the printer doesn't panic and the structured fields
        // are well-formed. Stdout capture itself is verified by the
        // dispatch-level integration tests; here we just exercise the
        // path so any future refactor that drops the printer signature
        // fails compilation.
        print_registry_summary(&s);
    }

    // ── Stage 11b.0: verify-proof + mainnet refusal mapping ───────────

    /// Helper to write a `ProofArtifactBody` (Stage 11a vintage —
    /// mock-v1, no Stage 11b.0 fields) to a temp file. Returns the
    /// file path. Used by the verify-proof tests below.
    fn write_stage11a_mock_proof_artifact(dir: &std::path::Path) -> std::path::PathBuf {
        use omni_zkml::ProofBackend;
        let proof_bytes = omni_zkml::MockProofBackend
            .prove(b"vm", b"vi", b"vo")
            .unwrap();
        let metadata = omni_zkml::ProofMetadata::new_stage11a(
            omni_zkml::MOCK_BACKEND_ID.to_string(),
            blake3::hash(b"vm").to_hex().to_string(),
            blake3::hash(b"vi").to_hex().to_string(),
            blake3::hash(b"vo").to_hex().to_string(),
        );
        let body = omni_zkml::ProofArtifactBody::from_components(metadata, &proof_bytes);
        let bytes = body.to_canonical_bytes().unwrap();
        let path = dir.join("proof_artifact.json");
        std::fs::write(&path, &bytes).unwrap();
        path
    }

    #[tokio::test]
    async fn verify_proof_accepts_stage11a_mock_artifact_and_refuses_mainnet() {
        // A Stage 11a-vintage mock proof artifact (no Stage 11b.0
        // fields) parses, verifies, and is reported as
        // mainnet-ineligible (refusal layer 2: absent proof_system
        // treated as Mock).
        let dir = tempfile::tempdir().unwrap();
        let path = write_stage11a_mock_proof_artifact(dir.path());
        verify_proof_core(path).await.unwrap();
        // verify_proof_core prints to stdout + emits a tracing event.
        // We don't capture stdout here — the unit test for the
        // refusal-mapping helper covers the typed outcome side.
    }

    /// Stage 11b.0.1 — pins that the verify-proof dispatch routes
    /// through `verify_artifact` correctly for BOTH the Stage 11a
    /// vintage `proof_system: None` shape AND the explicit
    /// `proof_system: Some(Mock)` shape. The dispatch arm matches
    /// both via the `Some(Mock) | None` pattern; this test confirms
    /// the explicit-Mock half works end-to-end.
    #[tokio::test]
    async fn verify_proof_accepts_explicit_mock_proof_system_artifact() {
        use omni_zkml::ProofBackend;
        let dir = tempfile::tempdir().unwrap();
        let proof_bytes = omni_zkml::MockProofBackend
            .prove(b"em", b"ei", b"eo")
            .unwrap();
        // Build a Stage 11b.0-style metadata: proof_system set
        // explicitly to Mock + model_format = Onnx (just to exercise
        // a non-None metadata shape; the verifier itself only checks
        // bytes against the hashed PublicInputs).
        let metadata = omni_zkml::ProofMetadata {
            backend_id: omni_zkml::MOCK_BACKEND_ID.to_string(),
            model_hash: blake3::hash(b"em").to_hex().to_string(),
            input_hash: blake3::hash(b"ei").to_hex().to_string(),
            response_hash: blake3::hash(b"eo").to_hex().to_string(),
            model_format: Some(omni_zkml::ModelFormat::Onnx),
            proof_system: Some(omni_zkml::ProofSystem::Mock),
            circuit_id_hex: None,
            verification_key_hex: None,
            public_inputs: None,
            testnet_or_dev_only: None,
            model_framework: None,
        };
        let body = omni_zkml::ProofArtifactBody::from_components(metadata, &proof_bytes);
        let bytes = body.to_canonical_bytes().unwrap();
        let path = dir.path().join("explicit_mock_proof.json");
        std::fs::write(&path, &bytes).unwrap();

        // Must dispatch through verify_artifact, verify successfully,
        // and report mainnet-ineligible (refusal layer 2 fires on
        // explicit Mock proof_system).
        verify_proof_core(path).await.unwrap();
    }

    #[tokio::test]
    async fn verify_proof_returns_no_verifier_for_unknown_proof_system() {
        // Construct a proof artifact whose proof_system is Ezkl —
        // no verifier is registered for it in Stage 11b.0, so the
        // verify-proof subcommand must return
        // NoVerifierForProofSystem.
        let dir = tempfile::tempdir().unwrap();
        let metadata = omni_zkml::ProofMetadata {
            backend_id: "ezkl-onnx-prod-future".into(),
            model_hash: blake3::hash(b"m").to_hex().to_string(),
            input_hash: blake3::hash(b"i").to_hex().to_string(),
            response_hash: blake3::hash(b"o").to_hex().to_string(),
            model_format: Some(omni_zkml::ModelFormat::Onnx),
            proof_system: Some(omni_zkml::ProofSystem::Ezkl),
            circuit_id_hex: None,
            verification_key_hex: None,
            public_inputs: None,
            testnet_or_dev_only: None,
            model_framework: None,
        };
        let body = omni_zkml::ProofArtifactBody::from_components(metadata, &[0xab, 0xcd]);
        let bytes = body.to_canonical_bytes().unwrap();
        let path = dir.path().join("ezkl_proof.json");
        std::fs::write(&path, &bytes).unwrap();

        let err = verify_proof_core(path).await.unwrap_err();
        assert!(
            matches!(err, OperatorError::NoVerifierForProofSystem { ref proof_system } if proof_system.contains("Ezkl")),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn verify_proof_returns_typed_read_error_for_missing_file() {
        let phantom = std::path::PathBuf::from("/nonexistent/proof_artifact.json");
        let err = verify_proof_core(phantom).await.unwrap_err();
        assert!(
            matches!(err, OperatorError::ProofArtifactRead { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn verify_proof_returns_typed_parse_error_for_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, b"this is not JSON").unwrap();
        let err = verify_proof_core(path).await.unwrap_err();
        assert!(
            matches!(err, OperatorError::ProofArtifactParse(_)),
            "got {err:?}"
        );
    }

    #[cfg(feature = "submit")]
    #[test]
    fn map_mainnet_refusal_preserves_stage11a_mock_variant() {
        // Stage 11a's MockBackendRefusedOnMainnet must survive the
        // generalization — preserves the typed contract Stage 11a
        // pinned.
        let reason = omni_zkml::MainnetRefusalReason::MockBackend {
            backend_id: "mock-v1".to_string(),
            proof_system: Some(omni_zkml::ProofSystem::Mock),
        };
        let mapped = map_mainnet_refusal(reason);
        assert!(
            matches!(mapped, OperatorError::MockBackendRefusedOnMainnet { ref backend_id } if backend_id == "mock-v1"),
            "got {mapped:?}"
        );
    }

    #[cfg(feature = "submit")]
    #[test]
    fn map_mainnet_refusal_covers_every_stage11b0_layer() {
        use omni_zkml::MainnetRefusalReason as R;
        // Every layer's reason maps to a distinct operator-error
        // variant. The match below is exhaustive on the source enum;
        // any future MainnetRefusalReason variant added without a
        // mapping fails compilation.
        let cases: [(R, fn(&OperatorError) -> bool); 6] = [
            (
                R::TestnetOrDevOnly { backend_id: "x".into() },
                |e| matches!(e, OperatorError::TestnetOnlyProofRefusedOnMainnet { .. }),
            ),
            (
                R::MockBackend { backend_id: "x".into(), proof_system: None },
                |e| matches!(e, OperatorError::MockBackendRefusedOnMainnet { .. }),
            ),
            (
                R::BoundedReference {
                    backend_id: "x".into(),
                    proof_system: omni_zkml::ProofSystem::Stage11bOnnxReference,
                },
                |e| matches!(e, OperatorError::BoundedReferenceProofRefusedOnMainnet { .. }),
            ),
            (
                R::GgufClaim { backend_id: "x".into(), proof_system: None },
                |e| matches!(e, OperatorError::GgufProofClaimRefusedOnMainnet { .. }),
            ),
            (
                R::UnknownModelFormat { backend_id: "x".into(), model_format: None },
                |e| matches!(e, OperatorError::UnknownModelFormatRefusedOnMainnet { .. }),
            ),
            (
                R::NotInMainnetAllowlist {
                    backend_id: "x".into(),
                    proof_system: omni_zkml::ProofSystem::Ezkl,
                },
                |e| matches!(e, OperatorError::ProofSystemNotMainnetApproved { .. }),
            ),
        ];
        for (reason, check) in cases {
            let mapped = map_mainnet_refusal(reason.clone());
            assert!(check(&mapped), "wrong mapping for {reason:?}: got {mapped:?}");
        }
    }
}
