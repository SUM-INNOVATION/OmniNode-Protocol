//! Issue #84 — operator CLI read surface for the InferenceSettlement
//! subprotocol.
//!
//! Feature-gated on `settlement-read`. Four subcommands consuming the
//! merged read APIs from #83:
//!
//! - `operator settlement status` — chain-side snapshot of the three
//!   settlement gates plus current head.
//! - `operator settlement session <session_id> [--json]` — composed
//!   multi-verifier view including consistency groups and per-verifier
//!   bond summaries when the session's mode requires them.
//! - `operator settlement claimable <session_id> --verifier <addr>` —
//!   chain-side pre-check for a specific verifier.
//! - `operator settlement verifier <addr>` — verifier registry entry.
//!
//! ## Posture
//!
//! - **Read-only.** No writes, no signing, no claim submission, no
//!   session admin. The `SumChainClient` used here is constructed with
//!   a zero seed as a deliberate sentinel — this module never invokes
//!   any signing-adjacent path.
//! - **Multi-gate dormancy rendered explicitly.** Every dormant
//!   branch surfaces the observed value + head verbatim and exits
//!   non-zero. Empty active reads (`session found=false`, empty claim
//!   list) are distinct from dormant and exit zero.
//! - **Testable I/O.** Every subcommand takes a `&mut dyn Write` so
//!   hermetic tests can capture stdout without subprocess spawn.

use std::collections::BTreeSet;
use std::io::Write;

use anyhow::Result;
use clap::{Args, Subcommand};
use serde_json::json;

use omni_sumchain::dto::BlockFinality;
use omni_sumchain::settlement::view::{
    BondState, ClaimState, DisputeState, PerVerifierView, SessionLifecycle,
    SettlementSessionView,
};
use omni_sumchain::settlement::wire::VerifierRegistryRaw;
use omni_sumchain::settlement::SettlementReadError;
use omni_sumchain::{JsonRpcTransport, SumChainClient};

// ── Issue #85 — observability markers ────────────────────────────────────────
//
// Stable tracing markers instrumented around the existing read-only
// operator settlement commands from #84. NO new subcommand, NO write
// path, NO signing surface — pure instrumentation of surfaces that
// already exist.
//
// All markers appear as an `event="<name>"` structured field on a
// `tracing` event. The constant names below are the ONLY grep targets
// for the marker taxonomy; consumers of this module must not
// hard-code the string values.

pub(crate) mod markers {
    /// Emitted at the start of every `operator settlement …` subcommand
    /// dispatch. Carries `subcommand`, and (when applicable)
    /// `session_id` / `address` fields.
    pub(crate) const QUERY: &str = "settlement_query";

    /// Emitted on the successful return path of every subcommand,
    /// including active-empty reads (`session found=false`,
    /// `verifier found=false`). Carries the same context fields as
    /// `QUERY`.
    pub(crate) const QUERY_OK: &str = "settlement_query_ok";

    /// Emitted when a subcommand encountered a locally-enforced
    /// dormant gate. Carries `gate` (chain-param name), `observed`
    /// (Option<u64> as `Some(N)` or `None`), and `head` alongside
    /// the subcommand context.
    pub(crate) const DORMANT: &str = "settlement_dormant";

    /// Emitted when a subcommand produced a normalized view that
    /// requires an additional chain-side gate to be active OR
    /// requires the caller to have fetched additional DTOs. Carries
    /// `gate` (the missing one), `session_id`, and `reason`.
    pub(crate) const VIEW_INCOMPLETE: &str = "settlement_view_incomplete";

    /// Emitted on RPC transport failure or wire-parse failure — the
    /// two `SettlementReadError` variants that are neither dormant
    /// nor view-incomplete. Carries `category` (`"chain_rpc"` or
    /// `"chain_response_malformed"`) and `error`.
    pub(crate) const QUERY_FAILED: &str = "settlement_query_failed";
}

fn emit_query_start(
    subcommand: &str,
    session_id: Option<&str>,
    address: Option<&str>,
) {
    tracing::info!(
        event = markers::QUERY,
        subcommand = subcommand,
        session_id = session_id.unwrap_or(""),
        address = address.unwrap_or(""),
        "settlement query start"
    );
}

fn emit_query_ok(
    subcommand: &str,
    session_id: Option<&str>,
    address: Option<&str>,
) {
    tracing::info!(
        event = markers::QUERY_OK,
        subcommand = subcommand,
        session_id = session_id.unwrap_or(""),
        address = address.unwrap_or(""),
        "settlement query ok"
    );
}

/// Inspect the [`SettlementReadError`] variant and emit the matching
/// terminal marker. Called by [`err_to_anyhow_with_marker`] on every
/// error propagation site in the four `run_*` functions.
fn emit_failure_marker(
    err: &SettlementReadError,
    subcommand: &str,
    session_id: Option<&str>,
    address: Option<&str>,
) {
    match err {
        SettlementReadError::Dormant { gate, observed, head } => {
            let observed_display = match observed {
                None => "None".to_string(),
                Some(n) => format!("Some({n})"),
            };
            tracing::warn!(
                event = markers::DORMANT,
                subcommand = subcommand,
                session_id = session_id.unwrap_or(""),
                address = address.unwrap_or(""),
                gate = gate.param_field_name(),
                observed = %observed_display,
                head = head,
                "settlement RPC gate dormant"
            );
        }
        SettlementReadError::ViewIncomplete {
            missing_gate,
            session_id: sid,
            reason,
        } => {
            tracing::warn!(
                event = markers::VIEW_INCOMPLETE,
                subcommand = subcommand,
                session_id = %sid,
                address = address.unwrap_or(""),
                gate = missing_gate.param_field_name(),
                reason = %reason,
                "settlement view incomplete"
            );
        }
        SettlementReadError::Rpc(inner) => {
            tracing::error!(
                event = markers::QUERY_FAILED,
                subcommand = subcommand,
                session_id = session_id.unwrap_or(""),
                address = address.unwrap_or(""),
                category = "chain_rpc",
                error = %inner,
                "settlement RPC failure"
            );
        }
        SettlementReadError::WireParse(msg) => {
            tracing::error!(
                event = markers::QUERY_FAILED,
                subcommand = subcommand,
                session_id = session_id.unwrap_or(""),
                address = address.unwrap_or(""),
                category = "chain_response_malformed",
                error = %msg,
                "settlement wire parse failure"
            );
        }
    }
}

/// Emit the failure marker AND convert the typed error to `anyhow`.
/// Use this wherever the existing `run_*` functions call
/// `settlement_read_error_to_anyhow`.
fn err_to_anyhow_with_marker(
    err: SettlementReadError,
    subcommand: &str,
    session_id: Option<&str>,
    address: Option<&str>,
) -> anyhow::Error {
    emit_failure_marker(&err, subcommand, session_id, address);
    settlement_read_error_to_anyhow(err)
}

/// Emit `settlement_query_failed` for a NON-typed direct RPC failure
/// (`chain_getChainParams`, `chain_getBlockHeight`,
/// `sum_listInferenceAttestations`). These paths don't produce a
/// [`SettlementReadError`] but still need the same marker + field
/// contract as `err_to_anyhow_with_marker` — including the
/// `session_id=""` / `address=""` empty-when-absent convention.
///
/// Called from `run_status` and `run_session`'s pre-compose RPCs.
fn emit_query_failed(
    subcommand: &str,
    session_id: Option<&str>,
    address: Option<&str>,
    category: &str,
    error: &dyn std::fmt::Display,
) {
    tracing::error!(
        event = markers::QUERY_FAILED,
        subcommand = subcommand,
        session_id = session_id.unwrap_or(""),
        address = address.unwrap_or(""),
        category = category,
        error = %error,
        "settlement RPC failure"
    );
}

// ── Public arg types ─────────────────────────────────────────────────────────

#[derive(Args, Debug, Clone)]
pub struct SettlementArgs {
    /// SUM Chain JSON-RPC endpoint. Falls back to
    /// $OMNINODE_SUMCHAIN_RPC_URL when omitted.
    #[arg(long)]
    pub rpc_url: Option<String>,

    #[command(subcommand)]
    pub cmd: SettlementCmd,
}

#[derive(Subcommand, Debug, Clone)]
pub enum SettlementCmd {
    /// Print the chain-side snapshot of the three InferenceSettlement
    /// gates and the current head. Never errors on dormancy — dormancy
    /// is the answer this subcommand exists to report.
    Status,

    /// Render the composed multi-verifier view for a session,
    /// including consistency groups and per-verifier bond summaries
    /// when the session's mode requires them.
    Session(SessionArgs),

    /// Chain-side claimable-reward pre-check for a specific verifier.
    /// Read-only; does not submit a claim.
    Claimable(ClaimableArgs),

    /// Read the verifier registry entry for the given chain address.
    /// Requires the bonding gate to be active.
    Verifier(VerifierArgs),

    /// Issue #87 — verifier self-claim submission. Boxed to keep the
    /// enum's largest variant size unchanged. Feature-gated on
    /// `settlement-submit` so `settlement-read` builds don't compile
    /// any of the claim code path.
    #[cfg(feature = "settlement-submit")]
    Claim(Box<ClaimArgs>),

    /// Issue #100 — operator verifier on-chain registration. Builds
    /// the chain's no-key `omninode_buildRegisterVerifier` tx, signs
    /// the returned hash with the `OMNINODE_VERIFIER_SEED_HEX` key,
    /// and submits via `sum_sendRawTransaction`. Boxed for enum-size
    /// parity with `Claim`. Feature-gated on `settlement-submit` —
    /// reuses the same builder-envelope / signer / submit helpers as
    /// the claim path.
    #[cfg(feature = "settlement-submit")]
    RegisterVerifier(Box<RegisterVerifierArgs>),

    /// Issue #81 — dispute open + resolve write path. Boxed for
    /// enum-size parity with `Claim`. Feature-gated on
    /// `settlement-dispute`; `settlement-read` / `settlement-submit`
    /// builds don't compile any of the dispute code path.
    #[cfg(feature = "settlement-dispute")]
    Dispute(Box<DisputeArgs>),
}

#[derive(Args, Debug, Clone)]
pub struct SessionArgs {
    /// Session identifier.
    pub session_id: String,

    /// Emit JSON instead of the default line-oriented human rendering.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug, Clone)]
pub struct ClaimableArgs {
    /// Session identifier.
    pub session_id: String,

    /// Verifier chain address.
    #[arg(long)]
    pub verifier: String,
}

#[derive(Args, Debug, Clone)]
pub struct VerifierArgs {
    /// Verifier chain address.
    pub address: String,
}

/// Issue #87 — argument surface for `operator settlement claim`.
#[cfg(feature = "settlement-submit")]
#[derive(Args, Debug, Clone)]
pub struct ClaimArgs {
    /// Session identifier for the attestation being claimed.
    pub session_id: String,

    /// Optional explicit-safety verifier address. When provided, MUST
    /// equal the address derived from OMNINODE_VERIFIER_SEED_HEX;
    /// otherwise the authority gate refuses. Provides a scripted-
    /// invocation double-check.
    #[arg(long)]
    pub verifier: Option<String>,

    /// Optional fee override. When absent, chain-side builder applies
    /// its default. When present, passed through to the builder
    /// request and asserted-equal on the returned envelope.
    #[arg(long)]
    pub fee: Option<u128>,

    /// Run prechecks + builder RPC + envelope + decoded-tx
    /// verification and STOP. Never loads the signing seed. Never
    /// invokes `sum_sendRawTransaction`. Used for operator dry-runs
    /// and CI checks.
    #[arg(long)]
    pub dry_run: bool,
}

/// Issue #100 — argument surface for
/// `operator settlement register-verifier`.
#[cfg(feature = "settlement-submit")]
#[derive(Args, Debug, Clone)]
pub struct RegisterVerifierArgs {
    /// Chain-id guardrail. The command REFUSES before building the tx
    /// unless `chain_getChainParams.chain_id` equals this value —
    /// preventing an accidental registration against the wrong chain.
    #[arg(long)]
    pub expect_chain_id: u64,

    /// Bond amount (native Koppa) to lock at registration. REQUIRED —
    /// clap rejects the command when `--bond` is omitted. Passed
    /// straight to the chain builder. The chain executor requires a
    /// positive bond (`Failed(365)` on zero), so the handler ALSO
    /// rejects `--bond 0` locally — before any preflight RPC or seed
    /// access — rather than building a guaranteed-doomed, fee-paying
    /// transaction.
    #[arg(long)]
    pub bond: u128,

    /// Optional fee override. When absent, the chain-side builder
    /// applies its default. When present, it is passed to the builder
    /// request and asserted-equal on the returned envelope.
    #[arg(long)]
    pub fee: Option<u128>,

    /// Run preflight + builder RPC + envelope + decoded-tx
    /// verification and STOP. Dry-run DERIVES the operator address from
    /// `OMNINODE_VERIFIER_SEED_HEX` (needed for the builder request and
    /// the envelope's `from` assertion) but never retains the secret,
    /// never signs, and never invokes `sum_sendRawTransaction`. Used
    /// for operator dry-runs and CI checks.
    #[arg(long)]
    pub dry_run: bool,
}

// ── Issue #81 — Dispute argument surface ─────────────────────────────────────

/// `operator settlement dispute { open | resolve }` argument tree.
#[cfg(feature = "settlement-dispute")]
#[derive(Args, Debug, Clone)]
pub struct DisputeArgs {
    #[command(subcommand)]
    pub cmd: DisputeCmd,
}

#[cfg(feature = "settlement-dispute")]
#[derive(Subcommand, Debug, Clone)]
pub enum DisputeCmd {
    /// Open a dispute against a verifier's attestation. Signer must
    /// be the session funder; local prechecks enforce this before the
    /// builder RPC is called.
    Open(Box<OpenDisputeArgs>),

    /// Resolve an open dispute with a validator-quorum-signed approval
    /// bundle. The submitting operator is fee payer only; authority
    /// comes from the approvals payload. Chain evaluates quorum.
    Resolve(Box<ResolveDisputeArgs>),
}

#[cfg(feature = "settlement-dispute")]
#[derive(Args, Debug, Clone)]
pub struct OpenDisputeArgs {
    /// Session identifier of the disputed attestation.
    pub session_id: String,
    /// Target verifier chain address (base58).
    #[arg(long)]
    pub verifier: String,
    /// 32-byte hex commitment to off-chain evidence. `0x` prefix
    /// optional.
    #[arg(long)]
    pub evidence: String,
    /// Optional fee override; passed to builder if provided.
    #[arg(long)]
    pub fee: Option<u128>,
    /// Verify prechecks + builder + decoded tx and STOP before
    /// signing/submitting.
    #[arg(long)]
    pub dry_run: bool,
}

#[cfg(feature = "settlement-dispute")]
#[derive(Args, Debug, Clone)]
pub struct ResolveDisputeArgs {
    /// Session identifier of the target dispute.
    pub session_id: String,
    /// Target verifier chain address (base58) — the one the dispute
    /// is against.
    #[arg(long)]
    pub verifier: String,
    /// `true` = allow claim to proceed; `false` = deny claim.
    #[arg(long)]
    pub allow_claim: bool,
    /// Path to a JSON file of validator approvals. Format:
    ///
    /// ```json
    /// [{"pubkey": "0x<64 hex>", "signature": "0x<128 hex>"}]
    /// ```
    ///
    /// `0x` prefix optional per hex field.
    #[arg(long)]
    pub approvals: std::path::PathBuf,
    /// Optional fee override.
    #[arg(long)]
    pub fee: Option<u128>,
    /// Verify prechecks + builder + decoded tx and STOP before
    /// signing/submitting.
    #[arg(long)]
    pub dry_run: bool,
}

// ── Entry point (production) ─────────────────────────────────────────────────

pub(crate) async fn dispatch(args: SettlementArgs) -> Result<()> {
    let url = resolve_rpc_url(args.rpc_url.clone())?;
    // Read-only surface — the seed is never consulted by any RPC we
    // issue. Zero seed makes any accidental signing surface loud
    // (derived address is deterministic and obviously wrong).
    let client = SumChainClient::new(url, [0u8; 32]);
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    dispatch_core(args, &client, &mut lock)
}

fn resolve_rpc_url(opt: Option<String>) -> Result<String> {
    if let Some(u) = opt {
        return Ok(u);
    }
    std::env::var("OMNINODE_SUMCHAIN_RPC_URL").map_err(|_| {
        anyhow::anyhow!(
            "--rpc-url is required or set OMNINODE_SUMCHAIN_RPC_URL"
        )
    })
}

// ── Testable core ────────────────────────────────────────────────────────────

pub(crate) fn dispatch_core<T: JsonRpcTransport>(
    args: SettlementArgs,
    client: &SumChainClient<T>,
    out: &mut dyn Write,
) -> Result<()> {
    match args.cmd {
        SettlementCmd::Status => run_status(client, out),
        SettlementCmd::Session(a) => run_session(client, &a, out),
        SettlementCmd::Claimable(a) => run_claimable(client, &a, out),
        SettlementCmd::Verifier(a) => run_verifier(client, &a, out),
        #[cfg(feature = "settlement-submit")]
        SettlementCmd::Claim(a) => {
            run_claim(client, &a, out, crate::operator::SeedSource::Env)
        }
        #[cfg(feature = "settlement-submit")]
        SettlementCmd::RegisterVerifier(a) => run_register_verifier(
            client,
            &a,
            out,
            crate::operator::SeedSource::Env,
        ),
        #[cfg(feature = "settlement-dispute")]
        SettlementCmd::Dispute(a) => match a.cmd {
            DisputeCmd::Open(inner) => {
                run_dispute_open(client, &inner, out, crate::operator::SeedSource::Env)
            }
            DisputeCmd::Resolve(inner) => {
                run_dispute_resolve(
                    client,
                    &inner,
                    out,
                    crate::operator::SeedSource::Env,
                )
            }
        },
    }
}

// ── status ───────────────────────────────────────────────────────────────────

fn run_status<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    out: &mut dyn Write,
) -> Result<()> {
    let subcommand = "status";
    emit_query_start(subcommand, None, None);

    let params = client.get_chain_params().map_err(|e| {
        let err = anyhow::anyhow!("chain_getChainParams: {e}");
        emit_query_failed(subcommand, None, None, "chain_rpc", &err);
        err
    })?;
    let head = client
        .get_block_height(BlockFinality::Latest)
        .map_err(|e| {
            let err = anyhow::anyhow!("chain_getBlockHeight: {e}");
            emit_query_failed(subcommand, None, None, "chain_rpc", &err);
            err
        })?
        .height;

    writeln!(out, "chain_id={}", params.chain_id)?;
    writeln!(out, "head={head}")?;
    for (name, observed) in [
        (
            "inference_settlement_enabled_from_height",
            params.inference_settlement_enabled_from_height,
        ),
        (
            "inference_settlement_consistency_enabled_from_height",
            params.inference_settlement_consistency_enabled_from_height,
        ),
        (
            "inference_verifier_bonding_enabled_from_height",
            params.inference_verifier_bonding_enabled_from_height,
        ),
    ] {
        writeln!(out, "{name} = {}", render_gate_state(observed, head))?;
    }

    emit_query_ok(subcommand, None, None);
    Ok(())
}

fn render_gate_state(observed: Option<u64>, head: u64) -> String {
    match observed {
        None => "Dormant (unset)".to_string(),
        Some(n) if head < n => {
            format!("Scheduled@{n} (Δ={} blocks)", n - head)
        }
        Some(n) => format!("Active (since={n})"),
    }
}

// ── session ──────────────────────────────────────────────────────────────────

fn run_session<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    args: &SessionArgs,
    out: &mut dyn Write,
) -> Result<()> {
    let subcommand = "session";
    let session_id = &args.session_id;
    emit_query_start(subcommand, Some(session_id), None);

    // Session lookup surfaces settlement-gate dormancy directly.
    let session = match client.omninode_get_inference_session(session_id) {
        Ok(Some(s)) => s,
        Ok(None) => {
            if args.json {
                writeln!(
                    out,
                    "{}",
                    serde_json::to_string(&json!({
                        "session_id": session_id,
                        "found": false,
                    }))?
                )?;
            } else {
                writeln!(out, "session_id={session_id} found=false")?;
            }
            emit_query_ok(subcommand, Some(session_id), None);
            return Ok(());
        }
        Err(e) => {
            return Err(err_to_anyhow_with_marker(
                e,
                subcommand,
                Some(session_id),
                None,
            ));
        }
    };

    let claims = client
        .omninode_get_inference_claims(session_id)
        .map_err(|e| err_to_anyhow_with_marker(e, subcommand, Some(session_id), None))?;
    let disputes = client
        .omninode_get_inference_disputes(session_id)
        .map_err(|e| err_to_anyhow_with_marker(e, subcommand, Some(session_id), None))?;

    // Attestations RPC is unconditional (no gate) and already lives
    // on SumChainClient.
    let attestations = client.list_attestations(session_id).map_err(|e| {
        let err = anyhow::anyhow!("sum_listInferenceAttestations: {e}");
        emit_query_failed(subcommand, Some(session_id), None, "chain_rpc", &err);
        err
    })?;

    let params = client.get_chain_params().map_err(|e| {
        let err = anyhow::anyhow!("chain_getChainParams: {e}");
        emit_query_failed(subcommand, Some(session_id), None, "chain_rpc", &err);
        err
    })?;
    let head = client
        .get_block_height(BlockFinality::Latest)
        .map_err(|e| {
            let err = anyhow::anyhow!("chain_getBlockHeight: {e}");
            emit_query_failed(subcommand, Some(session_id), None, "chain_rpc", &err);
            err
        })?
        .height;
    let consistency_gate_active = params
        .inference_settlement_consistency_enabled_from_height
        .is_some_and(|n| head >= n);
    let bonding_gate_active = params
        .inference_verifier_bonding_enabled_from_height
        .is_some_and(|n| head >= n);

    // Fetch consistency data when the session requires it AND the
    // gate is active. The compose guard refuses if either is missing.
    let consistency = if session.consistency_required && consistency_gate_active {
        Some(
            client
                .omninode_get_inference_consistency(session_id)
                .map_err(|e| {
                    err_to_anyhow_with_marker(e, subcommand, Some(session_id), None)
                })?,
        )
    } else {
        None
    };

    // Fetch per-verifier registry data when the session requires it
    // AND the bonding gate is active. The compose guard refuses if
    // either is missing.
    let verifier_registry = if session.bond_required && bonding_gate_active {
        let mut set: BTreeSet<String> = BTreeSet::new();
        for a in &attestations {
            set.insert(a.verifier_address.clone());
        }
        for c in &claims.claims {
            set.insert(c.verifier_address.clone());
        }
        for d in &disputes.disputes {
            set.insert(d.verifier_address.clone());
        }
        let mut entries: Vec<VerifierRegistryRaw> = Vec::new();
        for addr in set {
            if let Some(r) = client.omninode_get_verifier(&addr).map_err(|e| {
                err_to_anyhow_with_marker(e, subcommand, Some(session_id), Some(&addr))
            })? {
                entries.push(r);
            }
        }
        Some(entries)
    } else {
        None
    };

    let view = SettlementSessionView::compose(
        session,
        claims,
        disputes,
        attestations,
        consistency,
        verifier_registry,
        consistency_gate_active,
        bonding_gate_active,
    )
    .map_err(|e| err_to_anyhow_with_marker(e, subcommand, Some(session_id), None))?;

    if args.json {
        writeln!(out, "{}", serde_json::to_string_pretty(&view_to_json(&view))?)?;
    } else {
        render_session_view_human(out, &view)?;
    }

    emit_query_ok(subcommand, Some(session_id), None);
    Ok(())
}

fn render_session_view_human(
    out: &mut dyn Write,
    view: &SettlementSessionView,
) -> Result<()> {
    writeln!(out, "session_id={}", view.session_id)?;
    writeln!(
        out,
        "mode: consistency_required={} bond_required={}",
        view.mode.consistency_required, view.mode.bond_required
    )?;
    writeln!(out, "max_verifiers={}", view.max_verifiers)?;
    writeln!(
        out,
        "escrow_total={} escrow_remaining={}",
        view.escrow_total, view.escrow_remaining
    )?;
    writeln!(out, "claims_count={}", view.claims_count)?;
    writeln!(out, "lifecycle={}", render_lifecycle(&view.lifecycle))?;
    writeln!(out, "created_at_height={}", view.created_at_height)?;
    writeln!(
        out,
        "settled_at_height={}",
        opt_u64(view.settled_at_height)
    )?;
    writeln!(
        out,
        "refunded_at_height={}",
        opt_u64(view.refunded_at_height)
    )?;
    match &view.plurality_key {
        Some(k) => writeln!(
            out,
            "plurality_key=({}, {}, {}, {})",
            k.model_hash, k.manifest_root, k.response_hash, k.proof_root
        )?,
        None => writeln!(out, "plurality_key=none")?,
    }
    writeln!(out, "verifiers ({}):", view.verifiers.len())?;
    for v in &view.verifiers {
        render_per_verifier_human(out, v)?;
    }
    Ok(())
}

fn render_per_verifier_human(
    out: &mut dyn Write,
    v: &PerVerifierView,
) -> Result<()> {
    writeln!(out, "  address={}", v.verifier_address)?;
    match &v.attestation {
        Some(a) => writeln!(
            out,
            "    attestation: tx={} included_at={} finalized={}",
            a.tx_hash, a.included_at_height, a.finalized
        )?,
        None => writeln!(out, "    attestation: none")?,
    }
    match &v.digest_tuple {
        Some(d) => writeln!(
            out,
            "    digest_tuple=({}, {}, {}, {})",
            d.model_hash, d.manifest_root, d.response_hash, d.proof_root
        )?,
        None => writeln!(out, "    digest_tuple=none")?,
    }
    writeln!(out, "    claim={}", render_claim_state(&v.claim_state))?;
    writeln!(out, "    dispute={}", render_dispute_state(&v.dispute_state))?;
    match &v.bond_summary {
        Some(b) => writeln!(
            out,
            "    bond: amount={} state={} unbonding_since={} withdrawable_at={}",
            b.bond_amount,
            render_bond_state(&b.bond_state),
            opt_u64(b.unbonding_since_height),
            opt_u64(b.withdrawable_at_height),
        )?,
        None => writeln!(out, "    bond=none")?,
    }
    Ok(())
}

fn render_lifecycle(l: &SessionLifecycle) -> String {
    match l {
        SessionLifecycle::Active => "Active".into(),
        SessionLifecycle::Settled => "Settled".into(),
        SessionLifecycle::Refunded => "Refunded".into(),
        SessionLifecycle::Unknown(s) => format!("Unknown({s})"),
    }
}

fn render_bond_state(s: &BondState) -> String {
    match s {
        BondState::Bonded => "Bonded".into(),
        BondState::Unbonding => "Unbonding".into(),
        BondState::Withdrawn => "Withdrawn".into(),
        BondState::UnknownWire(s) => format!("UnknownWire({s})"),
    }
}

fn render_claim_state(c: &ClaimState) -> String {
    match c {
        ClaimState::NotSubmitted => "NotSubmitted".into(),
        ClaimState::Pending { claimed_at_height, reward_amount } => format!(
            "Pending(claimed_at={claimed_at_height}, amount={reward_amount})"
        ),
        ClaimState::Paid { paid_at_height, reward_amount } => {
            format!("Paid(paid_at={paid_at_height}, amount={reward_amount})")
        }
        ClaimState::Denied { denied_at_height, reason } => format!(
            "Denied(denied_at={}, reason={})",
            opt_u64(*denied_at_height),
            reason.as_deref().unwrap_or("none")
        ),
        ClaimState::UnknownWire(s) => format!("UnknownWire({s})"),
    }
}

fn render_dispute_state(d: &DisputeState) -> String {
    // Chain-team public terminology: the resolved variants render at
    // this display boundary as `ResolvedAllowClaim` /
    // `ResolvedDenyClaim`. The #83 internal enum names
    // (`ResolvedApproved` / `ResolvedDenied`) predate the chain-team
    // clarification; they are NOT renamed at the wire/view layer so
    // #83's public API stays stable — this display mapping is the
    // operator-facing boundary.
    match d {
        DisputeState::None => "None".into(),
        DisputeState::Open { opened_at_height } => {
            format!("Open(opened_at={opened_at_height})")
        }
        DisputeState::ResolvedApproved {
            resolved_at_height,
            approve_bps,
            deny_bps,
        } => format!(
            "ResolvedAllowClaim(resolved_at={}, approve_bps={}, deny_bps={})",
            opt_u64(*resolved_at_height),
            opt_u32(*approve_bps),
            opt_u32(*deny_bps)
        ),
        DisputeState::ResolvedDenied {
            resolved_at_height,
            approve_bps,
            deny_bps,
        } => format!(
            "ResolvedDenyClaim(resolved_at={}, approve_bps={}, deny_bps={})",
            opt_u64(*resolved_at_height),
            opt_u32(*approve_bps),
            opt_u32(*deny_bps)
        ),
        DisputeState::UnknownWire(s) => format!("UnknownWire({s})"),
    }
}

fn opt_u64(v: Option<u64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "none".to_string())
}

fn opt_u32(v: Option<u32>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "none".to_string())
}

// ── session JSON emission ────────────────────────────────────────────────────

fn view_to_json(view: &SettlementSessionView) -> serde_json::Value {
    json!({
        "session_id": view.session_id,
        "mode": {
            "consistency_required": view.mode.consistency_required,
            "bond_required": view.mode.bond_required,
        },
        "max_verifiers": view.max_verifiers,
        "escrow_total": view.escrow_total.to_string(),
        "escrow_remaining": view.escrow_remaining.to_string(),
        "claims_count": view.claims_count,
        "lifecycle": render_lifecycle(&view.lifecycle),
        "created_at_height": view.created_at_height,
        "settled_at_height": view.settled_at_height,
        "refunded_at_height": view.refunded_at_height,
        "plurality_key": view.plurality_key.as_ref().map(|k| json!({
            "model_hash": k.model_hash,
            "manifest_root": k.manifest_root,
            "response_hash": k.response_hash,
            "proof_root": k.proof_root,
        })),
        "verifiers": view.verifiers.iter().map(per_verifier_to_json).collect::<Vec<_>>(),
    })
}

fn per_verifier_to_json(v: &PerVerifierView) -> serde_json::Value {
    json!({
        "address": v.verifier_address,
        "attestation": v.attestation.as_ref().map(|a| json!({
            "tx_hash": a.tx_hash,
            "included_at_height": a.included_at_height,
            "finalized": a.finalized,
        })),
        "digest_tuple": v.digest_tuple.as_ref().map(|d| json!({
            "model_hash": d.model_hash,
            "manifest_root": d.manifest_root,
            "response_hash": d.response_hash,
            "proof_root": d.proof_root,
        })),
        "claim_state": render_claim_state(&v.claim_state),
        "dispute_state": render_dispute_state(&v.dispute_state),
        "bond_summary": v.bond_summary.as_ref().map(|b| json!({
            "bond_amount": b.bond_amount.to_string(),
            "bond_state": render_bond_state(&b.bond_state),
            "unbonding_since_height": b.unbonding_since_height,
            "withdrawable_at_height": b.withdrawable_at_height,
        })),
    })
}

// ── claimable ────────────────────────────────────────────────────────────────

fn run_claimable<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    args: &ClaimableArgs,
    out: &mut dyn Write,
) -> Result<()> {
    let subcommand = "claimable";
    let session_id = &args.session_id;
    let verifier = &args.verifier;
    emit_query_start(subcommand, Some(session_id), Some(verifier));

    let r = client
        .omninode_get_claimable_reward(session_id, verifier)
        .map_err(|e| {
            err_to_anyhow_with_marker(e, subcommand, Some(session_id), Some(verifier))
        })?;
    writeln!(out, "session_id={}", r.session_id)?;
    writeln!(out, "verifier_address={}", r.verifier_address)?;
    writeln!(out, "reward_amount={}", r.reward_amount)?;
    writeln!(
        out,
        "mature={} claim_ready_block={} blocks_until_ready={}",
        r.mature, r.claim_ready_block, r.blocks_until_ready
    )?;
    writeln!(
        out,
        "escrow_available={} cap_available={} dispute_clear={} claimable_now={}",
        r.escrow_available, r.cap_available, r.dispute_clear, r.claimable_now
    )?;

    emit_query_ok(subcommand, Some(session_id), Some(verifier));
    Ok(())
}

// ── verifier ─────────────────────────────────────────────────────────────────

fn run_verifier<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    args: &VerifierArgs,
    out: &mut dyn Write,
) -> Result<()> {
    let subcommand = "verifier";
    let address = &args.address;
    emit_query_start(subcommand, None, Some(address));

    match client.omninode_get_verifier(address).map_err(|e| {
        err_to_anyhow_with_marker(e, subcommand, None, Some(address))
    })? {
        None => {
            writeln!(out, "address={} found=false", address)?;
        }
        Some(v) => {
            writeln!(out, "verifier={}", v.verifier)?;
            writeln!(out, "bond={}", v.bond)?;
            writeln!(out, "status={}", v.status)?;
            writeln!(out, "registered_at_height={}", v.registered_at_height)?;
            if let Some(h) = v.unbonding_started_height {
                writeln!(out, "unbonding_started_height={h}")?;
            }
            if let Some(h) = v.unlock_height {
                writeln!(out, "unlock_height={h}")?;
            }
        }
    }

    emit_query_ok(subcommand, None, Some(address));
    Ok(())
}

// ── Error rendering ──────────────────────────────────────────────────────────

/// Convert a [`SettlementReadError`] into an `anyhow::Error` whose
/// display carries the observed value + head verbatim. Callers
/// propagate via `?`; the top-level `main.rs` prints the anyhow error
/// message to stderr and exits non-zero.
///
/// **Never renders zeros** for a dormant gate. Never panics.
fn settlement_read_error_to_anyhow(err: SettlementReadError) -> anyhow::Error {
    match err {
        SettlementReadError::Dormant { gate, observed, head } => anyhow::anyhow!(
            "settlement RPC gate '{}' dormant: observed={} head={head}",
            gate.param_field_name(),
            opt_u64(observed),
        ),
        SettlementReadError::ViewIncomplete {
            missing_gate,
            session_id,
            reason,
        } => anyhow::anyhow!(
            "settlement view for session '{session_id}' requires gate '{}' to be \
             active: {reason}",
            missing_gate.param_field_name(),
        ),
        SettlementReadError::Rpc(inner) => anyhow::anyhow!("chain RPC failure: {inner}"),
        SettlementReadError::WireParse(msg) => {
            anyhow::anyhow!("settlement wire parse failure: {msg}")
        }
    }
}

// ── Issue #87 — claim markers + run_claim (settlement-submit-gated) ─────────

#[cfg(feature = "settlement-submit")]
pub(crate) mod claim_markers {
    pub(crate) const CLAIM_READY: &str = "settlement_claim_ready";
    pub(crate) const CLAIM_REFUSED_DORMANCY: &str = "settlement_claim_refused_dormancy";
    pub(crate) const CLAIM_REFUSED_MATURITY: &str = "settlement_claim_refused_maturity";
    pub(crate) const CLAIM_REFUSED_AUTHORITY: &str = "settlement_claim_refused_authority";
    pub(crate) const CLAIM_REFUSED_BOND_PRECHECK: &str =
        "settlement_claim_refused_bond_precheck";
    pub(crate) const CLAIM_SUBMITTED: &str = "settlement_claim_submitted";
    pub(crate) const CLAIM_FAILED: &str = "settlement_claim_failed";
}

#[cfg(feature = "settlement-submit")]
fn run_claim<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    args: &ClaimArgs,
    out: &mut dyn Write,
    seed_source: crate::operator::SeedSource,
) -> Result<()> {
    use crate::settlement_signer::{
        BondPrecheckOutcome, ClaimSignerError, ClaimSignerIdentity,
    };
    use omni_sumchain::dto::BlockFinality;
    use omni_sumchain::settlement_submit::{
        decode_unsigned_tx, sign_and_submit, verify_builder_envelope,
        verify_decoded_transaction, BuildClaimRewardRequest,
        SettlementSubmitError,
    };

    let session_id = args.session_id.as_str();
    tracing::info!(
        event = "settlement_query",
        subcommand = "claim",
        session_id = session_id,
        "settlement claim start"
    );

    // ── Step 1: derive identity (settlement-read; no seed retention) ──
    let identity = match ClaimSignerIdentity::resolve(seed_source.clone()) {
        Ok(i) => i,
        Err(e) => {
            let err = anyhow::anyhow!("claim signer identity resolve: {e}");
            tracing::error!(
                event = claim_markers::CLAIM_FAILED,
                subcommand = "claim",
                session_id = session_id,
                category = "seed_missing_or_malformed",
                error = %err,
                "claim signer identity resolve failed"
            );
            return Err(err);
        }
    };
    let derived = identity.address().to_string();

    // ── Step 2: params + head + settlement gate ──────────────────────
    let params = client.get_chain_params().map_err(|e| {
        let err = anyhow::anyhow!("chain_getChainParams: {e}");
        tracing::error!(
            event = claim_markers::CLAIM_FAILED,
            subcommand = "claim",
            session_id = session_id,
            verifier = %derived,
            category = "chain_rpc",
            error = %err,
            "chain_getChainParams failed during claim precheck"
        );
        err
    })?;
    let head = client
        .get_block_height(BlockFinality::Latest)
        .map_err(|e| {
            let err = anyhow::anyhow!("chain_getBlockHeight: {e}");
            tracing::error!(
                event = claim_markers::CLAIM_FAILED,
                subcommand = "claim",
                session_id = session_id,
                verifier = %derived,
                category = "chain_rpc",
                error = %err,
                "chain_getBlockHeight failed during claim precheck"
            );
            err
        })?
        .height;
    let observed = params.inference_settlement_enabled_from_height;
    let gate_active = observed.is_some_and(|n| head >= n);
    if !gate_active {
        let observed_display = match observed {
            None => "None".to_string(),
            Some(n) => format!("Some({n})"),
        };
        tracing::warn!(
            event = claim_markers::CLAIM_REFUSED_DORMANCY,
            subcommand = "claim",
            session_id = session_id,
            verifier = %derived,
            gate = "inference_settlement_enabled_from_height",
            observed = %observed_display,
            head = head,
            "claim refused: settlement gate dormant"
        );
        return Err(anyhow::anyhow!(SettlementSubmitError::Dormant {
            observed,
            head,
        }));
    }

    // ── Step 3: fetch attestation for (session, derived) ─────────────
    let attestation_opt =
        client.get_attestation(session_id, &derived).map_err(|e| {
            let err = anyhow::anyhow!("sum_getInferenceAttestation: {e}");
            tracing::error!(
                event = claim_markers::CLAIM_FAILED,
                subcommand = "claim",
                session_id = session_id,
                verifier = %derived,
                category = "chain_rpc",
                error = %err,
                "attestation lookup failed"
            );
            err
        })?;
    let attestation = match attestation_opt {
        Some(a) => a,
        None => {
            let err_msg = format!(
                "no attestation found for (session_id={session_id}, verifier={derived})"
            );
            tracing::error!(
                event = claim_markers::CLAIM_FAILED,
                subcommand = "claim",
                session_id = session_id,
                verifier = %derived,
                category = "attestation_not_found",
                error = %err_msg,
                "claim refused: attestation not found"
            );
            return Err(anyhow::anyhow!(err_msg));
        }
    };

    // ── Step 4: authority ────────────────────────────────────────────
    // Uses #80's `ClaimSignerIdentity::verify_matches` — pure string
    // compare against the derived-only identity.
    if let Err(e) = identity.verify_matches(&attestation.verifier_address) {
        let msg = e.to_string();
        tracing::warn!(
            event = claim_markers::CLAIM_REFUSED_AUTHORITY,
            subcommand = "claim",
            session_id = session_id,
            verifier = %derived,
            attestation_verifier = %attestation.verifier_address,
            "claim refused: authority mismatch (attestation)"
        );
        return Err(anyhow::anyhow!(msg));
    }
    if let Some(explicit) = args.verifier.as_deref() {
        if explicit != derived {
            let msg = format!(
                "--verifier={explicit} does not match derived signer address={derived}"
            );
            tracing::warn!(
                event = claim_markers::CLAIM_REFUSED_AUTHORITY,
                subcommand = "claim",
                session_id = session_id,
                verifier = %derived,
                explicit_verifier = %explicit,
                "claim refused: authority mismatch (--verifier flag)"
            );
            return Err(anyhow::anyhow!(msg));
        }
    }

    // ── Step 5: maturity ─────────────────────────────────────────────
    let claimable = client
        .omninode_get_claimable_reward(session_id, &derived)
        .map_err(|e| {
            let err = anyhow::anyhow!("omninode_getClaimableReward: {e}");
            tracing::error!(
                event = claim_markers::CLAIM_FAILED,
                subcommand = "claim",
                session_id = session_id,
                verifier = %derived,
                category = "chain_rpc",
                error = %err,
                "claimable-reward read failed"
            );
            err
        })?;
    if !claimable.claimable_now || head < claimable.claim_ready_block {
        let blocks_until_ready = claimable.claim_ready_block.saturating_sub(head);
        tracing::warn!(
            event = claim_markers::CLAIM_REFUSED_MATURITY,
            subcommand = "claim",
            session_id = session_id,
            verifier = %derived,
            claim_ready_block = claimable.claim_ready_block,
            head = head,
            blocks_until_ready = blocks_until_ready,
            "claim refused: not mature"
        );
        return Err(anyhow::anyhow!(SettlementSubmitError::Immature {
            claim_ready_block: claimable.claim_ready_block,
            head,
            blocks_until_ready,
        }));
    }

    // ── Step 6: bond precheck (conditional) ──────────────────────────
    let session = client
        .omninode_get_inference_session(session_id)
        .map_err(|e| {
            let err = anyhow::anyhow!("omninode_getInferenceSession: {e}");
            tracing::error!(
                event = claim_markers::CLAIM_FAILED,
                subcommand = "claim",
                session_id = session_id,
                verifier = %derived,
                category = "chain_rpc",
                error = %err,
                "session read failed"
            );
            err
        })?;
    if let Some(s) = session.as_ref() {
        if s.bond_required {
            let outcome = identity
                .precheck_bond(client, true)
                .map_err(|e: ClaimSignerError| {
                    let err = anyhow::anyhow!("bond precheck: {e}");
                    tracing::error!(
                        event = claim_markers::CLAIM_FAILED,
                        subcommand = "claim",
                        session_id = session_id,
                        verifier = %derived,
                        category = "chain_rpc",
                        error = %err,
                        "bond precheck RPC failed"
                    );
                    err
                })?;
            if !matches!(outcome, BondPrecheckOutcome::Bonded { .. }) {
                let outcome_kind = format!("{outcome:?}");
                tracing::warn!(
                    event = claim_markers::CLAIM_REFUSED_BOND_PRECHECK,
                    subcommand = "claim",
                    session_id = session_id,
                    verifier = %derived,
                    outcome = %outcome_kind,
                    "claim refused: bond precheck"
                );
                return Err(anyhow::anyhow!(
                    SettlementSubmitError::BondPrecheckFailed { outcome_kind }
                ));
            }
        }
    }

    // ── Step 7: builder RPC ──────────────────────────────────────────
    let build_response = client
        .omninode_build_claim_inference_reward(&BuildClaimRewardRequest {
            from: derived.clone(),
            session_id: session_id.to_string(),
            fee: args.fee,
        })
        .map_err(|e| {
            let err = anyhow::anyhow!("{e}");
            tracing::error!(
                event = claim_markers::CLAIM_FAILED,
                subcommand = "claim",
                session_id = session_id,
                verifier = %derived,
                category = "chain_rpc",
                error = %err,
                "omninode_buildClaimInferenceReward failed"
            );
            err
        })?;

    // ── Step 8: verify envelope (before hex decode) ──────────────────
    if let Err(e) = verify_builder_envelope(
        &build_response,
        &derived,
        params.chain_id,
        args.fee,
    ) {
        let msg = e.to_string();
        tracing::error!(
            event = claim_markers::CLAIM_FAILED,
            subcommand = "claim",
            session_id = session_id,
            verifier = %derived,
            category = "builder_mismatch",
            error = %msg,
            "builder envelope mismatch"
        );
        return Err(anyhow::anyhow!(e));
    }

    // ── Step 9: decode + verify decoded TransactionV2 ────────────────
    let tx = decode_unsigned_tx(&build_response).map_err(|e| {
        let msg = e.to_string();
        tracing::error!(
            event = claim_markers::CLAIM_FAILED,
            subcommand = "claim",
            session_id = session_id,
            verifier = %derived,
            category = "wire_decode",
            error = %msg,
            "unsigned_tx decode failed"
        );
        anyhow::anyhow!(e)
    })?;
    if let Err(e) =
        verify_decoded_transaction(&tx, &build_response, &derived, session_id)
    {
        let msg = e.to_string();
        tracing::error!(
            event = claim_markers::CLAIM_FAILED,
            subcommand = "claim",
            session_id = session_id,
            verifier = %derived,
            category = "builder_mismatch",
            error = %msg,
            "decoded transaction mismatch"
        );
        return Err(anyhow::anyhow!(e));
    }

    // ── CLAIM_READY marker: all prechecks + verify passed ────────────
    tracing::info!(
        event = claim_markers::CLAIM_READY,
        subcommand = "claim",
        session_id = session_id,
        verifier = %derived,
        chain_id = params.chain_id,
        nonce = build_response.nonce,
        fee = build_response.fee.to_string(),
        claim_ready_block = claimable.claim_ready_block,
        head = head,
        "claim ready for submission"
    );

    // Emit human-visible summary either way.
    writeln!(out, "session_id={session_id}")?;
    writeln!(out, "verifier={derived}")?;
    writeln!(out, "chain_id={}", params.chain_id)?;
    writeln!(out, "nonce={}", build_response.nonce)?;
    writeln!(out, "fee={}", build_response.fee)?;
    writeln!(out, "claim_ready_block={}", claimable.claim_ready_block)?;
    writeln!(out, "head={head}")?;

    if args.dry_run {
        writeln!(out, "dry_run=true submitted=false")?;
        return Ok(());
    }

    // ── Steps 10-13: load signer, sign, submit ───────────────────────
    let signer = crate::settlement_signer::ClaimSigner::resolve(seed_source)
        .map_err(|e| {
            let err = anyhow::anyhow!("claim signer resolve: {e}");
            tracing::error!(
                event = claim_markers::CLAIM_FAILED,
                subcommand = "claim",
                session_id = session_id,
                verifier = %derived,
                category = "seed_missing_or_malformed",
                error = %err,
                "signer resolve failed"
            );
            err
        })?;
    // Belt-and-braces: `ClaimSigner::resolve` re-loads the seed from
    // env. Assert its derived address matches the one we've been
    // running prechecks against. If the env changed between §3.1 and
    // §3.10 (e.g. a shell replaced the value), refuse before signing.
    if signer.address() != derived.as_str() {
        let msg = format!(
            "signer re-derivation mismatch: prechecks used {derived}, \
             signer would sign as {}",
            signer.address()
        );
        tracing::error!(
            event = claim_markers::CLAIM_FAILED,
            subcommand = "claim",
            session_id = session_id,
            verifier = %derived,
            category = "seed_mismatch_between_stages",
            error = %msg,
            "signer re-derivation disagrees with precheck-time derivation"
        );
        return Err(anyhow::anyhow!(msg));
    }

    let receipt =
        sign_and_submit(client, &tx, signer.seed_for_signing(), session_id, &derived)
            .map_err(|e| {
                let err = anyhow::anyhow!("sign_and_submit: {e}");
                tracing::error!(
                    event = claim_markers::CLAIM_FAILED,
                    subcommand = "claim",
                    session_id = session_id,
                    verifier = %derived,
                    category = "chain_rpc",
                    error = %err,
                    "sign_and_submit failed"
                );
                err
            })?;

    tracing::info!(
        event = claim_markers::CLAIM_SUBMITTED,
        subcommand = "claim",
        session_id = session_id,
        verifier = derived.as_str(),
        tx_hash = receipt.tx_hash.as_str(),
        chain_id = receipt.chain_id,
        nonce = receipt.nonce,
        fee = receipt.fee.to_string().as_str(),
        "claim submitted"
    );
    writeln!(out, "tx_hash={}", receipt.tx_hash)?;
    writeln!(out, "submitted=true")?;
    Ok(())
}

// ── Issue #100 — verifier registration markers + run (settlement-submit) ──

#[cfg(feature = "settlement-submit")]
pub(crate) mod register_verifier_markers {
    /// Emitted once preflight (chain-id + settlement gate + bonding
    /// gate) and builder-envelope / decoded-tx verification all pass,
    /// immediately before the signing seed is loaded.
    pub(crate) const REGISTER_READY: &str = "settlement_register_verifier_ready";

    /// Emitted when `--bond` is zero. The chain executor rejects a zero
    /// bond (`Failed(365)`), so the handler refuses locally at the very
    /// top — before any preflight RPC or seed/identity access — rather
    /// than building a guaranteed-doomed, fee-paying transaction.
    pub(crate) const REGISTER_REFUSED_BOND_ZERO: &str =
        "settlement_register_verifier_refused_bond_zero";

    /// Emitted when the chain reports a `chain_id` different from the
    /// operator's `--expect-chain-id` guardrail.
    pub(crate) const REGISTER_REFUSED_CHAIN_ID: &str =
        "settlement_register_verifier_refused_chain_id";

    /// Emitted when the settlement base gate OR the verifier-bonding
    /// gate is dormant/scheduled at head. Carries the offending
    /// `gate` name.
    pub(crate) const REGISTER_REFUSED_DORMANCY: &str =
        "settlement_register_verifier_refused_dormancy";

    /// Emitted on a successful `sum_sendRawTransaction`.
    pub(crate) const REGISTER_SUBMITTED: &str =
        "settlement_register_verifier_submitted";

    /// Emitted on any RPC / envelope / wire / signer failure.
    pub(crate) const REGISTER_FAILED: &str = "settlement_register_verifier_failed";
}

/// Issue #100 — build + sign + submit an on-chain verifier registration.
///
/// Reuses the claim path's building blocks end-to-end: the
/// `ClaimSignerIdentity` / `ClaimSigner` seed helpers (#80/#87), the
/// generic `verify_builder_envelope` + `decode_unsigned_tx` +
/// `sign_and_submit` helpers, and the chain's no-key
/// `omninode_buildRegisterVerifier` builder. This handler adds ONLY
/// the register-verifier preflight (chain-id + the two activation
/// gates) and the register-verifier decoded-tx assertion — no new
/// crypto and no chain-side change.
#[cfg(feature = "settlement-submit")]
fn run_register_verifier<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    args: &RegisterVerifierArgs,
    out: &mut dyn Write,
    seed_source: crate::operator::SeedSource,
) -> Result<()> {
    use omni_sumchain::dto::BlockFinality;
    use omni_sumchain::settlement_submit::{
        decode_unsigned_tx, sign_and_submit_tx, verify_builder_envelope,
        verify_register_verifier_transaction, BuildRegisterVerifierRequest,
    };

    use register_verifier_markers as rvm;

    tracing::info!(
        event = "settlement_query",
        subcommand = "register-verifier",
        "settlement register-verifier start"
    );

    // ── Step 1: local bond validation (zero network, zero secrets) ───
    // The chain executor rejects a zero bond (`Failed(365)`), so a
    // `--bond 0` would build a guaranteed-doomed, fee-paying tx. Refuse
    // at the very TOP — before any preflight RPC and before any seed /
    // identity access. `--bond` is a REQUIRED clap arg, so an omitted
    // flag is rejected by clap before this handler is ever reached; the
    // only local case left to enforce is an explicit zero.
    if args.bond == 0 {
        tracing::warn!(
            event = rvm::REGISTER_REFUSED_BOND_ZERO,
            subcommand = "register-verifier",
            bond = args.bond.to_string(),
            "register-verifier refused: --bond must be positive"
        );
        return Err(anyhow::anyhow!(
            "--bond must be a positive amount: the chain executor rejects a \
             zero bond (Failed(365)); refusing to register verifier before \
             any RPC or seed access"
        ));
    }

    // ── Step 2: preflight RPC — params + head, chain-id, both gates ──
    // The entire preflight precedes signer-identity resolution: the
    // configured seed is never read for a command that a chain-id or
    // dormant-gate refusal would reject anyway.
    let params = client.get_chain_params().map_err(|e| {
        let err = anyhow::anyhow!("chain_getChainParams: {e}");
        tracing::error!(
            event = rvm::REGISTER_FAILED,
            subcommand = "register-verifier",
            category = "chain_rpc",
            error = %err,
            "chain_getChainParams failed during register-verifier preflight"
        );
        err
    })?;
    let head = client
        .get_block_height(BlockFinality::Latest)
        .map_err(|e| {
            let err = anyhow::anyhow!("chain_getBlockHeight: {e}");
            tracing::error!(
                event = rvm::REGISTER_FAILED,
                subcommand = "register-verifier",
                category = "chain_rpc",
                error = %err,
                "chain_getBlockHeight failed during register-verifier preflight"
            );
            err
        })?
        .height;

    // chain-id guardrail.
    if params.chain_id != args.expect_chain_id {
        let msg = format!(
            "chain-id mismatch: --expect-chain-id={}, chain reports {}; \
             refusing to register verifier",
            args.expect_chain_id, params.chain_id
        );
        tracing::warn!(
            event = rvm::REGISTER_REFUSED_CHAIN_ID,
            subcommand = "register-verifier",
            expected = args.expect_chain_id,
            actual = params.chain_id,
            "register-verifier refused: chain-id mismatch"
        );
        return Err(anyhow::anyhow!(msg));
    }

    // settlement base gate.
    let settlement_observed = params.inference_settlement_enabled_from_height;
    if !settlement_observed.is_some_and(|n| head >= n) {
        let observed_display = match settlement_observed {
            None => "None".to_string(),
            Some(n) => format!("Some({n})"),
        };
        tracing::warn!(
            event = rvm::REGISTER_REFUSED_DORMANCY,
            subcommand = "register-verifier",
            gate = "inference_settlement_enabled_from_height",
            observed = %observed_display,
            head = head,
            "register-verifier refused: settlement subsystem inactive"
        );
        return Err(anyhow::anyhow!(
            "settlement subsystem inactive: gate \
             'inference_settlement_enabled_from_height' observed={observed_display}, \
             head={head}; refusing to register verifier"
        ));
    }

    // verifier-bonding gate. RegisterVerifier is a bonding operation;
    // the chain gates it on `inference_verifier_bonding_enabled_from_height`
    // (executor `Failed(364)` when closed) in addition to the outer
    // settlement gate above. Refuse locally to avoid a fee-paying
    // doomed tx.
    let bonding_observed = params.inference_verifier_bonding_enabled_from_height;
    if !bonding_observed.is_some_and(|n| head >= n) {
        let observed_display = match bonding_observed {
            None => "None".to_string(),
            Some(n) => format!("Some({n})"),
        };
        tracing::warn!(
            event = rvm::REGISTER_REFUSED_DORMANCY,
            subcommand = "register-verifier",
            gate = "inference_verifier_bonding_enabled_from_height",
            observed = %observed_display,
            head = head,
            "register-verifier refused: verifier bonding inactive"
        );
        return Err(anyhow::anyhow!(
            "verifier bonding inactive: gate \
             'inference_verifier_bonding_enabled_from_height' observed={observed_display}, \
             head={head}; refusing to register verifier"
        ));
    }

    // ── Step 3: resolve signer identity (reads seed, derives address) ─
    // Only now — after local bond validation and every preflight
    // refusal — is the configured `OMNINODE_VERIFIER_SEED_HEX` seed
    // read. `ClaimSignerIdentity::resolve` derives the base58 address
    // and drops the secret; it is NOT retained. Dry-run needs this
    // address for the builder request and the envelope `from`
    // assertion, but never signs and never submits.
    let identity = match crate::settlement_signer::ClaimSignerIdentity::resolve(
        seed_source.clone(),
    ) {
        Ok(i) => i,
        Err(e) => {
            let err = anyhow::anyhow!("register-verifier signer identity resolve: {e}");
            tracing::error!(
                event = rvm::REGISTER_FAILED,
                subcommand = "register-verifier",
                category = "seed_missing_or_malformed",
                error = %err,
                "signer identity resolve failed"
            );
            return Err(err);
        }
    };
    let derived = identity.address().to_string();

    // ── Step 4: builder RPC ──────────────────────────────────────────
    let build_response = client
        .omninode_build_register_verifier(&BuildRegisterVerifierRequest {
            from: derived.clone(),
            bond: args.bond,
            fee: args.fee,
        })
        .map_err(|e| {
            let err = anyhow::anyhow!("{e}");
            tracing::error!(
                event = rvm::REGISTER_FAILED,
                subcommand = "register-verifier",
                verifier = %derived,
                category = "chain_rpc",
                error = %err,
                "omninode_buildRegisterVerifier failed"
            );
            err
        })?;

    // ── Step 5: verify envelope (before hex decode) ──────────────────
    if let Err(e) = verify_builder_envelope(
        &build_response,
        &derived,
        params.chain_id,
        args.fee,
    ) {
        let msg = e.to_string();
        tracing::error!(
            event = rvm::REGISTER_FAILED,
            subcommand = "register-verifier",
            verifier = %derived,
            category = "builder_mismatch",
            error = %msg,
            "builder envelope mismatch"
        );
        return Err(anyhow::anyhow!(e));
    }

    // ── Step 6: decode + verify decoded TransactionV2 ────────────────
    let tx = decode_unsigned_tx(&build_response).map_err(|e| {
        let msg = e.to_string();
        tracing::error!(
            event = rvm::REGISTER_FAILED,
            subcommand = "register-verifier",
            verifier = %derived,
            category = "wire_decode",
            error = %msg,
            "unsigned_tx decode failed"
        );
        anyhow::anyhow!(e)
    })?;
    if let Err(e) = verify_register_verifier_transaction(
        &tx,
        &build_response,
        &derived,
        args.bond,
    ) {
        let msg = e.to_string();
        tracing::error!(
            event = rvm::REGISTER_FAILED,
            subcommand = "register-verifier",
            verifier = %derived,
            category = "builder_mismatch",
            error = %msg,
            "decoded transaction mismatch"
        );
        return Err(anyhow::anyhow!(e));
    }

    // ── READY marker: all preflight + verify passed ──────────────────
    tracing::info!(
        event = rvm::REGISTER_READY,
        subcommand = "register-verifier",
        verifier = %derived,
        chain_id = params.chain_id,
        nonce = build_response.nonce,
        fee = build_response.fee.to_string(),
        bond = args.bond.to_string(),
        head = head,
        "register-verifier ready for submission"
    );

    writeln!(out, "verifier={derived}")?;
    writeln!(out, "chain_id={}", params.chain_id)?;
    writeln!(out, "nonce={}", build_response.nonce)?;
    writeln!(out, "fee={}", build_response.fee)?;
    writeln!(out, "bond={}", args.bond)?;
    writeln!(out, "head={head}")?;

    if args.dry_run {
        writeln!(out, "dry_run=true submitted=false")?;
        return Ok(());
    }

    // ── Step 7: load signer, re-derivation guard, sign + submit ──────
    let signer = crate::settlement_signer::ClaimSigner::resolve(seed_source)
        .map_err(|e| {
            let err = anyhow::anyhow!("register-verifier signer resolve: {e}");
            tracing::error!(
                event = rvm::REGISTER_FAILED,
                subcommand = "register-verifier",
                verifier = %derived,
                category = "seed_missing_or_malformed",
                error = %err,
                "signer resolve failed"
            );
            err
        })?;
    if signer.address() != derived.as_str() {
        let msg = format!(
            "signer re-derivation mismatch: preflight used {derived}, \
             signer would sign as {}",
            signer.address()
        );
        tracing::error!(
            event = rvm::REGISTER_FAILED,
            subcommand = "register-verifier",
            verifier = %derived,
            category = "seed_mismatch_between_stages",
            error = %msg,
            "signer re-derivation disagrees with preflight-time derivation"
        );
        return Err(anyhow::anyhow!(msg));
    }

    // Uses the generic `sign_and_submit_tx` core, which returns the
    // generic `SettlementSubmitReceipt` (tx hash + submit status). A
    // verifier registration has no session / verifier-reward fields to
    // report, so it does NOT reuse the claim path's `ClaimSubmitReceipt`
    // (which would force a fabricated empty `session_id`). The envelope
    // values (`chain_id` / `nonce` / `fee`), already asserted equal to
    // the signed tx during verification above, are reported directly.
    let receipt = sign_and_submit_tx(client, &tx, signer.seed_for_signing())
        .map_err(|e| {
            let err = anyhow::anyhow!("sign_and_submit: {e}");
            tracing::error!(
                event = rvm::REGISTER_FAILED,
                subcommand = "register-verifier",
                verifier = %derived,
                category = "chain_rpc",
                error = %err,
                "sign_and_submit failed"
            );
            err
        })?;

    let submitted = receipt.status.is_submitted();
    tracing::info!(
        event = rvm::REGISTER_SUBMITTED,
        subcommand = "register-verifier",
        verifier = derived.as_str(),
        tx_hash = receipt.tx_hash.as_str(),
        chain_id = params.chain_id,
        nonce = build_response.nonce,
        fee = build_response.fee.to_string().as_str(),
        bond = args.bond.to_string().as_str(),
        submitted = submitted,
        "register-verifier submitted"
    );
    writeln!(out, "tx_hash={}", receipt.tx_hash)?;
    writeln!(out, "submitted={submitted}")?;
    Ok(())
}

// ── Issue #81 — dispute markers + run_dispute_* (settlement-dispute-gated) ──

#[cfg(feature = "settlement-dispute")]
pub(crate) mod dispute_markers {
    pub(crate) const DISPUTE_READY: &str = "settlement_dispute_ready";
    pub(crate) const DISPUTE_REFUSED_DORMANCY: &str =
        "settlement_dispute_refused_dormancy";
    pub(crate) const DISPUTE_REFUSED_AUTHORITY: &str =
        "settlement_dispute_refused_authority";
    pub(crate) const DISPUTE_REFUSED_MATURITY: &str =
        "settlement_dispute_refused_maturity";
    pub(crate) const DISPUTE_REFUSED_STATE: &str = "settlement_dispute_refused_state";
    pub(crate) const DISPUTE_SUBMITTED: &str = "settlement_dispute_submitted";
    pub(crate) const DISPUTE_FAILED: &str = "settlement_dispute_failed";
}

// ── OpenDispute handler ─────────────────────────────────────────────────────

#[cfg(feature = "settlement-dispute")]
fn run_dispute_open<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    args: &OpenDisputeArgs,
    out: &mut dyn Write,
    seed_source: crate::operator::SeedSource,
) -> Result<()> {
    use crate::settlement_signer::{ClaimSignerIdentity};
    use omni_sumchain::dto::BlockFinality;
    use omni_sumchain::settlement_dispute::{
        decode_unsigned_tx_open, sign_and_submit_dispute,
        verify_open_builder_envelope, verify_open_decoded_transaction,
        wire::{evidence_bytes_to_wire, parse_evidence_hex},
        BuildOpenInferenceDisputeRequest, SettlementDisputeError,
    };

    let session_id = args.session_id.as_str();
    tracing::info!(
        event = "settlement_query",
        subcommand = "dispute",
        phase = "open",
        session_id = session_id,
        "dispute open start"
    );

    // ── Step 1: derive identity (settlement-read; no seed retention) ──
    let identity = ClaimSignerIdentity::resolve(seed_source.clone()).map_err(|e| {
        let err = anyhow::anyhow!("claim signer identity resolve: {e}");
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "open",
            session_id = session_id,
            category = "seed_missing_or_malformed",
            error = %err,
            "identity resolve failed"
        );
        err
    })?;
    let derived = identity.address().to_string();

    // ── Step 2: params + head + settlement gate ──────────────────────
    let params = client.get_chain_params().map_err(|e| {
        let err = anyhow::anyhow!("chain_getChainParams: {e}");
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "open",
            session_id = session_id,
            category = "chain_rpc",
            error = %err,
            "chain_getChainParams failed"
        );
        err
    })?;
    let head = client
        .get_block_height(BlockFinality::Latest)
        .map_err(|e| {
            let err = anyhow::anyhow!("chain_getBlockHeight: {e}");
            tracing::error!(
                event = dispute_markers::DISPUTE_FAILED,
                subcommand = "dispute",
                phase = "open",
                session_id = session_id,
                category = "chain_rpc",
                error = %err,
                "chain_getBlockHeight failed"
            );
            err
        })?
        .height;
    let observed = params.inference_settlement_enabled_from_height;
    if !observed.is_some_and(|n| head >= n) {
        let observed_display = match observed {
            None => "None".to_string(),
            Some(n) => format!("Some({n})"),
        };
        tracing::warn!(
            event = dispute_markers::DISPUTE_REFUSED_DORMANCY,
            subcommand = "dispute",
            phase = "open",
            session_id = session_id,
            gate = "inference_settlement_enabled_from_height",
            observed = %observed_display,
            head = head,
            "dispute open refused: settlement gate dormant"
        );
        return Err(anyhow::anyhow!(SettlementDisputeError::Dormant {
            observed,
            head,
        }));
    }

    // ── Step 3: fetch session; refuse if absent ──────────────────────
    let session = match client
        .omninode_get_inference_session(session_id)
        .map_err(|e| {
            let err = anyhow::anyhow!("omninode_getInferenceSession: {e}");
            tracing::error!(
                event = dispute_markers::DISPUTE_FAILED,
                subcommand = "dispute",
                phase = "open",
                session_id = session_id,
                category = "chain_rpc",
                error = %err,
                "session read failed"
            );
            err
        })? {
        Some(s) => s,
        None => {
            let msg = format!("session '{session_id}' not found on chain");
            tracing::warn!(
                event = dispute_markers::DISPUTE_REFUSED_STATE,
                subcommand = "dispute",
                phase = "open",
                session_id = session_id,
                reason = "session_not_found",
                "dispute open refused: session missing"
            );
            return Err(anyhow::anyhow!(msg));
        }
    };

    // ── Step 4: funder precheck ──────────────────────────────────────
    if session.funder != derived {
        tracing::warn!(
            event = dispute_markers::DISPUTE_REFUSED_AUTHORITY,
            subcommand = "dispute",
            phase = "open",
            session_id = session_id,
            expected_funder = session.funder.as_str(),
            derived = derived.as_str(),
            "dispute open refused: funder mismatch"
        );
        return Err(anyhow::anyhow!(SettlementDisputeError::FunderMismatch {
            derived: derived.clone(),
            session_funder: session.funder,
        }));
    }

    // ── Step 4b: session-open precheck ───────────────────────────────
    // Chain-team confirmed (sum-chain#110) that `OpenDispute` is only
    // accepted against sessions still in `Open` status. Refunded /
    // Settled / unknown terminal states must refuse locally so the
    // operator doesn't burn a fee on a tx the chain will reject.
    use omni_sumchain::settlement::view::SessionLifecycle;
    let session_lifecycle = SessionLifecycle::from_wire(&session.lifecycle);
    if !matches!(session_lifecycle, SessionLifecycle::Active) {
        tracing::warn!(
            event = dispute_markers::DISPUTE_REFUSED_STATE,
            subcommand = "dispute",
            phase = "open",
            session_id = session_id,
            reason = "session_not_open",
            wire_status = session.lifecycle.as_str(),
            "dispute open refused: session is not in Open status"
        );
        return Err(anyhow::anyhow!(
            "session '{session_id}' is not open: chain status='{}'",
            session.lifecycle
        ));
    }

    // ── Step 5: attestation lookup ───────────────────────────────────
    let attestation = match client
        .get_attestation(session_id, args.verifier.as_str())
        .map_err(|e| {
            let err = anyhow::anyhow!("sum_getInferenceAttestation: {e}");
            tracing::error!(
                event = dispute_markers::DISPUTE_FAILED,
                subcommand = "dispute",
                phase = "open",
                session_id = session_id,
                verifier = args.verifier.as_str(),
                category = "chain_rpc",
                error = %err,
                "attestation lookup failed"
            );
            err
        })? {
        Some(a) => a,
        None => {
            let msg = format!(
                "no attestation found for (session_id={session_id}, verifier={})",
                args.verifier
            );
            tracing::warn!(
                event = dispute_markers::DISPUTE_REFUSED_STATE,
                subcommand = "dispute",
                phase = "open",
                session_id = session_id,
                verifier = args.verifier.as_str(),
                reason = "attestation_not_found",
                "dispute open refused: attestation missing"
            );
            return Err(anyhow::anyhow!(msg));
        }
    };

    // ── Step 6: maturity precheck ────────────────────────────────────
    let maturity = attestation.included_at_height
        + params.finality_depth
        + session.dispute_window_blocks;
    if head >= maturity {
        tracing::warn!(
            event = dispute_markers::DISPUTE_REFUSED_MATURITY,
            subcommand = "dispute",
            phase = "open",
            session_id = session_id,
            verifier = args.verifier.as_str(),
            included_at_height = attestation.included_at_height,
            finality_depth = params.finality_depth,
            dispute_window_blocks = session.dispute_window_blocks,
            maturity = maturity,
            head = head,
            "dispute open refused: dispute window closed"
        );
        return Err(anyhow::anyhow!(
            SettlementDisputeError::MaturityWindowClosed {
                included_at_height: attestation.included_at_height,
                finality_depth: params.finality_depth,
                dispute_window_blocks: session.dispute_window_blocks,
                maturity,
                head,
            }
        ));
    }

    // ── Step 7: duplicate-dispute precheck ───────────────────────────
    let disputes = client
        .omninode_get_inference_disputes(session_id)
        .map_err(|e| {
            let err = anyhow::anyhow!("omninode_getInferenceDisputes: {e}");
            tracing::error!(
                event = dispute_markers::DISPUTE_FAILED,
                subcommand = "dispute",
                phase = "open",
                session_id = session_id,
                category = "chain_rpc",
                error = %err,
                "disputes read failed"
            );
            err
        })?;
    if disputes
        .disputes
        .iter()
        .any(|d| d.verifier_address == args.verifier)
    {
        tracing::warn!(
            event = dispute_markers::DISPUTE_REFUSED_STATE,
            subcommand = "dispute",
            phase = "open",
            session_id = session_id,
            verifier = args.verifier.as_str(),
            reason = "duplicate_dispute",
            "dispute open refused: dispute already exists for this verifier"
        );
        return Err(anyhow::anyhow!(SettlementDisputeError::DuplicateDispute {
            session_id: session_id.to_string(),
            verifier: args.verifier.clone(),
        }));
    }

    // ── Step 8: parse evidence hex ───────────────────────────────────
    let evidence_bytes = parse_evidence_hex(&args.evidence).map_err(|e| {
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "open",
            session_id = session_id,
            category = "evidence_parse",
            error = %e,
            "evidence hex parse failed"
        );
        anyhow::anyhow!(e)
    })?;

    // ── Step 9: call builder RPC ─────────────────────────────────────
    let build_response = client
        .omninode_build_open_inference_dispute(&BuildOpenInferenceDisputeRequest {
            from: derived.clone(),
            session_id: session_id.to_string(),
            verifier: args.verifier.clone(),
            evidence_commitment: evidence_bytes_to_wire(&evidence_bytes),
            fee: args.fee,
        })
        .map_err(|e| {
            let err = anyhow::anyhow!("{e}");
            tracing::error!(
                event = dispute_markers::DISPUTE_FAILED,
                subcommand = "dispute",
                phase = "open",
                session_id = session_id,
                category = "chain_rpc",
                error = %err,
                "OpenDispute builder failed"
            );
            err
        })?;

    // ── Step 10: verify envelope ─────────────────────────────────────
    if let Err(e) = verify_open_builder_envelope(
        &build_response,
        &derived,
        params.chain_id,
        args.fee,
    ) {
        let msg = e.to_string();
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "open",
            session_id = session_id,
            category = "builder_mismatch",
            error = %msg,
            "builder envelope mismatch"
        );
        return Err(anyhow::anyhow!(e));
    }

    // ── Step 11: decode + verify decoded tx ─────────────────────────
    let tx = decode_unsigned_tx_open(&build_response).map_err(|e| {
        let msg = e.to_string();
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "open",
            session_id = session_id,
            category = "wire_decode",
            error = %msg,
            "unsigned_tx decode failed"
        );
        anyhow::anyhow!(e)
    })?;
    if let Err(e) = verify_open_decoded_transaction(
        &tx,
        &build_response,
        &derived,
        session_id,
        args.verifier.as_str(),
        &evidence_bytes,
    ) {
        let msg = e.to_string();
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "open",
            session_id = session_id,
            category = "builder_mismatch",
            error = %msg,
            "decoded transaction mismatch"
        );
        return Err(anyhow::anyhow!(e));
    }

    tracing::info!(
        event = dispute_markers::DISPUTE_READY,
        subcommand = "dispute",
        phase = "open",
        session_id = session_id,
        verifier = args.verifier.as_str(),
        chain_id = params.chain_id,
        nonce = build_response.nonce,
        fee = build_response.fee.to_string().as_str(),
        maturity = maturity,
        head = head,
        "dispute open ready for submission"
    );

    writeln!(out, "phase=open")?;
    writeln!(out, "session_id={session_id}")?;
    writeln!(out, "verifier={}", args.verifier)?;
    writeln!(out, "chain_id={}", params.chain_id)?;
    writeln!(out, "nonce={}", build_response.nonce)?;
    writeln!(out, "fee={}", build_response.fee)?;
    writeln!(out, "maturity={maturity}")?;
    writeln!(out, "head={head}")?;

    if args.dry_run {
        writeln!(out, "dry_run=true submitted=false")?;
        return Ok(());
    }

    // ── Step 12-14: load signer, sign, submit ────────────────────────
    let signer = crate::settlement_signer::ClaimSigner::resolve(seed_source)
        .map_err(|e| {
            let err = anyhow::anyhow!("claim signer resolve: {e}");
            tracing::error!(
                event = dispute_markers::DISPUTE_FAILED,
                subcommand = "dispute",
                phase = "open",
                session_id = session_id,
                category = "seed_missing_or_malformed",
                error = %err,
                "signer resolve failed"
            );
            err
        })?;
    if signer.address() != derived.as_str() {
        let msg = format!(
            "signer re-derivation mismatch: prechecks used {derived}, signer would sign as {}",
            signer.address()
        );
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "open",
            session_id = session_id,
            category = "seed_mismatch_between_stages",
            error = %msg,
            "signer disagrees with precheck-time derivation"
        );
        return Err(anyhow::anyhow!(msg));
    }
    let receipt = sign_and_submit_dispute(
        client,
        &tx,
        signer.seed_for_signing(),
        "open",
        session_id,
        args.verifier.as_str(),
    )
    .map_err(|e| {
        let err = anyhow::anyhow!("sign_and_submit_dispute: {e}");
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "open",
            session_id = session_id,
            category = "chain_rpc",
            error = %err,
            "sign_and_submit failed"
        );
        err
    })?;

    tracing::info!(
        event = dispute_markers::DISPUTE_SUBMITTED,
        subcommand = "dispute",
        phase = "open",
        session_id = session_id,
        verifier = args.verifier.as_str(),
        tx_hash = receipt.tx_hash.as_str(),
        chain_id = receipt.chain_id,
        nonce = receipt.nonce,
        fee = receipt.fee.to_string().as_str(),
        "dispute open submitted"
    );
    writeln!(out, "tx_hash={}", receipt.tx_hash)?;
    writeln!(out, "submitted=true")?;
    Ok(())
}

// ── ResolveDispute handler ──────────────────────────────────────────────────

#[cfg(feature = "settlement-dispute")]
fn run_dispute_resolve<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    args: &ResolveDisputeArgs,
    out: &mut dyn Write,
    seed_source: crate::operator::SeedSource,
) -> Result<()> {
    use crate::settlement_signer::ClaimSignerIdentity;
    use omni_sumchain::dto::BlockFinality;
    use omni_sumchain::settlement_dispute::{
        decode_unsigned_tx_resolve, sign_and_submit_dispute,
        verify_resolve_builder_envelope, verify_resolve_decoded_transaction,
        wire::parse_approvals_json, BuildResolveInferenceDisputeRequest,
        SettlementDisputeError,
    };

    let session_id = args.session_id.as_str();
    tracing::info!(
        event = "settlement_query",
        subcommand = "dispute",
        phase = "resolve",
        session_id = session_id,
        "dispute resolve start"
    );

    // ── Step 1: derive identity (fee-payer only, no authority) ───────
    let identity = ClaimSignerIdentity::resolve(seed_source.clone()).map_err(|e| {
        let err = anyhow::anyhow!("claim signer identity resolve: {e}");
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "resolve",
            session_id = session_id,
            category = "seed_missing_or_malformed",
            error = %err,
            "identity resolve failed"
        );
        err
    })?;
    let derived = identity.address().to_string();

    // ── Step 2: params + head + settlement gate ──────────────────────
    let params = client.get_chain_params().map_err(|e| {
        let err = anyhow::anyhow!("chain_getChainParams: {e}");
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "resolve",
            session_id = session_id,
            category = "chain_rpc",
            error = %err,
            "chain_getChainParams failed"
        );
        err
    })?;
    let head = client
        .get_block_height(BlockFinality::Latest)
        .map_err(|e| {
            let err = anyhow::anyhow!("chain_getBlockHeight: {e}");
            tracing::error!(
                event = dispute_markers::DISPUTE_FAILED,
                subcommand = "dispute",
                phase = "resolve",
                session_id = session_id,
                category = "chain_rpc",
                error = %err,
                "chain_getBlockHeight failed"
            );
            err
        })?
        .height;
    let observed = params.inference_settlement_enabled_from_height;
    if !observed.is_some_and(|n| head >= n) {
        let observed_display = match observed {
            None => "None".to_string(),
            Some(n) => format!("Some({n})"),
        };
        tracing::warn!(
            event = dispute_markers::DISPUTE_REFUSED_DORMANCY,
            subcommand = "dispute",
            phase = "resolve",
            session_id = session_id,
            gate = "inference_settlement_enabled_from_height",
            observed = %observed_display,
            head = head,
            "dispute resolve refused: settlement gate dormant"
        );
        return Err(anyhow::anyhow!(SettlementDisputeError::Dormant {
            observed,
            head,
        }));
    }

    // ── Step 3: fetch disputes; find Open one for --verifier ─────────
    use omni_sumchain::settlement::view::DisputeState;
    let disputes = client
        .omninode_get_inference_disputes(session_id)
        .map_err(|e| {
            let err = anyhow::anyhow!("omninode_getInferenceDisputes: {e}");
            tracing::error!(
                event = dispute_markers::DISPUTE_FAILED,
                subcommand = "dispute",
                phase = "resolve",
                session_id = session_id,
                category = "chain_rpc",
                error = %err,
                "disputes read failed"
            );
            err
        })?;
    let target = disputes
        .disputes
        .iter()
        .find(|d| d.verifier_address == args.verifier);
    let is_open = match target {
        Some(d) => matches!(
            classify_dispute_wire_state(&d.state),
            DisputeState::Open { .. }
        ),
        None => false,
    };
    if !is_open {
        tracing::warn!(
            event = dispute_markers::DISPUTE_REFUSED_STATE,
            subcommand = "dispute",
            phase = "resolve",
            session_id = session_id,
            verifier = args.verifier.as_str(),
            reason = "dispute_not_open",
            "dispute resolve refused: no Open dispute for verifier"
        );
        return Err(anyhow::anyhow!(SettlementDisputeError::DisputeNotOpen {
            session_id: session_id.to_string(),
            verifier: args.verifier.clone(),
        }));
    }

    // ── Step 4: parse --approvals JSON ───────────────────────────────
    let approvals_bytes = std::fs::read_to_string(&args.approvals).map_err(|e| {
        let msg = format!("failed to read --approvals file {:?}: {e}", args.approvals);
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "resolve",
            session_id = session_id,
            category = "approvals_read",
            error = %msg,
            "approvals file read failed"
        );
        anyhow::anyhow!(msg)
    })?;
    let parsed_approvals = parse_approvals_json(&approvals_bytes).map_err(|e| {
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "resolve",
            session_id = session_id,
            category = "approvals_parse",
            error = %e,
            "approvals JSON parse failed"
        );
        anyhow::anyhow!(e)
    })?;

    // ── Step 4b: empty-approvals precheck ────────────────────────────
    // Chain-team confirmed (sum-chain#110): an empty approvals array
    // will BUILD but always FAIL authority at execution. Refuse
    // locally before builder/sign/submit so the operator doesn't burn
    // a fee on a predictably-rejected tx.
    if parsed_approvals.is_empty() {
        tracing::warn!(
            event = dispute_markers::DISPUTE_REFUSED_STATE,
            subcommand = "dispute",
            phase = "resolve",
            session_id = session_id,
            reason = "approvals_empty",
            "dispute resolve refused: empty approvals array cannot meet validator quorum"
        );
        return Err(anyhow::anyhow!(SettlementDisputeError::ApprovalsEmpty));
    }

    let wire_approvals: Vec<_> = parsed_approvals.iter().map(|p| p.to_wire()).collect();

    // ── Step 5: call builder RPC ─────────────────────────────────────
    let build_response = client
        .omninode_build_resolve_inference_dispute(
            &BuildResolveInferenceDisputeRequest {
                from: derived.clone(),
                session_id: session_id.to_string(),
                verifier: args.verifier.clone(),
                allow_claim: args.allow_claim,
                approvals: wire_approvals,
                fee: args.fee,
            },
        )
        .map_err(|e| {
            let err = anyhow::anyhow!("{e}");
            tracing::error!(
                event = dispute_markers::DISPUTE_FAILED,
                subcommand = "dispute",
                phase = "resolve",
                session_id = session_id,
                category = "chain_rpc",
                error = %err,
                "ResolveDispute builder failed"
            );
            err
        })?;

    // ── Step 6: verify envelope ──────────────────────────────────────
    if let Err(e) = verify_resolve_builder_envelope(
        &build_response,
        &derived,
        params.chain_id,
        args.fee,
    ) {
        let msg = e.to_string();
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "resolve",
            session_id = session_id,
            category = "builder_mismatch",
            error = %msg,
            "builder envelope mismatch"
        );
        return Err(anyhow::anyhow!(e));
    }

    // ── Step 7: decode + verify decoded tx (order-sensitive approvals) ──
    let tx = decode_unsigned_tx_resolve(&build_response).map_err(|e| {
        let msg = e.to_string();
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "resolve",
            session_id = session_id,
            category = "wire_decode",
            error = %msg,
            "unsigned_tx decode failed"
        );
        anyhow::anyhow!(e)
    })?;
    if let Err(e) = verify_resolve_decoded_transaction(
        &tx,
        &build_response,
        &derived,
        session_id,
        args.verifier.as_str(),
        args.allow_claim,
        &parsed_approvals,
    ) {
        let msg = e.to_string();
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "resolve",
            session_id = session_id,
            category = "builder_mismatch",
            error = %msg,
            "decoded transaction mismatch"
        );
        return Err(anyhow::anyhow!(e));
    }

    tracing::info!(
        event = dispute_markers::DISPUTE_READY,
        subcommand = "dispute",
        phase = "resolve",
        session_id = session_id,
        verifier = args.verifier.as_str(),
        allow_claim = args.allow_claim,
        approvals_count = parsed_approvals.len(),
        chain_id = params.chain_id,
        nonce = build_response.nonce,
        fee = build_response.fee.to_string().as_str(),
        "dispute resolve ready for submission"
    );

    writeln!(out, "phase=resolve")?;
    writeln!(out, "session_id={session_id}")?;
    writeln!(out, "verifier={}", args.verifier)?;
    writeln!(out, "allow_claim={}", args.allow_claim)?;
    writeln!(out, "approvals_count={}", parsed_approvals.len())?;
    writeln!(out, "chain_id={}", params.chain_id)?;
    writeln!(out, "nonce={}", build_response.nonce)?;
    writeln!(out, "fee={}", build_response.fee)?;

    if args.dry_run {
        writeln!(out, "dry_run=true submitted=false")?;
        return Ok(());
    }

    // ── Step 8-10: load signer, sign, submit ─────────────────────────
    let signer = crate::settlement_signer::ClaimSigner::resolve(seed_source)
        .map_err(|e| {
            let err = anyhow::anyhow!("claim signer resolve: {e}");
            tracing::error!(
                event = dispute_markers::DISPUTE_FAILED,
                subcommand = "dispute",
                phase = "resolve",
                session_id = session_id,
                category = "seed_missing_or_malformed",
                error = %err,
                "signer resolve failed"
            );
            err
        })?;
    if signer.address() != derived.as_str() {
        let msg = format!(
            "signer re-derivation mismatch: prechecks used {derived}, signer would sign as {}",
            signer.address()
        );
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "resolve",
            session_id = session_id,
            category = "seed_mismatch_between_stages",
            error = %msg,
            "signer disagrees with precheck-time derivation"
        );
        return Err(anyhow::anyhow!(msg));
    }
    let receipt = sign_and_submit_dispute(
        client,
        &tx,
        signer.seed_for_signing(),
        "resolve",
        session_id,
        args.verifier.as_str(),
    )
    .map_err(|e| {
        let err = anyhow::anyhow!("sign_and_submit_dispute: {e}");
        tracing::error!(
            event = dispute_markers::DISPUTE_FAILED,
            subcommand = "dispute",
            phase = "resolve",
            session_id = session_id,
            category = "chain_rpc",
            error = %err,
            "sign_and_submit failed"
        );
        err
    })?;

    tracing::info!(
        event = dispute_markers::DISPUTE_SUBMITTED,
        subcommand = "dispute",
        phase = "resolve",
        session_id = session_id,
        verifier = args.verifier.as_str(),
        tx_hash = receipt.tx_hash.as_str(),
        chain_id = receipt.chain_id,
        nonce = receipt.nonce,
        fee = receipt.fee.to_string().as_str(),
        "dispute resolve submitted"
    );
    writeln!(out, "tx_hash={}", receipt.tx_hash)?;
    writeln!(out, "submitted=true")?;
    Ok(())
}

/// Local helper: turn the raw wire state string into a
/// `DisputeState` variant. Only cares about `Open` vs everything else
/// for the resolve precheck; unknown states map to `UnknownWire`.
#[cfg(feature = "settlement-dispute")]
fn classify_dispute_wire_state(
    state: &str,
) -> omni_sumchain::settlement::view::DisputeState {
    use omni_sumchain::settlement::view::DisputeState;
    match state {
        "open" => DisputeState::Open { opened_at_height: 0 },
        "resolved_approved" => DisputeState::ResolvedApproved {
            resolved_at_height: None,
            approve_bps: None,
            deny_bps: None,
        },
        "resolved_denied" => DisputeState::ResolvedDenied {
            resolved_at_height: None,
            approve_bps: None,
            deny_bps: None,
        },
        other => DisputeState::UnknownWire(other.to_string()),
    }
}

// ── Hermetic tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};
    use omni_sumchain::FakeJsonRpcTransport;
    use serde_json::json;

    // ── Fixtures ────────────────────────────────────────────────────────

    fn make_client() -> (SumChainClient<FakeJsonRpcTransport>, FakeJsonRpcTransport) {
        let fake = FakeJsonRpcTransport::new();
        // Zero seed is the read-only sentinel — no signing surface is
        // ever reached in this module.
        let client = SumChainClient::with_transport([0u8; 32], fake.clone());
        (client, fake)
    }

    fn seed_params(fake: &FakeJsonRpcTransport, params: serde_json::Value) {
        fake.set_response("chain_getChainParams", Ok(params));
    }

    fn seed_head(fake: &FakeJsonRpcTransport, height: u64) {
        fake.set_response(
            "chain_getBlockHeight",
            Ok(json!({ "height": height, "finality": "latest" })),
        );
    }

    fn params_all_dormant() -> serde_json::Value {
        json!({
            "finality_depth": 12,
            "min_fee": 100,
            "chain_id": 1_800_100,
        })
    }

    fn params_settlement_only_active(h: u64) -> serde_json::Value {
        json!({
            "finality_depth": 12,
            "min_fee": 100,
            "chain_id": 1_800_100,
            "inference_settlement_enabled_from_height": h,
        })
    }

    fn params_all_gates_active(h: u64) -> serde_json::Value {
        json!({
            "finality_depth": 12,
            "min_fee": 100,
            "chain_id": 1_800_100,
            "inference_settlement_enabled_from_height": h,
            "inference_settlement_consistency_enabled_from_height": h,
            "inference_verifier_bonding_enabled_from_height": h,
        })
    }

    fn args_status() -> SettlementArgs {
        SettlementArgs {
            rpc_url: None,
            cmd: SettlementCmd::Status,
        }
    }

    fn args_session(id: &str) -> SettlementArgs {
        SettlementArgs {
            rpc_url: None,
            cmd: SettlementCmd::Session(SessionArgs {
                session_id: id.to_string(),
                json: false,
            }),
        }
    }

    fn args_session_json(id: &str) -> SettlementArgs {
        SettlementArgs {
            rpc_url: None,
            cmd: SettlementCmd::Session(SessionArgs {
                session_id: id.to_string(),
                json: true,
            }),
        }
    }

    fn args_claimable(id: &str, verifier: &str) -> SettlementArgs {
        SettlementArgs {
            rpc_url: None,
            cmd: SettlementCmd::Claimable(ClaimableArgs {
                session_id: id.to_string(),
                verifier: verifier.to_string(),
            }),
        }
    }

    fn args_verifier(addr: &str) -> SettlementArgs {
        SettlementArgs {
            rpc_url: None,
            cmd: SettlementCmd::Verifier(VerifierArgs {
                address: addr.to_string(),
            }),
        }
    }

    fn s(buf: &Vec<u8>) -> String {
        String::from_utf8(buf.clone()).unwrap()
    }

    // ── status ──────────────────────────────────────────────────────────

    #[test]
    fn status_reports_all_gates_dormant_without_error() {
        let (client, fake) = make_client();
        seed_params(&fake, params_all_dormant());
        seed_head(&fake, 200_000);

        let mut buf = Vec::new();
        dispatch_core(args_status(), &client, &mut buf).unwrap();
        let out = s(&buf);
        assert!(out.contains("chain_id=1800100"), "chain_id line missing: {out}");
        assert!(out.contains("head=200000"), "head line missing: {out}");
        assert!(
            out.contains("inference_settlement_enabled_from_height = Dormant (unset)"),
            "settlement gate must render Dormant, not zero; got: {out}"
        );
        assert!(
            out.contains(
                "inference_settlement_consistency_enabled_from_height = Dormant (unset)"
            ),
            "consistency gate must render Dormant; got: {out}"
        );
        assert!(
            out.contains(
                "inference_verifier_bonding_enabled_from_height = Dormant (unset)"
            ),
            "bonding gate must render Dormant; got: {out}"
        );
        // Never fabricate zeros:
        assert!(!out.contains("= 0"), "no zeros should appear as gate values: {out}");
        assert!(!out.contains("= Active (since=0)"), "no Active-since-zero: {out}");
    }

    #[test]
    fn status_reports_mixed_active_scheduled() {
        let (client, fake) = make_client();
        // Settlement + consistency active from genesis; bonding scheduled.
        seed_params(
            &fake,
            json!({
                "finality_depth": 12,
                "min_fee": 100,
                "chain_id": 1_800_100,
                "inference_settlement_enabled_from_height": 0,
                "inference_settlement_consistency_enabled_from_height": 0,
                "inference_verifier_bonding_enabled_from_height": 500_000,
            }),
        );
        seed_head(&fake, 200_000);

        let mut buf = Vec::new();
        dispatch_core(args_status(), &client, &mut buf).unwrap();
        let out = s(&buf);
        assert!(
            out.contains("inference_settlement_enabled_from_height = Active (since=0)"),
            "{out}"
        );
        assert!(
            out.contains(
                "inference_settlement_consistency_enabled_from_height = Active (since=0)"
            ),
            "{out}"
        );
        assert!(
            out.contains(
                "inference_verifier_bonding_enabled_from_height = Scheduled@500000 (Δ=300000 blocks)"
            ),
            "{out}"
        );
    }

    // ── session ─────────────────────────────────────────────────────────

    #[test]
    fn session_settlement_dormant_errors_and_does_not_call_gated_rpcs() {
        let (client, fake) = make_client();
        seed_params(&fake, params_all_dormant());
        seed_head(&fake, 100_000);

        let mut buf = Vec::new();
        let err = dispatch_core(args_session("s-1"), &client, &mut buf).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("inference_settlement_enabled_from_height"),
            "error must name the dormant gate verbatim: {msg}"
        );
        assert!(msg.contains("observed=none"), "error must render observed=none: {msg}");
        assert!(msg.contains("head=100000"), "error must render head verbatim: {msg}");
        // No stdout on dormant error:
        assert!(s(&buf).is_empty(), "no partial stdout on dormant error");
        // No gated settlement RPC issued:
        let methods: Vec<String> =
            fake.calls().into_iter().map(|(m, _)| m).collect();
        assert!(
            !methods.iter().any(|m| m == "omninode_getInferenceSession"),
            "settlement RPC MUST NOT be called with dormant gate; methods={methods:?}"
        );
    }

    #[test]
    fn session_active_but_not_found_exits_ok_with_found_false() {
        let (client, fake) = make_client();
        seed_params(&fake, params_settlement_only_active(0));
        seed_head(&fake, 500_000);
        fake.set_response("omninode_getInferenceSession", Ok(serde_json::Value::Null));

        let mut buf = Vec::new();
        dispatch_core(args_session("s-missing"), &client, &mut buf).unwrap();
        let out = s(&buf);
        assert!(out.contains("session_id=s-missing"), "{out}");
        assert!(out.contains("found=false"), "{out}");
    }

    fn seed_multi_verifier_session(fake: &FakeJsonRpcTransport, bond_required: bool) {
        seed_params(fake, params_all_gates_active(0));
        seed_head(fake, 500_000);
        fake.set_response(
            "omninode_getInferenceSession",
            Ok(json!({
                "session_id": "s-mv",
                "consistency_required": false,
                "bond_required": bond_required,
                "max_verifiers": 3,
                "escrow_total": "1000",
                "escrow_remaining": "600",
                "claims_count": 2,
                "lifecycle": "active",
                "created_at_height": 400_000,
            })),
        );
        fake.set_response(
            "omninode_getInferenceClaims",
            Ok(json!({
                "session_id": "s-mv",
                "claims": [
                    {
                        "verifier_address": "v-A",
                        "claimed_at_height": 450_000,
                        "reward_amount": "200",
                        "state": "paid",
                        "paid_at_height": 450_001,
                    },
                    {
                        "verifier_address": "v-B",
                        "claimed_at_height": 450_050,
                        "reward_amount": "200",
                        "state": "pending",
                    },
                ]
            })),
        );
        fake.set_response(
            "omninode_getInferenceDisputes",
            Ok(json!({ "session_id": "s-mv", "disputes": [] })),
        );
        fake.set_response(
            "sum_listInferenceAttestations",
            Ok(json!([
                {
                    "session_id": "s-mv",
                    "verifier_address": "v-A",
                    "model_hash": "0xaa",
                    "manifest_root": "0xbb",
                    "response_hash": "0xcc",
                    "proof_root": "0xdd",
                    "verifier_signature": "0xsig-a",
                    "included_at_height": 440_000,
                    "tx_hash": "0xtx-a",
                    "finalized": true,
                },
                {
                    "session_id": "s-mv",
                    "verifier_address": "v-B",
                    "model_hash": "0xaa",
                    "manifest_root": "0xbb",
                    "response_hash": "0xcc",
                    "proof_root": "0xdd",
                    "verifier_signature": "0xsig-b",
                    "included_at_height": 440_500,
                    "tx_hash": "0xtx-b",
                    "finalized": true,
                }
            ])),
        );
    }

    #[test]
    fn session_multi_verifier_renders_both_verifiers() {
        let (client, fake) = make_client();
        seed_multi_verifier_session(&fake, /*bond_required*/ false);

        let mut buf = Vec::new();
        dispatch_core(args_session("s-mv"), &client, &mut buf).unwrap();
        let out = s(&buf);
        assert!(out.contains("session_id=s-mv"), "{out}");
        assert!(
            out.contains("mode: consistency_required=false bond_required=false"),
            "{out}"
        );
        assert!(out.contains("verifiers (2):"), "multi-verifier count line: {out}");
        assert!(out.contains("address=v-A"), "{out}");
        assert!(out.contains("address=v-B"), "{out}");
        assert!(out.contains("Paid(paid_at=450001, amount=200)"), "{out}");
        assert!(out.contains("Pending(claimed_at=450050, amount=200)"), "{out}");
        // Non-bond session: bond=none for each verifier.
        assert_eq!(
            out.matches("bond=none").count(),
            2,
            "each verifier should render bond=none for a non-bond session: {out}"
        );
    }

    #[test]
    fn session_bond_required_fetches_registry_and_renders_bond_summaries() {
        let (client, fake) = make_client();
        seed_multi_verifier_session(&fake, /*bond_required*/ true);
        fake.set_response(
            "omninode_getVerifier",
            Ok(json!({
                "verifier": "v-A",
                "bond": 10_000,
                "status": "Active",
                "registered_at_height": 100
            })),
        );
        // FakeJsonRpcTransport returns the same seeded response for
        // every call of the same method. We want distinct responses
        // for v-A vs v-B: swap in a two-shot dance via a small helper.
        // Simpler: seed once with the v-A response, dispatch, and
        // assert both verifiers received the same shape — the
        // multi-verifier registry fetch loop is what we're
        // exercising here, not per-address routing.
        let mut buf = Vec::new();
        dispatch_core(args_session("s-mv"), &client, &mut buf).unwrap();
        let out = s(&buf);
        // Both verifiers should have a bond summary populated from
        // the seeded getVerifier response.
        assert!(
            out.contains("bond: amount=10000 state=Bonded"),
            "at least one verifier must show bond_summary from registry: {out}"
        );
        // Registry RPC must have been called for each verifier in the
        // union set (2 verifiers → 2 calls).
        let calls = fake.calls();
        let verifier_calls: Vec<_> = calls
            .iter()
            .filter(|(m, _)| m == "omninode_getVerifier")
            .collect();
        assert_eq!(
            verifier_calls.len(),
            2,
            "expected 2 omninode_getVerifier calls (one per verifier), got {}: {calls:?}",
            verifier_calls.len()
        );
    }

    #[test]
    fn session_dispute_states_render_chain_public_terms() {
        // Chain-team clarification: the resolved dispute states are
        // publicly named `ResolvedAllowClaim` / `ResolvedDenyClaim`.
        // Three verifiers, one each of `open` / `resolved_approved` /
        // `resolved_denied` on the wire, exercise the entire display
        // mapping. Legacy internal-enum names must never leak into
        // operator-facing output.
        let (client, fake) = make_client();
        seed_params(&fake, params_all_gates_active(0));
        seed_head(&fake, 500_000);
        fake.set_response(
            "omninode_getInferenceSession",
            Ok(json!({
                "session_id": "s-disputes",
                "consistency_required": false,
                "bond_required": false,
                "max_verifiers": 3,
                "escrow_total": "1000",
                "escrow_remaining": "1000",
                "claims_count": 0,
                "lifecycle": "active",
                "created_at_height": 400_000,
            })),
        );
        fake.set_response(
            "omninode_getInferenceClaims",
            Ok(json!({ "session_id": "s-disputes", "claims": [] })),
        );
        fake.set_response(
            "omninode_getInferenceDisputes",
            Ok(json!({
                "session_id": "s-disputes",
                "disputes": [
                    {
                        "verifier_address": "v-open",
                        "opened_at_height": 460_000,
                        "state": "open"
                    },
                    {
                        "verifier_address": "v-allow",
                        "opened_at_height": 460_100,
                        "state": "resolved_approved",
                        "resolved_at_height": 465_000,
                        "approve_bps": 6600,
                        "deny_bps": 3400
                    },
                    {
                        "verifier_address": "v-deny",
                        "opened_at_height": 460_200,
                        "state": "resolved_denied",
                        "resolved_at_height": 466_000,
                        "approve_bps": 3300,
                        "deny_bps": 6700
                    }
                ]
            })),
        );
        fake.set_response("sum_listInferenceAttestations", Ok(json!([])));

        let mut buf = Vec::new();
        dispatch_core(args_session("s-disputes"), &client, &mut buf).unwrap();
        let out = s(&buf);

        // Open dispute appears on its verifier row.
        assert!(out.contains("address=v-open"), "{out}");
        assert!(
            out.contains("dispute=Open(opened_at=460000)"),
            "open dispute must render on the verifier row: {out}"
        );

        // Resolved-allow renders using the chain-public term.
        assert!(out.contains("address=v-allow"), "{out}");
        assert!(
            out.contains(
                "dispute=ResolvedAllowClaim(resolved_at=465000, approve_bps=6600, deny_bps=3400)"
            ),
            "resolved-allow must render as ResolvedAllowClaim: {out}"
        );

        // Resolved-deny renders using the chain-public term.
        assert!(out.contains("address=v-deny"), "{out}");
        assert!(
            out.contains(
                "dispute=ResolvedDenyClaim(resolved_at=466000, approve_bps=3300, deny_bps=6700)"
            ),
            "resolved-deny must render as ResolvedDenyClaim: {out}"
        );

        // Legacy internal-enum names must not leak into display.
        assert!(
            !out.contains("ResolvedApproved"),
            "legacy 'ResolvedApproved' must not leak into operator display: {out}"
        );
        assert!(
            !out.contains("ResolvedDenied"),
            "legacy 'ResolvedDenied' must not leak into operator display: {out}"
        );
    }

    #[test]
    fn session_json_output_is_parseable_and_carries_multi_verifier_array() {
        let (client, fake) = make_client();
        seed_multi_verifier_session(&fake, /*bond_required*/ false);

        let mut buf = Vec::new();
        dispatch_core(args_session_json("s-mv"), &client, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&s)
            .expect("--json output must be parseable JSON");
        assert_eq!(parsed["session_id"], "s-mv");
        let verifiers = parsed["verifiers"].as_array().expect("verifiers array");
        assert_eq!(verifiers.len(), 2, "multi-verifier array in JSON: {parsed}");
        assert_eq!(parsed["escrow_total"], "1000");
    }

    // ── claimable ───────────────────────────────────────────────────────

    #[test]
    fn claimable_mature_renders_ready() {
        let (client, fake) = make_client();
        seed_params(&fake, params_all_gates_active(0));
        seed_head(&fake, 500_000);
        fake.set_response(
            "omninode_getClaimableReward",
            Ok(json!({
                "session_id": "s-1",
                "verifier_address": "v-mature",
                "mature": true,
                "claim_ready_block": 499_000,
                "blocks_until_ready": 0,
                "escrow_available": true,
                "cap_available": true,
                "dispute_clear": true,
                "claimable_now": true,
                "reward_amount": "200"
            })),
        );

        let mut buf = Vec::new();
        dispatch_core(args_claimable("s-1", "v-mature"), &client, &mut buf).unwrap();
        let out = s(&buf);
        assert!(out.contains("mature=true"), "{out}");
        assert!(out.contains("claimable_now=true"), "{out}");
        assert!(out.contains("blocks_until_ready=0"), "{out}");
    }

    #[test]
    fn claimable_immature_renders_blocks_until_ready() {
        let (client, fake) = make_client();
        seed_params(&fake, params_all_gates_active(0));
        seed_head(&fake, 500_000);
        fake.set_response(
            "omninode_getClaimableReward",
            Ok(json!({
                "session_id": "s-1",
                "verifier_address": "v-imm",
                "mature": false,
                "claim_ready_block": 550_000,
                "blocks_until_ready": 50_000,
                "escrow_available": true,
                "cap_available": true,
                "dispute_clear": true,
                "claimable_now": false,
                "reward_amount": "200"
            })),
        );

        let mut buf = Vec::new();
        dispatch_core(args_claimable("s-1", "v-imm"), &client, &mut buf).unwrap();
        let out = s(&buf);
        assert!(out.contains("mature=false"), "{out}");
        assert!(out.contains("claimable_now=false"), "{out}");
        assert!(out.contains("blocks_until_ready=50000"), "{out}");
    }

    // ── verifier ────────────────────────────────────────────────────────

    #[test]
    fn verifier_bonding_dormant_errors_and_does_not_call_verifier_rpc() {
        let (client, fake) = make_client();
        // Settlement active but bonding absent.
        seed_params(&fake, params_settlement_only_active(0));
        seed_head(&fake, 500_000);

        let mut buf = Vec::new();
        let err = dispatch_core(args_verifier("v-1"), &client, &mut buf).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("inference_verifier_bonding_enabled_from_height"),
            "error must name the bonding gate: {msg}"
        );
        assert!(msg.contains("observed=none"), "must render observed=none: {msg}");
        assert!(msg.contains("head=500000"), "must render head verbatim: {msg}");
        let methods: Vec<String> =
            fake.calls().into_iter().map(|(m, _)| m).collect();
        assert!(
            !methods.iter().any(|m| m == "omninode_getVerifier"),
            "verifier RPC MUST NOT be called with dormant bonding gate; methods={methods:?}"
        );
    }

    // ── Clap drift safety ───────────────────────────────────────────────

    /// Drift safety for the `operator settlement …` subcommand tree.
    /// Fails if any of the four subcommands or their documented flags
    /// gets renamed or dropped from the clap tree.
    #[test]
    fn clap_drift_settlement_subcommand_tree_remains_registered() {
        // Wrap SettlementArgs in a probe Parser so we can inspect the
        // clap tree it produces (SettlementArgs itself derives Args,
        // not Parser).
        #[derive(Parser)]
        struct Probe {
            #[command(subcommand)]
            #[allow(dead_code)]
            cmd: SettlementCmd,
        }
        let cmd = <Probe as CommandFactory>::command();

        for sub in ["status", "session", "claimable", "verifier"] {
            assert!(
                cmd.find_subcommand(sub).is_some(),
                "operator settlement {sub} must be a registered subcommand"
            );
        }

        let session = cmd.find_subcommand("session").unwrap();
        assert!(
            session.get_arguments().any(|a| a.get_long() == Some("json")),
            "operator settlement session must accept --json"
        );

        let claimable = cmd.find_subcommand("claimable").unwrap();
        assert!(
            claimable
                .get_arguments()
                .any(|a| a.get_long() == Some("verifier")),
            "operator settlement claimable must accept --verifier"
        );
    }

    // ── Issue #85 — observability marker tests ─────────────────────────

    /// Shared-buffer `MakeWriter` for capturing tracing output inside a
    /// test scope. Each test constructs one, sets it as the current
    /// tracing subscriber via `tracing::subscriber::with_default`, runs
    /// `dispatch_core`, and then greps the captured bytes for marker
    /// strings.
    #[derive(Clone)]
    struct CapturedLogs(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl CapturedLogs {
        fn new() -> Self {
            Self(std::sync::Arc::new(std::sync::Mutex::new(Vec::new())))
        }
        fn as_string(&self) -> String {
            String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
        }
    }

    struct CapturedLogsWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
    impl std::io::Write for CapturedLogsWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLogs {
        type Writer = CapturedLogsWriter;
        fn make_writer(&'a self) -> Self::Writer {
            CapturedLogsWriter(self.0.clone())
        }
    }

    /// Run `dispatch_core` inside a scoped tracing subscriber that
    /// captures every event into a `CapturedLogs`. Returns the
    /// captured tracing bytes as a `String` PLUS the `dispatch_core`
    /// result, so the same test can inspect both.
    fn run_with_capture(
        args: SettlementArgs,
        client: &SumChainClient<FakeJsonRpcTransport>,
    ) -> (String, Vec<u8>, Result<(), anyhow::Error>) {
        let logs = CapturedLogs::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(logs.clone())
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .without_time()
            .finish();
        let mut stdout_buf = Vec::new();
        let result = tracing::subscriber::with_default(subscriber, || {
            dispatch_core(args, client, &mut stdout_buf)
        });
        (logs.as_string(), stdout_buf, result)
    }

    // ── Marker constant pin ────────────────────────────────────────────

    #[test]
    fn marker_constant_names_are_pinned() {
        // These string values are the operator-visible marker names.
        // Renaming any of them is a coordinated observability change
        // that must update runbooks + dashboards + tests. Do NOT
        // change the right-hand sides without a matching runbook /
        // downstream update.
        assert_eq!(markers::QUERY, "settlement_query");
        assert_eq!(markers::QUERY_OK, "settlement_query_ok");
        assert_eq!(markers::DORMANT, "settlement_dormant");
        assert_eq!(markers::VIEW_INCOMPLETE, "settlement_view_incomplete");
        assert_eq!(markers::QUERY_FAILED, "settlement_query_failed");
    }

    // ── Happy path — status emits QUERY + QUERY_OK ─────────────────────

    #[test]
    fn status_happy_path_emits_query_and_query_ok_markers() {
        let (client, fake) = make_client();
        seed_params(&fake, params_all_gates_active(0));
        seed_head(&fake, 500_000);

        let (logs, _stdout, result) = run_with_capture(args_status(), &client);
        result.expect("status happy path must succeed");

        assert!(
            logs.contains("event=\"settlement_query\""),
            "must emit QUERY marker; logs=\n{logs}"
        );
        assert!(
            logs.contains("event=\"settlement_query_ok\""),
            "must emit QUERY_OK marker; logs=\n{logs}"
        );
        assert!(
            logs.contains("subcommand=\"status\""),
            "must record subcommand field; logs=\n{logs}"
        );
        // Failure markers must NOT appear on the happy path.
        for banned in [
            "settlement_dormant",
            "settlement_view_incomplete",
            "settlement_query_failed",
        ] {
            assert!(
                !logs.contains(&format!("event=\"{banned}\"")),
                "happy path must not emit {banned}; logs=\n{logs}"
            );
        }
    }

    // ── Dormant branch — settlement dormant on `session` ───────────────

    #[test]
    fn session_dormant_emits_dormant_marker_with_gate_observed_head() {
        let (client, fake) = make_client();
        seed_params(&fake, params_all_dormant());
        seed_head(&fake, 100_000);

        let (logs, _stdout, result) = run_with_capture(args_session("s-1"), &client);
        result.expect_err("dormant must produce non-Ok result");

        assert!(
            logs.contains("event=\"settlement_dormant\""),
            "must emit DORMANT marker; logs=\n{logs}"
        );
        // Required structured fields per #85 spec:
        assert!(
            logs.contains("subcommand=\"session\""),
            "DORMANT marker must carry subcommand field; logs=\n{logs}"
        );
        assert!(
            logs.contains("session_id=\"s-1\""),
            "DORMANT marker must carry session_id field; logs=\n{logs}"
        );
        assert!(
            logs.contains("gate=\"inference_settlement_enabled_from_height\""),
            "DORMANT marker must carry gate field; logs=\n{logs}"
        );
        assert!(
            logs.contains("observed=\"None\"") || logs.contains("observed=None"),
            "DORMANT marker must carry observed=None; logs=\n{logs}"
        );
        assert!(
            logs.contains("head=100000"),
            "DORMANT marker must carry head field; logs=\n{logs}"
        );
        // QUERY_OK must NOT appear.
        assert!(
            !logs.contains("event=\"settlement_query_ok\""),
            "dormant path must not emit QUERY_OK; logs=\n{logs}"
        );
    }

    // ── View-incomplete branch ─────────────────────────────────────────

    #[test]
    fn session_view_incomplete_emits_view_incomplete_marker() {
        // Settlement gate active + consistency gate DORMANT, and the
        // session is consistency-mode. The `omninode_getInferenceConsistency`
        // RPC would be gated at RPC level (dormant), so the CLI's
        // session pipeline hits ViewIncomplete when the consistency
        // fetch fails first. Instead: settlement + consistency BOTH
        // active but the session is bond-required and bonding is
        // dormant → ViewIncomplete { Bonding, .. } via compose guard.
        //
        // Simplest path: settlement + consistency active, bonding
        // dormant; session marked bond_required=true. compose fires
        // the view-incomplete branch after the RPCs succeed.
        let (client, fake) = make_client();
        seed_params(
            &fake,
            json!({
                "finality_depth": 12,
                "min_fee": 100,
                "chain_id": 1_800_100,
                "inference_settlement_enabled_from_height": 0,
                "inference_settlement_consistency_enabled_from_height": 0,
                // bonding intentionally omitted → dormant
            }),
        );
        seed_head(&fake, 500_000);
        fake.set_response(
            "omninode_getInferenceSession",
            Ok(json!({
                "session_id": "s-vi",
                "consistency_required": false,
                "bond_required": true,
                "max_verifiers": 3,
                "escrow_total": "1000",
                "escrow_remaining": "1000",
                "claims_count": 0,
                "lifecycle": "active",
                "created_at_height": 400_000,
            })),
        );
        fake.set_response(
            "omninode_getInferenceClaims",
            Ok(json!({ "session_id": "s-vi", "claims": [] })),
        );
        fake.set_response(
            "omninode_getInferenceDisputes",
            Ok(json!({ "session_id": "s-vi", "disputes": [] })),
        );
        fake.set_response("sum_listInferenceAttestations", Ok(json!([])));

        let (logs, _stdout, result) = run_with_capture(args_session("s-vi"), &client);
        result.expect_err("view-incomplete must produce non-Ok result");

        assert!(
            logs.contains("event=\"settlement_view_incomplete\""),
            "must emit VIEW_INCOMPLETE marker; logs=\n{logs}"
        );
        assert!(
            logs.contains("subcommand=\"session\""),
            "VIEW_INCOMPLETE marker must carry subcommand; logs=\n{logs}"
        );
        assert!(
            logs.contains("session_id=\"s-vi\""),
            "VIEW_INCOMPLETE marker must carry session_id; logs=\n{logs}"
        );
        assert!(
            logs.contains("gate=\"inference_verifier_bonding_enabled_from_height\""),
            "VIEW_INCOMPLETE marker must carry missing gate; logs=\n{logs}"
        );
        assert!(
            !logs.contains("event=\"settlement_query_ok\""),
            "view-incomplete path must not emit QUERY_OK; logs=\n{logs}"
        );
    }

    // ── Query-failed branch — RPC error ────────────────────────────────

    #[test]
    fn claimable_rpc_error_emits_query_failed_marker() {
        use omni_zkml::ChainClientError;
        let (client, fake) = make_client();
        seed_params(&fake, params_all_gates_active(0));
        seed_head(&fake, 500_000);
        fake.set_response(
            "omninode_getClaimableReward",
            Err(ChainClientError::Other("simulated RPC outage".into())),
        );

        let (logs, _stdout, result) = run_with_capture(
            args_claimable("s-1", "v-mature"),
            &client,
        );
        result.expect_err("RPC error must produce non-Ok result");

        assert!(
            logs.contains("event=\"settlement_query_failed\""),
            "must emit QUERY_FAILED marker; logs=\n{logs}"
        );
        assert!(
            logs.contains("subcommand=\"claimable\""),
            "QUERY_FAILED marker must carry subcommand; logs=\n{logs}"
        );
        assert!(
            logs.contains("session_id=\"s-1\""),
            "QUERY_FAILED marker must carry session_id; logs=\n{logs}"
        );
        assert!(
            logs.contains("address=\"v-mature\""),
            "QUERY_FAILED marker must carry address; logs=\n{logs}"
        );
        assert!(
            logs.contains("category=\"chain_rpc\""),
            "QUERY_FAILED marker must carry category='chain_rpc'; logs=\n{logs}"
        );
        assert!(
            logs.contains("simulated RPC outage"),
            "QUERY_FAILED marker must include underlying error; logs=\n{logs}"
        );
    }

    // ── Query-failed branch — wire parse error ─────────────────────────

    #[test]
    fn claimable_wire_parse_error_emits_query_failed_with_malformed_category() {
        // Chain-active seed + malformed JSON response so the DTO
        // deserialise fails on the claim reward, driving the
        // `SettlementReadError::WireParse` path.
        let (client, fake) = make_client();
        seed_params(&fake, params_all_gates_active(0));
        seed_head(&fake, 500_000);
        fake.set_response(
            "omninode_getClaimableReward",
            Ok(json!({ "not_the_right_shape": true })),
        );

        let (logs, _stdout, result) = run_with_capture(
            args_claimable("s-1", "v-mature"),
            &client,
        );
        result.expect_err("wire parse error must produce non-Ok result");

        assert!(
            logs.contains("event=\"settlement_query_failed\""),
            "must emit QUERY_FAILED marker; logs=\n{logs}"
        );
        assert!(
            logs.contains("category=\"chain_response_malformed\""),
            "wire parse failure must carry category='chain_response_malformed'; logs=\n{logs}"
        );
    }

    // ── Direct RPC failure (no SettlementReadError) emits fields ──────

    #[test]
    fn status_direct_rpc_failure_emits_query_failed_with_empty_session_and_address() {
        // `chain_getChainParams` is called BEFORE any settlement-specific
        // gated RPC. Its failure never routes through
        // `SettlementReadError::Rpc`; the emit_query_failed helper is
        // the only marker source. Test pins that the emitted marker
        // carries subcommand + session_id="" + address="" +
        // category="chain_rpc" + error.
        use omni_zkml::ChainClientError;
        let (client, fake) = make_client();
        fake.set_response(
            "chain_getChainParams",
            Err(ChainClientError::Other("simulated params outage".into())),
        );
        // `chain_getBlockHeight` also seeded so the fake doesn't fall
        // through to its "no response configured" default if fetched.
        seed_head(&fake, 500_000);

        let (logs, _stdout, result) = run_with_capture(args_status(), &client);
        result.expect_err("direct RPC failure must produce non-Ok result");

        assert!(
            logs.contains("event=\"settlement_query_failed\""),
            "must emit QUERY_FAILED marker; logs=\n{logs}"
        );
        assert!(
            logs.contains("subcommand=\"status\""),
            "QUERY_FAILED marker must carry subcommand; logs=\n{logs}"
        );
        // The subcommand has no session_id / address context — the
        // fields still must be emitted verbatim as empty strings so
        // downstream grep / filter rules stay uniform.
        assert!(
            logs.contains("session_id=\"\""),
            "QUERY_FAILED marker must carry empty session_id when absent; logs=\n{logs}"
        );
        assert!(
            logs.contains("address=\"\""),
            "QUERY_FAILED marker must carry empty address when absent; logs=\n{logs}"
        );
        assert!(
            logs.contains("category=\"chain_rpc\""),
            "QUERY_FAILED marker must carry category='chain_rpc'; logs=\n{logs}"
        );
        assert!(
            logs.contains("simulated params outage"),
            "QUERY_FAILED marker must carry the underlying error text; logs=\n{logs}"
        );
    }

    // ── Verifier happy path emits markers ──────────────────────────────

    #[test]
    fn verifier_happy_path_emits_markers_with_address() {
        let (client, fake) = make_client();
        seed_params(&fake, params_all_gates_active(0));
        seed_head(&fake, 500_000);
        fake.set_response(
            "omninode_getVerifier",
            Ok(json!({
                "verifier": "v-1",
                "bond": 10_000,
                "status": "Active",
                "registered_at_height": 100
            })),
        );

        let (logs, _stdout, result) = run_with_capture(args_verifier("v-1"), &client);
        result.expect("verifier happy path must succeed");

        assert!(
            logs.contains("event=\"settlement_query\""),
            "must emit QUERY marker; logs=\n{logs}"
        );
        assert!(
            logs.contains("event=\"settlement_query_ok\""),
            "must emit QUERY_OK marker; logs=\n{logs}"
        );
        assert!(
            logs.contains("subcommand=\"verifier\""),
            "must record subcommand=verifier; logs=\n{logs}"
        );
        assert!(
            logs.contains("address=\"v-1\""),
            "must record address field; logs=\n{logs}"
        );
    }

    // ── Verifier renders the canonical record; null → found=false ──────

    #[test]
    fn verifier_canonical_record_renders_and_null_is_found_false() {
        let (client, fake) = make_client();
        seed_params(&fake, params_all_gates_active(0));
        seed_head(&fake, 500_000);
        fake.set_response(
            "omninode_getVerifier",
            Ok(json!({
                "verifier": "v-1",
                "bond": 10_000,
                "status": "Active",
                "registered_at_height": 777,
                "unbonding_started_height": 800,
                "unlock_height": 900
            })),
        );
        let (_logs, stdout, result) = run_with_capture(args_verifier("v-1"), &client);
        result.expect("canonical verifier record must render");
        let out = String::from_utf8(stdout).expect("utf8 stdout");
        assert!(out.contains("verifier=v-1"), "{out}");
        assert!(out.contains("bond=10000"), "{out}");
        assert!(out.contains("status=Active"), "{out}");
        assert!(out.contains("registered_at_height=777"), "{out}");
        assert!(out.contains("unbonding_started_height=800"), "{out}");
        assert!(out.contains("unlock_height=900"), "{out}");
        // No leftover legacy field names in operator-facing output.
        assert!(!out.contains("bond_amount="), "{out}");
        assert!(!out.contains("bond_state="), "{out}");
        assert!(!out.contains("slash_history"), "{out}");

        // A null record renders `found=false`, not a parse failure.
        let (client2, fake2) = make_client();
        seed_params(&fake2, params_all_gates_active(0));
        seed_head(&fake2, 500_000);
        fake2.set_response("omninode_getVerifier", Ok(serde_json::Value::Null));
        let (_logs2, stdout2, result2) = run_with_capture(args_verifier("ghost"), &client2);
        result2.expect("null verifier must succeed as found=false");
        let out2 = String::from_utf8(stdout2).expect("utf8 stdout");
        assert!(out2.contains("address=ghost found=false"), "{out2}");
    }

    // ── Issue #87 — settlement claim tests ─────────────────────────────

    #[cfg(feature = "settlement-submit")]
    mod claim {
        use super::*;
        use crate::operator::SeedSource;
        use omni_sumchain::settlement_submit::{
            Address, ClaimInferenceRewardRequest, InferenceSettlementOperation,
            InferenceSettlementTxData, SignedTransaction, TransactionV2, TxPayload,
        };

        const TEST_SEED: [u8; 32] = [7u8; 32];
        const TEST_CHAIN_ID: u64 = 1_800_100;
        const TEST_NONCE: u64 = 123;
        const TEST_FEE: u128 = 1000;

        // ── Test-only fixture — lives inside the omni-node test module ──
        //
        // Previously exposed as `omni_sumchain::settlement_submit::fixtures`,
        // but that made the fixture reachable from any downstream consumer
        // compiling `settlement-submit`. Moved here so the test-only surface
        // never leaks into production API.
        //
        // Uses the small hex-encode helper from tx.rs (public within the
        // crate) and the chain-primitives types re-exported by
        // `omni_sumchain::settlement_submit`.

        pub(super) struct TestClaimTx {
            pub(super) unsigned_hex: String,
            pub(super) signing_hash_hex: String,
            pub(super) from_b58: String,
            pub(super) tx: TransactionV2,
        }

        fn encode_hex_local(bytes: &[u8]) -> String {
            use std::fmt::Write;
            let mut s = String::with_capacity(bytes.len() * 2);
            for b in bytes {
                let _ = write!(&mut s, "{b:02x}");
            }
            s
        }

        pub(super) fn build_test_claim_tx(
            seed: &[u8; 32],
            session_id: &str,
            chain_id: u64,
            nonce: u64,
            fee: u128,
        ) -> TestClaimTx {
            let pubkey = omni_zkml::signer_pubkey_bytes(seed)
                .expect("signer_pubkey_bytes");
            let from = Address::from_public_key(&pubkey);
            let from_b58 = from.to_base58();
            let tx = TransactionV2 {
                chain_id: chain_id.into(),
                from,
                fee: fee.into(),
                nonce: nonce.into(),
                payload: TxPayload::InferenceSettlement(InferenceSettlementTxData {
                    operation: InferenceSettlementOperation::ClaimReward(
                        ClaimInferenceRewardRequest {
                            session_id: session_id.to_string(),
                        },
                    ),
                }),
            };
            let unsigned_hex = format!("0x{}", encode_hex_local(&tx.to_bytes()));
            let signing_hash = tx.signing_hash();
            let signing_hash_hex =
                format!("0x{}", encode_hex_local(signing_hash.as_bytes()));
            TestClaimTx {
                unsigned_hex,
                signing_hash_hex,
                from_b58,
                tx,
            }
        }

        fn params_settlement_active() -> serde_json::Value {
            json!({
                "finality_depth": 12,
                "min_fee": 100,
                "chain_id": TEST_CHAIN_ID,
                "inference_settlement_enabled_from_height": 0,
            })
        }

        fn args_claim(session_id: &str) -> Box<ClaimArgs> {
            Box::new(ClaimArgs {
                session_id: session_id.to_string(),
                verifier: None,
                fee: None,
                dry_run: false,
            })
        }

        fn args_claim_dry_run(session_id: &str) -> Box<ClaimArgs> {
            Box::new(ClaimArgs {
                session_id: session_id.to_string(),
                verifier: None,
                fee: None,
                dry_run: true,
            })
        }

        fn seed_full_happy_path(fake: &FakeJsonRpcTransport) -> TestClaimTx {
            let fixture = build_test_claim_tx(
                &TEST_SEED,
                "s-1",
                TEST_CHAIN_ID,
                TEST_NONCE,
                TEST_FEE,
            );
            seed_params(fake, params_settlement_active());
            seed_head(fake, 500_000);
            fake.set_response(
                "sum_getInferenceAttestation",
                Ok(json!({
                    "session_id": "s-1",
                    "verifier_address": fixture.from_b58,
                    "model_hash": "0xaa",
                    "manifest_root": "0xbb",
                    "response_hash": "0xcc",
                    "proof_root": "0xdd",
                    "verifier_signature": "0xsig",
                    "included_at_height": 440_000,
                    "tx_hash": "0xtx-att",
                    "finalized": true,
                })),
            );
            fake.set_response(
                "omninode_getClaimableReward",
                Ok(json!({
                    "session_id": "s-1",
                    "verifier_address": fixture.from_b58,
                    "mature": true,
                    "claim_ready_block": 499_000,
                    "blocks_until_ready": 0,
                    "escrow_available": true,
                    "cap_available": true,
                    "dispute_clear": true,
                    "claimable_now": true,
                    "reward_amount": "200"
                })),
            );
            fake.set_response(
                "omninode_getInferenceSession",
                Ok(json!({
                    "session_id": "s-1",
                    "consistency_required": false,
                    "bond_required": false,
                    "max_verifiers": 3,
                    "escrow_total": "1000",
                    "escrow_remaining": "1000",
                    "claims_count": 0,
                    "lifecycle": "active",
                    "created_at_height": 400_000,
                })),
            );
            fake.set_response(
                "omninode_buildClaimInferenceReward",
                Ok(json!({
                    "unsigned_tx": fixture.unsigned_hex,
                    "signing_hash": fixture.signing_hash_hex,
                    "from": fixture.from_b58,
                    "nonce": TEST_NONCE,
                    "fee": TEST_FEE,
                    "chain_id": TEST_CHAIN_ID,
                })),
            );
            fake.set_response(
                "sum_sendRawTransaction",
                Ok(json!({ "tx_hash": "0xtxhash-happy" })),
            );
            fixture
        }

        fn run_with_capture_claim(
            args: SettlementArgs,
            client: &SumChainClient<FakeJsonRpcTransport>,
            seed_source: SeedSource,
        ) -> (String, Vec<u8>, Result<()>) {
            let logs = CapturedLogs::new();
            let subscriber = tracing_subscriber::fmt()
                .with_writer(logs.clone())
                .with_max_level(tracing::Level::TRACE)
                .with_ansi(false)
                .without_time()
                .finish();
            let mut stdout_buf = Vec::new();
            let result = tracing::subscriber::with_default(subscriber, || {
                let SettlementArgs { cmd, .. } = args;
                match cmd {
                    SettlementCmd::Claim(a) => {
                        super::super::run_claim(client, &a, &mut stdout_buf, seed_source)
                    }
                    _ => panic!("run_with_capture_claim called with non-Claim variant"),
                }
            });
            (logs.as_string(), stdout_buf, result)
        }

        fn call_methods(fake: &FakeJsonRpcTransport) -> Vec<String> {
            fake.calls().into_iter().map(|(m, _)| m).collect()
        }

        // ── Test 1 — dormancy refuses before builder/submit ────────

        #[test]
        fn claim_dormant_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            seed_params(&fake, params_all_dormant());
            seed_head(&fake, 100_000);

            let (logs, _stdout, result) = run_with_capture_claim(
                SettlementArgs { rpc_url: None, cmd: SettlementCmd::Claim(args_claim("s-1")) },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            result.expect_err("dormant must produce non-Ok");

            assert!(
                logs.contains("event=\"settlement_claim_refused_dormancy\""),
                "must emit CLAIM_REFUSED_DORMANCY; logs=\n{logs}"
            );
            assert!(
                logs.contains("gate=\"inference_settlement_enabled_from_height\""),
                "must carry gate field; logs=\n{logs}"
            );
            let methods = call_methods(&fake);
            assert!(
                !methods.iter().any(|m| m == "omninode_buildClaimInferenceReward"),
                "builder must not be called; methods={methods:?}"
            );
            assert!(
                !methods.iter().any(|m| m == "sum_sendRawTransaction"),
                "submit must not be called; methods={methods:?}"
            );
        }

        // ── Test 2 — missing attestation refuses before builder/submit ──

        #[test]
        fn claim_missing_attestation_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            seed_params(&fake, params_settlement_active());
            seed_head(&fake, 500_000);
            fake.set_response(
                "sum_getInferenceAttestation",
                Ok(serde_json::Value::Null),
            );

            let (logs, _stdout, result) = run_with_capture_claim(
                SettlementArgs { rpc_url: None, cmd: SettlementCmd::Claim(args_claim("s-1")) },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            result.expect_err("missing attestation must produce non-Ok");

            assert!(
                logs.contains("event=\"settlement_claim_failed\""),
                "must emit CLAIM_FAILED; logs=\n{logs}"
            );
            assert!(
                logs.contains("category=\"attestation_not_found\""),
                "must carry attestation_not_found category; logs=\n{logs}"
            );
            let methods = call_methods(&fake);
            assert!(!methods.iter().any(|m| m == "omninode_buildClaimInferenceReward"));
            assert!(!methods.iter().any(|m| m == "sum_sendRawTransaction"));
        }

        // ── Test 3 — authority mismatch refuses before builder/submit ──

        #[test]
        fn claim_authority_mismatch_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            seed_params(&fake, params_settlement_active());
            seed_head(&fake, 500_000);
            fake.set_response(
                "sum_getInferenceAttestation",
                Ok(json!({
                    "session_id": "s-1",
                    "verifier_address": "v-someone-else",
                    "model_hash": "0xaa",
                    "manifest_root": "0xbb",
                    "response_hash": "0xcc",
                    "proof_root": "0xdd",
                    "verifier_signature": "0xsig",
                    "included_at_height": 440_000,
                    "tx_hash": "0xtx-att",
                    "finalized": true,
                })),
            );

            let (logs, _stdout, result) = run_with_capture_claim(
                SettlementArgs { rpc_url: None, cmd: SettlementCmd::Claim(args_claim("s-1")) },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            result.expect_err("authority mismatch must produce non-Ok");

            assert!(logs.contains("event=\"settlement_claim_refused_authority\""));
            let methods = call_methods(&fake);
            assert!(!methods.iter().any(|m| m == "omninode_buildClaimInferenceReward"));
            assert!(!methods.iter().any(|m| m == "sum_sendRawTransaction"));
        }

        // ── Test 4 — immature refuses before builder/submit ────────

        #[test]
        fn claim_immature_refuses_and_never_calls_builder_or_submit() {
            let fixture =
                build_test_claim_tx(&TEST_SEED, "s-1", TEST_CHAIN_ID, TEST_NONCE, TEST_FEE);
            let (client, fake) = make_client();
            seed_params(&fake, params_settlement_active());
            seed_head(&fake, 400_000);
            fake.set_response(
                "sum_getInferenceAttestation",
                Ok(json!({
                    "session_id": "s-1",
                    "verifier_address": fixture.from_b58,
                    "model_hash": "0xaa",
                    "manifest_root": "0xbb",
                    "response_hash": "0xcc",
                    "proof_root": "0xdd",
                    "verifier_signature": "0xsig",
                    "included_at_height": 380_000,
                    "tx_hash": "0xtx-att",
                    "finalized": true,
                })),
            );
            fake.set_response(
                "omninode_getClaimableReward",
                Ok(json!({
                    "session_id": "s-1",
                    "verifier_address": fixture.from_b58,
                    "mature": false,
                    "claim_ready_block": 500_000,
                    "blocks_until_ready": 100_000,
                    "escrow_available": true,
                    "cap_available": true,
                    "dispute_clear": true,
                    "claimable_now": false,
                    "reward_amount": "200"
                })),
            );

            let (logs, _stdout, result) = run_with_capture_claim(
                SettlementArgs { rpc_url: None, cmd: SettlementCmd::Claim(args_claim("s-1")) },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            result.expect_err("immature must produce non-Ok");

            assert!(logs.contains("event=\"settlement_claim_refused_maturity\""));
            assert!(logs.contains("claim_ready_block=500000"));
            assert!(logs.contains("blocks_until_ready=100000"));
            let methods = call_methods(&fake);
            assert!(!methods.iter().any(|m| m == "omninode_buildClaimInferenceReward"));
            assert!(!methods.iter().any(|m| m == "sum_sendRawTransaction"));
        }

        // ── Test 5 — bond precheck failure refuses before builder/submit ──

        #[test]
        fn claim_bond_precheck_failed_refuses_and_never_calls_builder_or_submit() {
            let fixture =
                build_test_claim_tx(&TEST_SEED, "s-1", TEST_CHAIN_ID, TEST_NONCE, TEST_FEE);
            let (client, fake) = make_client();
            seed_params(
                &fake,
                json!({
                    "finality_depth": 12,
                    "min_fee": 100,
                    "chain_id": TEST_CHAIN_ID,
                    "inference_settlement_enabled_from_height": 0,
                    "inference_verifier_bonding_enabled_from_height": 0,
                }),
            );
            seed_head(&fake, 500_000);
            fake.set_response(
                "sum_getInferenceAttestation",
                Ok(json!({
                    "session_id": "s-1",
                    "verifier_address": fixture.from_b58,
                    "model_hash": "0xaa",
                    "manifest_root": "0xbb",
                    "response_hash": "0xcc",
                    "proof_root": "0xdd",
                    "verifier_signature": "0xsig",
                    "included_at_height": 440_000,
                    "tx_hash": "0xtx-att",
                    "finalized": true,
                })),
            );
            fake.set_response(
                "omninode_getClaimableReward",
                Ok(json!({
                    "session_id": "s-1",
                    "verifier_address": fixture.from_b58,
                    "mature": true,
                    "claim_ready_block": 499_000,
                    "blocks_until_ready": 0,
                    "escrow_available": true,
                    "cap_available": true,
                    "dispute_clear": true,
                    "claimable_now": true,
                    "reward_amount": "200"
                })),
            );
            fake.set_response(
                "omninode_getInferenceSession",
                Ok(json!({
                    "session_id": "s-1",
                    "consistency_required": false,
                    "bond_required": true,          // <-- forces bond precheck
                    "max_verifiers": 3,
                    "escrow_total": "1000",
                    "escrow_remaining": "1000",
                    "claims_count": 0,
                    "lifecycle": "active",
                    "created_at_height": 400_000,
                })),
            );
            fake.set_response(
                "omninode_getVerifier",
                Ok(json!({
                    "verifier": fixture.from_b58,
                    "bond": 0,
                    "status": "Withdrawn",           // <-- not Active/Bonded
                    "registered_at_height": 100
                })),
            );

            let (logs, _stdout, result) = run_with_capture_claim(
                SettlementArgs { rpc_url: None, cmd: SettlementCmd::Claim(args_claim("s-1")) },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            result.expect_err("bond precheck failure must produce non-Ok");

            assert!(logs.contains("event=\"settlement_claim_refused_bond_precheck\""));
            let methods = call_methods(&fake);
            assert!(!methods.iter().any(|m| m == "omninode_buildClaimInferenceReward"));
            assert!(!methods.iter().any(|m| m == "sum_sendRawTransaction"));
        }

        // ── Test 6 — builder response mismatch refuses BEFORE submit ──

        #[test]
        fn claim_builder_envelope_mismatch_refuses_before_submit() {
            let fixture = seed_full_happy_path(&FakeJsonRpcTransport::new()); // fixture only, drop the fake
            let (client, fake) = make_client();
            let _ = seed_full_happy_path(&fake);
            // Overwrite builder to return the WRONG `from`:
            fake.set_response(
                "omninode_buildClaimInferenceReward",
                Ok(json!({
                    "unsigned_tx": fixture.unsigned_hex,
                    "signing_hash": fixture.signing_hash_hex,
                    "from": "not-the-derived-address",
                    "nonce": TEST_NONCE,
                    "fee": TEST_FEE,
                    "chain_id": TEST_CHAIN_ID,
                })),
            );

            let (logs, _stdout, result) = run_with_capture_claim(
                SettlementArgs { rpc_url: None, cmd: SettlementCmd::Claim(args_claim("s-1")) },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            result.expect_err("envelope mismatch must produce non-Ok");

            assert!(logs.contains("event=\"settlement_claim_failed\""));
            assert!(logs.contains("category=\"builder_mismatch\""));
            let methods = call_methods(&fake);
            assert!(
                methods.iter().any(|m| m == "omninode_buildClaimInferenceReward"),
                "builder WAS called for envelope check; methods={methods:?}"
            );
            assert!(
                !methods.iter().any(|m| m == "sum_sendRawTransaction"),
                "submit MUST NOT be called after envelope mismatch; methods={methods:?}"
            );
        }

        // ── Test 7 — dry-run calls builder but never signs/submits ──

        #[test]
        fn claim_dry_run_calls_builder_but_never_signs_or_submits() {
            let (client, fake) = make_client();
            let _fixture = seed_full_happy_path(&fake);

            let (logs, stdout, result) = run_with_capture_claim(
                SettlementArgs {
                    rpc_url: None,
                    cmd: SettlementCmd::Claim(args_claim_dry_run("s-1")),
                },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            result.expect("dry-run happy path must succeed");

            assert!(
                logs.contains("event=\"settlement_claim_ready\""),
                "must emit CLAIM_READY; logs=\n{logs}"
            );
            let stdout_str = String::from_utf8(stdout).unwrap();
            assert!(stdout_str.contains("dry_run=true submitted=false"), "{stdout_str}");
            let methods = call_methods(&fake);
            assert!(
                methods.iter().any(|m| m == "omninode_buildClaimInferenceReward"),
                "builder MUST be called on dry-run; methods={methods:?}"
            );
            assert!(
                !methods.iter().any(|m| m == "sum_sendRawTransaction"),
                "submit MUST NOT be called on dry-run; methods={methods:?}"
            );
            assert!(
                !logs.contains("event=\"settlement_claim_submitted\""),
                "no CLAIM_SUBMITTED on dry-run; logs=\n{logs}"
            );
        }

        // ── Test 8 — happy path calls builder once + submit once ──

        #[test]
        fn claim_happy_path_calls_builder_once_and_submit_once() {
            let (client, fake) = make_client();
            let _fixture = seed_full_happy_path(&fake);

            let (logs, stdout, result) = run_with_capture_claim(
                SettlementArgs { rpc_url: None, cmd: SettlementCmd::Claim(args_claim("s-1")) },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            result.expect("happy path must succeed");

            assert!(logs.contains("event=\"settlement_claim_ready\""));
            assert!(logs.contains("event=\"settlement_claim_submitted\""));
            assert!(logs.contains("tx_hash=\"0xtxhash-happy\""));
            let stdout_str = String::from_utf8(stdout).unwrap();
            assert!(stdout_str.contains("submitted=true"), "{stdout_str}");
            assert!(stdout_str.contains("tx_hash=0xtxhash-happy"), "{stdout_str}");

            let methods = call_methods(&fake);
            let builder_count = methods
                .iter()
                .filter(|m| m.as_str() == "omninode_buildClaimInferenceReward")
                .count();
            let submit_count = methods
                .iter()
                .filter(|m| m.as_str() == "sum_sendRawTransaction")
                .count();
            assert_eq!(builder_count, 1, "builder called exactly once; got {builder_count}, methods={methods:?}");
            assert_eq!(submit_count, 1, "submit called exactly once; got {submit_count}, methods={methods:?}");
        }

        // ── Test 9 — submitted raw tx round-trips through SignedTransaction::from_hex ──

        #[test]
        fn claim_submitted_hex_round_trips_via_from_hex_not_raw_concat() {
            let (client, fake) = make_client();
            let fixture = seed_full_happy_path(&fake);

            let (_logs, _stdout, result) = run_with_capture_claim(
                SettlementArgs { rpc_url: None, cmd: SettlementCmd::Claim(args_claim("s-1")) },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            result.expect("happy path");

            // Extract the hex that was sent to sum_sendRawTransaction.
            let calls = fake.calls();
            let (_m, params) = calls
                .iter()
                .find(|(m, _)| m == "sum_sendRawTransaction")
                .expect("submit RPC must have been called");
            let hex_param = params
                .as_array().expect("params array")
                .first().expect("params[0]")
                .as_str().expect("hex string")
                .to_string();

            // 9a. Round-trips through the canonical decoder.
            let decoded = SignedTransaction::from_hex(&hex_param)
                .expect("submitted hex must round-trip via SignedTransaction::from_hex");
            // 9b. Decoded tx equals the fixture tx (proves we didn't reorder
            //     or corrupt fields).
            let inner_tx = decoded
                .staking_data()
                .map(|_| unreachable!("wrong payload"))
                .unwrap_or(());
            let _ = inner_tx; // silence unused-var; the assertion below is stronger.
            // 9c. Signing hash agrees with the fixture.
            assert_eq!(decoded.signing_hash(), fixture.tx.signing_hash());

            // 10. NOT a raw concat — a raw concat would omit the
            //     `TxInner::V2` discriminant, which bincode adds as an
            //     enum tag byte. Verify the submitted hex is strictly
            //     LONGER than raw concat would be.
            let raw_concat_bytes =
                fixture.unsigned_hex.strip_prefix("0x").unwrap().len() / 2 + 64 + 32;
            let submitted_len = hex_param.strip_prefix("0x").map(|s| s.len()).unwrap_or(hex_param.len()) / 2;
            assert!(
                submitted_len > raw_concat_bytes,
                "submitted wire ({submitted_len} bytes) must include the TxInner::V2 \
                 enum-tag wrapper — pure concat would only be {raw_concat_bytes} bytes"
            );
        }

        // ── Test 11 — marker constants pinned ─────────────────────

        #[test]
        fn claim_marker_constant_names_are_pinned() {
            use super::super::claim_markers;
            assert_eq!(claim_markers::CLAIM_READY, "settlement_claim_ready");
            assert_eq!(claim_markers::CLAIM_REFUSED_DORMANCY, "settlement_claim_refused_dormancy");
            assert_eq!(claim_markers::CLAIM_REFUSED_MATURITY, "settlement_claim_refused_maturity");
            assert_eq!(claim_markers::CLAIM_REFUSED_AUTHORITY, "settlement_claim_refused_authority");
            assert_eq!(claim_markers::CLAIM_REFUSED_BOND_PRECHECK, "settlement_claim_refused_bond_precheck");
            assert_eq!(claim_markers::CLAIM_SUBMITTED, "settlement_claim_submitted");
            assert_eq!(claim_markers::CLAIM_FAILED, "settlement_claim_failed");
        }

        // ── Test 12 — no claim-on-behalf / coordinator / contributor-triggered path ──

        /// The public `ClaimArgs` surface + `SettlementCmd` variants
        /// carry NO on-behalf / coordinator / contributor flags. This
        /// test compiles as long as `ClaimArgs`'s fields are the
        /// documented four (`session_id`, `verifier`, `fee`, `dry_run`)
        /// and no additional variants appear on `SettlementCmd`. If a
        /// future edit adds any of the banned surfaces, this test
        /// fails to compile — a grep-visible guard.
        #[test]
        fn claim_surface_carries_no_forbidden_flags() {
            // Compile-time destructure — any new field breaks the
            // pattern.
            let _ = ClaimArgs {
                session_id: "s".to_string(),
                verifier: None,
                fee: None,
                dry_run: false,
            };
            // Variant enumeration — if a new variant is added, the
            // exhaustive match fails to compile.
            fn variant_check(cmd: &SettlementCmd) -> &'static str {
                match cmd {
                    SettlementCmd::Status => "status",
                    SettlementCmd::Session(_) => "session",
                    SettlementCmd::Claimable(_) => "claimable",
                    SettlementCmd::Verifier(_) => "verifier",
                    SettlementCmd::Claim(_) => "claim",
                    SettlementCmd::RegisterVerifier(_) => "register-verifier",
                    #[cfg(feature = "settlement-dispute")]
                    SettlementCmd::Dispute(_) => "dispute",
                }
            }
            let _ = variant_check;
            // Runtime grep against the module source is done at code-
            // review time; no runtime source-string check needed here.
        }
    }

    // ── Issue #100 — settlement register-verifier tests ────────────────

    #[cfg(feature = "settlement-submit")]
    mod register_verifier {
        use super::*;
        use crate::operator::SeedSource;
        use omni_sumchain::settlement_submit::{
            verify_register_verifier_transaction, Address,
            ClaimInferenceRewardRequest, InferenceSettlementOperation,
            InferenceSettlementTxData, RegisterVerifierRequest,
            SettlementBuilderEnvelope, SignedTransaction, TransactionV2, TxInner,
            TxPayload,
        };

        const TEST_SEED: [u8; 32] = [7u8; 32];
        const TEST_CHAIN_ID: u64 = 1_800_100;
        const TEST_NONCE: u64 = 7;
        const TEST_FEE: u128 = 1000;
        const TEST_BOND: u128 = 5_000_000;

        struct TestRegisterTx {
            unsigned_hex: String,
            signing_hash_hex: String,
            from_b58: String,
            tx: TransactionV2,
        }

        fn encode_hex_local(bytes: &[u8]) -> String {
            use std::fmt::Write;
            let mut s = String::with_capacity(bytes.len() * 2);
            for b in bytes {
                let _ = write!(&mut s, "{b:02x}");
            }
            s
        }

        fn build_test_register_tx(
            seed: &[u8; 32],
            bond: u128,
            chain_id: u64,
            nonce: u64,
            fee: u128,
        ) -> TestRegisterTx {
            let pubkey =
                omni_zkml::signer_pubkey_bytes(seed).expect("signer_pubkey_bytes");
            let from = Address::from_public_key(&pubkey);
            let from_b58 = from.to_base58();
            let tx = TransactionV2 {
                chain_id: chain_id.into(),
                from,
                fee: fee.into(),
                nonce: nonce.into(),
                payload: TxPayload::InferenceSettlement(InferenceSettlementTxData {
                    operation: InferenceSettlementOperation::RegisterVerifier(
                        RegisterVerifierRequest { bond },
                    ),
                }),
            };
            let unsigned_hex = format!("0x{}", encode_hex_local(&tx.to_bytes()));
            let signing_hash = tx.signing_hash();
            let signing_hash_hex =
                format!("0x{}", encode_hex_local(signing_hash.as_bytes()));
            TestRegisterTx {
                unsigned_hex,
                signing_hash_hex,
                from_b58,
                tx,
            }
        }

        /// Chain params with BOTH the settlement base gate and the
        /// verifier-bonding gate active from height 0.
        fn params_all_active() -> serde_json::Value {
            json!({
                "finality_depth": 12,
                "min_fee": 100,
                "chain_id": TEST_CHAIN_ID,
                "inference_settlement_enabled_from_height": 0,
                "inference_verifier_bonding_enabled_from_height": 0,
            })
        }

        fn args_register(
            bond: u128,
            expect_chain_id: u64,
            dry_run: bool,
        ) -> Box<RegisterVerifierArgs> {
            Box::new(RegisterVerifierArgs {
                expect_chain_id,
                bond,
                fee: None,
                dry_run,
            })
        }

        /// Seed a full happy-path builder + submit fixture. Returns the
        /// fixture so tests can assert against the exact tx.
        fn seed_full_happy_path(fake: &FakeJsonRpcTransport) -> TestRegisterTx {
            let fixture = build_test_register_tx(
                &TEST_SEED,
                TEST_BOND,
                TEST_CHAIN_ID,
                TEST_NONCE,
                TEST_FEE,
            );
            seed_params(fake, params_all_active());
            seed_head(fake, 500_000);
            fake.set_response(
                "omninode_buildRegisterVerifier",
                Ok(json!({
                    "unsigned_tx": fixture.unsigned_hex,
                    "signing_hash": fixture.signing_hash_hex,
                    "from": fixture.from_b58,
                    "nonce": TEST_NONCE,
                    "fee": TEST_FEE,
                    "chain_id": TEST_CHAIN_ID,
                })),
            );
            fake.set_response(
                "sum_sendRawTransaction",
                Ok(json!({ "tx_hash": "0xregister-happy" })),
            );
            fixture
        }

        fn run_with_capture_register(
            args: SettlementArgs,
            client: &SumChainClient<FakeJsonRpcTransport>,
            seed_source: SeedSource,
        ) -> (String, Vec<u8>, Result<()>) {
            let logs = CapturedLogs::new();
            let subscriber = tracing_subscriber::fmt()
                .with_writer(logs.clone())
                .with_max_level(tracing::Level::TRACE)
                .with_ansi(false)
                .without_time()
                .finish();
            let mut stdout_buf = Vec::new();
            let result = tracing::subscriber::with_default(subscriber, || {
                let SettlementArgs { cmd, .. } = args;
                match cmd {
                    SettlementCmd::RegisterVerifier(a) => super::super::run_register_verifier(
                        client,
                        &a,
                        &mut stdout_buf,
                        seed_source,
                    ),
                    _ => panic!(
                        "run_with_capture_register called with non-RegisterVerifier variant"
                    ),
                }
            });
            (logs.as_string(), stdout_buf, result)
        }

        fn call_methods(fake: &FakeJsonRpcTransport) -> Vec<String> {
            fake.calls().into_iter().map(|(m, _)| m).collect()
        }

        // ── Test 1 — build + sign + submit round-trip ──────────────
        //
        // Drives the full handler, then decodes the exact bytes sent to
        // `sum_sendRawTransaction` and asserts they parse to a
        // `RegisterVerifier(bond)` operation carrying the expected
        // fields.
        #[test]
        fn register_verifier_build_sign_submit_round_trip() {
            let (client, fake) = make_client();
            let fixture = seed_full_happy_path(&fake);

            let (logs, stdout, result) = run_with_capture_register(
                SettlementArgs {
                    rpc_url: None,
                    cmd: SettlementCmd::RegisterVerifier(args_register(
                        TEST_BOND,
                        TEST_CHAIN_ID,
                        false,
                    )),
                },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            result.expect("happy path must succeed");

            assert!(
                logs.contains("event=\"settlement_register_verifier_ready\""),
                "READY marker missing; logs=\n{logs}"
            );
            assert!(
                logs.contains("event=\"settlement_register_verifier_submitted\""),
                "SUBMITTED marker missing; logs=\n{logs}"
            );
            let stdout_str = String::from_utf8(stdout).unwrap();
            assert!(stdout_str.contains("submitted=true"), "{stdout_str}");
            assert!(
                stdout_str.contains("tx_hash=0xregister-happy"),
                "{stdout_str}"
            );
            assert!(stdout_str.contains("bond=5000000"), "{stdout_str}");

            // Builder + submit each called exactly once.
            let methods = call_methods(&fake);
            assert_eq!(
                methods
                    .iter()
                    .filter(|m| m.as_str() == "omninode_buildRegisterVerifier")
                    .count(),
                1,
                "builder called once; methods={methods:?}"
            );
            assert_eq!(
                methods
                    .iter()
                    .filter(|m| m.as_str() == "sum_sendRawTransaction")
                    .count(),
                1,
                "submit called once; methods={methods:?}"
            );

            // Decode the exact submitted wire and prove it is a
            // RegisterVerifier op with the expected fields.
            let calls = fake.calls();
            let (_m, params) = calls
                .iter()
                .find(|(m, _)| m == "sum_sendRawTransaction")
                .expect("submit RPC must have been called");
            let hex_param = params
                .as_array()
                .expect("params array")
                .first()
                .expect("params[0]")
                .as_str()
                .expect("hex string")
                .to_string();

            let decoded = SignedTransaction::from_hex(&hex_param)
                .expect("submitted hex must round-trip via SignedTransaction::from_hex");
            // Signing hash agrees with the fixture (proves fields intact).
            assert_eq!(decoded.signing_hash(), fixture.tx.signing_hash());

            let inner_tx = match decoded.inner() {
                TxInner::V2(v2) => v2,
                other => panic!("expected TxInner::V2, got {other:?}"),
            };
            assert_eq!(u64::from(inner_tx.chain_id), TEST_CHAIN_ID);
            assert_eq!(u64::from(inner_tx.nonce), TEST_NONCE);
            assert_eq!(u128::from(inner_tx.fee), TEST_FEE);
            assert_eq!(inner_tx.from.to_base58(), fixture.from_b58);
            match &inner_tx.payload {
                TxPayload::InferenceSettlement(InferenceSettlementTxData {
                    operation:
                        InferenceSettlementOperation::RegisterVerifier(
                            RegisterVerifierRequest { bond },
                        ),
                }) => assert_eq!(*bond, TEST_BOND, "bond field must round-trip"),
                other => panic!("expected RegisterVerifier op, got {other:?}"),
            }
        }

        // ── Test 2 — dry-run stops before signing/submitting ───────
        #[test]
        fn register_verifier_dry_run_never_submits() {
            let (client, fake) = make_client();
            let _fixture = seed_full_happy_path(&fake);

            let (logs, stdout, result) = run_with_capture_register(
                SettlementArgs {
                    rpc_url: None,
                    cmd: SettlementCmd::RegisterVerifier(args_register(
                        TEST_BOND,
                        TEST_CHAIN_ID,
                        true,
                    )),
                },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            result.expect("dry-run must succeed through preflight + verify");

            let stdout_str = String::from_utf8(stdout).unwrap();
            assert!(
                stdout_str.contains("dry_run=true submitted=false"),
                "{stdout_str}"
            );
            assert!(
                logs.contains("event=\"settlement_register_verifier_ready\""),
                "READY marker missing; logs=\n{logs}"
            );
            let methods = call_methods(&fake);
            assert!(
                !methods.iter().any(|m| m == "sum_sendRawTransaction"),
                "dry-run must not submit; methods={methods:?}"
            );
        }

        // ── Test 3 — chain-id mismatch refusal (preflight) ─────────
        #[test]
        fn register_verifier_refuses_on_chain_id_mismatch() {
            let (client, fake) = make_client();
            let _fixture = seed_full_happy_path(&fake);

            let (logs, _stdout, result) = run_with_capture_register(
                SettlementArgs {
                    rpc_url: None,
                    cmd: SettlementCmd::RegisterVerifier(args_register(
                        TEST_BOND,
                        // Guardrail disagrees with the chain's chain_id.
                        TEST_CHAIN_ID + 1,
                        false,
                    )),
                },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            let err = result.expect_err("chain-id mismatch must refuse");
            assert!(
                err.to_string().contains("chain-id mismatch"),
                "error must name chain-id mismatch: {err}"
            );
            assert!(
                logs.contains("event=\"settlement_register_verifier_refused_chain_id\""),
                "chain-id refusal marker missing; logs=\n{logs}"
            );
            // Refused BEFORE the builder or submit RPCs.
            let methods = call_methods(&fake);
            assert!(
                !methods.iter().any(|m| m == "omninode_buildRegisterVerifier"),
                "must not call builder on chain-id mismatch; methods={methods:?}"
            );
            assert!(
                !methods.iter().any(|m| m == "sum_sendRawTransaction"),
                "must not submit on chain-id mismatch; methods={methods:?}"
            );
        }

        // ── Test 4 — settlement-inactive refusal (preflight) ───────
        #[test]
        fn register_verifier_refuses_when_settlement_dormant() {
            let (client, fake) = make_client();
            // Settlement base gate ABSENT (dormant); chain-id matches.
            seed_params(
                &fake,
                json!({
                    "finality_depth": 12,
                    "min_fee": 100,
                    "chain_id": TEST_CHAIN_ID,
                }),
            );
            seed_head(&fake, 500_000);

            let (logs, _stdout, result) = run_with_capture_register(
                SettlementArgs {
                    rpc_url: None,
                    cmd: SettlementCmd::RegisterVerifier(args_register(
                        TEST_BOND,
                        TEST_CHAIN_ID,
                        false,
                    )),
                },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            let err = result.expect_err("settlement-inactive must refuse");
            assert!(
                err.to_string().contains("settlement subsystem inactive"),
                "error must name settlement inactivity: {err}"
            );
            assert!(
                logs.contains("event=\"settlement_register_verifier_refused_dormancy\""),
                "dormancy refusal marker missing; logs=\n{logs}"
            );
            assert!(
                logs.contains("gate=\"inference_settlement_enabled_from_height\""),
                "must name the settlement gate; logs=\n{logs}"
            );
            let methods = call_methods(&fake);
            assert!(
                !methods.iter().any(|m| m == "omninode_buildRegisterVerifier"),
                "must not call builder when settlement dormant; methods={methods:?}"
            );
        }

        // ── Test 5 — bonding-inactive refusal (preflight) ──────────
        #[test]
        fn register_verifier_refuses_when_bonding_dormant() {
            let (client, fake) = make_client();
            // Settlement base gate ACTIVE, but verifier-bonding ABSENT.
            seed_params(
                &fake,
                json!({
                    "finality_depth": 12,
                    "min_fee": 100,
                    "chain_id": TEST_CHAIN_ID,
                    "inference_settlement_enabled_from_height": 0,
                }),
            );
            seed_head(&fake, 500_000);

            let (logs, _stdout, result) = run_with_capture_register(
                SettlementArgs {
                    rpc_url: None,
                    cmd: SettlementCmd::RegisterVerifier(args_register(
                        TEST_BOND,
                        TEST_CHAIN_ID,
                        false,
                    )),
                },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            let err = result.expect_err("bonding-inactive must refuse");
            assert!(
                err.to_string().contains("verifier bonding inactive"),
                "error must name bonding inactivity: {err}"
            );
            assert!(
                logs.contains("gate=\"inference_verifier_bonding_enabled_from_height\""),
                "must name the bonding gate; logs=\n{logs}"
            );
            let methods = call_methods(&fake);
            assert!(
                !methods.iter().any(|m| m == "omninode_buildRegisterVerifier"),
                "must not call builder when bonding dormant; methods={methods:?}"
            );
        }

        // ── Test 6 — marker constant names pinned ──────────────────
        #[test]
        fn register_verifier_marker_constant_names_are_pinned() {
            use super::super::register_verifier_markers as rvm;
            assert_eq!(rvm::REGISTER_READY, "settlement_register_verifier_ready");
            assert_eq!(
                rvm::REGISTER_REFUSED_BOND_ZERO,
                "settlement_register_verifier_refused_bond_zero"
            );
            assert_eq!(
                rvm::REGISTER_REFUSED_CHAIN_ID,
                "settlement_register_verifier_refused_chain_id"
            );
            assert_eq!(
                rvm::REGISTER_REFUSED_DORMANCY,
                "settlement_register_verifier_refused_dormancy"
            );
            assert_eq!(
                rvm::REGISTER_SUBMITTED,
                "settlement_register_verifier_submitted"
            );
            assert_eq!(rvm::REGISTER_FAILED, "settlement_register_verifier_failed");
        }

        // ── Test 7 — missing --bond is a clap required-arg error ───
        //
        // `--bond` is a REQUIRED clap arg (no default). Omitting it must
        // fail at parse time, before `run_register_verifier` is ever
        // entered — so no client / transport / seed is even constructed.
        #[test]
        fn register_verifier_missing_bond_is_clap_required_error() {
            #[derive(Debug, clap::Parser)]
            struct Probe {
                #[command(subcommand)]
                #[allow(dead_code)]
                cmd: SettlementCmd,
            }
            use clap::Parser as _;

            // Supply --expect-chain-id (also required) but omit --bond.
            let parsed = Probe::try_parse_from([
                "op",
                "register-verifier",
                "--expect-chain-id",
                "1800100",
            ]);
            let err = parsed.expect_err("missing --bond must be a clap error");
            let rendered = err.to_string();
            assert!(
                rendered.contains("--bond"),
                "clap error must name the missing --bond arg: {rendered}"
            );
        }

        // ── Test 8 — --bond 0 refused locally, before any RPC ──────
        #[test]
        fn register_verifier_bond_zero_refused_locally() {
            let (client, fake) = make_client();
            // Seed a full happy path; NONE of it must be reached.
            let _fixture = seed_full_happy_path(&fake);

            let (logs, _stdout, result) = run_with_capture_register(
                SettlementArgs {
                    rpc_url: None,
                    cmd: SettlementCmd::RegisterVerifier(args_register(
                        0,
                        TEST_CHAIN_ID,
                        false,
                    )),
                },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            let err = result.expect_err("bond 0 must be refused");
            assert!(
                err.to_string().contains("--bond must be a positive amount"),
                "error must name the zero-bond refusal: {err}"
            );
            assert!(
                logs.contains(
                    "event=\"settlement_register_verifier_refused_bond_zero\""
                ),
                "bond-zero marker missing; logs=\n{logs}"
            );
            // Refused before ANY RPC: the recorder is empty.
            let methods = call_methods(&fake);
            assert!(
                methods.is_empty(),
                "bond-0 must issue ZERO RPC; methods={methods:?}"
            );
        }

        // ── Test 9 — bond failures precede builder, signer, submit ──
        //
        // Proves BOTH bond-failure paths short-circuit before the
        // builder RPC, before any seed/identity resolution, and before
        // any submission. Zero seed access is proven by supplying
        // `SeedSource::AbsentForTest`: had the handler reached identity
        // resolution, it would have returned a seed-missing error rather
        // than the bond-zero error.
        #[test]
        fn register_verifier_bond_failures_precede_builder_signer_and_submit() {
            // (a) missing --bond: clap rejects before the handler is
            //     entered, so no transport/seed is ever touched.
            #[derive(Debug, clap::Parser)]
            struct Probe {
                #[command(subcommand)]
                #[allow(dead_code)]
                cmd: SettlementCmd,
            }
            use clap::Parser as _;
            let missing = Probe::try_parse_from([
                "op",
                "register-verifier",
                "--expect-chain-id",
                "1800100",
            ]);
            assert!(
                missing.is_err(),
                "missing --bond must fail at clap parse, before the handler"
            );

            // (b) --bond 0 with an ABSENT seed: refuses for bond-zero,
            //     NOT for a missing seed — proving no seed/identity
            //     resolution occurred.
            let (client, fake) = make_client();
            let _fixture = seed_full_happy_path(&fake);
            let (logs, _stdout, result) = run_with_capture_register(
                SettlementArgs {
                    rpc_url: None,
                    cmd: SettlementCmd::RegisterVerifier(args_register(
                        0,
                        TEST_CHAIN_ID,
                        false,
                    )),
                },
                &client,
                SeedSource::AbsentForTest,
            );
            let err = result.expect_err("bond 0 must be refused");
            let msg = err.to_string();
            assert!(
                msg.contains("--bond must be a positive amount"),
                "must refuse for bond-zero: {msg}"
            );
            // NOT the seed-missing / identity-resolve error path.
            assert!(
                !msg.contains("identity resolve") && !msg.contains("unavailable"),
                "bond-zero must fire BEFORE any seed/identity resolution \
                 (AbsentForTest would surface a seed error if reached): {msg}"
            );
            assert!(
                logs.contains(
                    "event=\"settlement_register_verifier_refused_bond_zero\""
                ),
                "bond-zero marker missing; logs=\n{logs}"
            );
            // ZERO builder RPC + ZERO submission.
            let methods = call_methods(&fake);
            assert!(
                methods.is_empty(),
                "bond-0 must issue ZERO RPC (no builder, no submit); \
                 methods={methods:?}"
            );
            assert!(
                !methods.iter().any(|m| m == "omninode_buildRegisterVerifier"),
                "bond-0 must not call the builder; methods={methods:?}"
            );
            assert!(
                !methods.iter().any(|m| m == "sum_sendRawTransaction"),
                "bond-0 must not submit; methods={methods:?}"
            );
        }

        // ── Test 10 — dry-run derives the address but never signs ──
        //
        // Pins the FINAL documented dry-run contract word-for-word in
        // intent: dry-run DERIVES the operator address from
        // `OMNINODE_VERIFIER_SEED_HEX` (for the builder request + the
        // envelope `from` assertion) but never retains the secret, never
        // signs, and never invokes `sum_sendRawTransaction`.
        #[test]
        fn register_verifier_dry_run_derives_address_but_never_signs_or_submits() {
            let (client, fake) = make_client();
            let fixture = seed_full_happy_path(&fake);

            let (logs, stdout, result) = run_with_capture_register(
                SettlementArgs {
                    rpc_url: None,
                    cmd: SettlementCmd::RegisterVerifier(args_register(
                        TEST_BOND,
                        TEST_CHAIN_ID,
                        true,
                    )),
                },
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            result.expect("dry-run must succeed through preflight + verify");
            let stdout_str = String::from_utf8(stdout).unwrap();

            // (1) The seed's address WAS derived and surfaced.
            assert!(
                stdout_str.contains(&format!("verifier={}", fixture.from_b58)),
                "dry-run must derive + report the seed's address; \
                 stdout=\n{stdout_str}"
            );
            // (2) Build + verify happened (READY), then STOP.
            assert!(
                logs.contains("event=\"settlement_register_verifier_ready\""),
                "READY marker missing; logs=\n{logs}"
            );
            assert!(
                stdout_str.contains("dry_run=true submitted=false"),
                "{stdout_str}"
            );
            // (3) ZERO signing / ZERO submission.
            assert!(
                !logs.contains(
                    "event=\"settlement_register_verifier_submitted\""
                ),
                "dry-run must not emit SUBMITTED; logs=\n{logs}"
            );
            let methods = call_methods(&fake);
            assert!(
                !methods.iter().any(|m| m == "sum_sendRawTransaction"),
                "dry-run must not sign/submit; methods={methods:?}"
            );
            // The builder RPC IS part of a dry-run (build + verify).
            assert!(
                methods.iter().any(|m| m == "omninode_buildRegisterVerifier"),
                "dry-run must still run the builder RPC; methods={methods:?}"
            );

            // Contract corollary: dry-run REQUIRES the seed. With no
            // seed, the identity resolution dry-run depends on fails —
            // proving dry-run reads the seed rather than fabricating an
            // address.
            let (client2, fake2) = make_client();
            let _f2 = seed_full_happy_path(&fake2);
            let (_logs2, _stdout2, result2) = run_with_capture_register(
                SettlementArgs {
                    rpc_url: None,
                    cmd: SettlementCmd::RegisterVerifier(args_register(
                        TEST_BOND,
                        TEST_CHAIN_ID,
                        true,
                    )),
                },
                &client2,
                SeedSource::AbsentForTest,
            );
            let err2 = result2
                .expect_err("dry-run without a seed must fail at identity resolve");
            assert!(
                err2.to_string().contains("identity resolve"),
                "dry-run must fail resolving the seed's address when absent: \
                 {err2}"
            );
        }

        // ── Tests 11-18 — tampered builder responses each rejected ──
        //
        // Each mutates exactly one aspect of the builder envelope / tx /
        // caller expectation and asserts `verify_register_verifier_transaction`
        // returns a DISTINCT, clearly-labeled error. A non-Ok verify
        // return is exactly what makes the driver refuse before Step 7
        // (sign + submit) — these never reach a submission.

        /// Canonical (untampered) envelope matching `fixture.tx`.
        fn canonical_envelope(
            fixture: &TestRegisterTx,
            chain_id: u64,
            nonce: u64,
            fee: u128,
        ) -> SettlementBuilderEnvelope {
            SettlementBuilderEnvelope {
                unsigned_tx: fixture.unsigned_hex.clone(),
                signing_hash: fixture.signing_hash_hex.clone(),
                from: fixture.from_b58.clone(),
                nonce,
                fee,
                chain_id,
            }
        }

        #[test]
        fn verify_register_rejects_wrong_chain_id() {
            let f = build_test_register_tx(
                &TEST_SEED,
                TEST_BOND,
                TEST_CHAIN_ID,
                TEST_NONCE,
                TEST_FEE,
            );
            // Envelope claims a different chain id than the tx.
            let env =
                canonical_envelope(&f, TEST_CHAIN_ID + 1, TEST_NONCE, TEST_FEE);
            let err = verify_register_verifier_transaction(
                &f.tx,
                &env,
                &f.from_b58,
                TEST_BOND,
            )
            .expect_err("wrong chain_id must be rejected");
            assert!(
                err.to_string().contains("field 'tx.chain_id'"),
                "distinct chain_id error expected: {err}"
            );
        }

        #[test]
        fn verify_register_rejects_wrong_from() {
            let f = build_test_register_tx(
                &TEST_SEED,
                TEST_BOND,
                TEST_CHAIN_ID,
                TEST_NONCE,
                TEST_FEE,
            );
            let mut env =
                canonical_envelope(&f, TEST_CHAIN_ID, TEST_NONCE, TEST_FEE);
            env.from = "WRONG_FROM_ADDRESS".to_string();
            let err = verify_register_verifier_transaction(
                &f.tx,
                &env,
                &f.from_b58,
                TEST_BOND,
            )
            .expect_err("wrong from must be rejected");
            assert!(
                err.to_string().contains("field 'tx.from vs envelope.from'"),
                "distinct from error expected: {err}"
            );
        }

        #[test]
        fn verify_register_rejects_wrong_nonce() {
            let f = build_test_register_tx(
                &TEST_SEED,
                TEST_BOND,
                TEST_CHAIN_ID,
                TEST_NONCE,
                TEST_FEE,
            );
            let env =
                canonical_envelope(&f, TEST_CHAIN_ID, TEST_NONCE + 1, TEST_FEE);
            let err = verify_register_verifier_transaction(
                &f.tx,
                &env,
                &f.from_b58,
                TEST_BOND,
            )
            .expect_err("wrong nonce must be rejected");
            assert!(
                err.to_string().contains("field 'tx.nonce'"),
                "distinct nonce error expected: {err}"
            );
        }

        #[test]
        fn verify_register_rejects_wrong_fee() {
            let f = build_test_register_tx(
                &TEST_SEED,
                TEST_BOND,
                TEST_CHAIN_ID,
                TEST_NONCE,
                TEST_FEE,
            );
            let env =
                canonical_envelope(&f, TEST_CHAIN_ID, TEST_NONCE, TEST_FEE + 1);
            let err = verify_register_verifier_transaction(
                &f.tx,
                &env,
                &f.from_b58,
                TEST_BOND,
            )
            .expect_err("wrong fee must be rejected");
            assert!(
                err.to_string().contains("field 'tx.fee'"),
                "distinct fee error expected: {err}"
            );
        }

        #[test]
        fn verify_register_rejects_wrong_operation() {
            // A non-RegisterVerifier op (ClaimReward) with otherwise
            // valid envelope metadata must be rejected at the operation
            // variant check.
            let pubkey =
                omni_zkml::signer_pubkey_bytes(&TEST_SEED).expect("pubkey");
            let from = Address::from_public_key(&pubkey);
            let from_b58 = from.to_base58();
            let claim_tx = TransactionV2 {
                chain_id: TEST_CHAIN_ID.into(),
                from,
                fee: TEST_FEE.into(),
                nonce: TEST_NONCE.into(),
                payload: TxPayload::InferenceSettlement(
                    InferenceSettlementTxData {
                        operation: InferenceSettlementOperation::ClaimReward(
                            ClaimInferenceRewardRequest {
                                session_id: "s-not-register".to_string(),
                            },
                        ),
                    },
                ),
            };
            let signing_hash = claim_tx.signing_hash();
            let env = SettlementBuilderEnvelope {
                unsigned_tx: format!(
                    "0x{}",
                    encode_hex_local(&claim_tx.to_bytes())
                ),
                signing_hash: format!(
                    "0x{}",
                    encode_hex_local(signing_hash.as_bytes())
                ),
                from: from_b58.clone(),
                nonce: TEST_NONCE,
                fee: TEST_FEE,
                chain_id: TEST_CHAIN_ID,
            };
            let err = verify_register_verifier_transaction(
                &claim_tx,
                &env,
                &from_b58,
                TEST_BOND,
            )
            .expect_err("non-RegisterVerifier op must be rejected");
            assert!(
                err.to_string()
                    .contains("field 'tx.payload.operation variant'"),
                "distinct operation-variant error expected: {err}"
            );
        }

        #[test]
        fn verify_register_rejects_wrong_bond() {
            let f = build_test_register_tx(
                &TEST_SEED,
                TEST_BOND,
                TEST_CHAIN_ID,
                TEST_NONCE,
                TEST_FEE,
            );
            let env =
                canonical_envelope(&f, TEST_CHAIN_ID, TEST_NONCE, TEST_FEE);
            // Caller expected a different bond than the tx carries.
            let err = verify_register_verifier_transaction(
                &f.tx,
                &env,
                &f.from_b58,
                TEST_BOND + 1,
            )
            .expect_err("wrong bond must be rejected");
            assert!(
                err.to_string()
                    .contains("field 'tx.payload.operation.bond'"),
                "distinct bond error expected: {err}"
            );
        }

        #[test]
        fn verify_register_rejects_wrong_signing_hash() {
            let f = build_test_register_tx(
                &TEST_SEED,
                TEST_BOND,
                TEST_CHAIN_ID,
                TEST_NONCE,
                TEST_FEE,
            );
            let mut env =
                canonical_envelope(&f, TEST_CHAIN_ID, TEST_NONCE, TEST_FEE);
            // A different (but well-formed) 32-byte signing hash.
            env.signing_hash = format!("0x{}", "11".repeat(32));
            let err = verify_register_verifier_transaction(
                &f.tx,
                &env,
                &f.from_b58,
                TEST_BOND,
            )
            .expect_err("wrong signing_hash must be rejected");
            assert!(
                err.to_string().contains(
                    "field 'tx.signing_hash() vs envelope.signing_hash'"
                ),
                "distinct signing-hash error expected: {err}"
            );
        }
    }

    // ── Issue #81 — settlement dispute tests ────────────────────────

    #[cfg(feature = "settlement-dispute")]
    mod dispute {
        use super::*;
        use crate::operator::SeedSource;
        use omni_sumchain::settlement_dispute::{
            Address, InferenceSettlementOperation, InferenceSettlementTxData,
            OpenInferenceDisputeRequest, ResolveInferenceDisputeRequest,
            SignedTransaction, TransactionV2, TxPayload, ValidatorApproval,
        };

        const TEST_SEED: [u8; 32] = [11u8; 32];
        const TEST_CHAIN_ID: u64 = 1_800_100;
        const TEST_NONCE: u64 = 7;
        const TEST_FEE: u128 = 1000;

        // ── Inline fixtures (test-only, not exposed via public API) ──

        fn encode_hex_local(bytes: &[u8]) -> String {
            use std::fmt::Write;
            let mut s = String::with_capacity(bytes.len() * 2);
            for b in bytes {
                let _ = write!(&mut s, "{b:02x}");
            }
            s
        }

        struct TestOpenTx {
            unsigned_hex: String,
            signing_hash_hex: String,
            from_b58: String,
            tx: TransactionV2,
        }

        fn build_test_open_tx(
            seed: &[u8; 32],
            session_id: &str,
            verifier_b58: &str,
            evidence: [u8; 32],
            chain_id: u64,
            nonce: u64,
            fee: u128,
        ) -> TestOpenTx {
            let pubkey = omni_zkml::signer_pubkey_bytes(seed).unwrap();
            let from = Address::from_public_key(&pubkey);
            let from_b58 = from.to_base58();
            let verifier_addr = Address::from_base58(verifier_b58).unwrap();
            let tx = TransactionV2 {
                chain_id: chain_id.into(),
                from,
                fee: fee.into(),
                nonce: nonce.into(),
                payload: TxPayload::InferenceSettlement(InferenceSettlementTxData {
                    operation: InferenceSettlementOperation::OpenDispute(
                        OpenInferenceDisputeRequest {
                            session_id: session_id.to_string(),
                            verifier: verifier_addr,
                            evidence_commitment: evidence,
                        },
                    ),
                }),
            };
            let unsigned_hex = format!("0x{}", encode_hex_local(&tx.to_bytes()));
            let signing_hash_hex =
                format!("0x{}", encode_hex_local(tx.signing_hash().as_bytes()));
            TestOpenTx {
                unsigned_hex,
                signing_hash_hex,
                from_b58,
                tx,
            }
        }

        struct TestResolveTx {
            unsigned_hex: String,
            signing_hash_hex: String,
            from_b58: String,
            tx: TransactionV2,
        }

        fn build_test_resolve_tx(
            seed: &[u8; 32],
            session_id: &str,
            verifier_b58: &str,
            allow_claim: bool,
            approvals: Vec<ValidatorApproval>,
            chain_id: u64,
            nonce: u64,
            fee: u128,
        ) -> TestResolveTx {
            let pubkey = omni_zkml::signer_pubkey_bytes(seed).unwrap();
            let from = Address::from_public_key(&pubkey);
            let from_b58 = from.to_base58();
            let verifier_addr = Address::from_base58(verifier_b58).unwrap();
            let tx = TransactionV2 {
                chain_id: chain_id.into(),
                from,
                fee: fee.into(),
                nonce: nonce.into(),
                payload: TxPayload::InferenceSettlement(InferenceSettlementTxData {
                    operation: InferenceSettlementOperation::ResolveDispute(
                        ResolveInferenceDisputeRequest {
                            session_id: session_id.to_string(),
                            verifier: verifier_addr,
                            allow_claim,
                            approvals,
                        },
                    ),
                }),
            };
            let unsigned_hex = format!("0x{}", encode_hex_local(&tx.to_bytes()));
            let signing_hash_hex =
                format!("0x{}", encode_hex_local(tx.signing_hash().as_bytes()));
            TestResolveTx {
                unsigned_hex,
                signing_hash_hex,
                from_b58,
                tx,
            }
        }

        // ── Derive a fake verifier address for tests. ───────────────

        fn derived_addr_from_seed(seed: &[u8; 32]) -> String {
            omni_zkml::signer_chain_address_base58(seed).unwrap()
        }

        // ── Fixture: verifier we're targeting on-chain ──────────────
        const VERIFIER_SEED: [u8; 32] = [22u8; 32];

        fn verifier_b58() -> String {
            derived_addr_from_seed(&VERIFIER_SEED)
        }

        const EVIDENCE_HEX: &str =
            "0xdeadbeef00112233445566778899aabbccddeeff00112233445566778899aabb";

        fn evidence_bytes() -> [u8; 32] {
            let mut out = [0u8; 32];
            let body = EVIDENCE_HEX.trim_start_matches("0x");
            for i in 0..32 {
                out[i] = u8::from_str_radix(&body[i * 2..i * 2 + 2], 16).unwrap();
            }
            out
        }

        // ── Fake-transport helper (mirrors #87 shape) ───────────────

        fn params_settlement_active() -> serde_json::Value {
            json!({
                "finality_depth": 12,
                "min_fee": 100,
                "chain_id": TEST_CHAIN_ID,
                "inference_settlement_enabled_from_height": 0,
            })
        }

        fn args_open() -> Box<OpenDisputeArgs> {
            Box::new(OpenDisputeArgs {
                session_id: "s-1".to_string(),
                verifier: verifier_b58(),
                evidence: EVIDENCE_HEX.to_string(),
                fee: None,
                dry_run: false,
            })
        }

        fn args_open_dry_run() -> Box<OpenDisputeArgs> {
            Box::new(OpenDisputeArgs {
                session_id: "s-1".to_string(),
                verifier: verifier_b58(),
                evidence: EVIDENCE_HEX.to_string(),
                fee: None,
                dry_run: true,
            })
        }

        fn seed_open_happy_path(fake: &FakeJsonRpcTransport) -> TestOpenTx {
            let derived = derived_addr_from_seed(&TEST_SEED);
            let fixture = build_test_open_tx(
                &TEST_SEED,
                "s-1",
                &verifier_b58(),
                evidence_bytes(),
                TEST_CHAIN_ID,
                TEST_NONCE,
                TEST_FEE,
            );
            seed_params(fake, params_settlement_active());
            seed_head(fake, 500_000);
            fake.set_response(
                "omninode_getInferenceSession",
                Ok(json!({
                    "session_id": "s-1",
                    "consistency_required": false,
                    "bond_required": false,
                    "max_verifiers": 3,
                    "escrow_total": "1000",
                    "escrow_remaining": "1000",
                    "claims_count": 0,
                    "status": "Open",
                    "created_at_height": 400_000,
                    "funder": derived,
                    "dispute_window_blocks": 5_000,
                })),
            );
            fake.set_response(
                "sum_getInferenceAttestation",
                Ok(json!({
                    "session_id": "s-1",
                    "verifier_address": verifier_b58(),
                    "model_hash": "0xaa",
                    "manifest_root": "0xbb",
                    "response_hash": "0xcc",
                    "proof_root": "0xdd",
                    "verifier_signature": "0xsig",
                    "included_at_height": 495_000,      // maturity = 495_000 + 12 + 5_000 = 500_012
                    "tx_hash": "0xtxatt",
                    "finalized": true,
                })),
            );
            fake.set_response(
                "omninode_getInferenceDisputes",
                Ok(json!({ "session_id": "s-1", "disputes": [] })),
            );
            fake.set_response(
                "omninode_buildOpenInferenceDispute",
                Ok(json!({
                    "unsigned_tx": fixture.unsigned_hex,
                    "signing_hash": fixture.signing_hash_hex,
                    "from": fixture.from_b58,
                    "nonce": TEST_NONCE,
                    "fee": TEST_FEE,
                    "chain_id": TEST_CHAIN_ID,
                })),
            );
            fake.set_response(
                "sum_sendRawTransaction",
                Ok(json!({ "tx_hash": "0xtxhash-open-happy" })),
            );
            fixture
        }

        fn run_open_with_capture(
            args: Box<OpenDisputeArgs>,
            client: &SumChainClient<FakeJsonRpcTransport>,
            seed_source: SeedSource,
        ) -> (String, Vec<u8>, Result<()>) {
            let logs = CapturedLogs::new();
            let subscriber = tracing_subscriber::fmt()
                .with_writer(logs.clone())
                .with_max_level(tracing::Level::TRACE)
                .with_ansi(false)
                .without_time()
                .finish();
            let mut stdout_buf = Vec::new();
            let result = tracing::subscriber::with_default(subscriber, || {
                super::super::run_dispute_open(client, &args, &mut stdout_buf, seed_source)
            });
            (logs.as_string(), stdout_buf, result)
        }

        fn run_resolve_with_capture(
            args: Box<ResolveDisputeArgs>,
            client: &SumChainClient<FakeJsonRpcTransport>,
            seed_source: SeedSource,
        ) -> (String, Vec<u8>, Result<()>) {
            let logs = CapturedLogs::new();
            let subscriber = tracing_subscriber::fmt()
                .with_writer(logs.clone())
                .with_max_level(tracing::Level::TRACE)
                .with_ansi(false)
                .without_time()
                .finish();
            let mut stdout_buf = Vec::new();
            let result = tracing::subscriber::with_default(subscriber, || {
                super::super::run_dispute_resolve(
                    client,
                    &args,
                    &mut stdout_buf,
                    seed_source,
                )
            });
            (logs.as_string(), stdout_buf, result)
        }

        fn call_methods_dispute(fake: &FakeJsonRpcTransport) -> Vec<String> {
            fake.calls().into_iter().map(|(m, _)| m).collect()
        }

        // ── OPEN tests ──────────────────────────────────────────────

        #[test]
        fn open_dormant_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            seed_params(&fake, params_all_dormant());
            seed_head(&fake, 100_000);
            let (logs, _s, r) = run_open_with_capture(
                args_open(),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect_err("dormant must fail");
            assert!(logs.contains("event=\"settlement_dispute_refused_dormancy\""));
            let m = call_methods_dispute(&fake);
            assert!(!m.iter().any(|x| x == "omninode_buildOpenInferenceDispute"));
            assert!(!m.iter().any(|x| x == "sum_sendRawTransaction"));
        }

        #[test]
        fn open_missing_session_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            seed_params(&fake, params_settlement_active());
            seed_head(&fake, 500_000);
            fake.set_response("omninode_getInferenceSession", Ok(serde_json::Value::Null));
            let (logs, _s, r) = run_open_with_capture(
                args_open(),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect_err("missing session must fail");
            assert!(logs.contains("event=\"settlement_dispute_refused_state\""));
            assert!(logs.contains("reason=\"session_not_found\""));
            let m = call_methods_dispute(&fake);
            assert!(!m.iter().any(|x| x == "omninode_buildOpenInferenceDispute"));
            assert!(!m.iter().any(|x| x == "sum_sendRawTransaction"));
        }

        #[test]
        fn open_funder_mismatch_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            seed_params(&fake, params_settlement_active());
            seed_head(&fake, 500_000);
            fake.set_response(
                "omninode_getInferenceSession",
                Ok(json!({
                    "session_id": "s-1",
                    "consistency_required": false,
                    "bond_required": false,
                    "max_verifiers": 3,
                    "escrow_total": "1000",
                    "escrow_remaining": "1000",
                    "claims_count": 0,
                    "status": "Open",
                    "created_at_height": 400_000,
                    "funder": "someone-else",       // <-- mismatched
                    "dispute_window_blocks": 5_000,
                })),
            );
            let (logs, _s, r) = run_open_with_capture(
                args_open(),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect_err("funder mismatch must fail");
            assert!(logs.contains("event=\"settlement_dispute_refused_authority\""));
            let m = call_methods_dispute(&fake);
            assert!(!m.iter().any(|x| x == "omninode_buildOpenInferenceDispute"));
            assert!(!m.iter().any(|x| x == "sum_sendRawTransaction"));
        }

        /// Chain-team confirmation (sum-chain#110): `OpenDispute` only
        /// accepted against `Open` sessions. Refunded / Settled /
        /// unknown terminal states must refuse locally so the operator
        /// doesn't burn a fee on a chain-rejected tx.
        ///
        /// Uses the confirmed chain-team session JSON shape:
        /// `"status": "Refunded"` (capital-cased, not the legacy
        /// `"lifecycle": "refunded"`).
        #[test]
        fn open_session_not_open_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            let derived = derived_addr_from_seed(&TEST_SEED);
            seed_params(&fake, params_settlement_active());
            seed_head(&fake, 500_000);
            fake.set_response(
                "omninode_getInferenceSession",
                Ok(json!({
                    "session_id": "s-1",
                    "consistency_required": false,
                    "bond_required": false,
                    "max_verifiers": 3,
                    "escrow_total": "1000",
                    "escrow_remaining": "1000",
                    "claims_count": 0,
                    "status": "Refunded",     // <-- terminal state; refuse
                    "created_at_height": 400_000,
                    "funder": derived,
                    "dispute_window_blocks": 5_000,
                })),
            );
            let (logs, _s, r) = run_open_with_capture(
                args_open(),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect_err("non-Open session must be refused");
            assert!(
                logs.contains("event=\"settlement_dispute_refused_state\""),
                "must emit DISPUTE_REFUSED_STATE; logs=\n{logs}"
            );
            assert!(
                logs.contains("reason=\"session_not_open\""),
                "reason must name session_not_open; logs=\n{logs}"
            );
            assert!(
                logs.contains("wire_status=\"Refunded\""),
                "wire_status field must carry the chain's raw status verbatim; logs=\n{logs}"
            );
            let m = call_methods_dispute(&fake);
            assert!(
                !m.iter().any(|x| x == "sum_getInferenceAttestation"),
                "attestation lookup NOT reached (comes after status precheck); methods={m:?}"
            );
            assert!(
                !m.iter().any(|x| x == "omninode_buildOpenInferenceDispute"),
                "builder MUST NOT be called; methods={m:?}"
            );
            assert!(
                !m.iter().any(|x| x == "sum_sendRawTransaction"),
                "submit MUST NOT be called; methods={m:?}"
            );
        }

        /// Chain-team shape pin: an `Open` session with the confirmed
        /// `"status": "Open"` wire field passes the open-status
        /// precheck and reaches the builder. This test also exercises
        /// the `#[serde(alias = "status")]` DTO reader — the fake
        /// response uses `"status"`, not `"lifecycle"`.
        #[test]
        fn open_session_status_open_wire_shape_reaches_builder() {
            let (client, fake) = make_client();
            let _fixture = seed_open_happy_path(&fake);
            // seed_open_happy_path already uses "status": "Open" per
            // the reviewer's #81 test-JSON update; assert it flows
            // through to CLAIM_READY (the marker emitted right after
            // envelope + decoded-tx verification pass).
            let (logs, _s, r) = run_open_with_capture(
                args_open_dry_run(),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect("`status: Open` session must pass the open-status precheck");
            assert!(
                logs.contains("event=\"settlement_dispute_ready\""),
                "must emit DISPUTE_READY; logs=\n{logs}"
            );
            let m = call_methods_dispute(&fake);
            assert!(
                m.iter().any(|x| x == "omninode_buildOpenInferenceDispute"),
                "builder MUST be called; methods={m:?}"
            );
        }

        #[test]
        fn open_missing_attestation_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            let derived = derived_addr_from_seed(&TEST_SEED);
            seed_params(&fake, params_settlement_active());
            seed_head(&fake, 500_000);
            fake.set_response(
                "omninode_getInferenceSession",
                Ok(json!({
                    "session_id": "s-1",
                    "consistency_required": false,
                    "bond_required": false,
                    "max_verifiers": 3,
                    "escrow_total": "1000",
                    "escrow_remaining": "1000",
                    "claims_count": 0,
                    "status": "Open",
                    "created_at_height": 400_000,
                    "funder": derived,
                    "dispute_window_blocks": 5_000,
                })),
            );
            fake.set_response("sum_getInferenceAttestation", Ok(serde_json::Value::Null));
            let (logs, _s, r) = run_open_with_capture(
                args_open(),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect_err("missing attestation must fail");
            assert!(logs.contains("event=\"settlement_dispute_refused_state\""));
            assert!(logs.contains("reason=\"attestation_not_found\""));
            let m = call_methods_dispute(&fake);
            assert!(!m.iter().any(|x| x == "omninode_buildOpenInferenceDispute"));
            assert!(!m.iter().any(|x| x == "sum_sendRawTransaction"));
        }

        #[test]
        fn open_maturity_elapsed_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            let derived = derived_addr_from_seed(&TEST_SEED);
            seed_params(&fake, params_settlement_active());
            seed_head(&fake, 600_000);
            fake.set_response(
                "omninode_getInferenceSession",
                Ok(json!({
                    "session_id": "s-1",
                    "consistency_required": false,
                    "bond_required": false,
                    "max_verifiers": 3,
                    "escrow_total": "1000",
                    "escrow_remaining": "1000",
                    "claims_count": 0,
                    "status": "Open",
                    "created_at_height": 400_000,
                    "funder": derived,
                    "dispute_window_blocks": 5_000,
                })),
            );
            fake.set_response(
                "sum_getInferenceAttestation",
                Ok(json!({
                    "session_id": "s-1",
                    "verifier_address": verifier_b58(),
                    "model_hash": "0xaa",
                    "manifest_root": "0xbb",
                    "response_hash": "0xcc",
                    "proof_root": "0xdd",
                    "verifier_signature": "0xsig",
                    // maturity = 400_000 + 12 + 5_000 = 405_012 < head 600_000
                    "included_at_height": 400_000,
                    "tx_hash": "0xtxatt",
                    "finalized": true,
                })),
            );
            let (logs, _s, r) = run_open_with_capture(
                args_open(),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect_err("maturity elapsed must fail");
            assert!(logs.contains("event=\"settlement_dispute_refused_maturity\""));
            let m = call_methods_dispute(&fake);
            assert!(!m.iter().any(|x| x == "omninode_buildOpenInferenceDispute"));
            assert!(!m.iter().any(|x| x == "sum_sendRawTransaction"));
        }

        #[test]
        fn open_duplicate_dispute_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            let _fixture = seed_open_happy_path(&fake);
            // Overwrite disputes with an existing one for the same verifier:
            fake.set_response(
                "omninode_getInferenceDisputes",
                Ok(json!({
                    "session_id": "s-1",
                    "disputes": [
                        {
                            "verifier_address": verifier_b58(),
                            "opened_at_height": 400_500,
                            "state": "open"
                        }
                    ]
                })),
            );
            let (logs, _s, r) = run_open_with_capture(
                args_open(),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect_err("duplicate dispute must fail");
            assert!(logs.contains("event=\"settlement_dispute_refused_state\""));
            assert!(logs.contains("reason=\"duplicate_dispute\""));
            let m = call_methods_dispute(&fake);
            assert!(!m.iter().any(|x| x == "omninode_buildOpenInferenceDispute"));
            assert!(!m.iter().any(|x| x == "sum_sendRawTransaction"));
        }

        #[test]
        fn open_invalid_evidence_hex_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            let _fixture = seed_open_happy_path(&fake);
            let mut args = args_open();
            args.evidence = "0xNOT_HEX".to_string();
            let (logs, _s, r) = run_open_with_capture(
                args,
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect_err("bad evidence hex must fail");
            assert!(logs.contains("event=\"settlement_dispute_failed\""));
            assert!(logs.contains("category=\"evidence_parse\""));
            let m = call_methods_dispute(&fake);
            assert!(!m.iter().any(|x| x == "omninode_buildOpenInferenceDispute"));
            assert!(!m.iter().any(|x| x == "sum_sendRawTransaction"));
        }

        #[test]
        fn open_builder_envelope_mismatch_refuses_before_submit() {
            let (client, fake) = make_client();
            let fixture = seed_open_happy_path(&fake);
            fake.set_response(
                "omninode_buildOpenInferenceDispute",
                Ok(json!({
                    "unsigned_tx": fixture.unsigned_hex,
                    "signing_hash": fixture.signing_hash_hex,
                    "from": "not-the-derived-address",  // <-- mismatch
                    "nonce": TEST_NONCE,
                    "fee": TEST_FEE,
                    "chain_id": TEST_CHAIN_ID,
                })),
            );
            let (logs, _s, r) = run_open_with_capture(
                args_open(),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect_err("envelope mismatch must fail");
            assert!(logs.contains("event=\"settlement_dispute_failed\""));
            assert!(logs.contains("category=\"builder_mismatch\""));
            let m = call_methods_dispute(&fake);
            assert!(m.iter().any(|x| x == "omninode_buildOpenInferenceDispute"));
            assert!(!m.iter().any(|x| x == "sum_sendRawTransaction"));
        }

        #[test]
        fn open_dry_run_calls_builder_but_never_signs_or_submits() {
            let (client, fake) = make_client();
            let _fixture = seed_open_happy_path(&fake);
            let (logs, stdout, r) = run_open_with_capture(
                args_open_dry_run(),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect("dry-run must succeed");
            assert!(logs.contains("event=\"settlement_dispute_ready\""));
            assert!(!logs.contains("event=\"settlement_dispute_submitted\""));
            let stdout_str = String::from_utf8(stdout).unwrap();
            assert!(stdout_str.contains("dry_run=true submitted=false"));
            let m = call_methods_dispute(&fake);
            assert!(m.iter().any(|x| x == "omninode_buildOpenInferenceDispute"));
            assert!(!m.iter().any(|x| x == "sum_sendRawTransaction"));
        }

        #[test]
        fn open_happy_path_calls_builder_once_and_submit_once_round_trips() {
            let (client, fake) = make_client();
            let fixture = seed_open_happy_path(&fake);
            let (logs, stdout, r) = run_open_with_capture(
                args_open(),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect("happy path");
            assert!(logs.contains("event=\"settlement_dispute_ready\""));
            assert!(logs.contains("event=\"settlement_dispute_submitted\""));
            let m = call_methods_dispute(&fake);
            let bcount = m.iter().filter(|x| x.as_str() == "omninode_buildOpenInferenceDispute").count();
            let scount = m.iter().filter(|x| x.as_str() == "sum_sendRawTransaction").count();
            assert_eq!(bcount, 1, "builder exactly once");
            assert_eq!(scount, 1, "submit exactly once");

            // Round-trip: SignedTransaction::from_hex on the submitted hex.
            let calls = fake.calls();
            let (_m, params) = calls.iter().find(|(m, _)| m == "sum_sendRawTransaction").unwrap();
            let hex_param = params.as_array().unwrap()[0].as_str().unwrap().to_string();
            let decoded = SignedTransaction::from_hex(&hex_param)
                .expect("submitted hex must round-trip via SignedTransaction::from_hex");
            assert_eq!(decoded.signing_hash(), fixture.tx.signing_hash());

            let stdout_str = String::from_utf8(stdout).unwrap();
            assert!(stdout_str.contains("submitted=true"));
        }

        // ── RESOLVE tests ───────────────────────────────────────────

        fn write_approvals_json(approvals: &[ValidatorApproval]) -> std::path::PathBuf {
            let items: Vec<_> = approvals
                .iter()
                .map(|a| {
                    json!({
                        "pubkey":    format!("0x{}", encode_hex_local(&a.pubkey)),
                        "signature": format!("0x{}", encode_hex_local(&a.signature)),
                    })
                })
                .collect();
            let body = serde_json::to_string(&items).unwrap();
            let mut path = std::env::temp_dir();
            path.push(format!(
                "omninode-test-approvals-{}.json",
                std::process::id()
            ));
            std::fs::write(&path, body).unwrap();
            path
        }

        fn make_test_approvals() -> Vec<ValidatorApproval> {
            vec![ValidatorApproval {
                pubkey: [3u8; 32],
                signature: [5u8; 64],
            }]
        }

        fn args_resolve(approvals_path: std::path::PathBuf, allow_claim: bool) -> Box<ResolveDisputeArgs> {
            Box::new(ResolveDisputeArgs {
                session_id: "s-1".to_string(),
                verifier: verifier_b58(),
                allow_claim,
                approvals: approvals_path,
                fee: None,
                dry_run: false,
            })
        }

        fn args_resolve_dry(approvals_path: std::path::PathBuf) -> Box<ResolveDisputeArgs> {
            Box::new(ResolveDisputeArgs {
                session_id: "s-1".to_string(),
                verifier: verifier_b58(),
                allow_claim: true,
                approvals: approvals_path,
                fee: None,
                dry_run: true,
            })
        }

        fn seed_resolve_happy_path(fake: &FakeJsonRpcTransport, approvals: &[ValidatorApproval]) -> TestResolveTx {
            let fixture = build_test_resolve_tx(
                &TEST_SEED,
                "s-1",
                &verifier_b58(),
                true,
                approvals.to_vec(),
                TEST_CHAIN_ID,
                TEST_NONCE,
                TEST_FEE,
            );
            seed_params(fake, params_settlement_active());
            seed_head(fake, 500_000);
            fake.set_response(
                "omninode_getInferenceDisputes",
                Ok(json!({
                    "session_id": "s-1",
                    "disputes": [{
                        "verifier_address": verifier_b58(),
                        "opened_at_height": 400_500,
                        "state": "open"
                    }]
                })),
            );
            fake.set_response(
                "omninode_buildResolveInferenceDispute",
                Ok(json!({
                    "unsigned_tx": fixture.unsigned_hex,
                    "signing_hash": fixture.signing_hash_hex,
                    "from": fixture.from_b58,
                    "nonce": TEST_NONCE,
                    "fee": TEST_FEE,
                    "chain_id": TEST_CHAIN_ID,
                })),
            );
            fake.set_response(
                "sum_sendRawTransaction",
                Ok(json!({ "tx_hash": "0xtxhash-resolve-happy" })),
            );
            fixture
        }

        #[test]
        fn resolve_dormant_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            let approvals = make_test_approvals();
            let path = write_approvals_json(&approvals);
            seed_params(&fake, params_all_dormant());
            seed_head(&fake, 100_000);
            let (logs, _s, r) = run_resolve_with_capture(
                args_resolve(path, true),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect_err("dormant must fail");
            assert!(logs.contains("event=\"settlement_dispute_refused_dormancy\""));
            let m = call_methods_dispute(&fake);
            assert!(!m.iter().any(|x| x == "omninode_buildResolveInferenceDispute"));
            assert!(!m.iter().any(|x| x == "sum_sendRawTransaction"));
        }

        #[test]
        fn resolve_dispute_not_open_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            let approvals = make_test_approvals();
            let path = write_approvals_json(&approvals);
            seed_params(&fake, params_settlement_active());
            seed_head(&fake, 500_000);
            fake.set_response(
                "omninode_getInferenceDisputes",
                Ok(json!({
                    "session_id": "s-1",
                    "disputes": [{
                        "verifier_address": verifier_b58(),
                        "opened_at_height": 400_500,
                        "state": "resolved_approved"       // <-- not open
                    }]
                })),
            );
            let (logs, _s, r) = run_resolve_with_capture(
                args_resolve(path, true),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect_err("dispute not open must fail");
            assert!(logs.contains("event=\"settlement_dispute_refused_state\""));
            assert!(logs.contains("reason=\"dispute_not_open\""));
            let m = call_methods_dispute(&fake);
            assert!(!m.iter().any(|x| x == "omninode_buildResolveInferenceDispute"));
            assert!(!m.iter().any(|x| x == "sum_sendRawTransaction"));
        }

        #[test]
        fn resolve_malformed_approvals_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            seed_params(&fake, params_settlement_active());
            seed_head(&fake, 500_000);
            fake.set_response(
                "omninode_getInferenceDisputes",
                Ok(json!({
                    "session_id": "s-1",
                    "disputes": [{
                        "verifier_address": verifier_b58(),
                        "opened_at_height": 400_500,
                        "state": "open"
                    }]
                })),
            );
            // Write bad JSON — non-hex chars in pubkey.
            let mut path = std::env::temp_dir();
            path.push(format!(
                "omninode-test-bad-approvals-{}.json",
                std::process::id()
            ));
            std::fs::write(
                &path,
                r#"[{"pubkey": "0xNOT_HEX", "signature": "0x00"}]"#,
            )
            .unwrap();

            let (logs, _s, r) = run_resolve_with_capture(
                args_resolve(path, true),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect_err("malformed approvals must fail");
            assert!(logs.contains("event=\"settlement_dispute_failed\""));
            assert!(logs.contains("category=\"approvals_parse\""));
            let m = call_methods_dispute(&fake);
            assert!(!m.iter().any(|x| x == "omninode_buildResolveInferenceDispute"));
            assert!(!m.iter().any(|x| x == "sum_sendRawTransaction"));
        }

        /// Chain-team confirmed (sum-chain#110): an empty approvals
        /// array will BUILD but always FAIL authority at execution
        /// because it cannot meet the validator-quorum threshold.
        /// OmniNode refuses locally BEFORE builder/sign/submit so the
        /// operator doesn't burn a fee on a predictably-rejected tx.
        #[test]
        fn resolve_empty_approvals_refuses_and_never_calls_builder_or_submit() {
            let (client, fake) = make_client();
            seed_params(&fake, params_settlement_active());
            seed_head(&fake, 500_000);
            fake.set_response(
                "omninode_getInferenceDisputes",
                Ok(json!({
                    "session_id": "s-1",
                    "disputes": [{
                        "verifier_address": verifier_b58(),
                        "opened_at_height": 400_500,
                        "state": "open"
                    }]
                })),
            );
            // Well-formed but empty approvals list.
            let mut path = std::env::temp_dir();
            path.push(format!(
                "omninode-test-empty-approvals-{}.json",
                std::process::id()
            ));
            std::fs::write(&path, "[]").unwrap();

            let (logs, _s, r) = run_resolve_with_capture(
                args_resolve(path, true),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect_err("empty approvals must refuse locally");

            assert!(
                logs.contains("event=\"settlement_dispute_refused_state\""),
                "must emit DISPUTE_REFUSED_STATE; logs=\n{logs}"
            );
            assert!(
                logs.contains("reason=\"approvals_empty\""),
                "reason must name approvals_empty; logs=\n{logs}"
            );

            let m = call_methods_dispute(&fake);
            assert!(
                !m.iter().any(|x| x == "omninode_buildResolveInferenceDispute"),
                "builder MUST NOT be called on empty approvals; methods={m:?}"
            );
            assert!(
                !m.iter().any(|x| x == "sum_sendRawTransaction"),
                "submit MUST NOT be called on empty approvals; methods={m:?}"
            );
        }

        #[test]
        fn resolve_approvals_json_accepts_hex_with_or_without_prefix() {
            // Parse two entries — one with `0x`, one without — and
            // both should yield the same bytes.
            use omni_sumchain::settlement_dispute::wire::parse_approvals_json;
            let with = "aa".repeat(32);
            let without = "bb".repeat(32);
            let sig_with = "cc".repeat(64);
            let sig_without = "dd".repeat(64);
            let json = format!(
                r#"[
                    {{ "pubkey": "0x{with}",   "signature": "0x{sig_with}" }},
                    {{ "pubkey": "{without}",  "signature": "{sig_without}" }}
                ]"#
            );
            let parsed = parse_approvals_json(&json).expect("both formats parse");
            assert_eq!(parsed.len(), 2);
            assert_eq!(parsed[0].pubkey, [0xaa; 32]);
            assert_eq!(parsed[1].pubkey, [0xbb; 32]);
            assert_eq!(parsed[0].signature, [0xcc; 64]);
            assert_eq!(parsed[1].signature, [0xdd; 64]);
        }

        #[test]
        fn resolve_dry_run_calls_builder_but_never_signs_or_submits() {
            let (client, fake) = make_client();
            let approvals = make_test_approvals();
            let _fixture = seed_resolve_happy_path(&fake, &approvals);
            let path = write_approvals_json(&approvals);
            let (logs, stdout, r) = run_resolve_with_capture(
                args_resolve_dry(path),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect("dry-run must succeed");
            assert!(logs.contains("event=\"settlement_dispute_ready\""));
            assert!(!logs.contains("event=\"settlement_dispute_submitted\""));
            let s = String::from_utf8(stdout).unwrap();
            assert!(s.contains("dry_run=true submitted=false"));
            let m = call_methods_dispute(&fake);
            assert!(m.iter().any(|x| x == "omninode_buildResolveInferenceDispute"));
            assert!(!m.iter().any(|x| x == "sum_sendRawTransaction"));
        }

        #[test]
        fn resolve_happy_path_calls_builder_once_and_submit_once() {
            let (client, fake) = make_client();
            let approvals = make_test_approvals();
            let fixture = seed_resolve_happy_path(&fake, &approvals);
            let path = write_approvals_json(&approvals);
            let (logs, _stdout, r) = run_resolve_with_capture(
                args_resolve(path, true),
                &client,
                SeedSource::Explicit(TEST_SEED),
            );
            r.expect("happy path");
            assert!(logs.contains("event=\"settlement_dispute_ready\""));
            assert!(logs.contains("event=\"settlement_dispute_submitted\""));
            let m = call_methods_dispute(&fake);
            let bcount = m.iter().filter(|x| x.as_str() == "omninode_buildResolveInferenceDispute").count();
            let scount = m.iter().filter(|x| x.as_str() == "sum_sendRawTransaction").count();
            assert_eq!(bcount, 1);
            assert_eq!(scount, 1);

            // Round-trip decode.
            let calls = fake.calls();
            let (_m, params) = calls.iter().find(|(m, _)| m == "sum_sendRawTransaction").unwrap();
            let hex_param = params.as_array().unwrap()[0].as_str().unwrap().to_string();
            let decoded = SignedTransaction::from_hex(&hex_param).expect("round-trip");
            assert_eq!(decoded.signing_hash(), fixture.tx.signing_hash());
        }

        // ── Structural tests ────────────────────────────────────────

        #[test]
        fn dispute_marker_constant_names_are_pinned() {
            use super::super::dispute_markers;
            assert_eq!(dispute_markers::DISPUTE_READY, "settlement_dispute_ready");
            assert_eq!(dispute_markers::DISPUTE_REFUSED_DORMANCY, "settlement_dispute_refused_dormancy");
            assert_eq!(dispute_markers::DISPUTE_REFUSED_AUTHORITY, "settlement_dispute_refused_authority");
            assert_eq!(dispute_markers::DISPUTE_REFUSED_MATURITY, "settlement_dispute_refused_maturity");
            assert_eq!(dispute_markers::DISPUTE_REFUSED_STATE, "settlement_dispute_refused_state");
            assert_eq!(dispute_markers::DISPUTE_SUBMITTED, "settlement_dispute_submitted");
            assert_eq!(dispute_markers::DISPUTE_FAILED, "settlement_dispute_failed");
        }

        #[test]
        fn dispute_surface_carries_no_forbidden_flags() {
            // Compile-time exhaustive match — any new DisputeCmd
            // variant or OpenArgs/ResolveArgs field breaks this test.
            fn dc_variant_check(cmd: &DisputeCmd) -> &'static str {
                match cmd {
                    DisputeCmd::Open(_) => "open",
                    DisputeCmd::Resolve(_) => "resolve",
                }
            }
            let _ = dc_variant_check;
            let _ = OpenDisputeArgs {
                session_id: "s".to_string(),
                verifier: "v".to_string(),
                evidence: "0x".to_string(),
                fee: None,
                dry_run: false,
            };
            let _ = ResolveDisputeArgs {
                session_id: "s".to_string(),
                verifier: "v".to_string(),
                allow_claim: false,
                approvals: std::path::PathBuf::from("/tmp/x"),
                fee: None,
                dry_run: false,
            };
        }
    }
}
