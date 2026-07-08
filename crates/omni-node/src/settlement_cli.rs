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
    }
}

// ── status ───────────────────────────────────────────────────────────────────

fn run_status<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    out: &mut dyn Write,
) -> Result<()> {
    let params = client
        .get_chain_params()
        .map_err(|e| anyhow::anyhow!("chain_getChainParams: {e}"))?;
    let head = client
        .get_block_height(BlockFinality::Latest)
        .map_err(|e| anyhow::anyhow!("chain_getBlockHeight: {e}"))?
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
    let session_id = &args.session_id;

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
            return Ok(());
        }
        Err(e) => return Err(settlement_read_error_to_anyhow(e)),
    };

    let claims = client
        .omninode_get_inference_claims(session_id)
        .map_err(settlement_read_error_to_anyhow)?;
    let disputes = client
        .omninode_get_inference_disputes(session_id)
        .map_err(settlement_read_error_to_anyhow)?;

    // Attestations RPC is unconditional (no gate) and already lives
    // on SumChainClient.
    let attestations = client.list_attestations(session_id).map_err(|e| {
        anyhow::anyhow!("sum_listInferenceAttestations: {e}")
    })?;

    let params = client
        .get_chain_params()
        .map_err(|e| anyhow::anyhow!("chain_getChainParams: {e}"))?;
    let head = client
        .get_block_height(BlockFinality::Latest)
        .map_err(|e| anyhow::anyhow!("chain_getBlockHeight: {e}"))?
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
                .map_err(settlement_read_error_to_anyhow)?,
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
            if let Some(r) = client
                .omninode_get_verifier(&addr)
                .map_err(settlement_read_error_to_anyhow)?
            {
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
    .map_err(settlement_read_error_to_anyhow)?;

    if args.json {
        writeln!(out, "{}", serde_json::to_string_pretty(&view_to_json(&view))?)?;
    } else {
        render_session_view_human(out, &view)?;
    }
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
            "    bond: amount={} state={} unbonding_since={} withdrawable_at={} slashes={}",
            b.bond_amount,
            render_bond_state(&b.bond_state),
            opt_u64(b.unbonding_since_height),
            opt_u64(b.withdrawable_at_height),
            b.slash_history_len,
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
            "slash_history_len": b.slash_history_len,
        })),
    })
}

// ── claimable ────────────────────────────────────────────────────────────────

fn run_claimable<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    args: &ClaimableArgs,
    out: &mut dyn Write,
) -> Result<()> {
    let r = client
        .omninode_get_claimable_reward(&args.session_id, &args.verifier)
        .map_err(settlement_read_error_to_anyhow)?;
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
    Ok(())
}

// ── verifier ─────────────────────────────────────────────────────────────────

fn run_verifier<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    args: &VerifierArgs,
    out: &mut dyn Write,
) -> Result<()> {
    match client
        .omninode_get_verifier(&args.address)
        .map_err(settlement_read_error_to_anyhow)?
    {
        None => {
            writeln!(out, "address={} found=false", args.address)?;
        }
        Some(v) => {
            writeln!(out, "address={}", v.address)?;
            writeln!(out, "bond_amount={}", v.bond_amount)?;
            writeln!(out, "bond_state={}", v.bond_state)?;
            if let Some(h) = v.unbonding_since_height {
                writeln!(out, "unbonding_since_height={h}")?;
            }
            if let Some(h) = v.withdrawable_at_height {
                writeln!(out, "withdrawable_at_height={h}")?;
            }
            writeln!(out, "slash_history_len={}", v.slash_history.len())?;
        }
    }
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
                "address": "v-A",
                "bond_amount": "10000",
                "bond_state": "bonded",
                "slash_history": []
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
}
