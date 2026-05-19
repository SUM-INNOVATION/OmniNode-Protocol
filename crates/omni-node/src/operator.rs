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

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Args, Subcommand};
use tracing::{debug, info, warn};

use omni_sumchain::{BlockFinality, JsonRpcTransport, SumChainClient, UreqTransport};
use omni_types::phase5::{InferenceAttestation, InferenceCommitment, SnipV2ObjectId};
use omni_zkml::{
    poll_attestations_workflow, retry_dropped_attestations_workflow,
    signer_chain_address_base58, submit_attestation_workflow_with_block,
    sweep_stale_attestations_workflow, AttestationRegistry, ChainClient,
    ChainClientError, RegistryError, StalenessPolicy,
};

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

    #[error(
        "mainnet smoke (chain_id 1) requires --attestation-json; refusing to \
         synthesize an artificial attestation onto mainnet"
    )]
    MainnetRequiresAttestationJson,

    #[error(
        "--synthetic is forbidden on mainnet (chain_id 1); Stage 8a has no \
         synthetic-mainnet path"
    )]
    SyntheticMainnetForbidden,

    #[error(
        "smoke needs an attestation source: pass --attestation-json, or \
         (non-mainnet only) --synthetic"
    )]
    SyntheticRequiresExplicitFlag,

    #[error("--attestation-json and --synthetic are mutually exclusive")]
    ConflictingSmokeSource,

    #[error("submission not permitted: pass --allow-submit to enable chain writes")]
    SubmitNotPermitted,

    #[error(
        "mainnet submission (chain_id 1) additionally requires \
         --allow-mainnet-submit"
    )]
    MainnetSubmitNotPermitted,

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

    #[error("smoke did not reach Finalized before --confirm-timeout-secs; last status: {last}")]
    SmokeConfirmTimeout { last: String },

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

    #[error("smoke interrupted (Ctrl-C) before finality was confirmed")]
    SmokeInterrupted,

    #[error("--tx-hash is mutually exclusive with --session-id/--verifier-address")]
    ConflictingQueryMode,

    #[error("query needs either --tx-hash, or both --session-id and --verifier-address")]
    QueryModeRequired,

    #[error("internal invariant violated: {0}")]
    Internal(String),

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
    // `#[cfg(test)]`-gated (not `#[allow(dead_code)]`-suppressed) so the
    // production build genuinely contains only `Env` — no dead code,
    // no lint suppression.
    #[cfg(test)]
    Explicit([u8; 32]),
    #[cfg(test)]
    AbsentForTest,
    #[cfg(test)]
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
            #[cfg(test)]
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

#[derive(Debug, PartialEq, Eq)]
enum SmokeSource {
    File(PathBuf),
    Synthetic,
}

/// Gate 7 / attestation-source resolution. **Pure** and seed-free, so
/// callers can (and the runner does) evaluate it before the verifier
/// seed is ever touched.
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
    Smoke(SmokeArgs),
    /// Periodic poll → stale-sweep → (retry) lifecycle loop.
    Loop(LoopArgs),
    /// Read-only: validate seed ↔ attestation-json (+ optional chain snapshot).
    Preflight(PreflightArgs),
    /// Read-only: query the chain by (session_id, verifier_address) OR --tx-hash.
    Query(QueryArgs),
    /// Read-only: print the chain address derived from OMNINODE_VERIFIER_SEED_HEX.
    DeriveAddress,
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
    #[arg(long)]
    allow_submit: bool,
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
                    allow_submit: a.allow_submit,
                    allow_mainnet_submit: a.allow_mainnet_submit,
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
    }
    Ok(())
}

// ── Configs ───────────────────────────────────────────────────────────────────

struct WatchConfig {
    expect_chain_id: u64,
    poll_interval_secs: u64,
    max_ticks: Option<u64>,
}

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
            chain_id = snap.chain_id,
            finality_depth = snap.finality_depth,
            min_fee = snap.min_fee,
            v2_enabled_from_height = ?snap.v2_from,
            omninode_enabled_from_height = ?snap.omninode_from,
            head = snap.head,
            "chain params"
        );
        info!(
            v2_blocks_remaining = ?blocks_remaining(snap.head, snap.v2_from),
            omninode_blocks_remaining = ?blocks_remaining(snap.head, snap.omninode_from),
            v2_active = snap.v2,
            omninode_active = snap.omninode,
            "activation status"
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

/// Structured result of a successful `smoke` run. Returned by
/// [`smoke_core`] (testable) and rendered by [`print_smoke_summary`]
/// for the operator. Only constructed once the tx reaches `Finalized`.
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
struct SmokePending {
    tx_id: String,
    attestation_id: String,
    session_id: String,
    verifier_address: String,
    submitted_at_block: Option<u64>,
}

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
                synth_attestation(&derived, head)
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

            let polled = poll_attestations_workflow(&registry, &read_client)?;
            let swept = sweep_stale_attestations_workflow(&registry, &read_client, &pol)?;

            let mut retried = Vec::new();
            let mut retry_skipped_reason: Option<&'static str> = None;
            if submit_permitted {
                let omninode = read_client.omninode_is_active()?;
                let v2 = read_client.v2_is_active()?;
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
                    chain_id = params.chain_id,
                    finality_depth = params.finality_depth,
                    min_fee = params.min_fee,
                    v2_enabled_from_height = ?params.v2_enabled_from_height,
                    omninode_enabled_from_height = ?params.omninode_enabled_from_height,
                    head = head,
                    "preflight: chain params"
                );
                info!(
                    v2_blocks_remaining = ?blocks_remaining(head, params.v2_enabled_from_height),
                    omninode_blocks_remaining = ?blocks_remaining(head, params.omninode_enabled_from_height),
                    v2_active = v2,
                    omninode_active = omninode,
                    activated = omninode && v2,
                    "preflight: chain snapshot (report-only — valid \
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

    fn status_json(status: &str) -> serde_json::Value {
        json!({"status": status, "included_at_height": 101, "reason": null})
    }

    /// Activated non-mainnet fake wired for a synthetic submit, with the
    /// status RPC seeded to `status` (overridable mid-test via the
    /// shared `Arc<Mutex>` for sequenced included→finalized coverage).
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
}
