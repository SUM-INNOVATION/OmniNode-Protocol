//! Phase 5 Stage 13.0 / 13.2 — chain-anchor CLI surface for
//! Stage 12.25 signed-chain-report artifacts.
//!
//! Lives on `omni-node operator …` as five subcommands:
//!
//! - `submit-integrity-evidence-anchor` (gated `--features submit`)
//! - `query-integrity-evidence-anchor`
//! - `reconcile-integrity-evidence-anchor` (Stage 13.2)
//! - `verify-integrity-evidence-anchor`         — registry-backed
//! - `verify-integrity-evidence-anchor-file`    — standalone JSON
//!
//! ## Stage 13.0 vs Stage 13.2 chain-mode fork
//!
//! - Stage 13.0 (stub mode, default): `submit` / `query` operate
//!   against the local stub client + registry only. Verify stays
//!   chain-untouched in either stage.
//! - Stage 13.2 (chain mode, opt-in via `--rpc-url` +
//!   `--expect-chain-id`): `submit` / `query` / `reconcile`
//!   talk to the real SUM Chain via `omni-sumchain`'s
//!   `EvidenceAnchorChainClient` impl. CLI preflight enforces
//!   chain_id, anchor-activation, and operator double-gates
//!   BEFORE the workflow is invoked; the adapter independently
//!   re-checks activation + same-key as defense-in-depth.
//!
//! The submit command performs Stage 12.25 wrapper parse +
//! signature verification BEFORE building / submitting the anchor
//! digest. Metadata (artifact schema, signer pubkey, signed_at) is
//! lifted from the wrapper; the artifact hash is BLAKE3 over the
//! raw on-disk bytes (NOT a re-serialised representation).

use std::path::PathBuf;

#[cfg(feature = "submit")]
use anyhow::Context;
use anyhow::{Result, anyhow, bail};
#[cfg(feature = "submit")]
use chrono::DateTime;
use clap::{Args, Subcommand};

use omni_contributor::{
    SignedIntegrityEvidenceChainReport, verify_signed_integrity_evidence_chain_report,
};
use omni_sumchain::SumChainClient;
use omni_zkml::EvidenceAnchorError;
use omni_zkml::{
    AnchorRecord, AnchorSelector, AnchorStatus, ChainClientError,
    INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION, IntegrityEvidenceAnchorTxData,
    LocalAnchorStatus, LocalEvidenceAnchorRegistry, anchor_hex_lower,
    evidence_anchor_reason_tag, parse_anchor_hex_32, query_evidence_anchor_workflow,
    reconcile_evidence_anchors_workflow, verify_anchor_against_registry,
    verify_anchor_file_against_artifact_bytes,
};
#[cfg(feature = "submit")]
use omni_zkml::{
    VerifiedWrapperMetadata, anchor_signer_pubkey_bytes, build_anchor_digest,
    submit_evidence_anchor_workflow,
};

// ── CLI Args ──────────────────────────────────────────────────────────────────

#[derive(Args)]
pub(crate) struct EvidenceAnchorArgs {
    #[command(subcommand)]
    cmd: EvidenceAnchorCmd,
}

#[derive(Subcommand)]
enum EvidenceAnchorCmd {
    /// Anchor a Stage 12.25 `SignedIntegrityEvidenceChainReport`.
    /// Without `--rpc-url` uses the local stub client + registry
    /// (Stage 13.0 path). With `--rpc-url` + `--expect-chain-id`
    /// submits to SUM Chain via `omni-sumchain`'s
    /// `EvidenceAnchorChainClient` adapter (Stage 13.2 path);
    /// chain writes additionally require `--allow-submit`
    /// (and `--allow-mainnet-submit` on `chain_id == 1`).
    /// Gated behind `--features submit`.
    #[cfg(feature = "submit")]
    SubmitIntegrityEvidenceAnchor(SubmitAnchorArgs),
    /// Query an anchor by `--artifact-hash-hex` or `--tx-id`.
    /// Without `--rpc-url` reads the local stub registry only
    /// (Stage 13.0 path). With `--rpc-url` + `--expect-chain-id`
    /// queries the chain by stored `tx_id` and applies any
    /// chain-returned status transition to the local record
    /// (Stage 13.2 — chain-read-only, local-registry-mutating).
    QueryIntegrityEvidenceAnchor(QueryAnchorArgs),
    /// Stage 13.2 — sweep the local registry; query the chain
    /// for every `Submitted` / `Included` record and apply
    /// chain-returned status transitions to the local registry.
    /// Chain-read-only, local-registry-mutating. Per-record RPC
    /// failures land in the per-record event line; the sweep
    /// continues.
    ReconcileIntegrityEvidenceAnchor(ReconcileAnchorArgs),
    /// Registry-backed verify — proves the on-disk artifact
    /// corresponds to a recorded anchor authored by the
    /// artifact's signer.
    VerifyIntegrityEvidenceAnchor(VerifyAnchorArgs),
    /// Standalone-JSON verify — checks an anchor JSON against
    /// local artifact bytes WITHOUT consulting the registry.
    /// Does not prove submission / inclusion.
    VerifyIntegrityEvidenceAnchorFile(VerifyAnchorFileArgs),
}

#[cfg(feature = "submit")]
#[derive(Args)]
struct SubmitAnchorArgs {
    /// Path to a Stage 12.25 `SignedIntegrityEvidenceChainReport`
    /// JSON wrapper.
    #[arg(long)]
    signed_chain_report: PathBuf,

    /// 32-byte raw Ed25519 seed file for the anchor submitter.
    /// Same-key-submitter rule (Stage 13.0): MUST derive a
    /// pubkey equal to the wrapper's `signer_pubkey_hex`.
    #[arg(long)]
    submitter_seed: PathBuf,

    /// Directory in which the anchor record + tx_index live.
    /// Distinct from the Stage 12.7 contributor `--state-dir`;
    /// the directory name makes the boundary unambiguous.
    #[arg(long)]
    anchor_registry_dir: PathBuf,

    /// Optional: write the produced anchor wire payload as
    /// pretty JSON to this path (atomic temp+rename). Useful
    /// for distributing the anchor to peers out-of-band.
    #[arg(long)]
    json_out: Option<PathBuf>,

    // ── Stage 13.2 chain-mode flags ────────────────────────────
    /// Stage 13.2 — chain mode opt-in. When supplied, the CLI
    /// runs preflight (chain_id, anchor activation, opt-ins)
    /// and submits to the real SUM Chain via `omni-sumchain`.
    /// When omitted, falls back to the Stage 13.0 stub-client
    /// path.
    #[arg(long)]
    rpc_url: Option<String>,
    /// Required iff `--rpc-url` is given. Sanity-checked against
    /// `params.chain_id` in CLI preflight; mismatches refuse
    /// with `chain_id_mismatch` before any anchor RPC fires.
    #[arg(long)]
    expect_chain_id: Option<u64>,
    /// Required for chain-mode submit. Mirrors Stage 9a
    /// `operator smoke --allow-submit` posture: explicit opt-in
    /// to chain writes.
    #[arg(long)]
    allow_submit: bool,
    /// Required additionally when `params.chain_id == 1`
    /// (mainnet). Mirrors Stage 9a `--allow-mainnet-submit`.
    /// Does NOT override the `mainnet_policy_unresolved`
    /// refusal — chain governance must have set anchor
    /// activation before mainnet submits are permitted.
    #[arg(long)]
    allow_mainnet_submit: bool,
}

#[derive(Args)]
struct QueryAnchorArgs {
    /// Directory in which the anchor record + tx_index live.
    #[arg(long)]
    anchor_registry_dir: PathBuf,

    /// Lookup selector. Mutually exclusive with `--tx-id`;
    /// either MUST be supplied.
    #[arg(long)]
    artifact_hash_hex: Option<String>,

    /// Lookup selector. Mutually exclusive with
    /// `--artifact-hash-hex`; either MUST be supplied.
    #[arg(long)]
    tx_id: Option<String>,

    // ── Stage 13.2 chain-mode flags ────────────────────────────
    /// Stage 13.2 — chain-mode opt-in. When supplied (with
    /// `--expect-chain-id`), the CLI queries `omni-sumchain`'s
    /// `sum_getIntegrityEvidenceAnchorStatus` for the record's
    /// stored `tx_id` and applies the chain-returned status to
    /// the local record. **Chain-read-only,
    /// local-registry-mutating.** Without `--rpc-url`, behaves
    /// as Stage 13.0 (registry-only).
    #[arg(long)]
    rpc_url: Option<String>,
    /// Required iff `--rpc-url` is given.
    #[arg(long)]
    expect_chain_id: Option<u64>,
}

#[derive(Args)]
struct ReconcileAnchorArgs {
    /// Directory in which the anchor record + tx_index live.
    #[arg(long)]
    anchor_registry_dir: PathBuf,
    /// SUM Chain JSON-RPC endpoint URL. Required.
    #[arg(long)]
    rpc_url: String,
    /// Chain-id sanity check. Required.
    #[arg(long)]
    expect_chain_id: u64,
}

#[derive(Args)]
struct VerifyAnchorArgs {
    /// Path to a Stage 12.25 `SignedIntegrityEvidenceChainReport`
    /// JSON wrapper. Hashed raw to recompute the artifact hash.
    #[arg(long)]
    signed_chain_report: PathBuf,

    /// Directory in which the anchor record + tx_index live.
    #[arg(long)]
    anchor_registry_dir: PathBuf,

    /// Optional: look up by stored `tx_id` instead of the
    /// recomputed artifact hash. The recorded hash MUST still
    /// match the recomputed hash; mismatches refuse with
    /// `artifact_hash_mismatch`.
    #[arg(long)]
    tx_id: Option<String>,
}

#[derive(Args)]
struct VerifyAnchorFileArgs {
    /// Path to a Stage 12.25 `SignedIntegrityEvidenceChainReport`
    /// JSON wrapper.
    #[arg(long)]
    signed_chain_report: PathBuf,

    /// Path to a free-floating anchor JSON
    /// (`IntegrityEvidenceAnchorTxData`).
    #[arg(long)]
    anchor_json: PathBuf,
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

pub(crate) async fn dispatch(args: EvidenceAnchorArgs) -> Result<()> {
    match args.cmd {
        #[cfg(feature = "submit")]
        EvidenceAnchorCmd::SubmitIntegrityEvidenceAnchor(a) => {
            tokio::task::spawn_blocking(move || run_submit(a))
                .await
                .map_err(|e| anyhow!("submit-anchor join error: {e}"))?
        }
        EvidenceAnchorCmd::QueryIntegrityEvidenceAnchor(a) => {
            tokio::task::spawn_blocking(move || run_query(a))
                .await
                .map_err(|e| anyhow!("query-anchor join error: {e}"))?
        }
        EvidenceAnchorCmd::ReconcileIntegrityEvidenceAnchor(a) => {
            tokio::task::spawn_blocking(move || run_reconcile(a))
                .await
                .map_err(|e| anyhow!("reconcile-anchor join error: {e}"))?
        }
        EvidenceAnchorCmd::VerifyIntegrityEvidenceAnchor(a) => {
            tokio::task::spawn_blocking(move || run_verify(a))
                .await
                .map_err(|e| anyhow!("verify-anchor join error: {e}"))?
        }
        EvidenceAnchorCmd::VerifyIntegrityEvidenceAnchorFile(a) => {
            tokio::task::spawn_blocking(move || run_verify_file(a))
                .await
                .map_err(|e| anyhow!("verify-anchor-file join error: {e}"))?
        }
    }
}

// ── Stage 13.2 chain-mode preflight helpers ───────────────────────────────────

/// Map any `ChainClientError` produced by `omni-sumchain` into
/// the closed Stage 13.2 [`EvidenceAnchorError`] set, using the
/// typed classifier as the single chokepoint (no CLI-side
/// prefix matching).
fn chain_client_error_to_evidence_anchor_error(err: ChainClientError) -> EvidenceAnchorError {
    use omni_sumchain::{ChainErrorCategory, classify_chain_client_error};
    let text = match &err {
        ChainClientError::Other(s) => s.clone(),
    };
    match classify_chain_client_error(&err) {
        ChainErrorCategory::Transport => EvidenceAnchorError::ChainRpc(text),
        ChainErrorCategory::JsonRpcError => EvidenceAnchorError::ChainSubmitRefused(text),
        ChainErrorCategory::Malformed => EvidenceAnchorError::ChainResponseMalformed(text),
        ChainErrorCategory::AdapterNotActivated => EvidenceAnchorError::NotActivated {
            // The adapter-level refusal carries no chain_id
            // context (the adapter doesn't know `--expect-chain-id`).
            // Use 0 as a sentinel; the CLI preflight path always
            // refuses earlier with the real chain_id, so this is
            // only reached when a non-CLI caller bypasses
            // preflight.
            chain_id: 0,
            activation_status: text,
        },
        // Adapter same-key check is a defense-in-depth catch
        // for a misconfigured caller — surface as the catch-all
        // chain_rpc tag per Stage 13.2 mapper rule.
        ChainErrorCategory::AdapterSameKeyFail => EvidenceAnchorError::ChainRpc(text),
        ChainErrorCategory::Unknown => EvidenceAnchorError::ChainRpc(text),
    }
}

/// Run CLI preflight gates for chain-mode submit. Returns on
/// first refusal; on success the caller proceeds to invoke
/// `submit_evidence_anchor_workflow` against the real
/// `SumChainClient`.
///
/// Gate order (matches the locked Stage 13.2 plan):
/// 1. Fetch chain params (single RPC).
/// 2. Chain-id sanity check (`expected == params.chain_id`).
/// 3. Anchor-activation check (mainnet-aware tagging):
///    - mainnet + `None` → `MainnetPolicyUnresolved`
///    - any + `Some(h)` but `head < h` → `NotActivated` (scheduled)
///    - non-mainnet + `None` → `NotActivated` (dormant)
/// 4. `--allow-submit` opt-in.
/// 5. mainnet AND `--allow-mainnet-submit` opt-in.
#[cfg(feature = "submit")]
fn run_chain_submit_preflight(
    client: &SumChainClient,
    expected_chain_id: u64,
    allow_submit: bool,
    allow_mainnet_submit: bool,
) -> Result<(), EvidenceAnchorError> {
    use omni_sumchain::BlockFinality;
    // Gate 1+2: chain params + chain_id sanity.
    let params = client
        .get_chain_params()
        .map_err(chain_client_error_to_evidence_anchor_error)?;
    if params.chain_id != expected_chain_id {
        return Err(EvidenceAnchorError::ChainIdMismatch {
            expected: expected_chain_id,
            actual: params.chain_id,
        });
    }
    // Gate 3: anchor-activation, mainnet-aware tagging.
    let activation = params.integrity_evidence_anchor_enabled_from_height;
    let is_mainnet = params.chain_id == 1;
    match activation {
        None => {
            if is_mainnet {
                return Err(EvidenceAnchorError::MainnetPolicyUnresolved);
            }
            return Err(EvidenceAnchorError::NotActivated {
                chain_id: params.chain_id,
                activation_status: "dormant (no activation height set)".to_string(),
            });
        }
        Some(h) => {
            let head = client
                .get_block_height(BlockFinality::Latest)
                .map_err(chain_client_error_to_evidence_anchor_error)?
                .height;
            if head < h {
                return Err(EvidenceAnchorError::NotActivated {
                    chain_id: params.chain_id,
                    activation_status: format!(
                        "scheduled at height {h}, chain head at {head}"
                    ),
                });
            }
        }
    }
    // Gate 4: --allow-submit opt-in.
    if !allow_submit {
        return Err(EvidenceAnchorError::ChainRpc(
            "chain writes not permitted: pass --allow-submit to enable submission".to_string(),
        ));
    }
    // Gate 5: mainnet --allow-mainnet-submit opt-in.
    if is_mainnet && !allow_mainnet_submit {
        return Err(EvidenceAnchorError::ChainRpc(
            "mainnet (chain_id 1) writes additionally require --allow-mainnet-submit"
                .to_string(),
        ));
    }
    Ok(())
}

/// Run CLI preflight for chain-mode read paths (query, reconcile).
/// Only checks chain_id sanity — read paths neither require
/// activation (anchors recorded pre-deactivation are still
/// queryable) nor the operator double-gates.
fn run_chain_read_preflight(
    client: &SumChainClient,
    expected_chain_id: u64,
) -> Result<(), EvidenceAnchorError> {
    let params = client
        .get_chain_params()
        .map_err(chain_client_error_to_evidence_anchor_error)?;
    if params.chain_id != expected_chain_id {
        return Err(EvidenceAnchorError::ChainIdMismatch {
            expected: expected_chain_id,
            actual: params.chain_id,
        });
    }
    Ok(())
}

// ── Submit ────────────────────────────────────────────────────────────────────

#[cfg(feature = "submit")]
fn run_submit(args: SubmitAnchorArgs) -> Result<()> {
    println!(
        "event=integrity_evidence_anchor_submit_started signed_chain_report={}",
        args.signed_chain_report.display()
    );

    // 1. Read raw on-disk bytes of the wrapper — the bytes the
    //    operator actually holds and the bytes that will be
    //    hashed into the anchor digest.
    let raw_bytes = std::fs::read(&args.signed_chain_report).map_err(|e| {
        let reason = "io";
        println!("event=integrity_evidence_anchor_submit_failed reason={reason} detail={e}");
        anyhow!(
            "read signed-chain-report {}: {e}",
            args.signed_chain_report.display()
        )
    })?;

    // 2. Parse the wrapper for metadata extraction.
    let wrapper: SignedIntegrityEvidenceChainReport =
        serde_json::from_slice(&raw_bytes).map_err(|e| {
            let reason = "malformed_json";
            println!("event=integrity_evidence_anchor_submit_failed reason={reason} detail={e}");
            anyhow!(
                "parse signed-chain-report {}: {e}",
                args.signed_chain_report.display()
            )
        })?;

    // 3. Pre-submit gate — verify the wrapper signature under
    //    its own embedded pubkey. We refuse to anchor an
    //    unverifiable artifact.
    if let Err(e) =
        verify_signed_integrity_evidence_chain_report(&wrapper, &wrapper.signer_pubkey_hex)
    {
        let reason = "wrapper_signature_invalid";
        println!("event=integrity_evidence_anchor_submit_failed reason={reason} detail={e}");
        bail!(
            "Stage 12.25 wrapper signature did not verify under its embedded signer_pubkey_hex: {e}"
        );
    }

    // 4. Build the verified-metadata struct the library expects.
    let metadata = wrapper_metadata(&wrapper).map_err(|err| {
        let reason = evidence_anchor_reason_tag(&err);
        println!("event=integrity_evidence_anchor_submit_failed reason={reason} detail={err}");
        anyhow!("extract Stage 12.25 wrapper metadata: {err}")
    })?;

    // 5. Read + validate submitter seed file.
    let submitter_seed = read_seed_file(&args.submitter_seed).map_err(|err| {
        let reason = evidence_anchor_reason_tag(&err);
        println!("event=integrity_evidence_anchor_submit_failed reason={reason} detail={err}");
        anyhow!("read submitter seed: {err}")
    })?;

    // 6. Open registry; build digest; submit through stub client.
    let registry =
        LocalEvidenceAnchorRegistry::open(args.anchor_registry_dir.clone()).map_err(|e| {
            let reason = "io";
            println!("event=integrity_evidence_anchor_submit_failed reason={reason} detail={e}");
            anyhow!(
                "open --anchor-registry-dir {}: {e}",
                args.anchor_registry_dir.display()
            )
        })?;

    let digest = build_anchor_digest(&metadata, &raw_bytes);

    // ── Stage 13.0 vs Stage 13.2 fork ─────────────────────────
    // Without --rpc-url: stub client (Stage 13.0 path).
    // With --rpc-url + --expect-chain-id: real SumChainClient
    // after CLI preflight gates.
    let record = match (args.rpc_url.as_deref(), args.expect_chain_id) {
        (Some(url), Some(expected_chain_id)) => {
            let client = SumChainClient::new(url.to_string(), submitter_seed);
            // CLI preflight (chain_id + activation + opt-ins).
            run_chain_submit_preflight(
                &client,
                expected_chain_id,
                args.allow_submit,
                args.allow_mainnet_submit,
            )
            .map_err(|err| {
                let reason = evidence_anchor_reason_tag(&err);
                println!(
                    "event=integrity_evidence_anchor_submit_failed reason={reason} detail={err}"
                );
                anyhow!("submit anchor refused at preflight: {err}")
            })?;
            // Submit via the real adapter. The adapter
            // independently re-checks activation + same-key as
            // defense-in-depth.
            submit_evidence_anchor_workflow(&registry, &client, digest, &submitter_seed)
                .map_err(|err| {
                    // Chain-side errors are routed through the
                    // typed classifier; surface the closed tag.
                    let mapped = if let EvidenceAnchorError::ChainClient(inner) = err {
                        chain_client_error_to_evidence_anchor_error(inner)
                    } else {
                        err
                    };
                    let reason = evidence_anchor_reason_tag(&mapped);
                    println!(
                        "event=integrity_evidence_anchor_submit_failed reason={reason} detail={mapped}"
                    );
                    anyhow!("submit anchor refused: {mapped}")
                })?
        }
        (None, None) => {
            // Stage 13.0 stub mode.
            let client = omni_zkml::StubEvidenceAnchorChainClient::new();
            submit_evidence_anchor_workflow(&registry, &client, digest, &submitter_seed)
                .map_err(|err| {
                    let reason = evidence_anchor_reason_tag(&err);
                    println!(
                        "event=integrity_evidence_anchor_submit_failed reason={reason} detail={err}"
                    );
                    anyhow!("submit anchor refused: {err}")
                })?
        }
        _ => {
            // Partial chain-mode flags (e.g. only --rpc-url
            // without --expect-chain-id, or vice versa). Refuse
            // at parse time with a clear message.
            let reason = "chain_rpc";
            let detail = "chain mode requires BOTH --rpc-url and --expect-chain-id";
            println!(
                "event=integrity_evidence_anchor_submit_failed reason={reason} detail={detail}"
            );
            bail!(detail);
        }
    };

    println!(
        "event=integrity_evidence_anchor_submit_ok artifact_hash_hex={} signer_pubkey_hex={} tx_id={} \
         anchor_schema_version={} artifact_schema_version={} artifact_kind={}",
        record.artifact_hash_hex,
        record.signer_pubkey_hex,
        record.receipt.tx_id,
        record.tx_data.digest.anchor_schema_version,
        record.tx_data.digest.artifact_schema_version,
        record.tx_data.digest.artifact_kind.as_str(),
    );

    if let Some(path) = args.json_out.as_ref() {
        write_anchor_tx_data_atomic(&record.tx_data, path).map_err(|e| {
            println!("event=integrity_evidence_anchor_submit_failed reason=io detail={e}");
            anyhow!("write --json-out {}: {e}", path.display())
        })?;
        println!(
            "event=integrity_evidence_anchor_json_written path={}",
            path.display()
        );
    }
    Ok(())
}

// ── Query ─────────────────────────────────────────────────────────────────────

fn run_query(args: QueryAnchorArgs) -> Result<()> {
    println!(
        "event=integrity_evidence_anchor_query_started anchor_registry_dir={}",
        args.anchor_registry_dir.display()
    );

    // Selector validation: exactly one of --artifact-hash-hex /
    // --tx-id must be supplied.
    let (selector_owned, selector_label) = match (
        args.artifact_hash_hex.as_deref(),
        args.tx_id.as_deref(),
    ) {
        (Some(_), Some(_)) => {
            let reason = "selector_conflict";
            println!(
                "event=integrity_evidence_anchor_query_failed reason={reason} \
                 detail=--artifact-hash-hex and --tx-id are mutually exclusive"
            );
            bail!(
                "--artifact-hash-hex and --tx-id are mutually exclusive on query-integrity-evidence-anchor"
            );
        }
        (None, None) => {
            let reason = "selector_missing";
            println!(
                "event=integrity_evidence_anchor_query_failed reason={reason} \
                 detail=one of --artifact-hash-hex / --tx-id is required"
            );
            bail!("query-integrity-evidence-anchor needs either --artifact-hash-hex or --tx-id");
        }
        (Some(h), None) => (
            SelectorOwned::ArtifactHashHex(h.to_string()),
            format!("artifact_hash={h}"),
        ),
        (None, Some(t)) => (SelectorOwned::TxId(t.to_string()), format!("tx_id={t}")),
    };

    let registry =
        LocalEvidenceAnchorRegistry::open(args.anchor_registry_dir.clone()).map_err(|e| {
            println!("event=integrity_evidence_anchor_query_failed reason=io detail={e}");
            anyhow!(
                "open --anchor-registry-dir {}: {e}",
                args.anchor_registry_dir.display()
            )
        })?;

    // ── Stage 13.0 vs Stage 13.2 fork ─────────────────────────
    // Without --rpc-url: stub client (Stage 13.0 path).
    // With --rpc-url + --expect-chain-id: real SumChainClient.
    // The workflow is identical in both branches; only the
    // client differs.
    let selector_for_workflow = match &selector_owned {
        SelectorOwned::ArtifactHashHex(h) => AnchorSelector::ArtifactHashHex(h),
        SelectorOwned::TxId(t) => AnchorSelector::TxId(t),
    };
    let outcome_result = match (args.rpc_url.as_deref(), args.expect_chain_id) {
        (Some(url), Some(expected_chain_id)) => {
            // The seed isn't used by read-only RPCs; pass a
            // throwaway-zero seed (matching Stage 8a's
            // DUMMY_SEED posture for read-only ops).
            let dummy_seed = [0u8; 32];
            let client = SumChainClient::new(url.to_string(), dummy_seed);
            if let Err(err) = run_chain_read_preflight(&client, expected_chain_id) {
                let reason = evidence_anchor_reason_tag(&err);
                println!(
                    "event=integrity_evidence_anchor_query_failed reason={reason} \
                     selector={selector_label} detail={err}"
                );
                bail!("query anchor refused at preflight: {err}");
            }
            query_evidence_anchor_workflow(&registry, &client, selector_for_workflow)
                .map_err(|err| match err {
                    EvidenceAnchorError::ChainClient(inner) => {
                        chain_client_error_to_evidence_anchor_error(inner)
                    }
                    other => other,
                })
        }
        (None, None) => {
            let client = omni_zkml::StubEvidenceAnchorChainClient::new();
            query_evidence_anchor_workflow(&registry, &client, selector_for_workflow)
        }
        _ => {
            let reason = "chain_rpc";
            let detail = "chain mode requires BOTH --rpc-url and --expect-chain-id";
            println!(
                "event=integrity_evidence_anchor_query_failed reason={reason} \
                 selector={selector_label} detail={detail}"
            );
            bail!(detail);
        }
    };
    let outcome = outcome_result.map_err(|err| {
        let reason = evidence_anchor_reason_tag(&err);
        println!(
            "event=integrity_evidence_anchor_query_failed reason={reason} \
             selector={selector_label} detail={err}"
        );
        anyhow!("query anchor refused: {err}")
    })?;

    let chain_status_tag = chain_status_tag(&outcome.chain_status);
    println!(
        "event=integrity_evidence_anchor_query_ok artifact_hash_hex={} tx_id={} \
         local_status={} chain_status={} transitioned={}",
        outcome.record.artifact_hash_hex,
        outcome.record.receipt.tx_id,
        outcome.record.status.as_str(),
        chain_status_tag,
        outcome.local_status_transitioned,
    );
    Ok(())
}

// ── Reconcile (Stage 13.2) ────────────────────────────────────────────────────

fn run_reconcile(args: ReconcileAnchorArgs) -> Result<()> {
    println!(
        "event=integrity_evidence_anchor_reconcile_started anchor_registry_dir={} \
         rpc_url={} expect_chain_id={}",
        args.anchor_registry_dir.display(),
        args.rpc_url,
        args.expect_chain_id,
    );

    let registry =
        LocalEvidenceAnchorRegistry::open(args.anchor_registry_dir.clone()).map_err(|e| {
            println!(
                "event=integrity_evidence_anchor_reconcile_failed reason=io detail={e}"
            );
            anyhow!(
                "open --anchor-registry-dir {}: {e}",
                args.anchor_registry_dir.display()
            )
        })?;

    let dummy_seed = [0u8; 32];
    let client = SumChainClient::new(args.rpc_url.clone(), dummy_seed);

    if let Err(err) = run_chain_read_preflight(&client, args.expect_chain_id) {
        let reason = evidence_anchor_reason_tag(&err);
        println!(
            "event=integrity_evidence_anchor_reconcile_failed reason={reason} detail={err}"
        );
        bail!("reconcile refused at preflight: {err}");
    }

    let entries = reconcile_evidence_anchors_workflow(&registry, &client).map_err(|err| {
        let reason = evidence_anchor_reason_tag(&err);
        println!(
            "event=integrity_evidence_anchor_reconcile_failed reason={reason} detail={err}"
        );
        anyhow!("reconcile sweep failed: {err}")
    })?;

    let mut ok_count = 0u64;
    let mut transitioned_count = 0u64;
    let mut err_count = 0u64;
    for (artifact_hash_hex, result) in &entries {
        match result {
            Ok(outcome) => {
                ok_count += 1;
                if outcome.local_status_transitioned {
                    transitioned_count += 1;
                }
                let chain_tag = chain_status_tag(&outcome.chain_status);
                println!(
                    "event=integrity_evidence_anchor_reconcile_record_ok \
                     artifact_hash_hex={artifact_hash_hex} tx_id={} \
                     local_status={} chain_status={chain_tag} transitioned={}",
                    outcome.record.receipt.tx_id,
                    outcome.record.status.as_str(),
                    outcome.local_status_transitioned,
                );
            }
            Err(err) => {
                err_count += 1;
                let reason = reconcile_record_reason_tag(err);
                println!(
                    "event=integrity_evidence_anchor_reconcile_record_failed \
                     artifact_hash_hex={artifact_hash_hex} reason={reason} detail={err}"
                );
            }
        }
    }
    println!(
        "event=integrity_evidence_anchor_reconcile_summary ok={ok_count} \
         transitioned={transitioned_count} failed={err_count}"
    );
    Ok(())
}

/// Resolve the closed-set `reason=<tag>` for a per-record
/// reconcile failure from a BORROWED `EvidenceAnchorError`.
///
/// Non-chain errors (`io` / `anchor_not_found` /
/// `malformed_json` / …) keep their real tag — they describe
/// local registry / data conditions, not chain transport
/// failures. Only `EvidenceAnchorError::ChainClient(inner)`
/// gets routed through the typed
/// [`omni_sumchain::classify_chain_client_error`] to pick
/// between `chain_rpc` / `chain_submit_refused` /
/// `chain_response_malformed`.
///
/// Pinned by [`reconcile_record_reason_tag_tests`] below so the
/// taxonomy can't drift back to "everything is chain_rpc".
fn reconcile_record_reason_tag(err: &EvidenceAnchorError) -> &'static str {
    match err {
        EvidenceAnchorError::ChainClient(inner) => {
            let mapped = chain_client_error_to_evidence_anchor_error(inner.clone());
            evidence_anchor_reason_tag(&mapped)
        }
        other => evidence_anchor_reason_tag(other),
    }
}

enum SelectorOwned {
    ArtifactHashHex(String),
    TxId(String),
}

fn chain_status_tag(status: &AnchorStatus) -> &'static str {
    match status {
        AnchorStatus::Submitted => "submitted",
        AnchorStatus::Included => "included",
        AnchorStatus::Finalized => "finalized",
        AnchorStatus::Failed { .. } => "failed",
        AnchorStatus::Unknown => "unknown",
    }
}

// ── Verify (registry-backed) ──────────────────────────────────────────────────

fn run_verify(args: VerifyAnchorArgs) -> Result<()> {
    println!(
        "event=integrity_evidence_anchor_verify_started signed_chain_report={} \
         anchor_registry_dir={}",
        args.signed_chain_report.display(),
        args.anchor_registry_dir.display()
    );

    // 1. Read raw on-disk bytes ONCE — these bytes are exactly
    //    what gets hashed AND what the wrapper is parsed from.
    //    No second `fs::read` of the same path.
    let raw_bytes = std::fs::read(&args.signed_chain_report).map_err(|e| {
        println!("event=integrity_evidence_anchor_verify_failed reason=io detail={e}");
        anyhow!(
            "read signed-chain-report {}: {e}",
            args.signed_chain_report.display()
        )
    })?;

    // 2. Parse the wrapper from the SAME raw_bytes buffer.
    let wrapper: SignedIntegrityEvidenceChainReport =
        serde_json::from_slice(&raw_bytes).map_err(|e| {
            let reason = "malformed_json";
            println!("event=integrity_evidence_anchor_verify_failed reason={reason} detail={e}");
            anyhow!(
                "parse signed-chain-report {}: {e}",
                args.signed_chain_report.display()
            )
        })?;

    // 3. Verify the wrapper signature under its embedded
    //    pubkey before binding the anchor.
    if let Err(e) =
        verify_signed_integrity_evidence_chain_report(&wrapper, &wrapper.signer_pubkey_hex)
    {
        let reason = "wrapper_signature_invalid";
        println!("event=integrity_evidence_anchor_verify_failed reason={reason} detail={e}");
        bail!(
            "Stage 12.25 wrapper signature did not verify under its embedded signer_pubkey_hex: {e}"
        );
    }

    // 4. Extract the wrapper signer pubkey — this is the
    //    same-key binding the registry-backed verify enforces.
    let expected_signer_pubkey =
        parse_anchor_hex_32(&wrapper.signer_pubkey_hex).map_err(|e| {
            println!(
                "event=integrity_evidence_anchor_verify_failed reason=signing detail={e}"
            );
            anyhow!("Stage 12.25 wrapper signer_pubkey_hex malformed: {e}")
        })?;

    let registry =
        LocalEvidenceAnchorRegistry::open(args.anchor_registry_dir.clone()).map_err(|e| {
            println!("event=integrity_evidence_anchor_verify_failed reason=io detail={e}");
            anyhow!(
                "open --anchor-registry-dir {}: {e}",
                args.anchor_registry_dir.display()
            )
        })?;
    let record = verify_anchor_against_registry(
        &registry,
        &raw_bytes,
        &expected_signer_pubkey,
        args.tx_id.as_deref(),
    )
    .map_err(|err| {
        let reason = evidence_anchor_reason_tag(&err);
        println!("event=integrity_evidence_anchor_verify_failed reason={reason} detail={err}");
        anyhow!("verify anchor refused: {err}")
    })?;

    print_verify_ok(&record);
    Ok(())
}

fn print_verify_ok(record: &AnchorRecord) {
    println!(
        "event=integrity_evidence_anchor_verify_ok artifact_hash_hex={} signer_pubkey_hex={} \
         tx_id={} local_status={} anchor_schema_version={} artifact_schema_version={} \
         artifact_kind={}",
        record.artifact_hash_hex,
        record.signer_pubkey_hex,
        record.receipt.tx_id,
        record.status.as_str(),
        record.tx_data.digest.anchor_schema_version,
        record.tx_data.digest.artifact_schema_version,
        record.tx_data.digest.artifact_kind.as_str(),
    );
}

// ── Verify file (standalone JSON) ─────────────────────────────────────────────

fn run_verify_file(args: VerifyAnchorFileArgs) -> Result<()> {
    println!(
        "event=integrity_evidence_anchor_verify_file_started signed_chain_report={} \
         anchor_json={}",
        args.signed_chain_report.display(),
        args.anchor_json.display()
    );

    // 1. Read raw on-disk bytes ONCE — these bytes are exactly
    //    what gets hashed AND what the wrapper is parsed from.
    let raw_bytes = std::fs::read(&args.signed_chain_report).map_err(|e| {
        println!("event=integrity_evidence_anchor_verify_file_failed reason=io detail={e}");
        anyhow!(
            "read signed-chain-report {}: {e}",
            args.signed_chain_report.display()
        )
    })?;

    // 2. Parse the wrapper from the SAME raw_bytes buffer.
    let wrapper: SignedIntegrityEvidenceChainReport =
        serde_json::from_slice(&raw_bytes).map_err(|e| {
            let reason = "malformed_json";
            println!(
                "event=integrity_evidence_anchor_verify_file_failed reason={reason} detail={e}"
            );
            anyhow!(
                "parse signed-chain-report {}: {e}",
                args.signed_chain_report.display()
            )
        })?;

    // 3. Verify the wrapper signature under its embedded
    //    pubkey before binding the anchor.
    if let Err(e) =
        verify_signed_integrity_evidence_chain_report(&wrapper, &wrapper.signer_pubkey_hex)
    {
        let reason = "wrapper_signature_invalid";
        println!("event=integrity_evidence_anchor_verify_file_failed reason={reason} detail={e}");
        bail!(
            "Stage 12.25 wrapper signature did not verify under its embedded signer_pubkey_hex: {e}"
        );
    }

    // 4. Extract the wrapper signer pubkey for same-key
    //    binding against the standalone anchor.
    let expected_signer_pubkey =
        parse_anchor_hex_32(&wrapper.signer_pubkey_hex).map_err(|e| {
            println!(
                "event=integrity_evidence_anchor_verify_file_failed reason=signing detail={e}"
            );
            anyhow!("Stage 12.25 wrapper signer_pubkey_hex malformed: {e}")
        })?;

    let anchor_bytes = std::fs::read(&args.anchor_json).map_err(|e| {
        println!("event=integrity_evidence_anchor_verify_file_failed reason=io detail={e}");
        anyhow!("read --anchor-json {}: {e}", args.anchor_json.display())
    })?;
    let tx_data: IntegrityEvidenceAnchorTxData =
        serde_json::from_slice(&anchor_bytes).map_err(|e| {
            println!(
                "event=integrity_evidence_anchor_verify_file_failed reason=malformed_json detail={e}"
            );
            anyhow!(
                "parse --anchor-json {}: {e}",
                args.anchor_json.display()
            )
        })?;

    verify_anchor_file_against_artifact_bytes(&tx_data, &raw_bytes, &expected_signer_pubkey)
        .map_err(|err| {
            let reason = evidence_anchor_reason_tag(&err);
            println!(
                "event=integrity_evidence_anchor_verify_file_failed reason={reason} detail={err}"
            );
            anyhow!("verify anchor file refused: {err}")
        })?;

    let artifact_hash_hex = anchor_hex_lower(&tx_data.digest.artifact_hash);
    let signer_pubkey_hex = anchor_hex_lower(&tx_data.digest.signer_pubkey);
    println!(
        "event=integrity_evidence_anchor_verify_file_ok artifact_hash_hex={} \
         signer_pubkey_hex={} anchor_schema_version={} artifact_schema_version={} \
         artifact_kind={}",
        artifact_hash_hex,
        signer_pubkey_hex,
        tx_data.digest.anchor_schema_version,
        tx_data.digest.artifact_schema_version,
        tx_data.digest.artifact_kind.as_str(),
    );
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[cfg(feature = "submit")]
fn wrapper_metadata(
    wrapper: &SignedIntegrityEvidenceChainReport,
) -> Result<VerifiedWrapperMetadata, EvidenceAnchorError> {
    let signer_pubkey = parse_anchor_hex_32(&wrapper.signer_pubkey_hex).map_err(|e| {
        EvidenceAnchorError::Signing(format!(
            "Stage 12.25 wrapper signer_pubkey_hex malformed: {e}"
        ))
    })?;
    let signed_at_utc_unix = parse_rfc3339_to_unix(&wrapper.signed_at_utc)?;
    Ok(VerifiedWrapperMetadata {
        artifact_schema_version: wrapper.schema_version,
        signer_pubkey,
        signed_at_utc_unix,
    })
}

#[cfg(feature = "submit")]
fn parse_rfc3339_to_unix(s: &str) -> Result<i64, EvidenceAnchorError> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.timestamp())
        .map_err(|e| EvidenceAnchorError::MalformedSignedAtUtc {
            raw: s.to_string(),
            reason: e.to_string(),
        })
}

#[cfg(feature = "submit")]
fn read_seed_file(path: &std::path::Path) -> Result<[u8; 32], EvidenceAnchorError> {
    let bytes = std::fs::read(path).map_err(|e| EvidenceAnchorError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    if bytes.len() != 32 {
        return Err(EvidenceAnchorError::MalformedSeedFile {
            path: path.to_path_buf(),
            reason: format!("expected 32 bytes, got {}", bytes.len()),
        });
    }
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&bytes);
    // Sanity: confirm the seed can derive a pubkey (Ed25519 has
    // no domain-bound rejection set; this is just a primitive
    // probe so we surface decode failures up-front).
    let _ = anchor_signer_pubkey_bytes(&seed)?;
    Ok(seed)
}

#[cfg(feature = "submit")]
fn write_anchor_tx_data_atomic(
    tx_data: &IntegrityEvidenceAnchorTxData,
    out: &std::path::Path,
) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(tx_data).context("serialize anchor tx_data JSON")?;
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create parent dir {}", parent.display()))?;
        }
    }
    let tmp = out.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).with_context(|| format!("write tmp {}", tmp.display()))?;
    std::fs::rename(&tmp, out)
        .with_context(|| format!("rename {} -> {}", tmp.display(), out.display()))?;
    Ok(())
}

// ── Compile-time guards ───────────────────────────────────────────────────────

const _: () = {
    // Force the schema version constant to be referenced from
    // this CLI surface so a future bump in the library shows up
    // as a compile-time visible reference here.
    let _v: u32 = INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION;
};

// Compile-time guard: keep LocalAnchorStatus reachable for
// downstream tooling; the variant is rendered via `as_str()` on
// the OK event.
const _: fn() = || {
    fn _accept<T>(_: T) {}
    _accept::<LocalAnchorStatus>(LocalAnchorStatus::Submitted);
};

#[cfg(test)]
mod reconcile_record_reason_tag_tests {
    //! Stage 13.2 — pin the reconcile per-record reason-tag
    //! taxonomy.
    //!
    //! These tests defend against a regression where every
    //! per-record failure was collapsed into `reason=chain_rpc`.
    //! Local registry / data errors (`io`,
    //! `anchor_not_found`, `malformed_json`) MUST surface their
    //! real closed-set tag; only chain-transport / chain-side
    //! refusals route through the typed classifier.

    use super::*;

    #[test]
    fn io_error_keeps_io_tag() {
        let err = EvidenceAnchorError::Io {
            path: std::path::PathBuf::from("/tmp/anchors/foo.json"),
            source: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "nope"),
        };
        assert_eq!(reconcile_record_reason_tag(&err), "io");
    }

    #[test]
    fn anchor_not_found_keeps_anchor_not_found_tag() {
        let err = EvidenceAnchorError::AnchorNotFound {
            selector: "tx_id=0xmissing".to_string(),
        };
        assert_eq!(reconcile_record_reason_tag(&err), "anchor_not_found");
    }

    #[test]
    fn malformed_json_keeps_malformed_json_tag() {
        let source = serde_json::from_str::<serde_json::Value>("not json")
            .expect_err("intentional parse failure");
        let err = EvidenceAnchorError::MalformedJson {
            path: std::path::PathBuf::from("/tmp/anchors/bogus.json"),
            source,
        };
        assert_eq!(reconcile_record_reason_tag(&err), "malformed_json");
    }

    #[test]
    fn artifact_hash_mismatch_keeps_artifact_hash_mismatch_tag() {
        let err = EvidenceAnchorError::ArtifactHashMismatch {
            recomputed_hex: "a".repeat(64),
            anchored_hex: "b".repeat(64),
        };
        assert_eq!(reconcile_record_reason_tag(&err), "artifact_hash_mismatch");
    }

    #[test]
    fn chain_client_transport_failure_maps_to_chain_rpc() {
        let err = EvidenceAnchorError::ChainClient(ChainClientError::Other(
            "HTTP transport failure: connection refused".to_string(),
        ));
        assert_eq!(reconcile_record_reason_tag(&err), "chain_rpc");
    }

    #[test]
    fn chain_client_jsonrpc_error_maps_to_chain_submit_refused() {
        let err = EvidenceAnchorError::ChainClient(ChainClientError::Other(
            "JSON-RPC error: {\"code\":-32601,\"message\":\"method not found\"}"
                .to_string(),
        ));
        assert_eq!(reconcile_record_reason_tag(&err), "chain_submit_refused");
    }

    #[test]
    fn chain_client_adapter_malformed_status_maps_to_chain_response_malformed() {
        let err = EvidenceAnchorError::ChainClient(ChainClientError::Other(
            "malformed sum_getIntegrityEvidenceAnchorStatus response: missing field"
                .to_string(),
        ));
        assert_eq!(
            reconcile_record_reason_tag(&err),
            "chain_response_malformed"
        );
    }

    #[test]
    fn chain_client_adapter_not_activated_maps_to_not_activated() {
        let err = EvidenceAnchorError::ChainClient(ChainClientError::Other(
            "integrity_evidence_anchor not activated".to_string(),
        ));
        assert_eq!(reconcile_record_reason_tag(&err), "not_activated");
    }

    #[test]
    fn chain_client_unknown_string_falls_back_to_chain_rpc() {
        let err = EvidenceAnchorError::ChainClient(ChainClientError::Other(
            "totally unrecognized text".to_string(),
        ));
        assert_eq!(reconcile_record_reason_tag(&err), "chain_rpc");
    }
}
