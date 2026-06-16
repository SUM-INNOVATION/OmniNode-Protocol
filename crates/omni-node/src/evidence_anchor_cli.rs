//! Phase 5 Stage 13.0 — chain-anchor CLI surface for Stage 12.25
//! signed-chain-report artifacts.
//!
//! Lives on `omni-node operator …` as four subcommands:
//!
//! - `submit-integrity-evidence-anchor` (gated `--features submit`)
//! - `query-integrity-evidence-anchor`
//! - `verify-integrity-evidence-anchor`         — registry-backed
//! - `verify-integrity-evidence-anchor-file`    — standalone JSON
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
#[cfg(feature = "submit")]
use omni_zkml::EvidenceAnchorError;
use omni_zkml::{
    AnchorRecord, AnchorSelector, AnchorStatus, INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
    IntegrityEvidenceAnchorTxData, LocalAnchorStatus, LocalEvidenceAnchorRegistry,
    anchor_hex_lower, evidence_anchor_reason_tag, parse_anchor_hex_32,
    query_evidence_anchor_workflow, verify_anchor_against_registry,
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
    /// Anchor a Stage 12.25 `SignedIntegrityEvidenceChainReport`
    /// to SUM Chain. Stage 13.0 uses a stub chain client; Stage
    /// 13.1 will swap in the real adapter. Gated behind
    /// `--features submit`.
    #[cfg(feature = "submit")]
    SubmitIntegrityEvidenceAnchor(SubmitAnchorArgs),
    /// Query the registry-stored status of an anchor by
    /// `--artifact-hash-hex` or `--tx-id`.
    QueryIntegrityEvidenceAnchor(QueryAnchorArgs),
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

    let client = omni_zkml::StubEvidenceAnchorChainClient::new();
    let digest = build_anchor_digest(&metadata, &raw_bytes);

    let record = submit_evidence_anchor_workflow(&registry, &client, digest, &submitter_seed)
        .map_err(|err| {
            let reason = evidence_anchor_reason_tag(&err);
            println!("event=integrity_evidence_anchor_submit_failed reason={reason} detail={err}");
            anyhow!("submit anchor refused: {err}")
        })?;

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
    let client = omni_zkml::StubEvidenceAnchorChainClient::new();
    let outcome = query_evidence_anchor_workflow(
        &registry,
        &client,
        match &selector_owned {
            SelectorOwned::ArtifactHashHex(h) => AnchorSelector::ArtifactHashHex(h),
            SelectorOwned::TxId(t) => AnchorSelector::TxId(t),
        },
    )
    .map_err(|err| {
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
