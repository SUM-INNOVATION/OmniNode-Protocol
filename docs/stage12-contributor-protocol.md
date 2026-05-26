# Stage 12.0 — Contributor Inference Node protocol

**Status**: shipped at Stage 12.0. Off-chain only. `AttestationOnly` evidence mode only.

This document is the engineering reference for the `ContributorJob` / `ContributorResult` schemas, the canonical-bytes encoding, hashing/signing, and the base-unit accounting placeholders that a future payment engine consumes. It is intentionally minimal — runtime semantics live in the executable tests under `crates/omni-contributor/tests/`.

## Posture

- **No chain wire / Stage 7b tx / SUM Chain RPC / `InferenceAttestationDigest` change.**
- **No on-chain proof verification.** Stage 11d.3 reframe preserved: the chain remains neutral; proof acceptance is a local verifier policy decision. `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` and `MAINNET_APPROVED_PROOF_SYSTEMS` stay empty.
- **SNIP replaces the legacy decentralized-storage term everywhere** in the new code/docs. The `posture_preserved` test grep-guards this.
- **Default `omni-node` build pulls zero halo2/prover/framework runtimes.** CI's `default-tree-check` re-asserts after `omni-contributor` enters the default graph.

## Schemas (v1, frozen)

See `crates/omni-contributor/src/job.rs` and `crates/omni-contributor/src/result.rs`. All hash / pubkey / signature fields are lowercase hex of raw bytes (no `0x` prefix); 64 chars for BLAKE3 + Ed25519 pubkeys; 128 chars for Ed25519 signatures. SNIP V2 IDs are `0x`-prefixed lowercase hex (66 chars total, mirroring `omni_types::phase5::SnipV2ObjectId`).

### `ContributorJob`

```text
ContributorJob {
    schema_version: 1                            // pinned
    job_id:                  hex64               // = lowercase_hex(job_hash)
    model_hash:              hex64
    manifest_snip_root:      0xhex66
    input_snip_root:         0xhex66
    input_hash:              hex64                // BLAKE3 of input bytes
    verification_requirement: AttestationOnly    // closed enum; v1 = 1 variant
    accounting:              JobAccounting
    dispatched_at_utc:       RFC3339 UTC
    expires_at_utc:          Option<RFC3339 UTC>
    dispatcher_pubkey_hex:   Option<hex64>        // both Some or both None
    dispatcher_signature_hex:Option<hex128>
    notes:                   Option<String>
}

JobAccounting {
    tokenizer_hash:          hex64                // BLAKE3 of canonical tokenizer
    tokenizer_id:            String (non-empty)   // e.g. "tiktoken/cl100k_base"
    input_token_count:       u64
    max_output_token_count:  u64
    base_unit_reward_policy: Unspecified          // closed enum; v1 = 1 variant
}
```

### `ContributorResult`

```text
ContributorResult {
    schema_version: 1
    job_id:                    hex64
    job_hash:                  hex64
    job_snip_root:             Option<0xhex66>    // convenience; verifier requires --job
    model_hash:                hex64
    input_hash:                hex64
    response_snip_root:        0xhex66
    response_hash:             hex64
    evidence:                  AttestationOnly    // closed enum; v1 = 1 variant
    measured_accounting:       MeasuredAccounting
    produced_at_utc:           RFC3339 UTC
    contributor_pubkey_hex:    hex64
    contributor_signature_hex: hex128
    notes:                     Option<String>
}

MeasuredAccounting {
    tokenizer_hash:       hex64                   // == job.accounting.tokenizer_hash
    input_token_count:    u64                     // == job.accounting.input_token_count
    output_token_count:   u64                     // ≤ job.accounting.max_output_token_count
    total_base_units:     u64                     // == input + output
    stage_contributions:  Vec<StageContribution>  // non-empty
}

StageContribution {
    contributor_pubkey_hex: hex64
    stage_label:            String
    work_unit_kind:         { Tokens | PrefillTokens | DecodeTokens | Layers | FlopsEstimate }
    work_units:             u64
}
```

## Canonical bytes / hashes / signatures

Encoding is bincode 1.3 (via the `bincode1` crate alias). Domain-separated to keep contributor IDs disjoint from chain-wire bytes:

```text
JOB_DOMAIN    = b"OMNINODE-CONTRIBUTOR-JOB:v1:"
RESULT_DOMAIN = b"OMNINODE-CONTRIBUTOR-RESULT:v1:"
```

### `job_hash`

```text
body_bytes  = bincode1::serialize(JobCanonicalBody)
              // JobCanonicalBody excludes `job_id` and `dispatcher_signature_hex`.
canonical   = JOB_DOMAIN || body_bytes
hash_bytes  = BLAKE3(canonical)                  // [u8; 32]
job_hash    = lowercase_hex(hash_bytes)          // 64-char string
job_id      = job_hash                           // verifier asserts equality
```

Dispatcher signature input: same `canonical` bytes (Ed25519 hashes internally).

### Contributor signature

```text
body_bytes               = bincode1::serialize(ResultCanonicalBody)
                           // excludes `contributor_signature_hex`.
contributor_signing_input = RESULT_DOMAIN || body_bytes
contributor_signature     = Ed25519_sign(seed, contributor_signing_input)
```

Field declaration order in both `JobCanonicalBody` and `ResultCanonicalBody` is the bincode wire order; frozen for `schema_version: 1`. Reorders are a `schema_version: 2` migration.

## Verification pipeline

`omni_contributor::verify::verify_result` (10 ordered steps; see `verify.rs`). The verifier:

1. Validates schemas on both job and result.
2. Recomputes `job_hash`; asserts `result.job_hash == recomputed` and `job_id == lowercase_hex(recomputed)`.
3. Checks `expires_at_utc` is not past `now()`.
4. Verifies dispatcher signature if present; emits `not_signed` otherwise.
5. Asserts `result.model_hash == job.model_hash` and `result.input_hash == job.input_hash`.
6. Fetches `input_snip_root` from SNIP; BLAKE3-checks bytes against `job.input_hash`.
7. Fetches `response_snip_root` from SNIP; BLAKE3-checks bytes against `result.response_hash`.
8. Checks accounting:
    - `tokenizer_hash` matches.
    - `input_token_count` matches.
    - `output_token_count ≤ max_output_token_count`.
    - `total_base_units == input + output`.
    - `stage_contributions` is non-empty.
9. Reconstructs contributor signing input; Ed25519-verifies.
10. Evidence-mode check + requirement satisfaction (trivial for v1: `AttestationOnly` satisfied by signature).

No chain authority is consulted at any step.

## Base-unit accounting (protocol axiom)

```text
1 base unit = 1 token.
total_base_units = input_token_count + output_token_count.
```

A 200k-input / 1M-output job is 1.2M base units. **Stage 12.0 records and verifies these numbers but does NOT compute, distribute, or settle any reward.** A future Stage 12.x payment engine consumes:

- `job.accounting.base_unit_reward_policy` (currently `Unspecified` only),
- `result.measured_accounting.total_base_units`,
- `result.measured_accounting.stage_contributions[*].(contributor_pubkey_hex, work_unit_kind, work_units)`,

to derive per-contributor amounts. Multi-contributor B/C/D splits emit one `stage_contributions` entry per participant.

## SNIP integration (Option A)

`omni-contributor` does NOT extend `omni-store` or invent any new SNIP wire. `publish_bytes` / `fetch_bytes` are tempfile-backed wrappers around the existing `omni_store::SnipV2Adapter::ingest_public(&Path)` / `download_public(&SnipV2ObjectId, &Path)` methods. Lifecycle is checked via the existing `omni_store::snip_v2::check_lifecycle(_, allow_non_active: false)`.

## Reserved future strings (NOT in v1 code)

The following names are **reserved** for a future `schema_version: 2` migration but are NOT present in any Rust enum at Stage 12.0:

- `VerificationRequirement::Stage11dProductionFixedPointMlpProof` — would require a contributor-side prover (Stage 11d.2 + a new `omni-node` feature; out of scope for 12.0).
- `Evidence::Stage11dProductionFixedPointMlpProof { proof_artifact_snip_root, proof_artifact_hash }` — matching evidence variant.
- Additional `WorkUnitKind` variants beyond the v1 five.
- Additional `BaseUnitRewardPolicy` variants (the future payment-engine input).

A v1 deserializer that encounters any of these in a JSON payload returns a closed-enum parse error (`deny_unknown_fields`).

## CLI

```text
omni-node operator contributor validate-job  --job <path-or-snip-root>
omni-node operator contributor run-job       --job <path> --out <result-path>
                                              --runner external|stub
                                              [--external-command <bin> | --stub-response <path>]
                                              --seed-file <32-byte raw seed>
omni-node operator contributor verify-result --job <path> --result <path>
```

`run-job` accepts only the runner-specific flags it can satisfy. `verify-result` ALWAYS requires both `--job` and `--result` explicitly — the optional `job_snip_root` field on the result is convenience-only and is NOT implicitly trusted.

## Contributor seed

Each contributor holds a **32-byte raw seed file**, distinct from any chain-attestation seed used by Stage 6's `sign_chain_attestation_digest`. Different role; less blast radius if compromised.

## Out of scope (Stage 12.0)

Items 1, 7, 8 of the numbered chart flow (prompt/service fee, human feedback, financial RLHF/slashing). The entire lettered (developer/training/federated/aggregate/reward) flow. Network-level job dispatch (gossip topic / posted-jobs board) — Stage 12.0 ships manual-CLI handoff only. Multi-contributor pipeline orchestration runtime (the schema supports B/C/D records; the runtime is Stage 12.2+).

---

# Stage 12.1 — filesystem-based job discovery

**Status**: shipped at Stage 12.1. Off-chain only. Extends Stage 12.0 with two new envelopes, one new `JobSource` impl, a long-running `watch-jobs` subcommand, and required cost guardrails. No chain wire / payment / proof changes.

## Posture (unchanged from 12.0)

No chain wire / Stage 7b tx / SUM Chain RPC / `InferenceAttestationDigest` changes. `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` stays empty. AttestationOnly only. No GGUF correctness claim. Default `omni-node` build pulls zero halo2/prover/framework runtimes (CI re-asserts).

## Why filesystem-only discovery in 12.1

SNIP V2 roots are **content-addressed and immutable**. A dispatcher cannot "append to the index by re-publishing under the same root" — re-publishing produces a *different* root. A contributor polling one fixed SNIP root would keep fetching the same snapshot forever. Real mutable discovery (libp2p gossip / local head pointer / off-chain registry) is Stage 12.2+ work; Stage 12.1 ships filesystem-watch instead so the contributor-job lifecycle works end-to-end without inventing a new wire surface.

What still uses SNIP in 12.1:
- The `ContributorJob` JSON itself (dispatcher publishes; contributor fetches by `PostedJob.job_snip_root`).
- The `ContributorResult` JSON the contributor produces.
- The `PostedResultLink` envelope (optional, on `--publish-result-link`).

## Schemas

### `PostedJob`

```text
PostedJob {
    schema_version: 1                            // pinned
    posted_id:               hex64               // = lowercase_hex(BLAKE3(POSTED_JOB_DOMAIN || canonical_body))
    job_snip_root:           0xhex66             // SNIP root of the ContributorJob JSON
    job_hash:                hex64               // drift guard; verifier recomputes from fetched job
    model_hash:              hex64               // copy for cheap pre-fetch filtering
    posted_at_utc:           RFC3339 UTC (Z suffix)
    expires_at_utc:          Option<RFC3339 UTC>
    poster_pubkey_hex:       Option<hex64>       // both Some or both None
    poster_signature_hex:    Option<hex128>      // signs canonical body (excludes posted_id + signature)
    notes:                   Option<String>      // part of canonical signing input
}
```

### `PostedResultLink`

```text
PostedResultLink {
    schema_version: 1
    posted_id:                 hex64             // copy of the PostedJob.posted_id
    result_snip_root:          0xhex66           // SNIP root of the ContributorResult JSON
    result_canonical_hash:     hex64             // BLAKE3(canonical_result_bytes(result))
                                                 // — the signature-domain hash. Distinct
                                                 // from BLAKE3 of the on-disk JSON.
    contributor_pubkey_hex:    hex64
    contributor_signature_hex: hex128            // signs canonical body (excludes signature)
    published_at_utc:          RFC3339 UTC (Z suffix)
}
```

## Canonical bytes (v1, frozen)

Two new domain separators:

```text
POSTED_JOB_DOMAIN    = b"OMNINODE-CONTRIBUTOR-POSTED-JOB:v1:"        (35 bytes)
POSTED_RESULT_DOMAIN = b"OMNINODE-CONTRIBUTOR-POSTED-RESULT-LINK:v1:" (43 bytes)
```

Both distinct from Stage 12.0's `JOB_DOMAIN` / `RESULT_DOMAIN` and from any chain-wire tag. Same bincode 1.3 encoding regime; field declaration order frozen for `schema_version: 1`.

## Discovery — `FilesystemSource`

A `JobSource` impl that walks a directory for `*.json` files, parses each as a `PostedJob`, validates schema, and recomputes `posted_id` from canonical bytes (refusing drift). Stateless: dedup-across-polls happens in the watch loop's in-memory `HashSet<posted_id>`, NOT in the source. No persistent dedup state across restarts; rely on `expires_at_utc` + `--max-jobs` to bound work.

Non-JSON files are silently skipped. Per-file errors (bad JSON, schema, posted_id drift) come back as `Err(...)` items in the source's per-entry result vector so the watch loop can log + skip + continue. Source-level errors (directory unreadable on first poll) propagate as `ContributorError::Discover(...)` and cause the loop to exit.

## `watch-jobs` pipeline

```text
poll FilesystemSource
  → for each new PostedJob (not in seen-set):
      1. Validate schema.
      2. Verify poster signature (if present).
      3. Refuse if expired.
      4. Apply --accept-model-hash / --accept-tokenizer-hash allow-lists.
      5. Apply cost caps (--max-input-tokens / --max-output-tokens
         / --max-total-base-units) against the job's declared bounds.
      6. Fetch ContributorJob from SNIP via PostedJob.job_snip_root.
      7. Drift guards: recompute job_hash from canonical bytes;
         verify result.model_hash matches PostedJob.model_hash.
      8. Invoke 12.0 run_job (which itself validates job consistency
         + dispatcher signature + then fetches + runs + signs).
      9. Call verify_result(&job, &result, &adapter). On
         overall_ok=false, write <result-out-dir>/<job_id>.rejected.json
         (audit trail) and SKIP the link-publish step.
     10. Otherwise write <result-out-dir>/<job_id>.json.
     11. If --publish-result-link: publish the result JSON to SNIP,
         sign + publish a PostedResultLink.
```

Per-job failures (bad signature, expired, cost-cap exceeded, drift) emit a `WatchEvent::Skip { reason }` and continue. The loop never short-circuits because one file was bad. Source-level errors (directory disappearing mid-loop) propagate.

## Required cost caps

`watch-jobs` requires three CLI flags with no defaults:

```text
--max-input-tokens        <u64>    # refuse if job.accounting.input_token_count > this
--max-output-tokens       <u64>    # refuse if job.accounting.max_output_token_count > this
--max-total-base-units    <u64>    # refuse if input + max_output > this (overflow-safe)
```

A conservative default that fits a dev box is dangerous for a production contributor and vice-versa; the operator must make an explicit cap decision before any pickup happens. Caps are applied **after fetching the job from SNIP** but **before invoking the runner** — a runner is never invoked for a job that would exceed any cap.

## Post-run `verify_result`

The watch loop invokes the full Stage 12.0 `verify_result` pipeline on the freshly-built result, against the same SNIP adapter the runner used. A failure writes `<result-out-dir>/<job_id>.rejected.json` (containing the result body + a structured `rejected_reason`) and skips the optional link-publish step. This catches a category of integration bug (orchestrator/runner schema drift, future verifier-side check the orchestrator forgot) before another party's verifier sees the result.

## Stage 12.1 CLI

```text
omni-node operator contributor post-job \
    --job <path> --posted-out <path> \
    [--seed-file <path>] [--expires-at-utc <RFC3339-Z>] [--notes <string>]
    [--snip-binary <bin>] [--snip-seed <path>]

omni-node operator contributor watch-jobs \
    --source fs --jobs-dir <path> \
    --max-input-tokens <N> --max-output-tokens <N> --max-total-base-units <N> \
    --runner external|stub \
    [--external-command <bin>] [--external-arg ...] [--external-env-allow ...] \
    [--stub-response <path>] [--stub-input-tokens N] [--stub-output-tokens N] \
    --seed-file <path> --result-out-dir <path> \
    [--accept-model-hash <hex64> ...] [--accept-tokenizer-hash <hex64> ...] \
    [--max-jobs <N>] [--max-polls <N>] [--poll-interval-secs <N>] \
    [--publish-result-link] [--snip-binary <bin>] [--snip-seed <path>]

omni-node operator contributor publish-result-link \
    --result <path> --posted-job <path> --link-out <path> --seed-file <path>
    [--snip-binary <bin>] [--snip-seed <path>]
```

`--source` is `fs` only in 12.1. `--max-polls` is primarily for tests + smoke runs; production typically omits it and lets the loop run until `--max-jobs` is reached or the surrounding process supervisor sends SIGTERM.

## Out of scope (Stage 12.1)

- SNIP index polling and `PostedJobsIndex` aggregator. Deferred to Stage 12.2+ when a real mutable discovery surface is designed.
- Libp2p gossip / pubsub discovery.
- Lease / claim primitives. Two contributors polling the same directory will both pick the same job and both publish results; dispatchers reading results choose which to trust.
- Persistent dedup state across `watch-jobs` restarts.
- Multi-dispatcher reconciliation.
- Anything in items 1 / 7 / 8 or the lettered A–F flow.
- Any chain-side interaction.
