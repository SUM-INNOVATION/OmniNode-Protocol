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

---

# Stage 12.2 — contributor mesh network relay

**Status**: shipped at Stage 12.2. Extends Stages 12.0 + 12.1 with the first real network surface: signed gossipsub announcements of SNIP-pointed posted envelopes. Filesystem-based discovery (12.1) and network-based discovery (12.2) are parallel paths — operators can run either or both via separate subcommands.

## Posture (unchanged)

No chain wire / Stage 7b tx / SUM Chain RPC / `InferenceAttestationDigest` change. No payment / reward / staking / slashing. AttestationOnly only. No on-chain verification, no chain proof allowlist. SNIP only. Default `omni-node` build still pulls zero halo2/prover/framework runtimes (CI re-asserts).

## What the mesh carries

**Pointers only.** Announcements are tiny signed envelopes carrying a SNIP V2 root + drift-guard hashes + sender identity. They never carry job bodies, input bytes, model bytes, or result bytes. SNIP remains the content store; the gossip mesh only tells subscribers what's worth fetching.

Two signature layers protect each path:
1. **Announcer signature** (required) on the network envelope — anti-spam + provenance for who introduced this pointer.
2. **Inner envelope signatures** — the existing 12.0 / 12.1 validators (`PostedJob.poster_signature_hex` optional; `PostedResultLink.contributor_signature_hex` required) run unchanged after fetching from SNIP.

A receiver discarding the announcer signature is not the same as discarding the inner envelope. Anyone may legitimately relay an announcement they observed; the receiver still validates the inner signed envelope semantically.

## Stage 12.2-pre dependency

This stage builds on `omni-net` topic constants added in [PR #19 / Stage 12.2-pre](#) — `TOPIC_CONTRIBUTOR_JOB = "omni/contributor/job/v1"` and `TOPIC_CONTRIBUTOR_RESULT = "omni/contributor/result/v1"` — plus the non-blocking `OmniNet::try_next_event` accessor and the typed-error refusal of unknown topics. Without the pre-PR, `OmniNet::publish` on a contributor topic would silently route to `TOPIC_TEST`; this stage explicitly depends on that having been fixed.

## Schemas

```text
NetworkPostedJobAnnouncement {
    schema_version:           1                        // pinned
    posted_job_snip_root:     0xhex66                  // SNIP root of PostedJob JSON
    posted_id:                hex64                    // copy of PostedJob.posted_id
    job_hash:                 hex64                    // copy of PostedJob.job_hash
    model_hash:               hex64                    // copy; enables pre-fetch filter
    tokenizer_hash:           Option<hex64>            // optional; advisory
    announced_at_utc:         RFC3339 UTC (Z suffix)
    announcer_pubkey_hex:     hex64                    // REQUIRED
    announcer_signature_hex:  hex128                   // REQUIRED; over canonical body
}

NetworkPostedResultAnnouncement {
    schema_version: 1
    posted_id:                       hex64
    posted_result_link_snip_root:    0xhex66
    result_canonical_hash:           hex64             // copy of PostedResultLink field
    contributor_pubkey_hex:          hex64             // copy of PostedResultLink field
    announced_at_utc:                RFC3339 UTC (Z suffix)
    announcer_pubkey_hex:            hex64             // REQUIRED
    announcer_signature_hex:         hex128            // REQUIRED
}
```

Both have `deny_unknown_fields`. Announcer signature is **required** on both — different from `PostedJob.poster_signature_hex` (which is legitimately optional for local-CLI handoffs in 12.1).

## Canonical bytes (v1, frozen)

Two new domain separators:

```text
NET_JOB_DOMAIN    = b"OMNINODE-CONTRIBUTOR-NET-JOB:v1:"      (32 bytes)
NET_RESULT_DOMAIN = b"OMNINODE-CONTRIBUTOR-NET-RESULT:v1:"   (35 bytes)
```

Both distinct from the 12.0 / 12.1 separators and from any chain-wire tag. Bincode 1.3 encoding; canonical body excludes `announcer_signature_hex` (the signer can't include its own signature) but **includes** `announcer_pubkey_hex` so the signature binds to the claimed pubkey.

## Relay abstraction

`omni_contributor::ContributorRelay` is the sync, transport-agnostic interface the watch and announce paths use:

```text
publish_job(&NetworkPostedJobAnnouncement)    -> Result<()>
publish_result(&NetworkPostedResultAnnouncement) -> Result<()>
poll_jobs()    -> Result<Vec<NetworkPostedJobAnnouncement>>
poll_results() -> Result<Vec<NetworkPostedResultAnnouncement>>
```

Two impls:

- **`InMemoryRelay`** — vec-backed in-process queues. Fully sync; no tokio runtime. **All Stage 12.2 tests use this.** The full schema + signature + drift + cost-cap + dedup pipeline is exercised end-to-end without any real networking.
- **`OmniNetRelay`** — production adapter behind the `network` feature flag on `omni-contributor`. Wraps `omni_net::OmniNet`, drains events via Stage 12.2-pre's `try_next_event`, and bridges sync `publish_*` to async `OmniNet::publish` via `tokio::task::block_in_place(|| handle.block_on(...))`. Per-topic dispatch routes incoming `MessageReceived` events into a job-queue or result-queue based on `topic` string, so `poll_jobs` and `poll_results` don't lose each other's messages.

## NetworkSource

`NetworkSource` is a `JobSource` impl (the same trait `FilesystemSource` implements in 12.1) that drains job announcements from a relay. The validation pipeline per announcement:

1. Schema validate.
2. Verify announcer signature against canonical body bytes.
3. Fetch `PostedJob` JSON from SNIP at `posted_job_snip_root`.
4. Parse + schema-validate the fetched `PostedJob`.
5. Recompute `posted_id` from canonical bytes; assert three-way agreement (announcement / fetched envelope / recomputed).
6. Assert `job_hash` and `model_hash` agree between announcement and fetched envelope.

Per-entry failures surface as `Err(DiscoverError::…)` items the watch loop logs + skips; the loop never short-circuits on one bad announcement. The result of step 6 plugs straight into the existing Stage 12.1 `run_watch_loop` — cost caps, poster-signature check, runner invocation, post-run `verify_result`, accepted/rejected write-out, optional `PostedResultLink` publish are all unchanged. Stage 12.2 only swaps the discovery source.

## `watch-network-jobs` pipeline

Same as 12.1 `watch-jobs` with one source-of-discovery change. Required cost caps remain mandatory (`--max-input-tokens` / `--max-output-tokens` / `--max-total-base-units`). If `--publish-result-link` is set, the watch loop additionally builds + signs + broadcasts a `NetworkPostedResultAnnouncement` for each successfully verified result.

## `watch-network-results` pipeline

A separate, simpler loop (no inner job to fetch, no runner, no `verify_result`):

1. Drain result announcements from the relay.
2. Schema-validate; verify announcer signature.
3. If `--posted-id` filter is non-empty, apply it.
4. Fetch `PostedResultLink` from SNIP; parse; schema-validate.
5. Drift guard: announcement `posted_id` / `result_canonical_hash` / `contributor_pubkey_hex` must agree with fetched link.
6. Write the link bytes to `<result-out-dir>/<posted_id>.link.json` — **byte-identical to what was on SNIP** (no re-serialization).

This stage **does not settle payment, write results, or invoke any verifier on the result content**. Downstream consumers of result links are out of scope (Stage 12.x+).

## Stage 12.2 CLI

```text
omni-node operator contributor announce-job \
    --posted-job <path> --seed-file <path>
    [--include-tokenizer-hash]                    # fetch inner ContributorJob to populate it
    [--listen-port <u16>] [--peer <multiaddr> ...]
    [--propagation-wait-ms <N>]
    [--snip-binary <bin>] [--snip-seed <path>]

omni-node operator contributor watch-network-jobs \
    --max-input-tokens <N> --max-output-tokens <N> --max-total-base-units <N>   # REQUIRED
    --runner external|stub [runner-specific flags]
    --seed-file <path> --result-out-dir <path>
    [--accept-model-hash <hex64> ...] [--accept-tokenizer-hash <hex64> ...]
    [--max-jobs <N>] [--max-polls <N>] [--poll-interval-secs <N>]
    [--publish-result-link]
    [--listen-port <u16>] [--peer <multiaddr> ...]
    [--snip-binary <bin>] [--snip-seed <path>]

omni-node operator contributor announce-result \
    --posted-result-link <path> --seed-file <path>
    [--listen-port <u16>] [--peer <multiaddr> ...]
    [--propagation-wait-ms <N>]

omni-node operator contributor watch-network-results \
    [--posted-id <hex64> ...]                     # repeat; empty = accept any
    --result-out-dir <path>
    [--listen-port <u16>] [--peer <multiaddr> ...]
    [--max-results <N>] [--max-polls <N>] [--poll-interval-secs <N>]
    [--snip-binary <bin>] [--snip-seed <path>]
```

`--listen-port` maps to `NetConfig.listen_port`; arbitrary `--listen <multiaddr>` is Stage 12.3+. `--peer <multiaddr>` repeatable maps to `NetConfig.bootstrap_peers`.

## Out of scope (Stage 12.2)

- Persistent job / result dedup state across restarts.
- Lease / claim primitives; double-pickup prevention. Multiple watchers will independently pick the same announcement and produce results; dispatchers reading results choose which to trust.
- Result-content verifier on the receiver side of `watch-network-results` (it only fetches + writes the link envelope; consuming the linked result is a follow-up stage).
- Arbitrary `--listen <multiaddr>` flexibility.
- `PostedJobsIndex` aggregator / signed multi-publisher index. Either out of scope or requires a separate design.
- Reputation / abuse-mitigation beyond the required announcer signature.
- Real-network CI test (would need bootstrap infrastructure).
- Anything in items 1 / 7 / 8 of the chart; the lettered A–F flow; any chain-side interaction.


---

# Stage 12.3 — multi-contributor pooled-memory sessions

**Status**: shipped at Stage 12.3. Extends Stage 12.2's mesh with a signed coordination shell for executing one posted job across multiple contributor devices.

## Posture (verbatim, unchanged)

No chain wire, no Stage 7b tx, no SUM Chain RPC, no `InferenceAttestationDigest`, no payment/reward/staking/slashing, no marketplace/auction/pricing/bid logic, no exclusive claim or lease model, no proof mode, no on-chain verification, no chain proof allowlist, no A–F flow. SNIP only. Default `omni-node` tree still pulls zero halo2/prover/framework runtimes (CI re-asserts).

The coordinator is a **process role**, not a chain authority. Contributors decide locally whether to join; verifiers decide locally whether to trust an aggregate. Assignments are cooperation hints within a session — multiple coordinators can run parallel sessions for the same `posted_id` with overlapping work.

## What 12.3 actually is

**Stage 12.3 is a coordination shell, not live pooled RAM.** It lets multiple machines participate in one inference job, record each participant's contribution under their own signature, and produce one aggregated `ContributorResult`. Inter-stage artifact handoff happens through SNIP — contributor *i+1* fetches contributor *i*'s `partial_artifact_snip_root`. Actual low-latency shared tensor / activation transport between participants is **Stage 12.4**.

This means 12.3 is useful for two shapes today:

1. **1-of-N pipeline through SNIP** — practical for any workload where activation size + SNIP round-trip latency are acceptable. The schemas + verifier are the load-bearing surface; the operator wires real cross-stage I/O.
2. **1-of-1 signed bookkeeping** — a single contributor running the whole job under a session envelope. Useful as a building block for future N-of-N workloads and as the happy-path integration test.

## Accounting rule (important)

The final `ContributorResult.measured_accounting.total_base_units` is the **job-level** input + output token count, exactly as in 12.0. It is **NOT** a sum of partial totals — that would scale incorrectly with pipeline depth (e.g. 3 contributors cooperating on one 200k-input / 1M-output task would inflate to 3.6M base units, which is wrong; the job is still 1.2M base units total).

Partials carry their own `measured_accounting.stage_contributions` for the contributor's slice of work. Those `work_units` exist to drive a future reward-split policy that divides the job-level base-unit pool — they are NOT summed into the final total. The verifier checks each partial's accounting is structurally valid (exactly one `stage_contribution`, matching pubkey) but does not require numerical equality with the final.

## Roles

- **Coordinator** — signs `ExecutionSession`, `WorkAssignment`s, and `AggregatedContributorResult`. Holds a key distinct from contributor/dispatcher seeds (new `CoordinatorSigner` role wrapper). Does not need to run inference.
- **Contributor** — signs `ContributorJoin` and `PartialContributorResult`. Can be one of many in a session.
- **Coordinator-as-contributor** — same person can hold both keys (or use the same seed under different role wrappers); the protocol does not care.

## Inner envelopes (5)

All pinned to `schema_version: 1`. Canonical bytes use bincode 1.3 with per-envelope domain separators. Signer's own signature excluded from the canonical body; `session_id` / `assignment_id` are BLAKE3 of the canonical bytes (mirrors 12.1's `posted_id` derivation).

```text
ExecutionSession {
    schema_version, session_id, posted_id, job_hash, model_hash,
    tokenizer_hash?, coordinator_pubkey_hex,
    created_at_utc, expires_at_utc,        // required (sessions are bounded)
    coordinator_signature_hex,
}

ContributorJoin {
    schema_version, session_id, contributor_pubkey_hex,
    available_ram_bytes, max_input_tokens, max_output_tokens,
    supported_work_unit_kinds: [WorkUnitKind],   // non-empty
    runner_kind,                                  // non-empty printable ASCII, ≤ 64
    joined_at_utc,
    contributor_signature_hex,
}

WorkAssignment {
    schema_version, session_id, assignment_id, stage_index,
    contributor_pubkey_hex,        // must be a joined contributor
    work_kind: WorkKind,
    expected_work_units,           // > 0
    expected_work_unit_kind: WorkUnitKind,
    assigned_at_utc,
    coordinator_signature_hex,     // verified against session.coordinator_pubkey_hex
}

PartialContributorResult {
    schema_version, session_id, assignment_id, contributor_pubkey_hex,
    partial_artifact_snip_root, partial_artifact_hash,
    measured_accounting: MeasuredAccounting,  // exactly one stage_contribution
    produced_at_utc,
    contributor_signature_hex,
}

AggregatedContributorResult {
    schema_version, session_id, posted_id,
    final_result_snip_root,         // points at a standalone v1 ContributorResult
    final_result_canonical_hash,    // BLAKE3 of canonical_result_bytes(final)
    partial_refs: [AggregatedPartialRef],   // non-empty; one per assignment
    aggregated_at_utc,
    coordinator_pubkey_hex,         // must equal session.coordinator_pubkey_hex
    coordinator_signature_hex,
}
```

**`WorkAssignment` deliberately does NOT carry `coordinator_pubkey_hex`** — the coordinator is identified by the session, and assignment signatures are verified against `session.coordinator_pubkey_hex`. This avoids drift between an assignment's coordinator claim and the session's.

## `WorkKind`

```rust
enum WorkKind {
    Prefill,
    Decode,
    Layers { start: u32, end: u32 },   // half-open; start < end
    Shard { index: u32 },
    Custom { label: String },          // non-empty printable ASCII, ≤ 64
}
```

`Custom` is intentional: at v1 freezing we cannot enumerate every real-world split strategy. Labels like `Custom("kv-cache-shard")`, `Custom("expert-7")` round-trip through canonical bytes and stay forward-compatible with any future schema_version: 2 promotion to a dedicated variant.

## Network announcements (5)

Each is pointer-only (SNIP root + drift-guard copies + required announcer signature). Subscribers fetch the inner body from SNIP and run the local verifier. Per-event topic lets subscribers filter:

```
omni/contributor/session/open/v1
omni/contributor/session/join/v1
omni/contributor/session/assign/v1
omni/contributor/session/partial/v1
omni/contributor/session/aggregated/v1
```

Topic safety lift (Stage 12.2-pre) means an unknown topic name is a typed `UnknownTopic` error, not a silent misroute.

## What the local verifier checks

Pure helpers, bytes in / typed outcome out:

- `verify_execution_session` — schema, `session_id == BLAKE3(canonical)`, coordinator signature.
- `verify_contributor_join` — schema, `join.session_id == session.session_id`, contributor signature.
- `verify_work_assignment` — schema, `assignment.session_id` matches, `assignment_id == BLAKE3(canonical)`, contributor is in the joined set, assignment signature verifies against the **session's** `coordinator_pubkey_hex`.
- `verify_partial_result` — schema, binding to assignment (session_id + assignment_id + contributor_pubkey_hex), contributor signature.
- `verify_aggregated_result` — schema, drift guards against session, every assignment has exactly one referenced partial (no `--allow-incomplete` in v1), `partial_canonical_hash` matches each referenced partial, coordinator signature verifies and equals `session.coordinator_pubkey_hex`.
- `check_not_expired` — `now_utc < expires_at_utc` (RFC 3339 Z lex-compare).

What we do NOT verify:
- Semantic correctness of any partial's output bytes.
- That contributors actually have the RAM they advertised.
- That contributor *i*'s output is the right input for contributor *i+1*'s stage.
- That `model_hash` names a "good" model.

## CLI surface

```
omni-node operator contributor open-session       --posted-job <p> --coordinator-seed <p> --expires-at-utc <ISO>
omni-node operator contributor join-session       --execution-session-snip-root 0x… --contributor-seed <p>
                                                  --available-ram-bytes <N> --max-input-tokens <N> --max-output-tokens <N>
                                                  --supported-work-unit-kind tokens|prefill-tokens|… (repeatable)
                                                  --runner-kind <str>
omni-node operator contributor assign-work        --execution-session-snip-root 0x… --coordinator-seed <p>
                                                  --assignments-file <path>
omni-node operator contributor run-assignment     --assignment-snip-root 0x… --execution-session-snip-root 0x…
                                                  --contributor-seed <p> --runner stub|external …
omni-node operator contributor aggregate-session  --execution-session-snip-root 0x… --coordinator-seed <p>
                                                  --final-result-snip-root 0x…
                                                  --join-snip-root 0x…       (repeatable, required)
                                                  --assignment-snip-root 0x… (repeatable, paired)
                                                  --partial-snip-root 0x…    (repeatable, paired)
omni-node operator contributor watch-sessions     --out-dir <dir> [--posted-id …] [--session-id …]
```

`assign-work` reads a small JSON file describing each assignment (`contributor_pubkey_hex`, `stage_index`, `work_kind`, `expected_work_units`, `expected_work_unit_kind`). There is no auto-selection policy at 12.3 — operator scripts implement RAM-greedy / FCFS / whatever they want on top.

Every publish-and-broadcast subcommand (open / join / assign / run / aggregate) takes the same `--peer-wait-secs` (default 30s) + `--mesh-stabilize-ms` (default 500ms) Stage 12.2 added to the announce-* paths, so a fresh `OmniNet` doesn't silently drop publish on an empty mesh.

## Deferred to 12.4+ (call out)

- Live tensor / activation transport between participants via `omni-net`'s Phase-4 `TensorRequest`/`TensorResponse`. Stage 12.3 hands off via SNIP only.
- Auto-selection policy for joins → assignments (RAM-greedy, capability-match).
- Filesystem source for sessions (analog of 12.1's `FilesystemSource`).
- Sybil / anti-spam beyond required signatures.
- Anything in items 1 / 7 / 8 of the chart; the lettered A–F flow; any chain-side interaction.
- Reward-split engine that consumes partial `work_units` to divide the job-level base-unit pool.

---

# Stage 12.4 — live tensor / activation transport

**Status**: shipped at Stage 12.4. Extends Stage 12.3 sessions with direct peer-to-peer activation handoff over `omni-net`'s existing tensor request/response codec. SNIP keeps its role as durable audit + fallback storage.

## Posture (unchanged)

No chain wire / Stage 7b tx / SUM Chain RPC / `InferenceAttestationDigest` / payment / marketplace / exclusive claim / proof mode / on-chain verification / A–F flow. SNIP only for durable storage / fallback. Default `omni-node` tree still pulls zero halo2/prover/framework runtimes.

## What 12.4 is

**A signed handoff layer for sending one stage's output activation directly to the next stage's contributor**, without forcing the bytes through SNIP first. The session/join/assign/partial chain from 12.3 is unchanged; 12.4 adds one new envelope (`ActivationHandoff`) carried inside `omni-net`'s existing `TensorRequest.data`.

**What 12.4 does NOT do**:
- Define model-specific tensor semantics. Bytes are opaque.
- Prove ML correctness. AttestationOnly only.
- Implement generalized distributed scheduling. Topology is **strict linear** at v1 (`to.stage_index == from.stage_index + 1`).
- Discover peer IDs from `ContributorJoin` schemas (the join envelope does not change). Recipient PeerId is **operator-supplied** at v1; signed peer-advertisement is Stage 12.5+.

## omni-net inspection — no lift required

`omni-net` already ships `TENSOR_XFER_PROTOCOL = "/omni/tensor-xfer/1"` (libp2p request/response), `OmniNet::request_tensor(peer_id, TensorRequest)`, `OmniNet::respond_tensor(channel_id, TensorResponse)`, and `OmniNetEvent::TensorReceived { request, channel_id }`. Single message cap 128 MiB. Async, direct-peer-targeted.

Stage 12.4 treats the outer `TensorRequest` as routing-only and carries a signed `ActivationHandoff` envelope (bincode 1.3) inside `TensorRequest.data`. **The outer fields are advisory; only the signed inner envelope is trusted on the receive side.** Outer `TensorRequest.micro_batch_index` is overloaded to carry `chunk_index` for transport-side routing — documented as Stage 12.4 transport overlay; the verifier never trusts it.

## Inner envelope

```text
ActivationHandoff {
    schema_version,                 // = 1
    session_id, from_assignment_id, to_assignment_id,
    from_contributor_pubkey_hex, to_contributor_pubkey_hex,
    dtype: F16 | BF16 | F32,        // closed enum
    shape: [u64; 1..=8],            // non-empty, every dim > 0
    byte_len,                       // total reassembled bytes
    tensor_hash,                    // BLAKE3 of reassembled bytes
    chunk_index, chunk_count,       // 1..=256 chunks per stream
    produced_at_utc,
    tensor_chunk_bytes,             // THIS chunk; ≤ 64 MiB
    sender_signature_hex,
}
```

**Canonical body excludes `tensor_chunk_bytes` and `sender_signature_hex`.** The signature binds `tensor_hash` (BLAKE3 of the full reassembled bytes), `byte_len`, all chunk metadata, and the from/to assignment / contributor identities. Receivers MUST:
1. Verify the sender signature against `from_contributor_pubkey_hex`.
2. After reassembling all `chunk_count` chunks in `chunk_index` order, recompute BLAKE3 and reject on mismatch with `tensor_hash`.
3. Reject if reassembled length ≠ `byte_len`.

This is a two-step content integrity check: signature → hash → bytes.

Bounds enforced in schema validation:
| Limit | Default | Field |
|---|---|---|
| Single chunk bytes | 64 MiB | `tensor_chunk_bytes.len()` |
| Total bytes | 16 GiB | `byte_len` |
| Max chunks | 256 | `chunk_count` |
| Shape rank | 8 | `shape.len()` |

## Stage 12.3 binding (`verify_activation_handoff`)

- `handoff.session_id == session.session_id`.
- `from_assignment.session_id == session.session_id` and `to_assignment.session_id == session.session_id`.
- `to.stage_index == from.stage_index + 1` (strict linear, v1).
- `handoff.from_contributor_pubkey_hex == from_assignment.contributor_pubkey_hex`.
- `handoff.to_contributor_pubkey_hex == to_assignment.contributor_pubkey_hex`.
- Sender signature verifies.
- `produced_at_utc < session.expires_at_utc`.

The reassembler (`HandoffReceiver::feed`) layers on:
- Drift-check every shared metadata field against the first chunk on the stream (chunk_count, byte_len, tensor_hash, dtype, shape, from/to pubkeys).
- Duplicate `chunk_index` is idempotent reject (stream stays live).
- On `Complete`: total length + BLAKE3 must match the signed values, else `TensorHashMismatch` / `ByteLenMismatch`.

## Transport

`TensorTransport` trait with two impls:
- `InMemoryTensorTransport` — sync VecDeque, no networking. Tests + single-process pipelines.
- `OmniNetTensorTransport` (feature `network`) — wraps `Arc<tokio::sync::Mutex<OmniNet>>`, packs the inner envelope into `TensorRequest.data`, ACKs every received envelope on the wire (acceptance is "received", not "trusted"), accumulates pending envelopes into a per-instance queue.

PeerId hint parsing accepts either a bare PeerId base58 or a multiaddr whose last component is `/p2p/<peer-id>`.

## Runner integration

```rust
pub trait InferenceRunner {
    fn run(&self, manifest_path: &Path, input_bytes: &[u8]) -> Result<RunOutput, RunnerError>;

    // Stage 12.4 — default impl delegates to run() + returns None for the
    // output activation. Stage 12.0/12.1/12.2/12.3 runners keep working.
    fn run_with_activations(
        &self,
        manifest_path: &Path,
        input_bytes: &[u8],
        upstream_activation_bytes: Option<&[u8]>,
    ) -> Result<RunOutputWithActivation, RunnerError> { ... }
}

pub struct RunOutputWithActivation {
    pub run_output: RunOutput,
    pub output_activation_bytes: Option<Vec<u8>>,
}
```

`ExternalCommandRunner` overrides `run_with_activations`: writes the optional upstream activation to a tempfile, always passes `--activation-out <path>` to the subprocess, reads that file back if non-empty. **No model-framework runtime deps in `omni-node`** — the framework lives in the external command.

## CLI surface

```
omni-node operator contributor run-assignment
    --assignment-snip-root 0x… --execution-session-snip-root 0x…
    --contributor-seed <p> --runner stub|external …
    [--upstream-from-assignment-snip-root 0x…]
    [--upstream-from-peer <peer-id|multiaddr>]      # advisory hint
    [--activation-in-mode none|live]                # default: none
    [--downstream-to-assignment-snip-root 0x…]
    [--downstream-to-peer <peer-id|multiaddr>]
    [--activation-out-mode snip|live|both|none]
    # default: `both` if downstream present, else `snip` (12.3 behavior)
    [--upstream-wait-secs <n>]                       # default 60
    [--handoff-chunk-max-bytes <n>]                  # default 64 MiB

omni-node operator contributor send-handoff
    --execution-session-snip-root 0x…
    --from-assignment-snip-root 0x… --to-assignment-snip-root 0x…
    --from-contributor-seed <p>
    --activation-file <path>
    --dtype f16|bf16|f32 --shape <comma-separated-u64s>
    --to-peer <peer-id|multiaddr>
    [--chunk-max-bytes <n>]                          # default 64 MiB
```

`run-assignment`'s effective default for `--activation-out-mode`:
- **Downstream peer + assignment supplied → `both`** (live for latency, SNIP for audit/fallback).
- No downstream supplied → `snip` (Stage 12.3 behavior).

`watch-handoffs` is intentionally not in v1; the important path is `run-assignment --activation-in-mode live`. Diagnostic sends use `send-handoff`.

## Fallback semantics

| Mode | Live send | SNIP publish |
|---|---|---|
| `snip` (default no-downstream) | no | yes (12.3 behavior) |
| `live` | yes | no |
| `both` (default with-downstream) | yes | yes |
| `none` | no | no |

`activation-in-mode snip` (read upstream partial from SNIP) is NOT implemented in v1 — explicit `none` (first stage) or `live` only. Operators wiring SNIP fallback today extract bytes off the upstream partial out-of-band.

## Deferred to 12.5+

- **Signed contributor peer-advertisement** (so the operator does not have to supply `--downstream-to-peer` out-of-band). This is the right shape for `ContributorJoin` schema v2.
- **Non-linear topologies** (MoE branching, ring attention, many-to-one handoffs).
- **`activation-in-mode snip`** as a first-class CLI flag.
- **Receive-side rate limiting + per-session pending-handoff caps** (today: enforced per-envelope by schema bounds; coarser per-session caps are out of scope).
- **Reward-split engine** that consumes 12.3 partial `work_units` + 12.4 handoff metadata to divide the job-level base-unit pool.

---

# Stage 12.5 — signed contributor peer advertisement / session routing

**Status**: shipped at Stage 12.5. Builds on Stage 12.4 live activation handoff. Removes manual `--downstream-to-peer` from the pooled inference flow by letting contributors publish signed, session-scoped reachability hints.

## Posture (unchanged)

No chain wire / Stage 7b tx / SUM Chain RPC / `InferenceAttestationDigest` change. No payment / reward / staking / slashing. No marketplace / auction / bid logic. No proof mode / on-chain verification / chain proof allowlist. SNIP only for durable storage; mesh announcements are pointer-only. Default `omni-node` build still pulls zero halo2/prover/framework runtimes.

## What 12.5 actually is

**Local routing data, not a marketplace, registry, or chain authority.** A contributor publishes a signed `ContributorPeerAdvertisement` that binds:

```
ExecutionSession.session_id
  → ContributorJoin.contributor_pubkey_hex
    → ContributorPeerAdvertisement.libp2p_peer_id (+ multiaddrs, capabilities)
```

The advertisement is **per-session** and **short-lived** (≤ 24h by schema validation). Receivers verify announcer signature + body signature + drift + matching join + expiry, then cache locally for [Stage 12.4 live handoff](crates/omni-contributor/src/handoff.rs) route resolution. Two coordinators running parallel sessions for the same `posted_id` cache their own advertisements independently. There is no global registry.

## PeerId binding is honest

`advertise-peer` derives `libp2p_peer_id` from `OmniNet::local_peer_id()` (Stage 12.5-pre lift). There is no `--libp2p-peer-id` flag — operators can't lie about what node they're running. Restart-cycle implications:

- `omni-net`'s `SwarmBuilder::with_new_identity()` regenerates the libp2p keypair on every `OmniNet::new`, so **restart = new PeerId**. Any advertisement published before a restart is dead afterwards.
- That matches the ≤ 24h freshness cap: Stage 12.5 advertisements are short-lived routing hints, not permanent identity records.
- Persistent libp2p identity (so PeerIds survive restart) is a separate, deferred concern. Not in 12.5.

## New envelopes (2)

```text
ContributorPeerAdvertisement {
    schema_version, advertisement_id, session_id, contributor_pubkey_hex,
    libp2p_peer_id,                  // base58, from OmniNet::local_peer_id()
    listen_multiaddrs,               // may be empty; /p2p tail must match libp2p_peer_id
    capabilities: PeerCapabilities {
        supports_live_handoff,       // must be true for route resolution
        max_handoff_chunk_bytes,     // 1..=HANDOFF_CHUNK_MAX_BYTES (Stage 12.4 bound)
        supported_dtypes,            // non-empty
    },
    advertised_at_utc,
    expires_at_utc,                  // > advertised_at, <= advertised_at + 24h
    contributor_signature_hex,       // over canonical body (excl. advertisement_id + signature)
}

NetworkPeerAdvertisementAnnouncement {
    schema_version, peer_advertisement_snip_root, advertisement_id, session_id,
    contributor_pubkey_hex, announced_at_utc, announcer_pubkey_hex,
    announcer_signature_hex,         // required; inner contributor sig is the trust root
}
```

New domain separators: `OMNINODE-CONTRIBUTOR-PEER-ADVERT:v1:` and `OMNINODE-CONTRIBUTOR-NET-PEER-ADVERT:v1:`.

New mesh topic: `omni/contributor/session/peer-advert/v1` (carried with the existing `UnknownTopic` safety lift).

## Local routing cache

`PeerRoutingCache` keys on `(session_id, contributor_pubkey_hex)`. **Newest non-expired advertisement wins.** `resolve(session_id, contributor_pubkey_hex, dtype, local_chunk_cap, now_utc)` returns:

- `Found(ResolvedPeerRoute { peer_id, multiaddrs, max_handoff_chunk_bytes, negotiated_dtype })` — `max_handoff_chunk_bytes` is `min(local_chunk_cap, advertised.max_handoff_chunk_bytes)`.
- `NoAdvertisement` — nothing cached for this key.
- `AllExpired { newest_expires_at }` — advertisement exists but `now_utc >= expires_at_utc`.
- `LiveHandoffNotSupported` — advertisement's `supports_live_handoff` is false.
- `DtypeNotSupported { requested, supported }` — requested dtype is not in the advertised list.

## CLI

```bash
# Publish a peer advertisement. PeerId comes from the live OmniNet.
omni-node operator contributor advertise-peer \
  --execution-session-snip-root 0x... \
  --join-snip-root 0x... \
  --contributor-seed <p> \
  [--listen-multiaddr /ip4/.../udp/.../quic-v1 ...] \
  --max-handoff-chunk-bytes 33554432 \
  --supported-dtype f16 --supported-dtype bf16 \
  --expires-in-secs 3600 \
  [--publish-announcement]

# Long-running passive watcher; writes verified adverts to disk.
omni-node operator contributor watch-peer-adverts \
  --out-dir <dir> \
  --joins-dir <watch-sessions-out-dir>          # source of verified joins
  [--session-id <hex64> ...] [--contributor-pubkey <hex64> ...] \
  [--max-adverts <N>] [--max-polls <N>]

# Run-assignment can now resolve its downstream peer from the cache.
omni-node operator contributor run-assignment \
  --assignment-snip-root 0x... \
  --execution-session-snip-root 0x... \
  --downstream-to-assignment-snip-root 0x... \
  --peer-advert-dir <watch-peer-adverts-out-dir> \
  --joins-dir <watch-sessions-out-dir>     # required for join verification
  --resolve-downstream-peer-from-session \
  [--downstream-resolve-dtype f16|bf16|f32] \
  --contributor-seed <p> --runner ...
  # `--downstream-to-peer` still works and TAKES PRECEDENCE if both are supplied.
```

The canonical workflow is:

```bash
omni-node operator contributor watch-sessions       --out-dir A
omni-node operator contributor watch-peer-adverts   --out-dir B --joins-dir A
omni-node operator contributor run-assignment       --peer-advert-dir B --joins-dir A \
                                                    --resolve-downstream-peer-from-session ...
```

Both `watch-peer-adverts` and `run-assignment` point at the same `watch-sessions --out-dir` for the join source. The loader walks the `<dir>/<session_id>/{session.json, joins/*.json}` layout: every join goes through `verify_execution_session` on the sibling `session.json` followed by `verify_contributor_join(&session, &join)` (Stage 12.3 verifier — schema + session binding + contributor signature). Joins that fail either check are dropped with a stderr warning. A forged local join file cannot make a forged peer advert pass the matching-join gate.

## What the announcement processor verifies

[`process_peer_advertisement_announcement`](crates/omni-contributor/src/peer_routing.rs) follows the Stage 12.3 / 12.4 processor pattern:

1. Announcement schema valid.
2. Announcer signature verifies against `announcer_pubkey_hex`.
3. Fetch advertisement from SNIP at `peer_advertisement_snip_root`.
4. Advertisement body schema valid (including 24h expiry bound + PeerId/Multiaddr parse + `/p2p` matching).
5. Inner contributor signature verifies against `contributor_pubkey_hex`.
6. Drift checks: `(advertisement_id, session_id, contributor_pubkey_hex)` agree between announcement and body.
7. **Matching-join check**: `(session_id, contributor_pubkey_hex)` must match a supplied verified `ContributorJoin`. Operators wire this from `watch-sessions` output.
8. Expiry: routine watchers reject `now_utc >= expires_at_utc`. Forensic re-runs can pass `now_utc = None` to skip.

Returns a typed `PeerAdvertisementOutcome` whose failure variants map 1:1 to operator-visible bare-stdout events.

## Out of scope for 12.5 (call out)

- Persistent libp2p identity that survives `omni-node` restart.
- Non-linear / branching pipelines (Stage 12.4's `+1` topology constraint still applies).
- Advert-driven peer reputation / scoring.
- Any chain-side interaction, payment, slashing, reward distribution.
- Anything in items 1 / 7 / 8 of the chart or the lettered A–F flow.

---

# Stage 12.6 — persistent libp2p mesh identity

**Status**: shipped at Stage 12.6. Builds on Stage 12.5-pre's `OmniNet::local_peer_id()` accessor + Stage 12.5's signed peer advertisement. Operational reliability hardening — no protocol or schema change.

## Posture (unchanged)

No chain wire / Stage 7b tx / SUM Chain RPC / `InferenceAttestationDigest` change. No payment / reward / staking / slashing. No marketplace / auction / bid logic. No proof mode / on-chain verification / chain proof allowlist. SNIP only for contributor artifacts. Default `omni-node` tree still pulls zero halo2/prover/framework runtimes. No Stage 12.0–12.5 schema or canonical-byte changes.

## Problem

Stage 12.5 honestly advertises `OmniNet::local_peer_id()`. But pre-12.6 `OmniNet::new` calls `SwarmBuilder::with_new_identity()`, so the libp2p Ed25519 keypair is regenerated on every process start. Restart `omni-node` → new `PeerId` → every peer advertisement published before the restart dies, no matter how much of its ≤24h freshness window was left.

## Solution

Add a persistent identity policy on `NetConfig` and a `--net-identity-file <path>` CLI flag plumbed through every contributor subcommand that opens `OmniNet`. When the flag is supplied:

- The CLI calls `omni_net::load_or_create_keypair_file_bytes(&path)` which **auto-creates** the file with `0o600` (Unix) on first use and returns the libp2p protobuf-encoded keypair bytes.
- Those bytes ride into `NetConfig.identity = NetIdentity::KeypairProtobufBytes(_)`.
- The swarm builder branches: `Ephemeral` keeps pre-12.6 behavior; `KeypairProtobufBytes(_)` decodes the bytes via `libp2p_identity::Keypair::from_protobuf_encoding` and feeds the result to `SwarmBuilder::with_existing_identity`.
- `OmniNet::local_peer_id()` (Stage 12.5-pre) returns the same value across restarts → Stage 12.5 advertisements stay valid for their full freshness window.

If the flag is omitted, the swarm uses an ephemeral identity — every existing 12.0–12.5 caller stays bit-identical.

## Failure posture (intentional)

- **Missing file** → auto-create at `0o600` on Unix. No separate `init-net-identity` command — matches what every other identity-bearing libp2p binary does.
- **Existing file with malformed bytes** → typed `IdentityError::Decode`. The loader does NOT silently fall back to a fresh identity or overwrite the file. Operators rotating identity delete the file explicitly.
- **Existing file with world/group-readable perms on Unix** → typed `IdentityError::PermissionsTooBroad { mode }`. No load — a key that may have leaked is not silently used. The file is not modified.
- **Non-regular file** (directory, symlink loop, etc.) → typed `IdentityError::NotARegularFile`.
- On non-Unix targets the permission check is `#[cfg(unix)]`-gated and behavior degrades to "create with default perms" (same posture Stage 12.0's contributor-seed loader takes).

## Roles are disjoint

The libp2p mesh transport identity and the contributor signing identity are **NOT the same role**. Stage 12.0–12.5 uses separate Ed25519 seeds for `ContributorSigner` / `CoordinatorSigner` / `DispatcherSigner` (passed via `--seed-file` / `--contributor-seed` / `--coordinator-seed`). The mesh identity goes through `--net-identity-file`. Do not reuse a contributor seed as a `--net-identity-file` — different role, different blast radius if compromised.

## Operator workflow

```bash
# First run: creates ./omni-net.key at 0600 with a fresh keypair.
omni-node operator contributor advertise-peer \
  --net-identity-file ./omni-net.key \
  --execution-session-snip-root 0x... \
  --join-snip-root 0x... \
  --contributor-seed ./contributor.seed \
  --max-handoff-chunk-bytes 33554432 \
  --supported-dtype f16 \
  --expires-in-secs 3600 \
  --publish-announcement

# Restart omni-node, re-run with the same --net-identity-file → same PeerId.
# Previously published peer advertisements remain valid until expires_at_utc.
```

The flag is available on every contributor subcommand that opens `OmniNet`:

- Stage 12.2: `announce-job`, `watch-network-jobs`, `announce-result`, `watch-network-results`
- Stage 12.3: `open-session`, `join-session`, `assign-work`, `run-assignment`, `aggregate-session`, `watch-sessions`
- Stage 12.4: `send-handoff`
- Stage 12.5: `advertise-peer`, `watch-peer-adverts`

## Rotation

Peer advertisements remain short-lived (≤24h) and session-scoped. Stage 12.6 does NOT make them permanent identity records. To rotate identity:

1. Delete `./omni-net.key` (or move it aside).
2. Restart `omni-node` with the same `--net-identity-file <path>` → a fresh keypair is auto-created → new `PeerId`.
3. Re-run `advertise-peer` so peers learn the new route.

Stage 12.5's matching-join check still gates which advertisements receivers will accept; an attacker who steals the identity file but not the contributor seed cannot publish valid advertisements.

## Manual re-publish before expiry

Stage 12.6 stabilizes the PeerId across restart, but advertisements still expire after `expires_at_utc`. Operators wanting continuous routing across the freshness boundary should re-run `advertise-peer` manually before expiry. Auto-refresh (a long-running daemon that re-publishes near expiry) is intentionally deferred — likely Stage 12.7+.

## What 12.6 does NOT do

- Does NOT make peer advertisements long-lived. The ≤24h schema cap stays.
- Does NOT introduce a centralized identity registry. Rotation is operator-controlled.
- Does NOT change any 12.0–12.5 schema, canonical bytes, or wire format.
- Does NOT add a chain-side anything. Mesh identity is local infrastructure.
- Does NOT pull libp2p transport into the default `omni-node` tree — the new helper uses only `libp2p::identity` (already a transitive dep via `omni-net`).

---

# Stage 12.7 — local contributor workflow state persistence

**Status**: shipped at Stage 12.7. Pairs with Stage 12.6 persistent mesh identity. Operational reliability hardening — no protocol or schema change.

## Posture (unchanged)

No chain wire / Stage 7b tx / SUM Chain RPC / `InferenceAttestationDigest` change. No payment / reward / staking / slashing. No marketplace / auction / bid logic. No proof mode / on-chain verification / chain proof allowlist. SNIP only for contributor artifacts. Default `omni-node` tree still pulls zero halo2/prover/framework runtimes. No Stage 12.0–12.6 schema or canonical-byte changes. No omni-net changes.

## Problem

Stage 12.6 keeps the libp2p `PeerId` stable across restart so peer advertisements survive their ≤24h freshness window. But the rest of the contributor workflow state — already-seen announcements, verified sessions/joins/assignments/partials/aggregates, verified peer advertisements, just-fetched posted-result-links — still lives in either the in-memory dedup `HashSet` of `run_watch_loop` or in operator-facing per-command output directories that were never designed as a "resume" source of truth.

Concretely, restart `omni-node` mid-run today and:

- `watch-jobs` / `watch-network-jobs` will re-pick up every posted job whose announcement is still cached at SNIP — the in-memory `seen` set is gone.
- `watch-sessions` re-fetches sessions/joins/assignments/partials/aggregates from SNIP for every announcement, even ones already verified to disk.
- `watch-peer-adverts` re-verifies every advertisement from scratch.
- `run-assignment --resolve-downstream-peer-from-session` needs two separate flags (`--peer-advert-dir`, `--joins-dir`) to find what it needs, and there is no single canonical "this is the resume state" location.

## Solution

Add a local `ContributorStateStore` (in `crates/omni-contributor/src/state.rs`) backed by a versioned, atomic-write JSON tree, and plumb a `--contributor-state-dir <path>` + `--no-prune-state-on-start` flag onto every restart-sensitive contributor subcommand.

### Layout

```text
<state-dir>/
  meta/state_version.json
  seen/posted-jobs/<posted_id>
  seen/network-job-announcements/<posted_id>
  seen/network-result-announcements/<posted_id>--<snip_root>
  seen/sessions/<session_id>
  seen/joins/<session_id>--<contributor_pubkey>
  seen/assignments/<session_id>--<assignment_id>
  seen/partials/<session_id>--<assignment_id>
  seen/aggregates/<session_id>
  seen/peer-adverts/<session_id>--<contributor_pubkey>
  verified/sessions/<session_id>/session.json
  verified/sessions/<session_id>/joins/<pubkey>.json
  verified/sessions/<session_id>/assignments/<id>.json
  verified/sessions/<session_id>/partials/<id>.json
  verified/sessions/<session_id>/aggregated.json
  verified/sessions/<session_id>/peer-adverts/<pubkey>.json
  results/contributor-results/<job_id>.json
  results/contributor-results/<job_id>.rejected.json
  results/result-links/<posted_id>.link.json
```

The `verified/sessions/<id>/...` subtree is **bit-identical** to the Stage 12.3 `watch-sessions --out-dir` layout and the Stage 12.5 `watch-peer-adverts --out-dir` layout. The `results/contributor-results/...` subtree mirrors `watch-jobs --result-out-dir`. See the gradual-migration note below.

### Versioning + atomic writes

- `meta/state_version.json` carries `state_version: 1`, written on first `open`. Future binaries that bump `STATE_VERSION` will refuse mismatched directories with `StateError::UnsupportedVersion`. A 12.8+ migration is therefore clean.
- Every write goes through tempfile-in-same-directory + `fs::rename`, so a torn write never appears at the final path.
- `mark_seen` is idempotent (zero-byte marker via `OpenOptions::create_new`; pre-existing marker is treated as success).

### Auto-prune

`ContributorStateStore::open` auto-prunes by default: any verified session whose `expires_at_utc` has passed `now_utc` is removed (with cascade through its joins/assignments/partials/aggregate/peer-adverts + every matching `seen/` marker). Same for individual peer advertisements that are individually expired but inside a still-fresh session. Pass `--no-prune-state-on-start` to disable for forensic re-runs.

### CLI surface

```text
--contributor-state-dir <path>     # opt-in; absent = pre-12.7 behavior
--no-prune-state-on-start          # disable auto-prune on open
```

Wired on every restart-sensitive subcommand:

- `watch-jobs`
- `watch-network-jobs`
- `watch-network-results`
- `watch-sessions`
- `watch-peer-adverts`
- `run-assignment` (replaces `--peer-advert-dir` + `--joins-dir` when `--resolve-downstream-peer-from-session` is set)

When supplied, each watcher additionally:

- Loads cross-restart seen markers via `is_seen(StateNamespace::*, ...)` BEFORE the existing in-memory dedup, so already-handled posted-jobs / sessions / joins / assignments / partials / aggregates / peer-adverts are skipped without a SNIP refetch.
- Dual-writes verified bodies into the state-dir's tree (in addition to the existing operator-facing `--out-dir` / `--result-out-dir` paths).
- Marks `seen/` after each verified write so the next restart sees the entry as handled.

### Conflict policy (run-assignment)

When `--resolve-downstream-peer-from-session` is set:

- Supplying `--contributor-state-dir <path>` alone uses the state-dir's `verified/sessions/<id>/{joins,peer-adverts}/...` subtree as the canonical source.
- Supplying the legacy `--peer-advert-dir` + `--joins-dir` flags alone preserves pre-12.7 behavior bit-for-bit.
- Supplying `--contributor-state-dir` together with either legacy flag is rejected with `StateError::AmbiguousSource { legacy_flag }`. The point of the state-dir is to *be* the single source of truth; mixing it with the per-stage trees would re-introduce the layered-sources problem 12.7 exists to remove.

### Operator workflow

```bash
# Stage 12.7 — one state directory drives the whole restart-resumable
# watch + assignment flow.
omni-node operator contributor watch-sessions \
  --net-identity-file ./omni-net.key \
  --contributor-state-dir ./contrib-state \
  --out-dir ./contrib-state/verified/sessions \
  ...

omni-node operator contributor watch-peer-adverts \
  --net-identity-file ./omni-net.key \
  --contributor-state-dir ./contrib-state \
  --out-dir ./contrib-state/verified/sessions \
  ...

omni-node operator contributor run-assignment \
  --net-identity-file ./omni-net.key \
  --contributor-state-dir ./contrib-state \
  --resolve-downstream-peer-from-session \
  ...
```

Restart any of the three at any time; the state directory carries the dedup + verified caches forward. The `--out-dir` flags above point INTO the state-dir's verified subtree on purpose — a single directory serves both the operator-facing "I want to inspect verified envelopes" use case and the state-store's restart-resume use case.

### Gradual migration from pre-12.7 layouts

The `verified/sessions/<id>/...` subtree **under** the state-dir is the same per-session shape as the pre-12.7 `watch-sessions --out-dir <X>` tree. But the state-dir's reader (`list_verified_sessions`) only walks `<state-dir>/verified/sessions/...`; it does NOT auto-discover a flat `<X>/<id>/...` tree at the top level. So pointing `--contributor-state-dir <X>` at an existing `watch-sessions --out-dir <X>` does not migrate it.

Two supported migration paths:

1. **Start fresh** (recommended). Pick a new path, e.g. `./contrib-state`. Run the watchers with `--contributor-state-dir ./contrib-state` AND `--out-dir ./contrib-state/verified/sessions`. The pre-12.7 `--out-dir <X>` directories keep working in parallel; new envelopes populate the state-dir as they're announced. Once enough have re-fetched, drop the legacy `--peer-advert-dir` + `--joins-dir` flags from `run-assignment`. Cost: re-fetches that the old tree had cached. Benefit: zero manual file moves.
2. **In-place move**. `mkdir -p <X>/verified/sessions && mv <X>/<id> <X>/verified/sessions/<id>` for each pre-12.7 per-session subdirectory, then point 12.7 commands at `--contributor-state-dir <X>`. Cost: one-time file-move. Benefit: keeps the already-validated envelopes.

Either way, the state-dir is purely local. Delete it to force a clean re-fetch; the protocol itself does not know it exists.

### Safety properties

- **No private key material**. The contributor signing seed lives in `--contributor-seed` (Stage 12.0), the libp2p mesh keypair lives in `--net-identity-file` (Stage 12.6). The state-dir only stores envelopes that are already public on SNIP plus marker files.
- **Schema-versioned, not chain-anchored**. The state-dir is a local cache; nothing trusts it as evidence. Verification of restored bodies still re-runs the Stage 12.3 / 12.4 / 12.5 verifiers before any action.
- **Inspectable**. Everything is pretty-printed JSON. Operators can `cat` and `jq` freely.
- **Concurrent-safe (single-host)**. Tempfile + `rename` lets two processes pointing at the same directory write without tearing each other's files. Two processes both running `auto_prune` may race on `remove_dir_all`; that race is harmless (best-effort cleanup) but operators who care should not run two `omni-node` processes against the same state-dir.

### What 12.7 does NOT do

- Does NOT make the state-dir a remote sync target. It is local-only; cross-host replication would need a separate stage.
- Does NOT change any 12.0–12.6 schema, canonical bytes, or wire format. The on-disk JSON inside `verified/sessions/<id>/...` is the same JSON the existing watchers wrote.
- Does NOT replace the operator-facing `--out-dir` / `--result-out-dir` flags. Both still work; the state-dir is additional.
- Does NOT add a chain-side anything. The state-dir is local infrastructure.
- Does NOT pull halo2/prover/framework runtimes into the default `omni-node` tree.

---

# Stage 12.8 — local pooled-session assignment planner

**Status**: shipped at Stage 12.8. Builds on Stage 12.7 state-dir loaders + Stage 12.3 `WorkAssignment` publish path. Coordination ergonomics, not protocol change.

## Posture (unchanged)

No chain wire / Stage 7b tx / SUM Chain RPC / `InferenceAttestationDigest` change. No payment / reward / staking / slashing. No marketplace / auction / bid / pricing logic. No proof mode / on-chain verification / chain proof allowlist. No omni-net change. No omni-store wire change. No Stage 12.0–12.7 schema or canonical-byte changes. Default `omni-node` tree still pulls zero halo2/prover/framework runtimes.

## Problem

Stage 12.3 ships `assign-work`, which takes a hand-edited `assignments.json` and publishes one `WorkAssignment` per entry. That's correct, but it forces a coordinator to spell out the contributor → stage mapping by hand for every session. Stage 12.7 stores verified joins and peer adverts in a state-dir; nothing reads those to propose an assignment shape. Operators end up either copy-pasting pubkeys out of `verified/sessions/<id>/joins/*.json` or writing one-off shell scripts.

## Solution

Add a local coordinator-side **assignment planner**. Given a Stage 12.7 state-dir, a session id, and a strategy, produce an `AssignmentPlan` JSON that names contributors for each stage. The plan is a **local review artifact** — not signed, not SNIP-published, not network-visible. A second subcommand reads the plan and publishes each entry as a normal Stage 12.3 `WorkAssignment` (signed by the coordinator, optionally broadcast on the mesh).

### What the planner is NOT

- **NOT a marketplace.** No bids, prices, scores, rewards, reputation, or winner-picking.
- **NOT a scheduler.** No RAM-weighted ranking, no historical-performance scoring, no global state.
- **NOT a network protocol.** `AssignmentPlan` is local-only and **unsigned** in v1. The signed trust artifact remains the Stage 12.3 `WorkAssignment` at publish time. Adding a plan signature is a `schema_version: 2` migration.
- **NOT a chain authority.** Coordinator stays a process role.

### Determinism

After eligibility filtering, contributors are sorted by `contributor_pubkey_hex` (lexicographic ASCII order on the lowercase hex string). All strategies consume that sorted list in order. Re-running the planner with the same inputs (and the same `now_utc` if any advert sits on the eligibility boundary) yields byte-identical output, including `plan_hash`.

### Eligibility filter

A `ContributorJoin` is eligible if:

1. `available_ram_bytes >= --min-available-ram-bytes` (operator floor, default 0). **Pass/fail** — the floor is "is this contributor able to participate at all," not a quality signal. Two contributors that both clear the floor are interchangeable to the planner.
2. If `--require-live-routing` is set: a non-expired `ContributorPeerAdvertisement` exists for the same `(session_id, contributor_pubkey_hex)` AND its `capabilities.supports_live_handoff == true` AND its `capabilities.supported_dtypes` contains `--required-dtype`.

No ranking. No RAM weighting. Filtering only.

### Strategies (v1, closed enum)

- `single-contributor`: one contributor handles the entire work envelope.
- `sequential-layers`:
  - With `--model-plan <path>`: one stage per model-plan entry, assigned round-robin across the sorted eligible set.
  - Without `--model-plan`, requires `--layer-count N`: equal split with the remainder absorbed by the LAST stage so total = N exactly.
- `round-robin`:
  - With `--model-plan <path>`: stage_index N → eligible[N mod eligible.len()].
  - Without `--model-plan`, requires `--layer-count N`: emits N single-layer stages cycling through the eligible set.

Adding a strategy is a `schema_version: 2` migration.

### CLI surface

```text
plan-session-assignments
  --contributor-state-dir <path>             required
  --session-id <hex64>                       required
  --strategy sequential-layers|single-contributor|round-robin   default sequential-layers
  --required-dtype f16|bf16|f32              default f16
  --min-available-ram-bytes <u64>            default 0 (no floor)
  --max-assignments <u32>                    optional
  --model-plan <path>                        optional
  --layer-count <u32>                        optional (required when no model-plan and strategy needs it)
  --require-live-routing                     flag; loads + filters peer adverts
  --out <path>                               required
  --no-prune-state-on-start                  optional (Stage 12.7 semantics)
```

```text
assign-session-plan
  --plan <path>                              required
  --session-snip-root <0x...>                required
  --coordinator-seed <path>                  required
  --snip-binary, --snip-seed                 SNIP plumbing
  --listen-port, --peer, --net-identity-file
  --peer-wait-secs, --mesh-stabilize-ms, --propagation-wait-ms
  --no-publish-announcements                 default false (broadcast in addition to SNIP publish)
  --dry-run                                  optional
  --contributor-state-dir <path>             optional; mirrors published assignments
  --no-prune-state-on-start                  optional
```

### `AssignmentPlan` shape (local-only, unsigned)

```json
{
  "schema_version": 1,
  "session_id": "<hex64>",
  "planner_version": "omni-contributor v0.1.0",
  "strategy": "SequentialLayers",
  "required_dtype": "f16",
  "created_at_utc": "2026-05-31T00:30:00Z",
  "coordinator_pubkey_hex": "<hex64>",
  "assignments": [
    {
      "stage_index": 0,
      "contributor_pubkey_hex": "<hex64>",
      "work_kind": {"layers": {"start": 0, "end": 16}},
      "expected_work_units": 16,
      "expected_work_unit_kind": "layers"
    }
  ],
  "plan_hash": "<blake3-hex>"
}
```

`plan_hash` is BLAKE3 over the canonical JSON body with `plan_hash` itself cleared. `assign-session-plan` recomputes it on read and refuses any plan whose hash drifts — guard against accidental edits between dry-run and publish.

### `ModelPlan` shape (operator-supplied, local-only)

```json
{
  "schema_version": 1,
  "stages": [
    {"stage_index": 0, "work_kind": {"layers": {"start": 0, "end": 16}}, "expected_work_units": 16, "expected_work_unit_kind": "layers"},
    {"stage_index": 1, "work_kind": {"layers": {"start": 16, "end": 32}}, "expected_work_units": 16, "expected_work_unit_kind": "layers"}
  ]
}
```

`stage_index` values must equal their position in the `stages` array (no gaps, no duplicates). `WorkKind::Layers { start, end }` requires `start < end`. `expected_work_units > 0`.

### Operator workflow

```bash
# 1. Open the session (12.3) + receive joins + accept peer adverts.
omni-node operator contributor open-session ...
omni-node operator contributor watch-sessions \
  --net-identity-file ./omni-net.key \
  --contributor-state-dir ./contrib-state \
  --out-dir ./contrib-state/verified/sessions \
  ...

# 2. Plan the assignments locally (no network, no SNIP).
omni-node operator contributor plan-session-assignments \
  --contributor-state-dir ./contrib-state \
  --session-id <hex64> \
  --strategy sequential-layers \
  --layer-count 32 \
  --out ./plan.json

# 3. Review ./plan.json. Optionally dry-run.
omni-node operator contributor assign-session-plan \
  --plan ./plan.json \
  --session-snip-root 0x... \
  --coordinator-seed ./coord.seed \
  --dry-run

# 4. Publish.
omni-node operator contributor assign-session-plan \
  --plan ./plan.json \
  --session-snip-root 0x... \
  --coordinator-seed ./coord.seed \
  --contributor-state-dir ./contrib-state \
  --net-identity-file ./omni-net.key
```

### Trust boundary

State-dir loaders are parse-only (Stage 12.7 review). `plan-session-assignments` re-runs `verify_execution_session`, `verify_contributor_join`, and `verify_peer_advertisement_body` on every artifact loaded from the state-dir before feeding it to the planner. The planner library entry also re-verifies internally (defense in depth) so an alternate caller cannot accidentally skip the check. `assign-session-plan` re-fetches and re-verifies the session from SNIP at publish time, then re-derives every signature — the on-disk plan is treated as a *suggestion*, not a trust anchor.

### Session expiry

Both `plan-session-assignments` (via the planner's internal `check_not_expired`) and `assign-session-plan` (via an explicit `check_not_expired` on the fetched session) refuse to operate when `now_utc >= session.expires_at_utc`. This matches the publish-time posture of every other Stage 12.x command and means `--no-prune-state-on-start` cannot reach a signed `WorkAssignment` against a stale session.

### `--max-assignments` semantics

The cap means different things to different strategies and is **never** silently truncating:

- `sequential-layers` WITHOUT `--model-plan`: cap is a **contributor cap** ("use at most N contributors"). The strategy splits the requested layer envelope across `min(eligible.len(), N)` contributors. Layer ranges still cover `0..layer_count` exactly.
- `sequential-layers` WITH `--model-plan`: cap is checked against `model_plan.stages.len()`. If the cap is smaller, the planner refuses with `PlannerError::MaxAssignmentsTooSmall` — dropping stages would leave the work envelope incomplete.
- `single-contributor`: cap is checked against `model_plan.stages.len()` (or 1 if no model-plan). Refused if too small.
- `round-robin`: cap is checked against `model_plan.stages.len()` (or `--layer-count` if no model-plan). Refused if too small.

Rationale: silently dropping the tail produces a plan whose `plan_hash` is valid but whose work coverage is incomplete. Refuse loudly instead.

### Dry-run validation

`assign-session-plan --dry-run` does not just print the plan — it builds, signs (locally; no SNIP, no mesh), and runs `WorkAssignment::validate_schema` on every planned entry. A hand-edited plan with `expected_work_units = 0` or an inverted `WorkKind::Layers` survives `plan_hash` recompute (the operator can re-stamp the hash after editing) but fails schema validation. Dry-run catches it before the real publish path.

### Why no RAM-weighting

A planner that ranks contributors by available RAM is a scheduler. A planner that signs winner declarations is a marketplace. Both directions are deliberately outside Stage 12.8's posture. The eligibility floor (`--min-available-ram-bytes`) exists so operators can refuse contributors that obviously can't run a workload — that's a feasibility check, not a quality signal.

### What 12.8 does NOT do

- Does NOT introduce a new signed envelope. `AssignmentPlan` is unsigned.
- Does NOT add a new gossipsub topic, SNIP root format, or canonical-bytes domain separator.
- Does NOT change any Stage 12.0–12.7 schema or canonical bytes.
- Does NOT add a chain-side anything. The planner is local infrastructure.
- Does NOT pull halo2/prover/framework runtimes into the default `omni-node` tree.
- Does NOT replace the Stage 12.3 `assign-work` CLI. `assign-work` keeps working for hand-edited spec files.

---

# Stage 12.9 — local pooled-session progress monitor

**Status**: shipped at Stage 12.9. Builds on Stage 12.7 state-dir loaders + Stage 12.3 verifier helpers. Read-only observability — no protocol or schema change.

## Posture (unchanged)

No chain wire / Stage 7b tx / SUM Chain RPC / `InferenceAttestationDigest` change. No payment / reward / staking / slashing. No marketplace / auction / bid / pricing logic. No proof mode / on-chain verification / chain proof allowlist. No omni-net change. No omni-store wire change. No Stage 12.0–12.8 schema or canonical-byte changes. Default `omni-node` tree still pulls zero halo2/prover/framework runtimes.

## Problem

Stage 12.8 can plan and publish assignments, but an operator inspecting a pooled session still has to walk `<state>/verified/sessions/<id>/...` by hand to check whether joins arrived, assignments were issued, partials came back, and the aggregate landed. There is no single command that says "is session X complete?" or "where is it stuck?"

## Solution

Add a read-only `session-status` subcommand that loads everything for a `--session-id` from a Stage 12.7 `--contributor-state-dir`, re-runs every relevant Stage 12.3 verifier, and emits a deterministic `SessionStatusReport`. The report is a **local snapshot** — never signed, never SNIP-published, never network-visible. Two operators on different machines will see different reports for the same session, and that's correct: each report describes *what its local state-dir contains*.

### What this is NOT

- **NOT a coordination enforcer.** The report doesn't fail a session, refuse work, trigger retries, or mutate anything on disk beyond an optional `--json-out` mirror.
- **NOT a network protocol.** `SessionStatusReport` is local-only and **unsigned**. Schema-versioned so a future Stage 12.10 can bump cleanly.
- **NOT a chain authority.** No transaction, no signature, no on-chain anchor.
- **NOT a TUI.** Just stdout events, JSON, or a compact one-screen table.
- **NOT a peer-advert-driven liveness probe.** Adverts inform `peer_advert_present` per assignment but never drive completion. A session can be `CompletePartials` or `Aggregated` with no advert at all in the state-dir.

### Trust boundary

State-dir loaders are parse-only (Stage 12.7 review). `build_session_status_report` re-runs:

- `verify_execution_session` on `session.json`,
- `verify_contributor_join` per join,
- `verify_work_assignment` per assignment (against the verified-joins pubkey set),
- `verify_partial_result` per partial (matched by `assignment_id`),
- `verify_peer_advertisement_body` per advert (against the verified joins),
- `verify_aggregated_result` full-chain when an aggregate is present.

Failing chain links (joins / assignments / partials / aggregate) drop the artifact from valid counts and set `overall_status = InvalidState`. Failing peer adverts surface as notes but never flip `overall_status` — they're routing helpers, not chain links.

A failed `verify_execution_session` is **fail-closed**: the reporter returns a minimal `InvalidState` report immediately, carrying no session-derived fields (`posted_id`, `model_hash`, `session_expires_at_utc` are all `None`) and zero counts. Continuing past a tampered session would feed an untrusted body into the downstream verifiers and let session-derived trust fields leak into the report. The single `notes` entry names `verify_execution_session` so an operator can find the bad file.

### `SessionOverallStatus` decision tree

Computed in priority order (first match wins):

1. `NoSession` — no `session.json` for the requested id, OR an inner-body session_id mismatch with the directory key.
2. `InvalidState` — any chain-link body failed individual re-verification, OR `verify_aggregated_result` rejected an aggregate that exists.
3. `Aggregated` — `aggregate_present && aggregate_valid`.
4. `ExpiredIncomplete` — `now_utc >= session.expires_at_utc` AND not aggregated.
5. `NoAssignments` — session valid, zero valid assignments.
6. `CompletePartials` — every valid assignment has exactly one valid partial; no aggregate yet.
7. `InProgress` — otherwise.

### Counts policy

Counts are **valid-only**. Tampered artifacts surface as `notes` + `InvalidState` overall, not as inflated counts.

### Duplicate partials in v1

The state-dir layout writes `verified/sessions/<id>/partials/<assignment_id>.json`. Partials are keyed by `assignment_id`, so a second write overwrites — **duplicates are filesystem-impossible in v1**. The `duplicate_partial_assignment_ids` field is always empty and is retained for forward compatibility (a future stage that grows a partials sidecar may need it).

### CLI surface

```text
session-status
  --contributor-state-dir <path>             required
  --session-id <hex64>                       required
  --format events|json|pretty                default events
  --json-out <path>                          optional (best-effort fs::write)
  --fail-on-incomplete                       flag (exit non-zero unless CompletePartials or Aggregated)
  --include-expired                          flag (count expired adverts too)
  --no-prune-state-on-start                  optional (inherits Stage 12.7)
```

### Output formats

**`events`** (default — matches every other Stage 12.x watcher):

```
event=session_status session_id=... status=InProgress assignments=3 partials=1 missing=2 aggregate=false expired=false
event=assignment_status session_id=... assignment_id=... stage_index=0 contributor=... join=present peer_advert=present partial=present
event=missing_partial session_id=... assignment_id=... stage_index=1 contributor=...
event=note context=session_status message=...
```

**`json`** — `serde_json::to_string_pretty(&report)` is the **only** thing written to stdout in this mode. Operational chatter (`event=state_store_opened`, `event=session_status_json_written`) is rerouted to stderr so `jq` and other parsers can consume stdout directly. The omni-node binary's `tracing` output also goes to stderr (Stage 12.9 lift), so a clean `omni-node operator contributor session-status --format json` invocation produces parseable JSON on stdout with no preprocessing required.

**`pretty`** — minimal terminal-friendly table. No TUI deps.

### `--json-out` posture

Best-effort `std::fs::write`. Failure produces a stderr warning but does not change the exit code. The report is a **dashboard snapshot, not a protocol artifact** — operator dashboards are fine with non-atomic writes. The state-store's atomic-write helper stays private to `state.rs`.

### `--fail-on-incomplete` exit code mapping

| `overall_status` | Exit code |
|---|---|
| `Aggregated` | `0` |
| `CompletePartials` | `0` |
| `NoSession` | `1` |
| `NoAssignments` | `1` |
| `InProgress` | `1` |
| `ExpiredIncomplete` | `1` |
| `InvalidState` | `1` |

Without `--fail-on-incomplete`, every status exits `0` so dashboard scrapers can run unconditionally.

### Operator workflow

```bash
# After running Stage 12.7 watchers + Stage 12.8 plan/assign:
omni-node operator contributor session-status \
  --contributor-state-dir ./contrib-state \
  --session-id <hex64>
# → event lines suitable for grep + tail.

# JSON for a dashboard:
omni-node operator contributor session-status \
  --contributor-state-dir ./contrib-state \
  --session-id <hex64> \
  --format json \
  --json-out ./status.json

# In a CI gate:
omni-node operator contributor session-status \
  --contributor-state-dir ./contrib-state \
  --session-id <hex64> \
  --fail-on-incomplete
```

### What 12.9 does NOT do

- Does NOT introduce a new signed envelope. `SessionStatusReport` is unsigned.
- Does NOT add a gossipsub topic, SNIP root format, or canonical-bytes domain separator.
- Does NOT change any Stage 12.0–12.8 schema or canonical bytes.
- Does NOT add a chain-side anything. The reporter is local infrastructure.
- Does NOT pull halo2/prover/framework runtimes into the default `omni-node` tree.
- Does NOT mutate any protocol artifact. The only write it performs is the optional `--json-out` mirror.

---

# Stage 12.10 — local pooled-session repair planner

**Status**: shipped at Stage 12.10. Builds on Stage 12.9 status reports + Stage 12.7 state-dir loaders + Stage 12.3 verifier helpers. Local operator hint generation — no protocol or schema change.

## Posture (unchanged)

No chain wire / Stage 7b tx / SUM Chain RPC / `InferenceAttestationDigest` change. No payment / reward / staking / slashing. No marketplace / auction / bid / pricing / reputation logic. No proof mode / on-chain verification / chain proof allowlist. **No new omni-net gossipsub topic — reuses `TOPIC_SESSION_WORK_ASSIGNED`.** No omni-store wire change. No Stage 12.0–12.9 schema or canonical-byte changes. Default `omni-node` tree still pulls zero halo2/prover/framework runtimes. **No state-dir mutation on apply.**

## Problem

Stage 12.9 tells the operator that a session is `InProgress` with N missing partials, but it doesn't act. An operator looking at "stage 1 has no partial 30 minutes after open-session" has no built-in way to nudge the contributor's libp2p subscription. The old protocol path is to re-run `assign-work` manually with the same spec — error-prone and not auditable.

## Solution

Add a local repair planner that converts a `SessionStatusReport` into a list of operator-reviewed follow-up actions, plus an applier that executes them. The plan is a **local review artifact** — never signed, never SNIP-published, never network-visible. The signed trust artifact remains the Stage 12.3 `WorkAssignment` at apply time, byte-preserved across the reannounce.

### What this is NOT

- **NOT a coordination enforcer.** The plan is an operator-visible suggestion. Contributors can ignore reannouncements just as they could ignore the original assignment.
- **NOT a cancellation/supersession system.** Replacement assignments are deferred to Stage 12.11+ (see halt-finding subsection below).
- **NOT a retry daemon.** Operators run `apply-session-repair` explicitly; nothing auto-fires.
- **NOT a penalty system.** Reannouncement is a libp2p subscription nudge, not a sanction.
- **NOT a new envelope.** Reuses `WorkAssignment` + `NetworkWorkAssignedAnnouncement` verbatim.
- **NOT a new gossipsub topic.** Reuses `TOPIC_SESSION_WORK_ASSIGNED`.
- **NOT a state mutation.** `apply-session-repair` writes nothing to the state-dir.
- **NOT a chain authority.** Coordinator stays a process role.

### Halt finding — why no `ReassignMissing` in v1

`verify_aggregated_result` requires every assignment in the supplied slice to be referenced exactly once in the aggregate's `partial_refs`. Adding a replacement assignment for a missing partial leaves the old assignment in the state-dir's `verified/sessions/<id>/assignments/` and would trip `AggregateMissingPartialFor` at aggregate time. There is no canonical "this assignment is superseded" envelope in Stage 12.3.

A supersession model would require one of:
- A new signed envelope declaring `supersedes: Vec<assignment_id>`, OR
- A new field on `WorkAssignment` itself, OR
- A new verifier mode that takes a `superseded: HashSet<String>` and skips matching assignments.

All three are `schema_version: 2` migrations to Stage 12.3 envelopes — out of scope for Stage 12.10's "no canonical-byte changes" posture. Stage 12.11+ will design supersession explicitly. The Stage 12.10 test suite includes `aggregate_verifier_rejects_extra_assignment_without_partial` as an explicit fixture pinning the constraint, so a future supersession design discussion has the failure case documented.

### Trust boundary

State-dir loaders are parse-only (Stage 12.7 review, restated). The planner consumes an already-built `SessionStatusReport` (which itself re-runs every Stage 12.3 verifier — Stage 12.9 contract). The applier:

1. Recomputes `repair_plan_hash` and refuses on drift.
2. Recomputes `source_status_hash` from the **current** state-dir and refuses on drift (a partial may have arrived between plan and apply).
3. Re-runs `check_repair_eligible` on the current status. The `source_status_hash` projection is intentionally narrow — it covers only `(session_id, sorted [(assignment_id, partial_present)] pairs)` — so a status flip from `InProgress` to `ExpiredIncomplete` (clock advanced past `session.expires_at_utc`) or `InvalidState` (an aggregate body added between plan and apply that failed the verifier) can leave the projection unchanged. The eligibility re-check catches that explicitly, using the same status→error matrix as the planner.
4. Re-fetches and re-verifies the session via `--session-snip-root` at apply time.
5. Re-verifies each referenced assignment from the state-dir via `verify_work_assignment` before any SNIP write.

The on-disk plan is treated as a **suggestion**, not a trust anchor.

### `--no-publish-announcements` is SNIP-only

When set, the applier skips the entire omni-net layer: no mesh open, no peer wait, no propagation sleep, no shutdown. The reannouncement is then a pure SNIP-republish loop (no observable latency tax for the 30s peer-wait default). With the flag unset (default), the apply opens the mesh, peer-waits, broadcasts a fresh `NetworkWorkAssignedAnnouncement` per action, sleeps for `--propagation-wait-ms`, and shuts down — matching Stage 12.8 `assign-session-plan`.

### `source_status_hash` projection

BLAKE3 over `(session_id, sorted [(assignment_id, partial_present)] pairs)`. Two status reports with different `generated_at_utc` or different `notes` but the same operator-meaningful shape produce the same projection — so a plan built from a long-running watcher's snapshot stays valid until a partial actually arrives. As soon as a partial lands, the projection changes and the applier refuses with `RepairError::SourceStatusDrift`.

### CLI surface

```text
plan-session-repair
  --contributor-state-dir <path>             required
  --session-id <hex64>                       required
  --status-report <path>                     EITHER this
  --build-status                             OR this (clap-enforced mutual exclusion)
  --strategy reannounce-missing              v1 only
  --coordinator-pubkey-hex <hex64>           optional operator hint
  --include-expired                          passed through when --build-status
  --no-prune-state-on-start                  Stage 12.7 inherited
  --out <path>                               required
```

```text
apply-session-repair
  --repair-plan <path>                       required
  --session-snip-root <0x...>                required
  --coordinator-seed <path>                  required
  --contributor-state-dir <path>             required (re-verify before publish)
  --snip-binary, --snip-seed                 SNIP plumbing
  --listen-port, --peer, --net-identity-file
  --peer-wait-secs, --mesh-stabilize-ms, --propagation-wait-ms
  --no-publish-announcements                 default false (matches Stage 12.8)
  --dry-run                                  optional
  --no-prune-state-on-start                  Stage 12.7 inherited
```

### Refused status branches

`plan-session-repair` refuses to emit actions for:

| Status | Error |
|---|---|
| `NoSession` | `RepairError::SessionNotPresent` |
| `NoAssignments` | `RepairError::NothingToRepair` |
| `CompletePartials` | `RepairError::NothingToRepair` |
| `Aggregated` | `RepairError::NothingToRepair` |
| `InvalidState` | `RepairError::InvalidState` (clean tampered artifacts first; no `--allow-invalid-state` flag) |
| `ExpiredIncomplete` | `RepairError::SessionExpired` (extend the session via `open-session` first if you mean it) |

Only `InProgress` produces a non-empty plan.

### `SessionRepairPlan` shape (local-only, unsigned)

```json
{
  "schema_version": 1,
  "session_id": "<hex64>",
  "source_status_hash": "<blake3-hex>",
  "strategy": "ReannounceMissing",
  "created_at_utc": "2026-06-01T00:30:00Z",
  "coordinator_pubkey_hex": "<hex64>",
  "actions": [
    {
      "ReannounceAssignment": {
        "assignment_id": "<hex64>",
        "stage_index": 1,
        "contributor_pubkey_hex": "<hex64>"
      }
    }
  ],
  "repair_plan_hash": "<blake3-hex>"
}
```

Actions sorted by `(stage_index ASC, assignment_id ASC)`. `RepairAction` is a closed Rust enum with one variant; `RepairStrategy` is a closed Rust enum with one variant. Both serialize externally-tagged so the introduction of `ReassignMissing` in Stage 12.11+ is observable from `schema_version` alone.

### Byte preservation on reannounce

The applier re-publishes the on-disk assignment JSON bytes verbatim. SNIP is content-addressed, so the returned root equals the original publish's root. The `assignment_id`, the coordinator signature, and the SNIP root are all preserved across reannounce. Only the network announcement is fresh (new `announced_at_utc` and new announcer signature). The aggregate verifier sees exactly one assignment per id — the same one it would have seen if the contributor had never lost the original announcement.

### Operator workflow

```bash
# 1. Inspect the session.
omni-node operator contributor session-status \
  --contributor-state-dir ./contrib-state \
  --session-id <hex64>
# → InProgress, 2 missing partials.

# 2. Plan the repair (no network, no SNIP).
omni-node operator contributor plan-session-repair \
  --contributor-state-dir ./contrib-state \
  --session-id <hex64> \
  --build-status \
  --out ./repair-plan.json

# 3. Review ./repair-plan.json. Optionally dry-run.
omni-node operator contributor apply-session-repair \
  --repair-plan ./repair-plan.json \
  --session-snip-root 0x... \
  --coordinator-seed ./coord.seed \
  --contributor-state-dir ./contrib-state \
  --dry-run

# 4. Apply.
omni-node operator contributor apply-session-repair \
  --repair-plan ./repair-plan.json \
  --session-snip-root 0x... \
  --coordinator-seed ./coord.seed \
  --contributor-state-dir ./contrib-state \
  --net-identity-file ./omni-net.key
```

### What 12.10 does NOT do

- Does NOT introduce a new signed envelope. `SessionRepairPlan` is unsigned.
- Does NOT add a new gossipsub topic, SNIP root format, or canonical-bytes domain separator.
- Does NOT change any Stage 12.0–12.9 schema or canonical bytes.
- Does NOT add a chain-side anything. The planner + applier are local infrastructure.
- Does NOT pull halo2/prover/framework runtimes into the default `omni-node` tree.
- Does NOT mutate the state-dir on apply.
- Does NOT include `ReassignMissing` or `AbandonLocal` actions in v1. Both are deferred.
- Does NOT include an `--allow-invalid-state` flag. Operators must clean tampered artifacts before repair.

### Test coverage map

| What's covered | Where |
|---|---|
| Planner refusal matrix (every `SessionOverallStatus`) | `repair_plan.rs` planner-refuses-* |
| Planner accepted path + deterministic action ordering | `planner_emits_one_reannounce_per_missing_partial`, `planner_deterministic_under_input_shuffle` |
| `repair_plan_hash` serde round-trip + hash drift | `repair_plan_hash_round_trips_through_json`, `apply_path_detects_plan_hash_drift` |
| `source_status_hash` projection stability + drift | `source_status_hash_ignores_*`, `source_status_hash_changes_when_partial_present_flips`, `source_status_hash_drifts_when_partial_arrives_after_planning` |
| Apply-time eligibility re-check (projection-unchanged status flips) | `apply_eligibility_check_catches_expired_*`, `apply_eligibility_check_catches_invalid_state_*` |
| Apply-time state-dir contract checks (missing + tampered assignments) | `apply_path_detects_missing_assignment_in_state`, `apply_path_detects_tampered_assignment_in_state` |
| End-to-end state-dir → status → plan | `end_to_end_state_dir_status_then_plan` |
| Halt-finding rationale (aggregate verifier coverage check) | `aggregate_verifier_rejects_extra_assignment_without_partial` |
| CLI flag matrix incl. `--build-status` / `--status-report` mutual exclusion | `session_repair_flag_parse_smoke` (omni-node) |
| Binary smoke: `plan-session-repair` end-to-end through clap + state-store + planner | Manual (`target/debug/omni-node operator contributor plan-session-repair ...`) |
| Binary smoke: `apply-session-repair --no-publish-announcements` fails fast (no peer-wait) | Manual (verified in PR review) |

**Residual gap (deliberate):** the apply path's final SNIP-republish + mesh-broadcast loop is not exercised in CI — it would require either spawning `sum-node` + a libp2p mesh, or refactoring `run_apply_session_repair` to be generic over `SnipV2Adapter + ContributorRelay` so a `MockSnipStore` + `InMemoryRelay` can be injected. The underlying primitives (`snip::publish_bytes`, `OmniNetRelay::publish_work_assigned`, `NetworkWorkAssignedAnnouncement` signing) are exercised by Stage 12.2 / 12.3 integration tests. Adding mocked end-to-end apply tests is a future refactoring opportunity that would also benefit Stage 12.8 `assign-session-plan`.
