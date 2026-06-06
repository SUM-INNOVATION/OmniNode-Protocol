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
3. Re-runs `check_repair_eligible` on the current status. The `source_status_hash` projection (`(session_id, sorted [(assignment_id, partial_present, superseded, superseded_by_supersession_id)] pairs)` as of Stage 12.11 — see the "`source_status_hash` projection" section below for the v1 → v2 lift) does NOT cover session-level health: a status flip from `InProgress` to `ExpiredIncomplete` (clock advanced past `session.expires_at_utc`) or `InvalidState` (an aggregate body added between plan and apply that failed the verifier) can leave the projection unchanged. The eligibility re-check catches that explicitly, using the same status→error matrix as the planner.
4. Re-fetches and re-verifies the session via `--session-snip-root` at apply time.
5. Re-verifies each referenced assignment from the state-dir via `verify_work_assignment` before any SNIP write.

The on-disk plan is treated as a **suggestion**, not a trust anchor.

### `--no-publish-announcements` is SNIP-only

When set, the applier skips the entire omni-net layer: no mesh open, no peer wait, no propagation sleep, no shutdown. The reannouncement is then a pure SNIP-republish loop (no observable latency tax for the 30s peer-wait default). With the flag unset (default), the apply opens the mesh, peer-waits, broadcasts a fresh `NetworkWorkAssignedAnnouncement` per action, sleeps for `--propagation-wait-ms`, and shuts down — matching Stage 12.8 `assign-session-plan`.

### `source_status_hash` projection

BLAKE3 over `(session_id, sorted [(assignment_id, partial_present, superseded, superseded_by_supersession_id)] pairs)`. Two status reports with different `generated_at_utc` or different `notes` but the same operator-meaningful shape produce the same projection — so a plan built from a long-running watcher's snapshot stays valid until a partial actually arrives **or** a supersession arrives that retires an assignment.

**Stage 12.11 lift:** prior to Stage 12.11 the projection only covered `(assignment_id, partial_present)`. A verified supersession arriving between plan and apply that retired a missing assignment would not flip any `partial_present` (the retired assignment never had a partial), so the projection stayed identical and the applier could reannounce or reassign an already-retired assignment. Stage 12.11 adds per-assignment `superseded` + `superseded_by_supersession_id` to the projection so any active-cover change trips drift, surfaced as `RepairError::SourceStatusDrift`. The projection still ignores `generated_at_utc`, `notes`, peer-advert state, aggregate state, and join/assignment counts that don't change shape.

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

---

# Stage 12.11 — signed assignment supersession

**Status**: shipped at Stage 12.11. Builds on Stage 12.7 state-dir + Stage 12.9 status reporter + Stage 12.10 repair planner. Introduces a **coordinator-signed** envelope (`WorkAssignmentSupersession`) that marks a set of assignments as retired and names their replacements — strictly **replacement-only in v1**. No chain wire / `InferenceAttestationDigest` / Stage 7b tx / SUM Chain RPC change. No proof-mode evidence change. No payment, reward, staking, slashing, marketplace, auction, bid, pricing, reputation, exclusive-claim, or lease change. SNIP remains the sole content-addressed surface.

## What changed

1. **New off-chain envelope.** `WorkAssignmentSupersession { schema_version: 1, session_id, supersession_id, superseded_assignment_ids, replacement_assignment_ids, reason, created_at_utc, coordinator_pubkey_hex, coordinator_signature_hex }`. Closed `SupersessionReason` enum: `MissingPartial | InvalidPartial | OperatorRebalance | Custom { label }`. Both ID lists are sorted ascending, unique, disjoint; `replacement_assignment_ids` is **non-empty** in v1 (review-mandated; the "operator abandons stage" happy path requires a separate aggregate-cancellation envelope that does not exist in v1, and is out of scope here).
2. **New gossipsub topic.** `omni/contributor/session/assignment-supersession/v1` carries the pointer-only `NetworkWorkAssignmentSupersessionAnnouncement`. Body bytes live in SNIP.
3. **Aggregate verifier extension.** `verify_aggregated_result_with_supersessions(session, joins, assignments, partials, supersessions, aggregate)`: union of every supersession's `superseded_assignment_ids` is excluded from the partial-coverage requirement. Sequential chains (A → B then B → C) accumulate `{A, B}` to superseded, `{C}` to active. The classic Stage 12.10 `verify_aggregated_result(...)` is now `verify_aggregated_result_with_supersessions(..., &[])` — bit-for-bit unchanged on zero supersessions.
4. **Stage 12.7 state-dir bump.** `STATE_VERSION = 2`. Adds `verified/sessions/<id>/supersessions/<supersession_id>.json` and `seen/assignment-supersessions/`. v1 stores are auto-migrated by extending the layout; existing artifacts are not rewritten.
5. **Stage 12.9 status report bump.** `STATUS_SCHEMA_VERSION = 2`. Adds `active_assignment_count`, `superseded_assignment_count`, `supersession_count`, `supersessions: Vec<SupersessionStatus>`, and `AssignmentStatus { superseded, superseded_by_supersession_id }`. `missing_assignment_ids` is now **active-only** (superseded assignments missing partials no longer block aggregation, mirroring the aggregate verifier's posture). The status reporter loads supersessions BEFORE partials so that a tampered partial under an `InvalidPartial` supersession does not flip `overall_status` to `InvalidState` — the assignment is already retired.
6. **Stage 12.10 repair planner bump.** `REPAIR_PLAN_SCHEMA_VERSION = 2`. Adds `RepairStrategy::ReassignMissing` and `RepairAction::ReassignAssignment { superseded_assignment_id, new_stage_index, new_contributor_pubkey_hex, new_work_kind, new_expected_work_units, new_expected_work_unit_kind, reason }`. The `ReannounceMissing` planner additionally **filters out** any active-missing assignment that is already named in a verified supersession (replacement-in-flight) — so `plan-session-repair` no longer racing-emits a reannounce while a reassignment is pending.

## CLI surface

Two new subcommands. Both live under `operator contributor`.

```text
plan-session-reassign
    --contributor-state-dir <path>
    --session-id <hex>
    (--build-status | --status-report <path>)
    [--reason <missing-partial | invalid-partial | operator-rebalance>]
    [--coordinator-pubkey-hex <hex>]
    [--include-expired]
    [--no-prune-state-on-start]
    --out <path>
```

Produces a `SessionRepairPlan` v2 with `strategy: ReassignMissing` and one `ReassignAssignment` action per active-missing assignment. Note: `--reason` only controls the reason copied into the planned supersession; `Custom { label }` is reachable only by hand-editing the JSON before apply (closed-enum, but operator-overridable).

```text
apply-session-reassign
    --reassignment-plan <path>
    --session-snip-root <hex>
    --coordinator-seed <path>
    --contributor-state-dir <path>
    [--no-prune-state-on-start]
    [--snip-binary <path>] [--snip-seed <path>]
    [--listen-port <u16>] [--peer <multiaddr>...]
    [--net-identity-file <path>]
    [--propagation-wait-ms <u64>] [--peer-wait-secs <u64>] [--mesh-stabilize-ms <u64>]
    [--no-publish-announcements]
    [--dry-run]
```

Validates the plan (refuses any non-`ReassignMissing` strategy and any non-`ReassignAssignment` action, refuses `InvalidState` per **v1 spec — InvalidState is for human triage, not automatic reassignment**), recomputes `source_status_hash` against the live state-dir and refuses on drift, re-runs `check_repair_eligible`, fetches + verifies the session via the supplied SNIP root, pins the coordinator seed to `session.coordinator_pubkey_hex`, loads + filters verified joins + assignments from the state-dir, builds + signs each replacement `WorkAssignment`, builds + signs **one** `WorkAssignmentSupersession` covering every superseded + replacement ID, and:

- on `--dry-run`: prints `event=would_reassign` per replacement plus `event=would_publish_supersession` and exits without writing.
- otherwise: **Phase A** — SNIP-publishes every body up-front (replacements first, then supersession) WITHOUT any state-dir mutation or mesh broadcast. SNIP is content-addressed and republish is idempotent, so a transient failure here is safe to retry and `source_status_hash` is unchanged. **Phase B** — once every SNIP body is durably content-addressed, mutates the state-dir + broadcasts on the mesh in publish order (replacements first, then supersession). The supersession state-dir write closes the loop; sleeps `--propagation-wait-ms`, shuts down the mesh.

Phase A / Phase B split was introduced in the Stage 12.11 review: prior to it, replacement assignments were dual-written to the state-dir immediately after each SNIP publish, BEFORE the supersession publish was even attempted. A failed supersession publish would leave extra active replacements in the state-dir without the retiring supersession — and a retry would then refuse on `source_status_hash` drift, forcing manual state cleanup. The two-phase ordering eliminates that window; the remaining window (one local FS op for the supersession state-dir write) is much narrower than the original SNIP round-trip window and matches the standard local-FS failure profile.

Mesh ordering still respects the invariant that replacements must be observable before any peer sees the supersession naming them, so a watcher receiving the supersession announcement can already fetch every replacement assignment body it names.

### Watcher integration

`watch-sessions` polls the new topic and routes verified announcements to `<out-dir>/<session_id>/supersessions/<supersession_id>.json` + the Stage 12.7 state-dir mirror. When the in-memory session cache and the state-dir's verified-assignment slice are both available, the watcher upgrades the processor to the full `verify_assignment_supersession` reference-resolution leg. Otherwise the announcement is accepted on its own (announcer sig + body schema + body sig + drift); the aggregate verifier re-checks references at aggregate time.

### Status report renderer

`events` adds: top-line `active=`, `superseded=`, `supersessions=`; per-assignment `superseded=yes|no` + `superseded_by=<id|->`; new `event=supersession_status` lines. `pretty` mirrors with new column + a supersessions table. `json` exposes everything through serde (no renderer change required; the schema bump is wire-visible).

## Out of scope (Stage 12.11)

- **Abandonment / cancellation.** v1 is replacement-only. A future stage will introduce a separate signed envelope (aggregate-cancellation) so an operator can mark a stage retired without naming a replacement. The empty-replacement-list path returns `SchemaError::SupersessionEmptyReplacement` today.
- **Work-kind-aware planner.** The Stage 12.10 reassignment planner copies the superseded assignment's `work_kind` and `expected_work_units` verbatim; live re-shaping (different stage range, different work-unit kind) is operator-policy and reachable by hand-editing the plan JSON. The planner intentionally does not encode shaping heuristics.
- **InvalidState auto-recovery.** `apply-session-reassign` refuses `InvalidState` (the `check_repair_eligible` gate). Recovery from a tampered partial that has not yet been superseded is operator manual: build a supersession with `reason = InvalidPartial`, sign it offline, and feed the SNIP root into `apply-session-reassign` via a hand-built plan. The automation lives at Stage 12.12+.
- **Chain wire / proof-mode evidence / payment / reward / staking / slashing / marketplace.** Unchanged. No `InferenceAttestationDigest` field touched. No A–F flow primitives reintroduced.

## Test coverage map

| What's covered | Where |
|---|---|
| Supersession body schema (incl. v1 empty-replacement refusal, disjointness) | `supersession_negatives.rs` |
| `verify_assignment_supersession` rejection matrix (schema / session binding / coord binding / ID derivation / sig / reference resolution) | `supersession_negatives.rs` |
| Aggregate verifier with supersessions (happy path, sequential chains, post-supersession-missing-partial) | `aggregate_supersession_integration.rs` |
| Aggregate verifier without supersessions remains bit-for-bit unchanged | Stage 12.10 `aggregate_verifier_*` (re-run under 12.11) |
| Status reporter v2 fields (active vs superseded counts, `superseded` flag) + tampered-partial-under-`InvalidPartial` not flipping `InvalidState` | `status_report.rs` (`tampered_partial_under_invalid_partial_supersession_is_not_invalid_state`) |
| Repair planner v2 (`ReassignMissing` strategy, `ReassignAssignment` action, in-flight supersession filter on `ReannounceMissing`) | `repair_plan.rs` |
| `process_assignment_supersession_announcement` (happy path with + without session context, drift on session_id, body-coord-pubkey drift under session) | `session_announcement_processing.rs` (`supersession_processor::*`) |
| Watch-sessions handler dispatch + state-dir dual-write + seen-marker | `omni_net::topics::all_topics_includes_every_public_constant` (topic count bump) + manual `watch-sessions` smoke |
| CLI flag matrix for `plan-session-reassign` + `apply-session-reassign` (mutual exclusion, `--reason` parse, `--no-publish-announcements`, `--dry-run`) | `session_reassign_flag_parse_smoke` (omni-node) |
| `source_status_hash` v2 projection lift — supersession arriving between plan and apply trips drift | `source_status_hash_drifts_when_supersession_arrives_after_planning` |

**Residual gap (deliberate):** the apply path's two-phase SNIP-republish + mesh-broadcast loop is not exercised in CI (same constraint as Stage 12.10's apply gap). The primitives are exercised by Stage 12.2 / 12.3 / 12.10 integration tests; the v1 invariants (replacement-only, single supersession covering all reassignments, publish ordering) are mechanically asserted by the verifier suite, the body schema, and the apply-time refusal arms.

---

# Stage 12.12 — Invalid-partial triage for supersession

**Status**: shipped at Stage 12.12. Builds on Stage 12.9 status reporter + Stage 12.11 supersession verifier + Stage 12.11 reassign CLI. Allows operators to use `apply-session-reassign --reason invalid-partial` to triage an `InvalidState` session when — and only when — every chain-link failure is a tampered partial (`SupersessionReason::InvalidPartial`) whose retiring assignment is named in the plan. Every other `InvalidState` cause (invalid session, invalid join, invalid assignment, invalid aggregate, invalid supersession, or even an invalid partial NOT in the plan) continues to refuse. No chain wire / Stage 7b tx / SUM Chain RPC / `InferenceAttestationDigest` change. No payment / reward / staking / slashing. No marketplace / auction / bid / pricing / reputation. No exclusive claim / lease. No proof mode / on-chain verification / chain proof allowlist. No A–F flow. SNIP only. No new gossipsub topic.

## What this stage is, in one sentence

Stage 12.11 introduced the `SupersessionReason::InvalidPartial` label so a coordinator can retire a tampered partial; Stage 12.12 introduces the **structured diagnostic surface** that tells an operator (and only the operator's local `apply-session-reassign`) which `InvalidState` sessions are safe to triage that way.

## Why structured diagnostics

Stage 12.9–12.11's `SessionStatusReport` had a single boolean (`any_chain_invalid`) driving `InvalidState`, with free-form strings appended to `notes`. Seven distinct failure modes — `verify_execution_session`, join, assignment, supersession, partial-orphan, partial-verify, aggregate — all routed through that one bit. A policy that reads "is this triagable?" has to know **which** failure(s) occurred; `notes` is documentation, not contract.

Stage 12.12 bumps `STATUS_SCHEMA_VERSION` 2 → 3 and adds one field:

```rust
pub invalid_artifacts: Vec<InvalidArtifactStatus>,
```

`InvalidArtifactStatus` is a closed externally-tagged enum (`kind` discriminator: `invalid_session` / `invalid_join` / `invalid_assignment` / `invalid_partial` / `invalid_supersession` / `invalid_aggregate`). Each variant carries the relevant id (`assignment_id`, `contributor_pubkey_hex`, `supersession_id`, or nothing for session / aggregate) plus a `reason_tag: String`.

**`reason_tag` is the stable string returned by `SessionVerifyOutcome::reason_tag()`.** It is NOT derived from `format!("{outcome:?}")`. Renaming a `SessionVerifyOutcome` variant must NOT silently change the wire-visible tag — the `session_verify_outcome_reason_tags_are_stable` test in `invalid_partial_triage.rs` pins every existing variant's tag.

### Special case: orphan partials

A partial body that references an `assignment_id` with no matching verified assignment surfaces as `InvalidPartial { assignment_id, reason_tag: "unmatched" }`. No separate `OrphanPartial` variant — the uniform shape lets `check_reassign_targets_active_missing` refuse via the existing `not_in_status` arm at apply time.

### Invariant

```
invalid_artifacts.is_empty()  <==>  overall_status != InvalidState
```

Pinned by `invalid_artifacts_is_empty_when_overall_status_is_not_invalid_state` and defended-in-depth by `check_reassign_eligible_allowing_invalid_partials` (refuses `invalid_state_without_diagnostics` when an attacker-supplied report violates it).

### `notes` is preserved alongside

`notes` strings are still populated at every site. Operator dashboards rendering free-form text keep working. **Automation must read `invalid_artifacts`** — `notes` content is documentation.

## Eligibility helper

```rust
pub fn check_reassign_eligible_allowing_invalid_partials(
    status: &SessionStatusReport,
    plan: &SessionRepairPlan,
) -> Result<(), RepairError>
```

Decision matrix:

| `status.overall_status` | Decision |
|---|---|
| `InProgress` | `Ok(())` |
| `NoSession` / `NoAssignments` / `CompletePartials` / `Aggregated` / `ExpiredIncomplete` | Forwarded to `check_repair_eligible` → unchanged typed error |
| `InvalidState`, `plan.strategy != ReassignMissing` | `RepairError::InvalidState` (e.g. `ReannounceMissing` never triages) |
| `InvalidState`, any entry in `invalid_artifacts` is not `InvalidPartial` | `RepairError::InvalidStateNotTriagable { kind: <invalid_*>, context: <id> }` |
| `InvalidState`, every `InvalidPartial.assignment_id` is in the plan's superseded set | `Ok(())` |
| `InvalidState`, an `InvalidPartial.assignment_id` is NOT in the plan | `RepairError::InvalidStateNotTriagable { kind: "invalid_partial_not_in_plan", context }` |
| `InvalidState`, `invalid_artifacts` is empty | `RepairError::InvalidStateNotTriagable { kind: "invalid_state_without_diagnostics", context: "" }` |

`MissingPartial` / `OperatorRebalance` reassigns + every `ReannounceMissing` reannounce continue to call `check_repair_eligible` directly and hard-refuse `InvalidState`. The Stage 12.12 split is **strictly** `--reason invalid-partial` plumbing.

## Planner change

`build_session_repair_plan_with_reason` branches on `(strategy, reason)`. Only `ReassignMissing + InvalidPartial` takes the relaxed path; it calls a dedicated **plan-time** precheck `check_invalid_partial_plan_eligible(status)` (mirror of the apply helper expressed against the implicit candidate target set the planner is about to emit) and then selects actions from `status.invalid_artifacts.iter().filter_map(InvalidPartial)` (filtering out superseded assignments and orphan-tagged entries). All other reason / strategy combinations keep the existing `check_repair_eligible` gate and the `partial_present == false && !superseded` selection.

The plan-time precheck enforces the **planner contract**: the planner emits a plan **if and only if** apply will accept it. Without this gate, a status with `InvalidPartial(A) + InvalidJoin(...)` would still produce a plan covering A, and the operator would only discover the refusal at `apply-session-reassign` after paying the cost of plan creation, review, and hash-pinning. The precheck returns the same typed `RepairError::InvalidStateNotTriagable { kind, context }` surface the apply helper uses, so scripts can write one rule across both layers.

## CLI changes

- **`plan-session-reassign --reason invalid-partial`** now emits actions for the active assignments whose status row carries an `InvalidPartial` diagnostic (no longer silently refused by the `InvalidState` gate).
- **`apply-session-reassign`** dispatches on `plan.actions[0].reason`:
  - `InvalidPartial` → `check_reassign_eligible_allowing_invalid_partials(&current_status, &plan)`
  - everything else → `check_repair_eligible(&current_status)`

### Apply-time mixed-reason defense

The planner emits a uniform `SupersessionReason` across one plan, but the plan is unsigned/local — hand-editing can produce mixed reasons that would smuggle one reason's relaxed gate onto another reason's actions. Stage 12.12 adds an early refusal:

```text
apply-session-reassign refuses mixed-reason plan: plan.actions[0].reason=InvalidPartial
but a later action has reason=MissingPartial
```

This check runs after the existing `ReassignAssignment`-only check and before any eligibility / drift / SNIP / mesh / state-dir work.

### Renderer additions

`events`: new `event=invalid_artifact session_id=... kind=... reason_tag=... [assignment_id=... | contributor_pubkey_hex=... | supersession_id=...]` lines, one per entry. `pretty`: new "Invalid artifacts" section under counts. `json`: unchanged (serde already exposes the field; v3 is additive).

## Out of scope (Stage 12.12)

- **Proof of partial wrongness.** Stage 12.12 trusts the coordinator's local triage label. No chain claim that the partial was objectively wrong.
- **Slashing, reputation, payment, chain reporting.** Untouched.
- **Automatic retry daemon.** Operator-initiated only.
- **Work-kind reshaping.** Replacement assignment copies `work_kind` and `expected_work_units` verbatim, same as Stage 12.11.
- **Abandonment / cancellation.** Still v1 replacement-only.
- **New gossipsub topic.** None added.
- **Chain wire / `InferenceAttestationDigest` / proof mode / chain proof allowlist / A–F flow.** Untouched.
- **`MissingPartial` / `OperatorRebalance` triage of `InvalidState`.** Refused — only `InvalidPartial` plus the tightly-scoped helper accepts.

## Test coverage map

| What's covered | Where |
|---|---|
| `SessionVerifyOutcome::reason_tag` stability across every variant | `session_verify_outcome_reason_tags_are_stable` (`invalid_partial_triage.rs`) |
| Status v3 emits `InvalidPartial` for a forged partial | `invalid_artifacts_emit_invalid_partial_for_tampered_partial` (`status_report.rs`) |
| Status v3 emits `InvalidJoin` for a forged join | `invalid_artifacts_emit_invalid_join_for_forged_join` |
| Status v3 emits `InvalidAssignment` for a tampered assignment | `invalid_artifacts_emit_invalid_assignment_for_tampered_assignment` |
| Status v3 emits `InvalidAggregate` for a tampered aggregate | `invalid_artifacts_emit_invalid_aggregate_for_aggregate_signed_by_wrong_coord` |
| Status v3 early-return for invalid session carries only `InvalidSession` | `invalid_artifacts_emit_invalid_session_returned_early` |
| Orphan partials emit `InvalidPartial { reason_tag: "unmatched" }` | `invalid_artifacts_emit_invalid_partial_unmatched_for_orphan_partial` |
| Reporter invariant — empty `invalid_artifacts` when not `InvalidState` | `invalid_artifacts_is_empty_when_overall_status_is_not_invalid_state` |
| Planner `--reason invalid-partial` selects only active assignments with `InvalidPartial` | `invalid_partial_plan_targets_only_assignments_with_invalid_partials` (`repair_plan.rs`) |
| Plan-time precheck refuses superseded `InvalidPartial` target (defense-in-depth) | `invalid_partial_planner_refuses_when_invalid_partial_targets_superseded_assignment` |
| Plan-time precheck refuses extra non-`InvalidPartial` artifacts (e.g. `InvalidJoin`) | `invalid_partial_planner_refuses_when_invalid_state_also_has_invalid_join` |
| Plan-time precheck refuses orphan `InvalidPartial { reason_tag: "unmatched" }` | `invalid_partial_planner_refuses_orphan_unmatched_invalid_partial` |
| `MissingPartial` / `OperatorRebalance` strategies still refuse `InvalidState` | `missing_partial_strategy_still_refuses_invalid_state` |
| `ReannounceMissing` never accepts `InvalidState` | `reannounce_missing_strategy_still_refuses_invalid_state` |
| Helper accepts when every `InvalidPartial` is targeted | `accepts_invalid_state_when_all_invalid_artifacts_are_planned_invalid_partials` (`invalid_partial_triage.rs`) |
| Helper refuses on additional `InvalidJoin` | `refuses_when_invalid_state_also_has_invalid_join` |
| Helper refuses on additional `InvalidAssignment` | `refuses_when_invalid_state_also_has_invalid_assignment` |
| Helper refuses on `InvalidAggregate` | `refuses_when_invalid_state_has_invalid_aggregate` |
| Helper refuses on `InvalidSupersession` | `refuses_when_invalid_state_has_invalid_supersession` |
| Helper refuses on `InvalidSession` | `refuses_when_invalid_state_has_invalid_session` |
| Helper refuses when an `InvalidPartial` is not in the plan | `refuses_when_invalid_partial_assignment_is_not_in_plan` |
| Helper refuses `ReannounceMissing` under `InvalidState` | `refuses_reannounce_missing_strategy_under_invalid_state` |
| Defense-in-depth: empty diagnostics array under `InvalidState` | `refuses_invalid_state_when_diagnostics_array_is_empty` |
| Helper forwards `Aggregated` / `ExpiredIncomplete` / `InProgress` to `check_repair_eligible` | `forwards_to_check_repair_eligible_for_non_invalid_state` |
| Apply-time mixed-reason defense | inline in `run_apply_session_reassign`; refuses before eligibility / drift / SNIP / mesh |

**Residual gap (deliberate):** the apply path's CLI dispatch + mixed-reason defense + dry-run + Phase A / Phase B publish under `--reason invalid-partial` is not exercised end-to-end in CI (same constraint as Stage 12.10 / 12.11 — would require mocked SNIP + InMemoryRelay end-to-end refactor). The library helper, the planner branch, and the structured reporter are mechanically tested; the CLI is a single dispatch line on top of the well-tested helper.

---

# Stage 12.13 — supersession-aware resume / watch hardening + audit ergonomics

**Status**: shipped at Stage 12.13. Builds on Stage 12.7 state-dir + Stage 12.11 supersession verifier + Stage 12.11 watch-sessions handler + Stage 12.11 apply-session-reassign Phase A/B + Stage 12.12 v3 status. Closes three runtime gaps and adds an operator-facing audit ergonomics layer — all **without** touching canonical bytes, gossipsub topics, or schema versions. No new envelope, no new wire format, no `schema_version` bump. Default `omni-node` tree still pulls zero halo2/prover/framework runtimes.

## What this stage is, in one sentence

Stage 12.11 / 12.12 shipped the supersession verifier, the reassign CLI, and the structured invalid-artifacts diagnostic surface; Stage 12.13 hardens the **consumers** of those facilities (watch-sessions restart preload, the watch-sessions supersession handler's out-of-order arrival path, the apply-session-reassign Phase B sequence, and the session-status renderer) and adds an audit ergonomics layer over the existing v3 report.

## Hardening 1 — restart preload across the supersession tree

Prior to Stage 12.13, `run_watch_sessions` preloaded ONLY `list_verified_sessions()` into the in-memory `sessions: HashMap` cache. Per-session joins, assignments, supersessions, partials, and aggregates were loaded lazily per announcement. Consequences:

- The watcher's supersession handler `handle_assignment_supersession` round-tripped through `state_store.list_verified_assignments_for(...)` on every single announcement — disk-cheap but not warmed.
- Cross-restart double-supersession detection at announcement time was absent.

Stage 12.13 adds a new library helper:

```rust
pub struct RestartSnapshot {
    pub sessions: HashMap<String, ExecutionSession>,
    pub joins_by_session: HashMap<String, Vec<ContributorJoin>>,
    pub assignments_by_session: HashMap<String, Vec<WorkAssignment>>,
    pub supersessions_by_session: HashMap<String, Vec<WorkAssignmentSupersession>>,
}

pub struct RestartReport {
    pub sessions_accepted: u32,
    pub sessions_rejected: u32,
    pub joins_accepted: u32,
    pub joins_rejected: u32,
    pub assignments_accepted: u32,
    pub assignments_rejected: u32,
    pub supersessions_accepted: u32,
    pub supersessions_rejected: u32,
    pub rejection_notes: Vec<String>,
}

pub fn load_verified_restart_snapshot(
    store: &ContributorStateStore,
) -> Result<(RestartSnapshot, RestartReport), StatusError>
```

Walks the state-dir in **dependency order** — sessions → joins → assignments → supersessions — and re-runs each Stage 12.3 / 12.11 verifier on the way (`verify_execution_session`, `verify_contributor_join`, `verify_work_assignment`, `verify_assignment_supersession`). The Stage 12.7 trust boundary stays in place: `list_verified_*` are still parse-only, and `load_verified_restart_snapshot` is the **caller** re-running the verifier. Corrupted local entries surface as structured rejection notes (`kind=… session_id=… id=… reason_tag=…`) rather than panicking or silently polluting the cache.

Aggregate re-verify is deliberately **skipped** at preload (Stage 12.13 decision 3) — the first `session-status` build re-runs `verify_aggregated_result_with_supersessions`. Partials are loaded lazily per-announcement (decision 5).

`run_watch_sessions` consumes the snapshot and warms three caches: `sessions`, `assignments_by_session`, and `supersessions_by_session` (decision 5; partials and aggregates remain lazy). One new bare-stdout line at startup:

```text
event=state_store_restart_loaded sessions_accepted=N sessions_rejected=N \
  joins_accepted=N joins_rejected=N \
  assignments_accepted=N assignments_rejected=N \
  supersessions_accepted=N supersessions_rejected=N
```

Followed by one `event=warn context=state_store_restart_load_rejected …` line per rejection note.

## Hardening 2 — out-of-order supersession handler

The Stage 12.11 supersession handler called `process_assignment_supersession_announcement` with `Some(session)` AND `Some(assignments)` — meaning the processor's full reference-resolution leg ran. If a supersession announcement arrived **before** every referenced replacement assignment was on disk (a race in gossipsub delivery between the assignment topic and the supersession topic), the verifier returned `SupersessionReferenceUnknown`, the processor classified that as `BodySchemaInvalid`, the handler logged an error and **dropped** the announcement.

Stage 12.13 reverses the two-pass strategy (Stage 12.13 decision 1):

1. **First pass** — call the processor with `Some(session), None` for assignments. This runs announcer-sig + body schema + body sig + drift + session pinning and returns the parsed body. Reference resolution is deferred.
2. **Second pass** — if the in-memory assignments cache covers every referenced `superseded_assignment_id` and `replacement_assignment_id`, the handler runs `verify_assignment_supersession` directly against the slice. Reference resolution completes; the watcher emits the existing `event=assignment_supersession ... references_resolved=true …` line.
3. **Otherwise** — the handler accepts the supersession on the first-pass checks alone and emits one new bare-stdout line:
   ```text
   event=supersession_partial_verify session_id=… supersession_id=… \
     unresolved_assignment_count=K
   ```

The `session-status` build at audit time re-runs the supersession-aware aggregate verifier against the full state-dir snapshot, so any reference that never resolves surfaces as a structured `invalid_supersession` diagnostic via Stage 12.12's `invalid_artifacts` field. The watcher's best-effort accept does NOT bypass body-sig — a forged supersession is still rejected (pinned by `supersession_with_forged_body_signature_rejected_even_in_best_effort_pass`).

The watcher also keeps the in-memory caches current: each `handle_assignment` success updates `assignments_by_session`, and each `handle_assignment_supersession` success appends to `supersessions_by_session`. Same-poll-cycle ordering is preserved.

## Hardening 3 — apply-session-reassign Phase B split

The Stage 12.11 round-1 fix made Phase A SNIP-publish every body before any local side effects. The Stage 12.11 round-1 review left a narrower window inside Phase B: state-dir writes were interleaved with mesh broadcasts per replacement, so a failure mid-loop left the state-dir half-applied.

Stage 12.13 splits Phase B into four sub-phases (decision 4):

- **B1** — write every replacement assignment to the state-dir + mark seen.
- **B2** — write the supersession to the state-dir + mark seen.
- **B3** — broadcast each replacement on the mesh. Failures emit `event=warn context=replacement_assignment_broadcast …` and the loop **continues**; the state-dir is already coherent.
- **B4** — broadcast the supersession on the mesh **only if** every B3 broadcast succeeded. Otherwise emit `event=warn context=supersession_broadcast_skipped …` and skip B4 so the "replacements observable on mesh before supersession" invariant is preserved.

The trailing `event=reassign_applied` line gains two new fields:

```text
event=reassign_applied session_id=… replacements_published=N \
  supersession_published=1 supersession_broadcast=<status> \
  replacement_broadcast_failures=N
```

`supersession_broadcast` ∈ `{broadcast_ok, broadcast_failed, skipped_replacement_broadcast_failed, skipped_no_publish}`. The closed status set lets log scrapers branch deterministically.

**Exit code policy** (Stage 12.13 review fix):

| `supersession_broadcast` | Process exit | Rationale |
|---|---|---|
| `broadcast_ok` | `0` | Mesh state coherent. |
| `skipped_no_publish` | `0` | Operator opted out of mesh via `--no-publish-announcements`. |
| `skipped_replacement_broadcast_failed` | nonzero | B3 partially broadcast replacements; B4 was skipped to preserve the "replacements observable before supersession" invariant. Mesh state is incoherent — operator must triage. State-dir is coherent; safe to re-run apply. |
| `broadcast_failed` | nonzero | All B3 broadcasts succeeded but B4 itself errored; replacements are visible on mesh without the retiring supersession. Operator re-runs apply or manually rebroadcasts. |

**What this closes**: a Phase B failure no longer requires manual state-dir cleanup unless both B1 AND B2 wrote partially (the only remaining trap is the supersession state-dir write — a single FS op). Mesh-broadcast failures degrade gracefully on the state-dir side but the process now exits **nonzero** when peers see a half-complete view, so automation can react.

## Audit ergonomics — `session-status --format events | pretty`

New derived helpers (no JSON schema change; v3 is frozen at Stage 12.12):

```rust
pub enum AuditCoherence {
    Coherent,
    PartialApplySupersession { supersession_id: String, unresolved_count: u32 },
    OrphanReplacementAssignments { assignment_ids: Vec<String> },
    NotReassignTriagable,
    ReassignTriagable,
}

pub struct AuditHealth {
    pub coherence: AuditCoherence,
    pub triagable_by_reassign: bool,
    pub recommended_action: &'static str,
}

pub fn compute_audit_health(report: &SessionStatusReport) -> AuditHealth;
```

`recommended_action` is a closed set of static strings:

- `"none"`
- `"run plan-session-reassign --reason missing-partial"`
- `"run plan-session-reassign --reason invalid-partial"`
- `"clean state-dir orphan replacements before retry"`
- `"operator triage required"`

`compute_audit_health` is a pure projection over the v3 report. The `events` renderer adds one new line at the end of every report:

```text
event=audit_health session_id=… coherence=<tag> \
  triagable_by_reassign=<bool> recommended_action="<closed-string>"
```

The `pretty` renderer adds an "Audit health" section under "Notes:" with coherence, triagable_by_reassign, and recommended_action. JSON is **unchanged** — `--format json` still emits exactly the same v3 `SessionStatusReport`.

## Out of scope (Stage 12.13)

- No new envelope, no new canonical bytes, no `schema_version` bump anywhere.
- No new gossipsub topic.
- No SNIP wire change.
- No automated state-dir cleanup tool — operators delete orphan replacements manually after consulting the audit_health roll-up.
- No retry daemon.
- No batched-atomic state-dir write (per-file tempfile+rename atomicity stays; batched atomicity is OS-dependent).
- No standalone `session-audit` subcommand — audit lives in the existing `session-status` renderers.
- Aggregate re-verify is NOT part of the restart preload — the first `session-status` build re-runs it.

## Test coverage map

| What's covered | Where |
|---|---|
| `load_verified_restart_snapshot` happy path + assignments cache shape | `restart_preload_loads_assignments_and_caches_them` (`state_resume_integration.rs`) |
| Status report bit-equal across restart on the same state-dir | `restart_after_supersession_yields_same_status_report` |
| Restart preload drops forged supersession with stable `reason_tag` note | `restart_preload_drops_corrupted_supersession_with_note` |
| Phase B partial-apply causes `source_status_hash` drift on retry | `phase_b_partial_apply_state_drifts_source_status_hash` (`phase_b_partial_apply.rs`) |
| Audit detects orphan replacement when supersession is invalid (Phase B2 corrupted) | `audit_detects_orphan_replacement_assignment` |
| Audit detects B1-only orphan replacement via duplicate active stage_index (no supersession file) | `audit_detects_b1_only_orphan_via_duplicate_active_stage_index` |
| Audit does NOT flag a properly-superseded original at the same stage as its replacement | `audit_does_not_flag_duplicate_when_one_assignment_is_superseded` |
| Audit detects unresolved supersession references (best-effort accept) | `audit_detects_partial_apply_supersession_with_unresolved_reference` |
| Audit reports `Coherent` on the happy path | `complete_apply_audits_coherent` |
| Audit reports `ReassignTriagable` when every InvalidState entry is `InvalidPartial` | `audit_reports_reassign_triagable_when_only_invalid_partial_entries` |
| Audit reports `NotReassignTriagable` when any non-`InvalidPartial` entry exists | `audit_reports_not_reassign_triagable_when_invalid_join_present` |
| Out-of-order supersession with missing reference → best-effort first pass | `supersession_with_missing_reference_accepts_via_best_effort_first_pass` (`watch_session_supersession_out_of_order.rs`) |
| Full verification path with complete slice | `supersession_with_complete_assignment_slice_runs_full_verification` |
| No-session best-effort accept | `supersession_with_no_session_pinning_accepts_via_best_effort` |
| Forged body signature still rejected in best-effort pass | `supersession_with_forged_body_signature_rejected_even_in_best_effort_pass` |

**Residual gap (deliberate):** the Phase B reorder + the watcher restart preload are still not exercised through the live CLI end-to-end (same Stage 12.10/11/12 CI constraint — would require mocked SNIP + InMemoryRelay full apply). The library helpers and the projection-only audit are mechanically tested; the CLI wires them in single-call sites that are independently visible in code review.

---

# Stage 12.14 — local session archival + state-dir compaction

**Status**: shipped at Stage 12.14. Builds on Stage 12.7 state-dir + Stage 12.12 v3 status + Stage 12.13 audit ergonomics. Adds operator lifecycle management for completed / aggregated / expired session subtrees: an explicit `archive-session` CLI that copies (or moves) the `verified/sessions/<session_id>/...` subtree + matching seen markers to an operator-chosen archive directory, BLAKE3-verifies every copy, and writes a manifest LAST so a partial copy is detectable without source corruption. No new protocol envelope. No canonical-byte changes. No `schema_version` bump on any in-tree envelope. No `STATE_VERSION` bump. No new gossipsub topic. No SNIP wire change. Default `omni-node` tree still pulls zero halo2/prover/framework runtimes.

## Why archival is purely local

Stage 12.7 made watcher state persistent; Stages 12.11–12.13 made the supersession / reassign / restart flows robust. The remaining operational gap is **disk growth**: completed, aggregated, expired, or abandoned sessions accumulate `verified/sessions/<id>/...` subtrees indefinitely unless the operator manually prunes them. `prune_expired` (Stage 12.7) handles expiry — but `Aggregated` sessions whose `expires_at_utc` is far in the future also pile up.

Stage 12.14 addresses this with a strictly local tool. Archives are **inert JSON files** — not signed, not gossiped, not chain-bound, not SNIP-bound. The archive manifest is a new local-only document that lives in `<archive-dir>`, NOT inside `<state-dir>`. Consequently `STATE_VERSION` stays at `2`.

## Library surface

```rust
pub const ARCHIVE_MANIFEST_SCHEMA_VERSION: u32 = 1;

pub enum ArchiveMode { Copy, Move }

pub enum ArchiveStatusRequirement {
    Any,
    /// Aggregated OR CompletePartials. The safe default.
    Complete,
    Aggregated,
    CompletePartials,
    ExpiredIncomplete,
}

pub struct ArchivedFile {
    pub source_relative: String,
    pub archive_relative: String,
    pub blake3_hex: String,
    pub bytes: u64,
}

pub struct ArchiveManifest {
    pub schema_version: u32,
    pub session_id: String,
    pub generated_at_utc: String,
    pub source_state_version: u32,
    pub omni_contributor_version: String,
    pub session_overall_status: String,   // Debug-stringified
    pub audit_coherence: String,          // Debug-stringified
    pub mode: ArchiveMode,
    pub require_status: ArchiveStatusRequirement,
    pub include_results: bool,
    pub files: Vec<ArchivedFile>,
}

pub struct ArchiveOptions<'a> {
    pub session_id: &'a str,
    pub archive_dir: &'a Path,
    pub mode: ArchiveMode,
    pub require_status: ArchiveStatusRequirement,
    pub include_results: bool,
    pub now_utc: &'a str,
    pub dry_run: bool,
}

pub fn archive_session(
    store: &ContributorStateStore,
    opts: &ArchiveOptions<'_>,
) -> Result<ArchiveManifest, ArchiveError>;
```

Also new: `ContributorStateStore::cascade_remove_session(&self, session_id)` — a **strict** public method. Stage 12.7's `prune_expired` continues to use a `cascade_remove_session_best_effort` private variant that swallows IO errors (matching its documented conservative posture — auto-prune shouldn't abort because one marker is unreadable). The Stage 12.14 review surfaced a real gap with a single shared cascade: operator-driven `--move` archival exposes the cascade outcome via the process exit code, so silent best-effort behavior would surface as `event=archive_complete` + zero exit while state-dir remnants remain. The strict variant therefore propagates every IO error other than `NotFound` as `StateError::Io { path, source }`; `NotFound` is benign so a partial-cascade retry idempotently re-runs to convergence.

## Safety contract

- **`--dry-run`** returns a fully-populated manifest without touching the filesystem. No archive dir written; no source removed.
- **`--copy`** (default) copies + BLAKE3-verifies; the source state-dir is untouched. The destination must NOT already contain `<archive-dir>/<session_id>/` — the call refuses with `ArchiveError::ArchiveAlreadyExists`.
- **`--move`** copies + BLAKE3-verifies + writes the manifest, THEN runs `ContributorStateStore::cascade_remove_session`. A partial-copy failure leaves the source intact (cascade never runs unless every byte landed and verified).
- **BLAKE3 mismatch** is **fail-fast**: no retry, no partial accept. Operator triages the FS / hardware and re-runs.

## Status policy

| `--require-status` | Accepts |
|---|---|
| `any` | every `SessionOverallStatus` (escape valve for InvalidState triage) |
| `complete` (default) | `Aggregated` OR `CompletePartials` |
| `aggregated` | only `Aggregated` |
| `complete-partials` | only `CompletePartials` |
| `expired-incomplete` | only `ExpiredIncomplete` |

`InProgress`, `InvalidState`, `NoAssignments`, and `NoSession` are refused by every requirement except `any`. Stage 12.14 intentionally does NOT add a `--require-status invalid-state` variant — operators must explicitly opt into `--require-status any` to archive a chain-failed session, signaling they understand the artifacts will not satisfy a reload-and-status pipeline.

## File walk

`archive_session` enumerates two layers:

1. **`verified/sessions/<session_id>/...`** — the full subtree, walked recursively. Stage 12.11 supersessions and Stage 12.5 peer-adverts are picked up naturally because they live under this root.
2. **Session-keyed seen markers** — `seen/sessions/<id>`, `seen/aggregates/<id>`, plus every `seen/joins/<id>--*`, `seen/assignments/<id>--*`, `seen/partials/<id>--*`, `seen/peer-adverts/<id>--*`, and `seen/assignment-supersessions/<id>--*` whose key starts with `<id>--`. This matches the existing Stage 12.7 prune-cascade definition.

`--include-results` additionally copies `results/result-links/<session.posted_id>.link.json` when present. Stage 12.14 leaves `results/contributor-results/<job_id>.json` alone by default — those are keyed by `job_id`, and the session→job mapping is not directly carried by the state-dir.

The top-level `seen/posted-jobs/`, `seen/network-job-announcements/`, `seen/network-result-announcements/` namespaces are NOT archived: they are operator-job-scoped, not session-scoped.

## CLI

```text
omni-node operator contributor archive-session
    --contributor-state-dir <path>
    --session-id <hex>
    --archive-dir <path>
    [--require-status any | complete | aggregated | complete-partials | expired-incomplete]
                                            # default: complete
    [--copy | --move]                       # default: copy
    [--include-results]                     # default: off
    [--dry-run]
    [--no-prune-state-on-start]
```

Closed-set bare-stdout events:

```text
event=archive_started session_id=... mode=<copy|move> \
  require_status=<any|complete|aggregated|complete_partials|expired_incomplete> \
  include_results=<bool> dry_run=<bool>

event=archive_file session_id=... source_relative=... archive_relative=... \
  blake3_hex=... bytes=...

event=archive_complete session_id=... files=N total_bytes=B mode=<copy|move> \
  manifest=<path>

event=would_archive_file ...
event=would_archive_complete session_id=... files=N total_bytes=B mode=<copy|move>
```

Any `ArchiveError` from the library — `SessionNotPresent` / `StatusRequirementUnmet` / `ArchiveAlreadyExists` / `BlakeMismatch` / `Io` / `Status` / `State` — produces a nonzero exit code so automation can detect refusals.

## Audit ergonomics

Stage 12.13's `compute_audit_health` `recommended_action` closed set grows from 5 to 7 strings:

- `"run archive-session --require-status aggregated"` — when `overall_status == Aggregated` and `coherence == Coherent`.
- `"run archive-session --require-status expired-incomplete"` — when `overall_status == ExpiredIncomplete` and `coherence == Coherent`.

`CompletePartials` is **deliberately** NOT recommended for archival — the operator may still want the aggregate to land. `InProgress` continues to recommend `plan-session-reassign --reason missing-partial` when there's an active-missing entry; otherwise `"none"`.

JSON renderer is unchanged. The audit recommendation lives in the `events` line and the `pretty` section; v3 `SessionStatusReport` JSON schema stays frozen.

## Out of scope (Stage 12.14)

- **No restore-from-archive command.** Archives are inspectable JSON files; a future stage can add `restore-session` if useful. Manual restore today is `cp -r <archive-dir>/<session_id>/verified/sessions/<id>/ <state-dir>/verified/sessions/<id>/`.
- **No remote backup destination.** `<archive-dir>` is local; operators can `rsync` / `s3 sync` on top.
- **No bulk-archive command.** Operators loop in shell; Stage 12.15 can revisit.
- **No `results/contributor-results/*.json` archival by default.** The mapping is per-job, not per-session.
- **No retry daemon. No automatic periodic compaction.**
- **No new envelope. No canonical-byte changes. No `schema_version` bump on any in-tree envelope. No `STATE_VERSION` bump. No new gossipsub topic. No SNIP wire change.**
- No chain / proof / payment / marketplace / staking / slashing / lease / reputation surfaces.

## Test coverage map

| What's covered | Where |
|---|---|
| `--dry-run` writes nothing | `dry_run_archives_nothing` (`archive_session_integration.rs`) |
| `--copy` writes manifest + verified BLAKE3 | `copy_writes_manifest_and_verifies_blake3` |
| `--copy` preserves source bit-for-bit | `copy_preserves_source_state_dir` |
| `--move` removes session subtree + matching seen markers | `move_removes_source_session_subtree_and_seen_markers` |
| `--require-status complete` refuses `InProgress` | `refuses_in_progress_when_require_status_complete` |
| `--require-status any` accepts `InvalidState` | `require_status_any_accepts_invalid_state` |
| Missing session refusal | `refuses_when_session_not_in_state_dir` |
| Existing archive dir refusal | `refuses_when_archive_dir_already_has_this_session` |
| `--include-results` copies only `result-links/`, not `contributor-results/` | `include_results_copies_posted_result_link_only` |
| Other sessions untouched by archive of a single session | `does_not_touch_other_sessions` |
| Manifest serde round-trip | `manifest_serde_roundtrip` |
| `ExpiredIncomplete` accepted under `expired-incomplete` requirement | `expired_incomplete_accepted_under_expired_incomplete_requirement` |
| Stage 12.11 supersession files + seen markers archived | `supersession_files_and_seen_markers_are_archived` |
| Public `ContributorStateStore::cascade_remove_session` matches Stage 12.7 prune cascade on the happy path | `cascade_remove_session_public_method_matches_prune_cascade` (`state_store_unit.rs`) |
| Strict cascade treats `NotFound` as benign (idempotent retry) | `cascade_remove_session_strict_accepts_missing_markers_as_benign` |
| Strict cascade propagates non-`NotFound` IO errors as `StateError::Io` | `cascade_remove_session_strict_propagates_non_notfound_errors` |
| `prune_expired` keeps its best-effort posture under the same failure | `prune_expired_keeps_best_effort_posture_under_same_failure` |
| Audit recommends archive for `Coherent + Aggregated` | `audit_recommends_archive_for_coherent_aggregated` (`phase_b_partial_apply.rs`) |
| Audit recommends archive for `Coherent + ExpiredIncomplete` | `audit_recommends_archive_for_coherent_expired_incomplete` |
| Audit does NOT recommend archive for `CompletePartials` | `audit_does_not_recommend_archive_for_complete_partials` |
| CLI flag matrix (`--require-status` parse, `--copy` xor `--move`, `--include-results`, `--dry-run`) | `archive_session_flag_parse_smoke` (omni-node) |

**Residual gap (deliberate):** the CLI run-fn `run_archive_session` is not exercised end-to-end (same constraint as Stage 12.10–12.13 — would require a CLI-process smoke harness). The library `archive_session` is exhaustively covered; the CLI is a thin dispatch on top.

---

# Stage 12.15 — local session archive restore / import validation

**Status**: shipped at Stage 12.15. Builds on Stage 12.14's archive layout + Stage 12.13's restart preload. Adds the **inverse** of `archive-session`: read `<archive-session-dir>/manifest.json`, validate every file against the manifest's BLAKE3 + path whitelist + state-dir version compatibility, then write each archived byte back into the operator's state-dir. **No new protocol envelope. No canonical-byte changes. No `schema_version` bump on any in-tree envelope. No `STATE_VERSION` bump. No `ARCHIVE_MANIFEST_SCHEMA_VERSION` bump. No new gossipsub topic. No SNIP wire change.** Default `omni-node` tree still pulls zero halo2/prover/framework runtimes.

## Why restore is byte-identical replay

Stage 12.14's archive captured per-file BLAKE3 + raw bytes. Stage 12.15 restore writes those bytes back at the same `source_relative` paths. Because every Stage 12.0–12.11 envelope is signed over canonical bytes that are frozen per `schema_version`, the restored bodies verify against `verify_execution_session` / `verify_contributor_join` / `verify_work_assignment` / `verify_assignment_supersession` exactly as if they had never left the state-dir. **No "restored" provenance flag is needed in the state-dir** — and adding one would change the verified-envelope contract.

## Library surface

```rust
pub enum RestoreSource<'a> {
    SessionDir(&'a Path),
    ArchiveRoot { archive_dir: &'a Path, session_id: &'a str },
}

pub struct RestoreOptions<'a> {
    pub source: RestoreSource<'a>,
    pub dry_run: bool,
    pub verify_only: bool,
    pub overwrite_existing: bool,
    pub include_results: bool,
    pub now_utc: &'a str,
}

pub struct RestoreReport {
    pub session_id: String,
    pub manifest_schema_version: u32,
    pub source_state_version: u32,
    pub files_restored: u32,
    pub files_skipped_results: u32,
    pub bytes_restored: u64,
    pub mode: &'static str,             // "restore" | "dry_run" | "verify_only"
    pub manifest_session_overall_status: String,
    pub manifest_audit_coherence: String,
}

pub fn verify_archive_manifest(
    source: &RestoreSource<'_>,
) -> Result<ArchiveManifest, RestoreError>;

pub fn restore_session_archive(
    store: &ContributorStateStore,
    opts: &RestoreOptions<'_>,
) -> Result<RestoreReport, RestoreError>;
```

Also new on the state store:

```rust
impl ContributorStateStore {
    pub fn write_archived_bytes(
        &self,
        source_relative: &str,
        bytes: &[u8],
        overwrite_existing: bool,
    ) -> Result<PathBuf, StateError>;
}
```

`write_archived_bytes` is the **single FS gate**: every byte that lands in the state-dir during restore goes through it. The function enforces:

1. UTF-8 relative path (no `..`, no absolute path, no backslash).
2. Whitelist match against the Stage 12.14 archive layout (`verified/sessions/<64-hex>/...` / `seen/sessions/<64-hex>` / `seen/aggregates/<64-hex>` / `seen/{joins,assignments,partials,peer-adverts,assignment-supersessions}/<64-hex>--*` / `results/result-links/<64-hex>.link.json`).
3. Pre-existing destination refuses unless `overwrite_existing == true`.
4. Atomic tempfile+rename via the existing Stage 12.7 helper.

Path violations surface as new typed `StateError::UnsafeRelativePath` / `StateError::DisallowedRelativePath` / `StateError::DestinationExists` variants; the restore module bubbles them up as the matching `RestoreError` variants for operator clarity.

## Safety contract (mode matrix)

| Mode | Manifest parse + schema/version/session_id check | Per-entry path safety | Read archive file + BLAKE3 verify | Destination existence preflight | Write destination |
|---|:---:|:---:|:---:|:---:|:---:|
| `--dry-run` | ✓ | ✓ | — | — | — |
| `--verify-only` | ✓ | ✓ | ✓ | — | — |
| (real restore) | ✓ | ✓ | ✓ | ✓ | ✓ |
| `--dry-run` + `--verify-only` | ✓ | ✓ | ✓ | — | — (verify-only wins) |

- **BLAKE3 mismatch** is **fail-fast** — no retry, no partial accept.
- **Existing-destination preflight** is **all-or-nothing**: if any destination file already exists and `--overwrite-existing` is false, the call refuses BEFORE writing anything. With `--overwrite-existing`, every existing destination is replaced.
- **`source_state_version`** is **strict equality only** in v1: `manifest.source_state_version == STATE_VERSION`. Anything else refuses with `RestoreError::IncompatibleSourceStateVersion`. A future stage can add a migration story when there's a `STATE_VERSION` bump.
- **`session_id` binding**: the manifest's `session_id` must equal the resolved expected id (the trailing path component when `RestoreSource::SessionDir` is supplied; the caller-supplied id when `RestoreSource::ArchiveRoot` is used). Defends against hand-renamed archive directories.

## CLI

```text
omni-node operator contributor restore-session-archive
    --contributor-state-dir <path>
    (--archive-session-dir <path> | --archive-dir <root> --session-id <hex>)
    [--dry-run]
    [--verify-only]
    [--overwrite-existing]
    [--include-results]
    [--no-prune-state-on-start]
```

`--archive-session-dir` and `--archive-dir + --session-id` are mutually exclusive (`conflicts_with_all` on the former; `requires` paired on the latter). `--dry-run` and `--verify-only` are NOT mutually exclusive — `--verify-only` is the strict superset, so passing both behaves as verify-only.

Closed-set bare-stdout events:

```text
event=restore_started session_id=... mode=<restore|dry_run|verify_only> \
  archive_dir=... overwrite_existing=<bool> include_results=<bool> \
  session_overall_status=<closed> audit_coherence=<closed>

event=restore_file session_id=... source_relative=... blake3_hex=... \
  bytes=... destination=...

event=verify_only_file ...           # verify-only mode only
event=would_restore_file ...         # dry-run mode only
event=restore_skipped_result_link session_id=... source_relative=... \
  reason=include_results_off

event=restore_complete session_id=... mode=<...> files=N \
  files_skipped_results=K bytes=B archive_dir=...
```

Any `RestoreError` exits nonzero so automation can detect refusals.

## Post-restore workflow

Stage 12.15 does **not** auto-run `session-status` after restore. Operators chain themselves:

```bash
omni-node operator contributor restore-session-archive \
  --contributor-state-dir <state> \
  --archive-session-dir /backup/contributor-archive/<session_id> \
  && omni-node operator contributor session-status \
       --contributor-state-dir <state> --session-id <session_id>
```

The runbook documents the chained pattern + the "stop the watcher before restoring" caveat (concurrent state-dir writes are Stage 12.7's operator policy).

## Out of scope (Stage 12.15)

- No restore from remote URLs or stdin — `<archive-dir>` is local.
- No bulk-restore command — operators loop in shell.
- No automatic chain reconciliation after restore.
- No `--move` restore variant — v1 is copy-only; archive stays intact.
- No restore-into-a-locked-watcher concurrency protection.
- No `STATE_VERSION` migration path — strict equality only.

## Test coverage map

| What's covered | Where |
|---|---|
| `--dry-run` writes nothing | `dry_run_validates_manifest_without_touching_state_dir` (`restore_session_archive.rs`) |
| `--verify-only` hashes intact archive without writing | `verify_only_hashes_intact_archive_without_writing` |
| Round-trip is byte-for-byte equal to archive | `restore_round_trips_bytes_byte_for_byte` |
| Restore after `archive --move` recreates state subtree | `restore_after_archive_move_recreates_full_state_subtree` |
| Stage 12.13 restart preload accepts restored session (zero rejections) | `restored_session_is_accepted_by_load_verified_restart_snapshot` |
| BLAKE3 mismatch refusal | `refuses_when_archive_file_blake3_mismatches_manifest` |
| Missing archive file refusal | `refuses_when_manifest_references_missing_archive_file` |
| Path traversal refusal | `refuses_path_traversal_in_manifest_entry` |
| Absolute path refusal | `refuses_absolute_path_in_manifest_entry` |
| Disallowed path-outside-whitelist refusal | `refuses_path_outside_session_whitelist` |
| Existing destination refuses BEFORE any write (all-or-nothing preflight) | `refuses_when_any_destination_exists_without_overwrite_existing` |
| `--overwrite-existing` replaces files | `overwrite_existing_replaces_destination_files` |
| Result link skipped unless `--include-results` | `result_link_skipped_unless_include_results_true` |
| Result link restored when `--include-results` | `result_link_restored_when_include_results_true` |
| Incompatible `source_state_version` refusal | `refuses_incompatible_source_state_version` |
| Unsupported manifest `schema_version` refusal | `refuses_unsupported_manifest_schema_version` |
| Session-id mismatch refusal (renamed archive dir) | `refuses_when_manifest_session_id_mismatches_supplied_session_id` |
| `verify_archive_manifest` helper returns typed manifest for valid archive | `verify_archive_manifest_returns_typed_manifest_for_valid_archive` |
| `write_archived_bytes` refuses path traversal | `write_archived_bytes_refuses_path_traversal` (`state_store_unit.rs`) |
| `write_archived_bytes` refuses absolute path | `write_archived_bytes_refuses_absolute_path` |
| `write_archived_bytes` refuses backslash separators | `write_archived_bytes_refuses_backslash` |
| `write_archived_bytes` refuses disallowed prefix | `write_archived_bytes_refuses_disallowed_prefix` |
| `write_archived_bytes` accepts every Stage 12.14 archive prefix | `write_archived_bytes_accepts_whitelisted_prefixes` |
| `write_archived_bytes` refuses existing destination unless `overwrite_existing` | `write_archived_bytes_refuses_existing_destination_unless_overwrite` |
| CLI flag matrix (`--archive-session-dir` xor `--archive-dir + --session-id`, `--dry-run`/`--verify-only` toggles, `--overwrite-existing`, `--include-results`) | `restore_session_archive_flag_parse_smoke` (omni-node) |
| `--dry-run --verify-only` still catches BLAKE3 mismatch (review fix — verify-only wins) | `dry_run_plus_verify_only_still_catches_blake3_mismatch` |
| `--dry-run --verify-only` still catches missing archive file (review fix) | `dry_run_plus_verify_only_still_catches_missing_archive_file` |
| `--dry-run --verify-only` happy path: BLAKE3 walk runs, no destination writes, mode=verify_only | `dry_run_plus_verify_only_accepts_intact_archive_and_writes_nothing` |

**Residual gap (deliberate):** the CLI run-fn `run_restore_session_archive` is not exercised end-to-end (same constraint as Stage 12.10–12.14). The library `restore_session_archive` is exhaustively covered; the CLI is a thin dispatch + event-emission layer on top.

# Stage 12.16 — local state-dir integrity scan + repair suggestions

## What this stage is, in one sentence

A purely read-only scanner that re-runs every Stage 12.3 / 12.11 verifier against the bodies on disk, walks seen markers ↔ verified bodies, reports stray files inside documented subtrees, rolls up the Stage 12.13 audit projection per session, and (optionally) walks a parallel archive directory via Stage 12.15 `restore_session_archive(verify_only=true, dry_run=true)` — emitting one typed `IntegrityFinding` per structural anomaly. The scanner **never writes** to the state-dir or archive directory; v1 emits *suggested* repair commands as closed-set labels but **never executes them**.

## Why integrity scans live outside the protocol surface

A contributor's state-dir is a local cache. Bytes drift from disk corruption, partial restores, half-applied repairs, or operator mistakes — none of which touch a protocol envelope. Stage 12.16 is therefore zero-protocol: no new schema_version, no new omni-net topic, no new SNIP wire, no chain interaction. It is a local diagnostics surface that composes existing parse-only loaders, verifier functions, Stage 12.13 restart-preload, Stage 12.13 audit health, and the Stage 12.15 archive-verify path.

## Library surface

`omni_contributor::integrity::scan_state_integrity(store, opts) -> Result<StateIntegrityReport, IntegrityError>`

- **Inputs**: a `&ContributorStateStore` (the caller controls auto-prune via Stage 12.7 open) and `ScanOptions { session_id_filter, archive_dir, now_utc }`.
- **Closed discriminators**:
  - `FindingSeverity` = `{ Ok, Warn, Error }`
  - `FindingKind` = `{ InvalidSession, InvalidJoin, InvalidAssignment, InvalidPartial, InvalidSupersession, InvalidAggregate, StaleSeenMarker, MissingSeenMarker, StrayVerifiedFile, StraySeenFile, OrphanReplacementAssignments, PartialApplySupersession, ReassignTriagable, NotReassignTriagable, ArchiveManifestMalformed, ArchiveBlakeMismatch, ArchiveCoveredSession }`
  - `RecommendedAction` = closed-set static-string labels (`run session-status`, `run plan-session-reassign --reason invalid-partial`, `run archive-session --verify-only`, `delete stale seen marker`, `clean state-dir orphan replacements before retry`, `operator triage required`, …).
- **Report shape**: `StateIntegrityReport { schema_version, generated_at_utc, state_dir, state_version, omni_contributor_version, sessions_scanned, sessions_verified, counts_ok, counts_warn, counts_error, sessions: Vec<SessionIntegritySummary>, findings: Vec<IntegrityFinding> }`. Findings are sorted deterministically by `(session_id, kind, path, reason_tag)`.
- **Frozen schema constant**: `STATE_INTEGRITY_REPORT_SCHEMA_VERSION = 1`. The report is a snapshot for tooling, not a protocol artifact; future v2 bumps follow the same closed-evolution rule as `STATUS_SCHEMA_VERSION`.

## Safety contract

- **Reads only.** No `write_verified_json`, no `mark_seen`, no `cascade_remove_session`, no archive writes. The scanner uses parse-only loaders (`list_verified_*`, `read_verified_aggregate_for`, `is_seen`) and re-invokes verifier functions on parsed bodies in memory.
- **No protocol surface bump.** No `schema_version` change anywhere in the contributor envelope set, no new SNIP wire, no chain/payment/marketplace surface.
- **No execution of repair actions.** Every `recommended_action` is a closed-set label, not an executable; v1 leaves all mutation to the existing Stage 12.10 / 12.11 / 12.14 / 12.15 CLI surfaces.
- **`--session-id` filter restricts everything session-scoped.** Stage 12.13 rejection-note findings, per-session deep scans, per-session seen-marker walks, and the archive walker all honor the filter. Cross-session structural findings (e.g. a junk file directly under `verified/sessions/`) still surface because no specific session owns them.
- **Auto-prune is forced OFF.** The CLI opens the state-store with `auto_prune = false` unconditionally so the open call never cascades-removes an expired session subtree before the scan runs. Without this, the Stage 12.7 default would silently delete the very state an integrity scan is supposed to surface. This is enforced in code (the `--no-prune-state-on-start` flag is deliberately absent from this subcommand) and pinned by a CLI clap regression that asserts the flag is rejected.

## CLI

```
omni-node operator contributor state-integrity \
  --contributor-state-dir <path>             (required)
  [--session-id <hex64>]                     (filter to one session)
  [--include-archives <archive-dir>]         (walk a parallel archive root)
  [--format events|json|pretty]              (default events)
  [--json-out <path>]                        (best-effort JSON mirror)
  [--fail-on-warn]                           (exit 1 when warn+err > 0)
```

- **Exit code policy**: default = exit 1 when `counts_error > 0`; with `--fail-on-warn`, exit 1 when `counts_warn + counts_error > 0`. Operators who want every warning to break CI opt in explicitly.
- **stdout posture**: `--format json` puts only the report JSON on stdout (state-store open notices, JSON mirror notices, and warnings go to stderr) so a `jq` pipeline works directly. `events` and `pretty` keep their prose stdout — same convention as `session-status`.
- **Auto-prune is unconditionally OFF** for this subcommand. Stage 12.7's expiry-driven cascade would silently delete the exact "expired/incomplete session" subtrees an integrity scan should *surface*, so the CLI opens the state-store with `auto_prune = false` regardless of any flag. The `--no-prune-state-on-start` flag that other subcommands expose is deliberately absent here — clap rejects it.

## Out of scope (Stage 12.16)

- **Repair execution.** v1 emits action labels only.
- **Verifier-result caching.** Each scan re-verifies from scratch; this matches Stage 12.13's restart preload cost profile and stays simple.
- **Cross-host federation.** The scanner is single-host; it has no mesh, no peer comparison, no chain query.
- **Schema bump.** No envelope or status / archive / restore report schema_version moves.

## Test coverage map

| Concern | Test |
| --- | --- |
| Clean state-dir → zero findings + scanner writes nothing | `clean_state_produces_no_findings_and_writes_nothing` (`state_integrity_scan.rs`) |
| Tampered session body → `InvalidSession` | `tampered_session_body_emits_invalid_session_finding` |
| Tampered join body → `InvalidJoin` | `tampered_join_body_emits_invalid_join_finding` |
| Tampered assignment body → `InvalidAssignment` | `tampered_assignment_body_emits_invalid_assignment_finding` |
| Tampered partial body → `InvalidPartial` | `tampered_partial_body_emits_invalid_partial_finding` |
| Tampered aggregate body → `InvalidAggregate` | `tampered_aggregate_body_emits_invalid_aggregate_finding` |
| Stale seen marker (no body) → `StaleSeenMarker` warn | `stale_seen_marker_without_body_emits_finding` |
| Missing seen marker (body exists) → `MissingSeenMarker` warn | `missing_seen_marker_with_body_emits_finding` |
| Stray file inside `verified/sessions/<id>/joins/` → `StrayVerifiedFile` | `stray_verified_file_emits_finding` |
| `--session-id` filter restricts session-scoped findings | `session_id_filter_restricts_session_scoped_findings` |
| `--include-archives` clean walk → `ArchiveCoveredSession` Ok, no errors | `clean_archive_dir_via_include_archives_emits_no_archive_findings` |
| `--include-archives` corrupt archive → `ArchiveBlakeMismatch` error | `corrupt_archive_blake3_emits_archive_blake_mismatch_finding` |
| `--include-archives` missing manifest → `ArchiveManifestMalformed` error | `missing_archive_manifest_emits_manifest_malformed_finding` |
| JSON round-trip preserves report | `json_roundtrip_preserves_report` |
| Findings deterministically ordered across re-runs | `findings_are_deterministically_ordered_across_repeat_runs` |
| Empty state-dir scans cleanly | `empty_state_dir_produces_clean_report` |
| Clean supersession seen marker round-trips without false `StaleSeenMarker` (review fix — reverse-walk maps `seen/assignment-supersessions/<sid>--<id>` to the correct `verified/sessions/<sid>/supersessions/<id>.json` path) | `supersession_seen_marker_with_body_emits_no_stale_finding` |
| Scanner leaves expired session subtree on disk (review fix — CLI opens with `auto_prune = false`) | `scanner_leaves_expired_session_subtree_on_disk` |
| CLI flag matrix (`--session-id` / `--include-archives` / `--format` / `--json-out` / `--fail-on-warn`) + CLI must reject `--no-prune-state-on-start` (review fix) | `state_integrity_flag_parse_smoke` (omni-node) |

**Residual gap (deliberate):** the CLI run-fn `run_state_integrity` is not exercised end-to-end (same constraint as Stage 12.10–12.15). The library `scan_state_integrity` is exhaustively covered; the CLI is a thin dispatch + renderer layer on top.

Two further deliberate gaps are documented for operators:

1. **Unparseable verified bodies.** Stage 12.7's parse-only loaders silently skip JSON that fails to parse, so a body whose JSON is structurally broken (e.g. truncated) disappears from the snapshot rather than surfacing as a finding. This is consistent with Stage 12.13's restart-preload behavior; operators who suspect a parse-level corruption should compare disk counts against `session-status`.
2. **No mutation in v1.** A finding's `recommended_action` is a string label, not an executable command. Stage 12.16 deliberately separates *detection* from *repair* — operators run the suggested Stage 12.10 / 12.11 / 12.14 / 12.15 CLI invocations explicitly.

# Stage 12.17 — local state-dir cleanup planner / applier

## What this stage is, in one sentence

A two-step `plan-state-cleanup` + `apply-state-cleanup` flow that composes Stage 12.16 findings + the Stage 12.13 audit projection into a deterministic JSON plan of closed-set cleanup actions, then walks that plan one action at a time with apply-time drift detection and a Stage 12.14-shaped quarantine — all without touching any protocol surface.

## Why cleanup is a separate command

Stage 12.16's `state-integrity` is read-only by contract. The Stage 12.17 cleanup commands honor that contract by living separately: the planner walks the state-dir to *build* a plan; the applier is the only mutating surface and it always re-runs the integrity scan first for drift detection. The two CLIs both open the store with `auto_prune = false` unconditionally (Stage 12.16 review precedent), so neither command can silently delete the expired/incomplete state the planner is supposed to see.

## Library surface

`omni_contributor::cleanup` ships:

- `pub const CLEANUP_PLAN_SCHEMA_VERSION: u32 = 1;` and `pub const QUARANTINE_MANIFEST_SCHEMA_VERSION: u32 = 1;` — closed-evolution per the Stage 12.15 precedent.
- Closed `CleanupActionKind` enum with 9 variants:
  - **Tier A** (no quarantine, reversible idempotency-hint edits): `RemoveSeenMarker`, `WriteSeenMarker`, `RemoveSeenFile`.
  - **Tier B** (quarantine-before-delete): `QuarantineVerifiedFile`, `QuarantineAndUnmarkJoin`, `QuarantineAndUnmarkAssignment`, `QuarantineAndUnmarkSupersession`.
  - **Tier B, gated**: `QuarantineAndUnmarkPartial` (`--allow-invalid-partial-cleanup`), `QuarantineAndUnmarkOrphanAssignment` (`--allow-orphan-assignments`).
- `CleanupAction { kind, session_id, path, seen_marker_path, source_finding_kind, source_reason_tag }` with `deny_unknown_fields`.
- `StateCleanupPlan { schema_version, plan_id, created_at_utc, state_dir, source_integrity_hash, omni_contributor_version, actions, cleanup_plan_hash }`. `plan_id` is the 16-char lowercase hex BLAKE3 prefix of `(state_dir || source_integrity_hash || created_at_utc)`. `source_integrity_hash` is the BLAKE3 of the integrity report's canonical projection with `generated_at_utc` and `state_dir` blanked; apply-time re-projection refuses on mismatch. `cleanup_plan_hash` mirrors Stage 12.11's `repair_plan_hash` recipe (BLAKE3 over the canonical plan with the field cleared).
- `pub fn plan_state_cleanup(report, audit_orphans, &PlanOptions) -> Result<StateCleanupPlan, CleanupError>` — pure projection over the Stage 12.16 report + the new Stage 12.17 orphan side-channel.
- `pub fn apply_state_cleanup(store, &StateCleanupPlan, &ApplyOptions) -> Result<CleanupReport, CleanupError>`.
- `QuarantineManifest` + `QuarantineEntry` — the Stage 12.14-shaped manifest written LAST under `<quarantine-dir>/<plan_id>/quarantine-manifest.json`.

**Scanner extensions (Stage 12.17-additive, no report-schema bump)**:

- `pub fn scan_state_integrity_with_audit_orphans(store, opts) -> Result<(StateIntegrityReport, HashMap<String, Vec<String>>), IntegrityError>` — same report struct as Stage 12.16; the extra map carries the per-session orphan-assignment ids the planner consumes for `QuarantineAndUnmarkOrphanAssignment`.
- `FindingKind::StraySeenFile` is now actively emitted by a new `scan_stray_seen_files` walker. The walker catches files directly under `seen/`, files under unknown namespace dirs, and shape-malformed keys (non-64-hex or missing `<sid>--` for prefixed namespaces). The reverse `scan_seen_marker_consistency` walk was made shape-strict in tandem so the same file never accumulates both a `StaleSeenMarker` and a `StraySeenFile` finding.

## Safety contract

- **Plan is read-only.** Walks the state-dir via Stage 12.16's scanner; writes only the operator-named `--out` plan file.
- **Closed mode matrix**: `--dry-run` runs every preflight (plan-hash, drift, gate, orphan-audit re-check, path-safety) without touching the FS. Real apply does the same preflights then mutates.
- **Path-safety preflight (Stage 12.17 review fix).** Before any IO, the applier validates every action's `path` and `seen_marker_path` against the per-kind whitelist: tier-B + `WriteSeenMarker` paths must start with `verified/sessions/`; `RemoveSeenMarker` / `RemoveSeenFile` paths and `seen_marker_path` fields must start with `seen/`; no path may contain `..` segments, a leading `/`, a backslash, or empty segments. A self-consistent `cleanup_plan_hash` does NOT vouch for path safety — hand-edited plans can recompute the hash trivially — so every path goes through `CleanupError::UnsafePlanPath { path, reason }` validation BEFORE any `std::fs::read` / `std::fs::write` / `remove_verified_relative` / `unmark_seen` call.
- **Apply ordering (Stage 12.17 review fix — three explicit phases).** A successful apply proceeds Phase A → Phase B → Phase C, and if Phase A or B fails the state-dir is byte-identical to pre-apply:
  - **Phase A — Quarantine.** For each tier-B action: read source bytes → compute BLAKE3 → write to `<quarantine-dir>/<plan_id>/<source_relative>` → re-read + BLAKE3 verify the quarantine copy. Build `QuarantineEntry` records in memory. **No source removal.**
  - **Phase B — Manifest.** Write `quarantine-manifest.json` atomically (tempfile + rename) under `<quarantine-dir>/<plan_id>/`. By the time Phase C runs the manifest is durable on disk; a failure here returns `CleanupError::Io` and Phase C is skipped.
  - **Phase C — State-dir mutation.** Walk every action in plan order: `RemoveSeenMarker` → `unmark_seen`; `RemoveSeenFile` → direct `fs::remove_file`; `WriteSeenMarker` → `mark_seen`; tier-B → `remove_verified_relative` then `unmark_seen` for the matching `seen_marker_path`.
- **Drift refusal**: `apply_state_cleanup` re-runs `scan_state_integrity_with_audit_orphans` and refuses with `CleanupError::SourceIntegrityDrift { expected, got }` on `source_integrity_hash` mismatch. For every `QuarantineAndUnmarkOrphanAssignment`, the per-session `compute_audit_health` projection is re-run and the orphan id set must equal the planner's — drift surfaces as `CleanupError::OrphanAuditDrift { session_id, plan_count, current_count }`.
- **Gate refusal**: gated actions are *planned freely* (so the operator can review a complete plan), but **refused at apply time** unless the matching `--allow-…` flag is set. Refusal is `CleanupError::GateRequired { kind, flag }` BEFORE any mutation.
- **Quarantine collision refusal**: pre-existing `<quarantine-dir>/<plan_id>/` is refused. Operator must pass a fresh directory or clear the subtree.
- **No `STATE_VERSION` / `STATE_INTEGRITY_REPORT_SCHEMA_VERSION` bump.** The quarantine subtree lives OUTSIDE the state-dir (operator-supplied path), and the scanner's orphan side-channel is a sibling return value rather than a finding field.

## CLI

```
omni-node operator contributor plan-state-cleanup \
  --contributor-state-dir <path>           (required)
  [--session-id <hex64>]                   (filter session-scoped actions)
  [--integrity-json <path>]                (consume a pre-baked report; drift warned)
  --out <path>                             (where the plan JSON lands; atomic write)
  [--format events|json|pretty]            (default events)

omni-node operator contributor apply-state-cleanup \
  --contributor-state-dir <path>           (required)
  --plan <path>                            (plan JSON to apply)
  --quarantine-dir <path>                  (required even for tier-A-only plans)
  [--dry-run]                              (validate without mutating)
  [--allow-invalid-partial-cleanup]        (gate for QuarantineAndUnmarkPartial)
  [--allow-orphan-assignments]             (gate for orphan-assignment actions)
  [--purge-stray]                          (skip quarantine for QuarantineVerifiedFile)
  [--format events|json|pretty]
```

Closed-set bare-stdout events: `event=cleanup_plan_written`, `event=cleanup_action_planned`, `event=cleanup_plan_built`, `event=cleanup_started`, `event=cleanup_action_applied`, `event=would_apply_action`, `event=cleanup_action_skipped`, `event=cleanup_complete`. Drift / hash / gate / collision refusals all exit non-zero with the typed `CleanupError` message on stderr.

## Out of scope (Stage 12.17)

- **`InvalidSession` / `InvalidAggregate` cleanup.** A session.json removal cascades the whole subtree; an aggregate removal flips the overall status. Both are operator-routed through Stage 12.14 `archive-session --move` or manual triage.
- **Auto-chain into reassign.** Stage 12.11 `plan-session-reassign` is a separate operator step. Cleanup never invokes it.
- **Quarantine retention / pruning.** The quarantine subtree is operator-managed metadata. v1 doesn't sweep old subtrees.
- **First-class quarantine restore via Stage 12.15.** The quarantine manifest is *not* an `ArchiveManifest`; manual `cp` back into the state-dir is the v1 rollback story.
- **No envelope, no canonical-byte changes, no `schema_version` bump on Stage 12.0–12.16 envelopes, no `STATE_VERSION` / `STATE_INTEGRITY_REPORT_SCHEMA_VERSION` bump, no new gossipsub topic, no SNIP / mesh / chain / payment / proof / marketplace surface.**

## Test coverage map

| Concern | Test |
| --- | --- |
| Clean state-dir → empty plan | `clean_state_produces_empty_plan` (`state_cleanup_plan.rs`) |
| Plan hash self-consistency + source-integrity-hash match | `plan_hash_is_self_consistent_and_drift_aware` |
| Tier A: stale seen marker round-trip | `stale_seen_marker_is_removed_by_cleanup_apply` |
| Tier A: missing seen marker round-trip | `missing_seen_marker_is_written_by_cleanup_apply` |
| Tier B: tampered join quarantined + unmarked + post-scan clean | `tampered_join_is_quarantined_and_unmarked` |
| Tier B: stray verified file quarantine round-trip | `stray_verified_file_quarantine_round_trip` |
| Source-integrity drift refusal | `apply_refuses_on_source_integrity_drift` |
| Plan-hash mismatch refusal | `apply_refuses_on_plan_hash_mismatch` |
| `--allow-invalid-partial-cleanup` gate refusal + accept | `invalid_partial_cleanup_is_gated` |
| Dry-run writes nothing (state-dir + quarantine) | `dry_run_writes_nothing` |
| Pre-existing `<quarantine-dir>/<plan_id>/` refusal | `apply_refuses_on_existing_quarantine_dir` |
| Plan ⇄ JSON round-trip preserves hashes | `plan_json_roundtrip_preserves_hash` |
| `InvalidSession` finding produces no v1 action | `invalid_session_finding_produces_no_action` |
| Cleanup doesn't collide with Stage 12.14 archive subtree | `cleanup_quarantine_does_not_collide_with_archive_layout` |
| Path-traversal refusal in `RemoveSeenFile` (review fix — `UnsafePlanPath`) | `apply_refuses_traversal_path_in_remove_seen_file` |
| Path-traversal refusal in `RemoveSeenMarker` (review fix) | `apply_refuses_traversal_path_in_remove_seen_marker` |
| Path-traversal refusal in tier-B `path` (review fix) | `apply_refuses_traversal_path_in_tier_b_action` |
| Path-traversal refusal in `seen_marker_path` (review fix) | `apply_refuses_traversal_path_in_seen_marker_path_field` |
| Quarantine-write failure leaves state-dir byte-identical (review fix — Phase A→B→C ordering) | `quarantine_write_failure_leaves_state_dir_untouched` |
| `StraySeenFile` emission: under `seen/` root | `stray_seen_file_under_seen_root_emits_finding` (`state_integrity_scan.rs`) |
| `StraySeenFile` emission: unknown namespace dir | `stray_seen_file_under_unknown_namespace_emits_finding` |
| `StraySeenFile` emission: shape-malformed key (no double-`StaleSeenMarker`) | `shape_malformed_seen_key_emits_stray_not_stale` |
| Orphan side-channel empty on clean state | `audit_orphan_side_channel_is_empty_on_clean_state` |
| CLI flag matrix `plan-state-cleanup` (including deliberate rejection of `--no-prune-state-on-start`) | `plan_state_cleanup_flag_parse_smoke` (omni-node) |
| CLI flag matrix `apply-state-cleanup` (including deliberate rejection of `--no-prune-state-on-start`) | `apply_state_cleanup_flag_parse_smoke` |

**Residual gap (deliberate):** the CLI run-fns `run_plan_state_cleanup` and `run_apply_state_cleanup` aren't exercised end-to-end (same constraint as Stage 12.10–12.16); the library functions are exhaustively covered and the CLI is a thin dispatch + renderer layer on top.
