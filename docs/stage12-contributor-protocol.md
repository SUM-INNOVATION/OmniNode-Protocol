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
