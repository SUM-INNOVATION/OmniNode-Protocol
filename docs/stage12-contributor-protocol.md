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
