# OmniNode Operator Runbook

Operational guide for running the `omni-node operator` surface against
SUM Chain. Covers the day-to-day commands, the build matrix
introduced in Stage 9a, registry inspection, recovery semantics,
observability, and a docs-only systemd sample.

For the per-stage feature breakdown see the [`README.md`](../README.md)
Phase 5 sections; for the historical first-mainnet-finalization
record see [`mainnet-smoke-audit.md`](./mainnet-smoke-audit.md).

---

## 1. Build matrix (Stage 9a + 9c)

OmniNode separates **read-only** operator commands from **submit**
commands at the cargo-feature level. A default build's binary
contains only the read-only subcommands; submit subcommands appear
only with `--features submit`. Both builds resolve entirely from
public sources (crates.io); **no GitHub credential is required for
either build** as of Stage 9c.

| Invocation | Compiled subcommands | `sumchain-primitives` / `sumchain-crypto` source |
|---|---|---|
| `cargo build -p omni-node` (default) | `watch-activation`, `preflight`, `query` (incl. `--tx-hash`), `derive-address`, `registry list/show`, `loop` (monitor-only) | absent from compile graph (Stage 9a feature gate keeps them out) |
| `cargo build -p omni-node --features submit` | All of the above plus `smoke` and `loop --allow-submit [--allow-mainnet-submit]` | crates.io `v0.1.0` (dual-licensed MIT OR Apache-2.0, byte-equivalent to chain rev `d83e45a4` for the InferenceAttestation surface) |

Read-only operator binaries stay small and submit code is an
explicit operator choice; both halves are enforced as HARD GATES by
the Stage 9c CI workflow on every push / PR.

### 1a. Choosing the right build — `--help` discoverability

The CLI is the source of truth for which mode a built binary is in:

```text
# Read-only binary  (default build):
$ omni-node operator --help
  watch-activation       Poll until OmniNode + V2 activation, then exit 0.
  loop                   Periodic poll → stale-sweep → (retry) lifecycle loop.
  preflight              Read-only: validate seed ↔ attestation-json …
  query                  Read-only: query the chain by (session_id, …) OR --tx-hash.
  derive-address         Read-only: print the chain address derived from …
  registry               Read-only: list / show records in a local …
# (no `smoke` subcommand; `loop --help` shows no --allow-submit / --allow-mainnet-submit)

# Submit-capable binary  (cargo … --features submit):
$ omni-node operator --help
  …all of the above, plus:
  smoke                  Submit one attestation and poll it to Finalized (Included = progress).
# (`loop --help` additionally shows --allow-submit / --allow-mainnet-submit)
```

If an operator accidentally invokes a submit command against a
read-only build (e.g. `omni-node operator smoke …`), `clap` reports
**`error: unrecognized subcommand 'smoke'`** — a clear, scriptable
discoverability cue rather than a runtime stub. To upgrade, rebuild
with `--features submit`.

#### Capturing `--help` for verification

Operators distributing a binary to a host other than their build
machine should capture the `--help` output of both `operator` and
`operator smoke` (the cfg-gated subcommand) once, alongside the
recorded SHA-256 of the binary (see §14). On the destination host,
re-run the same `--help` commands and diff against the captured
output — any difference means the binary you shipped is not the one
that produced the capture.

```bash
# Capture (build machine):
omni-node --version > /etc/omninode/version.txt
omni-node operator --help > /etc/omninode/help-operator.txt
# only on a submit build:
omni-node operator smoke --help > /etc/omninode/help-smoke.txt

# Verify (destination host):
diff /etc/omninode/help-operator.txt <(omni-node operator --help)
```

The `operator --help` output also contains the structured
`registry summary` subcommand (Stage 10a) which is present on both
default and submit builds. Use it to confirm the binary is at the
expected Stage 10a or later revision.

### 1b. CI workflow (Stage 9c restored gates)

The CI workflow at [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
runs on push to `main` and on PRs targeting `main`. All jobs are
unconditional — no secrets, no skip branches, runs on external-fork
PRs too:

| Job | Hard gate? | What it asserts |
|---|---|---|
| `default-build-test` | — | `cargo build -p omni-node` succeeds and `cargo test` passes for the workspace minus the PyO3 `omni-bridge` crate. |
| `default-tree-check` | **yes** | `cargo tree -p omni-node` (default features) does NOT contain `sumchain-(crypto\|primitives)`. The Stage 9a feature gating must keep them out of the default compile graph. |
| `submit-build-test` | — | `cargo build / test -p omni-sumchain -p omni-node --features submit` passes. Live `#[ignore]`'d tests are not run in CI. |
| `submit-tree-check` | **yes** | `cargo tree -p omni-node --features submit` DOES contain the public chain crates. Catches the submit feature silently going inert. |
| `stage6-fixture-check` | **yes** | `crates/omni-zkml/tests/fixtures/chain_attestation_vectors.json` and `crates/omni-zkml/src/chain_wire.rs` are byte-identical vs the PR base / push parent. |

Live `omni-sumchain` tests (`#[ignore]`'d, env-var-gated against
`OMNINODE_SUMCHAIN_RPC_URL` / `OMNINODE_VERIFIER_SEED_HEX`) are **not**
run in CI under any configuration — they stay operator-driven against
a local mirror / mainnet.

### 1d. How Stage 9c fixed the fresh-runner source-build issue (historical)

Stage 9a moved the submit code path behind a cargo `submit` feature.
That made a default build's **compiled binary** contain no submit
code — `operator smoke` and the loop's `--allow-submit` flags are
genuinely absent from `--help` on a default build. That part has
always been real.

What Stage 9a did **not** change is how Cargo evaluates a workspace
manifest. `optional = true` on a **git** dependency only controls
whether the dep is compiled / linked. Cargo still **clones** the
git source at workspace resolution time to read its own
`Cargo.toml` and validate the lockfile, regardless of whether any
feature pulls it in. Stage 9b's CI workflow on its first run on
`main` exposed this — `cargo build -p omni-node` failed at resolve
on a fresh runner with no credentials:

```text
error: failed to get `sumchain-crypto` as a dependency of package `omni-sumchain v0.1.0`
Caused by: unable to update https://github.com/SUM-INNOVATION/sum-chain?rev=d83e45a4
Caused by: failed to authenticate when downloading repository
```

**Stage 9c** retired the private git dependency. The chain team
published `sumchain-primitives` v0.1.0 and `sumchain-crypto` v0.1.0
to crates.io under MIT OR Apache-2.0 (Track A, confirmed in writing
2026-05-19). The workspace dep swap from git to `"0.1.0"` eliminates
the git source entirely; Cargo now resolves both crates through the
public crates.io index, no clone of any private repo is attempted on
any code path, and `cargo build -p omni-node` works on a fresh
runner without credentials. Stage 9b's cargo-side gates are restored
in Stage 9c and now run unconditionally — the green run is the
proof.

The audit note for the 2026-05-19 mainnet smoke
([`mainnet-smoke-audit.md`](./mainnet-smoke-audit.md)) is a frozen
historical record and is unaffected by the source-of-deps change
(the on-chain bytes for that tx were produced by chain code that
also ran at rev `d83e45a4`; crates.io v0.1.0 is byte-equivalent for
the InferenceAttestation surface).

### 1c. Distribution artifact convention (future)

When release automation lands in a later stage, the documented asset
naming will distinguish the two builds at download time without
renaming the in-Cargo `[[bin]]`:

```text
omni-node-readonly-<version>-<target>.tar.gz   # default-features build
omni-node-submit-<version>-<target>.tar.gz     # --features submit build
```

Stage 9b does **not** add a release / publishing pipeline — only the
naming convention is reserved here for future use.

---

## 2. Environment

| Variable / flag | Purpose |
|---|---|
| `OMNINODE_SUMCHAIN_RPC_URL` (or `--rpc-url`) | RPC endpoint. Required for any chain-touching command. |
| `OMNINODE_VERIFIER_SEED_HEX` | 64-char hex Ed25519 seed for the funded verifier. Required by `smoke`, `loop --allow-submit`, `derive-address`. Never read by `watch-activation` / `query` / `registry`. |
| `--expect-chain-id <N>` | Hard guardrail: refuses to act if `chain_getChainParams.chain_id != <N>`. Required on every chain-touching operator command. SUM Chain mainnet is `1`; local mirror is `31337`. |
| `--registry-path <P>` | Local attestation registry (a directory of JSON files). Required by `smoke`, `loop`, `registry list/show`. |
| `RUST_LOG` | Tracing filter. See § 7. |

Never commit the seed. The audit note explicitly excludes seed
material; runbook examples below use `$OMNINODE_VERIFIER_SEED_HEX` as
a placeholder.

---

## 3. Pre-activation / daily preflight

For mainnet:

```bash
export OMNINODE_SUMCHAIN_RPC_URL=https://rpc.sumchain.io

# Learn / confirm the verifier address from a seed (no chain access).
export OMNINODE_VERIFIER_SEED_HEX=<64 hex>
cargo run -p omni-node -- operator derive-address
# → prints a bare base58 address on stdout (scriptable).

# Confirm the chain is sane and (optionally) snapshot params.
cargo run -p omni-node -- operator preflight \
  --attestation-json ./first-attestation.json \
  --rpc-url $OMNINODE_SUMCHAIN_RPC_URL \
  --expect-chain-id 1
# → logs verifier_address, attestation_id, full chain params,
#   blocks-remaining (or activated=true). REPORT-ONLY: succeeds
#   before and after activation. Never submits.

# If the OmniNode subprotocol hasn't activated yet on the connected
# chain, wait it out:
cargo run -p omni-node -- operator watch-activation \
  --rpc-url $OMNINODE_SUMCHAIN_RPC_URL --expect-chain-id 1
# → exits 0 once both omninode_is_active() && v2_is_active().
```

None of these require `--features submit`.

---

## 4. Submitting an attestation

`smoke` is the canonical single-submission flow. It is the only
command that writes to the chain in this runbook.

```bash
# Submit requires the submit feature.
cargo run -p omni-node --features submit -- operator smoke \
  --rpc-url $OMNINODE_SUMCHAIN_RPC_URL \
  --expect-chain-id 1 \
  --registry-path ./mainnet-attestation-registry \
  --attestation-json ./first-attestation.json \
  --allow-submit --allow-mainnet-submit \
  --confirm-timeout-secs 60
```

Notes:
- The attestation file **must** be a real operator-provided
  `InferenceAttestation`. `--synthetic` is **forbidden on mainnet**
  and not even available in the CLI flag set. **Stage 11a** adds a
  second backstop: even if `--synthetic` were somehow taken, the
  mock proof backend (`mock-v1`) is hard-refused on `chain_id == 1`
  with `OperatorError::MockBackendRefusedOnMainnet`, **before any
  submit-side RPC**.
- On non-mainnet chains (`chain_id != 1`), `--synthetic` no longer
  fabricates placeholder bytes. Stage 11a wires the synthetic path
  through [`MockProofBackend`](../crates/omni-zkml/src/proof.rs) so
  the resulting commitment carries real BLAKE3 hashes of synthetic
  inputs and a real (non-cryptographic) proof envelope. The
  `verifier_signature` field in the registered `InferenceAttestation`
  is marked `stage11a-mock-v1` so operators reading the registry
  can tell which builder produced the record.
- `Included` is treated as progress only; success is reported solely
  on `Finalized`.
- On success, the consolidated `SMOKE SUMMARY` line carries
  `tx_hash` / `included_at_height` / `finalized` / `attestation_id` /
  `submitted_at_block` / `session_id` / `verifier_address`.

**Timeout guidance.** The first mainnet finalization observed ~**11 s**
/ ~1 block past submit (see [`mainnet-smoke-audit.md`](./mainnet-smoke-audit.md)).
The default `--confirm-timeout-secs 60` is comfortably sufficient on
SUM Chain mainnet today; raise it only if your endpoint or network
conditions warrant. No block time is hardcoded — you choose.

---

## 5. Lifecycle loop

`operator loop` reconciles the local registry against the chain on a
fixed-interval tick. Run it **monitor-only first**, only enable retry
after `smoke` is clean.

### Monitor-only (no chain writes)

```bash
cargo run -p omni-node -- operator loop \
  --rpc-url $OMNINODE_SUMCHAIN_RPC_URL \
  --expect-chain-id 1 \
  --registry-path ./mainnet-attestation-registry \
  --staleness-threshold-blocks 24 \
  --poll-interval-secs 30
```

Each tick: `poll_attestations_workflow` → `sweep_stale_attestations_workflow`.
Per-record RPC failures land as `Err` entries in the tick summary and
the sweep continues; the local record is **not** mutated on chain
failure. Ctrl-C triggers graceful shutdown.

### Retry-enabled

```bash
cargo run -p omni-node --features submit -- operator loop \
  --rpc-url $OMNINODE_SUMCHAIN_RPC_URL \
  --expect-chain-id 1 \
  --registry-path ./mainnet-attestation-registry \
  --staleness-threshold-blocks 24 \
  --poll-interval-secs 30 \
  --allow-submit --allow-mainnet-submit
```

The retry sweep submits **only** records in the local `Dropped`
state. Each retry independently fetches the chain head so the
`submitted_at_block` stamp is honest per attempt.

---

## 6. Registry inspection (read-only)

`registry list`, `registry show`, and (Stage 10a) `registry summary`
are local-only — no chain access, no mutation, no `--features submit`
required. `summary` accepts an **optional** `--rpc-url` for one extra
read; it never writes.

```bash
# List all records (sorted ascending by AttestationId hex):
cargo run -p omni-node -- operator registry list \
  --registry-path ./mainnet-attestation-registry

# Filter by local status (case-insensitive):
cargo run -p omni-node -- operator registry list \
  --registry-path ./mainnet-attestation-registry \
  --status dropped

# Pretty-print one record by AttestationId:
cargo run -p omni-node -- operator registry show \
  --registry-path ./mainnet-attestation-registry \
  --id d77d8e95c96e6ae2264cbe3baf1383d9a3ea82e59a49d7fbf97574a04d791f1d

# Stage 10a: counts-by-status (no chain access):
cargo run -p omni-node -- operator registry summary \
  --registry-path ./mainnet-attestation-registry

# Stage 10a: same, plus oldest-Submitted age (one chain RPC read):
cargo run -p omni-node -- operator registry summary \
  --registry-path ./mainnet-attestation-registry \
  --rpc-url https://rpc.sumchain.io \
  --expect-chain-id 1
```

`list` output is one bare-stdout line per record:

```
<id-hex>  <status>  tx=<tx_or_dash>  block=<submitted_or_dash>  updated=<iso8601>
```

`summary` output is exactly three lines (grep contract pinned by
`registry_summary_*` tests):

```
total=12
pending=2 submitted=4 included=1 finalized=3 failed=1 dropped=1
oldest_submitted_age_blocks=147 (id=ab12…, submitted_at_block=6049055, head=6049202)
```

When `--rpc-url` is **omitted**, the third line is replaced with a
stable skip comment so downstream pipelines never lose the line count:

```
# oldest_submitted_age: skipped (no --rpc-url)
```

Other skip reasons: `(no Submitted records)`, `(no Submitted record has
submitted_at_block)`. **`--expect-chain-id` is required when `--rpc-url`
is given** — `summary` never queries an unguarded chain endpoint.

So `operator registry list | grep submitted` and `operator registry
summary | grep finalized=` are the two recommended operator one-liners
for daily checks. `show` emits JSON for full-field inspection (including
`digest`, `attestation`, `receipt`, `error_message`,
`submitted_at_block`).

### 6a. `operator verify-proof` (Stage 11b.0, default build, read-only)

A new default-build (no `--features submit` required) read-only
subcommand inspects a proof artifact JSON and reports its mainnet
eligibility. Part of the **decentralized proof architecture** —
verification is universal, doesn't require a hosted service, runs on
pure CPU, and is available to every operator from Stage 11b.0 onward.

```bash
# Read a ProofArtifactBody JSON and report:
#   - backend_id / proof_system / model_format
#   - whether the proof verifies under the registered ProofVerifier
#   - mainnet eligibility (and refusal reason if refused)
omni-node operator verify-proof --proof-artifact ./some-proof.json
```

Output is bare stdout, five or six lines (the sixth appears only when
mainnet is refused — which is **every Stage 11b.0 artifact**, since
the mainnet allowlist is empty by design):

```
backend_id=mock-v1
proof_system=Mock
model_format=none
verified=true
mainnet_eligible=false
mainnet_refusal=proof artifact uses proof system Some(Mock) ...
```

**Stage 11b.0 ships only `MockProofVerifier`.** Verifying an artifact
whose `proof_system` is anything else (`Stage11bOnnxReference`,
`Ezkl`, `GgufStrategyTbd`) returns the typed
`OperatorError::NoVerifierForProofSystem` until backends land in
Stage 11c+.

**Mainnet eligibility at end of Stage 11b.0: zero.** The mainnet
allowlist (`MAINNET_APPROVED_PROOF_SYSTEMS` in `omni-zkml`) is empty
by design. Every proof artifact this command verifies will report
`mainnet_eligible=false` and carry an explicit refusal reason from
one of the six refusal layers documented in §11a. Mainnet
eligibility is a Stage 11c+ deliverable with chain-team review.

---

**No mutating repair commands ship in Stage 9a/10a/11b.0.** The registry is
durable across restarts; the lifecycle loop is the reconciliation
path. If a record is genuinely corrupt on disk, copy it aside and
let the operator team / chain team review — do not hand-edit JSON in
production.

---

## 7. Observability

Stage 10a is tracing-only; JSON output is deferred. Suggested filter:

```bash
export RUST_LOG=info,omni_node::operator=info,omni_sumchain=warn,omni_zkml=warn
```

### 7a. Stable `event=` markers (Stage 10a contract)

Stage 10a pins a small set of structured `event=` fields so operator
log pipelines can `grep 'event="<name>"'` without parsing free-form
message strings. The shape below is the **operator-facing contract**:
add a field freely; rename or remove → docs PR in the same commit.

| `event=` | Emitted by | Fields |
|---|---|---|
| `startup` | every `operator` invocation, before any chain/registry access | `subcommand`, `feature_submit` (bool), `version` |
| `chain_params` | first chain read of `watch-activation`, `smoke`, `preflight`, `query`, and loop entry | `chain_id`, `finality_depth`, `min_fee`, `omninode_enabled_from_height`, `v2_enabled_from_height`, `head` (Option — `None` when caller hasn't paid for a height read) |
| `activation_state` | `watch-activation` per tick, `smoke` pre-submit, `preflight` (when `--rpc-url` given), loop tick | `omninode_active` (bool), `v2_active` (bool), `activated` (bool — both gates), `head` (Option) |
| `registry_summary` | `operator registry summary` | `total`, `pending`, `submitted`, `included`, `finalized`, `failed`, `dropped`, `oldest_submitted_age_blocks` (Option) |

### 7b. Free-form markers (Stage 9a; still in place)

These predate the `event=` contract and are matched by message-string
grep. They remain stable, but new pipelines should prefer the
`event=` markers above when both are available.

| Marker | Where |
|---|---|
| `SMOKE SUMMARY` | end-of-`smoke` consolidated block with all activation/finalization fields |
| `loop tick complete` | per-tick counts: `polled`, `polled_errors`, `swept`, `swept_errors`, `retried`, `retried_errors`, `retry_skipped` |
| `submit_attestation_workflow_with_block: Submitted at chain head` | each block-aware submit |
| `mark_stale_if_overdue: dropped stale record` | each local Dropped from staleness |
| `chain reports Unknown for a locally-known record` | non-terminal observation; never mutates state |
| `retry_dropped_attestations_workflow: resubmitted dropped record` | each retry submission |

Per-record warnings emit at `WARN` with `id=...`, `error=...`. Per-record
skipped/no-op events emit at `DEBUG` with `status=...`.

---

## 8. Interpreting outcomes

| Observation | Meaning | What the operator does |
|---|---|---|
| `Unknown` from chain on `Submitted` | Mempool eviction, never-seen tx, or chain lag. **Observation-only** by Stage 5.1 contract. | Nothing on a single tick. After several ticks past the staleness threshold, the loop's staleness sweep transitions it to local `Dropped`. |
| `Failed { reason }` from chain | Terminal. The submission reached a final failure on chain. | Record `tx_hash` / `attestation_id` / `session_id` / `verifier_address` / `reason` and escalate to the chain team. No automatic retry. |
| Local `Dropped` | Synthetic OmniNode decision: staleness exceeded threshold OR Ctrl-C interrupted a poll mid-submit. Chain v1 never returns this. **Retryable.** | Retry-enabled loop will resubmit on its next tick. Or rerun `smoke` from a fresh `--registry-path`. |
| `SmokeConfirmTimeout` | The smoke tx never reached `Finalized` within `--confirm-timeout-secs`. | Check the last observed status from the error and the chain explorer. If `Included`, finalization is just slow; rerun smoke with the same registry — local idempotency reuses the tx hash and re-polls. |
| `SmokeInterrupted` | Ctrl-C during the smoke poll. The tx may have submitted but not been confirmed. | Re-run the same `smoke` command — local idempotency picks up the existing record. |
| `MainnetSubmitNotPermitted` | `--allow-submit` set on `chain_id 1` but `--allow-mainnet-submit` missing. | Add `--allow-mainnet-submit`; this is intentional double-gate, not a bug. |

---

## 9. Recovery after process interruption

The registry is the durable state. After a crash, kill, or Ctrl-C:

1. Start a **monitor-only** loop tick: `cargo run -p omni-node -- operator loop ... --max-ticks 1` (no submit flag). This reconciles in-flight records against the chain and surfaces any failures.
2. Inspect the registry: `operator registry list` and `operator registry list --status dropped`. Any `Dropped` records are eligible for retry; any unexpected `Failed` records need chain-team triage.
3. Resume the long-running loop in whichever mode (monitor-only / retry-enabled) you used previously.

No registry-rebuild or replay is needed. There is no separate WAL —
the JSON-per-record store written via atomic `.tmp`+rename **is** the
durable state.

---

## 10. Backup / rotation

The registry is a directory of `<hex_id>.json` files plus
transient `.tmp` files during writes. `cp -r ./mainnet-attestation-registry
./backups/registry-2026-05-19` is a sufficient backup.

Rotate by directing future runs at a fresh `--registry-path`. Old
directories remain readable by `operator registry list/show` after
rotation.

---

## 11. Failure triage + what to report (Stage 10a)

### 11a. Failure-state matrix

Each row maps a real failure state to its detection signal in the
operator surface as it exists today, the recommended local action,
and what to send to the chain team if escalation is needed. Where a
column says "partially," that means the typed surface funnels the
condition through a coarse-grained variant (currently
`ChainClientError::Other(String)`) and operator-side triage must
parse the message string — `omni-sumchain` does not yet expose a
typed taxonomy for sub-conditions like fee/balance vs. transport.

| Symptom | Detection signal | Detectable today? | Operator action | Escalation packet |
|---|---|---|---|---|
| `Unknown` persisting | `query_attestation_workflow` returns `AttestationStatus::Unknown` repeatedly | yes — Stage 5.1 makes Unknown explicitly non-terminal | wait `submitted_threshold_blocks`; if still Unknown, the loop's staleness sweep will move it to local `Dropped` | `tx_hash`, full status path, last `head`, `submitted_at_block` |
| `Failed { reason: String }` from chain | `AttestationStatus::Failed { reason }` — see [crates/omni-zkml/src/chain.rs](../crates/omni-zkml/src/chain.rs) | yes — **opaque string reason**, not a numeric code | record terminalises locally; no automatic retry | `tx_hash`, `attestation_id`, `session_id`, `verifier_address`, **verbatim `reason` string**, current local status |
| RPC error during read | `ChainClientError::Other(_)` surfaced through any chain call | partially — string-opaque single variant today | retry once with the same command; if persistent, treat as RPC outage and pause submits | full error string verbatim, RPC URL host, time of first failure, whether the failure is reproducible against a different mirror |
| Stale `Submitted` → local `Dropped` | `mark_stale_if_overdue: dropped stale record` + `event="registry_summary"` `dropped` count rises | yes — Stage 5.2 staleness | check the chain explorer for the original `tx_hash` first; if it's actually `Finalized` on chain, **do not** retry — the local Dropped is a misclassification, file a chain-team report. If chain truly doesn't know the tx, retry-enabled loop will resubmit on its next tick | dropped record id, original `submitted_at_block`, current head, observed chain status for the tx_hash |
| Duplicate / idempotency conflict | `RegistryError::ConflictingAttestation { id }` on `insert` | yes — Stage 5.1 contract | inspect both records via `operator registry show --id <hex>`; do NOT overwrite | both records' JSON, the conflict id |
| Verifier address mismatch | pre-flight gate in [crates/omni-sumchain/src/tx.rs](../crates/omni-sumchain/src/tx.rs) returns `ChainClientError::Other(_)` **before** any chain call | yes — never reaches chain | re-run `operator derive-address` to confirm `OMNINODE_VERIFIER_SEED_HEX` derives the expected address; check the attestation JSON's `verifier_address` field | derived vs. claimed addresses, seed source (env var name / file path, **never the seed itself**) |
| Insufficient balance / fee failure | surfaces as `ChainClientError::Other(_)` from `sum_sendRawTransaction` | **partially** — opaque string today; operator must parse | check `extra-alloc.json` funding for the verifier address; check `min_fee` from the most recent `event="chain_params"` log | full error string, verifier address, current balance if known from a separate RPC tool, the `min_fee` observed |
| Queryable record missing receipt | `RegistryError::SubmittedRecordMissingReceipt { id }` | yes — Stage 5.1 integrity defence | inspect the registry JSON file for hand-edits; **do not** auto-repair | the record id, JSON file contents, recent `cp -r` backup state if available |
| `SmokeConfirmTimeout` | `OperatorError::SmokeConfirmTimeout { last_status }` | yes — typed error | if `last_status` was `Included`, finalization may just be slow; rerun smoke with the same registry (local idempotency reuses the tx hash) | `tx_hash`, `last_status`, `--confirm-timeout-secs` value used |
| `SmokeInterrupted` | Ctrl-C during the smoke poll | yes — typed error | re-run the same `smoke` command; idempotency picks up the existing record | `tx_hash` (if submission completed before interrupt) |
| `MainnetSubmitNotPermitted` | `--allow-submit` given on `chain_id 1` but `--allow-mainnet-submit` missing | yes — typed error | add `--allow-mainnet-submit`; this is an intentional double-gate, not a bug | (no escalation needed; operator-side correction) |
| `MockBackendRefusedOnMainnet` (**Stage 11a**) | `OperatorError::MockBackendRefusedOnMainnet { backend_id }` — fires when the smoke `--synthetic` path is taken on `chain_id == 1`, even with `--allow-mainnet-submit` | yes — typed error, **before any submit RPC** | mainnet smoke requires a real attestation JSON (`--attestation-json`) produced off-binary by a real prover; the mock backend (`mock-v1`) is non-cryptographic by design and is hard-refused on mainnet | (no escalation needed; operator-side correction. If a real prover is available, use `--attestation-json` and re-run; if not, wait for Stage 11c's real backend) |
| `TestnetOnlyProofRefusedOnMainnet` (**Stage 11b.0**) | proof artifact carries `testnet_or_dev_only: Some(true)` (refusal layer 1) | yes — typed error | the artifact's producer explicitly disclaimed mainnet eligibility; use a mainnet-approved producer | backend_id, producer source |
| `BoundedReferenceProofRefusedOnMainnet` (**Stage 11b.0**) | proof_system is `Stage11bOnnxReference` (refusal layer 3) | yes — typed error | bounded reference fixtures are for architecture validation, not production; use a mainnet-approved producer (none ship at end of Stage 11b) | backend_id |
| `GgufProofClaimRefusedOnMainnet` (**Stage 11b.0**) | `model_format == Gguf` (refusal layer 4) | yes — typed error | no GGUF inference proof backend is approved at any stage through Stage 11b.0; wait for Stage 11d strategy + chain-team review. **Declaring GGUF prevents silent fake-GGUF claims**, which is the point. | backend_id, model_hash |
| `UnknownModelFormatRefusedOnMainnet` (**Stage 11b.0**) | `model_format = Other(_)` or absent on a non-mock backend (refusal layer 5) | yes — typed error | promote the format to a first-class enum variant via a chain-team-reviewed PR, or use an approved format | backend_id, model_format value |
| `ProofSystemNotMainnetApproved` (**Stage 11b.0**) | proof_system not in `MAINNET_APPROVED_PROOF_SYSTEMS` (refusal layer 6) | yes — typed error | **Stage 11b.0 ships with this allowlist empty by design.** No proof system is mainnet-eligible until Stage 11c+ with chain-team review. | backend_id, proof_system |
| `NoVerifierForProofSystem` (**Stage 11b.0**) | `operator verify-proof` was handed an artifact whose `proof_system` has no verifier registered | yes — typed error | Stage 11b.0 ships only `MockProofVerifier`; other proof systems are verifier-side stubs awaiting Stage 11c+ | proof_system |

> **Known limitation flagged by Stage 10a.** [`ChainClientError`](../crates/omni-zkml/src/error.rs)
> is currently the single-variant `Other(String)`. That is why several rows
> above are marked "partially" — fee, balance, transport, and signature
> failures all funnel through the same variant and require string-level
> triage. Stage 10a deliberately does **not** add new error variants;
> splitting `ChainClientError` into a typed taxonomy is a candidate for a
> future stage if operator feedback shows the parsing burden is real.

### 11c. GGUF proofs — what's possible today (Stage 11b.0)

OmniNode's canonical model format today is **GGUF** (llama.cpp). Stage
11b.0 ships the schema slot for declaring `model_format = "gguf"` on
proof artifacts, but **no GGUF inference proof backend is approved at
any stage through Stage 11b.0**.

The honest framing — to be repeated wherever Stage 11b is described:

- Full GGUF transformer inference proving is **not feasible** end-to-
  end with any current production-ready proof system. This is an open
  research problem.
- Stage 11d will pick **one or more** of the following strategies and
  document what each actually proves. None of them prove full
  transformer inference correctness:
  - **Shadow verifier circuit**: a small ONNX *verifier net* (provable
    via the Stage 11c ezkl path) ingests `(model_hash, input_hash,
    output_hash, derived features)` and outputs a binding. Proves the
    verifier-net computation, not the GGUF inference itself.
  - **Partial-inference proof**: a zkVM circuit re-executes a single
    layer of the transformer (e.g., the final softmax) given hashed
    intermediate state. Proves that the final emitted token follows
    from the claimed intermediate activation.
  - **Replay-based attestation**: K independent operators run the
    same `(model, input)` and submit attestations; matching
    `output_hash` values are treated as "K-replicated." This is a
    **consensus** mechanism, not a proof system.
- Until Stage 11d ships an approved strategy, **declaring
  `model_format = "gguf"` on a proof artifact and attempting mainnet
  submission is hard-refused** by Stage 11b.0's refusal layer 4
  (`GgufProofClaimRefusedOnMainnet`). The refusal is the point — it
  prevents silent fake-GGUF claims.

This is the **decentralized proof architecture** — not "decentralized
compute readiness." The criteria for the latter are documented in the
Stage 11b.0 README section and **explicitly unmet** at end of Stage
11b.0.

### 11b. Escalation packet

When escalating to the chain team, send:

- `tx_hash` (operator-side, from `SMOKE SUMMARY` or `registry show`).
- `attestation_id` (the 64-hex `(session_id, verifier_address)` de-dup key).
- `session_id`, `verifier_address`.
- The full status path you observed (`Submitted → Unknown → Unknown → Finalized`, or whatever).
- `submitted_at_block` and the chain head at the last poll.
- For chain `Failed`: the `reason` string verbatim.
- Current local registry status (`Submitted` / `Included` / `Finalized` / `Failed` / `Dropped`).
- The exact `operator` command line used (with secrets redacted — see below).
- The `event="chain_params"` line from the same session (chain_id, min_fee, activation heights, head observed at the time).

**Redact:** never include `OMNINODE_VERIFIER_SEED_HEX` or its file path in
escalation messages. The derived `verifier_address` is the safe identifier.

The exact channel (Slack, ticket system, email) is operator-defined.
The fields above are what the chain team needs.

---

## 12. Sample systemd unit (docs only — not installed)

For operators who supervise the lifecycle loop with systemd. **This
file is documentation; OmniNode ships no installer, service manager,
or daemon code.**

```ini
# /etc/systemd/system/omni-node-operator-loop.service
#
# Long-running monitor-only loop. For retry-enabled operation,
# rebuild omni-node with --features submit and add the
# --allow-submit / --allow-mainnet-submit flags to ExecStart.

[Unit]
Description=OmniNode operator lifecycle loop (monitor-only)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=omninode
Group=omninode
Environment=RUST_LOG=info,omni_node::operator=info,omni_sumchain=warn,omni_zkml=warn
Environment=OMNINODE_SUMCHAIN_RPC_URL=https://rpc.sumchain.io
# OMNINODE_VERIFIER_SEED_HEX is only needed for retry-enabled mode;
# leave UNSET for monitor-only and prefer EnvironmentFile= over an
# inline Environment= line so the seed never appears in process
# listings.
EnvironmentFile=-/etc/omninode/verifier-seed.env
WorkingDirectory=/var/lib/omninode
ExecStart=/usr/local/bin/omni-node operator loop \
  --rpc-url ${OMNINODE_SUMCHAIN_RPC_URL} \
  --expect-chain-id 1 \
  --registry-path /var/lib/omninode/registry \
  --staleness-threshold-blocks 24 \
  --poll-interval-secs 30
Restart=on-failure
RestartSec=15s
# Graceful Ctrl-C → SIGINT semantics: omni-node operator loop
# returns 0 on a clean shutdown, so systemd records `succeeded`.
KillSignal=SIGINT
TimeoutStopSec=30s
# Hardening (optional but recommended)
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/omninode

[Install]
WantedBy=multi-user.target
```

Adapt paths, the user, environment file, and the RPC URL to your
deployment. Build the binary appropriately for the desired mode:

```bash
# Monitor-only (default; resolves from crates.io only):
cargo build --release -p omni-node
sudo install -m 0755 target/release/omni-node /usr/local/bin/omni-node

# Retry-enabled (also crates.io only; Stage 9c — no GitHub access
# required; the chain primitives ship from public crates.io v0.1.0):
cargo build --release -p omni-node --features submit
sudo install -m 0755 target/release/omni-node /usr/local/bin/omni-node
```

---

## 13. Quick reference

| Need | Command | `--features submit`? |
|---|---|---|
| Print verifier address from a seed | `operator derive-address` | no |
| Validate JSON ↔ seed offline | `operator preflight --attestation-json ...` | no |
| Validate + chain snapshot | `operator preflight --rpc-url ... --expect-chain-id ...` | no |
| Wait for OmniNode activation | `operator watch-activation` | no |
| Query by `(session_id, verifier_address)` | `operator query --session-id ... --verifier-address ...` | no |
| Query by tx hash | `operator query --tx-hash 0x...` | no |
| List registry records | `operator registry list` | no |
| Show one record | `operator registry show --id <hex>` | no |
| Counts-by-status summary | `operator registry summary` | no |
| Summary + oldest-Submitted age | `operator registry summary --rpc-url ... --expect-chain-id ...` | no |
| Verify a proof artifact + report mainnet eligibility | `operator verify-proof --proof-artifact <path>` | no |
| Monitor-only loop | `operator loop` | no |
| Submit one attestation | `operator smoke ... --allow-submit --allow-mainnet-submit` | **yes** |
| Retry-enabled loop | `operator loop ... --allow-submit --allow-mainnet-submit` | **yes** |

Stage 9a is operations hardening, not protocol work. The SUM Chain
wire format is unchanged from Stage 7b; Stage 6's
`chain_attestation_vectors` fixture is byte-stable. Stage 10a adds
the `registry summary` subcommand, stable `event=` log markers, the
§11 failure-triage matrix, and the §14 release-readiness checklist
below — also without touching protocol bytes or transaction
construction.

**Stage 11b.0 — decentralized proof architecture.** Adds the
`ProofBackend` / `ProofVerifier` trait extensions, the `ModelFormat`
+ `ProofSystem` enums, the six-layer `check_mainnet_eligible` refusal
helper, and the `operator verify-proof` read-only subcommand. **Stage
11b.0 mainnet eligibility is intentionally zero** — the allowlist
ships empty, and every proof artifact `verify-proof` inspects will
report `mainnet_eligible=false`. This is **architecture**, not
production zkML; the criteria for "decentralized compute readiness"
are explicitly unmet at end of Stage 11b. See §11c for the honest
GGUF-proofs framing.

---

## 14. Release-readiness checklist (Stage 10a)

Run through this before tagging or distributing any `omni-node`
binary. **Stage 10a is documentation-only on the release side** — no
artifact workflow, no signing pipeline. Everything below is manual
and tracked by the operator.

1. **CI green on the tagged commit.** All six jobs in
   [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) must
   pass: `default-build-test`, `default-tree-check`,
   `submit-build-test`, `submit-tree-check`, `stage6-fixture-check`,
   and (Stage 11a) `stage11a-fixture-check`.

2. **Stage 6 byte-stability across the release window.** No
   functional change to chain wire bytes between the previous tag
   and this one:

   ```bash
   git diff --stat <previous-tag>..<this-tag> -- \
     crates/omni-zkml/tests/fixtures/chain_attestation_vectors.json \
     crates/omni-zkml/src/chain_wire.rs
   # must be empty
   ```

3. **Local test parity.** On the build host, both feature
   configurations green:

   ```bash
   cargo test -p omni-sumchain --features submit
   cargo test -p omni-node
   cargo test -p omni-node --features submit
   ```

   The `parity_vendored_primitives` (5 tests) and
   `chain_produced_fixture` (5 tests; Stage 9c.1) suites both
   contribute to the submit-build green light.

4. **Cargo version matches the git tag.** The workspace's
   `[workspace.package].version` (root `Cargo.toml`) equals the
   semver portion of the git tag being cut.

5. **Operator runbook is current.** Re-read §1, §6, §7, §11, §14 of
   this document on the head commit and confirm no claim references
   a removed flag, file, or behaviour. The Stage 9c PR review showed
   why a quick re-read is worth more than an automated linter here.

6. **Mainnet smoke audit immutability.** `git diff <previous-tag>..<this-tag> -- docs/mainnet-smoke-audit.md`
   must be empty. The audit is a historical record; release-prep
   does not edit it.

7. **Chain-produced fixture gate green.** Specifically
   `fixture_signed_tx_hash_matches_chain_produced_tx_hash` in
   [`crates/omni-sumchain/tests/chain_produced_fixture.rs`](../crates/omni-sumchain/tests/chain_produced_fixture.rs)
   passes — confirms the public crate's hashing has not drifted
   from the chain's TRANSACTIONS-CF key derivation.

7a. **Stage 11a proof pipeline fixture gate green.** The three
   vectors in [`crates/omni-zkml/tests/fixtures/proof_pipeline_vectors.json`](../crates/omni-zkml/tests/fixtures/proof_pipeline_vectors.json)
   are byte-identical against the committed values. Run
   `cargo test -p omni-zkml --test proof_pipeline_vectors` locally
   to confirm. Drift here (mock backend formula, canonical envelope
   shape, BLAKE3 chain, chain-wire path, Ed25519 signing) is a real
   regression — see §11a's `MockBackendRefusedOnMainnet` row.

8. **`omni-node --version` captured in release notes.** Run
   `omni-node --version` on the build host and paste the output
   verbatim into the release notes.

9. **SHA-256 recorded alongside the tag.** For each binary you
   intend to distribute:

   ```bash
   sha256sum target/release/omni-node
   # paste into release notes alongside (build host, build date, feature flags)
   ```

   **Signing is deferred to Stage 10b.** Until then, the SHA-256 +
   the operator's verification of `--help` output (see §1a "Capturing
   `--help` for verification") is the integrity path.

10. **Default + submit `--help` snapshots committed in the release
    notes.** Capture both via the §1a one-liners and include them
    next to the SHA-256 so the destination operator can diff
    locally without re-building.

The checklist intentionally has no GitHub Actions automation
binding. Stage 10b will evaluate whether a `workflow_dispatch`
artifact build + checksum publication is the right next step; until
that lands, items 8–10 are manual and recorded in the operator's
release log.
