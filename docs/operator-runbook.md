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
  and not even available in the CLI flag set.
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

`registry list` and `registry show` are local-only — no chain
access, no mutation, no `--features submit`.

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
```

`list` output is one bare-stdout line per record:

```
<id-hex>  <status>  tx=<tx_or_dash>  block=<submitted_or_dash>  updated=<iso8601>
```

So `operator registry list | grep submitted` is the recommended
operator one-liner for daily checks. `show` emits JSON for full-field
inspection (including `digest`, `attestation`, `receipt`,
`error_message`, `submitted_at_block`).

**No mutating repair commands ship in Stage 9a.** The registry is
durable across restarts; the lifecycle loop is the reconciliation
path. If a record is genuinely corrupt on disk, copy it aside and
let the operator team / chain team review — do not hand-edit JSON in
production.

---

## 7. Observability

Stage 9a is tracing-only; JSON output is deferred. Suggested filter:

```bash
export RUST_LOG=info,omni_node::operator=info,omni_sumchain=warn,omni_zkml=warn
```

Stable field markers worth grepping:

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

## 11. What to report on a chain incident

When escalating to the chain team:

- `tx_hash` (operator-side from `SMOKE SUMMARY` or `registry show`).
- `attestation_id` (the 64-hex `(session_id, verifier_address)` de-dup key).
- `session_id`, `verifier_address`.
- The full status path you observed (`Submitted → Unknown → Unknown → …`).
- `submitted_at_block` and the chain head at the last poll.
- For `Failed`: the `reason` string verbatim.
- Whether the local registry record is currently `Submitted`,
  `Included`, `Finalized`, `Failed`, or `Dropped`.

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
| Monitor-only loop | `operator loop` | no |
| Submit one attestation | `operator smoke ... --allow-submit --allow-mainnet-submit` | **yes** |
| Retry-enabled loop | `operator loop ... --allow-submit --allow-mainnet-submit` | **yes** |

Stage 9a is operations hardening, not protocol work. The SUM Chain
wire format is unchanged from Stage 7b; Stage 6's
`chain_attestation_vectors` fixture is byte-stable.
