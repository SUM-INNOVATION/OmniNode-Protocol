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

### 6a. `operator verify-proof` (Stage 11b.0 + 11b.0.1, default build, read-only)

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

**Stage 11b.0.1 — single dispatch entry point.** `verify-proof`
routes every artifact through `omni_zkml::ProofVerifier::verify_artifact(&body)`,
a defaulted trait method that takes the full `ProofArtifactBody`.
The default impl calls `self.verify(&proof, &public_inputs)` with
the hashed `PublicInputs`, so `MockProofVerifier` (and any future
Stage 11a-shape verifier) inherits the correct behaviour with zero
extra code. Real backends whose proof systems need backend-specific
public inputs (e.g., raw tensor values for a Stage 11c prover)
override `verify_artifact` and read `body.metadata.public_inputs`
directly. **No backend-specific helper logic in operator code** —
that's the architectural property Stage 11b.0.1 locks in for every
future backend.

**Mainnet eligibility at end of Stage 11b.0 / 11b.0.1 / 11b.1.a / 11b.1.b / 11c / 11d.0 / 11d.1: zero.**
The mainnet allowlist (`MAINNET_APPROVED_PROOF_SYSTEMS` in
`omni-zkml`) is empty by design. Every proof artifact this command
verifies will report `mainnet_eligible=false` and carry an explicit
refusal reason from one of the six refusal layers documented in §11a.
**Stage 11d defines and prepares the eligibility path; a later
Stage 11d.3 entry may add the first eligible proof system after
written chain-team sign-off** — see the Stage 11d.0 authoritative
docs:
  - [`docs/mainnet-eligibility-criteria.md`](mainnet-eligibility-criteria.md)
    — what qualifies a proof system; required `ProofArtifactBody`
    metadata; allowlist mechanics; chain-team review packet
    requirements; non-goals (including: `Stage11bHalo2Reference`
    stays testnet/dev-only in perpetuity).
  - [`docs/stage11d-review-packet.md`](stage11d-review-packet.md)
    — the structured template a Stage 11d.3 PR fills in.
  - [`docs/stage11d-mainnet-eligibility-FAQ.md`](stage11d-mainnet-eligibility-FAQ.md)
    — operator-facing Q&A.

**Stage 11b.1.a — multi-framework architectural scaffold (four equal primaries).**
Adds the `ModelFramework` enum (Rumus / PyTorch / TensorFlow / Caffe /
FrameworkAgnostic), `ModelFormat::Halo2ReferenceMlp`, and
`ProofSystem::Stage11bHalo2Reference`. The new
`crates/omni-proofs-halo2-reference/` ships a framework-neutral
canonical spec (`assets/canonical_spec.json`) for a bounded 4→8→4
int16 fixed-point MLP plus a pure-Rust canonical evaluator (the
**neutral reference implementation, not a fifth framework**) and
committed fixture manifests for all five variants. **RUMUS,
PyTorch, TensorFlow, and Caffe are all equal-status primary
compatibility targets**; each has its own developer-host exporter
under `tools/<framework>_export/` that reads the spec and emits its
manifest by running the canonical arithmetic through that
framework's own primitives. RUMUS is now LiveExport via `rumus =
"0.4.0"`'s `fixed::FixedLinear`. Caffe's exporter records an
auditable `PureNumpyCompatibility` fallback when the host lacks a
working Caffe binding (`generator_metadata.runtime_mode =
"pure-numpy-emulation"`); a host with real Caffe gets
`runtime_mode = "caffe-runtime"` and `generation_mode = LiveExport`.
The numeric contract is RUMUS-compatible: i16 tensors, i64
accumulator, bias added in widened scale² domain BEFORE
saturation, round-nearest-ties-away-from-zero requantization,
saturate to i16, ReLU `max(x, 0)`. **No halo2 circuit, no prover
dependency, no operator-binary changes.** The canonical evaluator
+ cross-framework equivalence test assert byte-for-byte agreement
plus identical `weights_hash` / `spec_hash` / `input_hash` /
`output_hash` across all five manifests. **OmniNode does not
invoke any framework runtime at any point** — framework manifests
are committed JSON files, parsed by pure Rust; the four exporters
live under `tools/` (excluded from the workspace) so
`cargo build -p omni-node` cannot transitively reach them. Layer 3
of `check_mainnet_eligible` refuses both `Stage11bOnnxReference`
AND `Stage11bHalo2Reference` (defense in depth alongside the
testnet flag + empty allowlist).

**Stage 11b.1.b — halo2 reference verifier (opt-in feature
`halo2-reference-verify`).** Adds a `Halo2ReferenceVerifier` to
`omni-proofs-halo2-reference` (gated by the `verify` cargo feature)
that overrides `omni_zkml::ProofVerifier::verify_artifact`. Default
`omni-node` builds **do not include** halo2 / pasta_curves / IPA
dependencies — `cargo tree -p omni-node` is empty of those crates,
gated by a CI tree-check. Operators who want to verify
`Stage11bHalo2Reference` artifacts in the field rebuild with
`cargo build -p omni-node --features halo2-reference-verify` (or
the equivalent `cargo install` form). The verifier:
  1. Asserts `proof_system == Stage11bHalo2Reference`,
     `model_format == Halo2ReferenceMlp`,
     `model_framework == FrameworkAgnostic`,
     `testnet_or_dev_only == Some(true)`, and `model_hash` equals
     the compile-pinned `EXPECTED_SPEC_HASH`.
  2. Decodes `metadata.public_inputs` JSON (backend-specific
     field) into `[i16; 4]` input + output, re-encodes LE, and
     verifies BLAKE3 matches `metadata.input_hash` /
     `metadata.response_hash`.
  3. Loads the embedded IPA params + re-derives the verifying key
     from the circuit (halo2_proofs 0.3.2 does not provide a
     stable on-disk VK format; re-derivation is deterministic).
  4. Calls `halo2_proofs::plonk::verify_proof` against the proof
     bytes from `body.proof_bytes_hex` and the field-lifted
     instance column built from the i16 input/output values.
The non-artifact entry point `verify(&[u8], &PublicInputs)` returns
`ProofVerifierError::RequiresArtifactDispatch` — the three-hash
`PublicInputs` cannot bind the raw i16 instance values into the
halo2 proof. Callers must use `verify_artifact(&ProofArtifactBody)`.
Mainnet refusal layers 1, 3, and 6 still hard-refuse; the feature
only enables verification of a testnet/dev artifact.

The committed proof fixture lives at
`crates/omni-proofs-halo2-reference/fixtures/halo2/` and is
regenerated via the workspace-excluded
`tools/halo2_reference_regen/` standalone Cargo package — pattern
identical to `tools/rumus_export/` in Stage 11b.1.a so the
operator binary's compile graph never reaches the prover.

**Stage 11c — arbitrary-input soundness for the bounded
`halo2-mlp-v1 / spec_version: 2` numeric contract.** Replaces the
Stage 11b.1.b "linear identity + remainder-range-check" gates
with a complete gadget chain: dense linear identity → round-half-
away-from-zero (RHAZ) gadget via signed-magnitude Euclidean
division → three-branch saturation gadget (`b_lo` / `b_in` /
`b_hi` with 17-bit aux witnesses) → ReLU sign-bit gadget. The
RHAZ gadget's Euclidean division `abs_w + S/2 = q_abs · S + r_pos`
with `r_pos ∈ [0, S)` is unique, so ties at ±S/2 and ±3S/2 are
pinned to the canonical round-half-AWAY branch without ambiguity.
A committed 8-entry test corpus
(`crates/omni-proofs-halo2-reference/tests/fixtures/corpus.json`)
exercises the canonical input, the bias-only path, four
hand-constructed tie cases, and the two extreme i16 inputs;
cross-framework corpus files (`{rumus,pytorch,tensorflow,caffe,
framework_agnostic}_corpus.json`) attest that every framework
reproduces the canonical output byte-for-byte for every entry.
`HALO2_K` bumped from 9 → 10. Mainnet posture unchanged: layers
1, 3, 6 still hard-refuse; `MAINNET_APPROVED_PROOF_SYSTEMS` stays
empty; all artifacts `testnet_or_dev_only: Some(true)`. **Stage
11c is still not "production zkML";** the bounded MLP is an
architectural-validation fixture. Production zkML and mainnet
eligibility are Stage 11d+ deliverables with chain-team review.

**Exit code: inspect/report, not strict-validator.** `verify-proof`
exits `0` on a successful *inspection* run regardless of whether
`verified=true` or `verified=false` — the boolean is the result
being reported, not a pass/fail gate. Operators consuming this from
scripts should grep stdout (e.g. `omni-node operator verify-proof
--proof-artifact ./p.json | grep '^verified='`) or consume the
structured `event="proof_verification"` tracing fields, not the
exit code. The CLI exits non-zero only on typed errors —
`ProofArtifactRead` (file unreadable), `ProofArtifactParse` (JSON
malformed), `NoVerifierForProofSystem` (proof system has no
registered verifier in Stage 11b.0). **A "strict validator" exit
code where `verified=false` returns non-zero is a deferred UX
decision** for Stage 11c (when real backends land) or Stage 10b
(release-artifact tooling) — not a behaviour change for Stage 11b.0.

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
| `TestnetOnlyProofRefusedOnMainnet` (**Stage 11b.0**, tightened in **Stage 11d.1**) | proof artifact's `testnet_or_dev_only` is not `Some(false)` (refusal layer 1). Both `Some(true)` (explicit disclaim) and `None` (absent declaration — treated as testnet/dev for safety per `docs/mainnet-eligibility-criteria.md` §1.5) fire this layer. Only `Some(false)` passes. | yes — typed error | either the producer explicitly disclaimed mainnet eligibility (`Some(true)`) or did not declare the flag (`None`); use a mainnet-approved producer that sets `Some(false)` | backend_id, testnet_or_dev_only value |
| `BoundedReferenceProofRefusedOnMainnet` (**Stage 11b.0**) | proof_system ∈ `{Stage11bOnnxReference, Stage11bHalo2Reference}` (refusal layer 3) | yes — typed error | bounded reference fixtures are for architecture validation, not production; use a mainnet-approved producer (none ship through Stage 11c) | backend_id |
| `GgufProofClaimRefusedOnMainnet` (**Stage 11b.0**) | `model_format == Gguf` (refusal layer 4) | yes — typed error | no GGUF inference proof backend is approved at any stage through Stage 11d.0; wait for a future Stage 11e research-track strategy + chain-team review. **Declaring GGUF prevents silent fake-GGUF claims**, which is the point. | backend_id, model_hash |
| `UnknownModelFormatRefusedOnMainnet` (**Stage 11b.0**) | `model_format = Other(_)` or absent on a non-mock backend (refusal layer 5) | yes — typed error | promote the format to a first-class enum variant via a chain-team-reviewed PR, or use an approved format | backend_id, model_format value |
| `ProofSystemNotMainnetApproved` (**Stage 11b.0**, schema-extended in **Stage 11d.1**) | artifact's `(proof_system, circuit_id_hex, model_hash)` triple does not match any entry in `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`, AND `proof_system` is not in the legacy `MAINNET_APPROVED_PROOF_SYSTEMS` back-compat alias (refusal layer 6) | yes — typed error | **Both lists ship empty through Stage 11d.1 / 11d.2 by design** — only a Stage 11d.3 entry with written chain-team sign-off populates the structured list. Stage 11d.1 introduced the structured `AllowlistEntry`-keyed list; the legacy `&[ProofSystem]` slice is preserved as an empty back-compat alias. See [docs/mainnet-eligibility-criteria.md](mainnet-eligibility-criteria.md). | backend_id, proof_system |
| `NoVerifierForProofSystem` (**Stage 11b.0**) | `operator verify-proof` was handed an artifact whose `proof_system` has no verifier registered | yes — typed error | Stage 11b.0 ships only `MockProofVerifier`; Stage 11b.1.b/11c add `Halo2ReferenceVerifier` under the opt-in `halo2-reference-verify` feature. Other proof systems are verifier-side stubs awaiting future stages. | proof_system |

> **Known limitation flagged by Stage 10a.** [`ChainClientError`](../crates/omni-zkml/src/error.rs)
> is currently the single-variant `Other(String)`. That is why several rows
> above are marked "partially" — fee, balance, transport, and signature
> failures all funnel through the same variant and require string-level
> triage. Stage 10a deliberately does **not** add new error variants;
> splitting `ChainClientError` into a typed taxonomy is a candidate for a
> future stage if operator feedback shows the parsing burden is real.

### 11c. GGUF proofs — what's possible today (Stage 11b.0–11d.0)

OmniNode's canonical model format today is **GGUF** (llama.cpp). Stage
11b.0 ships the schema slot for declaring `model_format = "gguf"` on
proof artifacts, but **no GGUF inference proof backend is approved at
any stage through Stage 11d.0**.

The honest framing — to be repeated wherever Stage 11b/c/d is described:

- Full GGUF transformer inference proving is **not feasible** end-to-
  end with any current production-ready proof system. This is an open
  research problem.
- **GGUF is explicitly NOT a Stage 11d candidate.** The Stage 11d.0
  mainnet-eligibility criteria record GGUF as deferred to Stage 11e
  research. The same criteria record that any future GGUF strategy
  would require its own chain-team plan and would not auto-inherit
  Stage 11d allowlist mechanics.
- A future Stage 11e research track could evaluate **one or more** of
  the following strategies. None of them prove full transformer
  inference correctness:
  - **Shadow verifier circuit**: a small *verifier net* (provable via
    whichever production proof class Stage 11d.2 selects, OR via a
    future ONNX-with-licensed-prover path) ingests `(model_hash,
    input_hash, output_hash, derived features)` and outputs a binding.
    Proves the verifier-net computation, not the GGUF inference itself.
    (Note: the Stage 11b.1 ezkl-discovery spike rejected ezkl due to
    licensing; this strategy presumes a different licensed prover.)
  - **Partial-inference proof**: a zkVM circuit re-executes a single
    layer of the transformer (e.g., the final softmax) given hashed
    intermediate state. Proves that the final emitted token follows
    from the claimed intermediate activation.
  - **Replay-based attestation**: K independent operators run the
    same `(model, input)` and submit attestations; matching
    `output_hash` values are treated as "K-replicated." This is a
    **consensus** mechanism, not a proof system.
- Until a future stage ships an approved GGUF strategy, **declaring
  `model_format = "gguf"` on a proof artifact and attempting mainnet
  submission is hard-refused** by `check_mainnet_eligible`'s refusal
  layer 4 (`GgufProofClaimRefusedOnMainnet`). The refusal is the point
  — it prevents silent fake-GGUF claims.

This is the **decentralized proof architecture** — not "decentralized
compute readiness." The criteria for the latter are documented in the
Stage 11b.0 README section and **explicitly unmet** at end of Stage
11d.0.

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

---

## Phase 5 — Contributor operations

Operator-side playbook for the Stage 12 contributor subsystem of `omni-node`. For the engineering reference (schemas, canonical bytes, verifier semantics, halt findings, byte-stability guarantees), see [`stage12-contributor-protocol.md`](./stage12-contributor-protocol.md). This section is the **playbook** — what to run, why, and what to look at when things break.

### Stage 12 — production posture

Stage 12 ships a complete local contributor operations subsystem under `omni-node operator contributor`. **43 subcommands** across Stage 12.0–12.25. State mutation is local and operator-directed: mutating commands (`run-job`, the `watch-*` daemons writing results / verified bodies, `archive-session --move`, `apply-state-cleanup`, `restore-session-archive`, `restore-state-cleanup-quarantine`) are explicit operator actions, and state-store opens are explicit via `--contributor-state-dir` (with auto-prune-on-open unless `--no-prune-state-on-start` is set). No command makes a chain transaction; nothing is published to a chain registry. Every signed artifact is **operator-local provenance**, not external attestation.

The lifecycle splits into five phases:

| Phase | Stages | Commands | Use |
| --- | --- | --- | --- |
| **Dispatch & run** | 12.0–12.1 | `validate-job`, `run-job`, `verify-result`, `post-job`, `watch-jobs`, `publish-result-link` | Run a single inference job end-to-end. |
| **Mesh announcement** | 12.2 | `announce-job`, `watch-network-jobs`, `announce-result`, `watch-network-results` | Discover jobs / publish results on libp2p gossipsub. |
| **Pooled-memory sessions** | 12.3–12.5 | `open-session`, `join-session`, `assign-work`, `run-assignment`, `aggregate-session`, `watch-sessions`, `send-handoff`, `advertise-peer`, `watch-peer-adverts` | Multi-contributor cooperation on one inference job. |
| **Session lifecycle** | 12.8–12.13 | `plan-session-assignments`, `assign-session-plan`, `session-status`, `plan-session-repair`, `apply-session-repair`, `plan-session-reassign`, `apply-session-reassign` | Plan, monitor, repair, and reassign pooled-memory sessions. |
| **State-dir archive & forensic record** | 12.14–12.25 | `archive-session`, `restore-session-archive`, `state-integrity`, `plan-state-cleanup`, `apply-state-cleanup`, `restore-state-cleanup-quarantine`, `state-integrity-diff`, sign/verify families | Capture, sign, and verify operator-local forensic records of state-dir contents. |

**What to run nightly vs on-demand**

- **Long-running daemons** (24/7 supervisor): `watch-jobs` or `watch-network-jobs` (job pickup), `watch-sessions` (mesh subscription), `watch-peer-adverts` (peer routing cache), `watch-network-results` (result-link mirror).
- **Nightly batch** (cron / systemd timer): baseline-only forensic-record path is Stage 12.16 `state-integrity --json-out` → Stage 12.20 `sign-state-integrity-baseline` → Stage 12.22 `build-integrity-evidence-bundle` (including the baseline and signed-baseline as `--include` entries) → Stage 12.23 `sign-integrity-evidence-bundle` → Stage 12.24 `verify-integrity-evidence-chain --json-out` → Stage 12.25 `sign-integrity-evidence-chain-report`. Produces one tamper-evident operator-local forensic record per night. For a diff-included variant, insert Stage 12.19 `state-integrity-diff --json-out` (against a previous baseline) and Stage 12.21 `sign-state-integrity-diff` between the baseline-sign and the bundle-build, and add the diff and signed-diff to the bundle's `--include` list. Stage 12.24 is the load-bearing step that produces the chain-report JSON Stage 12.25 signs — skipping it leaves nothing for `sign-integrity-evidence-chain-report` to consume.
- **Ad-hoc** (on demand): everything in the "Dispatch & run" + "Pooled-memory sessions" + "Session lifecycle" rows. Operators run these explicitly when they want to dispatch a job, open a session, or triage a stuck session.

**Shared flags across the subsystem**

A handful of flags appear on multiple subcommands and carry the same meaning everywhere:

- `--contributor-state-dir <path>` — Stage 12.7 single-source-of-truth local state directory. Auto-prunes expired sessions on open unless `--no-prune-state-on-start` is set. Opt-in on Stage 12.0–12.6; required on most Stage 12.8+ commands.
- `--no-prune-state-on-start` — disable auto-prune for forensic / triage runs. Stage 12.16+ verifier-side commands deliberately **reject** this flag because they don't open a state-dir; the rejection is pinned by clap regression.
- `--net-identity-file <path>` — Stage 12.6 persistent libp2p keypair. Same path across restarts → stable `PeerId`. Missing → ephemeral identity (pre-12.6 behavior).
- `--snip-binary <bin>` / `--snip-seed <path>` — SNIP V2 adapter plumbing. Required when a command publishes or fetches SNIP content.
- `--peer-wait-secs <N>` / `--mesh-stabilize-ms <N>` — Stage 12.2+ mesh-publish defaults. A fresh `OmniNet` silently drops `publish` on an empty mesh; these flags wait for at least one peer + a mesh stabilize period before publishing.
- `--format events|json|pretty` — Stage 12.9+ standardized output format. `events` = bare-stdout `event=...` key-value lines, `json` = single pretty JSON document on stdout (chatter to stderr so `jq` works), `pretty` = terminal-friendly summary.

**Seed file hygiene**

Every signing role uses a **separate 32-byte raw Ed25519 seed file**. Stage 12.0 establishes the contributor seed; subsequent stages add coordinator, dispatcher, baseline-signer, diff-signer, bundle-signer, chain-report-signer roles. **Do NOT reuse seeds across roles** — the integrity-artifact signing roles are deliberately distinct from chain-attestation and protocol-role seeds. Smaller blast radius if compromised.

### Stage 12.0 — Contributor Inference Node protocol

**Use when:** running a single inference job end-to-end (validate, run, verify a `ContributorResult`). This is the building block every other contributor command composes.

**Commands:** `validate-job`, `run-job`, `verify-result`.

**Workflow:**

```sh
# 1. Validate a job's schema + dispatcher signature.
omni-node operator contributor validate-job --job ./job.json

# 2. Run inference with a stub runner (deterministic, for tests) or external.
omni-node operator contributor run-job \
  --job ./job.json \
  --out ./result.json \
  --runner stub --stub-response ./response.bin \
  --seed-file ./contributor.seed

# 3. Verify the result against the job (does NOT trust result.job_snip_root implicitly).
omni-node operator contributor verify-result --job ./job.json --result ./result.json
```

**Key flags:**

- `--runner stub|external` — `stub` reads a fixed response from disk (deterministic; for tests). `external` invokes a subprocess that does the real inference — no framework runtime lives in `omni-node` itself.
- `--external-command <bin>` — required with `--runner external`. The subprocess gets `--manifest <path>` + `--input <path>` (+ `--activation-out <path>` on Stage 12.4 wired runners) and writes a `deny_unknown_fields` JSON envelope to stdout.
- `--seed-file <path>` — 32-byte raw Ed25519 contributor seed. DISTINCT from any chain-attestation seed.

**Cost guardrails:** none at this stage. `watch-jobs` (Stage 12.1) and `watch-network-jobs` (Stage 12.2) add required cost caps; one-off `run-job` invocations are at operator discretion.

**Verifier semantics:** `verify-result` runs the full 10-step `verify_result` pipeline (schemas, hash recomputation, expiry, dispatcher signature, model/input agreement, SNIP fetch + BLAKE3 check, accounting checks, contributor signature, evidence-mode check). Off-chain only.

**Output:** bare-stdout `event=...` lines with a final `event=verify_result overall_ok=true|false`. Automation should grep the final line.

**No protocol surface touched.** No chain wire, no on-chain proof verification, `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` stays empty. AttestationOnly evidence mode only.

For canonical-bytes encoding, signature recipes, and reserved future-stage strings, see [`stage12-contributor-protocol.md`](./stage12-contributor-protocol.md) Stage 12.0.

### Stage 12.1 — filesystem-based job discovery

**Use when:** running a long-lived pickup loop against jobs posted to a directory. Extends Stage 12.0 with a SNIP-published `PostedJob` envelope, a `FilesystemSource` `JobSource`, and a long-running `watch-jobs` subcommand. SNIP V2 roots are content-addressed, so filesystem watching replaces an index-polling story until Stage 12.2 ships gossipsub.

**Commands:** `post-job`, `watch-jobs`, `publish-result-link`.

**Workflow:**

```sh
# Dispatcher side — publish a job + a PostedJob envelope.
omni-node operator contributor post-job \
  --job ./job.json --posted-out ./posted.json \
  --seed-file ./dispatcher.seed \
  --snip-binary sum-node

# Contributor side — long-running pickup loop with REQUIRED cost caps.
omni-node operator contributor watch-jobs \
  --source fs --jobs-dir /var/lib/omni-node/posted-jobs \
  --max-input-tokens 200000 \
  --max-output-tokens 1000000 \
  --max-total-base-units 1200000 \
  --runner external --external-command /usr/local/bin/my-inference-runner \
  --seed-file ./contributor.seed \
  --result-out-dir /var/lib/omni-node/results \
  --publish-result-link --snip-binary sum-node

# Contributor side — manually publish a PostedResultLink for one result.
omni-node operator contributor publish-result-link \
  --result ./result.json --posted-job ./posted.json \
  --link-out ./link.json \
  --seed-file ./contributor.seed --snip-binary sum-node
```

**Required cost caps on `watch-jobs`:** `--max-input-tokens`, `--max-output-tokens`, `--max-total-base-units` are all **required with no defaults**. A conservative default that fits a dev box is dangerous for a production contributor and vice-versa; the operator must make an explicit cap decision before any pickup happens. Caps are applied AFTER fetching the job from SNIP but BEFORE invoking the runner — a runner is never invoked for a job that would exceed any cap.

**Allow-lists:** `--accept-model-hash <hex64>` and `--accept-tokenizer-hash <hex64>` are repeatable. Empty allow-list = accept any. Operators running known-tokenizer pipelines should pin both.

**Post-run verifier:** every accepted result re-runs `verify_result` against the SNIP adapter before write-out. A failure writes `<result-out-dir>/<job_id>.rejected.json` with a structured `rejected_reason` and skips the optional `--publish-result-link` step. This catches orchestrator/runner schema drift before another party's verifier sees the result.

**Operational tips:** `--max-polls <N>` is primarily for tests / smoke runs. Production typically omits it and lets the loop run until `--max-jobs` is reached or the process supervisor sends SIGTERM. There is no persistent dedup state across `watch-jobs` restarts — the in-memory `seen` set is rebuilt on every start. Rely on `expires_at_utc` + `--max-jobs` to bound work.

**No protocol surface touched.** SNIP-only; no chain wire, no lease/claim primitives. Two contributors pointing at the same directory will both pick the same job and both produce results; dispatchers reading results choose which to trust.

For full schema definitions of `PostedJob` / `PostedResultLink` and the per-step verification pipeline, see [`stage12-contributor-protocol.md`](./stage12-contributor-protocol.md) Stage 12.1.

### Stage 12.2 — contributor mesh network relay

**Use when:** running gossipsub-based discovery instead of filesystem polling. Two new signed network announcements (`NetworkPostedJobAnnouncement`, `NetworkPostedResultAnnouncement`) carry SNIP pointers across libp2p. SNIP still stores everything; the mesh just announces what to fetch.

**Commands:** `announce-job`, `watch-network-jobs`, `announce-result`, `watch-network-results`.

**Workflow:**

```sh
# Dispatcher side — announce an already-published PostedJob to the mesh.
omni-node operator contributor announce-job \
  --posted-job ./posted.json \
  --seed-file ./dispatcher.seed \
  --include-tokenizer-hash \
  --listen-port 4001 --peer /ip4/<bootstrap>/tcp/4001/p2p/<peer-id> \
  --snip-binary sum-node

# Contributor side — same pickup loop as Stage 12.1 but driven by mesh announcements.
omni-node operator contributor watch-network-jobs \
  --max-input-tokens 200000 --max-output-tokens 1000000 --max-total-base-units 1200000 \
  --runner external --external-command /usr/local/bin/my-inference-runner \
  --seed-file ./contributor.seed \
  --result-out-dir /var/lib/omni-node/results \
  --listen-port 4001 --peer /ip4/<bootstrap>/tcp/4001/p2p/<peer-id> \
  --publish-result-link --snip-binary sum-node

# Contributor side — announce a result link.
omni-node operator contributor announce-result \
  --posted-result-link ./link.json \
  --seed-file ./contributor.seed \
  --listen-port 4001 --peer /ip4/<bootstrap>/tcp/4001/p2p/<peer-id>

# Consumer side — passive watcher that mirrors result-link JSONs to disk.
omni-node operator contributor watch-network-results \
  --result-out-dir /var/lib/omni-node/result-links \
  --listen-port 4001 --peer /ip4/<bootstrap>/tcp/4001/p2p/<peer-id> \
  --snip-binary sum-node
```

**Mesh topics (frozen):**

- `omni/contributor/job/v1` — job announcements
- `omni/contributor/result/v1` — result announcements

Both carry the Stage 12.2-pre topic-safety lift, so unknown topics fail loudly with a typed `UnknownTopic` error instead of silently routing to `TOPIC_TEST`.

**Announcer signature is REQUIRED on both announcement types** — different from `PostedJob.poster_signature_hex` (which is legitimately optional for local-CLI handoffs). The inner envelope's own signature (`PostedResultLink.contributor_signature_hex`) is still verified after SNIP fetch.

**Bootstrap connectivity:** A bare `OmniNet::new()` is not yet connected to any peer, so `publish` on an empty mesh is a silent drop. Every `announce-*` and `watch-network-*` subcommand waits for at least one `PeerDiscovered` or `PeerConnected` event before publishing, with a `--peer-wait-secs <N>` (default 30) timeout. `--mesh-stabilize-ms <N>` (default 500) gives gossipsub a chance to form the topic mesh after the first peer event.

**`watch-network-results` filtering:** repeat `--posted-id <hex64>` to subscribe only to results for specific posted jobs. Empty filter = accept any.

**No protocol surface touched.** No persistent job/result dedup across restarts. No lease/claim primitives. No reputation. No real-network CI test (would need bootstrap infrastructure).

For full schemas + the announcement-then-fetch validation pipeline, see [`stage12-contributor-protocol.md`](./stage12-contributor-protocol.md) Stage 12.2.

### Stage 12.3 — multi-contributor pooled-memory sessions

**Use when:** multiple contributors cooperate on one inference job under a signed session envelope. Each participant signs a partial; one coordinator signs the aggregate. Inter-stage handoff at this stage uses SNIP (latency-tolerant); live tensor handoff is Stage 12.4.

**Commands:** `open-session`, `join-session`, `assign-work`, `run-assignment`, `aggregate-session`, `watch-sessions`.

**Workflow (1-of-N pipeline via SNIP):**

```sh
# 1. Coordinator opens a session bound to a previously posted job.
omni-node operator contributor open-session \
  --posted-job ./posted.json \
  --coordinator-seed ./coord.seed \
  --expires-at-utc 2026-06-30T00:00:00Z \
  --net-identity-file ./omni-net.key --listen-port 4001 --peer /ip4/.../

# 2. Each contributor joins the session, advertising RAM + token caps.
omni-node operator contributor join-session \
  --execution-session-snip-root 0x... \
  --contributor-seed ./contrib.seed \
  --available-ram-bytes 17179869184 \
  --max-input-tokens 200000 --max-output-tokens 1000000 \
  --supported-work-unit-kind prefill-tokens --supported-work-unit-kind decode-tokens \
  --runner-kind "llama-3-70b-q4" \
  --net-identity-file ./omni-net.key --listen-port 4001 --peer /ip4/.../

# 3. Coordinator assigns each stage. Spec file is hand-edited at 12.3;
#    Stage 12.8 ships a planner that generates it from verified joins.
omni-node operator contributor assign-work \
  --execution-session-snip-root 0x... \
  --coordinator-seed ./coord.seed \
  --assignments-file ./assignments.json

# 4. Each contributor runs its assignment + publishes a signed partial.
omni-node operator contributor run-assignment \
  --assignment-snip-root 0x... \
  --execution-session-snip-root 0x... \
  --contributor-seed ./contrib.seed \
  --runner external --external-command /usr/local/bin/my-stage-runner \
  --net-identity-file ./omni-net.key

# 5. Coordinator aggregates the partials into a final ContributorResult.
omni-node operator contributor aggregate-session \
  --execution-session-snip-root 0x... \
  --coordinator-seed ./coord.seed \
  --final-result-snip-root 0x... \
  --join-snip-root 0x... --assignment-snip-root 0x... --partial-snip-root 0x... \
  --join-snip-root 0x... --assignment-snip-root 0x... --partial-snip-root 0x...

# Long-running mesh watcher — verifies + writes session/join/assign/partial/aggregate
# announcements to --out-dir (or to a Stage 12.7 state-dir).
omni-node operator contributor watch-sessions \
  --out-dir /var/lib/omni-node/contrib-state/verified/sessions \
  --listen-port 4001 --peer /ip4/.../ \
  --net-identity-file ./omni-net.key
```

**Roles (separate seeds):**

- **Coordinator** — signs `ExecutionSession`, every `WorkAssignment`, and the `AggregatedContributorResult`. Does not need to run inference.
- **Contributor** — signs `ContributorJoin` and each `PartialContributorResult`.
- **Coordinator-as-contributor** — same operator can hold both keys; protocol doesn't care, but reuse the same seed and you lose blast-radius separation.

**Accounting rule (important):** the final `ContributorResult.measured_accounting.total_base_units` is the **job-level** input + output token count (same as Stage 12.0). It is NOT a sum of partial totals — that would inflate to N× the real cost for N stages. Partials carry their own `stage_contributions` for future reward-split policies; the verifier checks structure but does not require numerical equality with the final.

**Topology constraint (v1):** the verifier enforces strict linear topology — `aggregate.partial_refs` must cover every assignment exactly once, and aggregating without a partial for some assignment refuses with `AggregateMissingPartialFor`. Replacement / supersession is Stage 12.11.

**Mesh topics:** `omni/contributor/session/{open,join,assign,partial,aggregated,peer-advert}/v1`.

**No protocol surface touched.** No live tensor transport (Stage 12.4). No auto-selection policy. No sybil / anti-spam beyond required signatures.

For envelope schemas + the per-stage verifier helpers, see [`stage12-contributor-protocol.md`](./stage12-contributor-protocol.md) Stage 12.3.

### Stage 12.4 — live tensor / activation transport

**Use when:** the SNIP round-trip latency in Stage 12.3's pipeline is too slow. Stage 12.4 adds a direct peer-to-peer activation handoff over `omni-net`'s tensor request/response codec. SNIP keeps its role as durable backup / fallback.

**Commands:** `run-assignment` (extended with handoff flags), `send-handoff` (diagnostic).

**Workflow (live downstream handoff):**

```sh
# Receiver — wait for upstream activation, then run.
omni-node operator contributor run-assignment \
  --assignment-snip-root 0x... \
  --execution-session-snip-root 0x... \
  --contributor-seed ./contrib.seed \
  --runner external --external-command /usr/local/bin/my-stage-runner \
  --activation-in-mode live \
  --upstream-from-assignment-snip-root 0x... \
  --upstream-wait-secs 60 \
  --downstream-to-assignment-snip-root 0x... \
  --downstream-to-peer /ip4/<peer-ip>/tcp/4001/p2p/<peer-id> \
  --activation-out-mode both \
  --net-identity-file ./omni-net.key --listen-port 4001 --peer /ip4/<bootstrap>/

# Diagnostic — send a single activation file out of band.
omni-node operator contributor send-handoff \
  --execution-session-snip-root 0x... \
  --from-assignment-snip-root 0x... --to-assignment-snip-root 0x... \
  --from-contributor-seed ./contrib.seed \
  --activation-file ./activation.bin \
  --dtype f16 --shape 1,4096 \
  --to-peer /ip4/<peer-ip>/tcp/4001/p2p/<peer-id>
```

**Activation modes:**

| `--activation-out-mode` | Live send | SNIP publish | Default |
| --- | --- | --- | --- |
| `snip` | no | yes (12.3 behavior) | when no downstream peer supplied |
| `live` | yes | no | — |
| `both` | yes | yes | when downstream peer + assignment supplied |
| `none` | no | no | — |

`--activation-in-mode live` reads upstream activations from the live mesh (waits up to `--upstream-wait-secs`); `none` means this is the first stage. SNIP-side `activation-in-mode snip` is NOT implemented in v1 — operators wiring SNIP fallback extract bytes off the upstream partial out-of-band.

**Bounds (schema-enforced):** single chunk ≤ 64 MiB, total ≤ 16 GiB, max 256 chunks per stream, shape rank ≤ 8. Override the chunk size with `--handoff-chunk-max-bytes <N>` if your receiver has tighter memory limits.

**Two-step integrity check on receive:** signature → BLAKE3 hash → byte-length. A failing signature is rejected before reassembly; a failing hash after reassembly is `TensorHashMismatch`; a length mismatch is `ByteLenMismatch`. The outer `TensorRequest` fields are routing-only — the verifier never trusts them.

**Topology constraint:** `to.stage_index == from.stage_index + 1` (strict linear, v1). Non-linear / branching pipelines are Stage 12.4+ scope.

**PeerId hint:** `--downstream-to-peer` accepts either a bare PeerId base58 or a multiaddr whose last `/p2p/<peer-id>` component is the destination. Stage 12.5 ships signed peer advertisements so you don't have to wire this manually.

**No protocol surface touched.** No model-specific tensor semantics. Bytes are opaque. AttestationOnly only.

For envelope schema + the two-step integrity check details, see [`stage12-contributor-protocol.md`](./stage12-contributor-protocol.md) Stage 12.4.

### Stage 12.5 — signed contributor peer advertisement / session routing

**Use when:** running a multi-contributor session and wanting `run-assignment` to resolve downstream peers automatically instead of you wiring `--downstream-to-peer` manually. Each contributor publishes a signed, session-scoped peer advertisement; `run-assignment` looks it up from the local routing cache.

**Commands:** `advertise-peer`, `watch-peer-adverts`, `run-assignment --resolve-downstream-peer-from-session`.

**Workflow:**

```sh
# 1. Each contributor advertises its libp2p PeerId for one session.
#    PeerId comes from OmniNet::local_peer_id() — no --libp2p-peer-id flag,
#    so operators can't lie about what node they're running.
omni-node operator contributor advertise-peer \
  --execution-session-snip-root 0x... \
  --join-snip-root 0x... \
  --contributor-seed ./contrib.seed \
  --net-identity-file ./omni-net.key \
  --listen-multiaddr /ip4/<external-ip>/udp/4001/quic-v1 \
  --max-handoff-chunk-bytes 33554432 \
  --supported-dtype f16 --supported-dtype bf16 \
  --expires-in-secs 3600 \
  --publish-announcement

# 2. Long-running passive watcher — drains advertisement announcements,
#    verifies inner contributor signatures, matches against verified joins.
omni-node operator contributor watch-peer-adverts \
  --out-dir /var/lib/omni-node/contrib-state/verified/sessions \
  --joins-dir /var/lib/omni-node/contrib-state/verified/sessions \
  --net-identity-file ./omni-net.key

# 3. run-assignment resolves the downstream peer from the cache.
omni-node operator contributor run-assignment \
  --assignment-snip-root 0x... \
  --execution-session-snip-root 0x... \
  --downstream-to-assignment-snip-root 0x... \
  --peer-advert-dir /var/lib/omni-node/contrib-state/verified/sessions \
  --joins-dir /var/lib/omni-node/contrib-state/verified/sessions \
  --resolve-downstream-peer-from-session \
  --contributor-seed ./contrib.seed \
  --runner external --external-command /usr/local/bin/my-stage-runner
```

**Freshness:** advertisements are ≤ 24h by schema validation. PeerIds regenerate on every `omni-net` restart UNLESS you supply `--net-identity-file` (Stage 12.6). Without persistent identity, expect to re-run `advertise-peer` after every restart.

**Routing cache resolution outcomes:**

- `Found(ResolvedPeerRoute { ... })` — happy path
- `NoAdvertisement` — nothing cached for this `(session_id, contributor_pubkey)` key
- `AllExpired { newest_expires_at }` — advertisement exists but past expiry
- `LiveHandoffNotSupported` — advertisement's `supports_live_handoff` is false
- `DtypeNotSupported { requested, supported }` — your `--downstream-resolve-dtype` isn't in the advertised list

**Trust boundary:** `watch-peer-adverts` and `run-assignment` BOTH re-verify joins from the `--joins-dir` (running `verify_execution_session` on the sibling `session.json` followed by `verify_contributor_join`). A forged local join file cannot make a forged peer advert pass the matching-join gate.

**Precedence:** `--downstream-to-peer` still works and TAKES PRECEDENCE if both flags are supplied. Use this for one-off overrides.

**No protocol surface touched.** No registry, no marketplace. Two parallel sessions for the same `posted_id` cache their own advertisements independently. Reputation / scoring is out of scope.

For envelope schema + the advertisement processor pipeline, see [`stage12-contributor-protocol.md`](./stage12-contributor-protocol.md) Stage 12.5.

### Stage 12.6 — persistent libp2p mesh identity

**Use when:** running a contributor across restarts and you want peer advertisements to survive the freshness window. Stage 12.6 adds `--net-identity-file <path>` to every contributor subcommand that opens `OmniNet`.

**Commands:** none new — `--net-identity-file <path>` is a flag on every contributor subcommand that opens `OmniNet` (every announce-*, watch-*, open-session, join-session, assign-work, run-assignment, aggregate-session, send-handoff, advertise-peer, watch-peer-adverts).

**Workflow:**

```sh
# First run creates ./omni-net.key at 0600 with a fresh libp2p keypair.
omni-node operator contributor advertise-peer \
  --net-identity-file ./omni-net.key \
  --execution-session-snip-root 0x... \
  --join-snip-root 0x... \
  --contributor-seed ./contributor.seed \
  --max-handoff-chunk-bytes 33554432 \
  --supported-dtype f16 \
  --expires-in-secs 3600 \
  --publish-announcement

# Restart omni-node; re-run with the SAME --net-identity-file → same PeerId.
# Previously-published peer advertisements remain valid until expires_at_utc.
```

**Failure posture (intentional):**

| Condition | Behavior |
| --- | --- |
| Missing file | Auto-create at `0o600` (Unix) on first use. No separate `init-net-identity` command. |
| Existing file with malformed bytes | Typed `IdentityError::Decode`. Loader does NOT silently fall back or overwrite. |
| Existing file with world/group-readable perms (Unix) | Typed `IdentityError::PermissionsTooBroad { mode }`. File is NOT modified. |
| Non-regular file (directory, symlink loop) | Typed `IdentityError::NotARegularFile`. |

**Role separation:** the libp2p mesh transport identity is NOT the contributor signing identity. `--contributor-seed` / `--coordinator-seed` / `--dispatcher-seed` stay on their own files; `--net-identity-file` is a separate role. Don't reuse a contributor seed as a `--net-identity-file`.

**Rotation:** delete the identity file and re-run with the same path. A fresh keypair is auto-created; the next `advertise-peer` publishes a new advertisement with the new PeerId.

**Advertisements stay short-lived.** Stage 12.6 stabilizes the PeerId across restart but does NOT make advertisements permanent. The ≤ 24h schema cap stays. Operators wanting continuous routing across the freshness boundary re-run `advertise-peer` manually before expiry. Auto-refresh is intentionally deferred.

For the failure-mode error taxonomy + multi-binary identity-file precedent, see [`stage12-contributor-protocol.md`](./stage12-contributor-protocol.md) Stage 12.6.

### Stage 12.7 — local contributor workflow state persistence

**Use when:** running ANY long-running contributor command and you want restart-resumable dedup + verified-bodies cache. Stage 12.7 adds `--contributor-state-dir <path>` to every restart-sensitive subcommand, backed by a versioned, atomic-write JSON tree.

**Commands:** none new — `--contributor-state-dir <path>` is a flag on `watch-jobs`, `watch-network-jobs`, `watch-network-results`, `watch-sessions`, `watch-peer-adverts`, `run-assignment` (when `--resolve-downstream-peer-from-session` is set).

**Layout:**

```text
<state-dir>/
  meta/state_version.json
  seen/posted-jobs/<posted_id>                            # dedup markers (0-byte)
  seen/network-job-announcements/<posted_id>
  seen/sessions/<session_id>
  seen/joins/<session_id>--<contributor_pubkey>
  seen/assignments/<session_id>--<assignment_id>
  seen/partials/<session_id>--<assignment_id>
  seen/aggregates/<session_id>
  seen/peer-adverts/<session_id>--<contributor_pubkey>
  verified/sessions/<session_id>/session.json             # full verified bodies
  verified/sessions/<session_id>/joins/<pubkey>.json
  verified/sessions/<session_id>/assignments/<id>.json
  verified/sessions/<session_id>/partials/<id>.json
  verified/sessions/<session_id>/aggregated.json
  verified/sessions/<session_id>/peer-adverts/<pubkey>.json
  results/contributor-results/<job_id>.json
  results/contributor-results/<job_id>.rejected.json
  results/result-links/<posted_id>.link.json
```

**Workflow (canonical single-state-dir setup):**

```sh
# One state directory drives every restart-resumable command.
omni-node operator contributor watch-sessions \
  --net-identity-file ./omni-net.key \
  --contributor-state-dir ./contrib-state \
  --out-dir ./contrib-state/verified/sessions \
  --listen-port 4001 --peer /ip4/.../

omni-node operator contributor watch-peer-adverts \
  --net-identity-file ./omni-net.key \
  --contributor-state-dir ./contrib-state \
  --out-dir ./contrib-state/verified/sessions \
  --joins-dir ./contrib-state/verified/sessions

omni-node operator contributor run-assignment \
  --net-identity-file ./omni-net.key \
  --contributor-state-dir ./contrib-state \
  --resolve-downstream-peer-from-session \
  ...
```

`--out-dir` points INTO the state-dir's `verified/sessions` subtree on purpose — a single directory serves both the operator-facing "I want to inspect verified envelopes" use case and the state-store's restart-resume use case.

**Auto-prune:** `ContributorStateStore::open` auto-prunes by default — any verified session whose `expires_at_utc` has passed is removed (with cascade through joins/assignments/partials/aggregate/peer-adverts + every matching `seen/` marker). Pass `--no-prune-state-on-start` to disable for forensic re-runs.

**Conflict policy (run-assignment):** when `--resolve-downstream-peer-from-session` is set, supplying `--contributor-state-dir` together with the legacy `--peer-advert-dir` or `--joins-dir` flags refuses with `StateError::AmbiguousSource { legacy_flag }`. The point of the state-dir is to BE the single source of truth.

**Versioning + atomic writes:** `meta/state_version.json` carries the current `STATE_VERSION` (`2` as of Stage 12.16's lift). Restore (Stage 12.15) and diff (Stage 12.19) enforce strict equality against the binary's `STATE_VERSION` — an archive or report produced by a different state-version refuses cleanly with a typed error. Every write goes through tempfile-in-same-directory + `fs::rename`, so a torn write never appears at the final path. `mark_seen` is idempotent.

**Safety properties:**

- No private key material in the state-dir. Contributor seed and libp2p identity stay on their own files.
- Schema-versioned, not chain-anchored. The state-dir is a local cache; nothing trusts it as evidence. Verification of restored bodies still re-runs the Stage 12.3 / 12.4 / 12.5 verifiers.
- Inspectable. Everything is pretty-printed JSON; `cat` and `jq` freely.
- Concurrent-safe (single-host). Don't run two `omni-node` processes against the same state-dir.

**Gradual migration from pre-12.7 layouts:** the `verified/sessions/<id>/...` subtree under the state-dir is the same per-session shape as the pre-12.7 `--out-dir <X>` tree. Pointing `--contributor-state-dir <X>` at an existing flat `<X>/<id>/...` tree does NOT migrate it. Either (a) start fresh with a new path and let the new envelopes populate naturally, or (b) `mv <X>/<id> <X>/verified/sessions/<id>` for each per-session subdirectory in place.

For the loader pipeline + Stage 12.7-on-Stage-12.8/12.9 trust-boundary defense in depth, see [`stage12-contributor-protocol.md`](./stage12-contributor-protocol.md) Stage 12.7.

### Stage 12.8 — local pooled-session assignment planner

**Use when:** running a pooled-memory session and you want the assignment shape generated from the state-dir's verified joins + peer adverts instead of hand-edited spec files.

**Commands:** `plan-session-assignments`, `assign-session-plan`.

**Workflow:**

```sh
# 1. Open the session (12.3) + receive joins + accept peer adverts (12.5).
omni-node operator contributor open-session ...
omni-node operator contributor watch-sessions \
  --net-identity-file ./omni-net.key \
  --contributor-state-dir ./contrib-state \
  --out-dir ./contrib-state/verified/sessions ...

# 2. Plan the assignments locally (no network, no SNIP).
omni-node operator contributor plan-session-assignments \
  --contributor-state-dir ./contrib-state \
  --session-id <hex64> \
  --strategy sequential-layers \
  --layer-count 32 \
  --out ./plan.json

# 3. Review ./plan.json. Optionally dry-run (locally signs + schema-validates each entry).
omni-node operator contributor assign-session-plan \
  --plan ./plan.json \
  --session-snip-root 0x... \
  --coordinator-seed ./coord.seed \
  --dry-run

# 4. Publish each assignment (signed + SNIP-published + mesh-broadcast).
omni-node operator contributor assign-session-plan \
  --plan ./plan.json \
  --session-snip-root 0x... \
  --coordinator-seed ./coord.seed \
  --contributor-state-dir ./contrib-state \
  --net-identity-file ./omni-net.key
```

**Strategies (closed enum, v1):**

- `single-contributor` — one contributor handles the entire work envelope.
- `sequential-layers` (default) — equal layer split, remainder absorbed by the last stage. With `--model-plan <path>`: one stage per model-plan entry, round-robin across the sorted eligible set.
- `round-robin` — `stage_index N → eligible[N mod eligible.len()]`. Without `--model-plan`, requires `--layer-count N` and emits N single-layer stages cycling through the eligible set.

**Eligibility filter (pass/fail, NOT ranking):**

- `--min-available-ram-bytes <u64>` — floor (default 0). Two contributors that both clear the floor are interchangeable to the planner.
- `--require-live-routing` — require a non-expired peer advertisement for `(session_id, contributor_pubkey_hex)` with `supports_live_handoff == true` AND advertised dtype containing `--required-dtype`.

**Determinism:** after eligibility filtering, contributors are sorted by `contributor_pubkey_hex` (lexicographic ASCII on the lowercase hex). Re-running with the same inputs (and the same `now_utc` if any advert sits on the eligibility boundary) yields byte-identical output including `plan_hash`.

**Why no RAM-weighting:** a planner that ranks contributors by RAM is a scheduler; a planner that signs winner declarations is a marketplace. Both are deliberately outside Stage 12.8's posture.

**`--max-assignments` semantics:** never silently truncating. Refuses with `PlannerError::MaxAssignmentsTooSmall` if the cap would leave the work envelope incomplete.

**Trust boundary:** the planner re-runs `verify_execution_session`, `verify_contributor_join`, and `verify_peer_advertisement_body` on every artifact loaded from the state-dir before feeding it to the strategy. `assign-session-plan` re-fetches and re-verifies the session from SNIP at publish time, then re-derives every signature. The on-disk plan is a SUGGESTION, not a trust anchor — `assign-session-plan` recomputes `plan_hash` on read and refuses on drift.

**Session expiry:** both subcommands refuse to operate when `now_utc >= session.expires_at_utc`. `--no-prune-state-on-start` cannot reach a signed `WorkAssignment` against a stale session.

**No protocol surface touched.** `AssignmentPlan` is unsigned, local-only. Stage 12.3 `assign-work` still works for hand-edited spec files; Stage 12.8 is the planner ergonomics layer on top.

For the strategies' work-envelope coverage rules + the model-plan shape, see [`stage12-contributor-protocol.md`](./stage12-contributor-protocol.md) Stage 12.8.

### Stage 12.9 — local pooled-session progress monitor

**Use when:** inspecting whether a pooled session is complete, stuck, expired, or has tampered artifacts. Read-only — never signed, never SNIP-published, never network-visible.

**Commands:** `session-status`.

**Workflow:**

```sh
# Events output (default) — grep-friendly bare-stdout key=value lines.
omni-node operator contributor session-status \
  --contributor-state-dir ./contrib-state \
  --session-id <hex64>

# JSON for a dashboard scraper (single document on stdout, chatter to stderr).
omni-node operator contributor session-status \
  --contributor-state-dir ./contrib-state \
  --session-id <hex64> \
  --format json \
  --json-out ./status.json

# In a CI gate (exit nonzero unless CompletePartials or Aggregated).
omni-node operator contributor session-status \
  --contributor-state-dir ./contrib-state \
  --session-id <hex64> \
  --fail-on-incomplete
```

**`SessionOverallStatus` decision tree (first match wins):**

1. `NoSession` — no `session.json` for the requested id.
2. `InvalidState` — any chain-link body failed individual re-verification, OR `verify_aggregated_result` rejected an aggregate that exists.
3. `Aggregated` — `aggregate_present && aggregate_valid`.
4. `ExpiredIncomplete` — `now_utc >= session.expires_at_utc` AND not aggregated.
5. `NoAssignments` — session valid, zero valid assignments.
6. `CompletePartials` — every valid assignment has exactly one valid partial; no aggregate yet.
7. `InProgress` — otherwise.

**`--fail-on-incomplete` exit codes:**

| `overall_status` | Exit code |
| --- | --- |
| `Aggregated` / `CompletePartials` | 0 |
| Everything else | 1 |

Without `--fail-on-incomplete`, every status exits 0 so dashboard scrapers can run unconditionally.

**Counts policy:** counts are **valid-only**. Tampered artifacts surface as `notes` + `InvalidState` overall, not as inflated counts.

**Trust boundary:** state-dir loaders are parse-only (Stage 12.7 review). `session-status` re-runs `verify_execution_session`, `verify_contributor_join`, `verify_work_assignment`, `verify_partial_result`, `verify_peer_advertisement_body`, and `verify_aggregated_result` before counting anything. A failed `verify_execution_session` is **fail-closed** — returns a minimal `InvalidState` report carrying no session-derived fields, with a single `notes` entry naming the verifier so operators can find the bad file.

**Audit ergonomics (Stage 12.13 lift):** the events output includes an `event=audit_health session_id=... coherence=<state> recommended_action="..."` summary line. The closed-set coherence values (`coherent`, `orphan_replacement_assignments`, `partial_apply_supersession`, `reassign_triagable`, `invalid_state`, etc.) and the recommended-action strings are stable and grep-friendly.

**No protocol surface touched.** `SessionStatusReport` is unsigned, local-only. Two operators on different machines will see different reports for the same session, and that's correct — each describes its local state-dir.

For the per-`overall_status` recommended-action strings + the Stage 12.13 audit-health closed set, see [`stage12-contributor-protocol.md`](./stage12-contributor-protocol.md) Stage 12.9 + Stage 12.13.

### Stage 12.10 — local pooled-session repair planner

**Use when:** `session-status` reports `InProgress` with N missing partials and you want to nudge contributors via a re-broadcast (without changing the assignment IDs or coordinator signatures).

**Commands:** `plan-session-repair`, `apply-session-repair`.

**Workflow:**

```sh
# 1. Inspect.
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

**Strategy (v1, closed enum):** `reannounce-missing`. The applier re-publishes the on-disk assignment JSON bytes VERBATIM. SNIP is content-addressed → returned root equals the original publish's root. The `assignment_id`, the coordinator signature, and the SNIP root are all preserved across re-announce. Only the network announcement is fresh.

**`--build-status` xor `--status-report <path>`:** clap-enforced mutual exclusion. `--build-status` reads the state-dir directly; `--status-report` lets you re-use a saved `SessionStatusReport`.

**Refused status branches** (`plan-session-repair`):

| Status | Error |
| --- | --- |
| `NoSession` | `RepairError::SessionNotPresent` |
| `NoAssignments` / `CompletePartials` / `Aggregated` | `RepairError::NothingToRepair` |
| `InvalidState` | `RepairError::InvalidState` (clean tampered artifacts first; no `--allow-invalid-state` flag) |
| `ExpiredIncomplete` | `RepairError::SessionExpired` (extend the session via `open-session` first if you mean it) |

Only `InProgress` produces a non-empty plan.

**Trust boundary at apply time:** the applier (a) recomputes `repair_plan_hash` and refuses on drift, (b) recomputes `source_status_hash` from the CURRENT state-dir and refuses on drift (a partial may have arrived between plan and apply), (c) re-runs the eligibility check on the current status, (d) re-fetches and re-verifies the session via `--session-snip-root`, (e) re-verifies each referenced assignment from the state-dir before any SNIP write. The on-disk plan is a SUGGESTION.

**`--no-publish-announcements` is SNIP-only:** when set, the applier skips the entire omni-net layer (no mesh open, no peer wait, no propagation sleep, no shutdown). Useful for triage on a machine with no peers.

**Halt-finding context:** v1 deliberately ships ONLY `reannounce-missing`. `ReassignMissing` would require a supersession envelope; that landed in Stage 12.11 and is exercised through `plan-session-reassign` / `apply-session-reassign` (see the next section).

**Byte preservation on reannounce:** the aggregate verifier sees exactly one assignment per id — the same one it would have seen if the contributor had never lost the original announcement.

**No protocol surface touched.** `SessionRepairPlan` is unsigned, local-only. No new gossipsub topic. No state-dir mutation on apply.

For the `source_status_hash` projection rules + the `--no-prune-state-on-start` interaction with apply-time eligibility checks, see [`stage12-contributor-protocol.md`](./stage12-contributor-protocol.md) Stage 12.10.

### When to use which `--reason`

`plan-session-reassign` and `apply-session-reassign` accept a closed `--reason` flag controlling the `SupersessionReason` embedded in the signed supersession:

| `--reason` | Use when | Apply-time `InvalidState` policy |
|---|---|---|
| `missing-partial` (default) | An active assignment has no partial after the assigned contributor went offline / dropped the work. Status is `InProgress` with an active-missing entry. | **Refused.** Operator must clean tampered artifacts first. |
| `invalid-partial` | A contributor returned a partial whose `verify_partial_result` fails. Status flips to `InvalidState` with a structured `invalid_partial` diagnostic. | **Accepted** only when every `invalid_artifacts` entry is `InvalidPartial` whose `assignment_id` is in the plan. |
| `operator-rebalance` | Coordinator-driven re-assignment for capacity or fairness reasons; no chain failure. | **Refused.** Same posture as `missing-partial`. |

The Stage 12.13 audit roll-up at the bottom of `session-status` tells you which reason applies:

```text
event=audit_health session_id=... coherence=reassign_triagable \
  triagable_by_reassign=true \
  recommended_action="run plan-session-reassign --reason invalid-partial"
```

### Recovery from a Phase B partial-apply

`apply-session-reassign` is split into Phase A (SNIP publish every body) and Phase B (state-dir write + mesh broadcast). Phase B is further split B1..B4:

- **B1**: write each replacement assignment to the state-dir.
- **B2**: write the supersession to the state-dir.
- **B3**: broadcast each replacement on the mesh.
- **B4**: broadcast the supersession on the mesh (only if B3 was 100% clean).

If B1 succeeded but B2 failed (rare — single FS op), the state-dir has replacement assignments without the retiring supersession. Detection + recovery:

1. **Run `session-status --format events`** against the affected session. Look for the `event=audit_health` line:
   ```text
   event=audit_health ... coherence=orphan_replacement_assignments count=N ids=<id>,<id>... \
     recommended_action="clean state-dir orphan replacements before retry"
   ```
   The `ids` list names exactly the replacement assignments that landed in state without a retiring supersession.

2. **Verify the partial-apply suspicion** before deleting anything. Compare:
   - The reassignment plan you ran (its `actions[].replacement_assignment_id` set should equal the orphan list).
   - The contents of `<state-dir>/verified/sessions/<session_id>/assignments/<id>.json` for each orphan — `assigned_at_utc` should be close to your apply timestamp.
   - The contents of `<state-dir>/verified/sessions/<session_id>/supersessions/` — should NOT contain the supersession your plan would have produced.

3. **Manually delete the orphan assignment files** + their seen markers:
   ```bash
   for ID in <orphan-ids>; do
     rm <state-dir>/verified/sessions/<session_id>/assignments/$ID.json
     rm <state-dir>/seen/assignments/<session_id>--$ID
   done
   ```

4. **Re-run** `plan-session-reassign` (the status now matches the pre-apply shape; the new plan's `source_status_hash` will reflect that). Then `apply-session-reassign` with the new plan.

If B3 (replacement broadcast) failed but B1 + B2 succeeded, the state-dir is fully coherent — no cleanup needed. Operator can either re-run apply (no-op on SNIP, idempotent on state-dir, re-attempts mesh broadcasts) or manually re-broadcast via `--no-publish-announcements=false` on a fresh apply once the mesh issue is resolved.

**Exit codes**: `apply-session-reassign` exits **nonzero** when the mesh-broadcast state is incoherent (B3 partially broadcast and B4 skipped, OR B3 broadcast cleanly but B4 itself errored). The state-dir is coherent in every case — exit-nonzero is purely the "peers see a half-applied view" signal. Automation should treat any nonzero exit as a triage hand-off, NOT a state-dir corruption. The `event=reassign_applied … supersession_broadcast=<status> replacement_broadcast_failures=N` line right before exit names the failure mode (closed set; pattern-matchable).

### Out-of-order supersession arrival

The Stage 12.13 watch-sessions handler accepts a supersession in best-effort mode when local state hasn't yet seen the referenced replacement assignments. You'll see:

```text
event=supersession_partial_verify session_id=... supersession_id=... unresolved_assignment_count=K
```

This is **not** an error. The watcher stored the supersession; the aggregate verifier at `session-status` time will re-check references against the full state-dir snapshot. If references resolve later (the missing replacement assignment announcements arrive in a future poll cycle), `session-status` will report `coherence=coherent`. If they never resolve, `session-status` will report `coherence=partial_apply_supersession` and recommend operator triage.

### Reading the structured diagnostics

Stage 12.12's v3 status report surfaces every chain-failure mode as a typed `invalid_artifacts` entry. Each entry carries a stable `reason_tag` from `SessionVerifyOutcome::reason_tag()` (the tags are pinned by the test `session_verify_outcome_reason_tags_are_stable`). Closed `kind` set:

| `kind` | Triagable via `--reason invalid-partial`? | Operator action |
|---|---|---|
| `invalid_partial` | Yes, if every entry in `invalid_artifacts` is this kind. | Plan + apply reassign. |
| `invalid_join` | No. | Tampered local join file. Re-pull from chain / SNIP. |
| `invalid_assignment` | No. | Tampered local assignment file. Re-pull from coordinator. |
| `invalid_aggregate` | No. | Aggregator-side issue. Contact coordinator. |
| `invalid_supersession` | No. | Forged or schema-malformed supersession body. Delete from state-dir + re-pull. |
| `invalid_session` | No. | Session.json failed verification. State-dir compromised; rebuild from chain. |

### Startup logs

Stage 12.13 emits `event=state_store_restart_loaded` at watch-sessions startup with per-namespace accept/reject counts. Any rejection produces a structured warning:

```text
event=warn context=state_store_restart_load_rejected \
  kind=supersession session_id=<hex> id=<hex> reason_tag=<SessionVerifyOutcome variant>
```

The `reason_tag` set is closed — `grep`-able. A high rejection count after restart usually means either (a) the state-dir is corrupted (run a backup-and-rebuild) or (b) a coordinator key was rotated and old artifacts no longer verify (operator policy decision — re-pull from chain or accept the historical gap).

### Archiving completed sessions (Stage 12.14)

`prune_expired` (Stage 12.7) auto-removes sessions past `expires_at_utc` on watcher restart. **Aggregated sessions whose expires_at_utc is far in the future also accumulate** under `<state-dir>/verified/sessions/`, so over months the state-dir grows.

`omni-node operator contributor archive-session` is the operator-driven tool: copies (or moves) one session's subtree to a separate `<archive-dir>`, BLAKE3-verifies every copy, writes a manifest LAST, and (on `--move`) cascades the source out. **No chain, no SNIP, no mesh** — archives are inert JSON files.

**Default policy** is safe-by-default:

- `--require-status complete` → accepts `Aggregated` or `CompletePartials`. Refuses `InProgress`, `InvalidState`, `NoAssignments`, `NoSession`.
- `--copy` → source state-dir is untouched. `--move` is explicit opt-in.
- `--include-results` → off. Even when on, only `results/result-links/<posted_id>.link.json` is archived; per-job artifacts under `results/contributor-results/` are left in place.
- `--dry-run` → builds the manifest in memory and prints `event=would_archive_*` lines; does not touch the FS.

**When should I archive?** When the audit roll-up tells you:

```text
event=audit_health session_id=... coherence=coherent triagable_by_reassign=false \
  recommended_action="run archive-session --require-status aggregated"
```

The Stage 12.14 audit ergonomics emit that line for `Coherent + Aggregated` and `Coherent + ExpiredIncomplete` sessions. `CompletePartials` is intentionally NOT recommended for archive (operator may still want the aggregate to land).

**Typical workflow:**

```bash
# 1. Inspect — dry-run first.
omni-node operator contributor archive-session \
  --contributor-state-dir <state> \
  --session-id <hex> \
  --archive-dir /backup/contributor-archive \
  --dry-run

# (review the event=would_archive_* lines + manifest size estimate)

# 2. Copy. Source untouched; safe to repeat.
omni-node operator contributor archive-session \
  --contributor-state-dir <state> \
  --session-id <hex> \
  --archive-dir /backup/contributor-archive

# 3. Inspect the archive on disk.
cat /backup/contributor-archive/<session_id>/manifest.json
ls /backup/contributor-archive/<session_id>/verified/sessions/<session_id>/

# 4. (Optional) sync to remote storage.
rsync -a /backup/contributor-archive/<session_id>/ \
       s3://operator-bucket/contributor-archive/<session_id>/

# 5. After verifying the archive is durable, free state-dir.
omni-node operator contributor archive-session \
  --contributor-state-dir <state> \
  --session-id <hex> \
  --archive-dir /backup/contributor-archive2 \
  --move
```

Step 5 uses a fresh `--archive-dir` because step 2 already populated the first one (the call refuses to overwrite). Operators who don't need the safety belt can skip step 2 and go straight to `--move`.

**Recovery from a partial copy:** a partial copy under `--copy` leaves no manifest in `<archive-dir>/<session_id>/`. Just `rm -rf <archive-dir>/<session_id>/` and re-run. The source state-dir is untouched.

**Recovery from a BLAKE3 mismatch:** the command exits nonzero with `ArchiveError::BlakeMismatch` naming the failing file. Triage the filesystem / disk health, `rm -rf <archive-dir>/<session_id>/`, and re-run.

**Recovery from a partial `--move` cascade:** the `--move` path runs a **strict** cascade after the manifest write — every IO error other than `NotFound` propagates as `ArchiveError::State(StateError::Io { path, source })` and the process exits nonzero. The archive in `<archive-dir>/<session_id>/` IS durable (manifest written, files BLAKE3-verified); the source state-dir may be partially-cascaded (some files removed, some remnants left). Triage the FS error at the named `path`, then re-run `omni-node operator contributor archive-session --move` against the SAME `<archive-dir>` to refuse cleanly (`ArchiveAlreadyExists`) and instead manually call the cascade — or, simpler, `rm -rf` the source remnants by hand. The cascade is idempotent (NotFound is benign), so a retry after FS triage converges to the same final state.

**Manual restore (no Stage 12.14 restore command yet):**

```bash
cp -r /backup/contributor-archive/<session_id>/verified/sessions/<session_id>/ \
      <state-dir>/verified/sessions/<session_id>/
# (also restore seen markers if you need cross-restart dedup)
cp -r /backup/contributor-archive/<session_id>/seen/* \
      <state-dir>/seen/
```

Then run `omni-node operator contributor session-status --contributor-state-dir <state> --session-id <hex>` to confirm the chain re-verifies.

### Restoring an archived session (Stage 12.15)

`restore-session-archive` is the inverse of `archive-session`. It reads `<archive-session-dir>/manifest.json` (or `<archive-dir>/<session_id>/manifest.json` if you supply the pair), validates every file's BLAKE3 + safe-path + state-dir version compatibility, then writes the bytes back into your state-dir. **No chain, mesh, or SNIP wire is touched** — restore is local byte-identical replay.

**Stop the watcher first.** Stage 12.7 explicitly treats `<state-dir>` as operator-local; concurrent writes from a running `watch-sessions` and the restore command are not coordinated. The simplest pattern: SIGTERM the watcher, restore, restart the watcher.

**Three modes** — fastest to slowest:

| Mode | What it does | When to use |
|---|---|---|
| `--dry-run` | Parses the manifest, runs path safety + whitelist + schema/version/session-id checks. Touches no archive bytes beyond the manifest; touches no state-dir files. | Quick "is this archive shaped right?" check. |
| `--verify-only` | Adds full BLAKE3 walk over every archived file. Touches no state-dir files. | "Is this archive byte-intact on disk before I commit to a real restore?" |
| (real restore) | Adds destination-existence preflight (all-or-nothing) + writes every byte verbatim through the state-store's atomic helper. | The real thing. |

**Typical workflow:**

```bash
# 1. Stop the watcher (if any).
systemctl stop omni-node-watcher.service     # or your equivalent

# 2. Dry-run: cheap manifest validation.
omni-node operator contributor restore-session-archive \
  --contributor-state-dir <state> \
  --archive-session-dir /backup/contributor-archive/<session_id> \
  --dry-run

# 3. Verify-only: full BLAKE3 walk over the archive. Slow if the
#    archive is large but catches bit-rot.
omni-node operator contributor restore-session-archive \
  --contributor-state-dir <state> \
  --archive-session-dir /backup/contributor-archive/<session_id> \
  --verify-only

# 4. Real restore.
omni-node operator contributor restore-session-archive \
  --contributor-state-dir <state> \
  --archive-session-dir /backup/contributor-archive/<session_id>

# 5. Confirm with session-status. (NOT auto-run.)
omni-node operator contributor session-status \
  --contributor-state-dir <state> \
  --session-id <session_id>

# 6. Restart the watcher.
systemctl start omni-node-watcher.service
```

**Defaults are safe-by-default.** `--overwrite-existing` is OFF — restore refuses BEFORE writing if any destination file already exists. `--include-results` is OFF; result-link entries in the archive are skipped unless you opt in.

**Recovery from a refusal:**

- `RestoreError::BlakeMismatch` → archive bit-rot or hand-tamper. Re-pull the archive from your durable backup before re-running.
- `RestoreError::ManifestFileMissing` → the archive is incomplete (a partial copy from another tool). Re-pull or re-archive from source.
- `RestoreError::DestinationExists` → state-dir already holds files at the restore paths. Either re-run with `--overwrite-existing` (knowing what you're overwriting) or first delete the conflicting subtree (e.g. `omni-node operator contributor session-status` to identify it, then manual cleanup).
- `RestoreError::UnsafeRelativePath` / `RestoreError::DisallowedRelativePath` → the archive's manifest was hand-edited or produced by a non-Stage-12.14 tool. Refuse to ingest; treat the archive as untrusted.
- `RestoreError::IncompatibleSourceStateVersion` → the archive was produced by a state-dir at a different `STATE_VERSION`. v1 enforces strict equality (`== 2`). Either upgrade/downgrade your binary OR re-archive from a current state-dir.
- `RestoreError::SessionIdMismatch` → the archive directory was renamed and no longer matches the manifest's `session_id`. Rename it back, or use `--archive-dir + --session-id` to be explicit.

The restore command exits **nonzero** on every `RestoreError` variant so automation can detect refusals and trigger triage.

### Stage 12.16 — local state-dir integrity scan

**Use when:** routine health check, suspected disk corruption, after a half-finished repair, or as a CI gate before promoting a state-dir snapshot. The scan is fully read-only — safe to run against a live state-dir while the watcher is up.

**Workflow:**

```sh
# 1. Quick health check — events stream + exit-code policy.
omni-node operator contributor state-integrity \
  --contributor-state-dir <state>
# Exit code: 0 if counts_error == 0, else 1.

# 2. Drill into one session.
omni-node operator contributor state-integrity \
  --contributor-state-dir <state> \
  --session-id <session_id> \
  --format pretty

# 3. Pre-commit / pre-restore: also walk a parallel archive root.
#    Surfaces ArchiveCoveredSession (Ok), ArchiveManifestMalformed
#    (Error), ArchiveBlakeMismatch (Error).
omni-node operator contributor state-integrity \
  --contributor-state-dir <state> \
  --include-archives /backup/contributor-archive \
  --format json --json-out /tmp/integrity.json

# 4. CI strict mode: every warning trips the build.
omni-node operator contributor state-integrity \
  --contributor-state-dir <state> \
  --fail-on-warn
```

> **The scan never prunes.** Unlike `session-status` and the other
> Stage 12.x subcommands, `state-integrity` opens the state-store
> with auto-prune unconditionally OFF. Expired and incomplete
> session subtrees survive the open call so the scan can *report*
> on them rather than silently deleting them. There is therefore
> no `--no-prune-state-on-start` flag on this subcommand.

**Reading findings:**

Each finding carries closed-set `kind`, `severity`, optional `session_id`, optional `path`, `reason_tag`, and `recommended_action`. The action is a string label, not a command — operators run it explicitly using the existing Stage 12.10 / 12.11 / 12.14 / 12.15 surfaces.

| `recommended_action` | What to run |
| --- | --- |
| `run session-status` | `omni-node operator contributor session-status --session-id <id>` to triage further. |
| `run plan-session-reassign --reason invalid-partial` | Stage 12.11 reassign flow with reason `InvalidPartial`. |
| `run plan-session-reassign --reason missing-partial` | Stage 12.11 reassign flow with reason `MissingPartial`. |
| `run archive-session --verify-only` | Stage 12.14 archive in dry-run / verify mode. |
| `run restore-session-archive --verify-only` | Stage 12.15 restore in `--verify-only` mode. |
| `delete stale seen marker` | Manual: remove the `seen/<ns>/<key>` file the finding's `path` points to. Safe — seen markers are an idempotency hint, not a protocol artifact. |
| `clean state-dir orphan replacements before retry` | The Stage 12.13 audit detected orphan replacement assignments; inspect via `session-status` and either repair the supersession or remove the orphans before re-running reassign. |
| `operator triage required` | The finding does not map cleanly to a one-shot command. Read the `reason_tag` (e.g. `BindingMismatch`, `AggregateExtraPartialFor`) and consult [the Stage 12 protocol doc](./stage12-contributor-protocol.md). |

**Defaults are safe-by-default.** The scan writes nothing to the state-dir or archive-dir, and auto-prune is forced off (see the note above), so a scan against an old or expired tree never changes disk state. `--include-archives` is OFF; without it, the scan does not touch the archive directory at all.

**Two documented gaps in v1:**

- **No mutation.** v1 detects and recommends; it does not execute repairs. Run the suggested CLI invocations explicitly.
- **Unparseable bodies.** A verified body whose JSON is structurally broken (truncated, garbage prefix) is silently skipped by the Stage 12.7 parse-only loader, matching Stage 12.13's restart-preload behavior. If `sessions_scanned` is lower than expected, compare against `ls verified/sessions/` to spot bodies that fell off the snapshot.

The scan command exits **nonzero** when the configured threshold is tripped so CI gates and cron alerting can detect drift automatically:

- default: exit 1 when `counts_error > 0`
- `--fail-on-warn`: exit 1 when `counts_warn + counts_error > 0`

### Stage 12.17 — local state-dir cleanup planner / applier

**Use when:** `state-integrity` surfaces findings you want to act on locally without protocol-level intervention — stale or missing seen markers, stray files, tampered/invalid bodies that should be quarantined for forensics. The Stage 12.16 scanner stays read-only; the two Stage 12.17 commands are the only mutating surface.

**Workflow (the loop):**

```sh
# 1. Plan. Read-only. Writes only the plan JSON.
omni-node operator contributor plan-state-cleanup \
  --contributor-state-dir <state> \
  --out /tmp/plan.json
# Review /tmp/plan.json — every action carries source_finding_kind
# + source_reason_tag so you can trace it back to a Stage 12.16
# finding.

# 2. Dry-run the apply. No FS writes; the drift/hash/gate
#    preflights still run, so a dry-run that succeeds means a
#    real apply will succeed.
omni-node operator contributor apply-state-cleanup \
  --contributor-state-dir <state> \
  --plan /tmp/plan.json \
  --quarantine-dir /var/lib/omni-node/quarantine \
  --dry-run

# 3. Apply. Tier-B bodies land BLAKE3-verified under
#    <quarantine-dir>/<plan_id>/ BEFORE removal; a
#    quarantine-manifest.json is written LAST.
omni-node operator contributor apply-state-cleanup \
  --contributor-state-dir <state> \
  --plan /tmp/plan.json \
  --quarantine-dir /var/lib/omni-node/quarantine

# 4. Re-scan to confirm.
omni-node operator contributor state-integrity \
  --contributor-state-dir <state>
```

**When the plan contains gated actions:**

- `QuarantineAndUnmarkPartial` → pass `--allow-invalid-partial-cleanup`. The gate exists to stop you from accidentally pulling the rug from a planned Stage 12.11 reassign.
- `QuarantineAndUnmarkOrphanAssignment` → pass `--allow-orphan-assignments`. Apply additionally re-runs `compute_audit_health` per gated session and refuses if the orphan id set has changed since the plan was built.

**Refusal modes (typed `CleanupError`):**

- `SourceIntegrityDrift { expected, got }` → state-dir changed between plan and apply. Re-plan.
- `PlanHashMismatch { stored, recomputed }` → plan file was hand-edited or corrupted. Rebuild.
- `GateRequired { kind, flag }` → operator missed a gate flag. Re-run with the flag, or strip the gated action from the plan.
- `OrphanAuditDrift { session_id, plan_count, current_count }` → the gated session's orphan set moved. Re-plan.
- `QuarantineCollision { path }` → `<quarantine-dir>/<plan_id>/` already exists. Choose a fresh `--quarantine-dir` or clear the subtree.
- `UnsafePlanPath { path, reason }` → the plan file contains a `path` or `seen_marker_path` that violates the per-kind whitelist (`..` traversal, absolute, backslash, wrong prefix). The plan was hand-edited or generated by an unvetted tool — refuse to apply. Even a self-consistent `cleanup_plan_hash` does not vouch for path safety; the applier always re-validates every path against the whitelist before any IO.

**Rollback story (v1):**

The applier walks three explicit phases — **Phase A** (quarantine the bytes), **Phase B** (write `quarantine-manifest.json` atomically), **Phase C** (remove sources + unmark seen markers). If Phase A or B fails for any reason (BLAKE3 mismatch, FS error, manifest rename failure), Phase C is skipped and the state-dir is byte-identical to pre-apply. The quarantine subtree under `<plan_id>/` may contain partial body copies; the operator deletes it and re-runs the apply.

A successful apply leaves a byte-identical copy of every tier-B source at `<quarantine-dir>/<plan_id>/<source_relative>` together with a `quarantine-manifest.json` listing every entry's BLAKE3 + bytes. For first-class rollback, see the **Stage 12.18 cleanup-quarantine restore** section below. Stage 12.15 `restore-session-archive` does **not** consume a quarantine manifest — its manifest schema is `ArchiveManifest`, not `QuarantineManifest`.

**Defaults are safe-by-default.** Both commands open the state-store with `auto_prune = false` unconditionally — there is no `--no-prune-state-on-start` flag on either command (the Stage 12.16 review-fix precedent). The `--quarantine-dir` flag is *required* even for tier-A-only plans so a plan-id directory is always pinned consistently (no manifest is written when no bytes are quarantined).

**No protocol surface touched.** No envelope, no canonical-byte changes, no `STATE_VERSION` bump, no SNIP / mesh / chain / payment / proof / marketplace surface.

### Stage 12.18 — local cleanup-quarantine restore

**Use when:** Stage 12.17 cleanup ran (intentionally or by mistake) and you want to undo it bit-for-bit. The Stage 12.18 command consumes the `quarantine-manifest.json` Stage 12.17 wrote and reverses every body removal + (by default) every seen-marker unmark. Tier-A actions (`RemoveSeenMarker` / `WriteSeenMarker` / `RemoveSeenFile`) and `--purge-stray` entries are NOT recoverable from the manifest — operator re-runs the watcher or `mark_seen` manually for those.

**Workflow (the rollback loop):**

```sh
# 1. Inspect: parse + path-check the manifest only.
omni-node operator contributor restore-state-cleanup-quarantine \
  --contributor-state-dir <state> \
  --quarantine-plan-dir /var/lib/omni-node/quarantine/<plan_id> \
  --dry-run

# 2. Verify-only: BLAKE3 every quarantined byte against the
#    manifest. Catches bit-rot before any state-dir write.
omni-node operator contributor restore-state-cleanup-quarantine \
  --contributor-state-dir <state> \
  --quarantine-plan-dir /var/lib/omni-node/quarantine/<plan_id> \
  --verify-only

# 3. Real restore. Tier-B bodies go back to their state-dir
#    paths AND matching seen markers are written by default.
omni-node operator contributor restore-state-cleanup-quarantine \
  --contributor-state-dir <state> \
  --quarantine-plan-dir /var/lib/omni-node/quarantine/<plan_id>

# 4. Confirm with state-integrity. Pre-cleanup findings are
#    expected to re-appear (cleanup undid them; restore put
#    them back).
omni-node operator contributor state-integrity \
  --contributor-state-dir <state>
```

**Source-form options:**

- `--quarantine-plan-dir /path/to/<plan_id>` — direct path to the `<plan_id>` subdirectory.
- `--quarantine-dir /var/lib/omni-node/quarantine --plan-id <hex16>` — paired form, useful when scripting against a known quarantine root. Clap enforces that `--quarantine-dir` requires `--plan-id`.

**Optional flags:**

- `--overwrite-existing` — by default, any pre-existing destination refuses the whole restore (`DestinationExists`). With this flag, every destination is overwritten verbatim.
- `--no-restore-seen-markers` — restore only the verified bodies; skip writing the seen markers. Useful when you want the watcher to re-process announcements after the restore.
- `--allow-restore-orphan-assignments` — required to restore any entry whose `source_finding_kind == "orphan_replacement_assignments"`. Restoring re-introduces the Phase-B leftover that integrity-scan will detect; opt-in mirrors Stage 12.17's `--allow-orphan-assignments`.

**Refusal modes (typed `QuarantineRestoreError`):**

- `QuarantineDirNotFound { path }` → bad `--quarantine-plan-dir` / wrong `--plan-id`.
- `ManifestMissing { path }` → quarantine dir exists but `quarantine-manifest.json` is gone. Usually means Stage 12.17 apply crashed between Phase A and Phase B; nothing to restore.
- `MalformedManifest { path }` → manifest JSON is broken. Hand-edited or corrupted.
- `UnsupportedManifestVersion { got, expected }` → manifest came from a future stage; v1 binary refuses.
- `IncompatibleSourceStateVersion { manifest, current }` → quarantine was produced against a different `STATE_VERSION`. Strict equality; no migration story.
- `PlanIdMismatch { manifest_plan_id, supplied_plan_id }` → quarantine dir was renamed. Re-supply the correct name OR use the `--quarantine-dir + --plan-id` paired form.
- `UnsafeRelativePath { path, reason }` → manifest contains a `..` / absolute / backslash / out-of-whitelist path. Refused BEFORE any IO.
- `ManifestFileMissing { path }` → manifest names a body that's not on disk. Likely a partial copy.
- `BlakeMismatch { path, expected, got }` → bit-rot in the quarantine subtree. Re-pull from a durable backup or accept the loss.
- `DestinationExists { path }` → state-dir already has a file at the restore path. Re-run with `--overwrite-existing`.
- `SeenMarkerPathBlocked { path, reason }` → the seen-marker destination is occupied by a directory (or other non-file). Refused BEFORE any body write so the state-dir is byte-identical to pre-restore. Manually clear the marker path, or pass `--no-restore-seen-markers` to skip marker restoration entirely. `--overwrite-existing` does NOT cover this.
- `GatedRestoreRequired { kind, flag }` → operator missed `--allow-restore-orphan-assignments` for an orphan entry.
- `UnknownFindingKind { kind }` → manifest carries a `source_finding_kind` outside the Stage 12.17 closed set. Refused for forward-incompatibility safety.

**Rollback hash invariant:**

When `--no-restore-seen-markers` is NOT passed AND the original cleanup plan contained ONLY tier-B actions (no tier-A), the post-restore `source_integrity_hash` exactly equals the pre-cleanup hash. The restore is a bit-exact undo. Plans containing tier-A actions are partially un-restorable by design — those operations carried no payload to quarantine.

**Defaults are safe-by-default.** Auto-prune is forced OFF unconditionally; the `--no-prune-state-on-start` flag is deliberately absent (Stage 12.16/12.17 precedent). The quarantine subtree is **left intact** after the restore — you manage retention manually.

**No protocol surface touched.** No envelope, no canonical-byte changes, no `STATE_VERSION` / `QUARANTINE_MANIFEST_SCHEMA_VERSION` bump, no SNIP / mesh / chain / payment / proof / marketplace surface.

### Stage 12.19 — local integrity-report diff

**Use when:** you want to gate CI against "did anything new appear in the state-integrity scan since the last known-good baseline?" or forensically compare two captured snapshots from different times. The diff classifies every finding as `new`, `resolved`, or `unchanged` — read-only, no state-dir writes.

**CI gate workflow (live-vs-baseline):**

```sh
# 1. One-time: capture the baseline.
omni-node operator contributor state-integrity \
  --contributor-state-dir <state> \
  --format json \
  --json-out /etc/omni-node/baseline.integrity.json
# Commit /etc/omni-node/baseline.integrity.json to the repo
# (or store it under a build-agent artifact path).

# 2. Each CI run: scan live + diff against baseline.
omni-node operator contributor state-integrity \
  --contributor-state-dir <state> \
  --baseline /etc/omni-node/baseline.integrity.json \
  --fail-on-new-error
# Exit 1 if any NEW error-severity finding appeared since
# the baseline. Warnings stay green; existing baseline
# findings stay green.

# 3. (Forensic, two-JSON form.) Compare two captured
#    snapshots from different times or hosts.
omni-node operator contributor state-integrity-diff \
  --baseline /backup/yesterday.integrity.json \
  --current /backup/today.integrity.json \
  --format pretty
```

**Flag options:**

- `--require-state-dir-match` — defaults OFF. CI baselines are typically captured on a build agent with a different state-dir path than prod. Turn ON for host-pinned baselines.
- `--fail-on-new` — exit 1 if ANY new finding appears (any severity).
- `--fail-on-new-error` — exit 1 only when a new `error`-severity finding appears. Recommended CI mode; warnings stay green.
- Both flags can be set; either tripping → exit 1.
- `--summary-only` — omit the (often-large) `unchanged_findings` from rendered output. Applies uniformly to **events, pretty, JSON stdout, and `--json-out` mirror** alike (Stage 12.19 review fix routes them all through one redaction helper). `counts.unchanged` is preserved so scripts can still see the elided count. The library's `StateIntegrityDiffReport` returned by `diff_state_integrity_reports` always carries every finding; this is purely a CLI presentation flag.

**Refusal modes (typed `IntegrityDiffError`):**

- `UnsupportedBaselineSchemaVersion { got, expected }` → baseline came from a future stage; v1 binary refuses.
- `UnsupportedCurrentSchemaVersion { got, expected }` → same for the current report.
- `IncompatibleStateVersion { baseline, current }` → reports were captured against different `STATE_VERSION` values. Strict equality; no migration story.
- `StateDirMismatch { baseline, current }` → `state_dir` strings differ AND `--require-state-dir-match` was set. Drop the flag for host-mismatched comparisons.
- `FindingMetadataDrift { identity, baseline_severity, current_severity, baseline_recommended_action, current_recommended_action }` → two findings share the bit-exact identity tuple `(kind, session_id, path, reason_tag)` but disagree on severity or recommended action. v1 has no scanner path that produces such drift; the refusal indicates one of the reports was hand-edited or produced by a non-Stage-12.16 tool.
- `MalformedBaseline { path, source }` / `MalformedCurrent { path, source }` → JSON parse failed.

**Rollback hash invariant** (from Stage 12.18): a successful cleanup → quarantine → restore round-trip leaves the post-restore `source_integrity_hash` equal to the pre-cleanup hash. Use Stage 12.19's diff as the CI gate that *verifies* this invariant on every PR.

**Defaults are safe-by-default.** `state-integrity-diff` opens no state-store at all — it's pure JSON-to-JSON. `state-integrity --baseline` preserves the Stage 12.16 auto-prune-off posture (no `--no-prune-state-on-start` flag exposed; pinned by clap regression). The diff output is a v1 `StateIntegrityDiffReport` with its own forward-compatible schema (`STATE_INTEGRITY_DIFF_SCHEMA_VERSION = 1`); the v1 `StateIntegrityReport` it consumes is unchanged.

**No protocol surface touched.** No envelope, no canonical-byte changes, no `STATE_VERSION` / `STATE_INTEGRITY_REPORT_SCHEMA_VERSION` bump, no SNIP / mesh / chain / payment / proof / marketplace surface.

### Stage 12.20 — signed integrity baselines

**Use when:** your CI gate needs to prove the baseline JSON came from an expected Ed25519 key (and hasn't been hand-edited) before Stage 12.19 diffs it. The wrapper is local-only — no protocol, no SNIP, no chain. Sign the baseline once with an operator-local seed, then point the diff CLI at the signed wrapper plus the operator-supplied trust anchor (the pubkey hex).

**Workflow — signing the baseline:**

```sh
# 1. Capture a raw v1 baseline as before (Stage 12.16).
omni-node operator contributor state-integrity \
  --contributor-state-dir <state> \
  --format json \
  --json-out /tmp/baseline.json

# 2. Sign with a 32-byte Ed25519 seed file. Operators MUST
#    use a seed distinct from any chain-attestation or
#    protocol-role seed — the baseline-signing role is its
#    own key per Stage 12.20.
omni-node operator contributor sign-state-integrity-baseline \
  --baseline-in /tmp/baseline.json \
  --signer-seed /etc/omni-node/baseline-signer.seed \
  --signer-role operator \
  --out /etc/omni-node/baseline.signed.json
# event=signed_baseline_written path=... signer_role=operator signer_pubkey=<hex>
```

**Workflow — consuming a signed baseline (CI gate):**

```sh
# Capture the trust anchor (the signer pubkey) once. This is
# the same hex string the signing CLI emitted on its
# event=signed_baseline_written line.
SIGNER_PUBKEY=<64-hex>

# CI: live scan diff against the signed baseline. The verifier
# refuses BEFORE any diff if the wrapper isn't signed by the
# expected pubkey OR if the signature doesn't match.
omni-node operator contributor state-integrity \
  --contributor-state-dir <state> \
  --signed-baseline /etc/omni-node/baseline.signed.json \
  --baseline-pubkey-hex "$SIGNER_PUBKEY" \
  --fail-on-new-error

# Forensic two-JSON form:
omni-node operator contributor state-integrity-diff \
  --signed-baseline /backup/yesterday.baseline.signed.json \
  --baseline-pubkey-hex "$SIGNER_PUBKEY" \
  --current /backup/today.baseline.json
```

**Flag rules (clap-enforced):**

- `--baseline` and `--signed-baseline` are mutually exclusive on both subcommands. `state-integrity-diff` requires exactly one of the two; `state-integrity` requires neither (omit both for the raw scan posture).
- `--baseline-pubkey-hex` is required whenever `--signed-baseline` is set. Operators never trust the pubkey embedded in the wrapper alone.
- All four `--signer-role` values are accepted: `operator` / `contributor` / `dispatcher` / `coordinator`. The role tag is recorded for forensics; verifiers don't enforce policy on the role itself — the trust anchor is the pubkey.

**Refusal modes (typed `SignedBaselineError`):**

- `UnsupportedSchemaVersion { got, expected }` → wrapper came from a future stage; v1 binary refuses.
- `UnsupportedReportSchemaVersion { got, expected }` → embedded report came from a future Stage 12.16 lineage. Refuse to deserialize.
- `SignerPubkeyMismatch { expected, got }` → the wrapper's `signer_pubkey_hex` doesn't equal `--baseline-pubkey-hex`. Cheap pre-check — no crypto burn.
- `SignatureMismatch` → the wrapper was hand-edited (any field of the canonical body changed) OR the signature was tampered. Stage 12.20 covers `signer_role`, `signed_at_utc`, `signer_pubkey_hex`, AND every field of the embedded `report` — tampering any of them trips this.
- `MalformedJson { path, source }` → wrapper JSON parse failed.

**`--baseline-pubkey-hex` enforcement (uniform across both subcommands):**

- **Solo `--baseline-pubkey-hex`** (no `--baseline`, no `--signed-baseline`) is **rejected at parse time on both surfaces** — clap's `requires = "signed_baseline"` constraint fires cleanly when no baseline source is supplied at all.
- **`--baseline + --baseline-pubkey-hex`** (raw baseline with a trust anchor but no `--signed-baseline`) is **accepted at parse time on both surfaces** — the clap setups' `conflicts_with` / `required_unless_present` interactions short-circuit the `requires` check. The runtime backstop in `resolve_diff_baseline` emits `event=warn context=baseline_pubkey_hex_unused` on both subcommands so the operator notices the trust anchor was silently dropped before they ship the gate. The diff still runs because the raw baseline path doesn't need a trust anchor.

Operators who want a trust-anchored baseline MUST use `--signed-baseline` instead.

**Determinism:** signing the same report + same seed + same `signed_at_utc` produces a byte-identical wrapper — operators can commit a signed baseline to a repository and re-derive it deterministically.

**No protocol surface touched.** No envelope, no canonical-byte changes on Stage 12.0–12.19 surfaces, no `STATE_VERSION` / `STATE_INTEGRITY_REPORT_SCHEMA_VERSION` / `STATE_INTEGRITY_DIFF_SCHEMA_VERSION` bump, no SNIP / mesh / chain / payment / proof / marketplace surface.

### Stage 12.21 — signed integrity diffs

**Use when:** you want tamper-evident archival evidence of an integrity diff: which baseline was compared, which current report was compared, the exact diff that came out, and which key signed the artifact. Stage 12.20 covers the *baseline* trust path so a CI gate can prove the baseline JSON came from an expected key. Stage 12.21 covers the *diff* trust path so a forensic record of the diff itself can be archived. Both are local-only — no protocol, no SNIP, no chain.

**Workflow — signing a diff:**

```sh
# 1. Produce a raw v1 diff (Stage 12.19) the normal way.
omni-node operator contributor state-integrity-diff \
  --baseline /backup/yesterday.baseline.json \
  --current /backup/today.baseline.json \
  --json-out /tmp/diff.json

# 2. Sign with a 32-byte Ed25519 seed file. As with Stage 12.20,
#    operators MUST use a seed distinct from any chain-attestation
#    or protocol-role seed — the integrity-artifact signing role
#    is its own key.
omni-node operator contributor sign-state-integrity-diff \
  --diff-in /tmp/diff.json \
  --signer-seed /etc/omni-node/diff-signer.seed \
  --signer-role operator \
  --out /var/audit/2026-06-12.diff.signed.json
# event=signed_integrity_diff_signing_started
# event=signed_integrity_diff_written path=... signer_role=operator signer_pubkey=<hex>
```

**Workflow — verifying an archived signed diff:**

```sh
# Capture the trust anchor (the signer pubkey) once. This is
# the same hex string the signing CLI emitted on its
# event=signed_integrity_diff_written line.
SIGNER_PUBKEY=<64-hex>

omni-node operator contributor verify-state-integrity-diff-signature \
  --signed-diff /var/audit/2026-06-12.diff.signed.json \
  --expected-signer-pubkey-hex "$SIGNER_PUBKEY"
# event=signed_integrity_diff_verify_started
# event=signed_integrity_diff_verify_ok path=... signer_role=operator signer_pubkey=<hex>
```

The verifier opens no state-store and writes nothing. It exits nonzero on any refusal and emits a closed-tag reason line so log scrapers can classify failures:

- `--format events` (default) → one success line OR one `event=signed_integrity_diff_verify_failed reason=<tag>` line + nonzero exit.
- `--format json` → compact metadata view of the verified wrapper (`schema_version`, `signed_at_utc`, `signer_role`, `signer_pubkey_hex`, `signature_hex`, diff `schema_version` / `generated_at_utc` / `state_dir` / `baseline_state_dir` / `counts`). Operational chatter goes to stderr so `jq` works.
- `--format pretty` → terminal-friendly summary.

**Flag rules (clap-enforced):**

- `sign-state-integrity-diff` requires `--diff-in`, `--signer-seed`, `--signer-role`, and `--out`. Missing any of them refuses at parse time.
- `verify-state-integrity-diff-signature` requires `--signed-diff` and `--expected-signer-pubkey-hex`. The wrapper's own `signer_pubkey_hex` field is forensic context only; operators always supply the trust anchor.
- All four `--signer-role` values are accepted on signing: `operator` / `contributor` / `dispatcher` / `coordinator` — the Stage 12.20 closed enum is REUSED because the variants are role names, not artifact-type names.
- Neither subcommand exposes `--no-prune-state-on-start` — pinned by clap regression. Neither opens a contributor state-store.

**Refusal modes (typed `SignedIntegrityDiffError`, surfaced as `reason=<tag>`):**

- `signer_pubkey_mismatch` → the wrapper's `signer_pubkey_hex` doesn't equal `--expected-signer-pubkey-hex`. Cheap pre-check — no crypto burn.
- `signature_mismatch` → the wrapper was hand-edited (any field of the canonical body changed) OR the signature was tampered. Stage 12.21 covers `signer_role`, `signed_at_utc`, `signer_pubkey_hex`, AND every field of the embedded `diff` — tampering any of them trips this.
- `unsupported_schema_version` → wrapper came from a future stage; v1 binary refuses.
- `unsupported_diff_schema_version` → embedded diff came from a future Stage 12.19 lineage; refuse to deserialize.
- `signing` → Ed25519 decode/verify primitive returned an error (malformed hex, bad length, etc.).
- `canonical` → bincode encoding of the canonical body failed (closed-set struct; should be impossible in practice).
- `io` → FS read of the wrapper failed (missing file, permissions, etc.).
- `malformed_json` → wrapper JSON didn't parse as a v1 `SignedStateIntegrityDiff` (likely a hand-edit that broke the structure).

**What's covered by the signature (vs what isn't):**

- **Covered**: wrapper `schema_version`, `signed_at_utc`, `signer_pubkey_hex`, `signer_role`, AND every byte of the embedded `StateIntegrityDiffReport` (counts, all three findings buckets, both inputs' provenance — `baseline_generated_at_utc`, `current_generated_at_utc`, `baseline_state_dir`, `state_dir`, `baseline_state_version`, `current_state_version`, `baseline_omni_contributor_version`, `current_omni_contributor_version`). The wrapper's `signature_hex` is excluded (signing over your own signature is circular).
- **Not covered** (deliberate v1 scope): no separate signed-baseline-hash field linking back to a specific Stage 12.20 wrapper. The diff body already lifts both inputs' provenance, so the signed diff transitively pins the baseline-vs-current context without a redundant pointer.

**Determinism:** signing the same diff + same seed + same `signed_at_utc` produces a byte-identical wrapper — operators can commit a signed diff to an archive and re-derive it deterministically.

**Composition with Stage 12.20:** the two trust paths are complementary. A typical forensic chain looks like (a) a Stage 12.20 signed baseline captured nightly, (b) the next morning's diff produced via `state-integrity-diff --signed-baseline <a> --baseline-pubkey-hex <anchor> --current <today.json> --json-out /tmp/diff.json`, then (c) the Stage 12.21 signed wrapper around that diff archived to immutable storage. Stage 12.20 proves the baseline JSON came from an expected key; Stage 12.21 proves the diff JSON came from an expected key (often the same key, sometimes a different audit-only key — operators decide).

**No protocol surface touched.** No envelope, no canonical-byte changes on Stage 12.0–12.20 surfaces, no `STATE_VERSION` / `STATE_INTEGRITY_REPORT_SCHEMA_VERSION` / `STATE_INTEGRITY_DIFF_SCHEMA_VERSION` / `SIGNED_BASELINE_SCHEMA_VERSION` bump, no SNIP / mesh / chain / payment / proof / marketplace surface.

### Stage 12.22 — local integrity evidence bundles

**Use when:** you've captured a set of Stage 12.16–12.21 forensic artifacts (some signed, some not) and want a single tamper-evident pointer file to archive as the operator-side evidence record. The bundle fingerprints each file by `(artifact_kind, base-dir-relative path, byte_len, blake3_hex)`. It does not re-sign anything; Stage 12.20 baselines and Stage 12.21 diffs already carry their own Ed25519 signatures. It does not parse anything either; the verifier only re-hashes the referenced files and reports per-entry outcomes.

The bundle is local-only — no protocol, no SNIP, no chain.

**Workflow — building a bundle:**

```sh
# 1. Stage your audit artifacts under one base directory.
mkdir -p /tmp/audit-2026-06-12 && cd /tmp/audit-2026-06-12
omni-node operator contributor state-integrity \
  --contributor-state-dir <state> \
  --format json --json-out baseline.json
omni-node operator contributor sign-state-integrity-baseline \
  --baseline-in baseline.json \
  --signer-seed /etc/omni-node/baseline-signer.seed \
  --signer-role operator \
  --out baseline.signed.json
omni-node operator contributor state-integrity-diff \
  --baseline /backup/yesterday.baseline.json \
  --current baseline.json \
  --json-out diff.json
omni-node operator contributor sign-state-integrity-diff \
  --diff-in diff.json \
  --signer-seed /etc/omni-node/diff-signer.seed \
  --signer-role operator \
  --out diff.signed.json

# 2. Assemble the bundle. --base-dir is required; every
#    --include path is recorded relative to it.
omni-node operator contributor build-integrity-evidence-bundle \
  --base-dir /tmp/audit-2026-06-12 \
  --include state_integrity_report=baseline.json \
  --include signed_state_integrity_baseline=baseline.signed.json \
  --include state_integrity_diff_report=diff.json \
  --include signed_state_integrity_diff=diff.signed.json \
  --label "audit-2026-06-12" \
  --notes "captured by CI run #4231" \
  --out /var/audit/2026-06-12.bundle.json
# event=integrity_evidence_bundle_build_started base_dir=/tmp/audit-2026-06-12
# event=integrity_evidence_bundle_entry_hashed kind=signed_state_integrity_baseline path=baseline.signed.json bytes=2371 blake3=<hex>
# ... (one line per entry)
# event=integrity_evidence_bundle_written path=/var/audit/2026-06-12.bundle.json entry_count=4
```

**Workflow — verifying a bundle:**

```sh
# Same host, bundle's recorded base_dir still valid:
omni-node operator contributor verify-integrity-evidence-bundle \
  --bundle /var/audit/2026-06-12.bundle.json
# event=integrity_evidence_bundle_verify_started bundle=/var/audit/2026-06-12.bundle.json
# event=integrity_evidence_bundle_entry_ok kind=signed_state_integrity_baseline path=baseline.signed.json resolved_path=/tmp/audit-2026-06-12/baseline.signed.json
# ...
# event=integrity_evidence_bundle_verify_summary effective_base_dir=/tmp/audit-2026-06-12 ok=4 size_mismatch=0 hash_mismatch=0 not_found=0 read_error=0

# Different host or relocated tree (portability lever):
omni-node operator contributor verify-integrity-evidence-bundle \
  --bundle /var/audit/2026-06-12.bundle.json \
  --base-dir /srv/audit-mirror/2026-06-12
```

The verifier exits 0 iff every entry's outcome is `Ok`. Any non-`Ok` outcome (`SizeMismatch`, `HashMismatch`, `NotFound`, `ReadError`) → exit 1, but the verifier walks every entry first and emits a full summary. **No short-circuit.** Output formats:

- `--format events` (default) → one per-entry event line + a final summary line.
- `--format json` → full `BundleVerifyReport` to stdout (operational chatter to stderr so `jq` works).
- `--format pretty` → compact terminal-friendly summary + per-entry listing.

**Flag rules (clap-enforced):**

- `build-integrity-evidence-bundle` requires `--include` (≥1, repeatable), `--base-dir`, and `--out`. `--label` / `--notes` are optional; the closed format enum is required-with-default.
- `verify-integrity-evidence-bundle` requires `--bundle`. `--base-dir` is optional — when omitted, the verifier resolves entries against the bundle's recorded `base_dir`. When supplied, it overrides.
- All `--include kind=path` wire tags must come from the closed set (`state_integrity_report` / `signed_state_integrity_baseline` / `state_integrity_diff_report` / `signed_state_integrity_diff` / `state_cleanup_plan` / `cleanup_report` / `quarantine_manifest` / `quarantine_restore_report` / `archive_manifest` / `other`). Unknown tags refuse at CLI parse time with `event=integrity_evidence_bundle_build_failed reason=invalid_include`.
- Neither subcommand exposes `--no-prune-state-on-start` — pinned by clap regression. Neither opens a contributor state-store.

**Refusal modes (typed `EvidenceBundleError`, surfaced as `reason=<tag>`):**

- `empty_bundle` → no `--include` was supplied (must be at least 1).
- `duplicate_entry` → two `--include`s collided on `(artifact_kind, path)` after normalization.
- `too_many_entries` → more than 1024 entries. Cheap operator-typo defense; raise via constant if real-world bundles approach the cap.
- `bundle_label_too_long` / `notes_too_long` → `--label` exceeded 128 bytes or `--notes` exceeded 1024 bytes (UTF-8).
- `entry_too_large` → an entry exceeded the 256 MiB per-entry cap. Size check is cheap pre-check; no read attempted.
- `entry_not_found` → an `--include` pointed at a file that didn't exist at build time.
- `path_outside_base_dir` → an absolute `--include` path canonicalized to a target outside the canonical `--base-dir`. Refused so recorded paths stay base-dir-rooted.
- `invalid_relative_path` → a recorded entry path failed the strict relative-path validator. Detail carries the closed reason: `empty` / `absolute` (leading `/`) / `backslash` / `dot_segment` (a `.` segment like `./foo`) / `dotdot_segment` (a `..` traversal like `../outside`) / `empty_segment` (a `//` or trailing `/`). Refused at BUILD time on relative `--include` inputs BEFORE any hashing, and at VERIFY envelope-level on `bundle.entries[].path` BEFORE any per-entry FS work — so a hand-edited bundle pointing outside `base_dir` never opens a single byte.
- `base_dir_invalid` → `--base-dir` doesn't exist, isn't a directory, or couldn't be canonicalized.
- `effective_base_dir_not_found` → at verify time, neither the override nor the bundle's recorded `base_dir` exists. Refused at the envelope level so operators don't see N false `NotFound` outcomes from a bad root.
- `non_utf8_path` → a supplied path is not valid UTF-8.
- `invalid_include` → `--include` value was malformed (no `=`, unknown kind tag, empty path).
- `unsupported_schema_version` → bundle JSON came from a future stage; v1 binary refuses.
- `malformed_json` → bundle JSON couldn't be parsed as a v1 `IntegrityEvidenceBundle`.

**What the bundle covers (vs what it doesn't):**

- **Covered**: `(artifact_kind, base-dir-relative path, byte length, BLAKE3 of file bytes)` for each `--include`-d file, plus the bundle's own `schema_version`, `generated_at_utc`, `omni_contributor_version`, optional `label`, optional `notes`, and canonical `base_dir`.
- **Not covered** (deliberate v1 scope): no signature over the bundle JSON itself, no semantic validation of the referenced artifact bytes (e.g. the verifier does NOT re-run `verify_signed_state_integrity_baseline` on a signed-baseline entry — operators run Stage 12.20 / 12.21 verifiers separately), no recursive directory bundling, no automatic artifact discovery.

**Symlinks:** symlinks under `--base-dir` are followed at hash time. The recorded `path` is the operator-supplied form, NOT the symlink target. Operators wanting target-pinned paths should pass canonicalized paths themselves.

**Determinism:** same inputs + same `--base-dir` + same `generated_at_utc` + identical file contents produce a byte-identical bundle — operators can commit a bundle to an archive and re-derive it deterministically.

**Composition with Stage 12.20 / 12.21:** the bundle is the outer wrapper. A typical archival chain looks like (a) build daily integrity artifacts (Stage 12.16/12.19) + sign them (Stage 12.20/12.21) under a per-day `--base-dir`, (b) assemble a Stage 12.22 bundle pointing at all of them, (c) archive the bundle JSON to immutable storage. An operator reviewing the bundle + the artifact tree runs `verify-integrity-evidence-bundle` to confirm bytes haven't shifted, then runs Stage 12.20 / 12.21 verifiers on the signed entries to confirm the per-stage Ed25519 signatures still verify against the trust anchors they care about.

**No protocol surface touched.** No envelope, no canonical-byte changes on Stage 12.0–12.21 surfaces, no `STATE_VERSION` / `STATE_INTEGRITY_REPORT_SCHEMA_VERSION` / `STATE_INTEGRITY_DIFF_SCHEMA_VERSION` / `SIGNED_BASELINE_SCHEMA_VERSION` / `SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION` bump, no SNIP / mesh / chain / payment / proof / marketplace surface.

### Stage 12.23 — signed integrity evidence bundles

**Use when:** you've assembled a Stage 12.22 `IntegrityEvidenceBundle` and want to attest provenance — "this bundle came from a specific Ed25519 key at a specific time" — alongside the byte-fingerprint protection the bundle already provides. The signed wrapper is local-only — no protocol, no SNIP, no chain.

Bundle-byte verification (Stage 12.22 `verify-integrity-evidence-bundle`) and signature verification (Stage 12.23 `verify-integrity-evidence-bundle-signature`) are deliberately decoupled. Run signature verification when you care about WHO produced the bundle; run bundle-byte verification when you care about WHETHER the referenced artifact files still match.

**Workflow — signing a bundle:**

```sh
# 1. Assemble a v1 bundle as before (Stage 12.22).
omni-node operator contributor build-integrity-evidence-bundle \
  --base-dir /tmp/audit-2026-06-14 \
  --include state_integrity_report=baseline.json \
  --include signed_state_integrity_baseline=baseline.signed.json \
  --include state_integrity_diff_report=diff.json \
  --include signed_state_integrity_diff=diff.signed.json \
  --label "audit-2026-06-14" \
  --out /tmp/audit-2026-06-14.bundle.json

# 2. Sign with a 32-byte Ed25519 seed file. Operators MUST
#    use a seed distinct from any chain-attestation or
#    protocol-role seed — the integrity-artifact signing role
#    is its own key per Stage 12.20.
omni-node operator contributor sign-integrity-evidence-bundle \
  --bundle-in /tmp/audit-2026-06-14.bundle.json \
  --signer-seed /etc/omni-node/bundle-signer.seed \
  --signer-role operator \
  --out /var/audit/2026-06-14.bundle.signed.json
# event=signed_integrity_evidence_bundle_signing_started
# event=signed_integrity_evidence_bundle_written path=... signer_role=operator signer_pubkey=<hex>
```

On failure the signer emits `event=signed_integrity_evidence_bundle_sign_failed reason=<closed-tag>` + nonzero exit. Closed tags: `io` (bundle-in or seed-file FS read failure, atomic-write failure), `malformed_json` (bundle-in JSON parse failure), `signing` (seed-file decode/setup, Ed25519 primitive failure), `unsupported_bundle_schema_version` (the bundle JSON is from a future Stage 12.22 lineage), `canonical` (bincode encoding of the canonical body failed — closed-set struct; should be impossible in practice). Single-sourced via the same closed-tag mapper as the verifier.

**Workflow — verifying a signed bundle:**

```sh
# Capture the trust anchor once. Same hex string the signing
# CLI emitted on its event=signed_integrity_evidence_bundle_written
# line.
SIGNER_PUBKEY=<64-hex>

omni-node operator contributor verify-integrity-evidence-bundle-signature \
  --signed-bundle /var/audit/2026-06-14.bundle.signed.json \
  --expected-signer-pubkey-hex "$SIGNER_PUBKEY"
# event=signed_integrity_evidence_bundle_verify_started signed_bundle=/var/audit/2026-06-14.bundle.signed.json
# event=signed_integrity_evidence_bundle_verify_ok path=... signer_role=operator signer_pubkey=<hex>
```

The verifier opens no state-store and writes nothing. It exits nonzero on any refusal and emits a closed-tag reason line so log scrapers can classify failures. Output formats:

- `--format events` (default) → start line + one success/failure line.
- `--format json` → compact metadata view of the verified wrapper (`schema_version`, `signed_at_utc`, `signer_role`, `signer_pubkey_hex`, `signature_hex`, plus embedded `bundle.schema_version` / `bundle.generated_at_utc` / `bundle.omni_contributor_version` / `bundle.label` / `bundle.base_dir` / `bundle.entries.len()`). Operational chatter goes to stderr so `jq` works. Does NOT re-print the bundle's full entry list — operators who want the entries already have the wrapper on disk.
- `--format pretty` → terminal-friendly summary.

**Flag rules (clap-enforced):**

- `sign-integrity-evidence-bundle` requires `--bundle-in`, `--signer-seed`, `--signer-role`, and `--out`. Missing any refuses at parse time.
- `verify-integrity-evidence-bundle-signature` requires `--signed-bundle` and `--expected-signer-pubkey-hex`. The wrapper's own `signer_pubkey_hex` field is forensic context only; operators always supply the trust anchor.
- All four `--signer-role` values are accepted: `operator` / `contributor` / `dispatcher` / `coordinator` — the Stage 12.20 closed enum is REUSED because the variants are role names, not artifact-type names.
- Neither subcommand exposes `--no-prune-state-on-start` — pinned by clap regression. Neither opens a contributor state-store.
- Neither subcommand exposes `--base-dir` — that's a Stage 12.22 concept; this verifier only attests to bundle JSON bytes.

**Refusal modes (typed `SignedIntegrityEvidenceBundleError`, surfaced as `reason=<tag>`):**

- `signer_pubkey_mismatch` → the wrapper's `signer_pubkey_hex` doesn't equal `--expected-signer-pubkey-hex`. Cheap pre-check — no crypto burn.
- `signature_mismatch` → the wrapper was hand-edited (any field of the canonical body changed) OR the signature was tampered. Stage 12.23 covers `signer_role`, `signed_at_utc`, `signer_pubkey_hex`, AND every byte of the embedded `bundle` (every `BundleEntry`'s `artifact_kind` / `path` / `bytes` / `blake3_hex`, plus `bundle.schema_version` / `generated_at_utc` / `omni_contributor_version` / `label` / `notes` / `base_dir`) — tampering any of them trips this.
- `unsupported_schema_version` → wrapper came from a future stage; v1 binary refuses.
- `unsupported_bundle_schema_version` → embedded bundle came from a future Stage 12.22 lineage; refuse to deserialize.
- `signing` → Ed25519 decode/verify primitive returned an error (malformed hex, bad length, etc.).
- `canonical` → bincode encoding of the canonical body failed (closed-set struct; should be impossible in practice).
- `io` → FS read of the wrapper failed (missing file, permissions, etc.).
- `malformed_json` → wrapper JSON didn't parse as a v1 `SignedIntegrityEvidenceBundle`.

**What the signature covers (vs what it doesn't):**

- **Covered**: every field of the wrapper EXCEPT `signature_hex` — `schema_version`, `signed_at_utc`, `signer_pubkey_hex`, `signer_role`, AND every byte of the embedded `IntegrityEvidenceBundle` (including per-entry `blake3_hex` fingerprints). Because the bundle already records per-entry BLAKE3 hashes, the signature transitively pins the entire artifact tree without re-embedding any artifact bytes.
- **Not covered** (deliberate v1 scope): the artifact files the bundle references. Re-hashing them is Stage 12.22 `verify-integrity-evidence-bundle`'s job — operators run that separately if they care about byte integrity. Recursive verification of embedded `signed_state_integrity_baseline` / `signed_state_integrity_diff` artifacts is also out-of-scope — operators run Stage 12.20 / 12.21 verifiers separately.

**Determinism:** signing the same bundle + same seed + same `signed_at_utc` produces a byte-identical wrapper — operators can commit a signed bundle to an archive and re-derive it deterministically.

**Composition with Stage 12.20 / 12.21 / 12.22:** four trust paths, each with its own verifier, chained for full forensic confidence:

1. **Stage 12.20 / 12.21** — verify the per-artifact Ed25519 signatures on `signed_state_integrity_baseline` / `signed_state_integrity_diff` entries against their respective trust anchors.
2. **Stage 12.22** — re-hash every artifact file the bundle references; refuse on any byte drift.
3. **Stage 12.23** — verify the Ed25519 signature on the bundle wrapper itself against the operator's trust anchor. Confirms the bundle hasn't been hand-edited and came from the expected key.

A typical archival chain: (a) build daily integrity artifacts (Stage 12.16/12.19), (b) sign them (Stage 12.20/12.21), (c) assemble a Stage 12.22 bundle, (d) wrap it with a Stage 12.23 signature, (e) archive the signed bundle JSON to immutable storage. An operator reviewing the signed bundle runs Stage 12.23 verify first (cheapest — bundle JSON only); then Stage 12.22 verify (re-hashes all referenced files); then Stage 12.20/12.21 verify on each signed entry (re-checks each artifact's own signature).

**No protocol surface touched.** No envelope, no canonical-byte changes on Stage 12.0–12.22 surfaces, no `STATE_VERSION` / `STATE_INTEGRITY_REPORT_SCHEMA_VERSION` / `STATE_INTEGRITY_DIFF_SCHEMA_VERSION` / `SIGNED_BASELINE_SCHEMA_VERSION` / `SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION` / `INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION` bump, no SNIP / mesh / chain / payment / proof / marketplace surface.

### Stage 12.24 — local integrity-evidence chain verification

**Use when:** an operator has a signed bundle on disk and wants to run the full Stage 12.20–12.23 verification chain in ONE command. The chain verifier composes signed-bundle signature verify (Stage 12.23), bundle byte verify (Stage 12.22), and optional per-signed-child signature verify (Stage 12.20 / 12.21) into a single read-only operation.

**Workflow — full chain verify (every anchor supplied):**

```sh
# Capture the three trust anchors once.
BUNDLE_PUBKEY=<64-hex>      # Stage 12.23 bundle signer
BASELINE_PUBKEY=<64-hex>    # Stage 12.20 baseline signer
DIFF_PUBKEY=<64-hex>        # Stage 12.21 diff signer

omni-node operator contributor verify-integrity-evidence-chain \
  --signed-bundle /var/audit/2026-06-15.bundle.signed.json \
  --expected-bundle-signer-pubkey-hex "$BUNDLE_PUBKEY" \
  --expected-baseline-signer-pubkey-hex "$BASELINE_PUBKEY" \
  --expected-diff-signer-pubkey-hex "$DIFF_PUBKEY"
# event=integrity_evidence_chain_verify_started signed_bundle=... expected_baseline_pubkey=set expected_diff_pubkey=set
# event=integrity_evidence_chain_signed_bundle_ok signer_role=operator signer_pubkey=<64-hex>
# event=integrity_evidence_chain_bundle_byte_resolved effective_base_dir=...
# event=integrity_evidence_chain_bundle_byte_entry_ok kind=signed_state_integrity_baseline path=baseline.signed.json resolved_path=...
# event=integrity_evidence_chain_bundle_byte_entry_ok kind=signed_state_integrity_diff path=diff.signed.json resolved_path=...
# event=integrity_evidence_chain_child_ok kind=signed_state_integrity_baseline path=baseline.signed.json resolved_path=...
# event=integrity_evidence_chain_child_ok kind=signed_state_integrity_diff path=diff.signed.json resolved_path=...
# event=integrity_evidence_chain_verify_summary bundle_signature=ok bundle_byte_counts={ok=2 size_mismatch=0 hash_mismatch=0 not_found=0 read_error=0} child_counts={ok=2 skipped=0 failed=0}
```

**Workflow — verify only the bundle anchor (signed children skipped):**

```sh
# Omitted child anchors record Skipped — NOT silent passes.
# Operators see exactly which gates were skipped on the events
# output AND in the JSON report.
omni-node operator contributor verify-integrity-evidence-chain \
  --signed-bundle /var/audit/2026-06-15.bundle.signed.json \
  --expected-bundle-signer-pubkey-hex "$BUNDLE_PUBKEY"
# event=integrity_evidence_chain_child_skipped kind=signed_state_integrity_baseline ...
# event=integrity_evidence_chain_child_skipped kind=signed_state_integrity_diff ...
# event=integrity_evidence_chain_verify_summary ... child_counts={ok=0 skipped=2 failed=0}
# exit 0 — skipped children DON'T fail the exit code
```

**Workflow — verify bundle + diff but not baseline:**

```sh
omni-node operator contributor verify-integrity-evidence-chain \
  --signed-bundle /var/audit/2026-06-15.bundle.signed.json \
  --expected-bundle-signer-pubkey-hex "$BUNDLE_PUBKEY" \
  --expected-diff-signer-pubkey-hex "$DIFF_PUBKEY"
# baseline children → Skipped; diff children → Ok or Failed
```

**Workflow — relocated artifact tree (Stage 12.22 portability lever) + JSON archive:**

```sh
omni-node operator contributor verify-integrity-evidence-chain \
  --signed-bundle /var/audit/2026-06-15.bundle.signed.json \
  --expected-bundle-signer-pubkey-hex "$BUNDLE_PUBKEY" \
  --expected-baseline-signer-pubkey-hex "$BASELINE_PUBKEY" \
  --expected-diff-signer-pubkey-hex "$DIFF_PUBKEY" \
  --base-dir /srv/audit-mirror/2026-06-15 \
  --json-out /var/audit/2026-06-15.chain.json
# event=integrity_evidence_chain_json_written path=/var/audit/2026-06-15.chain.json
```

The `--json-out` write is **best-effort**: failure logs `event=integrity_evidence_chain_json_write_failed reason=<io|malformed_json>` and does NOT change exit code.

**Output formats:**

- `--format events` (default) → one per-step event line + a final summary line.
- `--format json` → full `IntegrityEvidenceChainReport` to stdout (operational chatter to stderr so `jq` works). Includes the embedded Stage 12.22 `BundleVerifyReport` and every per-child outcome.
- `--format pretty` → compact terminal-friendly summary + per-section listing.

**Exit policy:**

- Exit 0 iff: signed-bundle signature verified Ok AND every bundle-byte entry was Ok AND no signed-child signature failed.
- **`Skipped` child outcomes DON'T fail the exit.** They represent deliberate operator choice — explicit `--expected-*-signer-pubkey-hex` omission.
- Any envelope-level refusal (signed-bundle gate fails, Stage 12.22 base_dir missing, traversal path in embedded bundle) → exit nonzero, `event=integrity_evidence_chain_verify_failed reason=<closed-tag>`.

**Flag rules (clap-enforced):**

- `verify-integrity-evidence-chain` requires `--signed-bundle` and `--expected-bundle-signer-pubkey-hex`. The bundle signer trust anchor is non-negotiable.
- `--expected-baseline-signer-pubkey-hex` and `--expected-diff-signer-pubkey-hex` are independent optional gates. Pass either, both, or neither.
- `--base-dir` is optional — when omitted, the chain resolves entries against the bundle's recorded `base_dir`. When supplied, it overrides.
- `--json-out` is optional, best-effort.
- Subcommand does not expose `--no-prune-state-on-start` — pinned by clap regression. Does not open a contributor state-store.
- Pubkey hex format is NOT validated at clap parse time — the verifier surfaces typed errors (`signer_pubkey_mismatch`, `signing`) consistently with Stage 12.20–12.23.

**Refusal modes (envelope-level — emitted as `event=integrity_evidence_chain_verify_failed reason=<tag>` + nonzero exit):**

Closed reason tags carry a `signed_bundle_` or `bundle_byte_` prefix so the closed-set taxonomy stays self-disambiguating:

- `signed_bundle_io` / `signed_bundle_malformed_json` / `signed_bundle_unsupported_schema_version` / `signed_bundle_unsupported_bundle_schema_version` / `signed_bundle_signer_pubkey_mismatch` / `signed_bundle_signature_mismatch` / `signed_bundle_signing` / `signed_bundle_canonical` — Stage 12.23 envelope refusal on the outermost wrapper.
- `bundle_byte_unsupported_schema_version` / `bundle_byte_effective_base_dir_not_found` / `bundle_byte_invalid_relative_path` / etc. — Stage 12.22 envelope refusal on the embedded bundle. Per-entry outcomes are NOT here — they're inside the report's `bundle_byte_verify` field.

**Per-child refusal tags (inside `chain_child_failed` events, unprefixed because the surrounding `kind=...` field disambiguates):**

- For `signed_state_integrity_baseline` children: `io` (file missing / permissions) / `malformed_json` (parse fail) / `unsupported_schema_version` (wrapper schema) / `unsupported_report_schema_version` (embedded report schema) / `signer_pubkey_mismatch` (anchor mismatch) / `signature_mismatch` (crypto verify failed) / `signing` (decode error) / `canonical` (closed-set struct encoding).
- For `signed_state_integrity_diff` children: same except `unsupported_diff_schema_version` instead of `unsupported_report_schema_version`.

**Independent forensic facts (collect-all semantics):**

Bundle-byte verify and child-signature verify are independent. A signed-baseline entry can simultaneously:

- Have `bundle_byte_verify` outcome `HashMismatch` (the bytes on disk differ from what the bundle recorded), AND
- Have `child_signatures` outcome `Ok` (the wrapper as it stands on disk is still validly signed by the expected key)

These are different forensic facts — the file has drifted from what the bundle attested, but the file the bundle attested to was authentically signed at bundle-build time. Operators should see BOTH views. The chain verifier deliberately runs child signature verify even when bundle-byte said the bytes are wrong — a `HashMismatch` doesn't stop the child verifier from attempting to parse and verify whatever bytes ARE there.

**Composition with Stage 12.20 / 12.21 / 12.22 / 12.23:**

Stage 12.24 is the equivalent of running these four commands in sequence — but as a single composed operation that produces a unified report:

```sh
# Equivalent multi-command form (pre-Stage 12.24):
omni-node operator contributor verify-integrity-evidence-bundle-signature ...   # Stage 12.23
omni-node operator contributor verify-integrity-evidence-bundle ...             # Stage 12.22
omni-node operator contributor verify-signed-state-integrity-baseline ...       # Stage 12.20 (per signed-baseline entry)
omni-node operator contributor verify-state-integrity-diff-signature ...        # Stage 12.21 (per signed-diff entry)
```

Stage 12.24 collapses these into one invocation, with consistent exit-code semantics and a single tamper-evident JSON report that aggregates every gate's outcome. The individual stages remain available — operators who want fine-grained control (e.g. different log destinations per gate) can still run them separately.

**Determinism:** same inputs (signed bundle, anchors, `--base-dir`) + same `generated_at_utc` + same on-disk artifact files produce a byte-identical chain report — operators can commit a chain report to an archive and re-derive it deterministically.

**No protocol surface touched.** No envelope, no canonical-byte changes on Stage 12.0–12.23 surfaces, no `STATE_VERSION` / `STATE_INTEGRITY_REPORT_SCHEMA_VERSION` / `STATE_INTEGRITY_DIFF_SCHEMA_VERSION` / `SIGNED_BASELINE_SCHEMA_VERSION` / `SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION` / `INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION` / `SIGNED_INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION` bump, no SNIP / mesh / chain / payment / proof / marketplace surface.

### Stage 12.25 — signed integrity-evidence chain reports

**Use when:** you've run Stage 12.24 `verify-integrity-evidence-chain` and want to attest provenance — "this chain report came from a specific Ed25519 key at a specific time" — over the resulting JSON report. The signed wrapper is local-only — no protocol, no SNIP, no chain. Signature verification is **provenance-only**: it attests to bytes, not gate outcomes.

If you also want to re-verify the underlying Stage 12.24 gates (signed-bundle verify, bundle-byte verify, child signatures), run `verify-integrity-evidence-chain` directly against the chain report's `signed_bundle_path`. The Stage 12.25 wrapper deliberately does NOT do this — the two operations are decoupled so operators choose what to verify.

**Workflow — signing a chain report:**

```sh
# 1. Produce a v1 chain report (Stage 12.24).
omni-node operator contributor verify-integrity-evidence-chain \
  --signed-bundle /var/audit/2026-06-16.bundle.signed.json \
  --expected-bundle-signer-pubkey-hex "$BUNDLE_PUBKEY" \
  --expected-baseline-signer-pubkey-hex "$BASELINE_PUBKEY" \
  --expected-diff-signer-pubkey-hex "$DIFF_PUBKEY" \
  --json-out /tmp/audit-2026-06-16.chain.json

# 2. Sign with a 32-byte Ed25519 seed file. Operators MUST
#    use a seed distinct from any chain-attestation or
#    protocol-role seed — the integrity-artifact signing role
#    is its own key per Stage 12.20.
omni-node operator contributor sign-integrity-evidence-chain-report \
  --chain-report-in /tmp/audit-2026-06-16.chain.json \
  --signer-seed /etc/omni-node/chain-report-signer.seed \
  --signer-role operator \
  --out /var/audit/2026-06-16.chain.signed.json
# event=signed_integrity_evidence_chain_report_signing_started
# event=signed_integrity_evidence_chain_report_written path=... signer_role=operator signer_pubkey=<hex>
```

**Workflow — verifying a signed chain report:**

```sh
# Capture the trust anchor once. Same hex string the signing
# CLI emitted on its event=signed_integrity_evidence_chain_report_written
# line.
SIGNER_PUBKEY=<64-hex>

omni-node operator contributor verify-integrity-evidence-chain-report-signature \
  --signed-chain-report /var/audit/2026-06-16.chain.signed.json \
  --expected-signer-pubkey-hex "$SIGNER_PUBKEY"
# event=signed_integrity_evidence_chain_report_verify_started signed_chain_report=/var/audit/2026-06-16.chain.signed.json
# event=signed_integrity_evidence_chain_report_verify_ok path=... signer_role=operator signer_pubkey=<hex>
```

The verifier opens no state-store and writes nothing. It exits nonzero on any refusal and emits a closed-tag reason line so log scrapers can classify failures. Output formats:

- `--format events` (default) → start line + one success/failure line.
- `--format json` → compact metadata view of the verified wrapper (`schema_version`, `signed_at_utc`, `signer_role`, `signer_pubkey_hex`, `signature_hex`, plus embedded `chain_report.schema_version` / `chain_report.generated_at_utc` / `chain_report.signed_bundle_path` / `chain_report.effective_base_dir` / `chain_report.bundle_signer_role` / `chain_report.bundle_signer_pubkey_hex` / `chain_report_bundle_byte_counts` / `chain_report_child_counts`). Operational chatter goes to stderr so `jq` works. Does NOT re-print the embedded chain report's per-entry or per-child lists — operators who want them already have the wrapper on disk.
- `--format pretty` → terminal-friendly summary.

**Flag rules (clap-enforced):**

- `sign-integrity-evidence-chain-report` requires `--chain-report-in`, `--signer-seed`, `--signer-role`, and `--out`. Missing any refuses at parse time.
- `verify-integrity-evidence-chain-report-signature` requires `--signed-chain-report` and `--expected-signer-pubkey-hex`. The wrapper's own `signer_pubkey_hex` field is forensic context only; operators always supply the trust anchor.
- All four `--signer-role` values are accepted: `operator` / `contributor` / `dispatcher` / `coordinator` — the Stage 12.20 closed enum is REUSED because the variants are role names, not artifact-type names.
- Neither subcommand exposes `--no-prune-state-on-start` — pinned by clap regression. Neither opens a contributor state-store.
- Neither subcommand exposes `--base-dir` — that's a Stage 12.22 concept; this verifier only attests to chain-report JSON bytes.

**Refusal modes (typed `SignedIntegrityEvidenceChainReportError`, surfaced as `reason=<tag>`):**

- `signer_pubkey_mismatch` → the wrapper's `signer_pubkey_hex` doesn't equal `--expected-signer-pubkey-hex`. Cheap pre-check — no crypto burn.
- `signature_mismatch` → the wrapper was hand-edited (any field of the canonical body changed) OR the signature was tampered. Stage 12.25 covers `signer_role`, `signed_at_utc`, `signer_pubkey_hex`, AND every byte of the embedded `chain_report` (including Stage 12.24's minimal signer-metadata fields `bundle_signer_role` / `bundle_signer_pubkey_hex`, every `BundleEntryVerifyOutcome.outcome` enum payload in the embedded `BundleVerifyReport`, every `ChainChildEntryOutcome.signature_outcome` enum payload, and the summary counts) — tampering any of them trips this.
- `unsupported_schema_version` → wrapper came from a future stage; v1 binary refuses.
- `unsupported_chain_report_schema_version` → embedded chain report came from a future Stage 12.24 lineage; refuse to deserialize.
- `signing` → Ed25519 decode/verify primitive returned an error (malformed hex, bad length, etc.).
- `canonical` → bincode encoding of the canonical body failed (closed-set struct; should be impossible in practice).
- `io` → FS read of the wrapper failed (missing file, permissions, etc.).
- `malformed_json` → wrapper JSON didn't parse as a v1 `SignedIntegrityEvidenceChainReport`.

**What the signature covers (vs what it doesn't):**

- **Covered**: every field of the wrapper EXCEPT `signature_hex` — `schema_version`, `signed_at_utc`, `signer_pubkey_hex`, `signer_role`, AND every byte of the embedded `IntegrityEvidenceChainReport` (including per-child outcomes, the embedded `BundleVerifyReport` with its per-entry outcomes, Stage 12.24's minimal `bundle_signer_*` fields, and the summary counts). The signature transitively pins the entire forensic record of one chain verification under a single Ed25519 signature.
- **Not covered** (deliberate v1 scope): re-running any of the Stage 12.24 gates. Operators wanting to re-verify run `verify-integrity-evidence-chain` directly. Recursive verification of embedded `signed_state_integrity_baseline` / `signed_state_integrity_diff` / `signed_integrity_evidence_bundle` artifacts is also out-of-scope — operators run Stage 12.20 / 12.21 / 12.23 verifiers separately.

**Determinism:** signing the same chain report + same seed + same `signed_at_utc` produces a byte-identical wrapper — operators can commit a signed chain report to an archive and re-derive it deterministically.

**The six-stage forensic chain (terminating in Stage 12.25):**

```sh
# 1. Stage 12.20 — sign integrity baseline
# 2. Stage 12.21 — sign integrity diff
# 3. Stage 12.22 — build evidence bundle
# 4. Stage 12.23 — sign evidence bundle
# 5. Stage 12.24 — chain verify (signed-bundle + bundle-byte + child signatures)
# 6. Stage 12.25 — sign chain report
```

An operator reviewing a Stage 12.25 signed chain report has, in one file:
- A wrapper signature pinning the entire chain-verification record.
- The embedded chain report describing what Stage 12.24 found.
- The bundle signer's identity (via Stage 12.24's `bundle_signer_pubkey_hex`).
- The bundle byte verify outcomes for every artifact under `base_dir`.
- The per-child signature outcomes for every signed-baseline / signed-diff entry.

If the operator wants to **re-verify the chain**, they run Stage 12.24 `verify-integrity-evidence-chain` against the original `signed_bundle_path` (which is recorded in the chain report). The Stage 12.25 wrapper does NOT do this — the two operations are decoupled.

**No protocol surface touched.** No envelope, no canonical-byte changes on Stage 12.0–12.24 surfaces, no `STATE_VERSION` / `STATE_INTEGRITY_REPORT_SCHEMA_VERSION` / `STATE_INTEGRITY_DIFF_SCHEMA_VERSION` / `SIGNED_BASELINE_SCHEMA_VERSION` / `SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION` / `INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION` / `SIGNED_INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION` / `INTEGRITY_EVIDENCE_CHAIN_REPORT_SCHEMA_VERSION` bump, no SNIP / mesh / chain / payment / proof / marketplace surface.

### Stage 13.0 — chain anchoring for integrity evidence

**Use when:** you've produced a Stage 12.25 `SignedIntegrityEvidenceChainReport` and want SUM Chain to attest its existence — "this specific signed evidence artifact existed on disk and was submitted by its signer at this time" — without putting the full JSON wrapper on-chain. The chain stores a 32-byte BLAKE3 hash of the **raw on-disk wrapper bytes** + a small metadata digest (~70 bytes) under a 64-byte Ed25519 signature; the full forensic record stays local.

**Stage 13.0 ships a stub chain client only.** No real SUM Chain RPC is wired up — `submit-integrity-evidence-anchor` writes to a local on-disk anchor registry under `--anchor-registry-dir` and emits a deterministic stub `tx_id`. Stage 13.1 (deferred) replaces the stub with the real SUM Chain submission path after chain-team review of the wire spec frozen in `docs/stage13-evidence-anchor-spec.md`.

**Same-key submitter rule.** The Ed25519 seed supplied via `--submitter-seed` MUST derive the same public key embedded in the Stage 12.25 wrapper (`signer_pubkey_hex`). Stage 13.0 only allows the artifact signer to anchor their own artifact; relay / separate-submitter flows are deferred.

**Workflow — submit:**

```sh
# Stage 12.25 wrapper produced by sign-integrity-evidence-chain-report:
#   /var/omni-evidence/2026-06-15.chain.signed.json
# Anchor registry directory (DEDICATED — NOT the Stage 12.7 contributor --state-dir):
#   /var/omni-anchors/

omni-node operator evidence-anchor submit-integrity-evidence-anchor \
  --signed-chain-report /var/omni-evidence/2026-06-15.chain.signed.json \
  --submitter-seed /etc/omni-node/chain-report-signer.seed \
  --anchor-registry-dir /var/omni-anchors \
  --json-out /var/omni-evidence/2026-06-15.chain.anchor.json
# event=integrity_evidence_anchor_submit_started signed_chain_report=...
# event=integrity_evidence_anchor_submit_ok artifact_hash_hex=<hex> signer_pubkey_hex=<hex> \
#       tx_id=anchor-00000000-<hash-prefix> anchor_schema_version=1 \
#       artifact_schema_version=1 artifact_kind=signed_integrity_evidence_chain_report
# event=integrity_evidence_anchor_json_written path=/var/omni-evidence/2026-06-15.chain.anchor.json
```

The submitter command is **gated behind `--features submit`**, mirroring every other chain-touching operator subcommand. Without the feature, the command is absent from `--help` entirely and the binary contains no anchor-submit code path.

**Pre-submit gates (in order):**

1. Read the **raw on-disk bytes** of `--signed-chain-report` (these are exactly what gets hashed into the anchor).
2. Parse the wrapper to extract metadata: `schema_version`, `signer_pubkey_hex`, `signed_at_utc`.
3. **Verify the wrapper signature** under its embedded `signer_pubkey_hex`. We refuse to anchor a wrapper that fails its own signature check (`wrapper_signature_invalid`).
4. Read the 32-byte `--submitter-seed` file and check the derived pubkey equals the wrapper's `signer_pubkey_hex` (same-key submitter rule → `submitter_pubkey_mismatch` on divergence).
5. Compute `artifact_hash = blake3(raw_bytes)` — binds the chain record to the exact byte sequence the operator holds. Any whitespace, key-ordering, or formatting change in the on-disk file produces a different anchor.
6. Build the digest, sign canonical bytes with the submitter seed, submit through the stub chain client, persist the record under `<anchor-registry-dir>/<artifact_hash_hex>.json` + update `<anchor-registry-dir>/tx_index.json` atomically.

**Workflow — query (registry-backed):**

```sh
# By artifact hash:
omni-node operator evidence-anchor query-integrity-evidence-anchor \
  --anchor-registry-dir /var/omni-anchors \
  --artifact-hash-hex <64-hex>
# By tx_id:
omni-node operator evidence-anchor query-integrity-evidence-anchor \
  --anchor-registry-dir /var/omni-anchors \
  --tx-id anchor-00000000-<hash-prefix>
# event=integrity_evidence_anchor_query_started ...
# event=integrity_evidence_anchor_query_ok artifact_hash_hex=<hex> tx_id=... \
#       local_status=submitted chain_status=submitted transitioned=false
```

The query command opens **only** the anchor registry — no Stage 12.7 contributor state-store is touched. Status mirrors the Stage 5 chain v1 five-state model (`submitted | included | finalized | failed | unknown`); Stage 13.0's stub client defaults to `submitted` until per-`tx_id` overrides are configured by Stage 13.1.

**Workflow — verify (registry-backed — the primary verification command):**

```sh
omni-node operator evidence-anchor verify-integrity-evidence-anchor \
  --signed-chain-report /var/omni-evidence/2026-06-15.chain.signed.json \
  --anchor-registry-dir /var/omni-anchors \
  [--tx-id anchor-00000000-<hash-prefix>]
# event=integrity_evidence_anchor_verify_started ...
# event=integrity_evidence_anchor_verify_ok artifact_hash_hex=<hex> signer_pubkey_hex=<hex> \
#       tx_id=anchor-... local_status=submitted anchor_schema_version=1 \
#       artifact_schema_version=1 artifact_kind=signed_integrity_evidence_chain_report
```

Lookup semantics:

- If `--tx-id` is **absent** (default): recompute `artifact_hash = blake3(raw_bytes)`, look up by hash. Registry miss → `anchor_not_found`.
- If `--tx-id` is **supplied**: look up by `tx_id` via `tx_index.json`. The recorded `digest.artifact_hash` MUST equal the recomputed hash; mismatches refuse with `artifact_hash_mismatch`.

This command proves: "this on-disk artifact corresponds to a recorded anchor in the registry that was authored by the artifact signer." It does NOT prove on-chain inclusion (Stage 13.0 has no real chain) — that's a Stage 13.1 deliverable.

**Workflow — verify-file (standalone JSON — secondary):**

```sh
omni-node operator evidence-anchor verify-integrity-evidence-anchor-file \
  --signed-chain-report /var/omni-evidence/2026-06-15.chain.signed.json \
  --anchor-json       /var/omni-evidence/2026-06-15.chain.anchor.json
# event=integrity_evidence_anchor_verify_file_started ...
# event=integrity_evidence_anchor_verify_file_ok artifact_hash_hex=<hex> signer_pubkey_hex=<hex> ...
```

The standalone-JSON command does **not** consult the registry — it only proves the anchor JSON is internally consistent with the local on-disk wrapper bytes. Useful for offline forensic review or vetting an anchor JSON received out-of-band before importing it.

**`--anchor-registry-dir` is NOT `--state-dir`.** The anchor registry lives under a dedicated directory (typical convention: `/var/omni-anchors/`). It is distinct from the Stage 12.7 contributor `--state-dir` and from the Stage 5 attestation `--registry-path`. The flag name + dedicated directory make the boundary unambiguous.

**Refusal modes (typed `EvidenceAnchorError`, surfaced as `reason=<tag>`):**

- `wrapper_signature_invalid` — the Stage 12.25 wrapper's own signature did not verify under its embedded `signer_pubkey_hex`. Surfaced before any chain interaction. We do not anchor unverifiable artifacts.
- `submitter_pubkey_mismatch` — `--submitter-seed` derived pubkey ≠ wrapper `signer_pubkey_hex`. Same-key submitter rule.
- `submitter_signature_invalid` — anchor wire payload's `submitter_signature` did not verify under `digest.signer_pubkey`. Trips on tampered anchor JSON.
- `artifact_hash_mismatch` — recomputed `blake3(raw_bytes)` ≠ the recorded / anchor `digest.artifact_hash`. Trips when the on-disk wrapper file has been re-formatted, re-pretty-printed, or otherwise mutated after submit.
- `anchored_signer_pubkey_mismatch` — verify-time same-key binding: the stored / supplied anchor's `digest.signer_pubkey` does not equal the parsed Stage 12.25 wrapper's `signer_pubkey_hex`. Defends against a hand-edited registry record or a tampered standalone anchor that reuses the artifact hash but swaps in a different signer pubkey (with a valid signature by that other key).
- `anchor_not_found` — registry lookup miss (no record for the supplied `--artifact-hash-hex` / `--tx-id`).
- `unsupported_anchor_schema_version` — wire payload `anchor_schema_version` ≠ 1.
- `unsupported_artifact_schema_version` — wrapper `schema_version` outside the supported set (Stage 13.0 supports only v1 wrappers).
- `unsupported_artifact_kind` — wire payload `artifact_kind` not in the closed enum (Stage 13.0 ships one variant; new kinds require an `anchor_schema_version` bump).
- `malformed_seed_file` — `--submitter-seed` file did not parse / was not exactly 32 bytes.
- `malformed_json` — wrapper JSON or anchor JSON parse failure.
- `malformed_signed_at_utc` — wrapper's `signed_at_utc` field did not parse as RFC 3339.
- `chain_client` — stub chain-client failure (rare; configurable in tests).
- `io` — FS / registry IO failure.

**Stage 13.2 chain-mode additions** (fire only when `--rpc-url` + `--expect-chain-id` are supplied):

- `chain_id_mismatch` — `--expect-chain-id` ≠ `params.chain_id`. CLI preflight gate; no anchor RPC fires.
- `not_activated` — chain reports anchor RPC not yet active (non-mainnet dormant / scheduled, OR mainnet scheduled-but-not-yet-reached). Emitted by CLI preflight; the adapter additionally emits `not_activated` if a non-CLI caller bypasses preflight.
- `mainnet_policy_unresolved` — `chain_id == 1` AND `integrity_evidence_anchor_enabled_from_height == None` (chain governance has not set mainnet activation). Explicit, captures "mainnet anchors are not yet permitted" without implying a permanent refusal. Once mainnet sets `Some(h)`, refusal drops to `not_activated`.
- `chain_rpc` — transport-layer failure (HTTP / body read / JSON-RPC envelope malformed / missing `result` field). Routed through `omni_sumchain::classify_chain_client_error`; CLI never inspects raw error strings.
- `chain_submit_refused` — chain returned a JSON-RPC `error` object (chain refused at the application layer; chain's text surfaced verbatim).
- `chain_response_malformed` — chain returned success but the response shape could not be parsed into the expected DTO, OR the response carried an unrecognized status enum string (e.g. `status: "foo"`).

**What the chain attests (vs what it does not):**

- **Attested**: existence of `(artifact_hash, signer_pubkey, signed_at, artifact_kind, anchor_schema_version, artifact_schema_version)`. The chain proves the operator submitted a commitment to this exact byte sequence at this time, under this signing key.
- **NOT attested**: semantic correctness of the underlying integrity evidence. Stage 12.20-12.25's gates remain the source of truth for whether the evidence describes a successful verification — chain inclusion does not certify gate outcomes. Re-running `verify-integrity-evidence-chain` against the wrapper's `signed_bundle_path` is unchanged and unaffected by Stage 13.0.

**The chain-anchor pipeline:**

Stage 13.0 stub mode (default — local testing, regen, no chain):
```sh
# 1-6: Stage 12.20-12.25 (forensic chain — see above)
# 7. Stage 13.0 — submit chain anchor via stub client
omni-node operator evidence-anchor submit-integrity-evidence-anchor \
  --signed-chain-report /var/omni-evidence/2026-06-15.chain.signed.json \
  --submitter-seed     /etc/omni-node/chain-report-signer.seed \
  --anchor-registry-dir /var/omni-anchors \
  --json-out           /var/omni-evidence/2026-06-15.chain.anchor.json
# event=integrity_evidence_anchor_submit_ok ... tx_id=anchor-00000000-<hash-prefix>
```

Stage 13.2 chain mode (real SUM Chain RPC; gated `--features submit`):
```sh
# Submit
omni-node operator evidence-anchor submit-integrity-evidence-anchor \
  --signed-chain-report /var/omni-evidence/2026-06-15.chain.signed.json \
  --submitter-seed     /etc/omni-node/chain-report-signer.seed \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url            https://rpc.sumchain.example.com \
  --expect-chain-id    42 \
  --allow-submit
# event=integrity_evidence_anchor_submit_ok ... tx_id=0x<chain_tx_hash>

# Query (chain-read-only, local-registry-mutating)
omni-node operator evidence-anchor query-integrity-evidence-anchor \
  --anchor-registry-dir /var/omni-anchors \
  --tx-id              0x<chain_tx_hash> \
  --rpc-url            https://rpc.sumchain.example.com \
  --expect-chain-id    42

# Reconcile sweep (chain-read-only, local-registry-mutating)
omni-node operator evidence-anchor reconcile-integrity-evidence-anchor \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url            https://rpc.sumchain.example.com \
  --expect-chain-id    42
```

**Verify (unchanged — local-only in BOTH stub and chain modes):**

```sh
omni-node operator evidence-anchor verify-integrity-evidence-anchor \
  --signed-chain-report /var/omni-evidence/2026-06-15.chain.signed.json \
  --anchor-registry-dir /var/omni-anchors
```

**Mainnet posture (Stage 13.2):**

Mainnet (`chain_id == 1`) writes additionally require `--allow-mainnet-submit` AND chain governance must have set `integrity_evidence_anchor_enabled_from_height` to a value reached by the current chain head. Submission refuses with `mainnet_policy_unresolved` when activation is dormant, and `not_activated` when scheduled but not yet reached — neither can be overridden via flags.

**Activation state machine** (drives the mainnet/non-mainnet tagging split):

| chain_id | `integrity_evidence_anchor_enabled_from_height` | head vs. h | Refusal tag |
| --- | --- | --- | --- |
| `1` (mainnet) | `None` | — | `mainnet_policy_unresolved` |
| `1` (mainnet) | `Some(h)` | `head < h` | `not_activated` |
| `1` (mainnet) | `Some(h)` | `head >= h` | OK (subject to opt-ins) |
| ≠ 1 (non-mainnet) | `None` | — | `not_activated` |
| ≠ 1 (non-mainnet) | `Some(h)` | `head < h` | `not_activated` |
| ≠ 1 (non-mainnet) | `Some(h)` | `head >= h` | OK (subject to `--allow-submit`) |

**Determinism:** Anchoring the same on-disk wrapper bytes with the same submitter seed produces a byte-identical `IntegrityEvidenceAnchorTxData` payload — operators can commit the anchor JSON to an archive alongside the wrapper bytes and re-derive it.

**Implementation reference:** [`docs/stage13.2-chain-adapter.md`](stage13.2-chain-adapter.md) — Stage 13.2 engineering doc (RPC method names, DTO ownership, classifier).

### Stage 13.3 — anchor-lifecycle operations (summary, health, watch)

**Use when:** you want a fast local snapshot of anchor registry state, want to detect stale `Submitted` / `Included` records that the chain hasn't progressed, or want a long-running monitor that periodically reconciles + summarizes. All three commands are **operator-side hardening only** — Stage 13.0 wire / schema / domain / reason tags / same-key submitter model unchanged; no new chain RPCs.

**`summary-integrity-evidence-anchors`** — fully local, **no chain interaction**:

```sh
# Fast snapshot (default).
omni-node operator evidence-anchor summary-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors
# event=integrity_evidence_anchor_summary total=12 submitted=2 included=1 finalized=8 failed=1

# Optionally include stale rows (time-based — uses the existing
# `submitted_at` field, no record-shape change).
omni-node operator evidence-anchor summary-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors \
  --stale-threshold-secs 86400
# event=integrity_evidence_anchor_summary total=12 ...
# event=integrity_evidence_anchor_stale artifact_hash_hex=<hex> tx_id=0x<hash> status=submitted age_secs=129600 threshold_secs=86400
# event=integrity_evidence_anchor_stale_summary count=1 threshold_secs=86400

# Optionally emit a registry-health diagnostic (read-only).
omni-node operator evidence-anchor summary-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors \
  --include-health
# event=integrity_evidence_anchor_health records=12 malformed_records=0 orphan_tx_index_entries=0 orphan_tmp_files=1
```

**Summary is fully local.** No `--rpc-url` flag. Chain reconciliation is the `reconcile-integrity-evidence-anchor` (one-shot) and `watch-integrity-evidence-anchors` (continuous) commands' job — separation of concerns lets the summary stay a fast scriptable check.

**`watch-integrity-evidence-anchors`** — chain-read-only, local-registry-mutating, **never submits**:

```sh
omni-node operator evidence-anchor watch-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url            https://rpc.sumchain.example.com \
  --expect-chain-id    42 \
  --poll-interval-secs 60 \
  --stale-threshold-secs 86400
# event=integrity_evidence_anchor_watch_started anchor_registry_dir=/var/omni-anchors rpc_url=... expect_chain_id=42 poll_interval_secs=60
# event=integrity_evidence_anchor_watch_tick tick=1
# event=integrity_evidence_anchor_reconcile_record_ok artifact_hash_hex=... tx_id=0x... local_status=included chain_status=included transitioned=false
# event=integrity_evidence_anchor_summary total=12 ...
# event=integrity_evidence_anchor_stale ...
# event=integrity_evidence_anchor_stale_summary count=0 threshold_secs=86400
# ... (next tick after --poll-interval-secs)
# Ctrl-C → event=integrity_evidence_anchor_watch_stopped cause=ctrl_c ticks=N
```

Each tick re-runs Stage 13.2's `reconcile-integrity-evidence-anchor` sweep (one chain query per `Submitted`/`Included` record), emits a summary line, and optionally emits stale rows. **Ctrl-C** triggers graceful shutdown after the current tick. `--max-ticks N` stops after N ticks with `cause=max_ticks` — primary use case is scripted single-shot reconcile (`--max-ticks 1`) or bounded test runs.

**Locked `cause=` vs `reason=` invariant.** Watch-stop events use `cause=ctrl_c` / `cause=max_ticks` — the `reason=` key stays reserved for the closed refusal/failure taxonomy. Log scrapers can `grep 'reason='` for refusals only and `grep 'cause='` for informational stops:

```sh
# All refusals (any subcommand, any stage):
grep 'reason=' omni-node.log

# All clean watch stops:
grep 'event=integrity_evidence_anchor_watch_stopped cause=' omni-node.log
```

**Watch never submits.** Stage 13.3 watch is monitor-only — no `--allow-submit` flag, no retry-on-failed semantics, no resubmit-on-dropped path. Anchors are one-shot per signed wrapper; to re-submit, operators re-invoke `submit-integrity-evidence-anchor` explicitly. This is a locked Stage 13.3 invariant (Q5).

**Time-based staleness (locked Q1).** Stage 13.3 uses the existing `AnchorRecord::submitted_at: DateTime<Utc>` field — no record-shape change. Block-based staleness is deferred (would require adding `submitted_at_block: Option<u64>` to `AnchorRecord`, the Stage 5.2 posture for attestations).

**No new reason tags.** Stage 13.3 adds informational events only (`_summary`, `_stale`, `_stale_summary`, `_health`, `_watch_started`, `_watch_tick`, `_watch_stopped`, `_watch_tick_failed`); failures inside any command continue to route through the existing closed-set tags from Stage 13.0 / 13.2 (`chain_id_mismatch`, `not_activated`, `mainnet_policy_unresolved`, `chain_rpc`, `chain_submit_refused`, `chain_response_malformed`, `io`, `malformed_json`, `anchor_not_found`, …). The taxonomy stays frozen.

**When to use which command:**

| Need | Command |
| --- | --- |
| Fast local count snapshot | `summary-integrity-evidence-anchors` (no flags) |
| Quick local stale-record check | `summary-integrity-evidence-anchors --stale-threshold-secs N` |
| Registry directory diagnostic | `summary-integrity-evidence-anchors --include-health` |
| One-shot chain reconcile | `reconcile-integrity-evidence-anchor` (Stage 13.2) |
| Continuous chain monitor | `watch-integrity-evidence-anchors` (Stage 13.3) |

**Implementation reference:** [`docs/stage13.3-anchor-operations.md`](stage13.3-anchor-operations.md) — Stage 13.3 engineering doc (library helpers, time-based staleness rationale, event taxonomy, `cause=` vs `reason=` invariant).

### Stage 13.4 — anchor-registry cleanup with quarantine

**Use when:** Stage 13.3 detected one or more issues you want to act on — orphan `.tmp` files, orphan `tx_index.json` entries, malformed record files, or stale `Submitted` / `Included` records. Stage 13.4 turns those signals into a planned cleanup that defaults to dry-run and prefers quarantine over deletion. **Fully local — zero chain interaction.** Stage 13.0 wire / schema / domain unchanged.

**Three-phase workflow:**

```sh
# Phase 1 — Plan (default; no FS mutations)
omni-node operator evidence-anchor cleanup-integrity-evidence-anchor-registry \
  --anchor-registry-dir /var/omni-anchors \
  --plan-out           /tmp/cleanup-plan.json \
  --stale-threshold-secs 86400
# event=integrity_evidence_anchor_cleanup_plan_started ...
# event=integrity_evidence_anchor_cleanup_plan_written path=/tmp/cleanup-plan.json
# event=integrity_evidence_anchor_cleanup_plan_summary plan_id=<16-hex> actions=N tier_a=A tier_b=B gated=G

# Phase 2a — Apply, DRY-RUN (default — no --apply)
omni-node operator evidence-anchor cleanup-integrity-evidence-anchor-registry \
  --anchor-registry-dir /var/omni-anchors \
  --apply-plan         /tmp/cleanup-plan.json \
  --quarantine-dir     /var/omni-anchors-quarantine
# event=integrity_evidence_anchor_cleanup_apply_started ... mode=dry_run
# event=integrity_evidence_anchor_cleanup_apply_action_outcome ... status=would_apply
# event=integrity_evidence_anchor_cleanup_apply_summary mode=dry_run actions_dry_run=N ...

# Phase 2b — Apply, REAL (explicit --apply confirmation)
omni-node operator evidence-anchor cleanup-integrity-evidence-anchor-registry \
  --anchor-registry-dir /var/omni-anchors \
  --apply-plan         /tmp/cleanup-plan.json \
  --quarantine-dir     /var/omni-anchors-quarantine \
  --apply \
  --allow-stale-quarantine   # required iff plan contains stale-record actions
# event=integrity_evidence_anchor_cleanup_apply_started ... mode=apply
# event=integrity_evidence_anchor_cleanup_apply_action_outcome ... status=applied
# event=integrity_evidence_anchor_cleanup_apply_summary mode=apply actions_applied=N quarantine_manifest_relative=<plan_id>/quarantine_manifest.json

# Phase 3 — Restore (undo a prior apply via the quarantine manifest)
omni-node operator evidence-anchor cleanup-integrity-evidence-anchor-registry \
  --anchor-registry-dir /var/omni-anchors \
  --restore-manifest   /var/omni-anchors-quarantine/<plan_id>/quarantine_manifest.json
# event=integrity_evidence_anchor_cleanup_restore_started ...
# event=integrity_evidence_anchor_cleanup_restore_outcome ... status=restored
# event=integrity_evidence_anchor_cleanup_restore_summary restored=N skipped=K
```

**Closed mutex / required-with rules** (CLI refuses with a clap usage error before any FS read):

- `--apply-plan` and `--restore-manifest` are mutually exclusive.
- `--plan-out` and `--stale-threshold-secs` are plan-mode only.
- `--apply`, `--quarantine-dir`, `--allow-stale-quarantine` are apply-mode only.
- `--apply` requires `--apply-plan`.
- A plan containing Tier B actions requires `--quarantine-dir` at apply.

**Three locked invariants you should rely on:**

1. **Dry-run is the default.** Without `--apply`, the apply phase mutates nothing. You can review every action's `would_apply` outcome before committing.
2. **Quarantine first for Tier B.** Malformed records and stale records are copied to `<quarantine-dir>/<plan_id>/<source_relative>` with a manifest entry recording the BLAKE3 + bytes BEFORE the source is removed. Restore reads the manifest to put bytes back exactly where they came from.
3. **Drift refusal is the safety net.** If the registry changed between plan and apply (e.g. the watch loop transitioned an anchor while you were reviewing the plan), apply refuses with `reason=cleanup_drift` instead of acting on stale assumptions. Re-plan and re-apply.

**Stale-cleanup needs TWO opt-ins.** First, `--stale-threshold-secs` at plan time (without it, no stale-cleanup actions are emitted). Second, `--allow-stale-quarantine` at apply time. This protects against accidentally quarantining records that the chain might still finalize.

**Reason-tag taxonomy** (closed; surfaced on `event=integrity_evidence_anchor_cleanup_failed reason=<tag>` lines):

- `cleanup_drift` — registry drifted since plan was generated. **New tag string in Stage 13.4.**
- `cleanup_invalid_path` — plan or manifest entry has an invalid `source_relative` / `quarantine_relative` (absolute path, `..` traversal, separator-containing, or wrong per-kind shape). **New tag string added in the Stage 13.4 REJECT-fix loop to refuse operator-supplied JSON before any FS mutation.**
- `unsupported_cleanup_plan_schema_version` — plan's `schema_version` is not the locked `ANCHOR_CLEANUP_PLAN_SCHEMA_VERSION`. **New tag string added in the Stage 13.4 REJECT-fix loop; refused before the plan-hash check so future-schema plans cannot apply.**
- `cleanup_plan_hash_mismatch` — plan was hand-edited / corrupted. Reused from Stage 12.17.
- `gate_required` — gated action (`QuarantineStaleOpenRecord`) without `--allow-stale-quarantine`. Reused from Stage 12.17.
- `quarantine_blake3_mismatch` — quarantined bytes differ from manifest record. Reused from Stage 12.18.
- `restore_target_exists` — restore target path is already populated by something else. Reused from Stage 12.18.
- Plus the existing Stage 13.0 / 13.2 / 13.3 tags (`malformed_json`, `io`, etc.).

CLI mutex / required-with refusals exit non-zero via clap usage errors (NOT `reason=…` event lines) — they're argument-parse concerns, not the closed refusal taxonomy.

**Workflow recipe — "I noticed a stale anchor in Stage 13.3 watch output":**

```sh
# 1. Plan with a stale-threshold matching what watch showed.
omni-node operator evidence-anchor cleanup-integrity-evidence-anchor-registry \
  --anchor-registry-dir /var/omni-anchors \
  --plan-out           /tmp/p.json \
  --stale-threshold-secs 604800   # 1 week

# 2. Inspect /tmp/p.json. Confirm the actions look right.
jq '.actions' /tmp/p.json

# 3. Dry-run apply.
omni-node operator evidence-anchor cleanup-integrity-evidence-anchor-registry \
  --anchor-registry-dir /var/omni-anchors \
  --apply-plan         /tmp/p.json \
  --quarantine-dir     /var/omni-anchors-quarantine

# 4. Real apply with both opt-ins.
omni-node operator evidence-anchor cleanup-integrity-evidence-anchor-registry \
  --anchor-registry-dir /var/omni-anchors \
  --apply-plan         /tmp/p.json \
  --quarantine-dir     /var/omni-anchors-quarantine \
  --apply \
  --allow-stale-quarantine

# 5. If you change your mind, restore from the manifest.
omni-node operator evidence-anchor cleanup-integrity-evidence-anchor-registry \
  --anchor-registry-dir /var/omni-anchors \
  --restore-manifest   /var/omni-anchors-quarantine/<plan_id>/quarantine_manifest.json
```

**Implementation reference:** [`docs/stage13.4-anchor-cleanup.md`](stage13.4-anchor-cleanup.md) — Stage 13.4 engineering doc (action taxonomy, mutex rules, drift / plan-hash recipes, exact-FS-operation appendix).

### Stage 13.5 — local-only anchor export and verify

**Use when:** you need to hand off a subset of anchor records (plus optionally the raw artifact bytes and / or a Stage 12.25 signed-chain-report) to another operator host — for forensic record retention, cross-team review, or as an evidence artifact to keep alongside the on-disk registry snapshot. **Fully local — zero chain interaction.** Anchor registry is read-only. Stage 13.0 wire / domain / canonical-bytes / signing unchanged.

**Subcommands:**
- `omni-node operator evidence-anchor export-integrity-evidence-anchors` — produce a portable manifest + bytes subtree.
- `omni-node operator evidence-anchor verify-integrity-evidence-anchor-export` — re-check a previously-produced export tree.

**Key invariants:**

1. **At least one selector is required** (`--status`, `--tx-id`, or `--artifact-hash-hex`). No accidental "export everything." Misuse refuses at the clap layer with `at least one of --status, --tx-id, --artifact-hash-hex is required` — NOT via `reason=...`.
2. **No-clobber on `--export-out`.** The dir must be empty or non-existent. There is no `--force` — operator picks a fresh path or deletes the old one first.
3. **`--include-artifact-bytes` is a pair form** `<PATH>:<ARTIFACT_HASH_HEX>`. The file's BLAKE3 must equal the claimed hash; export refuses with `reason=export_entry_metadata_mismatch` on drift.
4. **Manifest does NOT carry `anchor_registry_dir`.** A portable handoff artifact does not leak host-local path layout. Operator-supplied `--label` / `--notes` are the provenance channel.
5. **What verify proves:** the export bytes match the manifest; each anchor's submitter signature is valid under its embedded `digest.signer_pubkey`; per-record metadata in the manifest matches the record's own fields; for paired `artifact_bytes`, `blake3(bytes) == record.tx_data.digest.artifact_hash` (the **artifact-hash binding**).
6. **What verify does NOT prove:** the Stage 12.25 **wrapper signer**. That binding lives in a signed-chain-report — Stage 12.25 own-signature verification is **out of scope** here. Operator runs Stage 12.25 verify on the signed-chain-report separately. Stage 13.5 verifies BLAKE3-of-bytes for any included signed-chain-report; it does NOT re-verify Stage 12.25's signature.
7. **`--strict` is a verification-time gate**, NOT a clap-level usage error. Missing `artifact_bytes` for an anchor record refuses with `reason=export_strict_mode_artifact_bytes_missing`.

**Reason-tag taxonomy** (closed; surfaced on `event=integrity_evidence_anchor_(export|verify_export)_failed reason=<tag>` lines):

- `unsupported_export_manifest_schema_version` — manifest declares a `schema_version` other than 1. **New tag in Stage 13.5.**
- `export_manifest_hash_mismatch` — recomputed canonical-bytes BLAKE3 of the manifest does not match the declared `export_manifest_hash`. **New tag in Stage 13.5.**
- `export_invalid_path` — manifest entry's `relative_path` is absolute, contains `..`, contains backslash, or violates the per-kind shape (anchors/<64-lower-hex>.json, artifacts/<64-lower-hex>, signed_chain_reports/<safe-basename>). **New tag in Stage 13.5.**
- `export_blake3_mismatch` — a copied file's BLAKE3 or length does not match the manifest's declaration. **New tag in Stage 13.5.**
- `export_entry_metadata_mismatch` — a manifest entry's metadata (artifact_hash / tx_id / status) does not match the record file's actual fields, OR a paired artifact_bytes file's `blake3(bytes)` does not equal `record.tx_data.digest.artifact_hash`. **New tag in Stage 13.5** (Q9 fold — single tag for every "manifest claim doesn't match the underlying byte fact" case).
- `export_strict_mode_artifact_bytes_missing` — verify-time `--strict` gate: an anchor_record entry has no paired artifact_bytes entry for the same `artifact_hash_hex`. **New tag in Stage 13.5.**
- `anchor_not_found` — `--tx-id` or `--artifact-hash-hex` selector points at a record that does not exist. **Reused** from Stage 13.0 (corrected from the v1 plan's `record_not_found` — that tag does not exist in the closed taxonomy).
- `malformed_json` — manifest or anchor record JSON parse failure. Reused.
- `io` — file presence / read errors; also covers `--export-out` non-empty refusal. Reused.
- `submitter_signature_invalid`, `unsupported_anchor_schema_version` — refusals from the Stage 13.0 verifier (`verify_anchor_tx_data`) when an exported record's signature or schema is bad. Reused.

CLI mutex / required-with refusals exit non-zero via clap usage errors (NOT `reason=…` event lines) — they're argument-parse concerns, not the closed refusal taxonomy.

**Workflow recipe — hand off a finalized anchor to another operator host for retention:**

```sh
# 1. On host A, plan-and-write the export.
omni-node operator evidence-anchor export-integrity-evidence-anchors \
  --anchor-registry-dir         /var/omni-anchors \
  --export-out                  /tmp/anchor-export-2026-06-17 \
  --status                      finalized \
  --include-artifact-bytes      /var/omni-artifacts/<hash>.signed.json:<artifact_hash_hex> \
  --include-signed-chain-report /var/omni-artifacts/<hash>.signed.json \
  --label                       "Q2-2026 forensic retention"

# 2. Inspect the manifest.
jq '.entries | map({kind, relative_path, status})' \
   /tmp/anchor-export-2026-06-17/evidence_anchor_export_manifest.json

# 3. Transport /tmp/anchor-export-2026-06-17 to host B.
tar czf anchor-export.tar.gz -C /tmp anchor-export-2026-06-17

# 4. On host B, verify the export.
omni-node operator evidence-anchor verify-integrity-evidence-anchor-export \
  --export-dir /var/anchor-retention/anchor-export-2026-06-17

# 5. Optional: strict mode for full artifact-hash binding.
omni-node operator evidence-anchor verify-integrity-evidence-anchor-export \
  --export-dir /var/anchor-retention/anchor-export-2026-06-17 \
  --strict
```

**Limitation — wrapper-signer binding:** Stage 13.5 verify does NOT re-verify the Stage 12.25 wrapper signer from artifact bytes alone. That binding requires the wrapper's own signature. To prove wrapper-signer binding, the operator runs Stage 12.25 verify (`omni-node operator evidence-anchor verify-integrity-evidence-anchor-file --signed-chain-report <SCR>` or equivalent) on the bundled signed-chain-report separately.

**Implementation reference:** [`docs/stage13.5-anchor-export.md`](stage13.5-anchor-export.md) — Stage 13.5 engineering doc (entry-kind taxonomy, mutex rules, manifest schema, file layout, verify preflight ordering, what's proven vs not, test inventory).

### Stage 13.6 — local-only anchor export import / registry restore

**Use when:** you have a Stage 13.5 export from another operator host and need to restore those anchor records into a local target registry — for forensic record retention, cross-team handoff, or rebuilding a registry after host loss. **Fully local — zero chain interaction.** Source export is read-only; target registry is write-once-per-record. No re-signing. Stage 13.0 wire / domain / canonical-bytes / signing unchanged.

**Subcommand:**
- `omni-node operator evidence-anchor import-integrity-evidence-anchor-export` — verify-first, default dry-run, byte-preserve.

**Key invariants:**

1. **Verify-first.** Apply calls `verify_anchor_export(--strict?)` before any FS write. A post-plan tamper of the export is caught by the durability fence on apply.
2. **Default dry-run.** Without `--apply`, the command reports `would_import` / `would_re_add_tx_index_entry` / `skipped_already_imported` outcomes; no record file is written, no `tx_index.json` is touched.
3. **No-clobber on real conflicts.** Conflict matrix:
    - Target has a byte-equal record AND tx_index maps the tx_id to the same hash → `skipped_already_imported`.
    - Target has a byte-equal record but the tx_index entry is missing → `re_added_tx_index_entry` (tx_index updated only; record file NOT rewritten).
    - Target has a byte-DIFFERENT record under the same hash → refuse with `reason=import_target_exists field=artifact_hash`.
    - Target's tx_index maps the same tx_id to a DIFFERENT hash → refuse with `reason=import_target_exists field=tx_id`.
4. **Preflight-all-before-mutate.** Apply classifies EVERY selected action before writing any record file. A tx_id collision on action #5 of 5 refuses with zero writes from actions 1–4.
5. **Byte-preserve.** Imported record files are copied verbatim from the export. Imported records carry historical `submitted_at` / `updated_at` from the source registry — Stage 13.3 stale-age views may report historical ages on imported records by design.
6. **`tx_index.json` is merged**, not rebuilt. Unrelated entries (pre-existing local submits, prior imports, Stage 13.4 restores) are preserved verbatim.
7. **`--strict` validates the WHOLE export tree** via Stage 13.5 `verify_anchor_export(strict=true)` — every `anchor_record` in the manifest must have a paired `artifact_bytes`. NOT just the records being imported.
8. **No-selector default imports all `anchor_record` entries from the manifest.** Asymmetric to Stage 13.5 export's required-selector (the export is already a curated subset).

**Reason-tag taxonomy** (closed; surfaced on `event=integrity_evidence_anchor_import_failed reason=<tag>` lines):

- `import_target_exists` — target registry already has a different record under the same artifact_hash, OR maps the import's tx_id to a different artifact_hash. **New tag in Stage 13.6.** Detail field carries `field=artifact_hash | tx_id` for disambiguation.
- `anchor_not_found` — `--tx-id` or `--artifact-hash-hex` selector points at a manifest entry that does not exist. Reused from Stage 13.0.
- `export_entry_metadata_mismatch` — manifest claim doesn't match the record's actual fields. Reused from Stage 13.5.
- `unsupported_export_manifest_schema_version`, `export_manifest_hash_mismatch`, `export_invalid_path`, `export_blake3_mismatch`, `export_strict_mode_artifact_bytes_missing` — propagated from `verify_anchor_export`. Reused from Stage 13.5.
- `submitter_signature_invalid`, `unsupported_anchor_schema_version`, `malformed_json`, `io` — reused from Stage 13.0.

CLI mutex / required-with refusals exit non-zero via clap usage errors (NOT `reason=…` event lines) — they're argument-parse concerns.

**Workflow recipe — restore a finalized anchor onto a remote operator host:**

```sh
# 1. On host A, build the export (Stage 13.5).
omni-node operator evidence-anchor export-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors \
  --export-out          /tmp/anchor-export-2026-06-18 \
  --status              finalized

# 2. Transport /tmp/anchor-export-2026-06-18 to host B (out-of-band).

# 3. On host B, sanity-verify the export (Stage 13.5; read-only).
omni-node operator evidence-anchor verify-integrity-evidence-anchor-export \
  --export-dir /var/anchor-retention/anchor-export-2026-06-18

# 4. On host B, dry-run the import.
omni-node operator evidence-anchor import-integrity-evidence-anchor-export \
  --export-dir          /var/anchor-retention/anchor-export-2026-06-18 \
  --anchor-registry-dir /var/omni-anchors

# 5. On host B, apply the import.
omni-node operator evidence-anchor import-integrity-evidence-anchor-export \
  --export-dir          /var/anchor-retention/anchor-export-2026-06-18 \
  --anchor-registry-dir /var/omni-anchors \
  --apply

# 6. (Optional) On host B, confirm the imported records are visible.
omni-node operator evidence-anchor summary-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors \
  --include-health

# 7. (Optional, recommended) On host B, run Stage 13.2 reconcile to
#    bring the imported records' status up to current chain truth.
omni-node operator evidence-anchor reconcile-integrity-evidence-anchor \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url             https://sum-chain.example/rpc \
  --expect-chain-id     <CHAIN_ID>
```

**Limitation — historical timestamps:** imported records' `submitted_at` / `updated_at` are facts from the SOURCE registry's clock, not the import time. Stage 13.3 `summary --stale-threshold-secs` will report imported records as "submitted last quarter" if they were submitted last quarter on the source host, even if they were imported on this host moments ago. This is intentional (forensic fidelity) — the import event line + file mtime are the only "this host first saw the record" traces.

**Limitation — wrapper-signer binding:** like Stage 13.5 verify, Stage 13.6 does NOT re-verify the Stage 12.25 wrapper signer. The imported anchor record's submitter signature is verified (`verify_anchor_tx_data`), but binding the anchor to its Stage 12.25 wrapper requires running Stage 12.25 verify on a paired signed-chain-report separately.

**Implementation reference:** [`docs/stage13.6-anchor-import.md`](stage13.6-anchor-import.md) — Stage 13.6 engineering doc (library surface, conflict matrix, mutex rules, verify-first invariant, byte-preserve + historical-timestamps note, test inventory).

### Stage 13.7 — local terminal-anchor archive / restore

**Use when:** the hot anchor registry has accumulated `Finalized` / `Failed` records you want to move out of the working set — for forensic retention, disk-pressure relief, or because the operational interest in those records has expired. **Fully local — zero chain interaction.** The chain is the source of truth for status; archive doesn't ask it anything. Stage 13.0 wire / schema / domain unchanged.

**Subcommand:**
- `omni-node operator evidence-anchor archive-integrity-evidence-anchors` — three-phase plan / apply / restore mirroring Stage 13.4 cleanup but targeting valid TERMINAL records.

**Key invariants:**

1. **Terminal records only.** `--status` is closed-set `finalized | failed` at the clap layer. `Submitted` / `Included` are not eligible. Default when no `--status` is given: `[FINALIZED]`. `Failed` requires explicit `--status FAILED`.
2. **Verify-first preflight.** Apply checks schema version → plan hash → drift → per-action path shape before any FS work. Each refusal routes through a new closed `reason=` tag (see taxonomy below).
3. **Two-phase durability — honest contract.**
   - **Phase 1** (before manifest lands): zero hot-registry mutation. A failure here leaves the registry byte-identical to its pre-apply state.
   - **Phase 2** (after manifest lands): destructive phase may be partially applied on IO failure. **Restore is the official recovery path** — run the same command in restore mode against the manifest the apply produced, and any source record that Phase 2 successfully deleted is re-established (row 2 idempotent for records Phase 2 didn't reach).
4. **Default dry-run for BOTH apply and restore.** `--apply` is the explicit operator confirmation in both mutation modes (Stage 13.7 REJECT-fix Finding 1). Restore without `--apply` reports `would_restore` outcomes without touching the hot registry.
5. **Byte-preserve.** Apply copies records verbatim into the archive subtree via atomic temp+rename. Restore copies archived bytes back via the same pattern. `submitted_at` / `updated_at` are preserved verbatim. Stage 13.3 stale-age views may show historical ages on restored records.
6. **Plan local, manifest portable.** The plan JSON includes `anchor_registry_dir` for local replay (drift recomputation). The archive manifest does NOT include host-local paths — it's a portable handoff artifact, parallel to Stage 13.5 export manifest.
7. **`registry_state_hash` includes `updated_at_unix`.** Deliberate divergence from Stage 13.4 because Stage 13.7's `--before <RFC3339>` selector reads `updated_at`; reconcile-driven timestamp bumps must trip drift.
8. **`tx_index.json` is merge-updated, not rebuilt** in both apply (remove archived tx_ids) and restore (add restored tx_ids). Unrelated entries preserved verbatim.

**Reason-tag taxonomy** (closed; surfaced on `event=integrity_evidence_anchor_archive_failed reason=<tag>` lines):

- `unsupported_archive_plan_schema_version` — plan's `schema_version` is not the locked `ANCHOR_ARCHIVE_PLAN_SCHEMA_VERSION = 1`. Refused FIRST. **New in Stage 13.7.**
- `archive_plan_hash_mismatch` — plan was hand-edited or corrupted. **New in Stage 13.7.**
- `archive_drift` — registry state changed since plan was generated. **New in Stage 13.7.**
- `archive_invalid_path` — manifest entry's `archive_relative` (or a plan action's `source_relative`) is absolute, contains `..`, contains backslash, or violates per-kind shape. **New in Stage 13.7.**
- `archive_blake3_mismatch` — archived bytes' BLAKE3 doesn't match the manifest's recorded `blake3_hex`. **New in Stage 13.7.**
- `archive_target_exists` — restore target conflict. `field=artifact_hash` (byte-different record at target) OR `field=tx_id` (tx_index maps the same tx_id to a different hash). **New in Stage 13.7.**
- `anchor_not_found` — selector miss. Operator-readable detail names the reason (`no such record`, `not terminal`, `excluded by --before`). Reused from Stage 13.0.
- `submitter_signature_invalid`, `unsupported_anchor_schema_version` — defense-in-depth refusals from Stage 13.0's `verify_anchor_tx_data`, re-run on every archived record at restore time. Reused.
- `malformed_json`, `io` — reused.

CLI mutex / required-with refusals exit non-zero via clap usage errors (NOT `reason=…` event lines).

**Workflow recipe — shrink the hot registry by archiving Finalized records older than 90 days:**

```sh
# 1. Plan with a date filter.
omni-node operator evidence-anchor archive-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors \
  --plan-out            /tmp/archive-plan.json \
  --status              finalized \
  --before              "$(date -u -v-90d +%Y-%m-%dT%H:%M:%SZ)"

# 2. Inspect the plan.
jq '.actions | length' /tmp/archive-plan.json
jq '.actions | map({status, artifact_hash_hex, tx_id})' /tmp/archive-plan.json

# 3. Dry-run the apply.
omni-node operator evidence-anchor archive-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors \
  --apply-plan          /tmp/archive-plan.json \
  --archive-dir         /var/omni-anchors-archive

# 4. Real apply.
omni-node operator evidence-anchor archive-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors \
  --apply-plan          /tmp/archive-plan.json \
  --archive-dir         /var/omni-anchors-archive \
  --apply

# 5. (Rare) Restore an archive subset later for re-investigation.
omni-node operator evidence-anchor archive-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors \
  --restore-manifest    /var/omni-anchors-archive/<plan_id>/archive_manifest.json \
  --apply
```

**Phase-2 partial-failure recovery recipe:**

```sh
# If a real apply fails mid-Phase-2 (e.g. disk fills, target FS
# error), the archive manifest IS durable but the hot registry
# may be partially mutated. Recover by running the same command
# in restore mode against the manifest the apply produced:
omni-node operator evidence-anchor archive-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors \
  --restore-manifest    /var/omni-anchors-archive/<plan_id>/archive_manifest.json \
  --apply

# Restore is idempotent: records Phase 2 didn't reach are
# byte-equal (row 2 → skipped_already_restored); records
# Phase 2 deleted are restored from the archive.
```

**Limitation — historical timestamps:** restored records carry the original `submitted_at` / `updated_at` from the source registry — exactly as in Stage 13.6 import. Stage 13.3 stale-age views may report historical ages by design.

**Limitation — terminal records only:** archiving `Submitted` / `Included` records would put records whose lifecycle is in flight beyond Stage 13.2 reconcile's reach. Explicitly disallowed at clap.

**Implementation reference:** [`docs/stage13.7-anchor-archive.md`](stage13.7-anchor-archive.md) — Stage 13.7 engineering doc (library surface, plan / manifest schemas, two-phase durability contract, restore conflict matrix, mutex rules, test inventory).

### Stage 13.8 — local integrity-evidence-anchor consistency report

**Use when:** you want a single read-only sweep across the hot registry, every Stage 13.7 archive subtree, and every Stage 13.5 export tree — at any operationally meaningful moment: after Stage 13.4 cleanup, after Stage 13.7 archive, before Stage 13.5 export, before Stage 13.9 chain reconcile (forward outlook), or as a periodic preflight stored to JSON for forensic retention. **Fully local — zero chain interaction.** Strictly read-only.

**Subcommand:**
- `omni-node operator evidence-anchor report-integrity-evidence-anchor-consistency`

**Key invariants:**

1. **Read-only.** The library uses only `std::fs::read` / `read_dir`. No registry, archive, export, or `tx_index.json` write helper is reachable from the consistency module.
2. **No new `reason=` tags.** Findings are typed report data ON the report, NOT refusals. The closed `EvidenceAnchorError` taxonomy is unchanged. The only `reason=` line the command can emit is `reason=io` when the required `--anchor-registry-dir` itself is unreadable at startup.
3. **Optional path unreadability is a finding, not a command failure.** Both `ArchiveDirUnreadable` and `ExportDirUnreadable` are `warning` severity. `ArchiveDirNoManifest` (warning) is distinct — it fires when the directory exists and is readable but contains no archive manifest.
4. **Severity principle (locked):** optional path can't be opened OR no manifest found → `warning`; readable AND claims to be an archive/export but fails integrity/schema checks → `error`.
5. **Cross-surface "expected duplicate" overlaps are summary counters, NOT per-item findings.** A 10k-record export overlapping the hot registry contributes at most 10k to `summary.hot_export_overlaps`, with **no per-item finding flood**. The `summary.archive_export_overlaps` counter behaves the same. Set semantics: each unique `artifact_hash_hex` contributes at most 1.
6. **Archive↔hot byte-equal duplicate IS emitted as a `warning` finding** (`ArchiveHotCollisionSameBytes`) because it indicates a likely partial-Phase-2 archive state — operationally distinct from the export "copy by design" case.
7. **Coarse export verification.** `verify_anchor_export(strict=false)` delegates per-export errors to Stage 13.5; a single `ExportVerifyFailed` finding carries the Stage 13.5 reason tag in `detail`.
8. **Bad `--json-out` parent is a CLI usage error (`bail!`)**, NOT a `reason=…` refusal.

**Finding kinds (24 closed):**

- **Hot (9):** `hot_record_malformed`, `hot_record_signature_invalid`, `hot_record_schema_unsupported`, `hot_filename_hash_mismatch`, `hot_tx_index_orphan`, `hot_tx_index_mismatch`, `hot_tx_id_duplicate`, `hot_stale_open_record`, `hot_tmp_orphan`.
- **Archive (13):** `archive_dir_unreadable`, `archive_dir_no_manifest`, `archive_manifest_malformed`, `archive_manifest_schema_unsupported`, `archive_entry_invalid_path`, `archive_entry_missing_file`, `archive_entry_blake3_mismatch`, `archive_entry_record_malformed`, `archive_entry_signature_invalid`, `archive_entry_metadata_mismatch`, `archive_hot_collision_same_bytes`, `archive_hot_collision_different_bytes`, `archive_tx_id_collision`.
- **Export (2):** `export_dir_unreadable`, `export_verify_failed`.

**Severity guide for operators:**

- `error` → stop and investigate before running mutation commands. Findings of this severity indicate real integrity problems that would cause downstream Stage 13.4 / 13.6 / 13.7 operations to refuse or behave incorrectly.
- `warning` → aware-of and decide. Includes optional-path-not-found, stale records, `.tmp` orphans, and tx_index orphans. Often resolved by running Stage 13.4 cleanup.
- `info` → currently unused in v1. Cross-surface "expected duplicate" overlaps live in summary counters instead.

**Workflow recipes:**

```sh
# 1. Periodic preflight; defaults to event-line output + summary.
omni-node operator evidence-anchor report-integrity-evidence-anchor-consistency \
  --anchor-registry-dir /var/omni-anchors

# 2. After Stage 13.4 cleanup, scan with archive + export surfaces too.
omni-node operator evidence-anchor report-integrity-evidence-anchor-consistency \
  --anchor-registry-dir /var/omni-anchors \
  --archive-dir         /var/omni-anchors-archive \
  --export-dir          /var/omni-exports/2026-Q1 \
  --export-dir          /var/omni-exports/2026-Q2

# 3. Include Stage 13.3-style stale findings.
omni-node operator evidence-anchor report-integrity-evidence-anchor-consistency \
  --anchor-registry-dir   /var/omni-anchors \
  --stale-threshold-secs  604800   # 1 week

# 4. Snapshot the report to JSON for forensic retention.
omni-node operator evidence-anchor report-integrity-evidence-anchor-consistency \
  --anchor-registry-dir /var/omni-anchors \
  --archive-dir         /var/omni-anchors-archive \
  --json-out            /var/omni-consistency-reports/2026-06-18.json

# 5. After Stage 13.7 archive apply, validate Phase-2 completeness.
#    Per the Stage 13.7 honest-durability contract, a Phase-2 IO
#    failure leaves the manifest durable but the hot registry
#    potentially partially mutated. A subsequent consistency
#    report surfaces this via ArchiveHotCollisionSameBytes
#    (warning) findings on records Phase 2 didn't reach.
omni-node operator evidence-anchor report-integrity-evidence-anchor-consistency \
  --anchor-registry-dir /var/omni-anchors \
  --archive-dir         /var/omni-anchors-archive
```

**Cross-surface overlap counter semantics:**

- `summary.hot_export_overlaps`: unique `artifact_hash_hex` count present in both hot registry and any export's `anchor_record` entries.
- `summary.archive_export_overlaps`: unique `artifact_hash_hex` count present in both any archive and any export.

Both use set semantics — duplicate manifest entries within a single export or archive do NOT inflate the counters.

**Limitation — no mutation.** Stage 13.8 reports problems; it does NOT repair them. `suggested_action` strings on findings point operators to the relevant mutation command (Stage 13.4 cleanup, Stage 13.7 restore, Stage 13.6 import, etc.).

**Limitation — no chain interaction.** Stage 13.8 cannot tell you whether the chain agrees with your local registry. Stage 13.9 (forward outlook) is the chain-facing reconcile / read stage.

**Implementation reference:** [`docs/stage13.8-anchor-consistency-report.md`](stage13.8-anchor-consistency-report.md) — Stage 13.8 engineering doc (library surface, finding taxonomy with all 24 closed kinds, severity matrix, cross-surface overlap counter semantics, test inventory).

### Stage 13.9 — SUM Chain read/reconcile integration

**Use when:** you want to bring local anchor statuses up to date against the SUM Chain. Stage 13.9 turns Stage 13.2's per-record reconcile into a batched flow against the chain's `sum_getIntegrityEvidenceAnchorStatusBatch` RPC (up to 100 tx_ids per call), and adds a new by-tuple lookup CLI for verifying a local record's canonical chain identity. **Read-only on the chain.** Reconcile applies the chain-returned status to local records per the locked transition table; the new by-tuple lookup is strictly read-only on the local registry.

**Subcommands:**
- `omni-node operator evidence-anchor reconcile-integrity-evidence-anchor` (existing) — now auto-batches via the trait's `query_anchor_status_batch` method when the chain client overrides it; falls back to per-record queries with fail-fast on transport errors when the client doesn't.
- `omni-node operator evidence-anchor watch-integrity-evidence-anchors` (existing) — gains batching transparently via reconcile.
- `omni-node operator evidence-anchor lookup-integrity-evidence-anchor-by-tuple` — **NEW.** Loads a local record via `--artifact-hash-hex` or `--tx-id`, extracts the locked 5-tuple from its `tx_data.digest`, and asks the chain for the canonical anchor. Read-only; useful for duplicate-anchor detection or after a chain prune event.

**Recommended preflight before any chain reconcile:** run Stage 13.8's `report-integrity-evidence-anchor-consistency`. Local-side problems (malformed records, tx_index drift, archive-hot collisions) should be resolved before talking to the chain.

**Key invariants:**

1. **Batch max 100.** The reconcile workflow chunks at `ANCHOR_STATUS_BATCH_MAX = 100` before calling the batch RPC. Oversize is never sent.
2. **Closed transition table.** Chain `Submitted` is **observation-only** for ALL local states (no reorg downgrade in 13.x). `Unknown` is observation-only. `Included` / `Finalized` / `Failed` apply forward-only from `Submitted` / `Included`; no downgrade overwrites of `Finalized` / `Failed`.
3. **Chunk-level failure fans out per-record.** A `ChainClientError` on a chunk (transport failure, malformed response shape) surfaces as ONE per-record `Err` entry for every tx_id in the chunk. Sweep continues with the next chunk.
4. **Per-item batch errors stay per-item.** When the chain echoes a per-item `error` field (e.g. operator typo'd a tx_id), the sibling records in the same chunk still succeed.
5. **By-tuple lookup is read-only on the local registry.** No `--apply` flag. The chain's canonical `tx_hash` and `included_at_height` are surfaced in event lines for operator inspection; mutation is out of scope for Stage 13.9.
6. **Raw-tuple flags are NOT exposed on the by-tuple CLI.** The tuple is always derived from a verified local record's digest. Operator-footgun guard.
7. **Read-path mapper.** JSON-RPC errors on read RPCs route to `chain_rpc`, NOT `chain_submit_refused`. The submit-path mapper is unchanged.
8. **By-tuple `null` outcome is informational.** Event line `event=integrity_evidence_anchor_tuple_lookup_no_chain_anchor ...` carries **no `reason=` key**; exit 0. Distinguishes "local found, chain says no" from `anchor_not_found` (local selector miss).
9. **`code` and `included_at_height` surface in event lines.** Stable on `failed`; opaque on other statuses. `AnchorStatus` enum unchanged.
10. **No live-chain tests.** All Stage 13.9 tests use `FakeJsonRpcTransport` + stub clients.

**Reason-tag taxonomy** — Stage 13.9 introduces **ZERO** new closed `reason=` tags:

- `chain_rpc` — per-item batch error; whole-chunk transport / fail-fast; JSON-RPC errors on read RPCs (NOT `chain_submit_refused`).
- `chain_response_malformed` — whole-chunk malformed response (shape / length / order mismatch).
- `io` — local FS / registry errors.
- By-tuple `null` outcome — **informational event line, no `reason=` key**.

CLI mutex / required-with refusals exit non-zero via clap usage errors (NOT `reason=…`).

**Workflow recipes:**

```sh
# 1. Periodic chain catch-up (now auto-batches against the real
#    chain). The Stage 13.2 chain-mode preflight + transitions
#    behavior is preserved; only the wire path changes.
omni-node operator evidence-anchor reconcile-integrity-evidence-anchor \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url             https://rpc.sumchain.io \
  --expect-chain-id     1

# 2. By-tuple cross-check (read-only). Operator picks a local
#    record; the command extracts the 5-tuple from that record's
#    tx_data.digest and asks the chain for the canonical anchor.
#    Useful when the operator suspects a duplicate-anchor race
#    (chain first-wins) or after a chain prune event.
omni-node operator evidence-anchor lookup-integrity-evidence-anchor-by-tuple \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url             https://rpc.sumchain.io \
  --expect-chain-id     1 \
  --artifact-hash-hex   <64-lower-hex>

# 3. Same, by tx_id (when the operator has the receipt but not
#    the hash).
omni-node operator evidence-anchor lookup-integrity-evidence-anchor-by-tuple \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url             https://rpc.sumchain.io \
  --expect-chain-id     1 \
  --tx-id               <chain-tx_id>
```

**Limitation — by-tuple lookup does NOT mutate local registry.** If the chain's canonical `tx_hash` differs from the local record's `receipt.tx_id`, Stage 13.9 reports the divergence via `tx_id_matches_canonical=false` on the event line. The operator decides what to do; Stage 13.9 itself does nothing.

**Limitation — no reorg-aware downgrade.** Chain `Submitted` on a record local-recorded as `Included` does NOT downgrade local. The 13.x track has no reorg model. If chain truly reverts an `Included` anchor, the operator's local record stays `Included` until a fresh terminal state arrives.

**Limitation — `tx_hash` status may become `unknown` if pruning is enabled in the future.** **By-tuple lookup is durable and never pruned.** If a local record's `tx_id` falls out of `getStatus` (returns `unknown`), run `lookup-integrity-evidence-anchor-by-tuple` to confirm the chain still has the anchor under the locked 5-tuple identity. This is the recommended long-term identity check.

**Failure-code 60–63 operator guide** (stable codes on `status == "failed"`):

| `code` | Meaning | Operator response |
| --- | --- | --- |
| `60` | not activated | Submit was rejected; chain hadn't activated anchor support. Re-submit later if applicable. |
| `61` | duplicate 5-tuple | First-wins. Local record may carry a non-canonical `tx_id`. Use by-tuple lookup to find the canonical one. |
| `62` | invalid submitter signature | Local record was tampered or submit was buggy. Investigate Stage 13.0 verifier output. |
| `63` | `tx.from != address(signer_pubkey)` | Submit-side configuration error — investigate the submit recipe. |
| other | parse `code` numerically; treat `reason` as opaque | — |

The `reason` field is **opaque** — log scrapers match on `code` for closed-set routing; `reason` is documentation, not a parsed token.

**Implementation reference:** [`docs/stage13.9-chain-read-reconcile.md`](stage13.9-chain-read-reconcile.md) — Stage 13.9 engineering doc (chain JSON-RPC wire shapes, locked transition table, read-path mapper, library surface, implementation locks, test inventory).

**Stage 13.x track closes here.** Stages 13.0–13.9 deliver the complete integrity-evidence-anchor lifecycle (submit → verify → reconcile → summary → cleanup → export → import → archive → consistency → chain read). Future evolution (operator UX, reorg model, multi-chain) is out of scope for the 13.x track.

**No protocol surface touched.** No envelope, no canonical-byte changes on Stage 12.0–12.25 surfaces, no `STATE_VERSION` / `STATE_INTEGRITY_REPORT_SCHEMA_VERSION` / `STATE_INTEGRITY_DIFF_SCHEMA_VERSION` / `SIGNED_BASELINE_SCHEMA_VERSION` / `SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION` / `INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION` / `SIGNED_INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION` / `INTEGRITY_EVIDENCE_CHAIN_REPORT_SCHEMA_VERSION` / `SIGNED_INTEGRITY_EVIDENCE_CHAIN_REPORT_SCHEMA_VERSION` bump, no SNIP / mesh / payment / proof / marketplace surface. Stage 13.0 ships a new `INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION = 1` constant on its own wire surface and a new `omni-zkml::evidence_anchor` module; Stage 13.5 ships `EVIDENCE_ANCHOR_EXPORT_MANIFEST_SCHEMA_VERSION = 1` on its own portable-manifest surface; Stage 13.6 reuses both verbatim and adds no new schema constants; Stage 13.7 ships `ANCHOR_ARCHIVE_PLAN_SCHEMA_VERSION = 1` and `ANCHOR_ARCHIVE_MANIFEST_SCHEMA_VERSION = 1` on its own plan + manifest surface; Stage 13.8 ships `ANCHOR_CONSISTENCY_REPORT_SCHEMA_VERSION = 1` on its own report surface; Stage 13.9 ships `ANCHOR_STATUS_BATCH_MAX = 100` and `FAILED_REASON_NULL_FALLBACK` as documentation constants alongside the new `AnchorStatusReport` DTO — **no new schema constants and no existing schema modified**.

### Stage 13.10 — operator acceptance: full lifecycle recipe + recovery playbook

**Use when:** you want the canonical end-to-end walkthrough that stitches every 13.x command together, plus the failure-mode playbooks for the situations the umbrella acceptance suite pins. Stage 13.10 ships **no new operator features**; it is an acceptance / hardening checkpoint for the 13.x track.

#### Full lifecycle recipe

The canonical happy path threads through stages 13.0 → 13.9. Each step's invariants are pinned by the umbrella suite at [`crates/omni-zkml/tests/evidence_anchor_stage_13_10_acceptance.rs`](../crates/omni-zkml/tests/evidence_anchor_stage_13_10_acceptance.rs).

```sh
# ── 0. Pre-flight (Stage 13.8): always run consistency first ────────────────
omni-node operator evidence-anchor report-integrity-evidence-anchor-consistency \
  --anchor-registry-dir /var/omni-anchors

# ── 1. Submit (Stage 13.0, requires --features submit) ─────────────────────
omni-node operator evidence-anchor submit-integrity-evidence-anchor \
  --signed-chain-report /var/signed/<report>.json \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url             https://rpc.sumchain.io \
  --expect-chain-id     1 \
  --allow-submit

# ── 2. Chain-state catch-up (Stage 13.9 batched reconcile) ─────────────────
omni-node operator evidence-anchor reconcile-integrity-evidence-anchor \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url             https://rpc.sumchain.io \
  --expect-chain-id     1

# ── 3. Inspect (Stage 13.3) ─────────────────────────────────────────────────
omni-node operator evidence-anchor summary-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors

# ── 4. Optional: keep an open record's identity in sync with the chain ─────
omni-node operator evidence-anchor lookup-integrity-evidence-anchor-by-tuple \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url             https://rpc.sumchain.io \
  --expect-chain-id     1 \
  --tx-id               <chain-tx_id>

# ── 5. Periodic cleanup of orphans / corrupted records (Stage 13.4) ────────
omni-node operator evidence-anchor cleanup-integrity-evidence-anchor-registry \
  --anchor-registry-dir /var/omni-anchors \
  --quarantine-dir      /var/omni-anchor-quarantine \
  --dry-run             # remove flag to apply

# ── 6. Portable evidence packaging (Stage 13.5 + 13.6) ─────────────────────
omni-node operator evidence-anchor export-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors \
  --export-out          /var/omni-anchor-export-2026-06

omni-node operator evidence-anchor verify-integrity-evidence-anchor-export \
  --export-dir          /var/omni-anchor-export-2026-06

omni-node operator evidence-anchor import-integrity-evidence-anchor-export \
  --export-dir          /var/omni-anchor-export-2026-06 \
  --anchor-registry-dir /var/omni-anchors-target

# ── 7. Cold archive of terminal records (Stage 13.7) ───────────────────────
omni-node operator evidence-anchor archive-integrity-evidence-anchors \
  --anchor-registry-dir /var/omni-anchors \
  --archive-dir         /var/omni-anchor-archive
```

The umbrella suite's scenario 1 (`scenario_1_happy_path_full_lifecycle_seed_reconcile_summary_export_import_archive_restore`) exercises steps 1–7 against `FakeJsonRpcTransport`. If you need to validate a change against the lifecycle without touching a real chain, that test is the canonical reproducer.

#### Recovery playbook

| Symptom | Diagnosis command | Recovery |
| --- | --- | --- |
| Hot registry contains a record file that won't parse | `report-integrity-evidence-anchor-consistency` flags it as `hot_record_malformed` (Error severity). | `cleanup-integrity-evidence-anchor-registry --dry-run` plans a `quarantine_malformed_record` action; rerun without `--dry-run` to apply. The quarantine manifest is durable — restore via `restore-…` if quarantine was wrong. Pinned by umbrella scenario 2. |
| `summary` reports a record stuck in `Submitted` past your expected confirmation window | `summary-…` + Stage 13.3 stale-list. `reconcile-…` with chain reporting `unknown` is **observation-only** and does NOT downgrade the local record. | If you suspect the chain truly never accepted the submit, use `lookup-integrity-evidence-anchor-by-tuple` to confirm; only the operator decides whether to mark the local record `Failed`. Pinned by umbrella scenario 3. |
| `tx_index.json` has an entry with no backing record file | `report-…-consistency` flags `hot_tx_index_orphan`. | `cleanup-…-registry` plans `remove_orphan_tx_index_entry`. Pinned by umbrella scenario 4. |
| A portable export bundle has been tampered with in transit | `verify-integrity-evidence-anchor-export` refuses with a Stage 13.5 reason tag. | **Do not import.** Re-export from the source registry, transfer again, re-verify. The import command re-runs verification internally, so a tampered bundle cannot accidentally be imported. Pinned by umbrella scenario 5. |
| Operator re-ran `import-…-export` against a target that already holds the bundle's records byte-equal | `import-…-export` either succeeds idempotently (no record bytes change) or refuses with `import_target_exists` whose closed-set `field` distinguishes byte-equal vs divergent collision. | Re-running a clean import is safe; the on-disk record bytes do not change. Investigate any `import_target_exists` refusal where `field=tx_id` or `field=artifact_hash` indicates divergent bytes. Pinned by umbrella scenario 6. |
| Archive apply failed mid-flight; some records gone from hot but the manifest is durable | `report-…-consistency` (with `--archive-dirs` pointing at the affected archive plan) cross-checks archive vs hot. | Run `restore-anchor-archive` on the archive manifest with the same `--anchor-registry-dir`. The restore is **idempotent** byte-equal: missing records are materialised from archive bytes, byte-equal records are left untouched (no rewrite), and tx_index entries are re-added. Pinned by umbrella scenario 7. |
| Want consistency over chain reconcile, but chain is unreachable | `report-…-consistency` is local-only — it makes **zero chain RPCs.** | Run consistency first, fix any orphans/malformed records, then run reconcile when the chain is reachable. Pinned by umbrella scenario 8. |
| By-tuple lookup reports `tx_id_matches_canonical=false` | The chain's canonical `tx_hash` differs from the local record's `receipt.tx_id` (cosmetic `0x`/case canonicalization, or genuine duplicate-anchor race). | **Stage 13.x does NOT auto-repair.** The local registry is not mutated by the lookup. Operator decides what to do (e.g. write a follow-up record manually, document the divergence). Pinned by umbrella scenario 9. |

#### Stage 13.10 acceptance reference

- Engineering doc: [`docs/stage13.10-acceptance.md`](stage13.10-acceptance.md) — operator-acceptance scope, scenario inventory, sign-off matrix.
- Umbrella test suite: [`crates/omni-zkml/tests/evidence_anchor_stage_13_10_acceptance.rs`](../crates/omni-zkml/tests/evidence_anchor_stage_13_10_acceptance.rs) — 12 hermetic tests covering the 9 scenarios above plus a `parse_event_line` self-test, a `_no_chain_anchor` event-line shape pin, and an error-taxonomy smoke.

**No new operator features.** Stage 13.10 changes no production code; the 13.x track closes as delivered through Stage 13.9.
