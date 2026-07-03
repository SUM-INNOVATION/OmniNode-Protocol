# OmniNode Protocol

[![license: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue)](LICENSE-MIT)
[![rust edition 2024](https://img.shields.io/badge/rust-2024-orange)](Cargo.toml)
[![toolchain 1.85](https://img.shields.io/badge/toolchain-1.85-orange)](rust-toolchain.toml)

**A Trustless, Peer-to-Peer AI Inference Network — Public Utility Infrastructure for AGI**

> Any device with a chip can become a node. Pool low-power devices into an omnipotent network.

---

## Executive Summary

OmniNode Protocol is a decentralized AI inference network that transforms
consumer hardware — MacBooks, PCs, mobile devices — into a unified compute
fabric capable of running massive Large Language Models (LLMs). Rather than
relying on centralized GPU clusters, OmniNode distributes model execution
across a peer-to-peer mesh, where each node contributes its memory and
compute to a collective inference pipeline.

The protocol is built on four pillars:

| Pillar | Mechanism | Outcome |
|---|---|---|
| **Compute** | Pipeline parallelism shards model layers across devices, routing hidden state tensors over a low-latency P2P mesh | Consumer devices pool their unified memory to run models that no single device could hold |
| **Storage** | Model weights (GGUF files) are chunked by transformer block, content-addressed (BLAKE3 → CIDv1), and distributed via a custom 64 MiB sliding-window protocol over libp2p request-response | No centralized model hosting. Weights are resilient, deduplicated, and globally available |
| **Privacy** | Federated Learning allows contributors to train locally on private data, uploading only mathematical weight gradients — never raw data | Data sovereignty is preserved. No central entity sees user data |
| **Incentives** | zkML (Zero-Knowledge Machine Learning) cryptographically proves correct inference. A Financial RLHF system stakes tokens, rewards quality, and slashes dishonest nodes | Trustless verification. Economic alignment between node operators and end users |

For the delivery record — every phase, every stage, what actually shipped —
see [`docs/project-status.md`](docs/project-status.md).

---

## Choosing your build (operator quick-reference)

OmniNode separates the **read-only operator surface** from the
**submit-capable surface** at the cargo-feature level. Both builds resolve
entirely from public sources (crates.io); no GitHub credential is required
at any step.

| You want to … | Build | Subcommands you get |
|---|---|---|
| Watch activation, query the chain, inspect your local registry, derive an address, run the lifecycle loop **monitor-only** | `cargo build -p omni-node` | `watch-activation`, `preflight`, `query` (incl. `--tx-hash`), `derive-address`, `registry list/show`, `loop` (monitor-only) |
| Submit attestations, run the lifecycle loop with retry | `cargo build -p omni-node --features submit` | all of the above **plus** `smoke` and `loop --allow-submit [--allow-mainnet-submit]` |

Default builds keep `sumchain-primitives` / `sumchain-crypto` out of the
compile graph entirely (the operator binary stays small and the submit code
path is an explicit operator choice). CI gates enforce both halves of this
on every push / PR. See [`docs/operator-runbook.md`](docs/operator-runbook.md)
for the full operator runbook.

---

## Production MLP Proofs

Off-chain zero-knowledge proof of a deterministic `16 → 32 → 16 → 8` int16
fixed-point MLP inference against the canonical
`production-fixedpoint-mlp-v1` spec. Distinct from the bounded `4 → 8 → 4`
`halo2-mlp-v1` reference (which is testnet/dev-only in perpetuity).

Cargo features:

| Feature | Gives you |
| --- | --- |
| `stage11d-production-verify` | `operator verify-proof` for production artifacts |
| `stage11d-production-prove` (superset) | above + `operator generate-production-mlp-proof` + `contributor run-job --emit-production-mlp-proof` (Stub + External runners) |

Default builds pull zero halo2 / `omni-proofs-halo2-*` / `rand_chacha` —
CI's tree gates enforce it. Under prover invocation, set
`RUST_MIN_STACK=67108864` (constraint-system walker needs headroom).

**Mainnet posture — dormant.** `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]`
by design. `check_mainnet_eligible` refuses every production artifact at
**layer 6** (empty registry). Lifting the refusal requires a chain-side
Proof Eligibility Registry record for `Stage11dProductionFixedPointMlp` —
currently blocked on chain-team-dependent work. Production proofs today
are safe for staging / testnet / evidence-gathering; **not** a mainnet
attestation-eligibility signal.

Full operator + integrator guide (build/use, math contract, pinned
constants, drift-detection, dependency posture, cross-references):
[`docs/production-mlp-proof.md`](docs/production-mlp-proof.md).

---

## Architecture Overview

Data and artifact movement across the five OmniNode workflows. The
diagram intentionally shows a **dormant boundary** between off-chain
verification and mainnet eligibility — production MLP proofs verify
off-chain today but do **not** cross into the SUM Chain attestation
path because the eligibility registry is empty by design.

```text
1) MODEL & STORAGE
   GGUF model
       │  chunk + BLAKE3
       ▼
   manifest (CIDv1) ──► SNIP content roots ──► omni-net peer fetch / serve


2) CONTRIBUTOR INFERENCE
   ContributorJob JSON
       │
       ▼
   operator contributor run-job  |  run-assignment
       │       ├── --runner stub       (StubRunner)
       │       └── --runner external   (ExternalCommandRunner)
       ▼
   ContributorResult JSON   (signed; local operator provenance)


3) PRODUCTION MLP PROOF                features: stage11d-production-{verify,prove}
   canonical spec: production-fixedpoint-mlp-v1
   input:  16 × i16
   output:  8 × i16       (derived by the canonical evaluator; you do not supply)
       │
       ▼
   Halo2 production prover
       (halo2_proofs 0.3.2 · HALO2_K = 11 · PROVER_RNG_SEED pinned)
       │
       ▼
   ProofArtifactBody JSON
       │
       ▼
   operator verify-proof
       │
       ▼
   verified          = true                      ← OFF-CHAIN VERIFY OK
   mainnet_eligible  = false                     ← REFUSED AT LAYER 6
   mainnet_refusal   = NotInMainnetAllowlist { … }

     ╳╳╳ DORMANT BOUNDARY ╳╳╳ no path from production-MLP artifacts to mainnet
     ╳╳╳                  ╳╳╳ MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]
     ═══════════════════════════════════════════════════════════════════════════
     (lifting the boundary requires a chain-side Proof Eligibility Registry
      record for Stage11dProductionFixedPointMlp — chain-team-dependent,
      currently blocked)


4) SUM CHAIN                                                    feature: submit
   eligible InferenceAttestation
       │  (today: NOT sourced from any Stage11dProductionFixedPointMlp artifact)
       ▼
   operator smoke  |  operator loop --allow-submit [--allow-mainnet-submit]
       │
       ▼
   SUM Chain   (submit / query / receipt / registry read)


5) FEATURE-GATE GUARDRAILS
   default                          read + query + operator + contributor
                                    NO submit
                                    NO halo2 / omni-proofs-halo2-* / rand_chacha
                                    (CI default-tree gate enforces)

   +submit                          adds SUM Chain submit path
                                    (sumchain-crypto / sumchain-primitives)

   +stage11d-production-verify      adds Halo2 production-MLP verifier
                                    (halo2_proofs + omni-proofs-halo2-production-mlp)

   +stage11d-production-prove       superset of verify; adds prover
                                    (+ rand_chacha)
                                    runtime: RUST_MIN_STACK=67108864
```

For the detailed workspace crate map, dependency inventory, per-phase
architecture, and inter-phase dependency graph, see
[`docs/project-status.md`](docs/project-status.md).

---

## Quick Start

```bash
# Prerequisites: Rust 1.85+ (edition 2024)
git clone https://github.com/SUM-INNOVATION/OmniNode-Protocol.git
cd OmniNode-Protocol
cargo build --workspace

# Run all tests
cargo test --workspace
```

### CLI commands

```bash
# Listen for mesh events and serve shard requests
RUST_LOG=info cargo run --bin omni-node -- listen

# Ingest a GGUF model: parse, chunk, store shards, announce on mesh
RUST_LOG=info cargo run --bin omni-node -- shard path/to/model.gguf

# Fetch a shard by CID from a LAN peer
RUST_LOG=info cargo run --bin omni-node -- fetch <cid>

# Send a test Gossipsub message
RUST_LOG=info cargo run --bin omni-node -- send "Hello from OmniNode"
```

### Two-Mac LAN demo

```bash
# Mac 1 — ingest model and serve shards
RUST_LOG=info cargo run --bin omni-node -- shard tinyllama.gguf

# Mac 2 — fetch a shard by CID (from Mac 1's manifest output)
RUST_LOG=info cargo run --bin omni-node -- fetch bafkr4i...
```

### End-to-end pipeline inference demo

`showcase_tui.py` ties all four phases together in a single runnable
script. Each machine loads only its assigned layer slice from a local GGUF
file.

```bash
# Prerequisites: maturin develop (builds omninode Python extension)
pip install mlx mlx-lm transformers rich

# Machine A — Sender (embed_tokens + first ½ of layers)
python showcase_tui.py sender /path/to/model.gguf

# Machine B — Receiver (second ½ of layers + norm + lm_head)
python showcase_tui.py receiver /path/to/model.gguf
```

Startup output (example with `tinyllama.gguf`):

```
[OmniStore] model_name  : TinyLlama
[OmniStore] model_hash  : a3f8c2...
[OmniStore] architecture: llama
[OmniStore] total_layers: 22
[OmniStore] shards      : 2
[OmniStore]   shard 0  layers 0-10   cid=bafkr4iabc123…  215 MB  [embedding]
[OmniStore]   shard 1  layers 11-21  cid=bafkr4ixyz789…  209 MB  [output_head]

[GGUF] arch=llama  hidden=2048  layers=22  heads=32/4  vocab=32000
[GGUF] Split: Sender layers 0-10 | Receiver layers 11-21
[GGUF] 198 tensors injected into bare Model()
[GGUF] RAM pool drop complete.
```

---

## Documentation

Deeper docs live in [`docs/`](docs/). Common entry points:

| Doc | Use it when … |
| --- | --- |
| [`operator-runbook.md`](docs/operator-runbook.md) | You are running `omni-node` in production and need the complete operator surface: build modes, subcommands, observability, release-readiness checklist. |
| [`production-mlp-proof.md`](docs/production-mlp-proof.md) | You need the operator + integrator guide for the Halo2 production MLP proof: feature flags, math contract, pinned constants, verifier drift-detection, dependency posture. |
| [`project-status.md`](docs/project-status.md) | You want the durable delivery record — every phase, every stage, what actually shipped. Formerly the bulk of this README. |
| [`stage14.8-proof-generation-readiness.md`](docs/stage14.8-proof-generation-readiness.md) | You need the closure and readiness packet for the Stage 14 proof-generation track (reference + production paths, StubRunner + ExternalCommandRunner sidecars, acceptance hardening). |
| [`stage15.2-release-artifact-workflow.md`](docs/stage15.2-release-artifact-workflow.md) | You are verifying a signed OmniNode release. Covers the cosign-keyless release workflow, the pinned production cert identity regex, and the operator verification recipe. |
| [`stage15.3-release-bundle-aarch64.md`](docs/stage15.3-release-bundle-aarch64.md) | You need the aarch64 release expansion detail: multi-arch matrix, per-arch ELF check, verification recipe on aarch64 hosts. |
| [`mainnet-eligibility-criteria.md`](docs/mainnet-eligibility-criteria.md) | You are evaluating what a proof system must satisfy before its eligibility registry entry may be added. Chain-team-facing canonical criteria. |
| [`stage11.d.3A-production-proof-eligibility-evidence.md`](docs/stage11.d.3A-production-proof-eligibility-evidence.md) | You are the chain team reviewing the production proof class candidate. Evidence bundle for eligibility review. |
| [`phase5-rc-audit-2026-06-24.md`](docs/phase5-rc-audit-2026-06-24.md) | You want the immutable Phase 5 release-candidate audit snapshot at 2026-06-24. Point-in-time record; do not edit. |

---

## License

Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE).

---

*OmniNode Protocol is open-source infrastructure for decentralized AGI. Built from the bottom up.*
