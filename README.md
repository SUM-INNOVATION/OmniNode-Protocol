# OmniNode Protocol

[![license: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue)](LICENSE-MIT) [![rust edition 2024](https://img.shields.io/badge/rust-2024-orange)](Cargo.toml) [![toolchain 1.85](https://img.shields.io/badge/toolchain-1.85-orange)](rust-toolchain.toml) [![SNIP V2 gated](https://img.shields.io/badge/SNIP-V2_gated-brightgreen)](#phase-5-stage-1-snip-v2-storage-integration--complete) [![SUM Chain InferenceAttestation v1 (dormant)](https://img.shields.io/badge/SUM_Chain-InferenceAttestation_v1_%28dormant%29-blueviolet)](#phase-5-stage-6-chain-wire-fixture--signing-spec-deliverables--complete)

**A Trustless, Peer-to-Peer AI Inference Network — Public Utility Infrastructure for AGI**

> Any device with a chip can become a node. Pool low-power devices into an omnipotent network.

---

## Executive Summary

OmniNode Protocol is a decentralized AI inference network that transforms consumer hardware — MacBooks, PCs, mobile devices — into a unified compute fabric capable of running massive Large Language Models (LLMs). Rather than relying on centralized GPU clusters, OmniNode distributes model execution across a peer-to-peer mesh, where each node contributes its memory and compute to a collective inference pipeline.

The protocol is built on four pillars:

| Pillar | Mechanism | Outcome |
|---|---|---|
| **Compute** | Pipeline parallelism shards model layers across devices, routing hidden state tensors over a low-latency P2P mesh | Consumer devices pool their unified memory to run models that no single device could hold |
| **Storage** | Model weights (GGUF files) are chunked by transformer block, content-addressed (BLAKE3 → CIDv1), and distributed via a custom 64 MiB sliding-window protocol over libp2p request-response | No centralized model hosting. Weights are resilient, deduplicated, and globally available |
| **Privacy** | Federated Learning allows contributors to train locally on private data, uploading only mathematical weight gradients — never raw data | Data sovereignty is preserved. No central entity sees user data |
| **Incentives** | zkML (Zero-Knowledge Machine Learning) cryptographically proves correct inference. A Financial RLHF system stakes tokens, rewards quality, and slashes dishonest nodes | Trustless verification. Economic alignment between node operators and end users |

---

## Choosing your build (operator quick-reference)

OmniNode separates the **read-only operator surface** from the
**submit-capable surface** at the cargo-feature level. Both builds
resolve entirely from public sources (crates.io); no GitHub
credential is required at any step.

| You want to … | Build | Subcommands you get |
|---|---|---|
| Watch activation, query the chain, inspect your local registry, derive an address, run the lifecycle loop **monitor-only** | `cargo build -p omni-node` | `watch-activation`, `preflight`, `query` (incl. `--tx-hash`), `derive-address`, `registry list/show`, `loop` (monitor-only) |
| Submit attestations, run the lifecycle loop with retry | `cargo build -p omni-node --features submit` | all of the above **plus** `smoke` and `loop --allow-submit [--allow-mainnet-submit]` |

Default builds keep `sumchain-primitives` / `sumchain-crypto` out of
the compile graph entirely (the operator binary stays small and the
submit code path is an explicit operator choice). Stage 9c's CI gates
enforce both halves of this on every push / PR. See
[`docs/operator-runbook.md`](docs/operator-runbook.md) for the full
operator runbook.

---

## Architecture Overview

```
                              OmniNode Protocol
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
    │  │  omni-node   │    │  omni-zkml   │    │  contracts   │  │
    │  │  (binary)    │    │  (proofs)    │    │  (SUM Chain) │  │
    │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
    │         │                   │                   │          │
    │  ┌──────┴───────────────────┴───────────────────┘          │
    │  │                                                         │
    │  │  ┌──────────────┐    ┌──────────────┐                   │
    │  │  │ omni-pipeline │    │ omni-bridge   │                   │
    │  │  │ (parallelism) │    │ (Rust↔Python) │                   │
    │  │  └──────┬────────┘    └──────┬────────┘                   │
    │  │         │                    │                            │
    │  │  ┌──────┴────────────────────┘                            │
    │  │  │                                                        │
    │  │  │  ┌──────────────┐    ┌──────────────┐                  │
    │  │  │  │  omni-store   │    │  omni-net     │                  │
    │  │  │  │  (sharding)   │    │  (libp2p)     │                  │
    │  │  │  └──────┬────────┘    └──────┬────────┘                  │
    │  │  │         │                    │                            │
    │  │  │  ┌──────┴────────────────────┘                            │
    │  │  │  │                                                        │
    │  │  │  │  ┌──────────────┐                                      │
    │  │  │  │  │  omni-types   │                                      │
    │  │  │  │  │  (shared)     │                                      │
    │  │  │  │  └───────────────┘                                      │
    │  └──┴──┴─────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌───────────────────┐                                           │
    │  │  python/omninode   │  (ML inference backends: llama.cpp, MLX) │
    │  └───────────────────┘                                           │
    └──────────────────────────────────────────────────────────────────┘
```

---

## Current Status

| Phase | Crate | Status |
|---|---|---|
| **Phase 1** — P2P Mesh Networking | `omni-net` | **Complete** |
| **Phase 2** — GGUF Model Sharding | `omni-store` | **Complete** |
| **Phase 3** — FFI Bridge & Local Inference | `omni-bridge` | **Complete** |
| **Phase 4** — Pipeline Parallelism | `omni-pipeline` | **Complete** |
| **Phase 5 Stage 1** — SNIP V2 types + CLI adapter | `omni-types`, `omni-store` | **Complete** |
| **Phase 5 Stage 2** — SNIP-backed model artifacts (publish / restore) | `omni-store` | **Complete** |
| **Phase 5 Stage 3** — Proof artifact flow (publish + commitment) | `omni-zkml` | **Complete** |
| **Phase 5 Stage 4** — Local verifier attestation envelope (canonical bytes + digest + Signer trait) | `omni-zkml` | **Complete** |
| **Phase 5 Stage 5** — Chain client abstraction + offline attestation registry | `omni-zkml` | **Complete** |
| **Phase 5 Stage 5.2** — Client-local staleness / retry policy (`StalenessPolicy`, `mark_stale_if_overdue`, `submitted_at_block`) | `omni-zkml` | **Complete** |
| **Phase 5 Stage 5.3** — End-to-end attestation orchestration (`OrchestrationClient` supertrait + `submit_attestation_workflow_with_block` + sweep / retry helpers) | `omni-zkml` | **Complete** |
| **Phase 5 Stage 6** — Chain wire fixture & signing-spec deliverables (Ed25519 + bs58 checksum address) | `omni-zkml` | **Complete** |
| **Phase 5 Stage 7a** — SUM Chain adapter (read/query against JSON-RPC; typed stub for submit) | `omni-sumchain` | **Complete** |
| **Phase 5 Stage 7b** — SUM Chain submit path (outer `SignedTransaction` via vendored chain primitives) | `omni-sumchain` | **Complete** |
| **Phase 5 Stage 8a** — Operator surface: activation watch / smoke / lifecycle loop (safe-by-default CLI) | `omni-node` | **Complete** |
| **Phase 5 Stage 8b** — Mainnet activation smoke hardening: read-only `preflight` + `query` + checklist runbook | `omni-node` | **Complete** |
| **Phase 5 Stage 8c** — Post-mainnet-smoke hardening: warning-clean build, `smoke` summary, `query --tx-hash`, `derive-address` | `omni-node` | **Complete** |
| **Phase 5 Stage 9a** — Production runbook + build decoupling (`submit` feature; default builds need no private repo); `operator registry list/show` | `omni-node`, `omni-sumchain` | **Complete** |
| **Phase 5 Stage 9b** — CI workflow + secret-gated submit matrix (superseded by 9c; original gates restored against the public dep source) | `.github/workflows`, docs | **Superseded by 9c** |
| **Phase 5 Stage 9c** — Public SUM Chain crates (`sumchain-primitives` / `sumchain-crypto` v0.1.0, crates.io, MIT OR Apache-2.0); cargo-side CI gates restored; fresh source builds need no private repo access | `Cargo.toml`, `.github/workflows`, docs | **Complete** |
| **Phase 5 Stage 9c.1** — Chain-produced signed-transaction fixture for the 2026-05-19 mainnet smoke tx; byte-roundtrip + `SignedTransaction::hash()` gate against authoritative on-chain bytes | `omni-sumchain/tests` | **Complete** |
| **Phase 5 Stage 10a** — Operator observability + release readiness: `operator registry summary`, stable `event=` log markers, failure-triage matrix, release-readiness checklist (docs-only) | `omni-node`, `docs/operator-runbook.md` | **Complete** |
| **Phase 5 Stage 11a** — Real zkML proof generation integration (trait + hermetic fixture harness): `ProofBackend` / `ProofVerifier` traits, `MockProofBackend` (mainnet-refused), `ProofMetadata` committed inside SNIP V2 proof artifact, end-to-end fixture pipeline, `build_mock_v1_attestation` replaces placeholder synthetic path | `omni-zkml`, `omni-node`, `.github/workflows` | **Complete** |
| Phase 5 Stage 10b — Release artifact workflow + signing (deferred from 10a) | `.github/workflows`, `docs/` | Planned |
| **Phase 5 Stage 11b.0** — Decentralized proof architecture: multi-backend trait extensions, `ModelFormat` / `ProofSystem` enums, expanded `ProofMetadata` schema (6 new optional fields, serde-skipped — Stage 11a fixture byte-stable), six-layer `check_mainnet_eligible` refusal helper (allowlist empty by design), read-only `operator verify-proof` subcommand. **Zero mainnet eligibility, zero new prover deps, zero chain-wire changes, no zkML claim.** | `omni-zkml`, `omni-node`, `docs/operator-runbook.md` | **Complete** |
| **Phase 5 Stage 11b.0.1** — Architecture correction: `ProofVerifier::verify_artifact(&ProofArtifactBody)` defaulted method as the single dispatch entry point; `operator verify-proof` refactored to use it. Zero new backends, zero prover deps, zero schema changes. | `omni-zkml`, `omni-node` | **Complete** |
| **Phase 5 Stage 11b.1.a** — Bounded multi-framework halo2 reference scaffold (no halo2 circuit yet, four equal-status primaries): `ModelFramework` enum (Rumus / PyTorch / TensorFlow / Caffe / FrameworkAgnostic), `ModelFormat::Halo2ReferenceMlp`, `ProofSystem::Stage11bHalo2Reference`, **framework-neutral** canonical MLP spec (4→8→4 with ReLU, i16 fixed-point, **i64 accumulator + round-half-away-from-zero + bias-before-saturation**), pure-Rust canonical evaluator (neutral reference implementation), `FrameworkManifest` schema with `spec_hash` / `input_hash` / `output_hash`, 5 committed framework manifests, cross-framework equivalence + hash-equivalence integration tests. **Four equal-status primary exporters** under `tools/` (excluded from the workspace): `rumus_export` (standalone Cargo pkg, `rumus = "0.4.0"` `fixed::FixedLinear`), `pytorch_export` (Python, `torch.int64` explicit), `tensorflow_export` (Python, `tf.int64` explicit), `caffe_export` (Python, real Caffe or auditable pure-NumPy fallback). **Zero prover deps, zero halo2 deps, zero operator-binary changes, zero framework runtimes in the OmniNode workspace, Stage 11a + Stage 11b.0 fixtures byte-stable.** | `omni-zkml`, `crates/omni-proofs-halo2-reference/` (new), `tools/<framework>_export/` (new, outside workspace) | **Complete** |
| **Phase 5 Stage 11b.1.b** — Halo2 circuit + `Halo2ReferenceVerifier` overriding `verify_artifact` + `verify()` returning `ProofVerifierError::RequiresArtifactDispatch` + opt-in `omni-node halo2-reference-verify` feature + frozen `params.bin`/`proof.bin`/`proof_artifact.json` fixtures + standalone `tools/halo2_reference_regen/` (workspace-excluded) + verifier-only CI gate. Default `omni-node` build pulls zero halo2 deps (verified by tree-check). All artifacts `testnet_or_dev_only: Some(true)`; `MAINNET_APPROVED_PROOF_SYSTEMS` stays empty. | `omni-zkml`, `omni-proofs-halo2-reference/` (extends), `omni-node`, `tools/halo2_reference_regen/`, `.github/workflows` | **Complete** |
| **Phase 5 Stage 11c** — Arbitrary-input soundness for the bounded `halo2-mlp-v1 / spec_version: 2` numeric contract: complete round-half-away-from-zero (RHAZ) gadget via signed-magnitude Euclidean division (no tie ambiguity) + three-branch saturation gadget (`b_lo` / `b_in` / `b_hi` with branch-correctness aux witnesses) + range checks at widths 8u/15u/16s/16u/17u/23u + 8-entry committed test corpus (`corpus.json`) exercising tie cases + cross-framework corpus equivalence across rumus/pytorch/tensorflow/caffe/framework_agnostic + gadget-isolated `SatTestCircuit` unit tests for boundary cases (real i16 inputs never saturate under the frozen spec; isolated tests exercise the b_lo/b_hi branches). `HALO2_K` bumped 9 → 10 (preauthorized ≤ 11; halt-and-report at ≥ 12). All four framework exporters extended with `--corpus` modes (developer-host only). Verifier-side `canonical_evaluate` defense in depth preserved. **Still not production zkML; still `testnet_or_dev_only: Some(true)`; `MAINNET_APPROVED_PROOF_SYSTEMS` stays empty.** | `omni-proofs-halo2-reference/` (extends), `tools/halo2_reference_regen/`, `tools/{rumus,pytorch,tensorflow,caffe}_export/`, `.github/workflows` | **Complete** |
| **Phase 5 Stage 11d.0** — Mainnet eligibility criteria + chain-team review packet (docs-only). Authoritative criteria document defining what qualifies a proof system for `MAINNET_APPROVED_PROOF_SYSTEMS`: required proof properties (soundness, completeness, determinism, verifier availability), required `ProofArtifactBody` metadata, distinguishability rules vs bounded reference (`Stage11bHalo2Reference` stays testnet/dev-only in perpetuity), allowlist mechanics design (Stage 11d.1 schema: `(proof_system, circuit_id_hex, model_hash)` triple keys), and the eight required chain-team review packet artifacts (R1–R8). Plus the structured review-packet template and an operator-facing FAQ. **No code, no schema, no allowlist entry.** Stage 11d defines and prepares the eligibility path; a later Stage 11d.3 entry may add the first eligible proof system after written sign-off. | `docs/mainnet-eligibility-criteria.md`, `docs/stage11d-review-packet.md`, `docs/stage11d-mainnet-eligibility-FAQ.md`, `README.md`, `docs/operator-runbook.md` | **Complete** |
| **Phase 5 Stage 11d.1** — Structured allowlist schema: `AllowlistEntry { proof_system, backend_id, circuit_id_hex, model_hash, model_format, verification_key_hash_hex, chain_team_review_ref }` + `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES: &[AllowlistEntry] = &[]` + `MAINNET_VK_HASH_DOMAIN_SEPARATOR = b"OMNINODE-VK:v1:"` + `mainnet_vk_hash(canonical_vk_bytes)` helper computing `BLAKE3(domain_separator \|\| canonical_vk_bytes)`. Layer 6 of `check_mainnet_eligible` rewired to consult the structured triple-key match (proof_system + circuit_id_hex + model_hash) with the legacy `MAINNET_APPROVED_PROOF_SYSTEMS: &[ProofSystem] = &[]` preserved as an empty back-compat alias. 12 new hardening tests including `stage11b_halo2_reference_never_in_allowlist`, `bounded_reference_refused_before_allowlist_lookup`, `testnet_or_dev_only_refused_before_allowlist_lookup`, `every_allowlist_entry_has_required_metadata` (forward-looking guard rejecting `Mock` / bounded-reference / `Gguf` / `Other(_)` entries), `wrong_circuit_id_hex_rejects`, `wrong_model_hash_rejects`, `vk_hash_helper_is_byte_stable`. Chain-team review tightenings folded into criteria / review-packet / FAQ / operator-runbook docs (board roster ref, benchmark requirements, deterministic byte-stability mandate, full failure-mode table covering all six `MainnetRefusalReason` layers, framework-binding location, `Gguf`/`Other(_)` invalid for 11d.3). **Allowlist stays empty.** | `crates/omni-zkml/src/proof.rs`, `crates/omni-zkml/src/lib.rs`, docs | **Complete** |
| **Phase 5 Stage 11d.2** — First production-grade proof class: a `16 → 32 → 16 → 8` int16 fixed-point MLP built on Stage 11c's RHAZ + saturation + ReLU gadget chain, in a new `omni-proofs-halo2-production-mlp` crate. `ProofSystem::Stage11dProductionFixedPointMlp` + `ModelFormat::ProductionFixedPointMlp` (off-chain proof metadata only) + `Halo2ProductionMlpVerifier` + standalone `tools/halo2_production_mlp_regen/` (workspace-excluded) + frozen `params.bin` / `proof.bin` / `proof_artifact.json` fixtures + 5 framework corpora (rumus / pytorch / tensorflow / caffe / framework_agnostic) + cross-framework equivalence test + opt-in `omni-node` `stage11d-production-verify` feature + verifier-only CI gates + isolation tree-checks + opt-in R9 chain-digest+signature roundtrip scaffold (`#[ignore]`, gated on `r9-chain-digest-roundtrip` feature). Verifier p95 ≈ 19.6 ms (≪ 1000 ms target); prover ≈ 2.3 s release (≪ 60 s target); `params.bin` 131 KB. **No chain wire / Stage 7b tx / `InferenceAttestationDigest` / RPC / validator-side verification changes. `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]` stays empty; every artifact under this proof class is refused by `check_mainnet_eligible` layer 6.** Mainnet allowlist entry remains a Stage 11d.3 deliverable (R4 reproducer + external cryptographer sign-off + full R9 chain-team sign-off all gated to 11d.3). | `crates/omni-proofs-halo2-production-mlp/` (new), `omni-zkml`, `omni-node` feature, `tools/halo2_production_mlp_regen/` (new, outside workspace), `.github/workflows`, `docs/stage11d.2-review-packet-production-fixedpoint-mlp.md`, `docs/stage11d.2-benchmark-record.md` | **Complete** |
| Phase 5 Stage 11d.3 — First allowlist entry. **Blocked on** external-cryptographer sign-off (Claim 1.1.S2 in the criteria doc) + completed review packet (R1–R8). Adds exactly one `AllowlistEntry`, commits the `mainnet_allowlist_snapshot.json` byte-stability fixture, references the signed chain-team-review document. | `crates/omni-zkml/src/proof.rs`, `docs/chain-team-review/…` (new), tests, CI | Planned (blocked on sign-off) |
| **Phase 5 Stage 12.0** — Contributor Inference Node workflow (numbered chart items 2–6, **AttestationOnly** evidence mode only). New `omni-contributor` crate with `ContributorJob` + `ContributorResult` schemas (forward-compatible base-unit accounting placeholders: `tokenizer_hash`, `input_token_count`, `max_output_token_count`, `total_base_units`, per-stage `work_unit_kind`/`work_units` for future B/C/D split). `InferenceRunner` trait + `StubRunner` + `ExternalCommandRunner` (file-path-based, `deny_unknown_fields` JSON stdout envelope; no framework runtime in the operator-binary graph). SNIP byte-publish/fetch wrappers (tempfile-backed, Option A — no `omni-store` lift). Direct Ed25519 signing via `libp2p-identity` inside the contributor crate (no `omni-zkml` dep). New `omni-node operator contributor` subcommand tree: `validate-job` / `run-job` / `verify-result`. Off-chain verification only. **No chain wire / Stage 7b tx / SUM Chain RPC / `InferenceAttestationDigest` change. No on-chain proof verification, no proof allowlist. No GGUF correctness-proof claim. No payment / reward distribution / slashing — those are Stage 12.x+.** Default `omni-node` build still pulls zero halo2/prover/framework runtimes (CI default-tree-check re-asserts). 26 tests pass. Production-proof evidence mode is a reserved future name documented in `docs/stage12-contributor-protocol.md` and is NOT present in any Rust enum at 12.0 — adding it is a `schema_version: 2` migration. | `crates/omni-contributor/` (new), `crates/omni-node/` (operator subcommand wiring), `.github/workflows`, `docs/stage12-contributor-protocol.md` | **Complete** |
| **Phase 5 Stage 12.1** — Filesystem-based contributor job discovery (extends 12.0). Two new schemas (`PostedJob`, `PostedResultLink`) with frozen `v1` canonical bytes under new domain separators `OMNINODE-CONTRIBUTOR-POSTED-JOB:v1:` and `OMNINODE-CONTRIBUTOR-POSTED-RESULT-LINK:v1:`. New `JobSource` trait + `FilesystemSource` impl (directory watch, dedup-by-`posted_id`, no persistent state). Three new `omni-node operator contributor` subcommands: `post-job` (publish `ContributorJob` to SNIP + write `PostedJob`), `watch-jobs --source fs` (long-running pickup loop), `publish-result-link` (publish `ContributorResult` + signed `PostedResultLink`). **Required cost guardrails** on `watch-jobs`: `--max-input-tokens`, `--max-output-tokens`, `--max-total-base-units` — no defaults; operator must explicitly cap workload before any pickup. Post-run `verify_result` check before writing accepted output; verification failure writes `<result-out-dir>/<job_id>.rejected.json` audit trail and skips link-publish. SNIP roots are content-addressed and immutable, so SNIP-based index polling is **deliberately not implemented** in 12.1 — deferred to Stage 12.2+ when a real mutable discovery surface is designed (libp2p gossip / local head pointer / off-chain registry). **No chain wire change, no payment, no leases/claims, no double-pickup prevention, no `PostedJobsIndex` aggregator, no proof mode, no A–F flow.** Default `omni-node` build still pulls zero halo2/prover/framework runtimes. 58 tests pass (6 posted byte-stability + 11 posted negatives + 8 watch end-to-end + Stage 12.0's 33). | `crates/omni-contributor/` (extends), `crates/omni-node/` (subcommand wiring), `docs/stage12-contributor-protocol.md` | **Complete** |
| **Phase 5 Stage 12.2-pre** — `omni-net` topic safety lift (prerequisite for Stage 12.2 contributor mesh). Adds `TOPIC_CONTRIBUTOR_JOB` + `TOPIC_CONTRIBUTOR_RESULT` as first-class gossipsub topics. **Fixes a real footgun**: `GossipManager::topic_by_name` previously silently routed any unknown topic to `TOPIC_TEST` — production-shape messages could land on the test topic. Post-lift, unknown topics fail loudly with a typed `UnknownTopic` error. Adds `OmniNet::try_next_event` (non-blocking event drain via `tokio::mpsc::Receiver::try_recv`) so synchronous consumers can poll without blocking. Two files changed, fully additive (no existing valid topic behavior changes). | `crates/omni-net/src/gossip.rs`, `crates/omni-net/src/lib.rs` | **Complete** |
| **Phase 5 Stage 12.5-pre** — Expose local `OmniNet` PeerId. Tiny additive lift so `advertise-peer` can bind to the actual running network identity rather than an operator-typed string. `OmniSwarm` + `OmniNet` gain a `local_peer_id()` accessor; the value is captured BEFORE the run task is spawned (no races against swarm events). Rustdoc documents the "restart = new PeerId" caveat: `SwarmBuilder::with_new_identity()` regenerates the libp2p keypair on every `OmniNet::new`, so peer advertisements are short-lived session routing hints — persistent libp2p identity is deferred. Mirrors the Stage 12.2-pre topic safety lift in shape + footprint. No wire change, no behaviour change. 2 new omni-net unit tests (stability across calls + distinct PeerIds across instances). | `crates/omni-net/src/lib.rs`, `crates/omni-net/src/swarm.rs` | **Complete** |
| **Phase 5 Stage 12.5** — Signed contributor peer advertisement / session routing. Removes manual `--downstream-to-peer` from pooled inference by letting contributors publish signed, **per-session, short-lived** (`expires_at_utc ≤ advertised_at_utc + 24h`) `ContributorPeerAdvertisement` envelopes binding `(session_id, contributor_pubkey_hex) → libp2p_peer_id + multiaddrs + capabilities`. PeerId derived from `OmniNet::local_peer_id()` (Stage 12.5-pre); no `--libp2p-peer-id` flag. **Local routing data only — not a marketplace, registry, permanent identity record, or chain authority.** Two coordinators running parallel sessions for the same `posted_id` cache their adverts independently. 2 new envelopes + 1 new gossipsub topic (`omni/contributor/session/peer-advert/v1`) + 2 new domain separators + new `PeerRoutingCache` (newest non-expired wins; dtype mismatch refused; chunk cap negotiated as `min(local, advertised)`) + new `process_peer_advertisement_announcement` (parallel to 12.3 processor shape, with `joins` argument for the matching-join check). 2 new `omni-node operator contributor` subcommands: `advertise-peer`, `watch-peer-adverts`. `run-assignment` gains `--peer-advert-dir` + `--resolve-downstream-peer-from-session` + `--downstream-resolve-dtype`; manual `--downstream-to-peer` still works and **takes precedence** when both are supplied. **No chain wire / payment / marketplace / proof-mode / on-chain change.** AttestationOnly only. SNIP only. Default `omni-node` tree still pulls zero halo2/prover/framework runtimes. 3 byte-stability + 18 schema/signature/drift/binding/expiry negatives + 7 routing-cache tests cover the new surface. | `crates/omni-contributor/src/peer_advert.rs` + `peer_routing.rs` (new); extended `canonical.rs` / `net.rs` / `relay.rs` / `error.rs` / `lib.rs` / `Cargo.toml`; `crates/omni-net/src/gossip.rs` + `lib.rs` (1 new topic constant); `crates/omni-node/src/contributor_cli.rs` (2 new subcommands + `run-assignment` extension); `docs/stage12-contributor-protocol.md` (Stage 12.5 section) | **Complete** |
| **Phase 5 Stage 12.4** — Live tensor / activation transport for pooled inference (extends 12.3; **no omni-net lift required**). New signed `ActivationHandoff` envelope (frozen `v1`) carried inside `omni-net`'s existing `/omni/tensor-xfer/1` request/response codec — outer `TensorRequest` fields are advisory routing hints, the signed inner envelope is the trust root. Two-step content integrity: sender signature binds `tensor_hash` + `byte_len` + chunk metadata + from/to assignment identities; receiver re-hashes reassembled bytes and rejects on hash or length mismatch. Chunking via `chunk_index` / `chunk_count` (≤ 256 chunks × ≤ 64 MiB each = 16 GiB cap). Strict linear pipeline at v1: `to.stage_index == from.stage_index + 1`. New `TensorTransport` trait + `InMemoryTensorTransport` (tests) + opt-in `OmniNetTensorTransport` (`network` feature). New `HandoffReceiver` reassembles + drift-checks across chunks. `InferenceRunner` gains `run_with_activations(manifest, input, upstream_activation_bytes) -> RunOutputWithActivation` with default impl delegating to `run()`, so Stage 12.0/12.1/12.2/12.3 runners stay valid; `ExternalCommandRunner` overrides to pass `--activation-in` / `--activation-out` paths to the subprocess (no model-framework runtime deps in `omni-node`). `run-assignment` gains `--activation-in-mode {none, live}` and `--activation-out-mode {snip, live, both, none}` — default `both` when downstream peer + assignment supplied, else `snip` (12.3 behavior preserved). New diagnostic subcommand `send-handoff`. Recipient PeerId is operator-supplied at v1; signed contributor peer-advertisement is Stage 12.5+. **No chain wire / payment / marketplace / proof-mode / on-chain change.** SNIP remains audit/fallback. Default `omni-node` tree still has zero halo2/prover/framework runtimes. 2 byte-stability + 24 schema/signature/hash/chunk negatives + 7 e2e integration tests (single-chunk + multi-chunk A→B, runner integration, tampered-chunk-bytes caught by reassembler, 12.3 chain still works under 12.4 deps). | `crates/omni-contributor/src/handoff.rs` + `handoff_verify.rs` + `tensor_transport.rs` (new); extended `canonical.rs` / `runner.rs` / `error.rs` / `lib.rs` / `Cargo.toml`; `crates/omni-node/src/contributor_cli.rs` (extended `run-assignment` + new `send-handoff`); `docs/stage12-contributor-protocol.md` (Stage 12.4 section) | **Complete** |
| **Phase 5 Stage 12.3** — Multi-contributor pooled-memory inference sessions. Cooperative compute pooling over the Stage 12 mesh: a coordinator opens a signed `ExecutionSession` for one posted job; contributors `ContributorJoin` with RAM + capability hints; the coordinator signs `WorkAssignment`s; each contributor publishes a signed `PartialContributorResult`; the coordinator wraps a standalone (v1) final `ContributorResult` in a signed `AggregatedContributorResult` that the local verifier accepts as a structurally-coherent multi-stage execution. **Coordination shell, not live pooled RAM** — inter-stage handoff goes through SNIP; live tensor / activation transport between participants is Stage 12.4. 5 new inner envelopes + 5 new network announcement envelopes + 5 new `omni/contributor/session/*/v1` gossipsub topics + new `CoordinatorSigner` role wrapper. 6 new `omni-node operator contributor` subcommands: `open-session`, `join-session`, `assign-work`, `run-assignment`, `aggregate-session`, `watch-sessions`. **Accounting rule**: final `total_base_units = input + output token count` (job-level), NOT a sum of partial totals — pipeline-correct. **No marketplace, no auction, no exclusive claim, no payment / reward / staking / slashing, no proof mode, no chain wire change.** AttestationOnly only. Aggregates require every assignment to have exactly one valid partial (no `--allow-incomplete` in v1). `WorkKind::Custom(label)` (non-empty printable ASCII, ≤ 64 chars) is a forward-compat escape hatch — `Custom("kv-cache-shard")`, `Custom("expert-7")` round-trip through canonical bytes today. Default `omni-node` build still pulls zero halo2/prover/framework runtimes. 7 inner byte-stability + 30 inner negatives + 5 net byte-stability + 9 e2e session integration tests (1-of-1, 3-stage pipeline, drift, wrong-coordinator, missing-partial, expired-session) cover the new surface. | `crates/omni-contributor/src/session.rs` + `session_verify.rs` (new), extended `canonical.rs` / `net.rs` / `relay.rs` / `signing.rs` / `error.rs` / `lib.rs`; `crates/omni-net/src/gossip.rs` (5 new topic constants); `crates/omni-node/src/contributor_cli.rs` (6 new subcommands); `docs/stage12-contributor-protocol.md` (Stage 12.3 section) | **Complete** |
| **Phase 5 Stage 12.2** — Contributor mesh network relay (extends 12.0 + 12.1; depends on 12.2-pre). Two new signed envelopes (`NetworkPostedJobAnnouncement`, `NetworkPostedResultAnnouncement`) under new domain separators `OMNINODE-CONTRIBUTOR-NET-JOB:v1:` / `OMNINODE-CONTRIBUTOR-NET-RESULT:v1:` — announcer signature **required**. New `ContributorRelay` trait + `InMemoryRelay` (tests) + `OmniNetRelay` (production, gated by `omni-contributor`'s opt-in `network` feature; wraps `omni_net::OmniNet`, uses 12.2-pre's `try_next_event` for non-blocking drain + `block_in_place + Handle::block_on` to bridge sync watch loop to async `publish`). New `NetworkSource` implements 12.1's `JobSource` trait — drains the relay, validates announcer signature, fetches `PostedJob` from SNIP, drift-guards against announcement copies of `posted_id` / `job_hash` / `model_hash`, and plugs the validated envelope into the existing 12.1 `run_watch_loop` (cost caps + poster signature + runner + post-run `verify_result` unchanged). Four new `omni-node operator contributor` subcommands: `announce-job`, `watch-network-jobs`, `announce-result`, `watch-network-results`. **Required cost caps** on `watch-network-jobs` mirror 12.1. **No chain wire change, no payment, no leases/claims, no double-pickup prevention, no `PostedJobsIndex`, no proof mode, no A–F flow.** Default `omni-node` build still pulls zero halo2/prover/framework runtimes. 9 watch-network-jobs e2e (incl. 2 ResultBroadcaster wiring tests) + 7 watch-network-results e2e (incl. tampered-link-contributor-signature) cover the network path on top of the 12.0 / 12.1 suite. Two bounded peer-wait flags on the announce-* paths (`--peer-wait-secs`, `--mesh-stabilize-ms`) so a fresh `OmniNet` doesn't silently drop publish before any peer is reachable. | `crates/omni-contributor/` (extends; new `net.rs` / `relay.rs` / extended `discover.rs` + `canonical.rs` + `watch.rs`), `crates/omni-node/src/contributor_cli.rs` (4 new subcommands), `docs/stage12-contributor-protocol.md` (Stage 12.2 section) | **Complete** |
| Phase 5 Stage 11e — GGUF-compatible proof strategy (shadow-verifier / partial-inference / replay-based); none of these prove full transformer correctness. Refused by `check_mainnet_eligible` layer 4 (`GgufClaim`) through end of Stage 11d.x — no GGUF candidate is on the Stage 11d table. | `omni-zkml`, `omni-pipeline` integration | Research / not yet planned |
| Phase 5 Stage 8+ — SUM Chain Tokenomics (staking / slashing / disputes / chain-side proof verification) — **gated on Stage 11d.3 sign-off + dedicated chain-team plan** | `contracts/` | Planned |

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

### CLI Commands

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

### Two-Mac LAN Demo

```bash
# Mac 1 — ingest model and serve shards
RUST_LOG=info cargo run --bin omni-node -- shard tinyllama.gguf

# Mac 2 — fetch a shard by CID (from Mac 1's manifest output)
RUST_LOG=info cargo run --bin omni-node -- fetch bafkr4i...
```

### End-to-End Pipeline Inference Demo

`showcase_tui.py` ties all four phases together in a single runnable script.
Each machine loads only its assigned layer slice from a local GGUF file.

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
[OmniStore]   shard 0  layers 0-10  cid=bafkr4iabc123…  215 MB  [embedding]
[OmniStore]   shard 1  layers 11-21  cid=bafkr4ixyz789…  209 MB  [output_head]

[GGUF] arch=llama  hidden=2048  layers=22  heads=32/4  vocab=32000
[GGUF] Split: Sender layers 0-10 | Receiver layers 11-21
[GGUF] 198 tensors injected into bare Model()
[GGUF] RAM pool drop complete.
```

---

## 5-Phase Implementation Roadmap

The protocol is constructed strictly bottom-up. Each phase produces a working milestone that can be demonstrated independently.

### Phase 1: P2P Mesh Networking — Complete

**Crate:** `omni-net` | **Foundation:** `omni-types`

Build the communication substrate. Nodes discover each other on the local network, establish encrypted QUIC connections, and exchange messages via Gossipsub pub/sub.

#### Phase 1a — LAN Mesh (Complete)

| Component | Implementation | Status |
|---|---|---|
| Transport | QUIC/v1 over UDP (TLS 1.3 built-in, no separate Noise step) | ✅ Done |
| LAN Discovery | mDNS — zero-configuration local peer discovery | ✅ Done |
| Messaging | Gossipsub pub/sub with signed messages (`ValidationMode::Strict`) | ✅ Done |
| Peer Exchange | Identify protocol — `/omni-node/0.1.0` | ✅ Done |
| Shard Transfer | `request-response` protocol — `/omni/shard-xfer/1` with `ShardCodec` | ✅ Done |
| Node API | `OmniNet` handle: `publish()`, `request_shard_chunk()`, `respond_shard()`, `next_event()`, `shutdown()` over async channels | ✅ Done |
| CLI | `omni-node listen` / `send` / `shard` / `fetch` | ✅ Done |

**Key design:** The swarm runs in a background `tokio` task, communicating with the `OmniNet` API handle via two async MPSC channels (256-slot capacity): commands flow in, events flow out. This keeps the libp2p internals fully encapsulated.

#### Phase 1b — Global WAN Transport & Decentralized NAT Traversal

OmniNode's networking layer has graduated from LAN-only mDNS discovery to a fully decentralized WAN transport — with **zero centralized servers**. No AWS instances, no Google STUN, no hosted relays. Every component of the NAT traversal stack is operated by the network's own participants.

##### Zero-Centralization Bootstrap

The Bootstrap Paradox — *"how do you find peers if you don't know any peers?"* — is solved with a **Kademlia DHT** (`/omni/kad/1.0.0`) seeded by community-operated bootstrap nodes. On startup:

1. The node dials bootstrap peers and seeds its Kademlia routing table.
2. A bootstrap query finds the k-closest peers to our own PeerId, rapidly expanding the routing table beyond the initial seeds.
3. Every `Identify::Event::Received` feeds the remote peer's **listen addresses** into Kademlia and registers its **observed address** as a local external address candidate.

This creates a **self-healing network graph**: if a bootstrap node goes offline, any 20+ node DHT continues to function. New nodes discover the network through any existing participant, not through a fixed list of servers.

##### Decentralized NAT Traversal

Most consumer devices sit behind NAT. OmniNode traverses firewalls in three stages — all without centralized infrastructure:

| Stage | Protocol | Function |
|-------|----------|----------|
| **1. Discovery** | AutoNAT | Probes random DHT peers with "can you dial me back?" requests. 3 confirmations = Public; 3 failures = Private. No Google STUN. |
| **2. Bridging** | Circuit Relay v2 | Firewalled nodes acquire a `/p2p-circuit` address from a volunteer relay (any open-NAT OmniNode). This is a **temporary bridge**, not a permanent proxy. |
| **3. Upgrade** | DCUtR | Both endpoints exchange observed addresses through the relay circuit and attempt simultaneous UDP hole-punching. On success, the relay circuit is replaced with a **direct QUIC connection**. |

Nodes with open NATs automatically volunteer as relays by enabling `relay_server: true`. The `active_relay_reservation` state machine prevents repeated reservation spam on flapping NAT status.

##### Mathematical Anti-DDoS Limits

Volunteer relay nodes are protected by three layers of rate limiting that make resource exhaustion **mathematically infeasible**:

| Limit | Value | Purpose |
|-------|-------|---------|
| `max_circuit_bytes` | **8 MiB** | Forces DCUtR upgrade before any bulk tensor transfer. A 215 MB TinyLlama shard cannot transit the relay — it *must* go direct. |
| `circuit_src_per_peer` | 4 / 60 s | Caps circuit opens per peer identity. |
| `circuit_src_per_ip` | 8 / 60 s | **Sybil-resistant.** A single IP spinning up 100 peer IDs still only gets 8 circuits per minute. This is the critical defense against multi-identity DDoS. |
| `reservation_rate_per_peer` | 2 / 60 s | Prevents reservation flooding. |
| `max_circuit_duration` | 120 s | Circuits that fail to upgrade are automatically torn down. |

The 8 MiB cap is the key architectural invariant: relay circuits exist exclusively as a DCUtR bootstrap mechanism. All tensor and shard traffic flows over direct QUIC connections — the relay never sees production data.

##### Systematic Memory Safety

The QUIC async event loop enforces **leak-proof garbage collection** on inbound request channels:

- Every inbound request creates a `PendingInbound<T>` struct pairing the `ResponseChannel` with its `InboundRequestId`.
- The channel is inserted into the pending HashMap **only if** the internal event queue accepts the notification (`try_send` returns `Ok`).
- If the queue is saturated, the `ResponseChannel` is **explicitly dropped** — physically releasing the memory and signaling to libp2p that no response is coming.
- On `InboundFailure` or `ResponseSent`, a reverse-index lookup (`InboundRequestId → channel_id`) cleans up both maps in O(1).

This guarantees that under sustained high-throughput tensor streams (e.g., two nodes running continuous pipeline inference), orphaned channels cannot accumulate — eliminating OOM as a failure mode in the transport layer.

| Component | Implementation | Status |
|---|---|---|
| WAN Discovery | Kademlia DHT — internet-scale peer lookup | **Complete** |
| NAT Traversal | AutoNAT detection → Circuit Relay → DCUtR hole-punching | **Complete** |
| Anti-DDoS Relay Limits | 8 MiB cap, per-peer + per-IP rate limiting, Sybil-resistant | **Complete** |
| Leak-Proof Event Loop | `PendingInbound<T>` with conditional insert + reverse-index cleanup | **Complete** |
| TCP Fallback | TCP + Noise transport for non-QUIC peers | ⏳ Deferred |
| Capability Ads | Custom request-response protocol — advertise RAM, platform, loaded layers | ⏳ Deferred |

**Gossipsub Topics:**
- `omni/test/v1` — integration test messages
- `omni/capability/v1` — periodic hardware capability heartbeats
- `omni/shard/v1` — shard availability announcements (bincode-serialized `ShardAnnouncement`)
- `omni/pipeline/v1` — pipeline coordination messages
- `omni/proof/v1` — zk proof announcements

---

### Phase 2: GGUF Model Sharding — Complete

**Crate:** `omni-store` | **Depends on:** `omni-net`, `omni-types`

Chunk GGUF model weight files by transformer block, content-address each shard with BLAKE3 → CIDv1, distribute them across the mesh via a custom 64 MiB sliding-window streaming protocol, and serve them from memory-mapped storage on demand.

#### The iroh Pivot

The original design called for [iroh](https://github.com/n0-computer/iroh) (n0.computer's BLAKE3-native blob storage) for shard distribution. During implementation, we discovered an **unresolvable `hickory-resolver` feature conflict** between iroh (any version) and libp2p 0.55. Both crates pull in `hickory-resolver` but require mutually exclusive feature sets — no version combination resolves.

**Solution:** We dropped iroh entirely and built a custom shard transfer protocol directly on top of `libp2p::request_response`. This eliminated the dependency conflict while giving us tighter control over the wire format, chunk windowing, and backpressure.

#### Storage Architecture

| Component | Implementation |
|---|---|
| GGUF Parsing | Custom zero-copy parser (`gguf.rs`) — `memmap2` maps the file, parser reads header/metadata/tensor index without touching tensor data. Supports GGUF v2 and v3, all 13 metadata value types |
| Layer-Wise Chunking | `chunker.rs` classifies tensors by name (`token_embd.*`, `blk.{N}.*`, `output.*`) and groups them into shards by `layers_per_shard` (default: 4). Embedding → first shard, output head → last shard |
| Content Addressing | BLAKE3 (multicodec `0x1e`) → CIDv1 with raw codec (`0x55`), base32lower encoding. Deterministic: same data always produces the same CID |
| On-Disk Store | `store.rs` — filesystem store at `~/.omninode/store/<cid>.shard`. Write-once, content-addressed (no race conditions) |
| Manifest | `manifest.rs` — CBOR-serialized `ModelManifest` listing every shard's CID, layer range, size, and BLAKE3 hash |
| Integrity | `verify.rs` — BLAKE3 hash and CID verification on every received shard |
| Memory Streaming | `mmap.rs` — memory-mapped shard files via `memmap2`, OS pages in data on demand |

#### 64 MiB Sliding-Window Shard Transfer

Large shards (hundreds of MB) are transferred in multiple request-response round-trips with offset/length windowing:

```
Requester                                  Responder
─────────                                  ─────────
ShardRequest{cid, offset=0, max=64MB}      →
                                           ← ShardResponse{total=350MB, data=[64MB]}
ShardRequest{cid, offset=64MB, max=64MB}   →
                                           ← ShardResponse{total=350MB, data=[64MB]}
... repeat until all bytes received ...
Reassemble → verify BLAKE3 → verify CID → store to disk
```

- **Wire format:** `[u32 BE length][bincode payload]` over QUIC substreams
- **Peak RAM:** ~128 MiB (one send + one receive buffer)
- **Chunk size:** Configurable via `StoreConfig::max_shard_msg_bytes` (default 64 MiB)
- **Safety limit:** Codec rejects any single message > 256 MiB
- **Request timeout:** 120 seconds per round-trip

The `FetchManager` state machine orchestrates multi-chunk fetches: tracks in-progress transfers, accumulates chunks, requests the next window automatically, and verifies + stores the reassembled shard on completion.

#### Chunking Example (LLaMA 7B, 32 layers, 4 per shard)

```
Shard 0: embedding + global + blocks 0-3     → CID_0  (~510 MB)
Shard 1: blocks 4-7                           → CID_1
Shard 2: blocks 8-11                          → CID_2
  ...
Shard 7: blocks 28-31 + output head           → CID_7
```

#### Shard Manifest Format

```json
{
  "model_name": "llama-7b-q4_k_m",
  "model_hash": "blake3:abc123...",
  "architecture": "llama",
  "total_layers": 32,
  "quantization": "Q4_K_M",
  "total_size_bytes": 4080218931,
  "gguf_version": 3,
  "shards": [
    {
      "shard_index": 0,
      "cid": "bafkr4i...",
      "layer_range": { "start": 0, "end": 3 },
      "includes_embedding": true,
      "includes_output_head": false,
      "size_bytes": 510027366,
      "blake3_hash": "deadbeef..."
    }
  ]
}
```

**Milestone:** CLI command `omni-node shard <model.gguf>` chunks a model, stores shards locally, announces them on the mesh, and serves them to peers. `omni-node fetch <cid>` fetches and verifies a shard from a LAN peer.

---

### Phase 3: FFI Bridge & Local Inference — Complete

**Crate:** `omni-bridge` | **Python package:** `python/omninode` | **Depends on:** `omni-store`, `omni-net`, `omni-types`

Bridge the Rust storage and networking layers to Python via PyO3 0.23 + maturin. The critical optimization is zero-copy weight transfer on Apple Silicon unified memory using the Python Buffer Protocol.

| Component | Implementation | Status |
|---|---|---|
| FFI Framework | PyO3 0.23 + maturin — Rust compiles to native Python extension (`omninode._omni_bridge`) | ✅ Done |
| Zero-Copy Shard Access | `PyShardView` implements `__getbuffer__` over `memmap2::Mmap` — struct-owned `shape`/`strides` arrays for pointer stability | ✅ Done |
| Store Bindings | `PyOmniStore` — `ingest_model()`, `mmap_shard()`, `has_shard()`, `get_shard()` | ✅ Done |
| Net Bindings | `PyOmniNet` — `publish()`, `request_shard()`, `next_event()`, `shutdown()` with context manager | ✅ Done |
| Type Wrappers | `PyNetConfig`, `PyStoreConfig`, `PyLayerRange`, `PyShardDescriptor`, `PyModelManifest` | ✅ Done |
| Event System | Flat `PyNetEvent` struct with `kind` discriminator + optional fields | ✅ Done |
| Async Strategy | `OnceLock<tokio::Runtime>` singleton — all async Rust methods wrapped via `block_on()` | ✅ Done |
| Error Mapping | `PyOmniError` / `PyStoreError` exception hierarchy via `create_exception!` | ✅ Done |

#### Zero-Copy Python Buffer Protocol

`PyShardView` wraps a `memmap2::Mmap` and exposes the raw memory pointer to Python via `__getbuffer__`. No data is copied — the kernel's virtual memory subsystem pages data in from the `.shard` file on demand.

```python
import numpy as np
from omninode import OmniStore

store = OmniStore()
manifest = store.ingest_model("models/tinyllama-1.1b-q4_k_m.gguf")

# Zero-copy: mmap → PyShardView → memoryview/numpy (120µs for 431 MB)
view = store.mmap_shard(manifest.shards[0].cid)
tensor_data = np.frombuffer(view, dtype=np.uint8)  # zero-copy
print(f"Shard: {len(view)} bytes, dtype={tensor_data.dtype}")

# Network: fetch a shard from a LAN peer
from omninode import OmniNet
with OmniNet() as net:
    event = net.next_event(timeout_secs=30.0)
    if event and event.kind == "peer_discovered":
        net.request_shard(event.peer_id, "bafkr4i...")
```

**Apple Silicon Unified Memory Path (Zero-Copy):**
```
Rust mmap (GGUF shard file)
    │  ← file is in unified memory (CPU + GPU share physical RAM)
    ▼
PyO3 __getbuffer__ exposes raw pointer to Python
    │  ← no copy: numpy.frombuffer() wraps the pointer (120µs for 431 MB)
    ▼
MLX mx.array wraps the numpy buffer
    │  ← no copy: Metal GPU reads directly from the same physical memory
    ▼
Inference executes on GPU — zero memory copies from disk to compute
```

#### Local GPU Inference (MLX)

The zero-copy path was validated end-to-end on Apple Silicon: Rust `mmap` → PyO3 `__getbuffer__` → NumPy → MLX `mx.array` → Metal GPU tensor math — with **zero memory copies** from disk to GPU compute.

```python
import mlx.core as mx
import numpy as np
from omninode import OmniStore

store = OmniStore()
manifest = store.ingest_model("models/tinyllama-1.1b-q4_k_m.gguf")

view = store.mmap_shard(manifest.shards[0].cid)       # 120 µs  (zero-copy mmap)
np_arr = np.frombuffer(view, dtype=np.float32)         # zero-copy buffer wrap
gpu_tensor = mx.array(np_arr)                          # 0.89s   (MLX realization)
result = mx.sum(gpu_tensor)                            # 0.43s   (Metal GPU math)
mx.eval(result)
```

| Stage | Time | Memory Copies |
|---|---|---|
| Rust mmap → PyShardView | 120 µs | 0 |
| NumPy `frombuffer()` | ~0 µs | 0 |
| MLX `mx.array()` realization | 0.89 s | 0 (unified memory) |
| GPU tensor summation (452 MB) | 0.43 s | 0 |
| **Total: disk → GPU result** | **~1.3 s** | **0 copies** |

> **452 MB** TinyLlama embedding layer — mapped, realized on GPU, and reduced in 1.3 seconds with zero memory copies on Apple Silicon unified memory.

**Milestone:** `maturin develop && python -c "import omninode"` — Python imports the native extension, ingests a GGUF model, and maps a 431 MB shard into a NumPy array in 120 microseconds with zero memory copies. MLX GPU inference validated end-to-end.

---

### Phase 4: Pipeline Parallelism — Complete

**Crate:** `omni-pipeline` | **Depends on:** `omni-net`, `omni-types`

Distribute inference across multiple nodes. Each node executes a contiguous range of model layers (a *stage*), forwarding hidden-state activation tensors to the next node over the existing libp2p QUIC mesh. `omni-pipeline` is a **coordination layer** — the actual forward pass (matrix multiply, attention, etc.) executes in Python via MLX/llama.cpp through `omni-bridge`.

| Component | Implementation | Status |
|---|---|---|
| TensorCodec | `/omni/tensor-xfer/1` request-response protocol — `[u32 BE len][bincode]` wire format, 128 MiB safety limit. Activation sizes: 7B ≈ 4 MB, 13B ≈ 5 MB, 70B ≈ 32 MB (f16) | ✅ Done |
| Session Formation | Gossipsub protocol on `omni/pipeline/v1` — `Propose → CapabilityOffer → ScheduleAssigned → StageReady → StartInference` | ✅ Done |
| Planner | RAM-proportional layer assignment — filter pipeline-ready nodes, sort by available RAM, assign contiguous `LayerRange`s proportionally | ✅ Done |
| Scheduler | GPipe micro-batch scheduling — deterministic execution grid, `efficiency = M / (M + S - 1)`, default `num_micro_batches = 2 × num_stages` | ✅ Done |
| Coordinator | `PipelineCoordinator` produces `PipelineAction::PublishMessage` values — no direct network I/O, caller dispatches via OmniNet | ✅ Done |
| Executor | `StageExecutor` manages per-node stage state — builds `TensorRequest`/`TensorResponse`, tracks micro-batch progress | ✅ Done |
| Session State Machine | `Forming → Scheduled → Running → Completed \| Failed` with strict transition validation | ✅ Done |
| Heartbeat Monitor | 3-second interval, 3× timeout factor — tracks liveness per `(session_id, stage_index)` | ✅ Done |
| Python FFI Bridge | `PyPipelineCoordinator` and `PyStageExecutor` exposed via PyO3 — drive distributed inference from Python/MLX | ✅ Done |

**Key design:** The coordinator and executor are pure synchronous state machines that produce action descriptors. They never call OmniNet directly — the caller (omni-node or Python) executes the network operations. This keeps `omni-pipeline` free of any libp2p dependency.

#### Hidden State Transfer Data Flow

```
Stage 0 (Node A)             Stage 1 (Node B)              Stage 2 (Node C)
────────────────             ────────────────              ────────────────
[token_ids]
     │
embed + layers 0-10
     │
     ├── /omni/tensor-xfer/1 ──►
     │   TensorRequest {              layers 11-21
     │     data: [f16 activations]         │
     │     (seq_len × hidden_dim × 2)      ├── /omni/tensor-xfer/1 ──►
     │   }                                 │   TensorRequest {              layers 22-31 + lm_head
     │                                     │     data: [f16]                      │
     ...                                   ...                                    ▼
                                                                            logits → token
```

**Activation sizes (f16):**

| Model | Hidden Dim | Seq Len | Transfer Size |
|---|---|---|---|
| 7B | 4096 | 512 | **4 MB** |
| 13B | 5120 | 512 | **5 MB** |
| 70B | 8192 | 2048 | **32 MB** |

All well within the 128 MiB TensorCodec limit and QUIC transport capacity.

#### GPipe Micro-Batch Schedule

```
Time →  0    1    2    3    4    5    6    7
S0:    [m0] [m1] [m2] [m3]
S1:         [m0] [m1] [m2] [m3]
S2:              [m0] [m1] [m2] [m3]

Pipeline bubble = S-1 time slots at start + end
Efficiency = M / (M + S - 1)
Default: num_micro_batches = 2 × num_stages (auto)
```

#### RAM-Proportional Planner

```
Inputs:
  capabilities: [(peer_id, available_ram_bytes, local_shard_cids, pipeline_ready)]
  total_layers:  32

Algorithm:
  1. Filter nodes with pipeline_ready = true
  2. Sort by available_ram_bytes descending
  3. Assign layers proportionally: node with 2× RAM gets 2× layers
  4. Enforce contiguity (each node gets a contiguous LayerRange)
  5. Embedding → first stage, output head → last stage
```

#### Python Bridge API

```python
from omninode import PipelineCoordinator, StageExecutor, PipelineCapability, OmniNet

# Coordinator proposes session → gets gossipsub bytes to publish
coord = PipelineCoordinator()
session_id, msg = coord.propose_session("llama-7b", "abc123", 32, local_peer_id)
net.publish("omni/pipeline/v1", msg)

# Collect capability offers from peers
cap = PipelineCapability("peer-b", ram_bytes=16e9, available_ram_bytes=8e9,
                         platform="AppleSilicon", local_shard_cids=["bafkr4i..."],
                         available_layers=[(0, 31)], pipeline_ready=True)
coord.handle_capability_offer(session_id, cap)

# Finalize schedule → JSON for executors + gossipsub bytes
schedule_json, msg = coord.finalize_schedule(session_id, hidden_dim=4096)

# Each node creates an executor for its assigned stage
executor = StageExecutor(stage_index=1, schedule_json=schedule_json)
# ... receive tensor → MLX forward pass → send to next stage
req = executor.build_forward_request(micro_batch_index=0, data=activations,
                                     seq_len=512, hidden_dim=4096, dtype=0)
```

**Milestone:** `PipelineCoordinator` and `StageExecutor` compiled and passed all 27 unit tests. `TensorCodec` compiled and passed all 8 networking tests. Python bridge (`PyPipelineCoordinator`, `PyStageExecutor`, `PyPipelineConfig`, `PyPipelineCapability`) fully exposed via PyO3 for driving distributed inference from MLX.

---

#### Phase 4b: End-to-End Inference Demo — `showcase_tui.py`

The showcase script is the full-stack integration test: it exercises Phase 2
(OmniStore CID verification), Phase 3 (PyO3 FFI bridge), and Phase 4 (QUIC
tensor routing) in a single two-process demo that runs real autoregressive
inference across two machines over a LAN.

---

##### Native GGUF-to-MLX Bridge

All model weights are loaded directly from the `.gguf` file using Apple's
low-level `mx.load()` API. There is no dependency on HuggingFace Hub, no
`config.json`, and no `mlx_lm.load()` call.

```python
# Low-level GGUF parse — raw tensors + metadata dict in one call
weights, metadata = mx.load(gguf_path, return_metadata=True)

# Architecture inferred entirely from the file
arch            = metadata["general.architecture"]          # e.g. "llama"
hidden_dim      = weights["token_embd.weight"].shape[1]     # e.g. 2048
vocab_size      = weights["token_embd.weight"].shape[0]     # e.g. 32000
total_layers    = len([k for k in weights if "attn_q.weight" in k])
n_sender_layers = total_layers // 2

# ModelArgs populated from GGUF metadata — no config.json required
args = ModelArgs(
    hidden_size             = hidden_dim,
    num_hidden_layers       = total_layers,
    intermediate_size       = int(metadata[f"{arch}.feed_forward_length"]),
    num_attention_heads     = int(metadata[f"{arch}.attention.head_count"]),
    num_key_value_heads     = int(metadata[f"{arch}.attention.head_count_kv"]),
    vocab_size              = vocab_size,
    rms_norm_eps            = float(metadata[f"{arch}.attention.layer_norm_rms_epsilon"]),
    max_position_embeddings = int(metadata[f"{arch}.context_length"]),
    rope_theta              = float(metadata[f"{arch}.rope.freq_base"]),
    tie_word_embeddings     = "output.weight" not in weights,
)
```

**GGUF → MLX tensor name map (Llama family):**

| GGUF key | MLX parameter path |
|---|---|
| `token_embd.weight` | `model.embed_tokens.weight` |
| `output_norm.weight` | `model.norm.weight` |
| `output.weight` | `lm_head.weight` |
| `blk.{N}.attn_norm.weight` | `model.layers.{N}.input_layernorm.weight` |
| `blk.{N}.ffn_norm.weight` | `model.layers.{N}.post_attention_layernorm.weight` |
| `blk.{N}.attn_q.weight` | `model.layers.{N}.self_attn.q_proj.weight` |
| `blk.{N}.attn_k.weight` | `model.layers.{N}.self_attn.k_proj.weight` |
| `blk.{N}.attn_v.weight` | `model.layers.{N}.self_attn.v_proj.weight` |
| `blk.{N}.attn_output.weight` | `model.layers.{N}.self_attn.o_proj.weight` |
| `blk.{N}.ffn_gate.weight` | `model.layers.{N}.mlp.gate_proj.weight` |
| `blk.{N}.ffn_up.weight` | `model.layers.{N}.mlp.up_proj.weight` |
| `blk.{N}.ffn_down.weight` | `model.layers.{N}.mlp.down_proj.weight` |

---

##### Decentralized RAM Pooling

After slicing out this node's assigned layers, the full model object is
explicitly destroyed and Apple Silicon's unified memory cache is flushed.
Each node physically holds only its layer slice in VRAM.

```python
# Slice out this node's layers
shard = SenderShard(model, n_sender_layers)   # or ReceiverShard

# Aggressively release the other 50% of weights from unified memory
del model, weights, mapped_weights
gc.collect()
mx.metal.clear_cache()
```

On a 22-layer model (TinyLlama 1.1B), each node retains ~215 MB of weights
instead of the full ~430 MB — a 50% RAM reduction per node, directly
demonstrating the protocol's core value proposition.

---

##### Pure-QUIC Autoregressive Pipeline

All tensor communication is routed over QUIC `request_tensor` streams.
Gossipsub is intentionally unused for ML operations — it does not support
the latency or ordering guarantees required for autoregressive inference.

**Wire protocol discriminator (encoded in the `hidden_dim` field):**

| `hidden_dim` value | Payload |
|---|---|
| `== model hidden size` | Float16 hidden-state activations `(1, seq_len, hidden_dim)` |
| `== 1` | 4-byte little-endian token ID (including EOS sentinel) |

**Autoregressive ping-pong loop:**

```
Sender                                    Receiver
──────                                    ────────
embed(prompt_tokens) → layers[0:N]
        ──── hidden_states (prefill) ──►
                                          layers[N:] → norm → argmax → token_1
        ◄─── token_id (hidden_dim=1) ────
embed(token_1) → layers[0:N]
        ──── hidden_states (decode) ──►
                                          layers[N:] → norm → argmax → token_2
        ◄─── token_id (hidden_dim=1) ────
... repeat until EOS ...
```

Both nodes maintain independent KV caches. The residual stream carries no
positional information, so caches stay synchronized as long as both nodes
process tokens in the same order — which the ping-pong protocol guarantees.

**Milestone:** Two machines running `showcase_tui.py` — one as Sender, one as
Receiver — discover each other via mDNS, verify GGUF shard CIDs via OmniStore,
load only their assigned layer slice from disk, and execute real autoregressive
LLM inference across the LAN with a Rich TUI displaying live token streaming.

---

### Phase 5 Stage 1: SNIP V2 Storage Integration — Complete

**Crates:** `omni-types` (data), `omni-store` (CLI adapter) | **Depends on:** `sum-node` v0.4.0-rc3+

Stage 1 of Phase 5 introduces a **storage substrate** for the future zkML / tokenomics work without touching any of the existing CIDv1 / BLAKE3 shard identity that Phases 2–4 depend on. SNIP V2 references are strictly **additive**: optional fields on the existing `ModelManifest` / `ShardDescriptor`, with `#[serde(default, skip_serializing_if = "Option::is_none")]` so every pre-Phase-5 JSON or CBOR manifest still deserializes unchanged.

| Concern | Where | Notes |
|---|---|---|
| 32-byte BLAKE3 Merkle root | `omni_types::phase5::SnipV2ObjectId` | `0x`-prefixed lowercase hex (66 chars); custom serde as string; distinct from CIDv1 |
| Lifecycle | `omni_types::phase5::SnipV2Lifecycle` | `Active` / `Pending` / `Abandoned`; serde for persistence, separate `FromStr` for stdout |
| Object reference | `omni_types::phase5::SnipV2ObjectRef` | `{ merkle_root, lifecycle, plaintext_size_bytes? }` |
| Inference data containers | `omni_types::phase5::{InferenceCommitment, InferenceAttestation}` | Carry session ID + model hash + manifest/proof Merkle roots + verifier address & signature (encoding deliberately opaque until SUM Chain spec is locked) |
| Optional manifest fields | `ShardDescriptor::snip_v2`, `ModelManifest::snip_v2` | Both `Option<SnipV2ObjectRef>`; CIDv1 / BLAKE3 fields are untouched |
| CLI adapter | `omni_store::snip_v2::{SnipV2Cli, SnipV2CliConfig, SnipV2Error}` | Wraps the documented `sum-node` CLI surface |

**Documented `sum-node` commands wrapped:**

```bash
# Ingest a local file into SNIP V2 Public storage
sum-node ingest-v2 <path> --visibility public
# stdout:
#   merkle_root: 0x<64 hex>
#   lifecycle:   <Active|Pending|Abandoned>

# Download a whole SNIP V2 object to a local path (no range reads in V2 today)
sum-node download <merkle_root_hex> --output <path>
```

**Module layout:** the parser is a pure function (`parse_ingest_stdout`) with no I/O, separated from process invocation (`SnipV2Cli::ingest_public` / `download_public`). All 14 parser/config tests run without ever spawning `sum-node`, alongside 2 CBOR backward-compatibility tests in `omni-store::manifest` that decode a legacy mirror manifest into the new struct shape.

**Typed errors (`SnipV2Error`, bridged into `StoreError::SnipV2`):** `CommandSpawn`, `NonZeroExit { code, stderr }`, `InputNotFound { path }`, `MissingMerkleRootLine`, `MissingLifecycleLine`, `InvalidMerkleRoot(SnipV2ParseError)`, `UnknownLifecycle(String)`, `UnsupportedLifecycle(SnipV2Lifecycle)`, `ParseFailure { reason }`, `DownloadFailed { code, stderr }`.

**What Stage 1 deliberately does not do:**
- No SNIP V1 — V1 is obsolete and unsupported.
- No Private V2 — encryption keys, access lists, and `download_private` are out of scope.
- No range reads — SNIP V2 does not expose them today; OmniNode's libp2p 64 MiB sliding window (Phase 2) is unchanged.
- No `omni-store::build_manifest` change — `snip_v2` defaults to `None` and is populated by a later stage.
- No PyO3 bridge surface, no `omni-pipeline` scheduler changes, no `omni-net` codec changes.
- No chain client, no tokenomics, no proof backend — `omni-zkml` remains a stub for Stage 2+.

---

### Phase 5 Stage 2: SNIP-backed Model Artifacts — Complete

**Crate:** `omni-store` | **Depends on:** Stage 1, `sum-node` v0.4.0-rc3+

Stage 2 wires the Stage 1 substrate into actionable storage flows: publishing local shards and their manifest to SNIP V2 Public, and restoring them on a fresh machine into the existing OmniNode shard cache. Both flows operate strictly through SNIP V2's Public surface — V1 is not used, Private V2 is not used, no range reads are issued. OmniNode's existing CIDv1 / BLAKE3 identity remains the only authority for cache key resolution; SNIP V2 roots are a *parallel* identifier used solely to address remote bytes.

| Operation | Function | Notes |
|---|---|---|
| Publish shards + manifest | `OmniStore::publish_to_snip` → `snip_v2_artifacts::publish_to_snip` | Resumable per-shard. Skips already-populated `snip_v2` refs. |
| Restore shards from a manifest | `OmniStore::restore_from_snip` → `snip_v2_artifacts::restore_from_snip` | Skips already-cached shards. Verifies each download against the manifest's BLAKE3 + CID. |
| Restore a manifest from its SNIP root | `OmniStore::restore_manifest_from_snip` → `snip_v2_artifacts::restore_manifest_from_snip` | Top-level `snip_v2` on the returned manifest is `None` by construction (see below). |
| Test seam | `trait SnipV2Adapter` in `omni-store::snip_v2`, implemented by `SnipV2Cli` | All 15 Stage-2 unit tests use a content-addressed in-memory fake; no `sum-node` shell-out. |

**Publish flow & resumability.** Each shard's ingest call is followed immediately by `write_manifest`, so the on-disk file always reflects the highest-water-mark of populated refs. A mid-loop failure preserves prior refs on disk; a re-run from the loaded manifest skips them and resumes from the failure point. After all shards succeed, the manifest file itself is ingested into SNIP; its root identifies the canonical "shards-populated, top-level-`None`" bytes.

**On-disk vs SNIP-stored manifest bytes — intentional divergence.** After SNIP returns the manifest root, the on-disk file is rewritten one final time to include a top-level `snip_v2` self-pointer. The SNIP-stored bytes do **not** carry this self-pointer; they remain stable under the canonical root. Restoring a manifest by its SNIP root therefore yields a `ModelManifest` whose top-level `snip_v2` is `None`, while every shard-level `snip_v2` survives intact — callers that want a local self-pointer add one explicitly after restore.

**Atomic restore.** Each shard is downloaded to `<cid>.shard.partial`, mmap-ed, verified against the manifest's BLAKE3 hash and CIDv1 via the existing `verify::verify_blake3` / `verify::verify_cid` primitives, and only then renamed to its final `<cid>.shard` path. Verification failure removes the partial; the cache state is unchanged. Crashes mid-download leave only a partial, which the next run overwrites or cleans.

**New typed errors** (additions to `StoreError`):
- `ShardFileMissing { cid, path }` — publish-side; the manifest references a CID whose local `<cid>.shard` file is not on disk.
- `ShardLacksSnipRef { cid }` — restore-side; the manifest's shard has no `snip_v2` ref to follow.

Verification failures continue to surface through the existing `StoreError::IntegrityMismatch`; SNIP CLI failures continue to surface through `StoreError::SnipV2(SnipV2Error)`. No new dependencies were added to any `Cargo.toml`.

**What Stage 2 deliberately does not do:**
- No SNIP V1, no Private V2, no range reads.
- No edits to `omni-net`, `omni-pipeline`, `omni-bridge`, `omni-zkml`, or `python/omninode`.
- No edits to `build_manifest` — SNIP publishing is an explicit, separate call against an already-ingested model.
- No chain client, no tokenomics, no proof backend — those land in Stage 3+.

---

### Phase 5 Stage 3: Proof Artifact Flow — Complete

**Crate:** `omni-zkml` | **Depends on:** Stage 1, Stage 2, `omni-store::SnipV2Adapter`

Stage 3 lays down the byte-shovel substrate that future zkML stages will sit on top of: the `omni-zkml` crate is no longer a stub. It accepts opaque proof and response byte files from a caller, publishes them to SNIP V2 Public through the existing [`omni-store::SnipV2Adapter`](crates/omni-store/src/snip_v2.rs), computes the response BLAKE3 hash, and assembles an `InferenceCommitment` ready for a future stage to sign and submit. **No real proof generation, no verifier, no signing, no chain client** is wired in this stage — proof bytes are opaque and the actual zk machinery is the subject of Stage 4+.

| Concern | Where | Notes |
|---|---|---|
| Proof byte file | `omni_zkml::ProofArtifact { local_path, snip_v2 }` | `snip_v2 == None` means "not yet published"; ingest skips when already populated |
| Response byte file + hash | `omni_zkml::ResponseArtifact { local_path, snip_v2, blake3_hash }` | `blake3_hash` is bare lowercase 64-char hex matching `ModelManifest::model_hash` convention |
| Publish flow | `omni_zkml::publish_proof_artifacts(&adapter, &mut response, &mut proof)` | Idempotent; per-case file-existence rules (see below) |
| Commitment builder | `omni_zkml::build_commitment(session_id, &manifest, &response, &proof)` | Strict pre-conditions; returns `omni_types::phase5::InferenceCommitment` |
| Local error type | `omni_zkml::ProofArtifactError` | Eight typed variants; SNIP and IO errors bridged via `#[from]` |

**Idempotent file-existence rules.** The publish flow only touches the local filesystem for what it actually needs:

| `response.snip_v2` | `response.blake3_hash` | Response file required? |
|---|---|---|
| `Some` | `Some` | No — response side fully skipped |
| `Some` | `None` | Yes — hash must be computed from the bytes (own preflight) |
| `None` | `Some` | Yes — needed for ingest |
| `None` | `None` | Yes — needed for both |

| `proof.snip_v2` | Proof file required? |
|---|---|
| `Some` | No — proof side fully skipped |
| `None` | Yes — needed for ingest |

A missing file always surfaces as the typed `ResponseFileNotFound { path }` or `ProofFileNotFound { path }`, never as a generic `Io` error. The hash-computation step on the response side has its **own** `is_file()` preflight, so a pre-supplied `snip_v2` with a missing local file still produces `ResponseFileNotFound` rather than leaking an I/O failure from `std::fs::read`.

**Strict commitment construction.** `build_commitment` enforces four pre-conditions, each mapped to a typed error: non-empty `session_id` (`EmptySessionId`), `manifest.snip_v2.is_some()` (`ManifestLacksSnipRoot`), `response.blake3_hash.is_some()` (`ResponseLacksHash`), and `proof.snip_v2.is_some()` (`ProofLacksSnipRoot`). Note that `response.snip_v2` is **not** required — `InferenceCommitment` carries only the response hash, not a response SNIP root.

**Restored-manifest annotation flow.** `OmniStore::restore_manifest_from_snip` returns a manifest with `snip_v2: None` at the top level (Stage 2's intentional contract — the canonical SNIP bytes never embed a self-pointer). To build a commitment from a restored manifest, callers explicitly annotate it with the root they restored from before calling `build_commitment`. This is exercised by [test `restored_manifest_requires_annotation_before_commitment`](crates/omni-zkml/src/artifact.rs).

**Dependencies added** (all workspace-declared; no new versions): `omni-store`, `blake3`, `thiserror`, `tracing` to `omni-zkml/Cargo.toml`; `tempfile` as a dev-dep. The root `Cargo.toml` is unchanged.

**What Stage 3 deliberately does not do:**
- No actual zk proof generation — no `ezkl`, no `risc0`, no witness construction.
- No proof verifier wiring; no circuit checks; no STARK/SNARK validation.
- No signing — `InferenceAttestation` is not produced this stage.
- No chain client, no on-chain submission, no tokenomics.
- No edits to `omni-store`, `omni-types`, `omni-net`, `omni-pipeline`, `omni-bridge`, or `python/omninode`.
- No SNIP V1, no Private V2, no range reads.

---

### Phase 5 Stage 4: Local Verifier Attestation Envelope — Complete

**Crate:** `omni-zkml` | **Depends on:** Stage 3, `omni-types::phase5::{InferenceCommitment, InferenceAttestation}`

Stage 4 turns an `InferenceCommitment` (Stage 3 output) into a signed `InferenceAttestation` through a deterministic, domain-separated pipeline. **No real cryptography, no chain submission, no verifier wiring.** The point of this stage is the canonical-bytes contract and the `Signer` trait seam — the same seam a future Ed25519-backed signer (libp2p identity ↔ chain address) and an eventual real chain submitter will plug into.

```
InferenceCommitment
  └─► compute_canonical_bytes (bincode 2.0 of CommitmentPayload { domain, commitment })
        └─► CommitmentDigest (BLAKE3 of canonical bytes — 32 bytes)
              └─► Signer::sign(digest) → verifier_signature
                    └─► InferenceAttestation { commitment, verifier_address, verifier_signature }
```

| Concern | Where | Notes |
|---|---|---|
| Domain tag | `omni_zkml::DOMAIN_TAG = "omninode.inference_attestation.v1"` | Encoded as the **first** field of `CommitmentPayload`; bumping the trailing `vN` is the contract for any breaking change to the envelope |
| Canonical envelope | `omni_zkml::CommitmentPayload { domain, commitment }` | bincode 2.0 + `config::standard()`, same configuration `omni-store::announce` / `omni-net::codec` already use |
| Digest | `omni_zkml::CommitmentDigest([u8; 32])` | BLAKE3 over the canonical bytes; `as_bytes`, `to_hex` (lowercase, no `0x`) |
| Signer abstraction | `omni_zkml::Signer { verifier_address() -> String, sign(&CommitmentDigest) -> Result<String, SignerError> }` | Both strings opaque; chain encoding (hex vs base58 vs bech32, sig scheme) deliberately pending |
| Builder | `omni_zkml::build_attestation(commitment, signer) -> AttestationResult<InferenceAttestation>` | Consumes the commitment by value (moves it into the attestation) |
| Local error type | `omni_zkml::{AttestationError, SignerError, AttestationResult<T>}` | `SignerError: Clone` so test fixtures can store and re-use canned outcomes; `AttestationError` is intentionally not Clone |

**Determinism guarantees** (each pinned by a named test):
- `canonical_bytes_are_deterministic` — same commitment → byte-equal canonical output across calls.
- `digest_changes_when_domain_tag_changes` — via the `pub(crate) compute_canonical_bytes_with_domain` test seam: same commitment, different domain string → different digest. Pins the version-bump contract.
- `digest_changes_when_session_id_changes` / `..._model_hash_changes` / `..._manifest_snip_root_changes` / `..._response_hash_changes` / `..._proof_snip_root_changes` — every commitment field flows into the digest.
- `digest_matches_independent_blake3_of_canonical_bytes` — `compute_digest(&c)` equals `blake3::hash(compute_canonical_bytes(&c))` byte-for-byte.

**Builder semantics:**
- Pre-validation rejects empty `session_id`, `model_hash`, `response_hash` *before* invoking the signer (asserted by spying on the fake signer's recorded digest).
- The signer receives the 32-byte `CommitmentDigest`, **not** the raw canonical bytes — pinned by `build_attestation_passes_digest_not_raw_bytes_to_signer`.
- Empty `verifier_address` and empty `verifier_signature` returned by the signer are rejected as typed errors.
- Signer failures propagate as `AttestationError::Signer(SignerError::Failed(msg))` with the diagnostic message preserved verbatim (`build_attestation_propagates_signer_failure_with_message`).

**Result-alias hygiene:** `omni-zkml` now exposes two non-colliding aliases:
- `omni_zkml::Result<T>` — Stage 3, `= std::result::Result<T, ProofArtifactError>`.
- `omni_zkml::AttestationResult<T>` — Stage 4, `= std::result::Result<T, AttestationError>`.

Callers import whichever they need; `attestation.rs` itself uses `AttestationResult` exclusively and never brings the Stage-3 `Result` alias into scope.

**Dependencies added** (both workspace-declared; no new versions, no root `Cargo.toml` edit):
- `bincode = { workspace = true }` — for the deterministic canonical envelope.
- `serde = { workspace = true }` — strict consequence of `CommitmentPayload` deriving `Serialize`/`Deserialize`.

**What Stage 4 deliberately does not do:**
- No real signature scheme — no Ed25519, no secp256k1, no actual key handling.
- No `Verifier` companion trait, no `verify_attestation` function. Stage 4 is producer-side only.
- No chain client, no RPC, no transaction encoding.
- No SUM Chain address/signature encoding decisions — strings remain opaque.
- No libp2p-identity ↔ chain-address binding implementation (the 32-byte seed convention is documented for Stage 5+).
- No edits to `omni-store`, `omni-types`, `omni-net`, `omni-pipeline`, `omni-bridge`, or `python/omninode`.
- No SNIP V1, no Private V2, no range reads, no tokenomics.

---

### Phase 5 Stage 5: Chain Client Abstraction & Offline Attestation Registry — Complete

**Crate:** `omni-zkml` | **Depends on:** Stage 4, `omni-types::phase5::InferenceAttestation`

Stage 5 ships the chain-shaped pieces OmniNode can own without depending on the unfinished SUM Chain spec: a synchronous `ChainClient` trait, a deterministic `AttestationId` keyed on `(session_id, verifier_address)`, a six-state `LocalAttestationStatus` state machine, and an on-disk JSON-per-record `AttestationRegistry` with atomic writes. Two workflow free functions (`submit_attestation_workflow`, `query_attestation_workflow`) compose the trait with the registry so a future real chain adapter slots in unchanged. **No real RPC, no on-chain calls, no tx encoding, no tokenomics.**

```
InferenceAttestation
  └─► compute_attestation_id  (BLAKE3 of bincode { domain, session_id, verifier_address })
        └─► AttestationRegistry::insert  → AttestationRecord { Pending }
              └─► submit_attestation_workflow ─► ChainClient::submit_attestation
                                                  └─► mark_submitted(receipt) → Submitted
                                                        └─► query_attestation_workflow ─► ChainClient::query_attestation_status(tx_id)
                                                              ├─► Submitted     → no transition
                                                              ├─► Included      → mark_included
                                                              ├─► Finalized     → mark_finalized   (terminal)
                                                              ├─► Failed{reason}→ mark_failed       (terminal)
                                                              └─► Unknown       → leave unchanged + tracing::warn!

(client-side, Stage 5.2 — separate from query_attestation_workflow:)
   staleness_policy + current_block ─► mark_stale_if_overdue ─► Submitted → Dropped (retryable)
```

| Concern | Where | Notes |
|---|---|---|
| Deterministic record key | `omni_zkml::compute_attestation_id` | Domain `"omninode.attestation_record.v1"` + `session_id` + `verifier_address`. Signature / commitment body deliberately excluded — matches the chain proposal's de-duplication rule. |
| Record identifier | `omni_zkml::AttestationId([u8; 32])` | Custom serde-as-lowercase-hex; filename is `<hex_id>.json` |
| Chain trait (sync) | `omni_zkml::ChainClient { submit_attestation, query_attestation_status(tx_id: &str) }` | No real RPC; opaque `String` ids and signature encodings; no `Pending` on the chain-returned status (chain only knows submitted-or-later). Query is keyed by `tx_id` (Stage 5.1) to match SUM Chain v1's `sum_getInferenceAttestationStatus(tx_hash)`. |
| Chain-returned status | `omni_zkml::AttestationStatus { Submitted, Included, Finalized, Failed { reason }, Unknown }` | Stage 5.1: matches SUM Chain v1's five-state lifecycle. `Unknown` replaces the earlier provisional `Dropped` — chain v1 surfaces unrecognized tx hashes (eviction, never-seen, or lag) via `Unknown`. |
| Local status | `omni_zkml::LocalAttestationStatus { Pending, Submitted, Included, Finalized, Failed { reason }, Dropped { reason: Option<String> } }` | `Pending` and `Dropped` are local-only. `Dropped` is a **client-side synthetic** state (chain never reports it); OmniNode sets it via staleness/timeout detection (Stage 5.2). The `Dropped → Submitted` retry hinge is unchanged. `Finalized` and `Failed` are terminal this stage. |
| Storage | `omni_zkml::AttestationRegistry::open(root)` | One JSON file per record at `<root>/<hex_id>.json`; atomic `.tmp` + rename; `list()` sorted by id hex ascending |
| Workflow seams | `submit_attestation_workflow`, `query_attestation_workflow` | Free functions generic over `ChainClient`; RPC failures propagate as `RegistryError::ChainClient(_)` and leave records unchanged |
| Local errors | `omni_zkml::{ChainClientError, RegistryError, RegistryResult<T>}` | Three error domains, three result aliases (`Result` Stage 3, `AttestationResult` Stage 4, `RegistryResult` Stage 5) coexist explicitly |

**Insert idempotency & conflict.** `insert(attestation)` computes the key from `(session_id, verifier_address)`:
- If no record exists → write a new `Pending` record.
- If a record exists and stores a byte-equal attestation → return it unchanged (no file rewrite, `updated_at` not bumped).
- If a record exists but stores a byte-different attestation → return `RegistryError::ConflictingAttestation { id }`, with the on-disk record preserved intact.

**RPC failures are not terminal.** Pinned by two named tests:
- `submit_workflow_returns_chain_error_and_leaves_record_pending` — fake client returns `Err(ChainClientError::Other("rpc gone"))`; workflow returns `Err(RegistryError::ChainClient(...))`; loaded record is still `Pending`.
- `query_workflow_returns_chain_error_and_leaves_record_unchanged` — same shape for the query path; record status unchanged after the failed RPC.

Only an explicit chain status of `Failed { reason }` (from `Submitted` or `Included`) transitions a record into a terminal local state. The retry hinge is `mark_submitted` accepting both `Pending` *and* `Dropped` as source states — `submit_workflow_resubmits_dropped_records` exercises the end-to-end Pending → Submitted → Dropped → Submitted path with a new receipt replacing the old one (`Dropped` is set via local staleness detection, not via the chain).

**Stage 5.1 — SUM Chain v1 alignment.** Stage 5.1 (in this section) realigned three pieces with the final chain spec:
- `ChainClient::query_attestation_status` is keyed by `tx_id: &str`, matching `sum_getInferenceAttestationStatus(tx_hash)`. The workflow externally takes an `AttestationId` (registry key) and internally reads `record.receipt.tx_id` before calling the chain.
- Chain-returned `AttestationStatus::Dropped` replaced with `Unknown` (no payload). The query workflow treats `Unknown` as observation-only: the record is left unchanged and `tracing::warn!` fires. `Unknown` is **never** translated to local `Failed` (terminal and wrong) or auto-translated to local `Dropped` (a staleness-policy decision belonging to Stage 5.2).
- New defensive error `RegistryError::SubmittedRecordMissingReceipt { id }` is returned when a queryable local record (`Submitted` or `Included`) has `receipt: None` — only reachable via hand-edited / corrupted JSON, since `mark_submitted` always sets the receipt. The chain is **not** called in that path.

**Stage 5.2 — staleness/timeout detection (Complete; see dedicated section below).** Optional `submitted_at_block: Option<u64>` on `AttestationRecord`, a `mark_submitted_with_block` registry method, and the `omni_zkml::staleness` module with `StalenessPolicy` + `is_record_stale` (pure) + `mark_stale_if_overdue` (workflow). Caller-supplied threshold; library hardcodes nothing.

**Atomic writes.** Every persisted change goes through `serde_json::to_vec_pretty` → write to `<hex_id>.json.tmp` → `fs::rename` to `<hex_id>.json`. Crashes mid-write leave at most a `.tmp` file behind; the `.json` is either present-and-complete or absent. Stray `.tmp` files are silently ignored by `list()`.

**`CommitmentDigest` JSON serde** was added to `attestation.rs` this stage (additive to Stage 4): `Serialize` / `Deserialize` impls that read/write the lowercase 64-char hex string. Required because `AttestationRecord` embeds the digest and persists to JSON. The Stage-1 `SnipV2ObjectId` hex-string serde pattern is reused.

**Dependencies added** (both workspace-declared; no new versions, no root `Cargo.toml` edit): `serde_json` (JSON persistence) and `chrono` (RFC3339 timestamps for `created_at` / `updated_at`).

**What Stage 5 deliberately does not do:**
- No real chain RPC, no `reqwest`/`hyper`/`alloy` clients.
- No tx encoding, no final receipt schema beyond two opaque strings.
- No staking, slashing, or reward formulas.
- No proof generation, no verifier wiring.
- No real cryptographic signer — Stage 4's `Signer` trait surface is unchanged.
- No SNIP V1, no Private V2, no range reads.
- No edits to `omni-types`, `omni-store`, `omni-net`, `omni-pipeline`, `omni-bridge`, or `python/omninode`.

---

### Phase 5 Stage 5.2: Client-Local Staleness / Retry Policy — Complete

**Crate:** `omni-zkml` (extends Stage 5 / 5.1) | **Depends on:** `tracing`, `chrono` (already in the workspace)

Stage 5.2 ships the **client-side staleness surface** that decides when a locally `Submitted` record has gone stale enough to mark `Dropped`. SUM Chain v1 does not return a chain-side `Dropped`; this module is the *only* writer that transitions `Submitted → Dropped`, and the resulting `Dropped` is a synthetic OmniNode decision. The retry hinge `Dropped → Submitted` is unchanged and continues to work via `mark_submitted` (legacy) or the new `mark_submitted_with_block`.

| Concern | Where | Notes |
|---|---|---|
| Policy type | `omni_zkml::StalenessPolicy` | Single knob `submitted_threshold_blocks: u64`. `StalenessPolicy::new(0)` returns `StalenessPolicyError::ZeroThreshold` — zero means "stale on the first new block past submit", almost always a config bug. Library hardcodes no formula like `finality_depth * N`; deriving the threshold from `chain_getChainParams` is the caller's responsibility. |
| Pure detection | `omni_zkml::is_record_stale(&record, current_block, &policy) -> bool` | True iff `status == Submitted` AND `submitted_at_block == Some(b)` AND `current_block.saturating_sub(b) > policy.submitted_threshold_blocks()`. Strict `>` (boundary `elapsed == threshold` is not stale). Pure: returns `false` (conservative) on height regression and on legacy records missing `submitted_at_block`; no logging. |
| Workflow helper | `omni_zkml::mark_stale_if_overdue(&registry, &id, current_block, &policy) -> RegistryResult<AttestationRecord>` | Reads the record; if `Submitted` and stale, calls `mark_dropped(Some("stale: submitted_at_block=N, current_block=M, threshold_blocks=T"))`. Non-`Submitted` source states are silent no-ops (mirrors `query_attestation_workflow` precedent for irrelevant inputs). Height regression and `submitted_at_block == None` paths each emit a typed `tracing::warn!` with the record id and both heights in scope. |
| Record extension | `AttestationRecord.submitted_at_block: Option<u64>` | New field. `#[serde(default, skip_serializing_if = "Option::is_none")]` — old Stage 5 / 5.1 / 7 records on disk deserialise cleanly to `None`. Pinned by `legacy_json_without_submitted_at_block_loads_with_none`. |
| Registry method | `AttestationRegistry::mark_submitted_with_block(id, receipt, current_block)` | Sibling of `mark_submitted`; both go through a private `mark_submitted_inner(.., Option<u64>)`. Legacy `mark_submitted` keeps its signature and **clears** any prior `submitted_at_block` on retry (no stale height can persist across submissions). Pinned by `mark_submitted_clears_prior_submitted_at_block_on_retry`. |
| Receipt + height preservation on Drop | `mark_stale_if_overdue` path | Preserves `receipt` and `submitted_at_block` on the dropped record so the next retry (`mark_submitted_with_block`) has the prior submission's traceability available. Pinned by `mark_stale_if_overdue_preserves_receipt_and_submitted_at_block_on_drop`. |
| Block height source | Caller-driven; no `ChainClient` trait change | The trait is unchanged at Stage 5.2 (`submit_attestation` + `query_attestation_status`). Production callers fetch the chain head via their adapter's own helper (for SUM Chain: `SumChainClient::get_block_height(BlockFinality::Latest)`) and pass `current_block: u64` into Stage 5.2's surface. `Latest` is the natural finality token for staleness; `Finalized` lags inclusion and would over-aggressively declare records stale. |

**State transitions — unchanged.** Stage 5.2 only adds a new caller path into the existing `mark_dropped`, which already rejects every non-`Submitted` source. `Pending`, `Included`, `Finalized`, `Failed`, and `Dropped` records remain immune from the staleness writer (silent no-op at the workflow layer).

**Stale predicate, in code:**

```rust
let elapsed = current_block.saturating_sub(submitted_at_block);
elapsed > policy.submitted_threshold_blocks()
```

`saturating_sub` is the height-regression safety. If a momentary chain re-org or a misconfigured caller passes a `current_block` below the stored `submitted_at_block`, the predicate evaluates to `false` (not stale) and a workflow-level `tracing::warn!` surfaces the regression for operator inspection. No typed error: callers iterating over many records shouldn't bail on one funny height.

**Wiring example (operator loop, sketch).** Stage 5.3 lands the
preferred entry points (`submit_attestation_workflow_with_block`,
`poll_attestations_workflow`, `sweep_stale_attestations_workflow`,
`retry_dropped_attestations_workflow`) — see the dedicated section
below. The lower-level Stage 5.2 stitch is preserved here for
reference and for callers that want finer-grained control: the
existing `submit_attestation_workflow` calls the legacy
`mark_submitted` and therefore does **not** stamp
`submitted_at_block`, so Stage 5.2 callers that want staleness
coverage either use the Stage 5.3 helper or hand-stitch insert +
client submit + the block-aware mark themselves:

```rust
use omni_sumchain::{BlockFinality, SumChainClient};
use omni_zkml::{
    mark_stale_if_overdue, AttestationRegistry, ChainClient,
    LocalAttestationStatus, StalenessPolicy,
};

let policy = StalenessPolicy::new(params.finality_depth * 4)?; // caller's choice

// Submit, stamping the chain head as `submitted_at_block`.
let record = registry.insert(attestation.clone())?;
if matches!(
    record.status,
    LocalAttestationStatus::Pending | LocalAttestationStatus::Dropped { .. },
) {
    let head = client.get_block_height(BlockFinality::Latest)?.height;
    let receipt = client.submit_attestation(&attestation)?;
    registry.mark_submitted_with_block(&record.id, receipt, head)?;
}

// Periodically: sweep Submitted records for staleness.
let head_now = client.get_block_height(BlockFinality::Latest)?.height;
for r in registry.list()? {
    let _ = mark_stale_if_overdue(&registry, &r.id, head_now, &policy)?;
}
// Records transitioned to Dropped can be retried by repeating the
// submit block above (insert is idempotent; `mark_submitted_with_block`
// accepts `Dropped` as a source state and stamps the fresh height).
```

**Stage 5.3 now ships `submit_attestation_workflow_with_block`** and
three companion sweep helpers — see the dedicated Stage 5.3 section
below for the recommended operator entry points. New code should
prefer those; the lower-level stitched sequence above remains useful
for custom orchestration that needs to do something between the
height fetch and the submit, or wants to interleave submits with other
chain calls.

**Tests** (default-on, hermetic):

- `staleness::tests` — 26 tests covering:
    * Policy construction (`policy_new_rejects_zero_threshold`, `policy_new_accepts_one_and_above`).
    * Pure `is_record_stale`: 5 negative-status branches (Pending / Included / Finalized / Failed / Dropped), 2 missing-or-zero-elapsed branches, the `>` boundary, true-when-strictly-past, height regression, and the minimum-threshold-one two-block-gap rule.
    * `mark_stale_if_overdue`: drops the stale Submitted, embeds the block-context triple in the reason, preserves `receipt` + `submitted_at_block`, leaves not-stale records alone, walks through the legacy-record + height-regression `tracing::warn!` branches, and silently no-ops on every non-Submitted source state.
    * End-to-end retry hinge — drop via staleness, retry via `mark_submitted_with_block` with a fresh height, sweep again at a fresh height that's within the new threshold.
- `registry::tests` — +5 persistence tests: `mark_submitted_does_not_set_submitted_at_block`, `mark_submitted_with_block_persists_submitted_at_block`, `mark_submitted_with_block_from_dropped_retry_records_new_height`, `mark_submitted_clears_prior_submitted_at_block_on_retry`, `legacy_json_without_submitted_at_block_loads_with_none`.
- Total: `cargo test -p omni-zkml` now reports 120 lib tests passing (up from 89 at the end of Stage 5.1 / Stage 6).

**What Stage 5.2 deliberately does not do:**
- No live polling loop, scheduler, retry backoff, or automatic resubmission. Stage 5.2 ships the *surface*; operators wire the cadence.
- No `ChainClient` trait extension — the trait stays at `submit_attestation` + `query_attestation_status`. `SumChainClient::get_block_height` already exists as an inherent method.
- No mainnet config, no hardcoded `chain_id`, no hardcoded finality depth, no hardcoded block-time assumptions.
- No chain-side `Dropped` state introduced anywhere.
- No edits to Stage 6 chain-wire code or fixtures.
- No edits to Stage 7b transaction construction / signing / submit path; `omni-sumchain` is untouched.
- No reward / slash / dispute logic, no async client changes.
- No edits to `omni-types` / `omni-store` / `omni-net` / `omni-pipeline` / `omni-bridge` / `python/omninode`.

---

### Phase 5 Stage 6: Chain Wire Fixture & Signing-Spec Deliverables — Complete

**Crate:** `omni-zkml` | **Depends on:** Stage 4 (for `DOMAIN_TAG`), `libp2p-identity` Ed25519, `bs58`

Stage 6 produces the **frozen signing spec** and **three deterministic test vectors** the chain team needs to build their `InferenceAttestation` adapter. It implements the v1 chain payload shape locally and exercises the full pipeline end-to-end. **No live RPC, no outer transaction encoding, no tokenomics.**

```
InferenceCommitment
  └─► commitment_to_chain_digest   ─► InferenceAttestationDigest
        └─► canonical_digest_bytes      = bincode(digest)
              └─► signing_input_bytes   = DOMAIN_TAG.as_bytes() || canonical_digest_bytes
                    └─► sign_chain_attestation_digest(seed, &digest) ─► [u8; 64]
```

| Concern | Where | Notes |
|---|---|---|
| Chain v1 wire structs | `omni_zkml::{InferenceAttestationDigest, InferenceAttestationTxData}` | Field order is bincode order; declaration order **frozen** for v1. `[u8; 32]` × 4 in the digest; `[u8; 64]` signature field on tx-data via a local `serde_signature_64` helper (no `serde-big-array` dep added). |
| Strict conversion | `commitment_to_chain_digest(&InferenceCommitment)` | Bare 64-char lowercase hex on both string hashes; 256-byte byte-length cap on `session_id`; SNIP roots as `[u8; 32]`. |
| Canonical bytes | `canonical_digest_bytes(&digest)` | **bincode 1.3** `bincode::serialize` — matches the chain team's reference implementation. Imported here via the crate-local renamed alias `bincode1 = { package = "bincode", version = "1.3" }`; the workspace `bincode = "2.0.0-rc.3"` is unchanged and still used by Stage 4 and by other crates. The encoder-significant difference is the string-length prefix: bincode 1.3 writes an 8-byte little-endian `u64` before UTF-8 bytes (bincode 2.0 standard writes a 1-byte varint for short strings). Fixed-size `[u8; N]` arrays encode identically (raw N bytes, no prefix). |
| Signing input | `signing_input_bytes(&digest)` | Exactly `DOMAIN_TAG.as_bytes() ‖ canonical_digest_bytes(&digest)` — the literal bytes of Stage 4's `"omninode.inference_attestation.v1"` constant. |
| Ed25519 signing | `sign_chain_attestation_digest(&seed, &digest) -> [u8; 64]` | Via `libp2p-identity::ed25519` (already a workspace dep). Returns the raw 64-byte signature. Verified deterministically using `RFC 8032` Ed25519: same seed + same digest → bit-identical signature. |
| Chain address | `derive_chain_address_base58(&pubkey)` / `signer_chain_address_base58(&seed)` | Implements the chain's exact rule: `BLAKE3(pubkey)[12..32] ‖ BLAKE3(BLAKE3(addr))[0..4]` → 24-byte payload → `bs58::encode(...)`. **No libp2p PeerId** in the chain-wire surface. |
| Local error type | `omni_zkml::{ChainWireError, ChainWireResult<T>}` | Four typed variants: `InvalidHex { field, reason }`, `SessionIdTooLong { got, max }`, `Signing`, `Serialization`. Fourth distinct result alias in `omni-zkml`. |

**Compatibility with Stage 4.** Stage 4's [`attestation.rs`](crates/omni-zkml/src/attestation.rs) is **byte-stable and unchanged**. `CommitmentDigest`, `compute_canonical_bytes`, `compute_digest`, `Signer`, and `build_attestation` all behave exactly as before. The Stage-4 local-digest pipeline (`CommitmentPayload { domain, commitment }` → bincode 2.0-rc.3 → BLAKE3) and the Stage-6 chain pipeline (`DOMAIN_TAG ‖ bincode 1.3 serialize(InferenceAttestationDigest)` → Ed25519) produce **different** byte sequences for the same `InferenceCommitment` by design. Stage 4 keeps the workspace bincode 2.0-rc.3 in `attestation.rs`; chain-wire uses bincode 1.3 via a renamed crate-local alias (`bincode1`). The two encoders intentionally diverge — chain-wire matches what the chain team's reference implementation uses. Both share the same versioned `DOMAIN_TAG` so a future version bump propagates to both consistently.

**Deliverable for the chain team.** [crates/omni-zkml/tests/fixtures/chain_attestation_vectors.json](crates/omni-zkml/tests/fixtures/chain_attestation_vectors.json) — three vectors with ASCII-only `session_id` strings (`"omninode-stage6-vec-1"`, `"omninode-stage6-vec-2"`, `"omninode-stage6-vec-3-abcdef-0123456789"`) and 32-byte fields rendered as one byte repeated 32 times (e.g. `"00…00"`, `"11…11"`, etc.). Each vector carries:

```json
{
  "session_id":             "...",
  "model_hash":             "<64 lowercase hex chars = 32 bytes>",
  "manifest_root":          "<64 lowercase hex chars = 32 bytes>",
  "response_hash":          "<64 lowercase hex chars = 32 bytes>",
  "proof_root":             "<64 lowercase hex chars = 32 bytes>",
  "verifier_ed25519_seed":  "<64 lowercase hex chars = 32 bytes>",
  "canonical_digest_bytes": "<lowercase hex, variable>",
  "signing_input_bytes":    "<lowercase hex, variable>",
  "signature_bytes":        "<128 lowercase hex chars = 64 bytes>",
  "signer_address_base58":  "<chain checksum-base58>",
  "signer_pubkey_hex":      "<64 lowercase hex chars = 32 bytes>"
}
```

The integration test at [crates/omni-zkml/tests/chain_attestation_vectors.rs](crates/omni-zkml/tests/chain_attestation_vectors.rs) re-derives all three vectors via the actual implementation path and asserts byte-equality against the committed JSON on every `cargo test` run. `OMNINODE_REGEN_VECTORS=1 cargo test -p omni-zkml --test chain_attestation_vectors` overwrites the fixture when intentionally bumping the spec.

**Spot-checks confirming the wire format:**
- Vector 1's `signer_pubkey_hex` is `3b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29` — exactly the RFC 8032 Ed25519 test-vector-1 public key for seed `0x00…00`. Confirms `libp2p-identity::ed25519` treats the seed as the standard Ed25519 secret seed.
- `canonical_digest_bytes` begins with `15 00 00 00 00 00 00 00` (= 21 as an 8-byte little-endian `u64`) for vector 1 and `27 00 00 00 00 00 00 00` (= 39) for vector 3 — the bincode 1.3 fixed-width length prefixes for the respective `session_id` strings. If this were bincode 2.0 standard, the prefix would be a single `15` / `27` byte.
- `signing_input_bytes` begins with `6f6d6e…7631` = ASCII `"omninode.inference_attestation.v1"`, the literal bytes of `DOMAIN_TAG`.

**Chain-address invariants — pinned by `derive_chain_address_uses_last_20_bytes_of_blake3_and_correct_checksum`:**
- Decoded base58 payload is exactly 24 bytes.
- Split is 20 (address) + 4 (checksum).
- Checksum equals `BLAKE3(BLAKE3(address_bytes))[..4]`.
- Address equals `BLAKE3(pubkey)[12..32]` (last 20 bytes, **positive assertion**).
- Address does **not** equal `BLAKE3(pubkey)[0..20]` (first 20 bytes, **explicit negative assertion**).

**Dependencies added** (one new workspace crate; both omni-zkml entries reference workspace-declared crates):
- Root `Cargo.toml` `[workspace.dependencies]` — added `bs58 = "0.5"`. The chain's address encoding requires Bitcoin-base58 over a manually-checksummed 24-byte payload; no existing workspace dep provides that alphabet.
- `crates/omni-zkml/Cargo.toml` — added `bs58 = { workspace = true }` and `libp2p-identity = { workspace = true }` (no speculative `features = [...]`).

**What Stage 6 deliberately does not do:**
- No live chain RPC, no `sum_sendRawTransaction`, no JSON-RPC client.
- No outer `SignedTransaction` encoding — `InferenceAttestationTxData` exists for type completeness only.
- No nonce, fee, gas, or mempool handling.
- No chain activation gate, no reward / slash / dispute logic, no tokenomics.
- No real `Signer` impl — `sign_chain_attestation_digest` is a standalone free function, not a Stage-4 `Signer` impl.
- No libp2p `PeerId` derivation in the chain-wire surface (identity-binding is a future-stage concern).
- No edits to Stage 4 `attestation.rs`, Stage 5 `registry.rs` / `chain.rs`, or anywhere outside `omni-zkml` (other than the one-line `bs58` workspace declaration).
- No SNIP V1, no Private V2, no range reads, no edits to `omni-net` / `omni-pipeline` / `omni-bridge` / `python/omninode`.

---

### Phase 5 Stage 7a: SUM Chain Adapter — Complete

**Crate:** `omni-sumchain` (new) | **Depends on:** `omni-zkml` (Stage 5/6 surface), `ureq` (sync HTTP)

Stage 7a ships the SUM Chain `ChainClient` adapter as a read/query implementation. Real RPC calls go through `ureq` against the local-mirror endpoint; default `cargo test` is fully hermetic via a `FakeJsonRpcTransport`. At Stage 7a the submit path landed as a typed `ChainClientError::Other(_)` stub with the chain-confirmed construction sequence documented in [crates/omni-sumchain/src/tx.rs](crates/omni-sumchain/src/tx.rs); the real implementation lands in Stage 7b (next section).

| Concern | Where | Notes |
|---|---|---|
| Adapter trait impl | `omni_sumchain::SumChainClient` | Generic over `T: JsonRpcTransport`; default `T = UreqTransport`. Implements `omni_zkml::ChainClient` (Stage 5). |
| Transport seam | `omni_sumchain::JsonRpcTransport` trait; `UreqTransport` (production) and `FakeJsonRpcTransport` (clonable `Arc<Mutex<_>>` test fixture) | Default tests never spawn HTTP. |
| Read RPCs implemented | `query_attestation_status(tx_id: &str)` (trait method), plus inherent `query_attestation_status_full`, `get_attestation`, `list_attestations`, `get_chain_params`, `get_block_height(BlockFinality)`, `get_nonce`, `omninode_is_active` | All six chain RPCs (status, attestation read, attestation list, chain params, block height, nonce) reachable. |
| Read DTOs | `omni_sumchain::{InferenceAttestationStatusInfo, InferenceAttestationInfo, BlockHeightInfo, BlockFinality, ChainParamsInfo}` | Owned by this crate; hex fields are `0x`-prefixed (as chain emits). `ChainParamsInfo` carries `finality_depth`, `min_fee`, `chain_id`, and an `#[serde(default)] omninode_enabled_from_height: Option<u64>` for forward-compat with the pending chain follow-up patch. |
| Status mapping | `omni_sumchain::map_status_info` | Strict matching against the five chain-confirmed lowercase variants: `"submitted" \| "included" \| "finalized" \| "failed" \| "unknown"`. Anything else is rejected (no silent fallback to `Unknown`). |
| Submit path (Stage 7a only) | `SumChainClient::submit_attestation` → typed `ChainClientError::Other(_)` | Delegated to `tx::build_and_submit_signed_transaction` so Stage 7b's body change was localised to one function. The Stage 5.1 workflow handles the error gracefully — record stays at `Pending`. **Superseded by Stage 7b's real implementation.** |
| `omninode_is_active` | Single body works pre-patch and post-patch | Pre-patch (`omninode_enabled_from_height` missing → parses as `None`): returns `Ok(false)`. Post-patch with `Some(h)` and `head >= h`: returns `Ok(true)`. No code change needed when the chain ships the follow-up patch. |

**Stage 5.1 contract preserved (Stage 7a integration):** chain `Unknown` → local record unchanged + `tracing::warn!`. Chain-RPC failures → `RegistryError::ChainClient(_)` with the record at its prior state. Tx-id keying flows from `SubmissionReceipt::tx_id` (Stage 5.1) through `query_attestation_workflow` into `ChainClient::query_attestation_status(&str)`.

**Operational setup for live tests** (documented in [crates/omni-sumchain/README.md](crates/omni-sumchain/README.md)):
1. Operator generates an Ed25519 seed locally (never committed).
2. Derives verifier address via Stage 6 `omni_zkml::signer_chain_address_base58`.
3. Funds the address by editing the chain's `extra-alloc.json` before first `docker-compose up`.
4. Live read tests: `OMNINODE_SUMCHAIN_RPC_URL=http://localhost:8545 cargo test -p omni-sumchain -- --ignored`.
5. Live tests **auto-skip** when the env var is unset, so CI invocations exit 0.

Documented local-mirror defaults: `http://localhost:8545`, `chain_id: 31337`, `min_fee: 1`. Final confirmation waits for the rebuilt local-mirror branch announcement.

**Dependencies added** (one new workspace crate):
- Root `Cargo.toml` `[workspace.dependencies]` — added `ureq = "2.10"` (sync HTTP). Used by `omni-sumchain` only. The `omni-sumchain` package additionally takes `omni-types`, `omni-zkml`, `serde`, `serde_json`, `thiserror`, `tracing` from the workspace (all pre-existing); `tempfile` as a `[dev-dependencies]` entry for the integration tests.

**Tests** (default-on, hermetic):
- `unit_status_mapping.rs` — 10 tests pinning every chain status variant + every negative case (uppercase rejected, unknown variant rejected, `failed` without `reason` errors, chain never returns `Dropped`).
- `unit_dto.rs` — 12 tests pinning DTO deserialisation for every read RPC, including the pre-patch and post-patch shapes of `ChainParamsInfo`.
- `unit_rpc_envelope.rs` — 16 tests asserting JSON-RPC method names and `params` arrays for every read helper, a `submit_attestation` happy-path receipt assertion (Stage 7b replaced the Stage 7a stub-error test with this end-to-end one), `omninode_is_active`'s three branches, and two Stage 5.1 integration tests (`Unknown` leaves the record unchanged; RPC errors propagate as `RegistryError::ChainClient(_)` without modifying state).
- `stage6_wire_parity.rs` — 1 cross-crate smoke that consumes the committed Stage 6 chain-team deliverable fixture from `omni-zkml/tests/fixtures/` and asserts byte-equal output.
- `live_local_mirror.rs` — 4 `#[ignore]`'d tests gated by `OMNINODE_SUMCHAIN_RPC_URL`; auto-skip when unset.

**What Stage 7a deliberately did not do** (delivered in Stage 7b, see next section):
- No real submit RPC call (`submit_attestation` was a typed stub at Stage 7a).
- No outer `SignedTransaction` construction at Stage 7a.
- No mainnet endpoint, no mainnet `chain_id`.
- No edits to `omni-zkml`, `omni-types`, `omni-store`, `omni-net`, `omni-pipeline`, `omni-bridge`, or `python/omninode`.
- No edits to `omni-zkml/Cargo.toml`.
- No staleness/timeout detection (Stage 5.2).
- No real cryptographic signing — Stage 4's `Signer` trait is unchanged.

---

### Phase 5 Stage 7b: SUM Chain Submit Path — Complete

**Crate:** `omni-sumchain` (extends Stage 7a) | **Depends on:** vendored `sumchain-primitives` + `sumchain-crypto` at chain rev `d83e45a4`

Stage 7b ships the real `submit_attestation` implementation. The Stage 5 `ChainClient` trait surface is unchanged; only the body in [crates/omni-sumchain/src/tx.rs](crates/omni-sumchain/src/tx.rs) flips from the typed-error stub to the chain-confirmed construction flow. All chain-side primitives (`TransactionV2`, `TxPayload`, `SignedTransaction`, `Address`) are pulled in via git deps at the chain-team-pinned rev; the local Stage 6 `InferenceAttestationDigest` and `InferenceAttestationTxData` remain the canonical OmniNode-side byte source, with byte-parity to the vendored types proven by three default-on tests in [tests/parity_vendored_primitives.rs](crates/omni-sumchain/tests/parity_vendored_primitives.rs).

| Concern | Where | Notes |
|---|---|---|
| Outer-tx construction | [src/tx.rs](crates/omni-sumchain/src/tx.rs) `build_and_submit_signed_transaction` | 11-step flow: gates → Stage 6 inner → local→vendored conversion → `TransactionV2` → outer-sign → bare-hex submit. |
| Outer signing | [src/outer_sign.rs](crates/omni-sumchain/src/outer_sign.rs) `outer_sign_transaction_v2` | Wraps `sumchain_crypto::sign(TransactionV2::signing_hash().as_bytes(), &PrivateKey)`. Outer signs the **32-byte BLAKE3 hash**, not the raw canonical bytes (differs from Stage 6's inner pattern). Replay protection is via the `chain_id` field of `TransactionV2`, not a domain tag. |
| Activation gates | `omninode_is_active()` and `v2_is_active()` inherent on `SumChainClient` | Both required `true` before `sum_getNonce` / `sum_sendRawTransaction` reaches the chain. Pre-patch (either activation field is `None`) returns `Ok(false)`. |
| Verifier-address consistency | Pre-flight gate in `tx::build_and_submit_signed_transaction` | `omni_zkml::signer_chain_address_base58(&seed)` must equal `attestation.verifier_address`; on mismatch returns typed `ChainClientError::Other(_)` and **the chain is never reached**. |
| Cached RPC reads | One `chain_getChainParams` call + at most one `chain_getBlockHeight` call per submit | Cached across both activation gates and the `TransactionV2` build. Pinned by `submit_attestation_calls_chain_get_chain_params_exactly_once` and `submit_attestation_calls_chain_get_block_height_at_most_once`. |
| Fee | `params.min_fee as u128` | Unconditional; no override path in the trait. `Balance` is `u128` on the chain; the `u64` DTO field is widened at construction time. |
| `ChainParamsInfo` gains | `#[serde(default)] v2_enabled_from_height: Option<u64>` | Symmetric to `omninode_enabled_from_height`; parser forward-compat with pre-patch mirrors that don't emit it. |
| Submission encoding | `signed_tx.to_hex()` returns **bare hex** (no `0x` prefix); chain accepts either | Pinned by `submit_attestation_passes_bare_hex_to_send_raw_transaction`. |
| Response propagation | Chain emits `{ "tx_hash": "0x..." }` (canonical) and the `0x`-prefixed `tx_hash` field flows verbatim into `SubmissionReceipt::tx_id` | Bare-string responses are also accepted as a backwards-compat fallback. Object form was confirmed via the Stage 7b live submit roundtrip against `sum-chain @ b586ff3f`. Stage 5.1's registry sees the chain-canonical tx-id and can later query against it via `query_attestation_status`. |

**Stage 5.1 contract preserved.** A submit failure (gate, RPC, or signing) propagates from `submit_attestation_workflow` as `RegistryError::ChainClient(_)` and leaves the local record at `Pending` (or `Dropped` for retry). The workflow never terminalises a record on a single submit failure.

**Local-mirror activation.** Stage 7b is gated at runtime on the chain mirror exposing **both** `omninode_enabled_from_height` and `v2_enabled_from_height` as `Some(_)`. The chain-team-confirmed local-mirror branch `snip-local-mirror-omninode @ b586ff3f` sets both to `0` (activation-from-genesis) on `chain_id = 31337`.

**Chain primitives** (as shipped at Stage 7b):
- `sumchain-primitives = { git = "https://github.com/SUM-INNOVATION/sum-chain", rev = "d83e45a4", package = "sumchain-primitives" }`
- `sumchain-crypto = { git = "...", rev = "d83e45a4", package = "sumchain-crypto" }`

> **Superseded by Stage 9c.** The chain team has since published both crates to the public crates.io index as `v0.1.0` (dual-licensed MIT OR Apache-2.0, byte-equivalent to rev `d83e45a4` for the InferenceAttestation surface). No GitHub credential is required for any build — `cargo fetch` / `cargo build` / `cargo test` resolve from crates.io alone. See the **Phase 5 Stage 9c** section below for the full migration. The text that follows describes the original Stage 7b state for historical record.

At Stage 7b, the chain repo was private. First `cargo fetch` / `cargo build` from a fresh clone needed `CARGO_NET_GIT_FETCH_WITH_CLI=true` plus GitHub credentials (PAT in credential helper, SSH key, or `gh auth login`) with read access to `SUM-INNOVATION/sum-chain`. CI needed an equivalent deploy key or PAT. Stage 9c removed this requirement entirely.

**Live-test setup additions.** Stage 7b's live submit roundtrip needs:
1. `OMNINODE_SUMCHAIN_RPC_URL=http://localhost:8545` (Stage 7a).
2. **`OMNINODE_VERIFIER_SEED_HEX=<64 hex chars>`** — the operator's Ed25519 seed.
3. The address derived from that seed must be pre-funded via the chain's `extra-alloc.json` before `docker-compose up`.

Live tests self-skip cleanly when either env var is unset.

**What Stage 7b deliberately does not do:**
- No mainnet endpoint, no mainnet `chain_id` assumption (live submit asserts `chain_id == 31337` as a guardrail).
- No retry / backoff machinery — a single submit failure surfaces as one error; the Stage 5.1 workflow handles resume.
- No fee override path on the trait — `submit_attestation(&InferenceAttestation)` has no fee parameter; `params.min_fee` is used unconditionally. A future stage may add an inherent `submit_attestation_with_fee(_, fee: u64)` if needed.
- No edits to Stage 4 / Stage 5 / Stage 6 source. Stage 6 chain-attestation-vectors fixture is byte-stable.
- No edits to `omni-zkml/Cargo.toml`.
- No staleness/timeout detection (Stage 5.2 — separate).

---

### Phase 5 Stage 5.3: End-to-End Attestation Orchestration — Complete

**Crate:** `omni-zkml` (new `orchestration` module) + `omni-sumchain` (additive `OrchestrationClient` impl on `SumChainClient`) | **Depends on:** Stage 5 / 5.1 / 5.2 / 6 / 7b — no new external dependencies

Stage 5.3 stitches every prior Phase 5 surface into one operator-facing module. Submit, poll, sweep-for-staleness, and retry now compose through four free functions over a sibling trait `OrchestrationClient: ChainClient` that adds exactly **one** method — `get_latest_block_height` — so the chain protocol surface (`submit_attestation` + `query_attestation_status`) is unchanged. `SumChainClient` opts into Stage 5.3 via an additive trait impl that wraps the existing inherent `get_block_height(BlockFinality::Latest)` helper.

| Helper | Source-state filter | Per-record RPC budget | Aggregate RPC budget | Where |
|---|---|---|---|---|
| `submit_attestation_workflow_with_block(reg, client, attestation)` | `Pending`, `Dropped` (insert is idempotent) | 1× height + 1× submit | (single record) | [crates/omni-zkml/src/orchestration.rs](crates/omni-zkml/src/orchestration.rs) |
| `poll_attestations_workflow(reg, client)` | `Submitted`, `Included` | 1× query | 1× per queryable record | same |
| `sweep_stale_attestations_workflow(reg, client, policy)` | `Submitted` | 0 (pure registry write on stale) | 1× height up-front, shared | same |
| `retry_dropped_attestations_workflow(reg, client)` | `Dropped` | 1× height + 1× submit | 1× height + 1× submit **per** dropped record | same |

**Invariants preserved (pinned by hermetic tests):**

- **Stage 5 idempotency.** `submit_attestation_workflow_with_block` calls `insert(attestation.clone())` first. An existing `Submitted` / `Included` / `Finalized` / `Failed` record returns unchanged with **zero** chain calls. A byte-different attestation under the same `(session_id, verifier_address)` key surfaces as `RegistryError::ConflictingAttestation` without reaching the chain. Pinned by `submit_with_block_is_noop_for_already_submitted_record`.
- **Height before submit.** The block-aware submit fetches `get_latest_block_height` **before** `submit_attestation`, so the `submitted_at_block` stamp reflects the chain head as seen by the chain at submit time, not after a multi-second submit RPC. Pinned by `submit_with_block_fetches_height_before_submit`.
- **Stage 5.1 `Unknown` is observation-only.** `poll_attestations_workflow` delegates to `query_attestation_workflow` per record; chain-returned `Unknown` leaves records unchanged and the sweep continues. Pinned by `poll_preserves_unknown_as_observation_only`.
- **Stage 5.1 RPC-failure containment in sweeps.** Per-record chain failures land as `Err` entries in the returned vec; the rest of the sweep continues; the failing record is **not** mutated. Pinned by `poll_per_record_failure_does_not_abort_sweep` and `retry_per_record_failure_does_not_abort_sweep`.
- **Stage 5.2 single-`Dropped`-writer rule.** Only `sweep_stale_attestations_workflow` writes `Submitted → Dropped`, and only via `mark_stale_if_overdue` (which itself only fires when the caller-constructed `StalenessPolicy` says so). Non-`Submitted` records are skipped (omitted from the sweep result vec).
- **Stage 5.2 height-source story.** The sibling trait exposes **only** `Latest` finality; `Finalized` lags inclusion and would over-aggressively declare records stale. Pinned by `sum_chain_client_get_latest_block_height_calls_chain_get_block_height_latest`.

**Sweep error model.** All three sweeps return `Vec<(AttestationId, RegistryResult<AttestationRecord>)>`. Records skipped by the source-state filter are **omitted** entirely (Q3 from the plan: operators wanting the full registry call `registry.list()`). The staleness sweep's up-front `get_latest_block_height` is fail-fast — without a reference height there's nothing to compare against, so the whole sweep returns `Err(RegistryError::ChainClient(_))` and no records are touched. Pinned by `staleness_sweep_height_failure_aborts_without_touching_records`.

**Wiring example (operator loop, replaces the lower-level Stage 5.2 sketch):**

```rust
use omni_sumchain::SumChainClient;
use omni_zkml::{
    poll_attestations_workflow, retry_dropped_attestations_workflow,
    submit_attestation_workflow_with_block, sweep_stale_attestations_workflow,
    AttestationRegistry, StalenessPolicy,
};

let policy = StalenessPolicy::new(params.finality_depth * 4)?; // caller's choice
let registry = AttestationRegistry::open("./attestations".into())?;
let client = SumChainClient::new(rpc_url, seed);

// 1. Submit new attestations (idempotent; stamps `submitted_at_block`).
let _ = submit_attestation_workflow_with_block(&registry, &client, attestation)?;

// 2. Periodically: reconcile chain-side state into the registry.
let polled = poll_attestations_workflow(&registry, &client)?;
for (id, result) in polled {
    if let Err(e) = result {
        tracing::warn!(id = %id, error = ?e, "poll failed; will retry next tick");
    }
}

// 3. Mark records that have aged out as locally Dropped.
let _ = sweep_stale_attestations_workflow(&registry, &client, &policy)?;

// 4. Resubmit anything in Dropped (fresh height stamp per retry).
let _ = retry_dropped_attestations_workflow(&registry, &client)?;
```

The four helpers compose freely — there is no required order, no retry cap, no backoff, and no scheduler. Operators pick the cadence.

**Tests** (default-on, hermetic):

- `omni_zkml::orchestration::tests` — 22 tests, all using a single in-module `FakeOrchestrationClient` that implements both `ChainClient` and `OrchestrationClient` with configurable per-call outcomes and a `Call` log for ordering / count assertions. Covers: 6 submit-with-block tests, 5 poll tests, 5 staleness-sweep tests, 5 retry tests, and 1 end-to-end lifecycle test (submit → poll → simulated chain forget → poll(Unknown) → staleness drop → retry → poll(Finalized)).
- `omni_sumchain::tests::unit_rpc_envelope` — +1 test confirming `SumChainClient::get_latest_block_height` posts `chain_getBlockHeight(["latest"])` and propagates the `height` field. Total in that file: 17 (was 16).
- Aggregate: `cargo test -p omni-zkml` reports **142 lib tests** (up from 120) + 1 fixture integration + 1 doc-test. `cargo test -p omni-zkml -p omni-sumchain` is fully green; the five `#[ignore]`'d `omni-sumchain` live tests continue to self-skip cleanly.

**What Stage 5.3 deliberately does not do:**
- No live polling loop, scheduler, retry backoff, or max-attempts cap. Operators wire the cadence.
- No `ChainClient` trait change. `OrchestrationClient` is a sibling that extends it via supertrait; existing Stage 5/5.1 fakes continue to compile untouched.
- No async. Everything stays sync.
- No new dependencies.
- No mainnet config. No hardcoded chain id, finality depth, or block time.
- No tokenomics / reward / slash / dispute logic — Stage 5.3 is glue, not policy.
- No edits to Stage 6 chain-wire or Stage 7b submit construction.
- No live-test additions; operator-level tests are hermetic against `FakeOrchestrationClient`.
- No edits to `omni-types` / `omni-store` / `omni-net` / `omni-pipeline` / `omni-bridge` / `python/omninode`.

---

### Phase 5 Stage 8a: Operator / Activation Monitoring Loop — Complete

**Crate:** `omni-node` (new `operator` module + nested subcommand) | **Depends on:** Stage 5.3 orchestration helpers + Stage 7a/7b `omni-sumchain` adapter — no new external crates, **zero edits outside `crates/omni-node/`**

Stage 8a turns the Stage 5.3 library helpers into a usable operator command surface on the `omni-node` binary. Three nested subcommands under `omni-node operator`, **safe by default**, no chain-protocol changes.

```
omni-node operator watch-activation --rpc-url <U> --expect-chain-id <N> [--poll-interval-secs 30] [--max-ticks <N>]
omni-node operator smoke            --rpc-url <U> --expect-chain-id <N> --registry-path <P>
                                    [--attestation-json <PATH>] [--synthetic]
                                    --allow-submit [--allow-mainnet-submit] [--confirm-timeout-secs 60]
omni-node operator loop             --rpc-url <U> --expect-chain-id <N> --registry-path <P>
                                    --staleness-threshold-blocks <N> [--poll-interval-secs 30]
                                    [--allow-submit] [--allow-mainnet-submit] [--max-ticks <N>]
```

| Command | What it does | Chain writes |
|---|---|---|
| `watch-activation` | Polls `chain_getChainParams` + `chain_getBlockHeight(Latest)`; logs params + blocks-remaining to each gate; exits `0` once `omninode_is_active() && v2_is_active()`, errors `ActivationNotReached` if `--max-ticks` exhausts first. | None (read-only). |
| `smoke` | Submits exactly one attestation, polls it to **`Finalized`** (`Included` is logged as progress, never treated as success) or errors `SmokeConfirmTimeout` at `--confirm-timeout-secs`. | One submit (gated). |
| `loop` | Per tick: `poll_attestations_workflow` → `sweep_stale_attestations_workflow` → (only if submit permitted **and** chain activated) `retry_dropped_attestations_workflow`. `tokio::select!` Ctrl-C graceful shutdown; `--max-ticks` bounds CI runs. | Retry submits only (gated); monitor-only without `--allow-submit`. |

**Mainnet smoke safety boundary (the load-bearing rule).** `operator smoke` resolves its attestation source through a pure, seed-free gate evaluated **before** the verifier seed is ever read:

- `chain_id == 1` (mainnet) **requires** `--attestation-json <path>` (deserialized to `InferenceAttestation`, submitted verbatim). There is **no** synthetic-mainnet path and no danger-flag to create one in Stage 8a.
- `--synthetic` is rejected outright on `chain_id == 1`.
- Synthetic generation is non-mainnet only and requires explicit `--synthetic` (no silent synthesis even on dev chains).
- `--attestation-json` + `--synthetic` together is a typed conflict error.

The gate ordering is asserted by hermetic tests that pass an **absent or malformed** seed and confirm the mainnet refusal still fires (`smoke_mainnet_without_attestation_json_is_refused_before_seed`, `smoke_mainnet_rejects_synthetic_before_malformed_seed`) — a missing/garbage `OMNINODE_VERIFIER_SEED_HEX` can never mask the real mainnet-misuse error.

**Runtime safety gates (defense-in-depth; smoke order):** chain-id guardrail → attestation-source/no-synthetic-mainnet (seed-free) → activation precheck (`omninode_is_active && v2_is_active`) → submit opt-in (`--allow-submit`) + mainnet double-gate (`--allow-mainnet-submit` when `chain_id == 1`) → seed resolution + verifier-address consistency → Stage 7b adapter-layer activation/verifier gates (unchanged backstop). The operator never even constructs a tx before activation; the Stage 7b adapter independently refuses too.

**Config / env.** `OMNINODE_SUMCHAIN_RPC_URL` (or `--rpc-url`); `OMNINODE_VERIFIER_SEED_HEX` (64 hex, only read by submit paths, only *after* the mainnet gate); `--expect-chain-id` required on every operator chain command (fail-loud if pointed at the wrong endpoint); `--staleness-threshold-blocks` passed verbatim to `StalenessPolicy::new` (rejects 0). Blocking SUM Chain RPC runs in `tokio::task::spawn_blocking` with an `Arc`'d client factory so no borrow escapes.

**Live / mainnet runbook.** (1) export `OMNINODE_SUMCHAIN_RPC_URL`; (2) `operator watch-activation --expect-chain-id 1` — logs blocks-remaining to V2 (5,200,000) then OmniNode (6,000,000), exits 0 when both live; (3) sanity-check logged params against expected mainnet values (chain_id 1, finality_depth 6, min_fee 1000); (4) export `OMNINODE_VERIFIER_SEED_HEX` for a funded verifier; (5) `operator smoke --expect-chain-id 1 --registry-path <P> --attestation-json ./first.json --allow-submit --allow-mainnet-submit` — submits a **real** operator-provided attestation, polls to `Finalized`; (6) `operator loop --expect-chain-id 1 --registry-path <P> --staleness-threshold-blocks <finality_depth*K> --allow-submit --allow-mainnet-submit`. Pre-activation only step 2 runs; steps 5–6 self-refuse until head ≥ 6,000,000.

**Tests** (default-on, hermetic; 20 in `omni_node::operator::tests`): pure gate matrices (`check_chain_id`, `submission_permitted`, `blocks_remaining`, `resolve_smoke_source`); `watch-activation` active/not-reached/chain-id-mismatch-before-height; the two gate-7-before-seed guarantees; non-mainnet synthetic requires explicit flag; attestation-json verifier mismatch + malformed file refused with zero submit calls; mainnet with valid attestation-json proceeds to finalized; mainnet without `--allow-mainnet-submit` refused; **smoke treats `Included` as progress not success — times out if the chain stays `Included`, succeeds only once `Finalized` (sequenced included→finalized via the shared fake)**; `loop` monitor-only never submits, submit-mode retries a seeded `Dropped` record exactly once, mainnet-without-flag fails at startup, zero staleness threshold rejected. Runners are exercised against `SumChainClient<FakeJsonRpcTransport>` — no live chain.

**What Stage 8a deliberately does not do:** no daemonization/systemd; no metrics backend (tracing-only); no scheduler beyond fixed-interval sleep; no `--allow-synthetic-mainnet-smoke` escape hatch; no `ChainClient`/`OrchestrationClient` trait changes; no async rewrite of `SumChainClient`; no keystore/multi-account; no proof/tokenomics; no edits to Stage 6 chain-wire/fixture, Stage 7b tx construction, or any crate outside `omni-node`.

---

### Phase 5 Stage 8b: Mainnet Activation Smoke + Operator Hardening — Complete

**Crate:** `omni-node` (two read-only subcommands added to the `operator` surface) | **Depends on:** Stage 8a — no new external crates, zero edits outside `crates/omni-node/`

Stage 8b de-risks the mainnet activation smoke. The concrete gap Stage 8a exposed: `smoke`'s verifier-address / JSON-parse checks sit at gate 5, **after** the gate-2 activation precheck — so pre-activation there was no way to confirm "the verifier seed derives to the JSON's `verifier_address`" until *after* mainnet activates, the worst possible time to discover a mismatch. Stage 8b adds a read-only preflight that validates the pairing decoupled from activation, plus a read-only chain query for post-activation verification, plus a checklist-driven runbook.

| Command | What it does | Chain writes |
|---|---|---|
| `operator preflight --attestation-json <P> [--rpc-url <U>] [--expect-chain-id <N>]` | Resolves the verifier seed, parses the attestation JSON, asserts `signer_chain_address_base58(seed) == attestation.verifier_address`, prints the `(session_id, verifier_address)` `AttestationId`. With `--rpc-url` (then `--expect-chain-id` required): one-shot chain snapshot — chain-id guardrail, params, head, blocks-remaining, activation status, **report-only** (succeeds whether or not the chain is already activated, Q2). Offline mode (no `--rpc-url`) skips the chain snapshot entirely. | **None** (read-only; never submits, never writes the registry, never requires activation or `--allow-submit`). |
| `operator query --rpc-url <U> --expect-chain-id <N> --session-id <S> --verifier-address <A>` | `sum_getInferenceAttestation(session_id, verifier_address)` behind the chain-id guardrail. Logs the chain's stored attestation when present; returns typed `AttestationNotFound` (non-zero, scriptable) when absent. | **None** (read-only). |

**Decisions:** Q1 yes (both `preflight` + `query` shipped); Q2 report-only when already activated (no error); Q3 documented timeout guidance, default `--confirm-timeout-secs` unchanged; Q4 no funding check in code — preflight prints the derived address, the operator verifies balance externally (no balance RPC in the adapter; adding one would be a forbidden Stage-7-area change).

**Mainnet activation runbook.** Pre-activation (head < 6,000,000):

1. `export OMNINODE_SUMCHAIN_RPC_URL=<mainnet RPC>` and `OMNINODE_VERIFIER_SEED_HEX=<64 hex>`.
2. `omni-node operator preflight --attestation-json ./first-attestation.json --rpc-url $OMNINODE_SUMCHAIN_RPC_URL --expect-chain-id 1` — one shot confirms seed↔JSON verifier match, the de-dup `AttestationId`, `chain_id=1` / `finality_depth=6` / `min_fee=1000` / `v2_enabled_from_height=5200000` / `omninode_enabled_from_height=6000000`, current head (must read `< 6,000,000`), and blocks-remaining. Funding: preflight logs the derived address; verify its balance via your chain explorer (out of band).
3. `omni-node operator watch-activation --expect-chain-id 1` — leave running; exits 0 when both gates live.

At/after activation (head ≥ 6,000,000):

4. `cargo run -p omni-node --features submit -- operator smoke --expect-chain-id 1 --registry-path <P> --attestation-json ./first-attestation.json --allow-submit --allow-mainnet-submit [--confirm-timeout-secs N]`. **Stage 9a:** the `submit` cargo feature is default-off; the binary used in steps 1–3 (read-only) does not need it, but `smoke` does. **Timeout guidance (empirically corrected, Stage 8c):** the first mainnet smoke (2026-05-19, see [docs/mainnet-smoke-audit.md](docs/mainnet-smoke-audit.md)) finalized in ~**11 s** / ~1 block past submit, so the default `--confirm-timeout-secs 60` is comfortably sufficient on SUM Chain mainnet — no need to inflate it. No block time is hardcoded; raise it only if your endpoint or network conditions warrant. Record the **tx hash**.
5. Confirm chain status reaches **`Finalized`** — Stage 8a returns success only on `Finalized`; `Included` is logged as progress.
6. `omni-node operator query --expect-chain-id 1 --session-id <S> --verifier-address <A>` — confirm the chain's stored attestation matches; record `included_at_height`, `tx_hash`, `finalized`.

**Duplicate / idempotency checks (split — they prove different things):**

1. **Local idempotency.** Re-run the *exact same* `smoke` command with the *same* `--registry-path`. Expected: the local record is already `Finalized`, so `submit_attestation_workflow_with_block`'s `insert` short-circuits — **no new chain submission**, the finalized record is reused, smoke re-polls to `Finalized`. This proves the local registry prevents accidental double-submit; it does **not** exercise chain-side duplicate rejection.
2. **Chain duplicate rejection.** Only if intentionally exercising chain rejection: from a fresh/empty registry (or a controlled one-off path that bypasses local idempotency), resubmit the same `(session_id, verifier_address)`. Expected: the chain rejects the duplicate or reports `Failed` with the documented duplicate behavior. **Stage 8b does not ship a bypass-local-idempotency duplicate-submit tool** — this remains a manual/external check or a later explicit diagnostic command.

**Operational sequencing.** Start the lifecycle loop **monitor-only first** (`cargo run -p omni-node -- operator loop …` with no `--allow-submit` and no feature flag) to confirm poll/sweep behave; only after the smoke is clean rebuild with `--features submit` and add `--allow-submit --allow-mainnet-submit` to enable the retry sweep.

**Results template to record:** `tx_hash`, `included_at_height`, finalized height + `finalized` status, `AttestationId`, local-idempotency rerun outcome.

> **Mainnet smoke: PASS (2026-05-19, SUM Chain `chain_id=1`).** First real OmniNode attestation finalized on mainnet — see the operational run record in [docs/mainnet-smoke-audit.md](docs/mainnet-smoke-audit.md). README stays the feature doc; the audit note holds the exact tx hash / heights / finality timing.

**Tests** (default-on, hermetic; **31** total in `omni_node::operator::tests`, +11 over Stage 8a): preflight — offline OK with matching seed (zero chain calls), verifier mismatch refused, malformed JSON, missing seed surfaced, `--rpc-url` requires `--expect-chain-id`, chain-id mismatch refused, happy path reports the snapshot and never submits, report-only when already activated; query — found attestation logged (never submits), not-found returns typed `AttestationNotFound`, chain-id guardrail fires before the attestation read. Every preflight/query path asserts `submit_calls == 0`.

**What Stage 8b deliberately does not do:** no Stage 6/7b changes (no live bug surfaced); no protocol changes; no synthetic mainnet attestation; no bypass-local-idempotency duplicate-submit tool; no daemonization/systemd; no balance RPC / new `omni-sumchain` methods; no `--confirm-timeout-secs` default change; no edits outside `crates/omni-node/`.

---

### Phase 5 Stage 8c: Operator Hardening After First Mainnet Smoke — Complete

**Crate:** `omni-node` | **Depends on:** Stage 8b + the 2026-05-19 mainnet smoke ([docs/mainnet-smoke-audit.md](docs/mainnet-smoke-audit.md)) | zero edits outside `crates/omni-node/`, no new deps

Folds the first real mainnet finalization back into the operator surface: a genuinely warning-clean production build, empirically-corrected runbook timing, and three read-only ergonomics wins.

| Change | Detail |
|---|---|
| Warning-clean build (no suppression) | `SeedSource`'s three test-injection variants (`Explicit`/`AbsentForTest`/`MalformedForTest`) and their `resolve()` match arms are `#[cfg(test)]`-gated, **not** `#[allow(dead_code)]`-suppressed. The production binary contains only `SeedSource::Env`; `cargo build -p omni-node` is genuinely warning-clean (verified). |
| `operator smoke` summary block | `smoke_core` now returns a testable `SmokeOutcome { attestation_id, session_id, verifier_address, tx_hash, submitted_at_block, included_at_height, finalized }`; the CLI renders it as one consolidated `SMOKE SUMMARY` line. `included_at_height` is populated by **one extra read-only** `query_attestation_status_full` call at finalization — submit semantics unchanged (success still gated solely on `Finalized`; `Included` remains progress). Ctrl-C during the poll now returns typed `SmokeInterrupted` rather than a false-success (a partial smoke must never look green). |
| `operator query --tx-hash` | New status-by-tx mode (`sum_getInferenceAttestationStatus`) alongside the Stage 8b `(session_id, verifier_address)` presence mode. Manual mutual-exclusion validation (Q4) with typed `ConflictingQueryMode` / `QueryModeRequired` errors so wording matches the rest of `operator`. Pair mode still returns `AttestationNotFound` when absent; tx-hash mode reports status (incl. `unknown`) and returns Ok — it's an observation, not a presence assertion. Read-only. |
| `operator derive-address` | New read-only subcommand: resolves `OMNINODE_VERIFIER_SEED_HEX` and prints the chain address (bare stdout for scripting + an `info!` context line). No chain access, no attestation file — removes the friction of needing a preflight/attestation just to learn your verifier address. Never echoes the seed. |
| Runbook timing corrected | Stage 8b runbook step 4 timeout guidance replaced with the empirical figure (mainnet finality ~11 s / ~1 block; default `--confirm-timeout-secs 60` is sufficient). |

**Tests** (default-on, hermetic; **37** total in `omni_node::operator::tests`, +6 over Stage 8b): `resolve_query_mode` matrix (tx-hash vs full pair vs conflict vs incomplete vs none); `query --tx-hash` reports status & never submits; tx-hash `unknown` is Ok not error; `derive_address` returns the seed-derived address / surfaces missing seed; `smoke_core` returns a populated `SmokeOutcome` (tx_hash, finalized, `included_at_height` from the extra read, session/verifier). Warning-clean is verified by `cargo build -p omni-node` (not a unit test). Stage 8a/8b tests carried forward (query tests renamed to the `query_pair_*` mode).

**Audit note immutability:** [docs/mainnet-smoke-audit.md](docs/mainnet-smoke-audit.md) records the 2026-05-19 run as a frozen historical record; Stage 8c added only a short "follow-up hardening" pointer at its foot — the smoke facts themselves are unedited.

**What Stage 8c deliberately does not do:** no Stage 6 chain-wire / Stage 7b tx-construction changes (no concrete bug surfaced); no synthetic mainnet support; no daemonization/systemd (first mainnet smoke just passed — observability/UX hardening is the correct increment before a supervised long-running process); no new chain RPCs / `omni-sumchain` changes; no `--confirm-timeout-secs` default change (docs corrected instead); no edits outside `crates/omni-node/`.

---

### Phase 5 Stage 9a: Production Operator Runbook + Build Decoupling — Complete

**Crates:** `omni-node`, `omni-sumchain` (additive feature) | **Depends on:** Stage 8c + the 2026-05-19 mainnet smoke ([docs/mainnet-smoke-audit.md](docs/mainnet-smoke-audit.md)) | no new external crates; zero protocol changes; Stage 6 fixture + Stage 7b tx-construction byte-stable

Two outcomes that make the now-proven operator path *adoptable*:

**1. Submit code path moves behind a `submit` cargo feature.** Stage 9a introduced the feature gate: the `sumchain-primitives` / `sumchain-crypto` deps (then sourced from the private chain repo) are marked `optional = true` and only pulled in by `--features submit`, and the `Smoke` / `loop --allow-submit` / `loop --allow-mainnet-submit` CLI variants are `#[cfg(feature = "submit")]`-gated. The original goal — that `cargo build -p omni-node` (default) compile without ever touching the chain repo — turned out to need more than the feature gate (Cargo still clones git sources at resolution time regardless of optional-ness). **Stage 9c later closed that gap** by moving both crates to the public crates.io index (v0.1.0, MIT OR Apache-2.0); fresh source builds are now credential-free for both feature configurations. See the Stage 9b / Stage 9c sections below for the full arc.

| Build invocation | Pulls chain crates? | Subcommands compiled in |
|---|---|---|
| `cargo build -p omni-node` (default) | **no** (feature off — sumchain-* absent from compile graph) | `watch-activation`, `preflight`, `query` (incl. `--tx-hash`), `derive-address`, `registry list/show`, `loop` (monitor-only) |
| `cargo build -p omni-node --features submit` | yes (sumchain-primitives + sumchain-crypto from crates.io as of Stage 9c) | all of the above plus `smoke`, `loop --allow-submit [--allow-mainnet-submit]` |

Verified empirically (Stage 9a CI gate):

```bash
cargo tree -p omni-node                  | grep -E 'sumchain-(crypto|primitives)'   # (no rows)
cargo tree -p omni-node --features submit | grep -E 'sumchain-(crypto|primitives)'  # sumchain-crypto + sumchain-primitives appear
```

When `submit` is off, `SumChainClient::submit_attestation` still satisfies the `ChainClient` trait but returns a typed `ChainClientError::Other("omni-sumchain built without the `submit` feature; …")` — read-only consumers (poll / query workflows) continue to compile and work generically. `OperatorCmd::Smoke` and the loop's `--allow-submit` / `--allow-mainnet-submit` flags are `#[cfg(feature = "submit")]`-gated at the CLI level (Q2): without the feature they don't appear in `--help` at all (cleaner than a flag that can only fail).

**2. Read-only `operator registry list / show`.** Two new subcommands inspect the local attestation registry without any chain access or mutation:

| Subcommand | Behaviour |
|---|---|
| `operator registry list --registry-path <P> [--status <S>]` | One bare-stdout line per record (`<id-hex>  <status>  tx=…  block=…  updated=…`) for grep-ability. Optional status filter (`pending\|submitted\|included\|finalized\|failed\|dropped`, case-insensitive); typed `InvalidStatusFilter` on an unknown value. |
| `operator registry show --registry-path <P> --id <hex>` | Pretty-prints the full `AttestationRecord` JSON (digest, attestation, receipt, error_message, submitted_at_block). Missing id → typed `RegistryError::RecordNotFound`. Malformed hex → typed `AttestationIdMalformed` (refused before any disk read). |

No mutating registry-repair tools ship in Stage 9a — registry inspection stays strictly read-only.

**3. Runbook.** A canonical operator runbook lives at [docs/operator-runbook.md](docs/operator-runbook.md): build matrix, daily preflight, submit / loop / monitor sequencing, registry inspection, recovery after process interruption, backup / rotation, what-to-report-on-incident fields, `RUST_LOG` guidance, the stable tracing-field markers to grep, and a **docs-only systemd unit sample** (no installer, no service-manager code in the binary). README's Stage 8b runbook step 4 was updated to use `cargo run -p omni-node --features submit -- operator smoke …` (read-only steps 1–3 stay plain).

**Audit-note immutability:** the 2026-05-19 [mainnet smoke audit](docs/mainnet-smoke-audit.md) remains historical. Stage 9a's hardening is described here in the README, not by editing the audit's recorded facts.

**Tests** (default-on, hermetic) — feature matrix:

| Suite | Default features | `--features submit` |
|---|---|---|
| `cargo test -p omni-node` | **33** (24 prior Stage 8c read-only + 9 new Stage 9a: 2 parse matrices + 7 registry list/show) | **46** (37 prior Stage 8c full set + 9 new) |
| `cargo test -p omni-sumchain` | **86** (read-only suites + the new `no_submit_feature` typed-error test) | **103** (full suite incl. parity, submit-construction; the 5 `#[ignore]`'d live tests still self-skip) |
| `cargo build -p omni-node` / `--features submit` | warning-clean / warning-clean | — |

**What Stage 9a deliberately does not do:** no Stage 6 chain-wire or fixture changes (byte-stable); no Stage 7b tx-construction changes — the cfg-gating moves modules without modifying bytes, and the parity tests still assert byte-equivalence with the chain primitives under `--features submit` (Stage 9a referred to these as "vendored" because the deps were still git-sourced at the time; Stage 9c later moved them to public crates.io, byte-equivalent); no synthetic mainnet support; no daemonization / systemd code (sample unit is docs-only); no JSON output mode (deferred); no mutating registry-repair tools; no new chain RPCs / `omni-sumchain` methods; no publishing of `sumchain-primitives` / `sumchain-crypto` (chain team's call — superseded by Stage 9c, where the chain team did publish); no edits outside `crates/omni-node/` + `crates/omni-sumchain/` (Cargo.toml + cfg attributes only, no tx bytes) + `docs/operator-runbook.md` + `README.md`.

---

### Phase 5 Stage 9b: CI Matrix + Operator Distribution Docs — Superseded by Stage 9c

**Crate:** none (workflow + docs only) | **Depends on:** Stage 9a | zero source changes; zero protocol changes

> **Resolved by Stage 9c.** Stage 9b's first CI run on `main` exposed
> a real bug in Stage 9a's framing: `optional = true` on a git
> dependency only controls whether the dep is **compiled and linked**
> — Cargo still **clones** the git source at workspace resolution
> time to read its `Cargo.toml`. So on a fresh runner without
> credentials, default `cargo build -p omni-node` failed at resolve.
> The corrective commit (`09282d2`) paused the cargo-touching gates
> rather than masking the bug with default-job credentials.
>
> Stage 9c retired the private git dependency in favour of the chain
> team's public crates.io publish (Track A — `sumchain-primitives` /
> `sumchain-crypto` v0.1.0, dual-licensed MIT OR Apache-2.0,
> byte-equivalent to chain rev `d83e45a4` for the
> InferenceAttestation surface OmniNode consumes). The cargo-side
> gates from Stage 9b's original design are restored in Stage 9c
> and now run unconditionally on every push / PR, including
> external-fork PRs — no secrets required.

Stage 9b's workflow at [`.github/workflows/ci.yml`](.github/workflows/ci.yml) converted Stage 9a's manual `cargo tree` and feature-matrix verification into an enforced workflow, and codified the read-only vs submit-capable build choice as a top-of-README pointer. No source-code changes — Stage 9b is purely operational hardening. The text below describes the original design; per Stage 9c the cargo-side jobs are now restored and the secret-gating machinery is deleted.

> The table below is the **historical Stage 9b design**. See the
> Stage 9c section below for the **currently-live** workflow (the
> secret-gating machinery is removed; the cargo-side gates run
> unconditionally now that the chain crates are public).

| CI job | Triggers | Secret required (Stage 9b) | Stage 9c status | Outcome on miss |
|---|---|---|---|---|
| `default-build-test` | push to `main`, PR → `main` | none | **live (restored)** | hard fail on build/test regression |
| `default-tree-check` | same | none | **live (restored)** | fails if `cargo tree -p omni-node` ever pulls `sumchain-crypto` / `sumchain-primitives` |
| `stage6-fixture-check` | same | none | **live (unchanged)** | fails if `crates/omni-zkml/tests/fixtures/chain_attestation_vectors.json` or `crates/omni-zkml/src/chain_wire.rs` diff is non-empty vs the PR base / push parent |
| `check-submit-auth` | same | reads `SUM_CHAIN_DEPLOY_KEY` / `SUM_CHAIN_PAT` presence | **removed in 9c** (no longer needed; chain crates are public) | — |
| `submit-build-test` | same | originally `SUM_CHAIN_DEPLOY_KEY` / `SUM_CHAIN_PAT` | **live (restored, no secret)** | runs `cargo build / test -p omni-sumchain -p omni-node --features submit`; live `#[ignore]`'d tests are never run in CI |
| `submit-skip-notice` | same | — | **removed in 9c** (no skip path needed) | — |
| `submit-tree-check` | same | none | **live (new in 9c)** | fails if `cargo tree -p omni-node --features submit` does NOT contain the public chain crates (submit feature gone inert) |

**Secret gating shape (Q1):** prefer `SUM_CHAIN_DEPLOY_KEY` (read-only deploy key on the chain team's repo — least privilege) via `webfactory/ssh-agent`; fall back to `SUM_CHAIN_PAT` rewritten as an `https://x-access-token:<PAT>@github.com/.insteadOf` git config. The probe job never echoes the secret value, only its presence as a string output (`deploy_key | pat | none`).

**Triggers (Q2):** push to `main` + PR targeting `main` only. No cron job in 9b. No live-mirror / mainnet jobs (the `omni-sumchain` `#[ignore]`'d live tests stay operator-driven).

**Release artifacts (Q3):** none in 9b. The runbook names the future convention (`omni-node-readonly-<version>-<target>` / `omni-node-submit-<version>-<target>`) so a later stage has a target; no release pipeline / signing / publishing is added.

**PR-vs-direct policy going forward (Q4):** with CI in place, **any stage touching** `crates/omni-sumchain/src/tx.rs`, `crates/omni-zkml/src/chain_wire.rs`, the Stage 6 fixture, vendored SUM Chain deps / `--features submit`, or protocol/signing behavior **routes through a PR** so CI runs before merge. Small docs/operator non-protocol fixes can still go direct to `main`.

**Distribution discoverability (Q5):** a short "Choosing your build" pointer table lives at the top of this README (linking to the runbook); the full table + `--help` discoverability notes (e.g., `operator smoke` is **absent** from `--help` without the feature; clap reports "unrecognized subcommand 'smoke'" rather than a runtime stub) live in [`docs/operator-runbook.md`](docs/operator-runbook.md) § 1.

**Local dry-run of the gates against the Stage 9a commit:**

```text
cargo tree -p omni-node | grep -E 'sumchain-(crypto|primitives)'                 → empty (PASS)
git diff HEAD~1..HEAD -- <stage-6 fixture> <chain_wire.rs>                       → empty (PASS)
cargo tree -p omni-node --features submit | grep -E 'sumchain-(crypto|primitives)' → vendored crates present (PASS, informational)
```

**What Stage 9b deliberately does not do:** no source changes (no bug surfaced); no release pipeline / signing / publishing automation (deferred); no `cargo clippy` / `cargo deny` / SBOM / supply-chain scanning jobs (scope creep, deferred); no cron / drift-detection jobs (Q2 deferred); no daemonization / systemd code; no edits to `docs/mainnet-smoke-audit.md` (immutability); no chain protocol changes; no `omni-sumchain` API changes; no `omni-zkml` changes; no publishing of `sumchain-primitives` / `sumchain-crypto` ourselves.

---

### Phase 5 Stage 9c: Public SUM Chain Crates + CI Gate Restoration — Complete

**Crates touched:** workspace `Cargo.toml` + `Cargo.lock`, `crates/omni-sumchain/tests/parity_vendored_primitives.rs`, `.github/workflows/ci.yml`, docs | **Depends on:** chain team's Track A publish (`sumchain-primitives` v0.1.0 + `sumchain-crypto` v0.1.0 on crates.io, 2026-05-19) | zero source changes to `omni-sumchain`'s `src/`, `omni-zkml`, or any other workspace member; Stage 6 fixture + Stage 7b tx construction **byte-stable** (the new parity gates re-prove this against the public source).

Closes the fresh-runner source-build issue Stage 9b exposed. The workspace dependency for `sumchain-primitives` / `sumchain-crypto` migrates from a private git source at rev `d83e45a4` to the chain team's public crates.io release, byte-equivalent for the InferenceAttestation surface OmniNode consumes. The cargo-side CI gates Stage 9b designed (and then had to pause) are restored in this stage and now actually run on a fresh runner — no secrets, no credentials, no special CI machinery.

**Build matrix (verified locally pre-push):**

| Build | Source of `sumchain-primitives` / `sumchain-crypto` | GitHub credential needed? |
|---|---|---|
| `cargo build -p omni-node` | absent from compile graph (feature off) | **no** |
| `cargo build -p omni-node --features submit` | crates.io `v0.1.0` | **no** |

The Stage 9a feature gating is preserved — `optional = true` in `omni-sumchain/Cargo.toml` and `submit = ["dep:sumchain-primitives", "dep:sumchain-crypto"]`. Default builds still keep the submit code path out of the compiled binary; the `submit` feature is still the explicit opt-in for operators who want `smoke` / retry-loop. The only thing that changed is the **source** of the deps; everything else about the Stage 9a/8a operator-surface separation stays intact.

**Dep swap (`Cargo.toml` workspace.dependencies):**

```diff
-sumchain-primitives = { git = "https://github.com/SUM-INNOVATION/sum-chain", rev = "d83e45a4", package = "sumchain-primitives" }
-sumchain-crypto     = { git = "https://github.com/SUM-INNOVATION/sum-chain", rev = "d83e45a4", package = "sumchain-crypto" }
+sumchain-primitives = "0.1.0"
+sumchain-crypto     = "0.1.0"
```

`Cargo.lock` rebuilt; the source for both crates is now `registry+https://github.com/rust-lang/crates.io-index`. No `https://github.com/SUM-INNOVATION/sum-chain` URL remains in the lockfile.

**New parity tests** (added to `crates/omni-sumchain/tests/parity_vendored_primitives.rs`, file-level `#![cfg(feature = "submit")]`):

| Test | What it pins |
|---|---|
| `parity_transaction_v2_signing_hash_is_stable` | Builds a deterministic `TransactionV2`, computes `signing_hash()`, asserts the 32-byte BLAKE3 output equals a frozen value captured against `sumchain-primitives v0.1.0`. A future chain rev bump that changes the bincode wire format or the hash function fails this loudly. |
| `parity_signed_transaction_hex_roundtrip_is_stable` | Builds a deterministic `SignedTransaction` (TxV2 + real Ed25519 signature from a known seed), calls `to_hex()`, parses back via `from_hex()`, re-emits, asserts the two hex strings are byte-equal. Also pins the bare-hex (no `0x` prefix) lowercase contract and that the reparsed `TxInner::V2` preserves `chain_id` / `fee` / `payload` arm. |

These join the three existing Stage 7b parity tests (`InferenceAttestationDigest` bincode, `InferenceAttestationTxData` bincode, `Address::from_public_key`). Test count: 5 / 5 pass under `--features submit`. All other suites unchanged.

**CI workflow restored.** The cargo-side gates Stage 9b designed are back in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) as five jobs, all **unconditional** (no secret-gating, no skip branch, no deploy-key / PAT machinery):

| Job | Hard gate? |
|---|---|
| `default-build-test` | builds + tests workspace (minus PyO3 `omni-bridge`) under default features |
| `default-tree-check` | **yes** — fails if `cargo tree -p omni-node` ever pulls `sumchain-(crypto\|primitives)` |
| `submit-build-test` | builds + tests `--features submit` for `omni-sumchain` and `omni-node` (no live tests) |
| `submit-tree-check` | **yes** — fails if `--features submit` does NOT contain the chain crates (catches inert submit-feature regressions) |
| `stage6-fixture-check` | **yes** — fails if `chain_attestation_vectors.json` or `chain_wire.rs` diff is non-empty vs the PR base / push parent |

The `decoupling-status-notice`, `check-submit-auth`, and `submit-skip-notice` jobs are **removed** (no longer needed; the decoupling is real now).

**Stage 9c.1 follow-up — landed (see section below):** the chain team handed over the raw `SignedTransaction::to_hex` payload + BLAKE3-recomputed provenance for the 2026-05-19 mainnet smoke tx (`0x3a9cbf85945136e55a3ab8bb04a09d406d52438d9c2fa1f77850a706a1c32a56`) on 2026-05-20. Stage 9c.1 commits the fixture and adds a byte-roundtrip + `SignedTransaction::hash()` test against the authoritative on-chain bytes. See the **Phase 5 Stage 9c.1** section below.

**What Stage 9c deliberately does not do:**
- No Stage 6 chain-wire / fixture changes — the byte-stability gate re-runs against the public crates and passes.
- No Stage 7b transaction-construction logic changes — `omni-sumchain/src/tx.rs` and `outer_sign.rs` are untouched; the only change is the source the vendored types resolve from.
- No `omni-sumchain`/`omni-zkml` API changes; no operator-surface changes.
- No release pipeline / signing / publishing automation (Stage 9d candidate).
- No `cargo clippy` / `cargo deny` / SBOM / supply-chain scanning jobs in CI.
- No cron / drift-detection jobs.
- No daemonization / systemd code.
- No chain-produced byte-roundtrip fixture (landed in Stage 9c.1; see below).
- No edits to `docs/mainnet-smoke-audit.md` (immutability rule stands).
- No edits to `omni-types` / `omni-store` / `omni-net` / `omni-pipeline` / `omni-bridge` / `python/omninode`.

---

### Phase 5 Stage 9c.1: Chain-Produced Signed Transaction Fixture — Complete

**Crates touched:** `crates/omni-sumchain/tests/` only (one fixture, one test, plus doc pointers in this README and `crates/omni-sumchain/README.md`) | **Depends on:** Stage 9c (public `sumchain-primitives` v0.1.0) + chain-team handoff 2026-05-20 of raw signed-tx bytes for the 2026-05-19 mainnet smoke (`0x3a9cbf85945136e55a3ab8bb04a09d406d52438d9c2fa1f77850a706a1c32a56`) | zero changes to `omni-sumchain/src/`, `omni-zkml`, Stage 6 fixture, or any other workspace member; Stage 6 + Stage 7b byte-stable.

Adds the **authoritative chain-produced byte-roundtrip gate** Stage 9c flagged as a follow-up. Where `parity_vendored_primitives.rs` builds a synthetic `TransactionV2` in-test and proves local-vs-public byte equivalence under deterministic inputs, this stage takes real bytes the chain wrote to its own TRANSACTIONS column family and proves they round-trip through the public crate intact.

**Files added:**

| Path | Purpose |
|---|---|
| `crates/omni-sumchain/tests/fixtures/chain_produced_signed_tx.json` | Bare-hex `signed_tx_hex` for the 2026-05-19 mainnet smoke tx, with `tx_hash`, `chain_id`, `session_id`, `verifier_address`, `fee_paid`, `included_at_block_height`, and a `provenance` block (chain repo + commit `82e83e4d0e5a8bc8f516f7333d10ab73f14ac050`, crate versions, extraction command, UTC timestamp, operator). Two methodology notes (chain-id source disclosure + BLAKE3-over-bincode-on-CF-write) and one smoke-pattern-digest disclosure are preserved verbatim. |
| `crates/omni-sumchain/tests/chain_produced_fixture.rs` | file-level `#![cfg(feature = "submit")]`; five tests pinning structure, byte-roundtrip, and chain-produced hash equivalence. |

**Tests** (all under `cargo test -p omni-sumchain --features submit`):

| Test | What it pins |
|---|---|
| `fixture_signed_tx_hex_is_bare_lowercase_ascii_hex` | The fixture string is stored bare (no `0x` prefix), ASCII hex digits only, lowercase. Lowercase pinned for *this fixture string only* — not asserted as a protocol-level guarantee about `SignedTransaction::to_hex()`. |
| `fixture_bare_hex_roundtrips_byte_identical` | `SignedTransaction::from_hex(bare).to_hex() == bare` byte-identically against the chain-produced bytes. |
| `fixture_0x_prefixed_hex_is_accepted_and_reemits_bare` | `SignedTransaction::from_hex("0x" + bare)` succeeds and re-emits identically to the bare form (pins the documented `from_hex` prefix-acceptance behaviour). |
| `fixture_decodes_to_v2_inference_attestation_with_expected_fields` | Decoded `TxInner::V2`; `chain_id == 1`, `fee == 1000`, `from.to_base58() == "2mvPk4h883B7DrcZvwy7yWKXyGYHuVzGP"`, payload arm is `TxPayload::InferenceAttestation`, `digest.session_id == "mainnet-smoke-2026-05-18-001"`. |
| `fixture_signed_tx_hash_matches_chain_produced_tx_hash` | `sumchain_primitives::SignedTransaction::hash().to_hex() == fixture.tx_hash`. The public hash method is `Hash::hash(bincode::serialize(self))`, exactly the chain's TRANSACTIONS-CF key derivation — so a match pins both that the fixture bytes are the exact stored row and that the public crate's hashing has not drifted from the chain's internal hashing. A future divergence fails loudly. |

**Smoke-pattern digest disclosure.** The decoded payload's `model_hash` (`0xaa…`), `manifest_root` (`0x11…`), `response_hash` (`0xbb…`), and `proof_root` (`0x22…`) are smoke-test placeholder bytes — the 2026-05-19 smoke exercised the submit path with non-semantic digests. Stage 9c.1 asserts structure and byte-roundtrip; it does not treat the digest payload as meaningful content. This is documented in both the fixture's `notes.digest_payload` block and the test module's preamble.

**What Stage 9c.1 deliberately does not do:**
- No changes to `omni-sumchain/src/tx.rs`, `outer_sign.rs`, or `client.rs`.
- No changes to Stage 6 chain-wire / fixture (`crates/omni-zkml/src/chain_wire.rs`, `chain_attestation_vectors.json`).
- No local reconstruction of the signed-tx hex from a verifier seed; no synthetic fallback.
- No edits to `docs/mainnet-smoke-audit.md` (immutability rule stands).
- No behaviour changes anywhere; purely additive test surface.
- No new CI workflow jobs — the existing `submit-build-test` job already runs `cargo test -p omni-sumchain --features submit`, which now picks up `chain_produced_fixture.rs` automatically.

---

### Phase 5 Stage 10a: Operator Observability + Release Readiness — Complete

**Crates touched:** [`crates/omni-node/src/operator.rs`](crates/omni-node/src/operator.rs) (new subcommand + 9 hermetic tests + 3 stable log marker helpers), [`docs/operator-runbook.md`](docs/operator-runbook.md) (§1a `--help` capture, §6 summary docs, §7 marker contract, §11 failure-triage matrix, §13 quick-reference rows, new §14 release checklist), README.md (this section + status row). | **Depends on:** Stage 9c.1; the 2026-05-19 mainnet smoke proved the submit path works on chain and Stage 9c/9c.1 closed the build-risk loop — Stage 10a is the operator-experience layer that follows. | **Zero protocol changes; zero new chain RPCs; Stage 6 chain-wire + fixture and Stage 7b tx construction byte-stable.**

Makes `omni-node` operable as a real operator binary: a counts-by-status registry summary, stable structured log markers, a failure-state triage matrix grounded in the typed surface that actually exists today, and a documented release-readiness checklist. No release artifact workflow in this stage (deferred to Stage 10b until there's a concrete operator audience asking for downloadable binaries).

**1. `operator registry summary` (new read-only subcommand).** Bare-stdout three-line shape:

```text
total=12
pending=2 submitted=4 included=1 finalized=3 failed=1 dropped=1
oldest_submitted_age_blocks=147 (id=ab12…, submitted_at_block=6049055, head=6049202)
```

The third line is replaced with a stable `# oldest_submitted_age: skipped (<reason>)` comment when:
- `--rpc-url` is omitted (default-build path; no chain access at all);
- the registry has no `Submitted` records; or
- the only Submitted records were inserted via the legacy `mark_submitted` (no `submitted_at_block`).

`--expect-chain-id` is **required** when `--rpc-url` is supplied — `summary` never queries an unguarded chain endpoint (matches every other chain-touching operator subcommand). When provided, it does exactly one `get_chain_params` + one `get_block_height(Latest)` and computes the oldest `Submitted` record's age in blocks. Never writes; never submits.

**2. Stable `event=` log markers.** Three new structured tracing markers join the existing free-form lines, pinned by [`docs/operator-runbook.md` §7a](docs/operator-runbook.md):

| `event=` | Emitter | Key fields |
|---|---|---|
| `startup` | every `operator` invocation, before any chain/registry access | `subcommand`, `feature_submit`, `version` |
| `chain_params` | first chain read in `watch-activation`, `smoke`, `preflight`, `query`, loop entry | `chain_id`, `finality_depth`, `min_fee`, `omninode_enabled_from_height`, `v2_enabled_from_height`, `head` |
| `activation_state` | `watch-activation` per tick, `smoke` pre-submit, `preflight` (with RPC), loop tick | `omninode_active`, `v2_active`, `activated`, `head` |
| `registry_summary` | `operator registry summary` | the six per-status counts + `oldest_submitted_age_blocks` |

These coexist with the existing Stage 9a free-form markers (`SMOKE SUMMARY`, `loop tick complete`, etc.) — operator pipelines can choose either; new pipelines should prefer `event=` for structured grep.

**3. Failure-triage matrix in [runbook §11a](docs/operator-runbook.md).** Maps eleven concrete failure states (chain `Unknown` persisting, chain `Failed { reason: String }`, RPC errors, stale Submitted → local Dropped, duplicate / idempotency conflicts, verifier-address mismatch, insufficient balance / fee, queryable record missing receipt, `SmokeConfirmTimeout`, `SmokeInterrupted`, `MainnetSubmitNotPermitted`) to:

- the typed surface signal the operator should grep / catch for;
- whether it's fully detectable today or only partially (string-opaque);
- the recommended local action;
- the escalation packet to send the chain team.

> **Known limitation flagged explicitly in §11a.** [`ChainClientError`](crates/omni-zkml/src/error.rs) is currently the single-variant `Other(String)`. Fee, balance, transport, and signature failures all funnel through that variant — operator triage must parse the string. Stage 10a does **not** add new error variants (that would be a typed-surface change). Splitting `ChainClientError` into a typed taxonomy is a candidate for a later stage if operator feedback shows the parsing burden is real.

**4. Release-readiness docs ([runbook §1a "Capturing `--help` for verification"](docs/operator-runbook.md) + [new §14](docs/operator-runbook.md)).** Documentation-only, no automation. Covers:
- the exact build commands for read-only vs. submit binaries;
- a `--help` capture + diff workflow for verifying the binary on a destination host without rebuilding;
- a manual SHA-256 recording step alongside the git tag;
- a 10-item release checklist (CI green, Stage 6 byte-stable across the release window, local test parity, Cargo version = git tag, runbook re-read, audit-doc immutability, Stage 9c.1 hash gate green, `--version` capture, SHA-256, `--help` snapshots).

**5. No release-artifact workflow.** Confirmed deferred to Stage 10b. There is currently no external operator audience expecting downloadable `omni-node` binaries; until there is, the release checklist is a manual operator checkpoint and not a CI job. Stage 10b will evaluate `workflow_dispatch` artifact builds + checksum publication when readiness arrives.

**Tests.** Nine new hermetic tests in [`crates/omni-node/src/operator.rs`](crates/omni-node/src/operator.rs) cover the summary subcommand:

| Test | What it pins |
|---|---|
| `registry_summary_zero_records_emits_zero_total` | empty registry → `total=0`, all six per-status fields zero, skip reason `(no --rpc-url)` |
| `registry_summary_counts_by_status_match_seed` | one record in each of the six statuses → exact counts |
| `registry_summary_age_line_skipped_when_no_rpc` | no `--rpc-url` → skip reason set; `oldest_submitted_age` is `None` |
| `registry_summary_age_line_skipped_when_no_submitted_records` | RPC available but no Submitted records → skip reason `(no Submitted records)` |
| `registry_summary_age_line_skipped_when_submitted_has_no_block` | Submitted record exists but `submitted_at_block` is None (legacy `mark_submitted` path) → skip reason `(no Submitted record has submitted_at_block)` |
| `registry_summary_age_line_picks_oldest_submitted` | multiple Submitted records → picks the one with the smallest `submitted_at_block`; never picks Included / Failed / Dropped records that happen to also carry a block |
| `registry_summary_age_uses_fake_head_height` | `age_blocks = head − submitted_at_block` against a `FakeJsonRpcTransport`-supplied head |
| `registry_summary_chain_id_mismatch_is_typed_error` | `--expect-chain-id` vs chain `chain_id` mismatch → typed `OperatorError::ChainIdMismatch` before the height read |
| `print_registry_summary_renders_age_line_when_present` | the bare-stdout renderer signature stays stable (compilation gate) |

Result counts: **`cargo test -p omni-node` 42/42 pass** (was 33 in Stage 9a); **`cargo test -p omni-node --features submit` 55/55 pass** (was 46 in Stage 9a). All other suites unchanged.

**What Stage 10a deliberately does not do:**
- No Stage 6 chain-wire / fixture changes (byte-stable).
- No Stage 7b transaction-construction changes; no edits to `crates/omni-sumchain/src/{tx,outer_sign,client}.rs`.
- No new chain RPCs.
- No new `ChainClientError` variants (the opaque-string reality is documented honestly in §11a; splitting the taxonomy is a candidate for a future stage).
- No Prometheus / OTLP / metrics endpoint; no `--json` output mode for operator commands; no metrics backend of any kind.
- No release-artifact workflow / signing pipeline (deferred to Stage 10b).
- No daemonization / systemd code (the sample unit in §12 stays docs-only).
- No edits to `docs/mainnet-smoke-audit.md` (immutability rule stands).
- No edits to `omni-types` / `omni-store` / `omni-net` / `omni-pipeline` / `omni-bridge` / `python/omninode`.

---

### Phase 5 Stage 11a: Real zkML Proof Generation Integration — Complete

**Crates touched:** new [`crates/omni-zkml/src/proof.rs`](crates/omni-zkml/src/proof.rs) (traits + mock impls + orchestrator), [`crates/omni-zkml/src/error.rs`](crates/omni-zkml/src/error.rs) (typed errors), [`crates/omni-zkml/tests/proof_pipeline_vectors.rs`](crates/omni-zkml/tests/proof_pipeline_vectors.rs) + [fixture JSON](crates/omni-zkml/tests/fixtures/proof_pipeline_vectors.json), [`crates/omni-node/src/operator.rs`](crates/omni-node/src/operator.rs) (new `build_mock_v1_attestation` + mainnet guard + 2 new tests), [`crates/omni-node/Cargo.toml`](crates/omni-node/Cargo.toml) (adds `blake3` dep), [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (new `stage11a-fixture-check` job), [`docs/operator-runbook.md`](docs/operator-runbook.md) (§4 + §11a + §14 updates), README. | **Depends on:** Stage 9c.1 chain-produced fixture + Stage 10a operator observability + Phase 5 RC audit `d695e41`. | **Zero protocol changes; zero new chain RPCs; Stage 6 chain-wire + fixture and Stage 7b tx construction byte-stable.**

Replaces the pre-Stage-11a placeholder proof surface — where [`crates/omni-node/src/operator.rs`](crates/omni-node/src/operator.rs)'s `synth_attestation` fabricated `model_hash="a".repeat(64)` / `manifest_snip_root=[0x11;32]` / `response_hash="b".repeat(64)` / `proof_snip_root=[0x22;32]` for every commitment field (the bytes that landed on chain in the 2026-05-19 mainnet smoke, see [`docs/mainnet-smoke-audit.md`](docs/mainnet-smoke-audit.md) and [`docs/phase5-rc-audit.md`](docs/phase5-rc-audit.md)) — with a real backend-trait-driven pipeline. Stage 11a is **trait + hermetic fixture harness first**: the real proof system (ezkl / risc0 / sp1 / …) is deliberately deferred to Stage 11b so we don't ship heavyweight prover dependencies before the trait shape has been validated end-to-end.

**1. `ProofBackend` + `ProofVerifier` traits ([crates/omni-zkml/src/proof.rs](crates/omni-zkml/src/proof.rs)).**

```rust
pub trait ProofBackend {
    fn prove(&self, model: &[u8], input: &[u8], output: &[u8])
        -> Result<Vec<u8>, ProofBackendError>;
    fn backend_id(&self) -> &'static str;
}

pub trait ProofVerifier {
    fn verify(&self, proof: &[u8], public_inputs: &PublicInputs)
        -> Result<bool, ProofVerifierError>;
}
```

`PublicInputs` is a typed struct (not opaque `&[u8]`) carrying `model_hash` / `input_hash` / `output_hash` as 32-byte BLAKE3 hashes. Stage 11b can extend the struct without rewriting verifier impls.

**2. `MockProofBackend` (`backend_id = "mock-v1"`) + `MockProofVerifier`.** Deterministic 64-byte "proof" computed as two domain-separated BLAKE3s over `(model_hash || input_hash || output_hash)`. **Not cryptographic.** Tampering with proof bytes or public inputs causes verification to fail — useful for hermetic tests of the pipeline shape, useless for mainnet soundness.

**3. `ProofMetadata` committed inside the SNIP V2 proof artifact.** Per Stage 11a OQ4: `input_hash` does NOT go on the chain-wire digest. Instead, a `ProofMetadata { backend_id, model_hash, input_hash, response_hash }` struct lives inside a canonical JSON envelope (`ProofArtifactBody`), and the BLAKE3 of that envelope becomes the `proof_snip_root` already committed on the chain digest. **Stage 6 chain-wire bytes are unchanged** — and the verifier still has every field it needs (via SNIP V2 fetch + envelope parse).

**4. End-to-end orchestrator [`produce_proof_artifact`](crates/omni-zkml/src/proof.rs).** Wraps backend invocation, metadata composition, canonical JSON serialization, file write, and SNIP V2 publish via the existing Stage 3 [`publish_proof_artifacts`](crates/omni-zkml/src/artifact.rs#L103) plumbing. `build_commitment` is **untouched** — proof concerns live in the new orchestrator (per Stage 11a correction 1).

**5. Operator binary wiring.** New [`build_mock_v1_attestation`](crates/omni-node/src/operator.rs) replaces the production smoke `--synthetic` call to `synth_attestation`. The new function:
- runs `MockProofBackend.prove(...)` over deterministic synthetic `(model, input, output)` bytes tied to `(verifier_address, head)`;
- builds a real `InferenceCommitment` with BLAKE3-derived hashes + BLAKE3-derived `proof_snip_root` (matching what a SNIP V2 publish would emit);
- hard-refuses `chain_id == 1` with `OperatorError::MockBackendRefusedOnMainnet { backend_id }` **before any submit-side RPC**, even with `--allow-mainnet-submit` (Stage 11a OQ5 hard requirement, defense-in-depth alongside Stage 8b's `SyntheticMainnetForbidden`).

`synth_attestation` is preserved as a `#[cfg(test)]`-only fixture so the existing dozen-plus test sites that exercise the registry / chain plumbing don't need rewriting. The production binary genuinely no longer contains the placeholder fabricator.

**6. Hermetic Stage 11a fixture.** New [`crates/omni-zkml/tests/proof_pipeline_vectors.rs`](crates/omni-zkml/tests/proof_pipeline_vectors.rs) + [`tests/fixtures/proof_pipeline_vectors.json`](crates/omni-zkml/tests/fixtures/proof_pipeline_vectors.json) with **3 vectors** covering short bytes, empty-input edge case, and longer ASCII bodies. Each vector pins (committed verbatim):

| Field group | Bytes captured |
|---|---|
| Raw inputs | model bytes, input bytes, output bytes, verifier Ed25519 seed |
| Expected hashes | `model_hash` / `input_hash` / `response_hash` (all BLAKE3 hex) |
| Expected proof | 64 bytes from `MockProofBackend`, hex-encoded |
| Expected envelope | full canonical `ProofArtifactBody` JSON bytes (hex) + their BLAKE3 = `proof_snip_root` |
| Expected `manifest_root` | deterministic BLAKE3 over `("omninode.stage11a.fixture.manifest" || model_bytes)` |
| Expected commitment | full `InferenceCommitment` field-by-field (session_id, model_hash, manifest_snip_root, response_hash, proof_snip_root) |
| Expected chain-wire | canonical-digest bytes + signing-input bytes (both via Stage 6 `compute_chain_attestation_vector`) |
| Expected signature | Ed25519 sig bytes + signer address (base58) + signer pubkey (hex) |

Regen via `OMNINODE_REGEN_PROOF_PIPELINE_VECTORS=1`. Verify-by-default. A drift at **any** layer (mock backend, canonical envelope shape, BLAKE3 chain, commitment field ordering, chain-wire bincode, Ed25519 signing) fails this test loudly.

**7. New CI hard gate: `stage11a-fixture-check`.** Mirrors `stage6-fixture-check`: asserts `proof_pipeline_vectors.json` is byte-identical against the PR base / push parent. The CI matrix is now **six jobs**: `default-build-test`, `default-tree-check`, `submit-build-test`, `submit-tree-check`, `stage6-fixture-check`, `stage11a-fixture-check`.

**Tests.** Counts under Stage 11a:

| Suite | Default features | `--features submit` |
|---|---|---|
| `cargo test -p omni-zkml` | **152** (was 142 at Stage 10a; +10: 8 new proof.rs unit tests + 3 new fixture tests minus internal rearrangement) | same |
| `cargo test -p omni-node` | **43** | **58** (was 56 at Stage 10a; +2 new: `build_mock_v1_attestation_produces_real_blake3_commitment_on_nonmainnet`, `build_mock_v1_attestation_refuses_mainnet_with_explicit_error`) |
| `cargo test -p omni-sumchain --features submit` | — | same (Stage 11a doesn't touch sumchain) |

**Mainnet smoke history.** The 2026-05-19 mainnet smoke (tx `0x3a9cbf85…c32a56`, pinned by Stage 9c.1) is **unchanged** by Stage 11a. The mainnet-smoke-audit doc and the Phase 5 RC audit doc remain immutable; Stage 11a's effect is only forward — future submits cannot route the mock backend onto mainnet.

**What Stage 11a deliberately does not do:**
- **No Stage 6 chain-wire / fixture changes.** `chain_attestation_vectors.json` and `chain_wire.rs` byte-stable per the Stage 6 baseline `509f7fd`; CI's `stage6-fixture-check` enforces this on every push/PR.
- **No Stage 7b changes.** No edits to `crates/omni-sumchain/src/tx.rs`, `outer_sign.rs`, or `client.rs`. Submit transaction construction doesn't know proofs got real.
- **No `build_commitment` signature change** — proof concerns live in the new orchestrator (Stage 11a correction 1).
- **No real prover dependency.** ezkl / risc0 / sp1 / halo2 / plonky2 not added — Stage 11b's job.
- **No tokenomics / staking / slashing / disputes / chain-side proof verification.** Stage 8+ territory.
- **No retry-loop proof regeneration.** Retry of a dropped Submitted record reuses the existing attestation in the registry; it does NOT regenerate a new proof. Whether the dropped-record workflow can ever regenerate proofs depends on whether the model / input / output bytes are persisted; in Stage 11a they are not (synthetic inputs are derived from `(verifier_address, head)` on each call; that's enough for the synthetic path's idempotency but not enough to re-derive a different proof from a real attestation). **Documented as deferred.**
- **No `--proof-backend` CLI flag.** Stage 11a ships only one backend (`mock-v1`); a flag with one accepted value would be pure scaffold. Stage 11b adds the flag when there's more than one option.
- **No `docs/mainnet-smoke-audit.md` or `docs/phase5-rc-audit.md` edits** (both immutable).
- **No edits to `omni-types` / `omni-store` / `omni-net` / `omni-pipeline` / `omni-bridge` / `python/omninode`.**

> **Stage 11b.0 framing correction.** Stage 11a's "Real zkML proof generation integration" header above is the historical commit description and is preserved as the milestone record. The corrected Stage 11b scope framing is **"decentralized proof architecture"** — not "production zkML" and not "decentralized compute readiness." The bounded MockProofBackend Stage 11a ships proves *a deterministic BLAKE3 chain over hashes of inputs*, not the operator's ML inference. See the Stage 11b.0 section below for the load-bearing scope clarification, the new typed `ProofSystem` enum, and the conservative mainnet-refusal layers.

---

### Phase 5 Stage 11b.0: Decentralized Proof Architecture — Complete

**Crates touched:** [`crates/omni-zkml/src/proof.rs`](crates/omni-zkml/src/proof.rs) (trait extensions, `ModelFormat` + `ProofSystem` enums, `MainnetRefusalReason`, `check_mainnet_eligible`, 11 new unit tests), [`crates/omni-zkml/src/lib.rs`](crates/omni-zkml/src/lib.rs) (re-exports), [`crates/omni-node/src/operator.rs`](crates/omni-node/src/operator.rs) (6 new typed `OperatorError` variants, `OperatorCmd::VerifyProof` subcommand on default build, `map_mainnet_refusal` helper, 6 new tests), runbook (§6a, §11a, §11c, §13, §14 updates), README. | **Depends on:** Stage 11a (PR #6 merged at `34d1bb8`). | **Zero protocol changes, zero new chain RPCs, Stage 6 chain-wire + fixture byte-stable, Stage 11a `proof_pipeline_vectors.json` byte-identical.**

This stage replaces the misleading "production zkML" framing of the Stage 11a single-backend swap-in with an honest **proof architecture**:

**1. Chain Compatibility Contract — hard constraint, enforced by code + CI:**
- [`crates/omni-zkml/src/chain_wire.rs`](crates/omni-zkml/src/chain_wire.rs) and [Stage 6 fixture](crates/omni-zkml/tests/fixtures/chain_attestation_vectors.json): untouched (CI `stage6-fixture-check` gates).
- [`crates/omni-sumchain/src/tx.rs`](crates/omni-sumchain/src/tx.rs) and [`outer_sign.rs`](crates/omni-sumchain/src/outer_sign.rs): untouched.
- No new SUM Chain RPCs.
- All Stage 11b.0 proof-system metadata lives inside the SNIP V2 proof artifact (the canonical `ProofArtifactBody` JSON committed by the existing `proof_snip_root`). **The chain payload is byte-identical to Stage 7b.**
- Chain-side verification and tokenomics are Stage 12+ and require separate approval.

**2. Multi-backend trait extensions ([`omni-zkml`](crates/omni-zkml/src/proof.rs)):**

The Stage 11a `ProofBackend` / `ProofVerifier` traits gain four new methods, three defaulted for back-compat and one required:

| Method | Defaulted? | Purpose |
|---|---|---|
| `proof_system() -> ProofSystem` | yes (`Mock`) | Verifier-routing key; consulted by `check_mainnet_eligible`. |
| `supported_formats() -> &[ModelFormat]` | yes (empty) | Format-aware adapter dispatch; the mock returns empty (format-agnostic). |
| `circuit_id() -> Option<[u8; 32]>` | yes (`None`) | Compiled-circuit / image-id binding. |
| `is_local_only() -> bool` | **no — required** | Decentralization invariant: every backend must explicitly declare. |

`ProofVerifier` gains `proof_system()` to mirror the backend side.

**3. Expanded `ProofMetadata` schema, with byte-stable serde:**

Six new optional fields, every one declared `#[serde(skip_serializing_if = "Option::is_none", default)]`:

- `model_format: Option<ModelFormat>` — `Onnx` | `Gguf` | `Other(String)`.
- `proof_system: Option<ProofSystem>` — `Mock` | `Stage11bOnnxReference` | `Ezkl` | `GgufStrategyTbd`.
- `circuit_id_hex: Option<String>` — 64-char lowercase hex.
- `verification_key_hex: Option<String>` — for backends that need an explicit verification key.
- `public_inputs: Option<serde_json::Value>` — backend-specific opaque shape.
- `testnet_or_dev_only: Option<bool>` — explicit "this artifact is NOT mainnet-eligible" producer flag.

The Stage 11a `proof_pipeline_vectors.json` fixture is **byte-identical** under this schema — every new field is absent on Stage 11a vintage artifacts, so the JSON serializes the same. A new `ProofMetadata::new_stage11a(...)` constructor is the canonical "Stage 11a-shape" factory; every existing call site uses it (verified by the `stage11a_compat_metadata_serializes_without_stage11b_fields` unit test).

**4. `check_mainnet_eligible()` — six-layer conservative refusal:**

| Layer | Refusal reason | Fires when |
|---|---|---|
| 1 | `TestnetOrDevOnly` | `testnet_or_dev_only != Some(false)` — i.e., either `Some(true)` (producer explicitly disclaimed) OR `None` (no declaration; treated as testnet/dev for safety). Stage 11d.1 tightening: only an explicit `Some(false)` passes. |
| 2 | `MockBackend` | `proof_system ∈ {Mock, None}` — preserves Stage 11a's `MockBackendRefusedOnMainnet` guarantee |
| 3 | `BoundedReference` | `proof_system ∈ {Stage11bOnnxReference, Stage11bHalo2Reference}` — bounded reference fixtures never mainnet |
| 4 | `GgufClaim` | `model_format == Gguf` — no GGUF inference proof backend is approved at any stage through Stage 11d.x |
| 5 | `UnknownModelFormat` | `model_format ∈ {Other(_), None}` (non-mock) — stringly-typed formats and absent formats refused |
| 6 | `NotInMainnetAllowlist` | artifact's `(proof_system, circuit_id_hex, model_hash)` triple does not match any entry in `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` (Stage 11d.1 structured allowlist), AND `proof_system` is not in the legacy `MAINNET_APPROVED_PROOF_SYSTEMS` back-compat alias |

**Both `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` and the legacy `MAINNET_APPROVED_PROOF_SYSTEMS` are empty by design through Stage 11d.1 / 11d.2.** Only a Stage 11d.3 PR with written chain-team sign-off populates the structured list. Stage 11d.1 introduced the structured `AllowlistEntry`-keyed schema (`AllowlistEntry { proof_system, backend_id, circuit_id_hex, model_hash, model_format, verification_key_hash_hex, chain_team_review_ref }`); the legacy `&[ProofSystem]` slice is preserved as an empty back-compat alias. Mainnet eligibility lands when a Stage 11d.3 entry is added — see [docs/mainnet-eligibility-criteria.md](docs/mainnet-eligibility-criteria.md). The `mainnet_allowlist_is_empty_at_stage_11b0`, `mainnet_allowlist_entries_empty_after_stage_11d1`, `legacy_MAINNET_APPROVED_PROOF_SYSTEMS_empty_after_stage_11d1`, and `every_proof_system_is_refused_on_mainnet_at_stage_11b0` tests pin this invariant.

**5. `operator verify-proof` (new read-only subcommand, default build):**

```bash
omni-node operator verify-proof --proof-artifact ./some-proof.json
```

Output (every Stage 11b.0 artifact reports `mainnet_eligible=false`, since the allowlist is empty):

```
backend_id=mock-v1
proof_system=none
model_format=none
verified=true
mainnet_eligible=false
mainnet_refusal=proof artifact uses proof system None (or none declared, treated as Mock); …
```

Available on default builds (no `--features submit` required). Stage 11b.0 registers only `MockProofVerifier` in the lookup table; other proof systems return the typed `OperatorError::NoVerifierForProofSystem` until backends land in Stage 11c+.

**6. Six new typed `OperatorError` refusal variants** (in addition to Stage 11a's `MockBackendRefusedOnMainnet`, preserved): `TestnetOnlyProofRefusedOnMainnet`, `BoundedReferenceProofRefusedOnMainnet`, `GgufProofClaimRefusedOnMainnet`, `UnknownModelFormatRefusedOnMainnet`, `ProofSystemNotMainnetApproved`, `NoVerifierForProofSystem`. The `map_mainnet_refusal` helper centralizes the 1:1 mapping between `omni_zkml::MainnetRefusalReason` variants and the operator's typed errors.

**7. Decentralized proof architecture — naming + scope:**

This stage is called **"decentralized proof architecture,"** never "decentralized compute readiness" or "production zkML." The criteria for the *readiness* claim are documented and explicitly unmet:

- ✗ at least one non-mock backend (deferred to Stage 11b.1 → Stage 11c).
- ✗ local prover reproducibility demonstrated (deferred).
- ✗ verifier-only acceptance by independent nodes (architecture present; demonstration deferred).
- ✗ clear mainnet semantics matching chain attestation (Stage 11c + chain-team review).

Local proving is the only required path. **No Bonsai, no hosted prover dependency, no centralized service** in any Stage 11b.0 code path or documentation. Remote proving is not mentioned in the Stage 11b.0 codebase.

**Tests:**
- `cargo test -p omni-zkml`: **152 → 163** (+11 Stage 11b.0 tests: mock-backend trait declarations, mock-verifier declaration, `MAINNET_APPROVED_PROOF_SYSTEMS` emptiness, every-proof-system-refused, six refusal-layer cases, Stage 11a-compat serialization byte-stability).
- `cargo test -p omni-node`: **43 → 47** (+4 verify-proof tests under default build).
- `cargo test -p omni-node --features submit`: **58 → 64** (+6: 4 verify-proof + 2 map_mainnet_refusal coverage).
- `cargo test -p omni-zkml --test proof_pipeline_vectors`: still **3/3** byte-stable — Stage 11a fixture verified intact.
- Stage 6 byte-stability (`git diff 509f7fd..HEAD` on `chain_wire.rs` + the chain-attestation fixture): empty.

**What Stage 11b.0 deliberately does not do:**
- **No Stage 11b.1 work in this PR.** No `omni-proofs-ezkl-reference` crate. No ezkl dep. No bounded ONNX reference backend. No prover fixture regen workflow. Stage 11b.1 is a separate plan + PR.
- **No mainnet allowlist additions.** The `MAINNET_APPROVED_PROOF_SYSTEMS` const slice stays empty.
- **No public-API moves.** Stage 11a's `omni_zkml::ProofBackend`, `omni_zkml::ProofVerifier`, etc. stay where they are; the trait extensions are additive and back-compat for any Stage 11a implementor.
- **No new heavy prover deps anywhere.**
- **No new CI workflow jobs.** Stage 11b.0's checks are pure-Rust unit tests that run inside the existing `default-build-test` job. No `stage11b0-arch-check` shell job; the same invariants are pinned by `mainnet_allowlist_is_empty_at_stage_11b0`, `every_proof_system_is_refused_on_mainnet_at_stage_11b0`, and `stage11a_compat_metadata_serializes_without_stage11b_fields`.
- **No "zkML" claim** on the bounded reference circuit (Stage 11b.1 deferred) or anywhere in 11b.0 documentation.
- **No GGUF inference proof readiness claim.** Stage 11b.0 ships the *refusal* infrastructure for GGUF — declaring `model_format = "gguf"` is honest, mainnet refuses the claim, and the runbook (§11c) documents the open menu of future Stage 11e research-track strategies.
- **No edits to** [`docs/mainnet-smoke-audit.md`](docs/mainnet-smoke-audit.md) **or** [`docs/phase5-rc-audit.md`](docs/phase5-rc-audit.md) (both immutable).
- **No edits to** `omni-types` / `omni-store` / `omni-net` / `omni-pipeline` / `omni-bridge` / `python/omninode` **source.**

---

### Phase 5 Stage 8+: zkML Proof Generation & SUM Chain Tokenomics — Planned

**Crate:** `omni-zkml` | **Smart Contracts:** `contracts/` | **Depends on:** `omni-pipeline`, `omni-net`, `omni-types`

Cryptographically prove that inference was executed correctly. Tie proofs to a staking/slashing economy on SUM Chain.

| Component | Implementation status |
|---|---|
| Proof Generation | **Stage 11d.2 production fixed-point MLP shipped** (16→32→16→8 `production-fixedpoint-mlp-v1`) alongside the original Stage 11b.1.b + 11c bounded reference (4→8→4 `halo2-mlp-v1`, testnet/dev-only). The production proof class is reachable via the opt-in `stage11d-production-verify` cargo feature on `omni-node` (default builds pull zero halo2); **off-chain only and still mainnet-refused** — `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]` is empty by design, and `check_mainnet_eligible` layer 6 refuses every artifact under this proof class until a Stage 11d.3 allowlist PR with external-cryptographer sign-off lands. The Stage 11d.0 mainnet-eligibility criteria + the Stage 11d.2 review packet + benchmark record are committed; full R9 chain-team sign-off is the remaining 11d.3 gate. No ezkl dependency (rejected at Stage 11b.1 due to license posture — `v23.0.5` has no `LICENSE` file and no `Cargo.toml` `license` field). RISC Zero remains an unevaluated future option. |
| Proof Aggregation | Future work, not yet planned. |
| On-Chain Verification | Future work — not in Stage 11d. The Stage 11d.1 allowlist schema is shaped so a future on-chain verifier is a refactor, not a schema change. See [docs/mainnet-eligibility-criteria.md §5](docs/mainnet-eligibility-criteria.md). |
| Staking | Future Stage 8+ — gated on Stage 11d.3 sign-off + dedicated chain-team plan. |
| Slashing | Future Stage 8+ — same gating. |
| Financial RLHF | Future Stage 8+ — same gating. |

**Reference + future prover landscape (current honest state):**

| Backend | Shipped today | Notes |
|---|---|---|
| halo2 (Pasta IPA) reference | **Stage 11b.1.b + 11c** — bounded `halo2-mlp-v1` verifier behind opt-in `halo2-reference-verify` feature; testnet/dev-only; not a production proof | Used for architecture validation, not real inference |
| halo2 production proof class | **Stage 11d.2 shipped** — `production-fixedpoint-mlp-v1` (16→32→16→8 int16 fixed-point MLP) behind opt-in `stage11d-production-verify` feature; off-chain only and **still mainnet-refused** (`MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]`) | Built on Stage 11c gadgets (RHAZ + saturation + ReLU + range checks); `circuit_id_hex` + `verification_key_hash_hex` pinned and drift-checked; allowlist entry blocked on Stage 11d.3 external-cryptographer sign-off + full R9 chain-team sign-off |
| ezkl (Halo2 SNARK on ONNX) | Not depended upon; **rejected at Stage 11b.1** due to licensing | Revisitable if upstream license posture changes |
| RISC Zero (STARK) | Not depended upon; unevaluated | Could become a future Stage 11e+ research track |
| GGUF/LLM inference | No proof strategy; refused by `check_mainnet_eligible` layer 4 | Future Stage 11e+ research track |

**Smart Contract Architecture:**
```
StakingContract
  ├── stake(node_id, amount)              // Join the network
  ├── unstake(node_id)                    // Begin 7-day unbonding
  ├── slash(node_id, proof_of_fault)      // Burn stake for provably bad inference
  └── reward(node_id, amount, proof)      // Distribute reward after verified inference

VerifierContract
  ├── verifyProof(proof, public_inputs)   // Verify single-stage proof
  └── verifyAggregatedProof(agg_proof)    // Verify composite multi-stage proof

RewardDistributor
  ├── submitFeedback(inference_id, score) // Human feedback score (0.0–1.0)
  ├── computeReward(inference_id)         // Calculate weighted reward
  └── distribute()                        // Batch-distribute pending rewards
```

> **SUM Chain Note:** The smart contract language, SDK, RPC endpoints, and deployment tooling are dependent on SUM Chain's architecture. The contract interfaces above are defined abstractly. Specific implementation will begin only after consulting the SUM Chain team for exact specifications.

**Milestone:** A node generates a zk proof of correct inference, submits it on-chain, and receives a token reward proportional to compute contribution and human feedback quality.

---

## Data Flow: End-to-End Shard Transfer

```
Node A: omni-node shard model.gguf
  1. mmap model.gguf
  2. Parse GGUF header, metadata, tensor index (zero-copy)
  3. Classify tensors: token_embd → embedding, blk.N → block N, output → head
  4. Plan chunks: group by layers_per_shard (default 4)
  5. For each chunk: slice bytes → BLAKE3 hash → CIDv1 → write <cid>.shard
  6. Build ModelManifest (CBOR) with all shard descriptors
  7. Wait for mDNS peer → publish ShardAnnouncement per shard on omni/shard/v1

Node B: omni-node fetch <cid>
  1. Listen for ShardAnnouncement on omni/shard/v1, or discover peer via mDNS
  2. Send ShardRequest{cid, offset=0, max_bytes=64MB} via request-response
  3. Node A: mmap shard → slice [0..64MB] → ShardResponse
  4. Node B: accumulate chunk → request next window → repeat
  5. All chunks received → verify BLAKE3 → verify CID → write <cid>.shard
```

---

## Workspace Directory Structure

```
OmniNode-Protocol/
│
├── Cargo.toml                          # Workspace manifest
├── Cargo.lock
├── rust-toolchain.toml                 # Rust 2024 edition, MSRV 1.85+
│
├── crates/
│   ├── omni-types/                     # Shared types, errors, config
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── node.rs                 # NodeId, PeerId wrappers, NodeCapability
│   │       ├── model.rs                # ModelManifest, ShardDescriptor (+ optional snip_v2), LayerRange, GgmlType
│   │       ├── pipeline.rs             # PipelineStage, PipelineSchedule, PipelineMessage, HiddenStateHeader, TensorDtype
│   │       ├── phase5.rs               # SnipV2ObjectId, SnipV2Lifecycle, SnipV2ObjectRef, InferenceCommitment, InferenceAttestation
│   │       ├── error.rs                # Unified error types (Network, Storage, GgufParse, ...)
│   │       └── config.rs              # NetConfig, StoreConfig, PipelineConfig
│   │
│   ├── omni-net/                       # Phase 1: P2P mesh networking (libp2p 0.55)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # OmniNet API handle (publish, request_shard_chunk, respond_shard)
│   │       ├── behaviour.rs            # Composed NetworkBehaviour: mDNS + Gossipsub + Identify + ShardXfer + TensorXfer
│   │       ├── swarm.rs                # Swarm lifecycle, event loop, command dispatch
│   │       ├── codec.rs                # ShardCodec: [u32 BE len][bincode] wire format for /omni/shard-xfer/1
│   │       ├── discovery.rs            # mDNS event handling and peer registration
│   │       ├── gossip.rs               # GossipManager: topic subscriptions and publishing
│   │       ├── events.rs               # OmniNetEvent enum (clean domain events, no raw libp2p)
│   │       ├── tensor_codec.rs         # TensorCodec: [u32 BE len][bincode] for /omni/tensor-xfer/1
│   │       ├── capability.rs           # Custom capability advertisement protocol (deferred)
│   │       ├── nat.rs                  # AutoNAT, relay, DCUtR (deferred)
│   │       └── transport.rs           # TCP/Noise fallback transport (deferred)
│   │
│   ├── omni-store/                     # Phase 2: GGUF model sharding & shard storage
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # OmniStore API (ingest_model, announce_shards, fetch, serve)
│   │       ├── gguf.rs                 # Zero-copy GGUF v2/v3 parser (memmap2, all 13 metadata types)
│   │       ├── chunker.rs              # Layer-wise tensor classification and chunk planning
│   │       ├── content_id.rs           # BLAKE3 → CIDv1 content addressing
│   │       ├── manifest.rs             # ModelManifest build, CBOR serialization, JSON export
│   │       ├── store.rs                # On-disk shard store (~/.omninode/store/<cid>.shard)
│   │       ├── mmap.rs                 # Memory-mapped file I/O
│   │       ├── verify.rs               # BLAKE3 and CID integrity verification
│   │       ├── announce.rs             # Gossipsub shard announcements (bincode on omni/shard/v1)
│   │       ├── fetch.rs                # FetchManager: windowed 64MB chunk fetching state machine
│   │       ├── serve.rs                # Inbound request handler: mmap → slice → respond
│   │       ├── snip_v2.rs              # Phase 5 Stage 1: sum-node ingest-v2/download CLI adapter + pure parser + SnipV2Adapter trait
│   │       ├── snip_v2_artifacts.rs    # Phase 5 Stage 2: publish_to_snip / restore_from_snip / restore_manifest_from_snip
│   │       └── error.rs               # StoreError enum (crate-local; bridges SnipV2Error, adds ShardFileMissing & ShardLacksSnipRef)
│   │
│   ├── omni-bridge/                    # Phase 3: Rust↔Python FFI (PyO3 0.23)
│   │   ├── Cargo.toml                 # cdylib + rlib, depends on omni-store + omni-net + pyo3
│   │   └── src/
│   │       ├── lib.rs                 # #[pymodule] _omni_bridge entry point
│   │       ├── errors.rs              # PyOmniError / PyStoreError exception hierarchy
│   │       ├── runtime.rs             # OnceLock<tokio::Runtime> singleton for block_on()
│   │       ├── types.rs               # PyNetConfig, PyStoreConfig, PyLayerRange, PyShardDescriptor, PyModelManifest
│   │       ├── store.rs               # PyOmniStore: ingest_model, mmap_shard, has_shard, get_shard
│   │       ├── shard_view.rs          # PyShardView: zero-copy __getbuffer__ over memmap2::Mmap
│   │       ├── net.rs                 # PyOmniNet: publish, request_shard, request_tensor, next_event, shutdown + context manager
│   │       ├── pipeline.rs            # PyPipelineCoordinator, PyStageExecutor, PyPipelineConfig, PyPipelineCapability
│   │       └── events.rs             # PyNetEvent: flat struct with kind discriminator (shard + tensor events)
│   │
│   ├── omni-pipeline/                  # Phase 4: Pipeline-parallel inference coordination
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # Module declarations and re-exports
│   │       ├── coordinator.rs          # PipelineCoordinator: session lifecycle, produces PipelineAction
│   │       ├── executor.rs             # StageExecutor: per-node stage state, TensorRequest/Response builders
│   │       ├── planner.rs              # RAM-proportional layer-to-node assignment
│   │       ├── scheduler.rs            # GPipe micro-batch scheduling (MicroBatchSchedule, ScheduleCell)
│   │       ├── session.rs              # PipelineSession state machine (Forming → Scheduled → Running → Done)
│   │       ├── heartbeat.rs            # HeartbeatMonitor: 3s interval, 3× timeout liveness detection
│   │       ├── transport.rs            # PipelineMessage bincode encode/decode helpers
│   │       └── error.rs               # PipelineError enum (thiserror)
│   │
│   ├── omni-zkml/                      # Phase 5: Proof artifact flow + attestation envelope + chain abstraction & registry + chain wire + staleness + orchestration + Stage 11a proof backend trait + mock impl
│   │   ├── Cargo.toml                 # depends on omni-store + omni-types + blake3 + bincode + bs58 + libp2p-identity + serde + serde_json + chrono + thiserror + tracing
│   │   ├── src/
│   │   │   ├── lib.rs                 # module declarations and root re-exports
│   │   │   ├── artifact.rs            # Stage 3: ProofArtifact, ResponseArtifact, publish_proof_artifacts, build_commitment
│   │   │   ├── attestation.rs         # Stage 4: DOMAIN_TAG, CommitmentPayload, CommitmentDigest (+ Stage-5 serde), Signer trait, build_attestation
│   │   │   ├── chain.rs               # Stage 5: ChainClient trait, SubmissionReceipt, AttestationStatus
│   │   │   ├── registry.rs            # Stage 5/5.1/5.2: AttestationId, AttestationRecord (incl. submitted_at_block), LocalAttestationStatus, AttestationRegistry (incl. mark_submitted_with_block), submit/query workflows
│   │   │   ├── chain_wire.rs          # Stage 6: InferenceAttestationDigest, InferenceAttestationTxData, canonical/signing bytes, Ed25519 signing, chain-address bs58, ChainAttestationVector
│   │   │   ├── staleness.rs           # Stage 5.2: StalenessPolicy, StalenessPolicyError, is_record_stale (pure), mark_stale_if_overdue (workflow)
│   │   │   ├── orchestration.rs       # Stage 5.3: OrchestrationClient trait + submit_attestation_workflow_with_block + poll_attestations_workflow + sweep_stale_attestations_workflow + retry_dropped_attestations_workflow
│   │   │   ├── proof.rs               # Stage 11a: ProofBackend / ProofVerifier traits, MockProofBackend (mock-v1, mainnet-refused), MockProofVerifier, PublicInputs, ProofMetadata, ProofArtifactBody, produce_proof_artifact orchestrator
│   │   │   └── error.rs               # ProofArtifactError + SignerError + AttestationError + ChainClientError + RegistryError + ChainWireError + Stage 11a ProofBackendError + ProofVerifierError + ProofPipelineError (with Result / AttestationResult / RegistryResult / ChainWireResult)
│   │   └── tests/
│   │       ├── chain_attestation_vectors.rs        # Stage 6 integration test (verify-by-default, regen via OMNINODE_REGEN_VECTORS=1)
│   │       ├── proof_pipeline_vectors.rs           # Stage 11a integration test: 3 byte-stable vectors covering MockProofBackend → ProofArtifactBody → SNIP root → InferenceCommitment → chain digest → Ed25519 signature (regen via OMNINODE_REGEN_PROOF_PIPELINE_VECTORS=1)
│   │       └── fixtures/
│   │           ├── chain_attestation_vectors.json  # Stage 6 frozen deliverable: 3 chain attestation test vectors
│   │           └── proof_pipeline_vectors.json     # Stage 11a frozen fixture: 3 proof-pipeline vectors (CI hard-gated by stage11a-fixture-check)
│   │
│   ├── omni-sumchain/                  # Phase 5 Stage 7a + 7b + 9a + 9c: SUM Chain adapter (read/query default-on; submit + public chain primitives from crates.io v0.1.0 behind the `submit` cargo feature, default-off)
│   │   ├── Cargo.toml                 # depends on omni-zkml + omni-types + ureq + sumchain-primitives + sumchain-crypto + serde + serde_json + thiserror + tracing (dev: bincode1)
│   │   ├── README.md                  # operational setup (extra-alloc.json funding, env vars, live-test guide, Stage 7b submission flow)
│   │   ├── src/
│   │   │   ├── lib.rs                 # module decls + re-exports
│   │   │   ├── client.rs              # SumChainClient<T: JsonRpcTransport>, ChainClient + OrchestrationClient impls, inherent read helpers, v2_is_active, derived_verifier_address
│   │   │   ├── dto.rs                 # InferenceAttestationStatusInfo, InferenceAttestationInfo, BlockHeightInfo, BlockFinality, ChainParamsInfo (omninode_enabled_from_height + v2_enabled_from_height)
│   │   │   ├── rpc.rs                 # JsonRpcTransport trait, UreqTransport (sync HTTP), FakeJsonRpcTransport (Arc<Mutex<_>> test fixture)
│   │   │   ├── status.rs              # map_status_info: chain status JSON -> omni_zkml::AttestationStatus (strict variant matching)
│   │   │   ├── outer_sign.rs          # Stage 7b: outer_sign_transaction_v2 — TransactionV2::signing_hash() + sumchain_crypto::sign + SignedTransaction::new_v2
│   │   │   └── tx.rs                  # Stage 7b: build_and_submit_signed_transaction (4 gates → Stage 6 inner pipeline → local→chain type conversion → TransactionV2 → outer-sign → bare-hex sum_sendRawTransaction)
│   │   └── tests/
│   │       ├── unit_status_mapping.rs        # 10 hermetic status-mapping tests
│   │       ├── unit_dto.rs                   # 14 hermetic DTO parse tests (pre-patch + post-patch ChainParamsInfo including v2_enabled_from_height)
│   │       ├── unit_rpc_envelope.rs          # 16 hermetic RPC envelope + Stage 5.1 integration tests (includes Stage 7b happy-path receipt)
│   │       ├── unit_submit_construction.rs   # 15 hermetic Stage 7b construction tests (gate ordering, RPC caching, bare-hex shape, min-fee round-trip, {tx_hash} object + bare-string + 3 negative parse paths)
│   │       ├── parity_vendored_primitives.rs # 5 byte-equivalence tests: Stage 6 local digest/tx-data + bs58 address derivation == public chain types (sumchain-primitives v0.1.0) under bincode 1.3, plus Stage 9c TransactionV2 signing-hash + SignedTransaction hex-roundtrip stability
│   │       ├── chain_produced_fixture.rs     # Stage 9c.1: 5 tests pinning byte-roundtrip + SignedTransaction::hash() against real on-chain bytes from the 2026-05-19 mainnet smoke (`--features submit` gated)
│   │       ├── fixtures/
│   │       │   └── chain_produced_signed_tx.json # Stage 9c.1: raw signed_tx_hex + chain-team provenance + methodology notes for tx 0x3a9cbf85…c32a56
│   │       ├── stage6_wire_parity.rs         # 1 cross-crate smoke against the Stage 6 chain-team fixture
│   │       └── live_local_mirror.rs          # 5 #[ignore]'d live tests, env-gated by OMNINODE_SUMCHAIN_RPC_URL (+ OMNINODE_VERIFIER_SEED_HEX for Stage 7b submit roundtrip)
│   │
│   └── omni-node/                      # Binary: CLI entry point
│       ├── Cargo.toml                 # + omni-zkml + omni-sumchain + thiserror (dev: tempfile)
│       └── src/
│           ├── main.rs                # listen | shard <path> | fetch <cid> | send <msg> | operator <…>
│           └── operator.rs            # Stage 8a/8b/8c/9a/10a/11a: operator watch-activation | smoke[submit] | loop[+submit retry] | preflight | query [--tx-hash] | derive-address | registry list/show/summary (Stage 10a: + summary subcommand, + stable event= log markers for startup / chain_params / activation_state / registry_summary. Stage 11a: replaces synth_attestation placeholder with build_mock_v1_attestation (real MockProofBackend pipeline; hard-refuses mainnet); synth_attestation kept as #[cfg(test)] fixture only. Safe-by-default, hermetic-tested, warning-clean; smoke + retry gated behind `--features submit`)
│
├── pyproject.toml                     # maturin build config for omninode Python package
├── python/                            # Python ML package (Phase 3)
│   └── omninode/
│       ├── __init__.py                # Re-exports from native extension _omni_bridge
│       └── py.typed                   # PEP 561 type checker marker
│
├── contracts/                          # Phase 5: SUM Chain smart contracts
│
├── proto/                              # Protobuf schema definitions
│
├── .github/
│   └── workflows/
│       └── ci.yml                      # Stage 9c: cargo-side gates restored against public sumchain crates — default-build-test + default-tree-check (HARD GATE) + submit-build-test + submit-tree-check (HARD GATE) + stage6-fixture-check (HARD GATE); no secrets required
│
├── docs/                              # Architecture documentation
│   ├── mainnet-smoke-audit.md         # Stage 8b: frozen historical record of the 2026-05-19 first mainnet finalization
│   └── operator-runbook.md            # Stage 9a + 9b: canonical operator runbook (build matrix, daily commands, recovery, RUST_LOG, systemd sample, --help discoverability)
│
├── scripts/                           # Development scripts
│
├── tests/                              # Workspace-level integration tests
│
├── benches/                            # Workspace-level benchmarks (criterion)
│
└── README.md
```

---

## Workspace Cargo.toml

```toml
[workspace]
resolver = "2"
members = [
    "crates/omni-types",
    "crates/omni-net",
    "crates/omni-store",
    "crates/omni-bridge",
    "crates/omni-pipeline",
    "crates/omni-zkml",
    "crates/omni-node",
]

[workspace.package]
version      = "0.1.0"
edition      = "2024"
license      = "MIT OR Apache-2.0"
repository   = "https://github.com/SUM-INNOVATION/OmniNode-Protocol"
rust-version = "1.85"

[workspace.dependencies]
# Async runtime
tokio = { version = "1.43", features = ["full"] }

# Serialization
serde       = { version = "1.0",         features = ["derive"] }
serde_json  = "1.0"
bincode     = { version = "2.0.0-rc.3", features = ["serde"] }
ciborium    = "0.2"

# Error handling
thiserror = "2.0"
anyhow    = "1.0"

# Logging / tracing
tracing            = "0.1"
tracing-subscriber = { version = "0.3", features = ["fmt", "env-filter"] }

# ─── Phase 1: Networking ──────────────────────────────────────────────────────

libp2p = { version = "0.55", features = [
    "macros", "tokio", "noise", "quic", "tcp", "dns",
    "kad", "mdns", "gossipsub", "identify",
    "autonat", "relay", "dcutr", "request-response",
] }

libp2p-identity = "0.2"
prost           = "0.13"
prost-build     = "0.13"

# ─── Phase 2: Storage ────────────────────────────────────────────────────────
# iroh removed — unresolvable hickory-resolver conflict with libp2p 0.55.
# Shard transfer uses libp2p request-response instead.

memmap2    = "0.9"
blake3     = "1.5"
cid        = "0.11"
multihash  = "0.19"
tempfile   = "3.14"

# ─── Phase 3: FFI ────────────────────────────────────────────────────────────

pyo3 = { version = "0.23", features = ["extension-module"] }

# ─── Phase 4: Pipeline ───────────────────────────────────────────────────────

safetensors    = "0.7"
ndarray        = "0.16"
tokio-util     = "0.7"
dashmap        = "6.1"
priority-queue = "2.1"
petgraph       = "0.7"

# ─── Async Utilities ─────────────────────────────────────────────────────────

futures     = "0.3"
async-trait = "0.1"

# ─── Utilities ───────────────────────────────────────────────────────────────

bytes   = "1.9"
uuid    = { version = "1.11", features = ["v4"] }
clap    = { version = "4.5",  features = ["derive"] }
rand    = "0.9"
chrono  = { version = "0.4",  features = ["serde"] }

# ─── Internal Crates ─────────────────────────────────────────────────────────

omni-types    = { path = "crates/omni-types" }
omni-net      = { path = "crates/omni-net" }
omni-store    = { path = "crates/omni-store" }
omni-bridge   = { path = "crates/omni-bridge" }
omni-pipeline = { path = "crates/omni-pipeline" }
omni-zkml     = { path = "crates/omni-zkml" }
```

---

## Dependency Inventory

### Workspace-Wide (Shared Dependencies)

| Crate | Version | Purpose |
|---|---|---|
| `tokio` | 1.43 | Async runtime (features: full) |
| `serde` | 1.0 | Serialization framework (features: derive) |
| `serde_json` | 1.0 | JSON serialization |
| `bincode` | 2.0.0-rc.3 | Binary serialization (features: serde) |
| `ciborium` | 0.2 | CBOR serialization for shard manifests |
| `thiserror` | 2.0 | Ergonomic error type derivation |
| `anyhow` | 1.0 | Application-level error handling |
| `tracing` | 0.1 | Structured logging and instrumentation |
| `tracing-subscriber` | 0.3 | Log subscriber (features: fmt, env-filter) |
| `futures` | 0.3 | Async stream utilities (`StreamExt::select_next_some()`) |
| `async-trait` | 0.1 | Async trait methods (used by libp2p Codec impl) |
| `bytes` | 1.9 | Efficient byte buffer type |
| `clap` | 4.5 | CLI argument parser (features: derive) |
| `uuid` | 1.11 | Unique identifiers (features: v4) |
| `chrono` | 0.4 | Timestamps (features: serde) |
| `rand` | 0.9 | Random number generation |

### Phase 1: P2P Mesh Networking (`omni-net`)

| Crate | Version | Purpose |
|---|---|---|
| `libp2p` | 0.55 | Core P2P framework |
| — feature `macros` | — | `#[derive(NetworkBehaviour)]` proc-macro (required 0.53+) |
| — feature `noise` | — | Noise protocol encrypted channels |
| — feature `quic` | — | QUIC transport (UDP, NAT-friendly) |
| — feature `tcp` | — | TCP fallback transport |
| — feature `dns` | — | DNS resolution |
| — feature `kad` | — | Kademlia DHT (WAN discovery) |
| — feature `mdns` | — | mDNS (LAN discovery) |
| — feature `gossipsub` | — | Pub/sub messaging |
| — feature `identify` | — | Peer identification |
| — feature `autonat` | — | NAT detection |
| — feature `relay` | — | Circuit relay |
| — feature `dcutr` | — | Direct Connection Upgrade through Relay |
| — feature `request-response` | — | Custom request-response protocols (shard transfer) |
| — feature `tokio` | — | Tokio runtime integration |
| `libp2p-identity` | 0.2 | Peer identity / keypair management |
| `prost` | 0.13 | Protobuf serialization |
| `prost-build` | 0.13 | Build-time protobuf codegen |

### Phase 2: GGUF Model Sharding (`omni-store`)

| Crate | Version | Purpose |
|---|---|---|
| `memmap2` | 0.9 | Memory-mapped file I/O for zero-copy GGUF parsing and shard streaming |
| `blake3` | 1.5 | BLAKE3 hashing for content addressing |
| `cid` | 0.11 | CIDv1 construction (BLAKE3 hash → content identifier) |
| `multihash` | 0.19 | Multihash encoding for CID compatibility |
| `tempfile` | 3.14 | Temporary files for tests |

> **Note:** The GGUF parser is implemented from scratch within `omni-store`. No mature Rust GGUF crate exists. The parser is built directly from the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), supports v2 and v3, and handles all 13 metadata value types.

### Phase 3: FFI Bridge (`omni-bridge`)

**Rust side:**

| Crate | Version | Purpose |
|---|---|---|
| `pyo3` | 0.23 | Rust ↔ Python FFI bindings (features: extension-module) |
| `numpy` (PyO3) | 0.23 | Expose Rust buffers as numpy arrays |
| `maturin` | 1.10 | Build tool: compile Rust into Python wheel |

**Python side (`python/pyproject.toml`):**

| Library | Version | Purpose |
|---|---|---|
| `llama-cpp-python` | >=0.3.8 | llama.cpp Python bindings for GGUF inference |
| `mlx` | >=0.22 | Apple Silicon ML framework |
| `mlx-lm` | >=0.22 | MLX language model utilities |
| `numpy` | >=1.26 | Tensor manipulation and buffer interface |
| `safetensors` | >=0.4 | Safe tensor serialization |
| `pytest` | >=8.0 | Testing framework |

### Phase 4: Pipeline Parallelism (`omni-pipeline`)

| Crate | Version | Purpose |
|---|---|---|
| `uuid` | 1.11 | Session ID generation (v4) |
| `chrono` | 0.4 | Timestamps for session creation and heartbeats |
| `bincode` | 2.0.0-rc.3 | PipelineMessage serialization for gossipsub transport |
| `thiserror` | 2.0 | PipelineError enum derivation |
| `async-trait` | 0.1 | Async trait methods for TensorCodec (in omni-net) |

> **Note:** The original plan called for `safetensors`, `ndarray`, `petgraph`, `dashmap`, and `priority-queue`. In practice, none were needed — activations are raw bytes (not safetensors), scheduling is deterministic GPipe (not graph-based), and planning is a simple proportional split (not dynamic programming). These workspace dependencies remain available for future phases.

### Phase 5: zkML & Tokenomics (`omni-zkml`)

| Crate | Version | Purpose | Status |
|---|---|---|---|
| `halo2_proofs` | 0.3.2 | Halo2 IPA proving system (Pasta curves) — backs the Stage 11b.1.b / 11c bounded `halo2-mlp-v1` reference verifier | **Shipped** behind the opt-in `halo2-reference-verify` feature on `omni-node`; pulled by the standalone `tools/halo2_reference_regen/` package outside the workspace. Default `omni-node` build pulls zero halo2. |
| `pasta_curves` | 0.5.x | Pasta curves backing `halo2_proofs` | Transitive of `halo2_proofs` |
| `ezkl` | n/a | Originally planned as the small/medium ONNX prover | **Rejected at Stage 11b.1.** `v23.0.5` has no `LICENSE` file and no `Cargo.toml` `license` field — unsafe to depend on. Revisitable if upstream license posture changes. |
| `risc0-zkvm` | n/a | Originally planned as a general-purpose STARK prover for large-model paths | **Not depended upon; unevaluated.** Could become a future research-track candidate (Stage 11e+); not on the Stage 11d table. |
| `alloy` | n/a | Originally planned for EVM interaction | **Not depended upon.** SUM Chain is not EVM; existing chain integration uses `sumchain-primitives` / `sumchain-crypto` v0.1.0 (Stage 9c). |

**Smart contract tooling (pending SUM Chain specification):**

| Tool | Purpose |
|---|---|
| Foundry (forge, cast, anvil) | Solidity development, testing, deployment |
| OpenZeppelin Contracts 5.x | Battle-tested staking, access control primitives |

---

## Inter-Phase Dependency Graph

```
Phase 1: omni-net (P2P Mesh)
    │
    │  omni-store uses Gossipsub for shard announcements,
    │  request-response for shard transfer
    ▼
Phase 2: omni-store (Model Sharding)
    │
    │  omni-bridge receives mmap'd weight buffers from omni-store,
    │  exposes them as zero-copy numpy arrays across FFI
    ▼
Phase 3: omni-bridge (FFI → Python Inference)
    │
    │  omni-pipeline orchestrates multi-node inference:
    │  omni-net for tensor transport, omni-store for shard loading,
    │  omni-bridge for per-stage inference execution
    ▼
Phase 4: omni-pipeline (Pipeline Parallelism)
    │
    │  omni-zkml wraps inference in zk proofs,
    │  aggregates per-stage proofs, submits on-chain
    ▼
Phase 5: omni-zkml + contracts (Verification & Tokenomics)
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Hidden state transfer latency dominates inference time | Pipeline parallelism slower than single-node | F16 quantized hidden states (halve transfer size); prioritize LAN peers; tensor compression |
| GGUF format evolution (llama.cpp breaking changes) | Parser breaks on new models | Version the parser; GGUF v3+ support; format version detection |
| Apple Silicon unified memory path is MLX-version-dependent | Zero-copy fails on some MLX versions | Fallback to explicit copy path; runtime feature detection |
| Production proof generation too slow for real-time inference | Users wait minutes/hours per proof | Stage 11d.0 documents this as an explicit non-goal for the first production proof class; async/background proof generation + proof caching are deferred design choices. ezkl is **not** used (rejected at Stage 11b.1 due to licensing). |
| NAT traversal failure rate | Nodes behind strict NATs can't participate | Deploy relay infrastructure; WebRTC transport as fallback |
| SUM Chain specification not finalized | Phase 5 contracts may need redesign | Abstract chain interface; defer chain-specific code until spec is confirmed |

---

## References

### Core Technologies

1. **libp2p** — Modular peer-to-peer networking framework.
   [GitHub: libp2p/rust-libp2p](https://github.com/libp2p/rust-libp2p) | [Docs](https://docs.rs/libp2p/latest/libp2p/)

2. **llama.cpp** — LLM inference in C/C++; defines the GGUF format.
   [GitHub: ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

3. **llama-cpp-python** — Python bindings for llama.cpp.
   [GitHub: abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

4. **PyO3** — Rust bindings for the Python interpreter.
   [GitHub: PyO3/pyo3](https://github.com/PyO3/pyo3) | [Docs](https://pyo3.rs/)

5. **maturin** — Build and publish PyO3 crates as Python packages.
   [GitHub: PyO3/maturin](https://github.com/PyO3/maturin)

6. **safetensors** — Safe, zero-copy tensor serialization by Hugging Face.
   [GitHub: huggingface/safetensors](https://github.com/huggingface/safetensors)

7. **halo2** (Zcash) — PLONK-based zero-knowledge proving system. Backs the Stage 11b.1.b / 11c bounded `halo2-mlp-v1` reference verifier.
   [GitHub: zcash/halo2](https://github.com/zcash/halo2)

8. **ezkl** — zkML engine for proving neural network inference via Halo2. **Not depended upon by OmniNode** — rejected at Stage 11b.1 due to license posture (`v23.0.5` has no `LICENSE` file or `Cargo.toml` `license` field). Listed here as a reference to the broader ecosystem; revisitable if upstream license posture changes.
   [GitHub: zkonduit/ezkl](https://github.com/zkonduit/ezkl)

9. **RISC Zero** — General-purpose zero-knowledge virtual machine. **Not depended upon by OmniNode**; unevaluated; could become a future research-track candidate.
   [GitHub: risc0/risc0](https://github.com/risc0/risc0) | [Docs](https://dev.risczero.com/)

### Reference Projects

9. **Exo** — Distributed AI inference on consumer hardware using pipeline parallelism. Key reference for placement engine design, heterogeneous device coordination, and MLX integration.
    [GitHub: exo-explore/exo](https://github.com/exo-explore/exo)

10. **Petals** — Collaborative LLM inference ("BitTorrent for LLMs"). Reference for the concept of distributed layer hosting across untrusted peers.
    [GitHub: bigscience-workshop/petals](https://github.com/bigscience-workshop/petals)

### Academic Papers

11. McMahan, B. et al. (2017). **"Communication-Efficient Learning of Deep Networks from Decentralized Data."** AISTATS 2017. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)
    — Foundational federated learning paper. Informs OmniNode's decentralized training architecture.

12. Huang, Y. et al. (2019). **"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism."** NeurIPS 2019. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
    — Defines the micro-batch pipeline parallelism strategy used in Phase 4.

13. Narayanan, D. et al. (2019). **"PipeDream: Generalized Pipeline Parallelism for DNN Training."** SOSP 2019.
    — Introduces 1F1B scheduling. Informs pipeline schedule design for inference workloads.

### Specifications

14. **GGUF Format Specification** — Binary format for quantized LLM weights.
    [Spec: ggml-org/ggml](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

15. **Noise Protocol Framework** — Cryptographic handshake protocol used by libp2p.
    [noiseprotocol.org](http://www.noiseprotocol.org/)

16. **QUIC (RFC 9000)** — UDP-based transport with built-in encryption and multiplexing.
    [RFC 9000](https://www.rfc-editor.org/rfc/rfc9000)

17. **CIDv1 Specification** — Content Identifier format for content-addressed systems.
    [multiformats/cid](https://github.com/multiformats/cid)

18. Maymounkov, P. and Mazieres, D. (2002). **"Kademlia: A Peer-to-peer Information System Based on the XOR Metric."** IPTPS 2002.
    — Distributed hash table algorithm used for WAN peer discovery.

---

## License

Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE).

---

*OmniNode Protocol is open-source infrastructure for decentralized AGI. Built from the bottom up.*
