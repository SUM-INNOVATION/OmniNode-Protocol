# OmniNode Protocol

**A Trustless, Peer-to-Peer AI Inference Network — Public Utility Infrastructure for AGI**

> Any device with a chip can become a node. Pool low-power devices into an omnipotent network.

---

## Executive Summary

OmniNode Protocol is a decentralized AI inference network that transforms consumer hardware — MacBooks, PCs, mobile devices — into a unified compute fabric capable of running massive Large Language Models (LLMs). Rather than relying on centralized GPU clusters, OmniNode distributes model execution across a peer-to-peer mesh, where each node contributes its memory and compute to a collective inference pipeline.

The protocol is built on four pillars:

| Pillar | Mechanism | Outcome |
|---|---|---|
| **Compute** | Pipeline parallelism shards model layers across devices, routing hidden state tensors over a low-latency P2P mesh | Consumer devices pool their unified memory to run models that no single device could hold |
| **Storage** | Model weights (GGUF files) are chunked by transformer block, content-addressed (CID), and distributed via IPFS. Nodes dynamically fetch only the shards they need | No centralized model hosting. Weights are resilient, deduplicated, and globally available |
| **Privacy** | Federated Learning allows contributors to train locally on private data, uploading only mathematical weight gradients — never raw data | Data sovereignty is preserved. No central entity sees user data |
| **Incentives** | zkML (Zero-Knowledge Machine Learning) cryptographically proves correct inference. A Financial RLHF system stakes tokens, rewards quality, and slashes dishonest nodes | Trustless verification. Economic alignment between node operators and end users |

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
    │  │  │  │  (IPFS/iroh)  │    │  (libp2p)     │                  │
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

## 5-Phase Implementation Roadmap

The protocol is constructed strictly bottom-up. Each phase produces a working milestone that can be demonstrated independently.

### Phase 1: P2P Mesh Networking

**Crate:** `omni-net` | **Foundation:** `omni-types` | **Status: ✅ Phase 1a complete — pending two-Mac LAN verification**

Build the communication substrate. Nodes discover each other, advertise hardware capabilities, and establish encrypted, low-latency connections.

#### Phase 1a — LAN Mesh (Implemented)

| Component | Implementation | Status |
|---|---|---|
| Transport | QUIC/v1 over UDP (TLS 1.3 built-in, no separate Noise step) | ✅ Done |
| LAN Discovery | mDNS — zero-configuration local peer discovery | ✅ Done |
| Messaging | Gossipsub pub/sub with signed messages (`ValidationMode::Strict`) | ✅ Done |
| Peer Exchange | Identify protocol — `/omni-node/0.1.0` | ✅ Done |
| Node API | `OmniNet` handle: `publish()`, `next_event()`, `shutdown()` over async channels | ✅ Done |
| CLI | `omni-node listen` / `omni-node send "<message>"` (two-Mac test harness) | ✅ Done |

#### Phase 1b — WAN & Capabilities (Deferred)

| Component | Implementation | Status |
|---|---|---|
| WAN Discovery | Kademlia DHT — internet-scale peer lookup | ⏳ Deferred |
| NAT Traversal | AutoNAT detection → Circuit Relay → DCUtR hole-punching | ⏳ Deferred |
| TCP Fallback | TCP + Noise transport for non-QUIC peers | ⏳ Deferred |
| Capability Ads | Custom request-response protocol — advertise RAM, platform, loaded layers | ⏳ Deferred |
| Codec | Custom request-response framing codec | ⏳ Deferred |

**Gossipsub Topics:**
- `omni/test/v1` — integration test messages
- `omni/capability/v1` — periodic hardware capability heartbeats
- `omni/shard/v1` — shard availability announcements
- `omni/pipeline/v1` — pipeline coordination messages
- `omni/proof/v1` — zk proof announcements

**Verification (pending):**
```bash
# Mac 1 (listener)
RUST_LOG=info cargo run --bin omni-node -- listen

# Mac 2 (sender)
RUST_LOG=info cargo run --bin omni-node -- send "Hello from OmniNode"
```

**Milestone:** Two Apple Silicon Macs on the same LAN discover each other via mDNS and exchange a Gossipsub message on `omni/test/v1`.

---

### Phase 2: IPFS Model Sharding

**Crate:** `omni-store` | **Depends on:** `omni-net`, `omni-types`

Chunk model weight files, content-address each chunk, distribute them across the mesh, and stream them into memory on demand.

| Component | Implementation |
|---|---|
| GGUF Parsing | Custom zero-copy parser — memory-maps the file, reads header/metadata/tensor index without loading tensor data |
| Chunking Strategy | Layer-wise — each chunk contains all tensors for N consecutive transformer blocks (preserves inference locality) |
| Content Addressing | BLAKE3 hash → CIDv1 for each chunk |
| Distribution | iroh blob protocol — efficient large-blob transfer with range requests over QUIC |
| Memory Streaming | `memmap2` — memory-mapped shard files, OS pages in data on demand |
| Shard Registry | Distributed via Kademlia DHT — maps CID → list of PeerIds that hold the shard |

**Chunking Example (LLaMA 7B, 32 layers):**
```
Chunk 0: embedding layer + blocks 0-3    → CID_0  (~510 MB)
Chunk 1: blocks 4-7                       → CID_1
Chunk 2: blocks 8-11                      → CID_2
  ...
Chunk 8: blocks 28-31 + output head       → CID_8
```

**Shard Manifest Format:**
```json
{
  "model_name": "llama-7b-q4_k_m",
  "model_hash": "blake3:abc123...",
  "architecture": "llama",
  "total_layers": 32,
  "quantization": "Q4_K_M",
  "total_size_bytes": 4080218931,
  "shards": [
    {
      "shard_index": 0,
      "cid": "bafybeig...",
      "layer_range": [0, 3],
      "includes_embedding": true,
      "size_bytes": 510027366,
      "blake3_hash": "deadbeef..."
    }
  ]
}
```

**Why iroh over rust-ipfs:**
1. Actively maintained by n0.computer (rust-ipfs is unmaintained)
2. BLAKE3-native content addressing (faster than SHA-256)
3. Built-in QUIC connectivity that complements the libp2p mesh
4. `iroh-blobs` provides efficient large-blob transfer with range requests

**Milestone:** CLI command `omni-node --shard <model.gguf>` chunks a model, distributes shards across the mesh, and reconstructs them on a different node.

---

### Phase 3: FFI Bridge & Local Inference

**Crate:** `omni-bridge` | **Python package:** `python/omninode` | **Depends on:** `omni-store`, `omni-types`

Bridge the Rust storage layer to Python ML backends. The critical optimization is zero-copy weight transfer on Apple Silicon unified memory.

| Component | Implementation |
|---|---|
| FFI Framework | PyO3 + maturin — compile Rust into a native Python extension module |
| Weight Loading | Expose `memmap2::Mmap` regions as numpy arrays across the FFI boundary (zero-copy) |
| llama.cpp Backend | `llama-cpp-python` — GGUF inference on CPU/CUDA/Metal |
| MLX Backend | Apple's `mlx` framework — native Apple Silicon GPU acceleration |
| Engine Abstraction | Python ABC `InferenceEngine` with `load_weights()` and `forward()` methods |

**Apple Silicon Unified Memory Path (Zero-Copy):**
```
Rust mmap (GGUF shard file)
    │  ← file is in unified memory (CPU + GPU share physical RAM)
    ▼
PyO3 exposes buffer pointer to Python
    │  ← no copy: numpy wraps the Rust pointer
    ▼
MLX mx.array wraps the numpy buffer
    │  ← no copy: Metal GPU reads directly from the same physical memory
    ▼
Inference executes on GPU — zero memory copies from disk to compute
```

**Milestone:** CLI command `omni-node --infer "Hello, world"` loads a GGUF model via Rust, passes weights to Python, and generates tokens.

---

### Phase 4: Pipeline Parallelism

**Crate:** `omni-pipeline` | **Depends on:** `omni-net`, `omni-store`, `omni-bridge`, `omni-types`

Distribute inference across multiple nodes. Each node executes a contiguous range of model layers, forwarding hidden state tensors to the next node over the mesh.

| Component | Implementation |
|---|---|
| Pipeline Schedule | GPipe-style micro-batching — prompts split into M micro-batches, staggered across stages |
| Tensor Serialization | safetensors wire format — hidden states serialized between pipeline stages |
| Placement Engine | Dynamic programming — bin-pack layers to nodes minimizing `max(stage_latency) + communication_time` |
| Latency Probing | RTT measurements over the mesh to inform placement decisions |
| Fault Tolerance | Heartbeat (5s interval, 15s timeout) → re-assign failed node's layers → re-queue in-flight micro-batches |
| Coordinator | Orchestrator managing pipeline lifecycle: placement → shard loading → execution → result collection |

**Pipeline Schedule (GPipe Micro-Batching):**
```
Time →
Stage 0 (Node A): [MB0] [MB1] [MB2] [MB3]
Stage 1 (Node B):       [MB0] [MB1] [MB2] [MB3]
Stage 2 (Node C):             [MB0] [MB1] [MB2] [MB3]
Stage 3 (Node D):                   [MB0] [MB1] [MB2] [MB3]
```

**Placement Algorithm:**
```
Inputs:
  nodes:  [(node_id, vram_bytes, ram_bytes, flops, bandwidth)]
  layers: [(layer_index, param_bytes, activation_bytes, flop_cost)]

Objective:  minimize max(stage_latency) + inter_stage_communication_time
Constraint: sum(layer_memory) per node ≤ node_available_memory

Solution:   O(L × N) dynamic programming (partition problem variant)
```

**Milestone:** CLI command `omni-node --pipeline-infer "Explain quantum computing"` distributes inference across 2+ nodes on the mesh and streams the response.

---

### Phase 5: zkML & SUM Chain Tokenomics

**Crate:** `omni-zkml` | **Smart Contracts:** `contracts/` | **Depends on:** `omni-pipeline`, `omni-net`, `omni-types`

Cryptographically prove that inference was executed correctly. Tie proofs to a staking/slashing economy on SUM Chain.

| Component | Implementation |
|---|---|
| Proof Generation | Dual backend: ezkl (Halo2 SNARK) for small-medium models, RISC Zero (STARK) for general computation |
| Proof Aggregation | Combine per-stage proofs into a single composite proof before on-chain submission |
| On-Chain Verification | Smart contract verifies aggregated proof and triggers reward/slash |
| Staking | Nodes stake SUM tokens to join the network; 7-day unbonding period |
| Slashing | Provably incorrect inference → stake is partially burned |
| Financial RLHF | `reward_i = compute_share_i × quality_score × stake_weight_i × block_reward` |

**Dual Prover Strategy:**

| Backend | Best For | Proof Type | Trade-off |
|---|---|---|---|
| ezkl (Halo2) | Small-medium models, high throughput | SNARK | Faster proving, smaller proofs, model-specific circuit |
| RISC Zero | Large models, general computation | STARK | Slower proving, larger proofs, proves arbitrary Rust code |

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

## Workspace Directory Structure

```
OmniNode-Protocol/
│
├── Cargo.toml                          # Workspace manifest
├── Cargo.lock
├── rust-toolchain.toml                 # Rust 2024 edition, MSRV 1.85+
├── deny.toml                           # cargo-deny: license & advisory audit
│
├── .github/
│   └── workflows/
│       ├── ci.yml                      # Lint → test → build (all crates)
│       └── release.yml                 # Tagged release builds
│
├── crates/
│   ├── omni-types/                     # Shared types, errors, config
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── node.rs                 # NodeId, PeerId wrappers, NodeCapability
│   │       ├── model.rs                # ModelManifest, ShardDescriptor, LayerRange
│   │       ├── pipeline.rs             # PipelineStage, MicroBatch, HiddenState
│   │       ├── error.rs                # Unified error types
│   │       └── config.rs              # Global configuration structs
│   │
│   ├── omni-net/                       # Phase 1: P2P mesh networking
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── transport.rs            # QUIC + Noise transport
│   │       ├── discovery.rs            # mDNS (LAN) + Kademlia (WAN)
│   │       ├── capability.rs           # Custom capability advertisement protocol
│   │       ├── gossip.rs               # Gossipsub pub/sub layer
│   │       ├── nat.rs                  # AutoNAT, relay, DCUtR
│   │       ├── behaviour.rs            # Composed NetworkBehaviour
│   │       ├── swarm.rs                # Swarm lifecycle management
│   │       ├── codec.rs                # Request-response codec
│   │       └── events.rs              # Network event types and handlers
│   │
│   ├── omni-store/                     # Phase 2: IPFS model sharding
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── gguf.rs                 # GGUF file parser (zero-copy, from spec)
│   │       ├── chunker.rs              # Layer-wise chunking logic
│   │       ├── cid.rs                  # BLAKE3 → CIDv1 generation
│   │       ├── manifest.rs             # Shard manifest (CBOR-serialized registry)
│   │       ├── ipfs.rs                 # iroh blob integration
│   │       ├── mmap.rs                 # Memory-mapped shard streaming
│   │       ├── registry.rs             # Distributed shard → peer registry
│   │       └── verification.rs        # CID integrity verification on download
│   │
│   ├── omni-bridge/                    # Phase 3: Rust↔Python FFI
│   │   ├── Cargo.toml                  # crate-type = ["cdylib"]
│   │   ├── pyproject.toml              # maturin build config
│   │   └── src/
│   │       ├── lib.rs                  # #[pymodule] definition
│   │       ├── shard_loader.rs         # Expose mmap'd buffers as numpy arrays
│   │       ├── inference.rs            # Bidirectional inference FFI
│   │       ├── memory.rs               # Unified memory management
│   │       └── config.rs              # Runtime config across FFI boundary
│   │
│   ├── omni-pipeline/                  # Phase 4: Pipeline parallelism
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── scheduler.rs            # GPipe / 1F1B micro-batch scheduling
│   │       ├── router.rs               # Layer-to-node mapping
│   │       ├── tensor_serde.rs         # Hidden state serialization (safetensors)
│   │       ├── stage.rs                # Pipeline stage execution
│   │       ├── latency.rs              # RTT-aware scheduling
│   │       ├── fault.rs                # Heartbeat, re-routing on node dropout
│   │       ├── coordinator.rs          # Pipeline lifecycle orchestrator
│   │       └── placement.rs           # DP-based layer placement algorithm
│   │
│   ├── omni-zkml/                      # Phase 5: Zero-knowledge proofs
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── prover.rs               # ezkl / RISC Zero proof generation
│   │       ├── verifier.rs             # Local proof verification
│   │       ├── aggregator.rs           # Multi-stage proof aggregation
│   │       ├── onnx.rs                 # Model → ONNX export for ezkl
│   │       └── receipt.rs             # RISC Zero receipt types
│   │
│   └── omni-node/                      # Binary: full node executable
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs                 # CLI entry point (clap)
│           ├── cli.rs                  # CLI argument definitions
│           ├── daemon.rs               # Long-running node daemon
│           ├── api.rs                  # Local HTTP API for node management
│           └── metrics.rs             # Prometheus metrics exporter
│
├── python/
│   ├── omninode/                       # Python ML package
│   │   ├── __init__.py
│   │   ├── inference/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py               # Abstract InferenceEngine ABC
│   │   │   ├── llama_engine.py         # llama-cpp-python backend
│   │   │   ├── mlx_engine.py           # MLX backend (Apple Silicon)
│   │   │   └── weight_loader.py       # Load weights from Rust mmap'd buffers
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── stage_worker.py         # Python-side pipeline stage worker
│   │   │   └── tensor_io.py           # Hidden state (de)serialization
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── device.py               # Device detection (CUDA, MPS, CPU)
│   │       └── memory.py              # Memory profiling utilities
│   ├── tests/
│   │   ├── test_inference.py
│   │   ├── test_weight_loading.py
│   │   └── test_pipeline_stage.py
│   ├── pyproject.toml
│   └── requirements.txt
│
├── contracts/                          # Phase 5: SUM Chain smart contracts
│   ├── README.md                       # SUM Chain integration notes
│   ├── src/
│   │   ├── staking.sol                 # Staking / slashing logic
│   │   ├── verifier.sol                # On-chain zk proof verifier
│   │   ├── rewards.sol                 # Financial RLHF reward distribution
│   │   └── registry.sol               # On-chain node + model registry
│   └── test/
│
├── proto/                              # Protobuf schema definitions
│   ├── capability.proto                # Node capability advertisement
│   ├── pipeline.proto                  # Pipeline stage messages
│   └── manifest.proto                 # Model manifest schema
│
├── docs/
│   ├── architecture.md
│   ├── phase1-networking.md
│   ├── phase2-storage.md
│   ├── phase3-ffi.md
│   ├── phase4-pipeline.md
│   ├── phase5-zkml.md
│   ├── glossary.md
│   └── threat-model.md
│
├── scripts/
│   ├── dev-setup.sh                    # Developer environment setup
│   ├── run-local-cluster.sh            # Spin up N local nodes for testing
│   └── benchmark.sh                   # End-to-end benchmark runner
│
├── tests/                              # Workspace-level integration tests
│   └── integration/
│       ├── mesh_formation.rs
│       ├── shard_distribution.rs
│       └── pipeline_inference.rs
│
├── benches/                            # Workspace-level benchmarks (criterion)
│   ├── shard_throughput.rs
│   ├── tensor_serde.rs
│   └── mesh_latency.rs
│
├── README.md
├── LICENSE                             # MIT OR Apache-2.0
├── CONTRIBUTING.md
└── CHANGELOG.md
```

---

## Dependency Inventory

### Workspace-Wide (Shared Dependencies)

| Crate | Version | Purpose |
|---|---|---|
| `tokio` | 1.43 | Async runtime (features: full) |
| `serde` | 1.0 | Serialization framework (features: derive) |
| `serde_json` | 1.0 | JSON serialization |
| `thiserror` | 2.0 | Ergonomic error type derivation |
| `anyhow` | 1.0 | Application-level error handling |
| `tracing` | 0.1 | Structured logging and instrumentation |
| `tracing-subscriber` | 0.3 | Log subscriber (features: fmt, env-filter) |
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
| — feature `request-response` | — | Custom request-response protocols |
| — feature `tokio` | — | Tokio runtime integration |
| `libp2p-identity` | 0.2 | Peer identity / keypair management |
| `futures` | 0.3 | `StreamExt::select_next_some()` for the swarm event loop |
| `prost` | 0.13 | Protobuf serialization |
| `prost-build` | 0.13 | Build-time protobuf codegen |
| `bincode` | 2.0.0-rc.3 | Binary serialization for capability structs |

### Phase 2: IPFS Model Sharding (`omni-store`)

| Crate | Version | Purpose |
|---|---|---|
| `iroh` | 0.32 | Content-addressed blob storage (BLAKE3-native) |
| `iroh-blobs` | 0.32 | Blob transfer protocol with range requests |
| `memmap2` | 0.9 | Memory-mapped file I/O for weight streaming |
| `blake3` | 1.5 | BLAKE3 hashing for CID computation |
| `cid` | 0.11 | CID construction and parsing |
| `multihash` | 0.19 | Multihash encoding for CID compatibility |
| `ciborium` | 0.2 | CBOR serialization for shard manifests |
| `tempfile` | 3.14 | Temporary files during chunking operations |

> **Note:** The GGUF parser is implemented from scratch within `omni-store`. No mature Rust GGUF crate exists. The parser is built directly from the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

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
| `safetensors` | 0.7 | Tensor serialization for hidden state transfer |
| `ndarray` | 0.16 | N-dimensional arrays for tensor manipulation |
| `tokio-util` | 0.7 | Codec framework for framed tensor streams |
| `dashmap` | 6.1 | Concurrent hashmap for routing tables |
| `priority-queue` | 2.1 | Priority queue for latency-aware scheduling |
| `petgraph` | 0.7 | Graph data structure for pipeline DAG |

### Phase 5: zkML & Tokenomics (`omni-zkml`)

| Crate | Version | Purpose |
|---|---|---|
| `ezkl` | >=15.0 | zkML proof generation (Halo2 SNARK for ONNX models) |
| `risc0-zkvm` | 3.0 | General-purpose zkVM (STARK prover) |
| `risc0-zkvm-platform` | 3.0 | RISC Zero platform support |
| `alloy` | 0.9 | EVM interaction (ABI encoding, contract calls) |

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
    │  request-response for shard fetching,
    │  Kademlia DHT as distributed shard registry
    ▼
Phase 2: omni-store (IPFS Sharding)
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
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"
repository = "https://github.com/SUM-INNOVATION/OmniNode-Protocol"
rust-version = "1.85"

[workspace.dependencies]
# Async runtime
tokio = { version = "1.43", features = ["full"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "2.0.0-rc.3"
ciborium = "0.2"

# Error handling
thiserror = "2.0"
anyhow = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["fmt", "env-filter"] }

# Async stream utilities
futures = "0.3"

# Networking (Phase 1)
libp2p = { version = "0.55", features = [
    "macros", "tokio", "noise", "quic", "tcp", "dns",
    "kad", "mdns", "gossipsub", "identify",
    "autonat", "relay", "dcutr", "request-response",
] }

# Content-addressed storage (Phase 2)
iroh = "0.32"
iroh-blobs = "0.32"
memmap2 = "0.9"
blake3 = "1.5"

# FFI (Phase 3)
pyo3 = { version = "0.23", features = ["extension-module"] }

# Tensor serialization (Phase 4)
safetensors = "0.7"

# Utilities
bytes = "1.9"
uuid = { version = "1.11", features = ["v4"] }
clap = { version = "4.5", features = ["derive"] }
rand = "0.9"
chrono = { version = "0.4", features = ["serde"] }
dashmap = "6.1"

# Internal crates
omni-types = { path = "crates/omni-types" }
omni-net = { path = "crates/omni-net" }
omni-store = { path = "crates/omni-store" }
omni-bridge = { path = "crates/omni-bridge" }
omni-pipeline = { path = "crates/omni-pipeline" }
omni-zkml = { path = "crates/omni-zkml" }
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Hidden state transfer latency dominates inference time | Pipeline parallelism slower than single-node | F16 quantized hidden states (halve transfer size); prioritize LAN peers; tensor compression |
| GGUF format evolution (llama.cpp breaking changes) | Parser breaks on new models | Version the parser; GGUF v3+ support; format version detection |
| iroh API instability (pre-1.0) | Breakage on updates | Pin exact version; wrap iroh in abstraction layer; maintain fork if needed |
| Apple Silicon unified memory path is MLX-version-dependent | Zero-copy fails on some MLX versions | Fallback to explicit copy path; runtime feature detection |
| ezkl proof generation too slow for real-time inference | Users wait minutes for proof | Async/background proof generation; RISC Zero for faster (larger) proofs; proof caching |
| NAT traversal failure rate | Nodes behind strict NATs can't participate | Deploy relay infrastructure; WebRTC transport as fallback |
| SUM Chain specification not finalized | Phase 5 contracts may need redesign | Abstract chain interface; defer chain-specific code until spec is confirmed |

---

## References

### Core Technologies

1. **libp2p** — Modular peer-to-peer networking framework.
   [GitHub: libp2p/rust-libp2p](https://github.com/libp2p/rust-libp2p) | [Docs](https://docs.rs/libp2p/latest/libp2p/)

2. **iroh** — Content-addressed blob storage and connectivity by n0.computer.
   [GitHub: n0-computer/iroh](https://github.com/n0-computer/iroh) | [Docs](https://docs.rs/iroh)

3. **llama.cpp** — LLM inference in C/C++; defines the GGUF format.
   [GitHub: ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

4. **llama-cpp-python** — Python bindings for llama.cpp.
   [GitHub: abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

5. **PyO3** — Rust bindings for the Python interpreter.
   [GitHub: PyO3/pyo3](https://github.com/PyO3/pyo3) | [Docs](https://pyo3.rs/)

6. **maturin** — Build and publish PyO3 crates as Python packages.
   [GitHub: PyO3/maturin](https://github.com/PyO3/maturin)

7. **safetensors** — Safe, zero-copy tensor serialization by Hugging Face.
   [GitHub: huggingface/safetensors](https://github.com/huggingface/safetensors)

8. **ezkl** — zkML engine for proving neural network inference via Halo2.
   [GitHub: zkonduit/ezkl](https://github.com/zkonduit/ezkl)

9. **RISC Zero** — General-purpose zero-knowledge virtual machine.
   [GitHub: risc0/risc0](https://github.com/risc0/risc0) | [Docs](https://dev.risczero.com/)

### Reference Projects

10. **Exo** — Distributed AI inference on consumer hardware using pipeline parallelism. Key reference for placement engine design, heterogeneous device coordination, and MLX integration.
    [GitHub: exo-explore/exo](https://github.com/exo-explore/exo)

11. **Petals** — Collaborative LLM inference ("BitTorrent for LLMs"). Reference for the concept of distributed layer hosting across untrusted peers.
    [GitHub: bigscience-workshop/petals](https://github.com/bigscience-workshop/petals)

### Academic Papers

12. McMahan, B. et al. (2017). **"Communication-Efficient Learning of Deep Networks from Decentralized Data."** AISTATS 2017. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)
    — Foundational federated learning paper. Informs OmniNode's decentralized training architecture.

13. Huang, Y. et al. (2019). **"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism."** NeurIPS 2019. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
    — Defines the micro-batch pipeline parallelism strategy used in Phase 4.

14. Narayanan, D. et al. (2019). **"PipeDream: Generalized Pipeline Parallelism for DNN Training."** SOSP 2019.
    — Introduces 1F1B scheduling. Informs pipeline schedule design for inference workloads.

### Specifications

15. **GGUF Format Specification** — Binary format for quantized LLM weights.
    [Spec: ggml-org/ggml](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

16. **Noise Protocol Framework** — Cryptographic handshake protocol used by libp2p.
    [noiseprotocol.org](http://www.noiseprotocol.org/)

17. **QUIC (RFC 9000)** — UDP-based transport with built-in encryption and multiplexing.
    [RFC 9000](https://www.rfc-editor.org/rfc/rfc9000)

18. **CIDv1 Specification** — Content Identifier format for content-addressed systems.
    [multiformats/cid](https://github.com/multiformats/cid)

19. Maymounkov, P. and Mazieres, D. (2002). **"Kademlia: A Peer-to-peer Information System Based on the XOR Metric."** IPTPS 2002.
    — Distributed hash table algorithm used for WAN peer discovery.

---

## License

Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE).

---

*OmniNode Protocol is open-source infrastructure for decentralized AGI. Built from the bottom up.*
