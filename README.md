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
| **Storage** | Model weights (GGUF files) are chunked by transformer block, content-addressed (BLAKE3 → CIDv1), and distributed via a custom 64 MiB sliding-window protocol over libp2p request-response | No centralized model hosting. Weights are resilient, deduplicated, and globally available |
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
| Phase 4 — Pipeline Parallelism | `omni-pipeline` | Planned |
| Phase 5 — zkML & Tokenomics | `omni-zkml` | Planned |

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

#### Phase 1b — WAN & Capabilities (Deferred)

| Component | Implementation | Status |
|---|---|---|
| WAN Discovery | Kademlia DHT — internet-scale peer lookup | ⏳ Deferred |
| NAT Traversal | AutoNAT detection → Circuit Relay → DCUtR hole-punching | ⏳ Deferred |
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
│   │       ├── model.rs                # ModelManifest, ShardDescriptor, LayerRange, GgmlType
│   │       ├── pipeline.rs             # PipelineStage, MicroBatch, HiddenState (stub)
│   │       ├── error.rs                # Unified error types (Network, Storage, GgufParse, ...)
│   │       └── config.rs              # NetConfig, StoreConfig
│   │
│   ├── omni-net/                       # Phase 1: P2P mesh networking (libp2p 0.55)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # OmniNet API handle (publish, request_shard_chunk, respond_shard)
│   │       ├── behaviour.rs            # Composed NetworkBehaviour: mDNS + Gossipsub + Identify + ShardXfer
│   │       ├── swarm.rs                # Swarm lifecycle, event loop, command dispatch
│   │       ├── codec.rs                # ShardCodec: [u32 BE len][bincode] wire format for /omni/shard-xfer/1
│   │       ├── discovery.rs            # mDNS event handling and peer registration
│   │       ├── gossip.rs               # GossipManager: topic subscriptions and publishing
│   │       ├── events.rs               # OmniNetEvent enum (clean domain events, no raw libp2p)
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
│   │       └── error.rs               # StoreError enum (crate-local)
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
│   │       ├── net.rs                 # PyOmniNet: publish, request_shard, next_event, shutdown + context manager
│   │       └── events.rs             # PyNetEvent: flat struct with kind discriminator
│   │
│   ├── omni-pipeline/                  # Phase 4: Pipeline parallelism (stub)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   │
│   ├── omni-zkml/                      # Phase 5: Zero-knowledge proofs (stub)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   │
│   └── omni-node/                      # Binary: CLI entry point
│       ├── Cargo.toml
│       └── src/
│           └── main.rs                # listen | shard <path> | fetch <cid> | send <msg>
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
├── docs/                              # Architecture documentation
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
| ezkl proof generation too slow for real-time inference | Users wait minutes for proof | Async/background proof generation; RISC Zero for faster (larger) proofs; proof caching |
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

7. **ezkl** — zkML engine for proving neural network inference via Halo2.
   [GitHub: zkonduit/ezkl](https://github.com/zkonduit/ezkl)

8. **RISC Zero** — General-purpose zero-knowledge virtual machine.
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
