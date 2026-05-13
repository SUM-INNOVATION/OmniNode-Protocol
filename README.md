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
| **Phase 4** — Pipeline Parallelism | `omni-pipeline` | **Complete** |
| **Phase 5 Stage 1** — SNIP V2 types + CLI adapter | `omni-types`, `omni-store` | **Complete** |
| **Phase 5 Stage 2** — SNIP-backed model artifacts (publish / restore) | `omni-store` | **Complete** |
| **Phase 5 Stage 3** — Proof artifact flow (publish + commitment) | `omni-zkml` | **Complete** |
| **Phase 5 Stage 4** — Local verifier attestation envelope (canonical bytes + digest + Signer trait) | `omni-zkml` | **Complete** |
| Phase 5 Stage 5+ — zkML proof generation & SUM Chain Tokenomics | `omni-zkml`, `contracts/` | Planned |

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

### Phase 5 Stage 5+: zkML Proof Generation & SUM Chain Tokenomics — Planned

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
│   ├── omni-zkml/                      # Phase 5: Proof artifact flow + attestation envelope (later: zk proofs)
│   │   ├── Cargo.toml                 # depends on omni-store + omni-types + blake3 + bincode + serde + thiserror + tracing
│   │   └── src/
│   │       ├── lib.rs                 # module declarations and root re-exports
│   │       ├── artifact.rs            # Stage 3: ProofArtifact, ResponseArtifact, publish_proof_artifacts, build_commitment
│   │       ├── attestation.rs         # Stage 4: DOMAIN_TAG, CommitmentPayload, CommitmentDigest, Signer trait, build_attestation
│   │       └── error.rs               # ProofArtifactError + Stage-4 SignerError + AttestationError (with AttestationResult<T>)
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
| `uuid` | 1.11 | Session ID generation (v4) |
| `chrono` | 0.4 | Timestamps for session creation and heartbeats |
| `bincode` | 2.0.0-rc.3 | PipelineMessage serialization for gossipsub transport |
| `thiserror` | 2.0 | PipelineError enum derivation |
| `async-trait` | 0.1 | Async trait methods for TensorCodec (in omni-net) |

> **Note:** The original plan called for `safetensors`, `ndarray`, `petgraph`, `dashmap`, and `priority-queue`. In practice, none were needed — activations are raw bytes (not safetensors), scheduling is deterministic GPipe (not graph-based), and planning is a simple proportional split (not dynamic programming). These workspace dependencies remain available for future phases.

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
