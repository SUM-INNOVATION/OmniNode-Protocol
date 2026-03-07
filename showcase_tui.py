#!/usr/bin/env python3
"""
showcase_tui.py — OmniNode Phase 4: Pipeline-Parallel Distributed Inference
Native GGUF-to-MLX bridge: zero mlx_lm.load() — weights served entirely from
a local GGUF file via Apple's mx.load() low-level API.
Architecture (hidden_dim, layer count) is inferred at runtime from the file.
Sender owns embed_tokens + layers[0:n_sender_layers].
Receiver owns layers[n_sender_layers:] + norm + lm_head.
All communication is pure QUIC via request_tensor / tensor_received.
  hidden_dim == hidden_dim  →  hidden states (prefill seq_len>1, or decode seq_len==1)
  hidden_dim == 1           →  4-byte little-endian token ID (including EOS sentinel)
"""

import gc
import re
import sys
import time
from datetime import datetime

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_lm.models.llama import Model, ModelArgs

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from omninode import OmniNet, NetConfig, OmniStore

try:
    from mlx_lm.models.base import create_attention_mask as _mlx_cam
except ImportError:
    _mlx_cam = None

# ── Constants ─────────────────────────────────────────────────────────────────

SESSION_ID = "showcase-001"
DTYPE      = 0  # F16 metadata tag

# GGUF tensor name → MLX parameter path (static, non-block tensors)
_GGUF_STATIC: dict[str, str] = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

# Per-block suffix → MLX sub-path (applied as model.layers.{i}.{value})
_GGUF_BLK: dict[str, str] = {
    "attn_norm.weight":   "input_layernorm.weight",
    "ffn_norm.weight":    "post_attention_layernorm.weight",
    "attn_q.weight":      "self_attn.q_proj.weight",
    "attn_q.bias":        "self_attn.q_proj.bias",
    "attn_k.weight":      "self_attn.k_proj.weight",
    "attn_k.bias":        "self_attn.k_proj.bias",
    "attn_v.weight":      "self_attn.v_proj.weight",
    "attn_v.bias":        "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_gate.weight":    "mlp.gate_proj.weight",
    "ffn_up.weight":      "mlp.up_proj.weight",
    "ffn_down.weight":    "mlp.down_proj.weight",
}

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# ── Causal mask ───────────────────────────────────────────────────────────────

def _causal_mask(h: mx.array):
    T = h.shape[1]
    if T <= 1:
        return None
    if _mlx_cam is not None:
        return _mlx_cam(h)
    mask = mx.tril(mx.ones((T, T)))
    return mx.log(mask).astype(h.dtype)

# ── Model Shards ──────────────────────────────────────────────────────────────

class SenderShard:
    """Holds embed_tokens + layers[0:n_sender_layers]. Runs prefill and decode steps."""

    def __init__(self, model, n_sender_layers: int):
        self.embed_tokens = model.model.embed_tokens
        self.layers       = model.model.layers[:n_sender_layers]
        self._init_cache()

    def _init_cache(self):
        self.kv_cache = [KVCache() for _ in range(len(self.layers))]

    def prefill(self, token_ids: list[int]) -> mx.array:
        x    = mx.array([token_ids])
        h    = self.embed_tokens(x)
        mask = _causal_mask(h)
        for i, layer in enumerate(self.layers):
            h = layer(h, mask=mask, cache=self.kv_cache[i])
        mx.eval(h)
        return h

    def decode_step(self, token_id: int) -> mx.array:
        x = mx.array([[token_id]])
        h = self.embed_tokens(x)
        for i, layer in enumerate(self.layers):
            h = layer(h, mask=None, cache=self.kv_cache[i])
        mx.eval(h)
        return h

    def reset(self):
        self._init_cache()


class ReceiverShard:
    """Holds layers[n_sender_layers:] + norm + lm_head. Runs decode steps."""

    def __init__(self, model, n_sender_layers: int):
        self.layers = model.model.layers[n_sender_layers:]
        self.norm   = model.model.norm

        # Tied embeddings: no separate lm_head weight (e.g. Qwen 0.5B).
        if hasattr(model, "lm_head"):
            self.lm_head      = model.lm_head
            self.embed_tokens = None
        else:
            self.lm_head      = None
            self.embed_tokens = model.model.embed_tokens

        self._init_cache()

    def _init_cache(self):
        self.kv_cache = [KVCache() for _ in range(len(self.layers))]

    def decode_step(self, hidden: mx.array, is_prefill: bool = False) -> int:
        mask = _causal_mask(hidden) if is_prefill else None
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, mask=mask, cache=self.kv_cache[i])
        hidden = self.norm(hidden)

        if self.lm_head is not None:
            logits = self.lm_head(hidden[:, -1, :])
        else:
            # .as_linear() safely unpacks 4-bit quantized weights
            logits = self.embed_tokens.as_linear(hidden[:, -1, :])

        token_id = int(mx.argmax(logits, axis=-1).item())
        mx.eval(token_id)
        return token_id

    def reset(self):
        self._init_cache()

# ── Native GGUF → MLX Bridge ──────────────────────────────────────────────────

def _load_and_slice(gguf_path: str, is_sender: bool):
    """
    Native GGUF-to-MLX adapter. No mlx_lm.load() or HuggingFace weight download.

    Pipeline:
      1. mx.load() — Apple low-level GGUF parse; raw tensors + metadata dict
      2. Architecture extracted directly from weights/metadata
      3. GGUF key names translated to MLX Llama parameter paths
      4. ModelArgs populated from GGUF metadata
      5. Bare Model() instantiated (no weights), then injected via load_weights()
      6. Shard sliced; full model + raw weights dropped from unified memory
    """
    from transformers import AutoTokenizer

    # ── 1. Low-level load ────────────────────────────────────────────────────
    print(f"[GGUF] Loading {gguf_path} ...")
    weights, metadata = mx.load(gguf_path, return_metadata=True)

    # ── 2. Dynamic architecture extraction ───────────────────────────────────
    hidden_dim      = weights["token_embd.weight"].shape[1]
    vocab_size      = weights["token_embd.weight"].shape[0]
    total_layers    = len([k for k in weights if "attn_q.weight" in k])
    n_sender_layers = total_layers // 2

    arch = metadata.get("general.architecture", "llama")

    def _meta(key, default=None):
        val = metadata.get(f"{arch}.{key}", default)
        return val if val is not None else default

    n_heads    = int(_meta("attention.head_count", 32))
    kv_val     = _meta("attention.head_count_kv")
    n_kv_heads = int(kv_val) if kv_val is not None else n_heads

    print(f"[GGUF] arch={arch}  hidden={hidden_dim}  layers={total_layers}  heads={n_heads}/{n_kv_heads}  vocab={vocab_size}")
    print(f"[GGUF] Split: Sender layers 0-{n_sender_layers-1} | Receiver layers {n_sender_layers}-{total_layers-1}")

    # ── 3. GGUF key → MLX parameter path ─────────────────────────────────────
    mapped_weights: dict[str, mx.array] = {}
    for gguf_name, tensor in weights.items():
        if gguf_name in _GGUF_STATIC:
            mapped_weights[_GGUF_STATIC[gguf_name]] = tensor
            continue
        m = _BLK_RE.match(gguf_name)
        if m:
            i, suffix = m.group(1), m.group(2)
            if suffix in _GGUF_BLK:
                mapped_weights[f"model.layers.{i}.{_GGUF_BLK[suffix]}"] = tensor

    # ── 4. Bare-metal ModelArgs from GGUF metadata ────────────────────────────
    args = ModelArgs(
        model_type              = "llama",
        hidden_size             = hidden_dim,
        num_hidden_layers       = total_layers,
        intermediate_size       = int(_meta("feed_forward_length", hidden_dim * 4)),
        num_attention_heads     = n_heads,
        num_key_value_heads     = n_kv_heads,
        vocab_size              = vocab_size,
        rms_norm_eps            = float(_meta("attention.layer_norm_rms_epsilon", 1e-5)),
        max_position_embeddings = int(_meta("context_length", 2048)),
        rope_theta              = float(_meta("rope.freq_base", 10000.0)),
        rope_traditional        = True,
        rope_scaling            = None,
        tie_word_embeddings     = "output.weight" not in weights,
    )

    # ── 5. Instantiate + inject weights ──────────────────────────────────────
    model = Model(args)
    model.load_weights(list(mapped_weights.items()), strict=False)
    mx.eval(model.parameters())
    print(f"[GGUF] {len(mapped_weights)} tensors injected into bare Model()")

    # ── 6. Slice shard + RAM pool drop ───────────────────────────────────────
    shard = SenderShard(model, n_sender_layers) if is_sender else ReceiverShard(model, n_sender_layers)
    del model, weights, mapped_weights
    gc.collect()
    mx.metal.clear_cache()
    print(f"[GGUF] RAM pool drop complete.")

    # ── 7. Tokenizer ─────────────────────────────────────────────────────────
    print(f"[GGUF] Loading tokenizer (TinyLlama) ...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    return shard, tokenizer, hidden_dim


def load_from_omnistore(gguf_path: str, is_sender: bool):
    """
    Hybrid Lazy-Load: OmniStore proves Phase 2 CID/chunking (BLAKE3 hashes,
    CID generation, shard layout), then the native GGUF bridge handles weight
    ingestion and memory pooling.
    """
    print(f"\n[OmniStore] Ingesting {gguf_path} ...")

    # ── Phase 2 Verification ─────────────────────────────────────────────────
    store    = OmniStore()
    manifest = store.ingest_model(gguf_path)

    print(f"[OmniStore] model_name  : {manifest.model_name}")
    print(f"[OmniStore] model_hash  : {manifest.model_hash}")
    print(f"[OmniStore] architecture: {manifest.architecture}")
    print(f"[OmniStore] total_layers: {manifest.total_layers}")
    print(f"[OmniStore] shards      : {len(manifest.shards)}")
    for desc in manifest.shards:
        flags = []
        if desc.includes_embedding:
            flags.append("embedding")
        if desc.includes_output_head:
            flags.append("output_head")
        tag = f"  [{', '.join(flags)}]" if flags else ""
        print(
            f"[OmniStore]   shard {desc.shard_index}"
            f"  layers {desc.layer_range.start}-{desc.layer_range.end}"
            f"  cid={desc.cid[:16]}…"
            f"  {desc.size_bytes // (1024 * 1024)} MB"
            f"{tag}"
        )

    # ── Native GGUF bridge ────────────────────────────────────────────────────
    print()
    shard, tokenizer, hidden_dim = _load_and_slice(gguf_path, is_sender)
    return shard, tokenizer, hidden_dim

# ── State & Layout ────────────────────────────────────────────────────────────

class State:
    def __init__(self, role: str):
        self.role          = role.upper()
        self.stage         = "Waiting"
        self.stage_color   = "yellow"
        self.peer_id       = None
        self.status        = "Waiting for peer..."
        self.transfer_time = None
        self.prompt        = ""
        self.full_response = ""
        self.generating    = False
        self.token_count   = 0
        self._logs: list[str] = []

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._logs.append(f"[dim]{ts}[/dim]  {msg}")
        if len(self._logs) > 200:
            self._logs.pop(0)

    def set_stage(self, stage: str, color: str = "cyan") -> None:
        self.stage       = stage
        self.stage_color = color

    def reset_for_next_turn(self) -> None:
        self.prompt        = ""
        self.full_response = ""
        self.token_count   = 0
        self.generating    = False
        self.transfer_time = None
        self.set_stage("Ready", "green")


def build_layout(state: State) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=12),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3),
    )

    header_grid = Table.grid(expand=True)
    header_grid.add_column()
    header_grid.add_column(justify="right")
    header_grid.add_row(
        Text.assemble(
            ("OmniNode Pipeline Inference", "bold white"),
            ("  •  ", "dim"),
            (f"Role: {state.role}", "bold cyan"),
        ),
        Text.assemble(
            ("Session: ", "dim"),
            (SESSION_ID, "bold green"),
        ),
    )
    layout["header"].update(Panel(header_grid, style="on #1a1a2e"))

    peer_table = Table.grid(padding=(0, 1))
    peer_table.add_column(style="dim", justify="right")
    peer_table.add_column()

    if state.peer_id:
        short_id = state.peer_id[:20] + "..."
        peer_table.add_row("Peer:", f"[bold green]● {short_id}[/bold green]")
    else:
        peer_table.add_row("Peer:", "[dim]Searching...[/dim]")

    peer_table.add_row("Status:", f"[bold]{state.status}[/bold]")

    if state.transfer_time is not None:
        peer_table.add_row("Transfer:", f"[bold yellow]{state.transfer_time:.3f}s[/bold yellow]")

    if state.token_count > 0:
        peer_table.add_row("Tokens:", f"[bold yellow]{state.token_count}[/bold yellow]")

    layout["left"].update(Panel(
        peer_table,
        title="[bold]Peer Status[/bold]",
        border_style="blue",
        box=box.ROUNDED,
    ))

    stage_text = Text.assemble(
        ("► ", "bold"),
        (state.stage, f"bold {state.stage_color}"),
    )
    parts: list = [stage_text]

    if state.prompt:
        parts.append(Text(""))
        parts.append(Text.assemble(("Q: ", "bold dim"), (state.prompt, "italic dim")))

    if state.full_response:
        display = state.full_response[-600:]
        rt = Text(display, overflow="fold")
        if state.generating:
            rt.append("  [Generating...]", style="dim")
        parts.append(Text(""))
        parts.append(rt)

    layout["right"].update(Panel(
        Group(*parts),
        title="[bold]Pipeline Stage[/bold]",
        border_style="magenta",
        box=box.ROUNDED,
    ))

    log_lines = state._logs[-8:] if state._logs else ["[dim]No events yet...[/dim]"]
    layout["footer"].update(Panel(
        Text.from_markup("\n".join(log_lines)),
        title="[bold]Log Console[/bold]",
        border_style="dim",
        box=box.ROUNDED,
    ))

    return layout

# ── Receiver ──────────────────────────────────────────────────────────────────

def run_receiver(shard: ReceiverShard, tokenizer, hidden_dim: int) -> None:
    n_recv      = len(shard.layers)
    state       = State("receiver")
    net         = OmniNet(NetConfig())
    eos_id      = tokenizer.eos_token_id
    target_peer = None
    rbatch      = 0

    def _step(h: mx.array, is_prefill: bool, live) -> None:
        nonlocal rbatch
        token_id = shard.decode_step(h, is_prefill=is_prefill)

        rbatch += 1
        net.request_tensor(
            peer_id           = target_peer,
            session_id        = f"tok-{rbatch}",
            micro_batch_index = rbatch,
            from_stage        = 1,
            to_stage          = 0,
            seq_len           = 1,
            hidden_dim        = 1,
            dtype             = DTYPE,
            data              = token_id.to_bytes(4, "little"),
        )

        if token_id == eos_id:
            shard.reset()
            state.generating = False
            state.log("[bold green]EOS — sent to Sender. Resetting for next turn.[/bold green]")
            state.reset_for_next_turn()
        else:
            text = tokenizer.decode([token_id], skip_special_tokens=True)
            state.full_response += text
            state.token_count   += 1
            state.set_stage("Decoding", "yellow")
            state.log(f"Token {state.token_count}: {text!r}")

        live.update(build_layout(state))

    state.log(f"Receiver online — {n_recv} layers. Waiting for peer...")

    with Live(build_layout(state), refresh_per_second=8, screen=True) as live:
        while True:
            ev = net.next_event(timeout_secs=0.1)

            if ev is None:
                live.update(build_layout(state))
                continue

            if ev.kind == "peer_discovered":
                state.peer_id = ev.peer_id
                state.status  = "Discovered"
                state.log(f"Discovered: [bold]{ev.peer_id[:20]}...[/bold]")
                live.update(build_layout(state))

            elif ev.kind == "peer_connected":
                target_peer   = ev.peer_id
                state.peer_id = ev.peer_id
                state.status  = "Connected"
                state.set_stage("Ready", "green")
                state.log("Connected. Ready for pipeline inference.")
                live.update(build_layout(state))

            elif ev.kind == "tensor_received":
                net.respond_tensor(
                    channel_id        = ev.channel_id,
                    session_id        = ev.session_id,
                    micro_batch_index = ev.micro_batch_index,
                    stage_index       = ev.to_stage,
                    accepted          = True,
                )

                if ev.hidden_dim == hidden_dim:
                    raw = np.frombuffer(ev.data, dtype=np.float16).copy()
                    if ev.seq_len > 1:
                        h = mx.array(raw).reshape(1, ev.seq_len, hidden_dim)
                        state.generating = True
                        state.set_stage(f"Prefill ({n_recv} layers)", "yellow")
                        state.log(f"Prefill: hidden_states (1,{ev.seq_len},{hidden_dim})")
                        live.update(build_layout(state))
                        _step(h, is_prefill=True, live=live)
                    else:
                        h = mx.array(raw).reshape(1, 1, hidden_dim)
                        state.set_stage(f"Decode ({n_recv} layers)", "yellow")
                        _step(h, is_prefill=False, live=live)

            elif ev.kind == "peer_disconnected":
                state.status = "Disconnected"
                state.log("Peer disconnected.")
                live.update(build_layout(state))

# ── Sender ────────────────────────────────────────────────────────────────────

def run_sender(shard: SenderShard, tokenizer, hidden_dim: int) -> None:
    n_send  = len(shard.layers)
    console = Console()
    net     = OmniNet(NetConfig())
    eos_id  = tokenizer.eos_token_id

    console.print(f"[dim]Sender: {n_send} layers. Connecting via mDNS...[/dim]")
    target_peer = None
    while target_peer is None:
        ev = net.next_event(timeout_secs=0.5)
        if ev is None:
            continue
        if ev.kind == "peer_discovered":
            console.print(f"[dim]Discovered {ev.peer_id[:20]}...[/dim]")
        elif ev.kind == "peer_connected":
            target_peer = ev.peer_id
            console.print(f"[bold green]Connected to {ev.peer_id[:20]}...[/bold green]")

    turn   = 0
    mbatch = 0

    while True:
        try:
            user_prompt = console.input("\n[bold cyan]Sender >[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_prompt.lower() in ("quit", "exit", "q"):
            break
        if not user_prompt:
            continue

        turn += 1
        state         = State("sender")
        state.peer_id = target_peer
        state.status  = "Connected"
        state.prompt  = user_prompt

        messages   = [{"role": "user", "content": user_prompt}]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        token_ids = tokenizer.encode(prompt_str)
        console.print(f"[dim]Prefill: {len(token_ids)} tokens → {n_send} layers...[/dim]")
        t_pre  = time.time()
        hidden = shard.prefill(token_ids)
        console.print(f"[dim]Done in {time.time()-t_pre:.2f}s. Sending to Receiver...[/dim]")

        hidden_bytes = np.array(hidden).astype(np.float16).tobytes()

        with Live(build_layout(state), refresh_per_second=8, screen=True) as live:
            state.set_stage("Sending Prefill", "cyan")
            live.update(build_layout(state))

            mbatch += 1
            t0 = time.time()
            net.request_tensor(
                peer_id           = target_peer,
                session_id        = f"{SESSION_ID}-t{turn}",
                micro_batch_index = mbatch,
                from_stage        = 0,
                to_stage          = 1,
                seq_len           = hidden.shape[1],
                hidden_dim        = hidden_dim,
                dtype             = DTYPE,
                data              = hidden_bytes,
            )
            state.transfer_time = time.time() - t0
            state.log(f"Prefill sent: (1,{hidden.shape[1]},{hidden_dim}) — {len(hidden_bytes)//1024} KB in {state.transfer_time:.3f}s")
            state.set_stage("Waiting for Token 1", "yellow")
            live.update(build_layout(state))

            while True:
                ev = net.next_event(timeout_secs=0.1)

                if ev is None:
                    live.update(build_layout(state))
                    continue

                if ev.kind == "tensor_response_received":
                    if state.stage.startswith("Waiting") or state.stage.startswith("Sending"):
                        state.log("Prefill ACK — pipeline active.")
                        state.set_stage("Receiving Tokens", "magenta")
                        live.update(build_layout(state))

                elif ev.kind == "tensor_received":
                    net.respond_tensor(
                        channel_id        = ev.channel_id,
                        session_id        = ev.session_id,
                        micro_batch_index = ev.micro_batch_index,
                        stage_index       = ev.to_stage,
                        accepted          = True,
                    )

                    if ev.hidden_dim == 1:
                        token_id = int.from_bytes(ev.data[:4], "little")

                        if token_id == eos_id:
                            state.log("[bold green]EOS — generation complete.[/bold green]")
                            state.set_stage("Complete", "green")
                            shard.reset()
                            live.update(build_layout(state))
                            break

                        text = tokenizer.decode([token_id], skip_special_tokens=True)
                        state.full_response += text
                        state.token_count   += 1
                        state.set_stage("Receiving Tokens", "magenta")
                        live.update(build_layout(state))

                        h       = shard.decode_step(token_id)
                        h_bytes = np.array(h).astype(np.float16).tobytes()
                        mbatch += 1
                        net.request_tensor(
                            peer_id           = target_peer,
                            session_id        = f"{SESSION_ID}-t{turn}",
                            micro_batch_index = mbatch,
                            from_stage        = 0,
                            to_stage          = 1,
                            seq_len           = 1,
                            hidden_dim        = hidden_dim,
                            dtype             = DTYPE,
                            data              = h_bytes,
                        )

                elif ev.kind == "tensor_request_failed":
                    state.log(f"[bold red]Transfer failed:[/bold red] {ev.error}")
                    state.set_stage("Failed", "red")
                    shard.reset()
                    live.update(build_layout(state))
                    break

        console.print(f"\n[bold green]Response ({state.token_count} tokens):[/bold green]")
        console.print(state.full_response)

    net.shutdown()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python showcase_tui.py [sender|receiver] /path/to/model.gguf")
        sys.exit(1)

    role      = sys.argv[1].lower()
    gguf_path = sys.argv[2]

    if role == "receiver":
        shard, tokenizer, hidden_dim = load_from_omnistore(gguf_path, is_sender=False)
        run_receiver(shard, tokenizer, hidden_dim)

    elif role == "sender":
        shard, tokenizer, hidden_dim = load_from_omnistore(gguf_path, is_sender=True)
        run_sender(shard, tokenizer, hidden_dim)

    else:
        print("Usage: python showcase_tui.py [sender|receiver] /path/to/model.gguf")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
