#!/usr/bin/env python3
"""Stage 11b.1.a — PyTorch canonical-spec exporter.

Developer-host manual generator. Reads the framework-neutral
canonical_spec.json and emits / verifies the committed
pytorch_manifest.json fixture by running the canonical arithmetic
through PyTorch with explicit integer math (NO torch.quantization /
torch.ao APIs — those have their own rounding semantics).

This script is NOT invoked by CI. The cross-framework equivalence
test in crates/omni-proofs-halo2-reference validates the committed
manifest using pure Rust, with no PyTorch dependency.

Usage:
    python3 tools/pytorch_export/pytorch_export.py verify-only
    python3 tools/pytorch_export/pytorch_export.py regen
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Hard requirement: numpy and blake3 must be present.
try:
    import numpy as np
except ImportError as e:
    sys.exit(f"numpy not installed: {e}. Run `pip install -r tools/pytorch_export/requirements.txt`.")

try:
    import blake3
except ImportError as e:
    sys.exit(f"blake3 not installed: {e}. Run `pip install -r tools/pytorch_export/requirements.txt`.")

try:
    import torch
except ImportError as e:
    sys.exit(f"torch not installed: {e}. Run `pip install -r tools/pytorch_export/requirements.txt`.")


def workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_spec_path() -> Path:
    return workspace_root() / "crates/omni-proofs-halo2-reference/assets/canonical_spec.json"


def default_manifest_path() -> Path:
    return workspace_root() / "crates/omni-proofs-halo2-reference/tests/fixtures/pytorch_manifest.json"


def round_half_away_from_zero_div(n: torch.Tensor, scale_log2: int) -> torch.Tensor:
    """Integer-domain round-half-away-from-zero divide by 2**scale_log2.

    Matches the canonical spec's `quantization.rounding` exactly.
    Operates on int64 tensors; returns int64.
    """
    assert n.dtype == torch.int64
    d = 1 << scale_log2
    half = d >> 1  # d/2; safe for d>=2.
    sign = torch.where(n >= 0, torch.tensor(1, dtype=torch.int64), torch.tensor(-1, dtype=torch.int64))
    abs_n = torch.where(n >= 0, n, -n)
    q = (abs_n + half) // d
    return sign * q


def saturate_to_i16(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -32768, 32767).to(torch.int16)


def dense(input_i16: torch.Tensor, weights_i16: torch.Tensor, biases_i16: torch.Tensor, scale_log2: int) -> torch.Tensor:
    """Canonical dense layer per Stage 11b.1.a §"Numeric contract".

    input_i16:   shape [in],     dtype int16 (interpreted as fixed-point at scale 2**scale_log2)
    weights_i16: shape [in, out], dtype int16 (same scale)
    biases_i16:  shape [out],    dtype int16 (same scale)
    Returns int16 tensor of shape [out] at the same scale.
    """
    x = input_i16.to(torch.int64)
    w = weights_i16.to(torch.int64)
    b = biases_i16.to(torch.int64)
    acc = torch.matmul(x.unsqueeze(0), w).squeeze(0)              # int64, shape [out]
    with_bias = acc + (b << scale_log2)                            # bias enters scale² domain
    q = round_half_away_from_zero_div(with_bias, scale_log2)       # int64, requantized to scale S
    return saturate_to_i16(q)


def relu_i16(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0).to(torch.int16)


def encode_le(t: list[int]) -> bytes:
    assert len(t) == 4
    out = bytearray()
    for v in t:
        out += int(v).to_bytes(2, "little", signed=True)
    return bytes(out)


def hex_blake3(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def compute(spec: dict) -> tuple[list[int], list[int], str]:
    scale_log2 = int(spec["quantization"]["scale_log2"])
    w1 = torch.tensor(spec["weights"]["W1"]["values"], dtype=torch.int16)  # [4, 8]
    b1 = torch.tensor(spec["weights"]["B1"]["values"], dtype=torch.int16)  # [8]
    w2 = torch.tensor(spec["weights"]["W2"]["values"], dtype=torch.int16)  # [8, 4]
    b2 = torch.tensor(spec["weights"]["B2"]["values"], dtype=torch.int16)  # [4]
    input_i16 = torch.tensor(spec["canonical_evaluation"]["input"], dtype=torch.int16)

    h_pre = dense(input_i16, w1, b1, scale_log2)
    h_post = relu_i16(h_pre)
    out = dense(h_post, w2, b2, scale_log2)

    return (h_post.tolist(), out.tolist(), f"torch {torch.__version__}")


def build_manifest(spec_bytes: bytes, spec: dict, output: list[int]) -> dict:
    spec_hash = hex_blake3(spec_bytes)
    input_list = list(spec["canonical_evaluation"]["input"])
    return {
        "framework": "PyTorch",
        "framework_version": f"torch {torch.__version__} (CPU int64 explicit; no torch.quantization)",
        "weights_hash": spec_hash,
        "input": input_list,
        "output": output,
        "generated_at_utc": "2026-05-22T00:00:00Z",
        "generated_by": "tools/pytorch_export/pytorch_export.py — developer-host manual script; explicit integer arithmetic via int64 tensors. Not invoked by CI.",
        "generation_mode": "LiveExport",
        "spec_hash": spec_hash,
        "input_hash": hex_blake3(encode_le(input_list)),
        "output_hash": hex_blake3(encode_le(output)),
        "generator_metadata": {
            "runtime_mode": "pytorch-int64-explicit",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "torch_quantization_apis_used": False,
            "regen_command": "python3 tools/pytorch_export/pytorch_export.py regen"
        },
        "notes": "PyTorch reproduces the canonical contract via explicit integer math on int64 tensors (no `torch.quantization` / `torch.ao` APIs). The export script implements i16×i16 → i64 accumulation, bias promotion to scale² domain, round-half-away-from-zero requantization, saturate-to-i16, and ReLU max(x,0) directly. Verified byte-for-byte against the canonical evaluator on developer host."
    }


def main() -> int:
    p = argparse.ArgumentParser(description="PyTorch canonical-spec exporter.")
    p.add_argument("mode", choices=["verify-only", "regen"], help="Operating mode.")
    p.add_argument("--spec", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    args = p.parse_args()

    spec_path = args.spec or default_spec_path()
    manifest_path = args.manifest or default_manifest_path()

    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    h_post, output, version_label = compute(spec)

    # Cross-check against the spec's pinned canonical_evaluation.output.
    pinned = list(spec["canonical_evaluation"]["output"])
    if output != pinned:
        sys.exit(
            f"PyTorch-computed output {output} does not match "
            f"spec.canonical_evaluation.output {pinned} — refusing to write a manifest"
        )

    manifest = build_manifest(spec_bytes, spec, output)

    if args.mode == "verify-only":
        on_disk = json.loads(manifest_path.read_bytes())
        for key in ("input", "output", "spec_hash", "input_hash", "output_hash"):
            if on_disk.get(key) != manifest.get(key):
                sys.exit(
                    f"pytorch_manifest.json {key} drift:\n  on_disk = {on_disk.get(key)}\n  computed = {manifest.get(key)}"
                )
        print(f"pytorch_export verify-only OK")
        print(f"  output      = {output}")
        print(f"  spec_hash   = {manifest['spec_hash']}")
        print(f"  input_hash  = {manifest['input_hash']}")
        print(f"  output_hash = {manifest['output_hash']}")
        return 0

    if args.mode == "regen":
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"pytorch_export regen wrote {manifest_path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
