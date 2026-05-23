#!/usr/bin/env python3
"""Stage 11b.1.a — TensorFlow canonical-spec exporter.

Developer-host manual generator. Reads the framework-neutral
canonical_spec.json and emits / verifies the committed
tensorflow_manifest.json fixture by running the canonical arithmetic
through TensorFlow with explicit integer math (NO tf.quantization /
tf.lite quantization APIs — those have their own rounding
semantics).

This script is NOT invoked by CI. The cross-framework equivalence
test in crates/omni-proofs-halo2-reference validates the committed
manifest using pure Rust, with no TensorFlow dependency.

Usage:
    python3 tools/tensorflow_export/tensorflow_export.py verify-only
    python3 tools/tensorflow_export/tensorflow_export.py regen
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError as e:
    sys.exit(f"numpy not installed: {e}. Run `pip install -r tools/tensorflow_export/requirements.txt`.")

try:
    import blake3
except ImportError as e:
    sys.exit(f"blake3 not installed: {e}. Run `pip install -r tools/tensorflow_export/requirements.txt`.")

try:
    import tensorflow as tf
except ImportError as e:
    sys.exit(f"tensorflow not installed: {e}. Run `pip install -r tools/tensorflow_export/requirements.txt`.")


def workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_spec_path() -> Path:
    return workspace_root() / "crates/omni-proofs-halo2-reference/assets/canonical_spec.json"


def default_manifest_path() -> Path:
    return workspace_root() / "crates/omni-proofs-halo2-reference/tests/fixtures/tensorflow_manifest.json"


def round_half_away_from_zero_div(n: tf.Tensor, scale_log2: int) -> tf.Tensor:
    """Integer-domain round-half-away-from-zero divide by 2**scale_log2.

    Operates on tf.int64; returns tf.int64.
    """
    assert n.dtype == tf.int64
    d = tf.constant(1 << scale_log2, dtype=tf.int64)
    half = tf.constant((1 << scale_log2) >> 1, dtype=tf.int64)
    sign = tf.where(n >= 0, tf.constant(1, dtype=tf.int64), tf.constant(-1, dtype=tf.int64))
    abs_n = tf.where(n >= 0, n, -n)
    q = (abs_n + half) // d
    return sign * q


def saturate_to_i16(x: tf.Tensor) -> tf.Tensor:
    return tf.cast(tf.clip_by_value(x, -32768, 32767), tf.int16)


def dense(input_i16: tf.Tensor, weights_i16: tf.Tensor, biases_i16: tf.Tensor, scale_log2: int) -> tf.Tensor:
    x = tf.cast(input_i16, tf.int64)
    w = tf.cast(weights_i16, tf.int64)
    b = tf.cast(biases_i16, tf.int64)
    acc = tf.linalg.matvec(tf.transpose(w), x)                  # int64, shape [out]
    with_bias = acc + tf.bitwise.left_shift(b, tf.constant(scale_log2, dtype=tf.int64))
    q = round_half_away_from_zero_div(with_bias, scale_log2)
    return saturate_to_i16(q)


def relu_i16(x: tf.Tensor) -> tf.Tensor:
    return tf.cast(tf.maximum(x, tf.constant(0, dtype=x.dtype)), tf.int16)


def encode_le(t: list[int]) -> bytes:
    assert len(t) == 4
    out = bytearray()
    for v in t:
        out += int(v).to_bytes(2, "little", signed=True)
    return bytes(out)


def hex_blake3(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def compute(spec: dict) -> list[int]:
    scale_log2 = int(spec["quantization"]["scale_log2"])
    w1 = tf.constant(spec["weights"]["W1"]["values"], dtype=tf.int16)  # [4, 8]
    b1 = tf.constant(spec["weights"]["B1"]["values"], dtype=tf.int16)  # [8]
    w2 = tf.constant(spec["weights"]["W2"]["values"], dtype=tf.int16)  # [8, 4]
    b2 = tf.constant(spec["weights"]["B2"]["values"], dtype=tf.int16)  # [4]
    input_i16 = tf.constant(spec["canonical_evaluation"]["input"], dtype=tf.int16)

    h_pre = dense(input_i16, w1, b1, scale_log2)
    h_post = relu_i16(h_pre)
    out = dense(h_post, w2, b2, scale_log2)
    return [int(v) for v in out.numpy().tolist()]


def build_manifest(spec_bytes: bytes, spec: dict, output: list[int]) -> dict:
    spec_hash = hex_blake3(spec_bytes)
    input_list = list(spec["canonical_evaluation"]["input"])
    return {
        "framework": "TensorFlow",
        "framework_version": f"tensorflow {tf.__version__} (CPU int64 explicit; no tf.quantization)",
        "weights_hash": spec_hash,
        "input": input_list,
        "output": output,
        "generated_at_utc": "2026-05-22T00:00:00Z",
        "generated_by": "tools/tensorflow_export/tensorflow_export.py — developer-host manual script; explicit integer arithmetic via int64 tensors. Not invoked by CI.",
        "generation_mode": "LiveExport",
        "spec_hash": spec_hash,
        "input_hash": hex_blake3(encode_le(input_list)),
        "output_hash": hex_blake3(encode_le(output)),
        "generator_metadata": {
            "runtime_mode": "tensorflow-int64-explicit",
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "tf_quantization_apis_used": False,
            "regen_command": "python3 tools/tensorflow_export/tensorflow_export.py regen"
        },
        "notes": "TensorFlow reproduces the canonical contract via explicit int64 integer arithmetic (avoiding `tf.quantization` higher-level machinery, which has its own rounding semantics). The export script mirrors the spec's pipeline: i16×i16 → i64 accumulation, bias promotion via shift, round-half-away-from-zero requantization, saturate-to-i16, ReLU max(x,0). Verified byte-for-byte against the canonical evaluator on developer host."
    }


def main() -> int:
    p = argparse.ArgumentParser(description="TensorFlow canonical-spec exporter.")
    p.add_argument("mode", choices=["verify-only", "regen"], help="Operating mode.")
    p.add_argument("--spec", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    args = p.parse_args()

    spec_path = args.spec or default_spec_path()
    manifest_path = args.manifest or default_manifest_path()

    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    output = compute(spec)

    pinned = list(spec["canonical_evaluation"]["output"])
    if output != pinned:
        sys.exit(
            f"TensorFlow-computed output {output} does not match "
            f"spec.canonical_evaluation.output {pinned} — refusing to write a manifest"
        )

    manifest = build_manifest(spec_bytes, spec, output)

    if args.mode == "verify-only":
        on_disk = json.loads(manifest_path.read_bytes())
        for key in ("input", "output", "spec_hash", "input_hash", "output_hash"):
            if on_disk.get(key) != manifest.get(key):
                sys.exit(
                    f"tensorflow_manifest.json {key} drift:\n  on_disk = {on_disk.get(key)}\n  computed = {manifest.get(key)}"
                )
        print(f"tensorflow_export verify-only OK")
        print(f"  output      = {output}")
        print(f"  spec_hash   = {manifest['spec_hash']}")
        print(f"  input_hash  = {manifest['input_hash']}")
        print(f"  output_hash = {manifest['output_hash']}")
        return 0

    if args.mode == "regen":
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"tensorflow_export regen wrote {manifest_path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
