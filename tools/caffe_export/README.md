# `tools/caffe_export` — Caffe canonical-spec exporter (with auditable pure-NumPy fallback)

Developer-host manual Python script that produces / verifies the committed `crates/omni-proofs-halo2-reference/tests/fixtures/caffe_manifest.json` fixture.

Caffe is an **equal primary compatibility target** alongside RUMUS, PyTorch, and TensorFlow. However Caffe's Python bindings are not reliably installable on modern developer hosts (especially macOS 2026), so this exporter supports an explicit, auditable pure-NumPy fallback that runs the **exact same canonical arithmetic**. The manifest itself records which path produced the committed bytes — fallbacks are never silent.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r tools/caffe_export/requirements.txt

# (Optional) install real Caffe if your host supports it. Then the
# exporter will switch to generation_mode = LiveExport automatically.
# pip install caffe   # not generally pip-installable on 2026 stacks
```

## Usage

```bash
# Verify the committed manifest matches what this exporter produces today.
python3 tools/caffe_export/caffe_export.py verify-only

# Regenerate the committed manifest (developer-host only).
python3 tools/caffe_export/caffe_export.py regen

# Force the pure-NumPy path even if real Caffe is importable
# (useful for verifying the fallback path explicitly).
python3 tools/caffe_export/caffe_export.py regen --force-numpy
```

## Runtime-mode contract

| Real Caffe available | `generation_mode`           | `generator_metadata.runtime_mode` | `caffe_runtime_present` |
|----------------------|-----------------------------|-----------------------------------|-------------------------|
| Yes                  | `LiveExport`                | `caffe-runtime`                   | `true`                  |
| No (or `--force-numpy`) | `PureNumpyCompatibility` | `pure-numpy-emulation`            | `false`                 |

The output bytes are identical in both modes because the canonical arithmetic is deterministic and integer-only. The manifest distinguishes them so a future audit can tell which path produced the committed bytes.

## Arithmetic contract

Same as the other framework exporters — on `np.int64` arrays:

1. `acc_i64 = input.int64 @ W.int64` — exact, no overflow.
2. `with_bias = acc + (bias.int64 << scale_log2)` — bias enters widened scale² domain BEFORE requantization.
3. `q = round_half_away_from_zero_div(with_bias, scale_log2)` — manual nearest-ties-away rounding.
4. `output = clip(q, -32768, 32767).astype(int16)`.
5. ReLU is `maximum(x, 0)`.

## CI

This script is **not** invoked by CI. The cross-framework equivalence test in `crates/omni-proofs-halo2-reference` validates the committed manifest using pure Rust, with no Caffe / NumPy dependency.
