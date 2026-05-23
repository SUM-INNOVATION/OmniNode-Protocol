# `tools/tensorflow_export` — TensorFlow canonical-spec exporter

Developer-host manual Python script that produces / verifies the committed `crates/omni-proofs-halo2-reference/tests/fixtures/tensorflow_manifest.json` fixture by running the framework-neutral canonical spec through TensorFlow with **explicit integer arithmetic** (no `tf.quantization` / `tf.lite` quantization APIs).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r tools/tensorflow_export/requirements.txt
```

The script needs `tensorflow`, `numpy`, and `blake3` (Python). CPU-only is sufficient.

## Usage

```bash
# Verify the committed manifest matches what TensorFlow produces today.
python3 tools/tensorflow_export/tensorflow_export.py verify-only

# Regenerate the committed manifest (developer-host only).
python3 tools/tensorflow_export/tensorflow_export.py regen
```

Pass `--spec` / `--manifest` to override paths.

## Arithmetic contract

The script implements the canonical contract directly on `tf.int64` tensors:

1. `acc_i64 = matvec(W.T.int64, input.int64)` — exact, no overflow.
2. `with_bias = acc + (bias.int64 << scale_log2)` via `tf.bitwise.left_shift` — bias enters widened scale² domain BEFORE requantization.
3. `q = round_half_away_from_zero_div(with_bias, scale_log2)` — manual nearest-ties-away rounding via `(abs_n + d/2) // d` with sign preservation.
4. `output = clip_by_value(q, -32768, 32767).cast(int16)`.
5. ReLU is `maximum(x, 0)`.

**No `tf.quantization` / `tf.lite` quantization APIs are used.**

## CI

This script is **not** invoked by CI. The cross-framework equivalence test in `crates/omni-proofs-halo2-reference` validates the committed manifest using pure Rust, with no TensorFlow dependency.
