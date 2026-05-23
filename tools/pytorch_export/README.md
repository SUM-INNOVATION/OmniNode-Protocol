# `tools/pytorch_export` — PyTorch canonical-spec exporter

Developer-host manual Python script that produces / verifies the committed `crates/omni-proofs-halo2-reference/tests/fixtures/pytorch_manifest.json` fixture by running the framework-neutral canonical spec through PyTorch with **explicit integer arithmetic** (no `torch.quantization` / `torch.ao` APIs).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r tools/pytorch_export/requirements.txt
```

The script needs `torch`, `numpy`, and `blake3` (Python). No GPU build is required — the canonical contract is integer-only and runs on CPU.

## Usage

```bash
# Verify the committed manifest matches what PyTorch produces today.
python3 tools/pytorch_export/pytorch_export.py verify-only

# Regenerate the committed manifest (developer-host only).
python3 tools/pytorch_export/pytorch_export.py regen
```

Pass `--spec` / `--manifest` to override paths.

## Arithmetic contract

The script implements the canonical contract directly on `torch.int64` tensors:

1. `acc_i64 = matmul(input.int64, W.int64)` — exact, no overflow.
2. `with_bias = acc + (bias.int64 << scale_log2)` — bias enters widened scale² domain BEFORE requantization.
3. `q = round_half_away_from_zero_div(with_bias, scale_log2)` — manual nearest-ties-away rounding via `(abs_n + d/2) // d` with sign preservation.
4. `output = clamp(q, -32768, 32767).to(int16)`.
5. ReLU is `clamp(x, min=0)`.

**No `torch.quantization` / `torch.ao` APIs are used** — those have their own rounding semantics that may not match the canonical spec.

## CI

This script is **not** invoked by CI. The cross-framework equivalence test in `crates/omni-proofs-halo2-reference` validates the committed manifest using pure Rust, with no PyTorch dependency.
