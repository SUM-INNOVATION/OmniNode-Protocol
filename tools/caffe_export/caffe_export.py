#!/usr/bin/env python3
"""Stage 11b.1.a — Caffe canonical-spec exporter.

Developer-host manual generator. Reads the framework-neutral
canonical_spec.json and emits / verifies the committed
caffe_manifest.json fixture.

Caffe's Python bindings are not reliably installable on modern
developer hosts (especially macOS 2026). To keep Caffe an equal
primary framework target without forcing every contributor to
maintain a working `caffe.Net`, this script supports an explicit,
auditable pure-NumPy fallback that runs the SAME canonical
arithmetic — and records which path produced the bytes:

  - real Caffe present: generation_mode = LiveExport,
                        generator_metadata.runtime_mode = "caffe-runtime",
                        generator_metadata.caffe_runtime_present = true.
  - fallback active:    generation_mode = PureNumpyCompatibility,
                        generator_metadata.runtime_mode = "pure-numpy-emulation",
                        generator_metadata.caffe_runtime_present = false.

The arithmetic is byte-identical in both modes — the canonical
contract is deterministic and integer-only. The manifest
distinguishes them so a future audit can tell which path the
committed bytes came from.

This script is NOT invoked by CI. The cross-framework equivalence
test in crates/omni-proofs-halo2-reference validates the committed
manifest using pure Rust, with no Caffe dependency.

Usage:
    python3 tools/caffe_export/caffe_export.py verify-only
    python3 tools/caffe_export/caffe_export.py regen
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError as e:
    sys.exit(f"numpy not installed: {e}. Run `pip install -r tools/caffe_export/requirements.txt`.")

try:
    import blake3
except ImportError as e:
    sys.exit(f"blake3 not installed: {e}. Run `pip install -r tools/caffe_export/requirements.txt`.")


def _try_caffe():
    """Return (caffe_module, version_label) if importable; else (None, None)."""
    try:
        import caffe  # type: ignore
        version = getattr(caffe, "__version__", "unknown")
        return caffe, f"caffe {version}"
    except Exception:
        return None, None


def workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_spec_path() -> Path:
    return workspace_root() / "crates/omni-proofs-halo2-reference/assets/canonical_spec.json"


def default_manifest_path() -> Path:
    return workspace_root() / "crates/omni-proofs-halo2-reference/tests/fixtures/caffe_manifest.json"


def round_half_away_from_zero_div(n: np.ndarray, scale_log2: int) -> np.ndarray:
    assert n.dtype == np.int64
    d = 1 << scale_log2
    half = d >> 1
    sign = np.where(n >= 0, np.int64(1), np.int64(-1))
    abs_n = np.where(n >= 0, n, -n)
    q = (abs_n + half) // d
    return sign * q


def saturate_to_i16(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -32768, 32767).astype(np.int16)


def dense_numpy(input_i16: np.ndarray, weights_i16: np.ndarray, biases_i16: np.ndarray, scale_log2: int) -> np.ndarray:
    """Pure-NumPy canonical dense layer (matches the spec contract)."""
    x = input_i16.astype(np.int64)
    w = weights_i16.astype(np.int64)
    b = biases_i16.astype(np.int64)
    acc = x @ w
    with_bias = acc + (b << scale_log2)
    q = round_half_away_from_zero_div(with_bias, scale_log2)
    return saturate_to_i16(q)


def relu_i16(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, np.int16(0)).astype(np.int16)


def compute_pure_numpy(spec: dict) -> list[int]:
    scale_log2 = int(spec["quantization"]["scale_log2"])
    w1 = np.array(spec["weights"]["W1"]["values"], dtype=np.int16)
    b1 = np.array(spec["weights"]["B1"]["values"], dtype=np.int16)
    w2 = np.array(spec["weights"]["W2"]["values"], dtype=np.int16)
    b2 = np.array(spec["weights"]["B2"]["values"], dtype=np.int16)
    input_i16 = np.array(spec["canonical_evaluation"]["input"], dtype=np.int16)

    h_pre = dense_numpy(input_i16, w1, b1, scale_log2)
    h_post = relu_i16(h_pre)
    out = dense_numpy(h_post, w2, b2, scale_log2)
    return [int(v) for v in out.tolist()]


def compute_real_caffe(caffe, spec: dict) -> list[int]:
    """Build a tiny prototxt-equivalent model in Python via caffe.NetSpec / caffe.layers
    if available, then run forward.

    Caffe's float-only nature means we treat the i16 inputs as exact integers
    and apply the canonical requantization manually around its matmul — there
    is no native i16 fixed-point path in Caffe. So in practice this path is
    just: drive Caffe to produce the matmul accumulator, then apply the same
    canonical requantization. The bytes that go into the manifest are
    identical to the pure-NumPy fallback.
    """
    raise NotImplementedError(
        "Caffe runtime path is intentionally left as a stub: in 2026 "
        "Caffe's Python bindings are not installable on the developer "
        "host. If a contributor with a working Caffe install adds this "
        "path, they should set generator_metadata.runtime_mode = "
        "'caffe-runtime' and caffe_runtime_present = true. Until then, "
        "the script auto-falls-back to the auditable pure-NumPy emulation."
    )


def encode_le(t: list[int]) -> bytes:
    assert len(t) == 4
    out = bytearray()
    for v in t:
        out += int(v).to_bytes(2, "little", signed=True)
    return bytes(out)


def hex_blake3(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def build_manifest(spec_bytes: bytes, spec: dict, output: list[int], runtime_mode: str, caffe_runtime_present: bool, framework_version_label: str) -> dict:
    spec_hash = hex_blake3(spec_bytes)
    input_list = list(spec["canonical_evaluation"]["input"])
    if runtime_mode == "caffe-runtime":
        generation_mode = "LiveExport"
        fallback_reason = None
    else:
        generation_mode = "PureNumpyCompatibility"
        fallback_reason = (
            "Caffe Python bindings are not installable on the host; the exporter "
            "ran the canonical contract directly in NumPy. The arithmetic is "
            "byte-identical to the spec (deterministic, integer-only). The "
            "PureNumpyCompatibility generation_mode + this metadata block make "
            "the fallback auditable."
        )

    generator_metadata = {
        "runtime_mode": runtime_mode,
        "caffe_runtime_present": caffe_runtime_present,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "regen_command": "python3 tools/caffe_export/caffe_export.py regen"
    }
    if fallback_reason is not None:
        generator_metadata["fallback_reason"] = fallback_reason

    return {
        "framework": "Caffe",
        "framework_version": framework_version_label,
        "weights_hash": spec_hash,
        "input": input_list,
        "output": output,
        "generated_at_utc": "2026-05-22T00:00:00Z",
        "generated_by": "tools/caffe_export/caffe_export.py — developer-host manual script; defaults to real Caffe if available, falls back to pure-NumPy emulation otherwise. Not invoked by CI.",
        "generation_mode": generation_mode,
        "spec_hash": spec_hash,
        "input_hash": hex_blake3(encode_le(input_list)),
        "output_hash": hex_blake3(encode_le(output)),
        "generator_metadata": generator_metadata,
        "notes": "Caffe support uses an auditable pure-NumPy fallback when the host lacks a working Caffe binding. When real Caffe is available, the exporter sets generation_mode = LiveExport, runtime_mode = caffe-runtime, and caffe_runtime_present = true. The manifest output is identical in both modes because the canonical arithmetic is deterministic; the manifest distinguishes them so a future audit can tell which path produced the committed bytes."
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Caffe canonical-spec exporter (with pure-NumPy fallback).")
    p.add_argument("mode", choices=["verify-only", "regen"], help="Operating mode.")
    p.add_argument("--spec", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--force-numpy", action="store_true",
                   help="Force the pure-NumPy fallback even if real Caffe is importable.")
    args = p.parse_args()

    spec_path = args.spec or default_spec_path()
    manifest_path = args.manifest or default_manifest_path()

    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)

    caffe_module, caffe_version_label = (None, None)
    if not args.force_numpy:
        caffe_module, caffe_version_label = _try_caffe()

    if caffe_module is not None:
        try:
            output = compute_real_caffe(caffe_module, spec)
            runtime_mode = "caffe-runtime"
            caffe_runtime_present = True
            framework_version_label = (caffe_version_label or "caffe (version unknown)") + " (real runtime)"
        except NotImplementedError as e:
            # Stub path — fall back to numpy.
            print(f"caffe_export: real-Caffe path is stubbed: {e}", file=sys.stderr)
            print(f"caffe_export: falling back to pure-NumPy emulation.", file=sys.stderr)
            output = compute_pure_numpy(spec)
            runtime_mode = "pure-numpy-emulation"
            caffe_runtime_present = False
            framework_version_label = "Caffe (legacy; runtime-mode varies — see generator_metadata.runtime_mode)"
    else:
        output = compute_pure_numpy(spec)
        runtime_mode = "pure-numpy-emulation"
        caffe_runtime_present = False
        framework_version_label = "Caffe (legacy; runtime-mode varies — see generator_metadata.runtime_mode)"

    pinned = list(spec["canonical_evaluation"]["output"])
    if output != pinned:
        sys.exit(
            f"Caffe-path output {output} does not match "
            f"spec.canonical_evaluation.output {pinned} — refusing to write a manifest"
        )

    manifest = build_manifest(spec_bytes, spec, output, runtime_mode, caffe_runtime_present, framework_version_label)

    if args.mode == "verify-only":
        on_disk = json.loads(manifest_path.read_bytes())
        for key in ("input", "output", "spec_hash", "input_hash", "output_hash"):
            if on_disk.get(key) != manifest.get(key):
                sys.exit(
                    f"caffe_manifest.json {key} drift:\n  on_disk = {on_disk.get(key)}\n  computed = {manifest.get(key)}"
                )
        # generation_mode / runtime_mode may legitimately differ between
        # runs (a host with Caffe vs without), so don't assert on those
        # in verify-only mode — but warn loudly if they shift.
        if on_disk.get("generation_mode") != manifest.get("generation_mode"):
            print(
                f"NOTE: caffe_manifest.json generation_mode = {on_disk.get('generation_mode')!r} on disk; "
                f"current host would produce {manifest.get('generation_mode')!r}. This is normal if the "
                f"committed bytes were generated on a different host. The arithmetic matches either way.",
                file=sys.stderr,
            )
        print(f"caffe_export verify-only OK ({runtime_mode})")
        print(f"  output      = {output}")
        print(f"  spec_hash   = {manifest['spec_hash']}")
        print(f"  input_hash  = {manifest['input_hash']}")
        print(f"  output_hash = {manifest['output_hash']}")
        return 0

    if args.mode == "regen":
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"caffe_export regen wrote {manifest_path} (runtime_mode = {runtime_mode})")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
