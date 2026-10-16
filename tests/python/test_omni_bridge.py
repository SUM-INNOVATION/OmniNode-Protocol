"""The `omninode._omni_bridge` extension, exercised through its Python API.

These tests run in CI (`.github/workflows/ci.yml`, default job) against the
module built by maturin for Python 3.12. They cover what only a real
interpreter can check: exception classes and messages, the buffer protocol of
`ShardView`, dict-returning builders, and object lifetimes. The Rust-side unit
tests in `crates/omni-bridge/src` cover the conversions that need no
interpreter.

Every store is created in `tmp_path`. A store built without a configuration
would write under `$HOME/.omninode`, which no test does.
"""

import gc
import json
import struct

import pytest

import omninode
from omninode import (
    LayerRange,
    ModelManifest,
    NetConfig,
    NetEvent,
    OmniError,
    OmniNet,
    OmniStore,
    PipelineCapability,
    PipelineConfig,
    PipelineCoordinator,
    ShardView,
    StageExecutor,
    StoreConfig,
    StoreError,
)

GIB = 1 << 30


# ── Module surface ──────────────────────────────────────────────────────────


def test_the_package_exports_exactly_the_bridge_names():
    from omninode import _omni_bridge

    for name in omninode.__all__:
        assert getattr(omninode, name) is getattr(_omni_bridge, name)


def test_exceptions_form_a_hierarchy_under_exception():
    assert issubclass(OmniError, Exception)
    assert issubclass(StoreError, OmniError)


# ── Configuration ───────────────────────────────────────────────────────────


def test_net_config_default_and_override():
    assert NetConfig(listen_port=4242).listen_port == 4242
    assert repr(NetConfig(listen_port=4242)) == "NetConfig(listen_port=4242)"
    assert isinstance(NetConfig().listen_port, int)


def test_store_config_overrides_and_validation(tmp_path):
    c = StoreConfig(
        store_dir=str(tmp_path),
        layers_per_shard=2,
        max_shard_msg_bytes=1024,
        max_shard_total_bytes=4096,
    )
    assert (c.store_dir, c.layers_per_shard) == (str(tmp_path), 2)
    assert (c.max_shard_msg_bytes, c.max_shard_total_bytes) == (1024, 4096)

    with pytest.raises(ValueError, match="max_shard_msg_bytes"):
        StoreConfig(store_dir=str(tmp_path), max_shard_msg_bytes=0)
    with pytest.raises(ValueError, match="max_shard_total_bytes"):
        StoreConfig(store_dir=str(tmp_path), max_shard_total_bytes=0)


def test_out_of_range_integers_are_refused_by_type_conversion(tmp_path):
    with pytest.raises(OverflowError):
        NetConfig(listen_port=70000)
    with pytest.raises(OverflowError):
        StoreConfig(store_dir=str(tmp_path), layers_per_shard=-1)


def test_layer_range_is_inclusive():
    r = LayerRange(3, 7)
    assert (r.start, r.end, len(r)) == (3, 7, 5)
    assert len(LayerRange(4, 4)) == 1


# ── Model manifest ──────────────────────────────────────────────────────────


def _manifest_json():
    return json.dumps(
        {
            "model_name": "m",
            "model_hash": "h" * 64,
            "architecture": "llama",
            "total_layers": 2,
            "quantization": "F16",
            "total_size_bytes": 128,
            "gguf_version": 3,
            "shards": [
                {
                    "shard_index": 0,
                    "cid": "bafy0",
                    "layer_range": {"start": 0, "end": 1},
                    "includes_embedding": True,
                    "includes_output_head": True,
                    "size_bytes": 128,
                    "blake3_hash": "b" * 64,
                }
            ],
        }
    )


def test_manifest_json_round_trip():
    m = ModelManifest.from_json(_manifest_json())
    assert (m.model_name, m.architecture, m.total_layers, len(m)) == ("m", "llama", 2, 1)
    s = m.shards[0]
    assert (s.shard_index, s.cid, s.size_bytes) == (0, "bafy0", 128)
    assert (s.layer_range.start, s.layer_range.end) == (0, 1)
    assert s.includes_embedding and s.includes_output_head
    again = ModelManifest.from_json(m.to_json())
    assert again.to_json() == m.to_json()


@pytest.mark.parametrize("text", ["", "[]", '{"model_name": 5}', "not json"])
def test_manifest_from_json_refuses_malformed_input(text):
    with pytest.raises(ValueError):
        ModelManifest.from_json(text)


# ── Store and zero-copy shard views ─────────────────────────────────────────


def _gguf_string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _synthetic_gguf():
    """GGUF v3, two blocks, one 1-D F16 tensor each: 64 x 0xAA then 64 x 0xBB."""
    raw = b"GGUF" + struct.pack("<IQQ", 3, 2, 1)
    raw += _gguf_string("general.architecture") + struct.pack("<I", 8) + _gguf_string("llama")
    for name, offset in (("blk.0.attn_q.weight", 0), ("blk.1.attn_q.weight", 64)):
        raw += _gguf_string(name) + struct.pack("<IQIQ", 1, 32, 1, offset)
    raw += b"\0" * ((-len(raw)) % 32)
    return raw + b"\xaa" * 64 + b"\xbb" * 64


def _store(tmp_path):
    return OmniStore(StoreConfig(store_dir=str(tmp_path / "store"), layers_per_shard=1))


def _ingested(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(_synthetic_gguf())
    store = _store(tmp_path)
    return store, store.ingest_model(str(model))


def test_ingest_stores_one_shard_per_block_with_exactly_its_bytes(tmp_path):
    store, m = _ingested(tmp_path)
    assert store.store_dir == str(tmp_path / "store")
    assert (m.architecture, m.gguf_version, len(m.shards)) == ("llama", 3, 2)
    for i, (s, byte) in enumerate(zip(m.shards, (0xAA, 0xBB))):
        assert (s.shard_index, s.layer_range.start, s.layer_range.end) == (i, i, i)
        assert store.has_shard(s.cid)
        # Byte results are `bytes`, not a list of ints; a pyo3 upgrade must
        # keep that.
        assert store.get_shard(s.cid) == bytes([byte]) * 64
    assert (tmp_path / "store" / "manifest.cbor").is_file()


def test_a_shard_view_is_a_read_only_zero_copy_buffer(tmp_path):
    store, m = _ingested(tmp_path)
    cid = m.shards[1].cid
    view = store.mmap_shard(cid)
    assert isinstance(view, ShardView)
    assert (len(view), view.size, view.cid) == (64, 64, cid)

    mv = memoryview(view)
    assert (mv.readonly, mv.format, mv.itemsize, mv.ndim, mv.shape) == (True, "B", 1, 1, (64,))
    assert mv.tobytes() == b"\xbb" * 64
    assert mv[0] == 0xBB and mv[-1] == 0xBB
    with pytest.raises(TypeError):
        mv[0] = 0
    mv.release()


def test_the_exporter_itself_refuses_a_writable_buffer(tmp_path):
    # Asking through the C API with PyBUF_WRITABLE reaches ShardView's own
    # refusal. (Higher-level consumers such as ctypes reject a read-only buffer
    # on their own, so they would not notice the exporter granting one.)
    import ctypes

    store, m = _ingested(tmp_path)
    view = store.mmap_shard(m.shards[0].cid)
    get_buffer = ctypes.pythonapi.PyObject_GetBuffer
    get_buffer.argtypes = [ctypes.py_object, ctypes.c_void_p, ctypes.c_int]
    get_buffer.restype = ctypes.c_int
    release = ctypes.pythonapi.PyBuffer_Release
    release.argtypes = [ctypes.c_void_p]
    release.restype = None
    py_buffer = ctypes.create_string_buffer(256)  # larger than any Py_buffer
    PyBUF_WRITABLE = 0x0001
    try:
        get_buffer(view, ctypes.addressof(py_buffer), PyBUF_WRITABLE)
    except BufferError as e:
        assert "read-only" in str(e)
    else:
        release(ctypes.addressof(py_buffer))
        raise AssertionError("ShardView granted a writable buffer")


def test_a_memoryview_keeps_its_mapping_alive_after_the_view_is_dropped(tmp_path):
    store, m = _ingested(tmp_path)
    mv = memoryview(store.mmap_shard(m.shards[0].cid))
    del store
    gc.collect()
    # The memoryview holds a reference to the ShardView, which owns the mapping.
    assert mv.tobytes() == b"\xaa" * 64
    mv.release()


def test_numpy_reads_the_view_without_copying(tmp_path):
    np = pytest.importorskip("numpy")
    store, m = _ingested(tmp_path)
    view = store.mmap_shard(m.shards[0].cid)
    arr = np.frombuffer(view, dtype=np.uint8)
    assert arr.shape == (64,)
    assert not arr.flags.writeable
    assert int(arr.sum()) == 0xAA * 64
    assert arr.base is not None


def test_missing_shards_raise_store_error(tmp_path):
    store = _store(tmp_path)
    cid = "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"
    assert not store.has_shard(cid)
    with pytest.raises(StoreError):
        store.get_shard(cid)
    with pytest.raises(StoreError):
        store.mmap_shard(cid)


def test_a_malformed_model_raises_store_error_and_writes_no_shard(tmp_path):
    store = _store(tmp_path)
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"GGUF\x03\x00\x00\x00truncated")
    with pytest.raises(StoreError):
        store.ingest_model(str(bad))
    with pytest.raises(StoreError):
        store.ingest_model(str(tmp_path / "absent.gguf"))
    assert not list((tmp_path / "store").glob("*.shard"))


# ── Pipeline coordination ───────────────────────────────────────────────────


def _cap(peer, available_ram, ready=True):
    return PipelineCapability(peer, 2 * available_ram, available_ram, "linux-x86_64", [], [(0, 3)], ready)


def _scheduled(micro_batches=2):
    c = PipelineCoordinator(PipelineConfig(num_micro_batches=micro_batches, max_seq_len=128))
    sid, msg = c.propose_session("tiny", "h" * 64, 4, "peer-a")
    assert isinstance(msg, bytes) and len(msg) > 0
    c.handle_capability_offer(sid, _cap("peer-a", 8 * GIB))
    c.handle_capability_offer(sid, _cap("peer-b", 4 * GIB))
    c.handle_capability_offer(sid, _cap("peer-c", 64 * GIB, ready=False))
    schedule_json, msg = c.finalize_schedule(sid, 64)
    assert isinstance(msg, bytes) and len(msg) > 0
    return c, sid, schedule_json


def test_a_session_from_proposal_to_completion():
    c, sid, schedule_json = _scheduled()
    stages = json.loads(schedule_json)["stages"]
    assert [s["peer_id"] for s in stages] == ["peer-a", "peer-b"]
    assert c.session_state(sid) == "Scheduled"
    assert c.handle_stage_ready(sid, 0) is None
    start = c.handle_stage_ready(sid, 1)
    assert isinstance(start, bytes) and len(start) > 0
    assert c.session_state(sid) == "Running"
    assert c.active_session_ids() == [sid]
    assert len(c.handle_session_complete(sid, 42)) > 0
    assert c.session_state(sid) == "Completed"
    assert c.active_session_ids() == []


def test_a_stage_failure_fails_the_session():
    c, sid, _ = _scheduled()
    assert len(c.handle_stage_failure(sid, "peer-b", 1, "out of memory")) > 0
    assert c.session_state(sid) == "Failed"


def test_unknown_sessions_raise_omni_error():
    c = PipelineCoordinator(PipelineConfig())
    with pytest.raises(OmniError, match="unknown session"):
        c.finalize_schedule("nope", 64)
    with pytest.raises(OmniError):
        c.handle_capability_offer("nope", _cap("p", GIB))
    assert c.session_state("nope") is None


def test_executor_builders_return_complete_dicts():
    _, sid, schedule_json = _scheduled()
    first = StageExecutor(0, schedule_json)
    last = StageExecutor(1, schedule_json)
    assert (first.is_first_stage, first.is_last_stage, last.is_last_stage) == (True, False, True)
    assert (first.next_stage_peer_id, last.prev_stage_peer_id) == ("peer-b", "peer-a")

    req = first.build_forward_request(0, b"\x01\x02\x03", 16, 64, 1)
    assert req == {
        "session_id": sid,
        "micro_batch_index": 0,
        "from_stage": 0,
        "to_stage": 1,
        "seq_len": 16,
        "hidden_dim": 64,
        "dtype": 1,
        "data": b"\x01\x02\x03",
    }
    assert isinstance(req["data"], bytes)

    ack = last.build_ack(0, False, "shape mismatch")
    assert ack == {
        "session_id": sid,
        "micro_batch_index": 0,
        "stage_index": 1,
        "accepted": False,
        "error": "shape mismatch",
    }
    assert last.build_ack(1, True)["error"] is None

    first.mark_micro_batch_completed()
    first.mark_micro_batch_completed()
    assert (first.micro_batches_completed, first.is_complete) == (2, True)


def test_an_executor_refuses_a_bad_schedule_or_stage():
    _, _, schedule_json = _scheduled()
    with pytest.raises(OmniError, match="invalid schedule JSON"):
        StageExecutor(0, "not json")
    with pytest.raises(OmniError):
        StageExecutor(5, schedule_json)


# ── Networking lifecycle ────────────────────────────────────────────────────


def test_net_event_is_not_constructible_from_python():
    with pytest.raises(TypeError):
        NetEvent()


def test_a_node_listens_and_shuts_down_cleanly():
    net = OmniNet(NetConfig(listen_port=0))
    try:
        assert repr(net) == "OmniNet(status='active')"
        seen = []
        for _ in range(20):
            ev = net.next_event(timeout_secs=0.5)
            if ev is None:
                continue
            seen.append(ev.kind)
            if ev.kind == "listening":
                assert ev.address and ev.address.startswith("/ip")
                break
        assert "listening" in seen
    finally:
        net.shutdown()
    assert repr(net) == "OmniNet(status='shut down')"
    with pytest.raises(OmniError, match="shut down"):
        net.publish("omni/test/v1", b"x")
    # Shutting down twice is harmless.
    net.shutdown()


def test_the_context_manager_shuts_the_node_down():
    with OmniNet(NetConfig(listen_port=0)) as net:
        assert repr(net) == "OmniNet(status='active')"
    assert repr(net) == "OmniNet(status='shut down')"
    with pytest.raises(OmniError):
        net.next_event(timeout_secs=0.1)


def test_an_invalid_peer_id_is_refused_before_any_network_call():
    with OmniNet(NetConfig(listen_port=0)) as net:
        with pytest.raises(OmniError, match="invalid PeerId"):
            net.request_shard("not-a-peer-id", "bafy0")
        with pytest.raises(OmniError, match="invalid PeerId"):
            net.request_tensor("not-a-peer-id", "s", 0, 0, 1, 1, 1, 0, b"")
