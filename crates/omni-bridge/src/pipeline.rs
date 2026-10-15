//! #[pyclass] wrappers for pipeline coordination types.
//!
//! Exposes `PipelineCoordinator` and `StageExecutor` from `omni-pipeline`
//! to Python, enabling distributed pipeline-parallel inference driven from
//! MLX/Python.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use omni_pipeline::{PipelineAction, PipelineCoordinator, StageExecutor};
use omni_types::config::PipelineConfig;
use omni_types::model::LayerRange;
use omni_types::pipeline::{PipelineCapability, PipelineSchedule};

use crate::errors::PyOmniError;

// ── PipelineConfig ──────────────────────────────────────────────────────────

#[pyclass(name = "PipelineConfig", from_py_object)]
#[derive(Clone)]
pub struct PyPipelineConfig {
    pub inner: PipelineConfig,
}

#[pymethods]
impl PyPipelineConfig {
    #[new]
    #[pyo3(signature = (
        num_micro_batches=None,
        max_seq_len=None,
        heartbeat_interval_secs=None,
        heartbeat_timeout_factor=None,
        tensor_timeout_secs=None,
    ))]
    fn new(
        num_micro_batches: Option<u32>,
        max_seq_len: Option<u32>,
        heartbeat_interval_secs: Option<u64>,
        heartbeat_timeout_factor: Option<u32>,
        tensor_timeout_secs: Option<u64>,
    ) -> Self {
        let mut cfg = PipelineConfig::default();
        if let Some(m) = num_micro_batches {
            cfg.num_micro_batches = Some(m);
        }
        if let Some(s) = max_seq_len {
            cfg.max_seq_len = s;
        }
        if let Some(h) = heartbeat_interval_secs {
            cfg.heartbeat_interval_secs = h;
        }
        if let Some(f) = heartbeat_timeout_factor {
            cfg.heartbeat_timeout_factor = f;
        }
        if let Some(t) = tensor_timeout_secs {
            cfg.tensor_timeout_secs = t;
        }
        Self { inner: cfg }
    }

    #[getter]
    fn num_micro_batches(&self) -> Option<u32> {
        self.inner.num_micro_batches
    }

    #[getter]
    fn max_seq_len(&self) -> u32 {
        self.inner.max_seq_len
    }

    #[getter]
    fn heartbeat_interval_secs(&self) -> u64 {
        self.inner.heartbeat_interval_secs
    }

    #[getter]
    fn heartbeat_timeout_factor(&self) -> u32 {
        self.inner.heartbeat_timeout_factor
    }

    #[getter]
    fn tensor_timeout_secs(&self) -> u64 {
        self.inner.tensor_timeout_secs
    }

    fn __repr__(&self) -> String {
        format!(
            "PipelineConfig(micro_batches={:?}, max_seq_len={}, heartbeat={}s, timeout={}s)",
            self.inner.num_micro_batches,
            self.inner.max_seq_len,
            self.inner.heartbeat_interval_secs,
            self.inner.tensor_timeout_secs,
        )
    }
}

// ── PipelineCapability ──────────────────────────────────────────────────────

#[pyclass(name = "PipelineCapability", skip_from_py_object)]
#[derive(Clone)]
pub struct PyPipelineCapability {
    pub inner: PipelineCapability,
}

#[pymethods]
impl PyPipelineCapability {
    #[new]
    fn new(
        peer_id: String,
        ram_bytes: u64,
        available_ram_bytes: u64,
        platform: String,
        local_shard_cids: Vec<String>,
        available_layers: Vec<(u32, u32)>,
        pipeline_ready: bool,
    ) -> Self {
        Self {
            inner: PipelineCapability {
                peer_id,
                ram_bytes,
                available_ram_bytes,
                platform,
                local_shard_cids,
                available_layers: available_layers
                    .into_iter()
                    .map(|(s, e)| LayerRange { start: s, end: e })
                    .collect(),
                pipeline_ready,
            },
        }
    }

    #[getter]
    fn peer_id(&self) -> &str {
        &self.inner.peer_id
    }

    #[getter]
    fn ram_bytes(&self) -> u64 {
        self.inner.ram_bytes
    }

    #[getter]
    fn available_ram_bytes(&self) -> u64 {
        self.inner.available_ram_bytes
    }

    #[getter]
    fn platform(&self) -> &str {
        &self.inner.platform
    }

    #[getter]
    fn pipeline_ready(&self) -> bool {
        self.inner.pipeline_ready
    }

    fn __repr__(&self) -> String {
        format!(
            "PipelineCapability(peer='{}', ram={}MB, ready={})",
            self.inner.peer_id,
            self.inner.available_ram_bytes / (1024 * 1024),
            self.inner.pipeline_ready,
        )
    }
}

// ── PipelineCoordinator ─────────────────────────────────────────────────────

#[pyclass(name = "PipelineCoordinator")]
pub struct PyPipelineCoordinator {
    inner: PipelineCoordinator,
}

#[pymethods]
impl PyPipelineCoordinator {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyPipelineConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: PipelineCoordinator::new(cfg),
        }
    }

    /// Propose a new pipeline session.
    ///
    /// Returns `(session_id, message_bytes)` — publish `message_bytes` on
    /// `omni/pipeline/v1` via `OmniNet.publish()`.
    fn propose_session(
        &mut self,
        model_name: &str,
        model_hash: &str,
        total_layers: u32,
        local_peer_id: &str,
    ) -> PyResult<(String, Vec<u8>)> {
        let (session_id, action) = self
            .inner
            .propose_session(
                model_name.to_string(),
                model_hash.to_string(),
                total_layers,
                local_peer_id.to_string(),
            )
            .map_err(|e| PyOmniError::new_err(e.to_string()))?;

        let PipelineAction::PublishMessage { data } = action;
        Ok((session_id, data))
    }

    /// Register a capability offer from a peer.
    fn handle_capability_offer(
        &mut self,
        session_id: &str,
        capability: &PyPipelineCapability,
    ) -> PyResult<()> {
        self.inner
            .handle_capability_offer(session_id, capability.inner.clone())
            .map_err(|e| PyOmniError::new_err(e.to_string()))
    }

    /// Run the planner and produce a schedule.
    ///
    /// Returns `(schedule_json, message_bytes)` — publish `message_bytes` on
    /// `omni/pipeline/v1`. Use `schedule_json` to create `StageExecutor`.
    fn finalize_schedule(
        &mut self,
        session_id: &str,
        hidden_dim: u32,
    ) -> PyResult<(String, Vec<u8>)> {
        let (schedule, action) = self
            .inner
            .finalize_schedule(session_id, hidden_dim)
            .map_err(|e| PyOmniError::new_err(e.to_string()))?;

        let schedule_json = serde_json::to_string(&schedule)
            .map_err(|e| PyOmniError::new_err(e.to_string()))?;

        let PipelineAction::PublishMessage { data } = action;
        Ok((schedule_json, data))
    }

    /// Handle a StageReady report.
    ///
    /// Returns `message_bytes` if all stages are ready (publish to start
    /// inference), otherwise `None`.
    fn handle_stage_ready(
        &mut self,
        session_id: &str,
        stage_index: u32,
    ) -> PyResult<Option<Vec<u8>>> {
        let result = self
            .inner
            .handle_stage_ready(session_id, stage_index)
            .map_err(|e| PyOmniError::new_err(e.to_string()))?;

        Ok(match result {
            Some(PipelineAction::PublishMessage { data }) => Some(data),
            None => None,
        })
    }

    /// Mark a session as completed.
    ///
    /// Returns `message_bytes` — publish on `omni/pipeline/v1`.
    fn handle_session_complete(
        &mut self,
        session_id: &str,
        total_tokens: u64,
    ) -> PyResult<Vec<u8>> {
        let action = self
            .inner
            .handle_session_complete(session_id, total_tokens)
            .map_err(|e| PyOmniError::new_err(e.to_string()))?;

        let PipelineAction::PublishMessage { data } = action;
        Ok(data)
    }

    /// Handle a stage failure and abort the session.
    ///
    /// Returns `message_bytes` — publish on `omni/pipeline/v1`.
    fn handle_stage_failure(
        &mut self,
        session_id: &str,
        peer_id: &str,
        stage_index: u32,
        error: &str,
    ) -> PyResult<Vec<u8>> {
        let action = self
            .inner
            .handle_stage_failure(session_id, peer_id, stage_index, error)
            .map_err(|e| PyOmniError::new_err(e.to_string()))?;

        let PipelineAction::PublishMessage { data } = action;
        Ok(data)
    }

    /// Get the current state of a session as a string.
    fn session_state(&self, session_id: &str) -> Option<String> {
        self.inner.session(session_id).map(|s| s.state.to_string())
    }

    /// List all active (non-terminal) session IDs.
    fn active_session_ids(&self) -> Vec<String> {
        self.inner
            .active_session_ids()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn __repr__(&self) -> String {
        let n = self.inner.active_session_ids().len();
        format!("PipelineCoordinator(active_sessions={n})")
    }
}

// ── StageExecutor ───────────────────────────────────────────────────────────

#[pyclass(name = "StageExecutor")]
pub struct PyStageExecutor {
    inner: StageExecutor,
}

#[pymethods]
impl PyStageExecutor {
    /// Create an executor for a specific stage.
    ///
    /// `schedule_json` is the JSON string returned by
    /// `PipelineCoordinator.finalize_schedule()`.
    #[new]
    fn new(stage_index: u32, schedule_json: &str) -> PyResult<Self> {
        let schedule: PipelineSchedule = serde_json::from_str(schedule_json)
            .map_err(|e| PyOmniError::new_err(format!("invalid schedule JSON: {e}")))?;

        let inner = StageExecutor::new(stage_index, schedule)
            .map_err(|e| PyOmniError::new_err(e.to_string()))?;

        Ok(Self { inner })
    }

    #[getter]
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }

    #[getter]
    fn stage_index(&self) -> u32 {
        self.inner.stage_index()
    }

    #[getter]
    fn is_first_stage(&self) -> bool {
        self.inner.is_first_stage()
    }

    #[getter]
    fn is_last_stage(&self) -> bool {
        self.inner.is_last_stage()
    }

    #[getter]
    fn next_stage_peer_id(&self) -> Option<&str> {
        self.inner.next_stage_peer_id()
    }

    #[getter]
    fn prev_stage_peer_id(&self) -> Option<&str> {
        self.inner.prev_stage_peer_id()
    }

    #[getter]
    fn micro_batches_completed(&self) -> u32 {
        self.inner.micro_batches_completed()
    }

    #[getter]
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// Build a TensorRequest dict to forward activations to the next stage.
    ///
    /// Returns a dict with keys: session_id, micro_batch_index, from_stage,
    /// to_stage, seq_len, hidden_dim, dtype, data.
    fn build_forward_request<'py>(
        &self,
        py: Python<'py>,
        micro_batch_index: u32,
        data: Vec<u8>,
        seq_len: u32,
        hidden_dim: u32,
        dtype: u8,
    ) -> PyResult<Bound<'py, PyDict>> {
        let req =
            self.inner
                .build_forward_request(micro_batch_index, data, seq_len, hidden_dim, dtype);
        let dict = PyDict::new(py);
        dict.set_item("session_id", &req.session_id)?;
        dict.set_item("micro_batch_index", req.micro_batch_index)?;
        dict.set_item("from_stage", req.from_stage)?;
        dict.set_item("to_stage", req.to_stage)?;
        dict.set_item("seq_len", req.seq_len)?;
        dict.set_item("hidden_dim", req.hidden_dim)?;
        dict.set_item("dtype", req.dtype)?;
        dict.set_item("data", PyBytes::new(py, &req.data))?;
        Ok(dict)
    }

    /// Build a TensorResponse dict to acknowledge a received tensor.
    ///
    /// Returns a dict with keys: session_id, micro_batch_index, stage_index,
    /// accepted, error.
    #[pyo3(signature = (micro_batch_index, accepted, error=None))]
    fn build_ack<'py>(
        &self,
        py: Python<'py>,
        micro_batch_index: u32,
        accepted: bool,
        error: Option<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let resp = self.inner.build_ack(micro_batch_index, accepted, error);
        let dict = PyDict::new(py);
        dict.set_item("session_id", &resp.session_id)?;
        dict.set_item("micro_batch_index", resp.micro_batch_index)?;
        dict.set_item("stage_index", resp.stage_index)?;
        dict.set_item("accepted", resp.accepted)?;
        dict.set_item("error", resp.error)?;
        Ok(dict)
    }

    /// Increment the completed micro-batch counter.
    fn mark_micro_batch_completed(&mut self) {
        self.inner.mark_micro_batch_completed();
    }

    fn __repr__(&self) -> String {
        format!(
            "StageExecutor(stage={}, session='{}', completed={})",
            self.inner.stage_index(),
            self.inner.session_id(),
            self.inner.micro_batches_completed(),
        )
    }
}
#[cfg(test)]
mod tests {
    //! The coordinator and executor wrappers through a whole session, as a
    //! Python driver would use them: propose, collect offers, schedule, build
    //! executors from the schedule JSON, report readiness, complete or fail.
    //! The dict-returning builders need an interpreter and are covered by
    //! `tests/python/`. Refusals are checked with `is_err()` only.

    use super::*;
    use crate::test_support::ok;

    const GIB: u64 = 1 << 30;

    fn capability(peer: &str, available_ram: u64, ready: bool) -> PyPipelineCapability {
        PyPipelineCapability::new(
            peer.into(),
            2 * available_ram,
            available_ram,
            "linux-x86_64".into(),
            vec![],
            vec![(0, 3)],
            ready,
        )
    }

    fn coordinator(micro_batches: u32) -> PyPipelineCoordinator {
        PyPipelineCoordinator::new(Some(PyPipelineConfig::new(
            Some(micro_batches),
            Some(128),
            None,
            None,
            None,
        )))
    }

    /// A scheduled two-stage session over four layers; returns the session id
    /// and the schedule JSON.
    fn scheduled(c: &mut PyPipelineCoordinator) -> (String, String) {
        let (sid, msg) = ok(
            c.propose_session("tiny", &"h".repeat(64), 4, "peer-a"),
            "proposes",
        );
        assert!(!msg.is_empty(), "the proposal is a message to publish");
        assert_eq!(c.session_state(&sid).as_deref(), Some("Forming"));
        ok(
            c.handle_capability_offer(&sid, &capability("peer-a", 8 * GIB, true)),
            "offer a",
        );
        ok(
            c.handle_capability_offer(&sid, &capability("peer-b", 4 * GIB, true)),
            "offer b",
        );
        // A node that is not pipeline-ready is never scheduled.
        ok(
            c.handle_capability_offer(&sid, &capability("peer-c", 64 * GIB, false)),
            "offer c",
        );
        let (json, msg) = ok(c.finalize_schedule(&sid, 64), "schedules");
        assert!(!msg.is_empty());
        assert_eq!(c.session_state(&sid).as_deref(), Some("Scheduled"));
        (sid, json)
    }

    #[test]
    fn the_schedule_covers_every_layer_contiguously_and_skips_unready_nodes() {
        let mut c = coordinator(2);
        let (sid, json) = scheduled(&mut c);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["session_id"], serde_json::json!(sid));
        let stages = v["stages"].as_array().unwrap();
        assert_eq!(stages.len(), 2);
        let peers: Vec<&str> = stages
            .iter()
            .map(|s| s["peer_id"].as_str().unwrap())
            .collect();
        assert_eq!(
            peers,
            ["peer-a", "peer-b"],
            "most available RAM first; peer-c is not ready"
        );
        let mut next = 0;
        for s in stages {
            let (start, end) = (
                s["layer_range"]["start"].as_u64().unwrap(),
                s["layer_range"]["end"].as_u64().unwrap(),
            );
            assert_eq!(start, next);
            assert!(end >= start);
            next = end + 1;
        }
        assert_eq!(next, 4);
        assert_eq!(stages[0]["includes_embedding"], serde_json::json!(true));
        assert_eq!(stages[1]["includes_output_head"], serde_json::json!(true));
    }

    #[test]
    fn executors_built_from_the_schedule_know_their_neighbours() {
        let mut c = coordinator(2);
        let (sid, json) = scheduled(&mut c);
        let first = ok(PyStageExecutor::new(0, &json), "stage 0");
        let last = ok(PyStageExecutor::new(1, &json), "stage 1");
        assert_eq!((first.session_id(), first.stage_index()), (sid.as_str(), 0));
        assert!(first.is_first_stage() && !first.is_last_stage());
        assert!(!last.is_first_stage() && last.is_last_stage());
        assert_eq!(first.next_stage_peer_id(), Some("peer-b"));
        assert_eq!(first.prev_stage_peer_id(), None);
        assert_eq!(last.prev_stage_peer_id(), Some("peer-a"));
        assert_eq!(last.next_stage_peer_id(), None);

        let mut e = first;
        assert!(!e.is_complete());
        e.mark_micro_batch_completed();
        assert_eq!(e.micro_batches_completed(), 1);
        assert!(!e.is_complete());
        e.mark_micro_batch_completed();
        assert!(
            e.is_complete(),
            "complete after the configured two micro-batches"
        );
    }

    #[test]
    fn an_executor_refuses_an_unknown_stage_or_malformed_schedule() {
        let mut c = coordinator(1);
        let (_, json) = scheduled(&mut c);
        assert!(PyStageExecutor::new(2, &json).is_err());
        assert!(PyStageExecutor::new(0, "not json").is_err());
        assert!(PyStageExecutor::new(0, "{}").is_err());
    }

    #[test]
    fn a_session_runs_once_every_stage_is_ready_and_then_completes() {
        let mut c = coordinator(1);
        let (sid, _) = scheduled(&mut c);
        assert_eq!(
            c.handle_stage_ready(&sid, 0).ok(),
            Some(None),
            "one of two ready"
        );
        assert_eq!(c.session_state(&sid).as_deref(), Some("Scheduled"));
        let start = ok(c.handle_stage_ready(&sid, 1), "ready");
        assert!(
            start.is_some_and(|m| !m.is_empty()),
            "all ready: start message"
        );
        assert_eq!(c.session_state(&sid).as_deref(), Some("Running"));
        assert_eq!(c.active_session_ids(), vec![sid.clone()]);
        let done = ok(c.handle_session_complete(&sid, 42), "completes");
        assert!(!done.is_empty());
        assert_eq!(c.session_state(&sid).as_deref(), Some("Completed"));
        assert!(
            c.active_session_ids().is_empty(),
            "a completed session is not active"
        );
    }

    #[test]
    fn a_stage_failure_fails_the_session() {
        let mut c = coordinator(1);
        let (sid, _) = scheduled(&mut c);
        let msg = ok(
            c.handle_stage_failure(&sid, "peer-b", 1, "out of memory"),
            "failure is reported",
        );
        assert!(!msg.is_empty());
        assert_eq!(c.session_state(&sid).as_deref(), Some("Failed"));
        assert!(c.active_session_ids().is_empty());
    }

    #[test]
    fn unknown_sessions_and_unschedulable_offers_are_refused() {
        let mut c = coordinator(1);
        assert!(c
            .handle_capability_offer("nope", &capability("p", GIB, true))
            .is_err());
        assert!(c.finalize_schedule("nope", 64).is_err());
        assert!(c.handle_stage_ready("nope", 0).is_err());
        assert_eq!(c.session_state("nope"), None);

        let (sid, _) = ok(c.propose_session("tiny", "h", 4, "peer-a"), "proposes");
        // No pipeline-ready node: the planner refuses rather than scheduling.
        ok(
            c.handle_capability_offer(&sid, &capability("idle", 8 * GIB, false)),
            "offer",
        );
        assert!(c.finalize_schedule(&sid, 64).is_err());
        assert_eq!(c.session_state(&sid).as_deref(), Some("Forming"));
    }

    #[test]
    fn config_and_capability_wrappers_expose_what_they_were_given() {
        let cfg = PyPipelineConfig::new(Some(3), Some(256), Some(5), Some(4), Some(9));
        assert_eq!(
            (
                cfg.num_micro_batches(),
                cfg.max_seq_len(),
                cfg.heartbeat_interval_secs(),
                cfg.heartbeat_timeout_factor(),
                cfg.tensor_timeout_secs(),
            ),
            (Some(3), 256, 5, 4, 9)
        );
        let d = PyPipelineConfig::new(None, None, None, None, None);
        let def = PipelineConfig::default();
        assert_eq!(
            (
                d.num_micro_batches(),
                d.max_seq_len(),
                d.tensor_timeout_secs()
            ),
            (
                def.num_micro_batches,
                def.max_seq_len,
                def.tensor_timeout_secs
            )
        );
        let cap = capability("peer-z", 3 * GIB, true);
        assert_eq!(
            (
                cap.peer_id(),
                cap.available_ram_bytes(),
                cap.ram_bytes(),
                cap.pipeline_ready()
            ),
            ("peer-z", 3 * GIB, 6 * GIB, true)
        );
        assert_eq!(
            cap.__repr__(),
            "PipelineCapability(peer='peer-z', ram=3072MB, ready=true)"
        );
    }
}
