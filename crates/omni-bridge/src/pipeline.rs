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

#[pyclass(name = "PipelineConfig")]
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

#[pyclass(name = "PipelineCapability")]
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
