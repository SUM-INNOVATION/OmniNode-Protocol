//! #[pyclass] wrapper for OmniNet.

use std::time::Duration;

use pyo3::prelude::*;

use omni_net::OmniNet;

use crate::errors::{anyhow_to_pyerr, PyOmniError};
use crate::events::PyNetEvent;
use crate::runtime::get_runtime;
use crate::types::PyNetConfig;

#[pyclass(name = "OmniNet")]
pub struct PyOmniNet {
    inner: Option<OmniNet>,
}

#[pymethods]
impl PyOmniNet {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyNetConfig>) -> PyResult<Self> {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        let net = anyhow_to_pyerr(get_runtime().block_on(OmniNet::new(cfg)))?;
        Ok(Self { inner: Some(net) })
    }

    /// Publish a message to a Gossipsub topic.
    fn publish(&self, topic: &str, data: Vec<u8>) -> PyResult<()> {
        let net = self
            .inner
            .as_ref()
            .ok_or_else(|| PyOmniError::new_err("OmniNet is shut down"))?;
        anyhow_to_pyerr(get_runtime().block_on(net.publish(topic, data)))
    }

    /// Request a shard chunk from a remote peer.
    #[pyo3(signature = (peer_id, cid, offset=None, max_bytes=None))]
    fn request_shard(
        &self,
        peer_id: &str,
        cid: &str,
        offset: Option<u64>,
        max_bytes: Option<u64>,
    ) -> PyResult<()> {
        let net = self
            .inner
            .as_ref()
            .ok_or_else(|| PyOmniError::new_err("OmniNet is shut down"))?;
        let pid: libp2p::PeerId = peer_id
            .parse()
            .map_err(|e| PyOmniError::new_err(format!("invalid PeerId: {e}")))?;
        anyhow_to_pyerr(get_runtime().block_on(net.request_shard_chunk(
            pid,
            cid.to_string(),
            offset,
            max_bytes,
        )))
    }

    /// Respond to a shard request (for serving shards to peers).
    #[pyo3(signature = (channel_id, cid, offset, total_bytes, data, error=None))]
    fn respond_shard(
        &self,
        channel_id: u64,
        cid: &str,
        offset: u64,
        total_bytes: u64,
        data: Vec<u8>,
        error: Option<String>,
    ) -> PyResult<()> {
        let net = self
            .inner
            .as_ref()
            .ok_or_else(|| PyOmniError::new_err("OmniNet is shut down"))?;
        let response = omni_net::ShardResponse {
            cid: cid.to_string(),
            offset,
            total_bytes,
            data,
            error,
        };
        anyhow_to_pyerr(get_runtime().block_on(net.respond_shard(channel_id, response)))
    }

    /// Poll for the next network event.
    ///
    /// Blocks until an event is available or `timeout_secs` elapses.
    /// Returns `None` on timeout or if the swarm has stopped.
    #[pyo3(signature = (timeout_secs=None))]
    fn next_event(&mut self, timeout_secs: Option<f64>) -> PyResult<Option<PyNetEvent>> {
        let net = self
            .inner
            .as_mut()
            .ok_or_else(|| PyOmniError::new_err("OmniNet is shut down"))?;
        let rt = get_runtime();
        let event = match timeout_secs {
            Some(t) => rt.block_on(async {
                tokio::time::timeout(Duration::from_secs_f64(t), net.next_event())
                    .await
                    .ok()
                    .flatten()
            }),
            None => rt.block_on(net.next_event()),
        };
        Ok(event.map(PyNetEvent::from))
    }

    /// Gracefully shut down the P2P node.
    fn shutdown(&mut self) -> PyResult<()> {
        if let Some(net) = self.inner.as_ref() {
            anyhow_to_pyerr(get_runtime().block_on(net.shutdown()))?;
        }
        self.inner = None;
        Ok(())
    }

    /// Context manager: `with OmniNet() as net: ...`
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager exit — shuts down the node.
    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.shutdown()?;
        Ok(false)
    }

    fn __repr__(&self) -> String {
        let status = if self.inner.is_some() { "active" } else { "shut down" };
        format!("OmniNet(status='{status}')")
    }
}
