//! omni-bridge — PyO3 bindings for the OmniNode Protocol.
//!
//! This crate produces a native Python extension module (`omninode._omni_bridge`)
//! that exposes zero-copy shard access, model ingestion, and P2P networking to
//! Python code.

use pyo3::prelude::*;

pub mod errors;
pub mod events;
pub mod net;
pub mod runtime;
pub mod shard_view;
pub mod store;
pub mod types;

/// Native extension module entry point: `omninode._omni_bridge`
#[pymodule]
fn _omni_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── Exceptions ───────────────────────────────────────────────────────
    m.add("OmniError", m.py().get_type::<errors::PyOmniError>())?;
    m.add("StoreError", m.py().get_type::<errors::PyStoreError>())?;

    // ── Configuration ────────────────────────────────────────────────────
    m.add_class::<types::PyNetConfig>()?;
    m.add_class::<types::PyStoreConfig>()?;

    // ── Model types ──────────────────────────────────────────────────────
    m.add_class::<types::PyLayerRange>()?;
    m.add_class::<types::PyShardDescriptor>()?;
    m.add_class::<types::PyModelManifest>()?;

    // ── Storage ──────────────────────────────────────────────────────────
    m.add_class::<store::PyOmniStore>()?;
    m.add_class::<shard_view::PyShardView>()?;

    // ── Networking ───────────────────────────────────────────────────────
    m.add_class::<net::PyOmniNet>()?;
    m.add_class::<events::PyNetEvent>()?;

    Ok(())
}
