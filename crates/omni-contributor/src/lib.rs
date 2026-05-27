//! Phase 5 Stage 12.0 — Contributor Inference Node workflow.
//!
//! Off-chain only. Implements numbered chart items 2–6: dispatch +
//! verification requirement → fetch model/input from SNIP → run local
//! inference → produce final response + evidence artifact →
//! deliver/publish response.
//!
//! Stage 12.0 ships **AttestationOnly** evidence mode only. Production
//! proof-mode evidence is a closed-enum extension reserved for a future
//! `schema_version: 2` migration and is **not** present in any Rust
//! enum in this crate.
//!
//! No chain wire / Stage 7b tx / SUM Chain RPC / `InferenceAttestationDigest`
//! / on-chain verification / proof-system allowlist touched. Stage 11d.3
//! reframe posture preserved: the chain remains neutral; proof acceptance
//! is a local verifier policy decision, and this crate's verifier is one
//! such local policy gate.
//!
//! ## Module map
//!
//! - [`job`]       — `ContributorJob` + `JobAccounting` + closed
//!                   `VerificationRequirement` + `BaseUnitRewardPolicy`.
//! - [`result`]    — `ContributorResult` + `MeasuredAccounting` +
//!                   `StageContribution` + closed `Evidence` +
//!                   closed `WorkUnitKind`.
//! - [`canonical`] — Domain separators + bincode-1.3 canonical bodies
//!                   for hashing/signing. Frozen for `schema_version: 1`.
//! - [`signing`]   — Ed25519 wrappers via `libp2p-identity` directly.
//!                   Does NOT depend on `omni-zkml`; contributor keys
//!                   are role-separate from any chain-attestation seed.
//! - [`snip`]      — `publish_bytes` / `fetch_bytes` tempfile wrappers
//!                   around the existing `omni_store::SnipV2Adapter`
//!                   path API. No new SNIP wire / root format.
//! - [`runner`]    — `InferenceRunner` trait + `StubRunner` +
//!                   `ExternalCommandRunner`. The only compute boundary.
//! - [`run`]       — `run_job` orchestrator (validate → fetch → run →
//!                   sign → publish → emit result).
//! - [`verify`]    — `verify_result` off-chain verification pipeline.
//! - [`error`]     — Typed error surface.

pub mod canonical;
pub mod discover;
pub mod error;
pub mod job;
pub mod net;
pub mod posted;
pub mod relay;
pub mod result;
pub mod run;
pub mod runner;
pub mod signing;
pub mod snip;
pub mod verify;
pub mod watch;

pub use discover::{DiscoveredEntry, FilesystemSource, JobSource, NetworkSource};
pub use error::{
    CanonicalError, ContributorError, DiscoverError, RelayError, RunnerError, SchemaError,
    SigningError, SnipError, VerifyError,
};
pub use net::{
    NetworkPostedJobAnnouncement, NetworkPostedResultAnnouncement, NET_SCHEMA_VERSION,
};
pub use relay::{ContributorRelay, InMemoryRelay};
#[cfg(feature = "network")]
pub use relay::OmniNetRelay;
pub use job::{
    BaseUnitRewardPolicy, ContributorJob, JobAccounting, VerificationRequirement,
};
pub use posted::{PostedJob, PostedResultLink, POSTED_SCHEMA_VERSION};
pub use result::{
    ContributorResult, Evidence, MeasuredAccounting, StageContribution, WorkUnitKind,
};
pub use run::{run_job, RunJobOptions};
pub use runner::{ExternalCommandRunner, InferenceRunner, RunOutput, StubRunner};
pub use signing::{ContributorSigner, DispatcherSigner};
pub use verify::{verify_result, VerifyOutcome};
pub use watch::{
    process_result_announcement, publish_result_link_for, run_watch_loop, AcceptFilters,
    CostCaps, EventEmitter, PublishedResultLink, ResultAnnouncementOutcome,
    ResultBroadcaster, SkipReason, StdoutEmitter, WatchEvent, WatchOptions,
};
