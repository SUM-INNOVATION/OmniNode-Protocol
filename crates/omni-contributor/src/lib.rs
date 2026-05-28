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
pub mod handoff;
pub mod handoff_verify;
pub mod job;
pub mod net;
pub mod posted;
pub mod relay;
pub mod result;
pub mod run;
pub mod runner;
pub mod session;
pub mod session_verify;
pub mod signing;
pub mod snip;
pub mod tensor_transport;
pub mod verify;
pub mod watch;

pub use discover::{DiscoveredEntry, FilesystemSource, JobSource, NetworkSource};
pub use error::{
    CanonicalError, ContributorError, DiscoverError, RelayError, RunnerError, SchemaError,
    SigningError, SnipError, VerifyError,
};
pub use net::{
    NetworkAggregatedResultAnnouncement, NetworkContributorJoinedAnnouncement,
    NetworkPartialResultAnnouncement, NetworkPostedJobAnnouncement,
    NetworkPostedResultAnnouncement, NetworkSessionOpenedAnnouncement,
    NetworkWorkAssignedAnnouncement, NET_SCHEMA_VERSION,
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
pub use runner::{
    ExternalCommandRunner, InferenceRunner, RunOutput, RunOutputWithActivation,
    RunnerOutputActivation, StubRunner,
};
pub use session::{
    AggregatedContributorResult, AggregatedPartialRef, ContributorJoin, ExecutionSession,
    PartialContributorResult, WorkAssignment, WorkKind, RUNNER_KIND_MAX,
    SESSION_SCHEMA_VERSION, WORK_KIND_CUSTOM_LABEL_MAX,
};
pub use handoff::{
    ActivationHandoff, TensorDtype, HANDOFF_BYTE_LEN_MAX, HANDOFF_CHUNK_COUNT_MAX,
    HANDOFF_CHUNK_MAX_BYTES, HANDOFF_SCHEMA_VERSION, HANDOFF_SHAPE_RANK_MAX,
};
pub use handoff_verify::{
    verify_activation_handoff, ChunkOutcome, HandoffReceiver, HandoffStreamKey,
    HandoffVerifyOutcome,
};
pub use tensor_transport::{
    InMemoryTensorTransport, TensorTransport, TensorTransportError,
};
#[cfg(feature = "network")]
pub use tensor_transport::OmniNetTensorTransport;
pub use session_verify::{
    check_not_expired, process_aggregated_result_announcement,
    process_contributor_joined_announcement, process_partial_result_announcement,
    process_session_opened_announcement, process_work_assigned_announcement,
    verify_aggregated_result, verify_contributor_join, verify_execution_session,
    verify_partial_result, verify_work_assignment, AnnouncementOutcome,
    SessionVerifyOutcome,
};
pub use signing::{ContributorSigner, CoordinatorSigner, DispatcherSigner};
pub use verify::{verify_result, VerifyOutcome};
pub use watch::{
    process_result_announcement, publish_result_link_for, run_watch_loop, AcceptFilters,
    CostCaps, EventEmitter, PublishedResultLink, ResultAnnouncementOutcome,
    ResultBroadcaster, SkipReason, StdoutEmitter, WatchEvent, WatchOptions,
};
