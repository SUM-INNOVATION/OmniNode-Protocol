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
pub mod peer_advert;
pub mod peer_routing;
pub mod planner;
pub mod posted;
pub mod relay;
pub mod repair;
pub mod result;
pub mod run;
pub mod runner;
pub mod session;
pub mod session_verify;
pub mod signing;
pub mod snip;
pub mod state;
pub mod status;
pub mod supersession;
pub mod supersession_verify;
pub mod tensor_transport;
pub mod verify;
pub mod watch;

pub use discover::{DiscoveredEntry, FilesystemSource, JobSource, NetworkSource};
pub use error::{
    CanonicalError, ContributorError, DiscoverError, PlannerError, RelayError, RepairError,
    RunnerError, SchemaError, SigningError, SnipError, StatusError, VerifyError,
};
pub use repair::{
    build_session_repair_plan, build_session_repair_plan_with_reason,
    check_reassign_targets_active_missing, check_repair_eligible, repair_plan_hash_hex,
    source_status_hash_hex, RepairAction, RepairStrategy, SessionRepairPlan,
    REPAIR_PLAN_SCHEMA_VERSION,
};
pub use status::{
    build_session_status_report, AssignmentStatus, SessionOverallStatus,
    SessionStatusReport, SupersessionStatus, STATUS_SCHEMA_VERSION,
};
pub use planner::{
    plan_assignments, plan_hash_hex, AssignmentPlan, ModelPlan, ModelPlanStage,
    PlannedAssignment, PlannerInputs, PlannerStrategy, MODEL_PLAN_SCHEMA_VERSION,
    PLANNER_SCHEMA_VERSION,
};
pub use net::{
    NetworkAggregatedResultAnnouncement, NetworkContributorJoinedAnnouncement,
    NetworkPartialResultAnnouncement, NetworkPeerAdvertisementAnnouncement,
    NetworkPostedJobAnnouncement, NetworkPostedResultAnnouncement,
    NetworkSessionOpenedAnnouncement, NetworkWorkAssignedAnnouncement,
    NetworkWorkAssignmentSupersessionAnnouncement, NET_SCHEMA_VERSION,
};
pub use peer_advert::{
    ContributorPeerAdvertisement, PeerCapabilities, PEER_ADVERTISEMENT_MAX_LIFETIME_SECS,
    PEER_ADVERTISEMENT_SCHEMA_VERSION,
};
pub use peer_routing::{
    process_peer_advertisement_announcement, verify_peer_advertisement_body,
    PeerAdvertisementOutcome, PeerRoutingCache, ResolvedPeerRoute, RouteResolution,
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
    process_assignment_supersession_announcement,
    process_contributor_joined_announcement, process_partial_result_announcement,
    process_session_opened_announcement, process_work_assigned_announcement,
    verify_aggregated_result, verify_aggregated_result_with_supersessions,
    verify_contributor_join, verify_execution_session, verify_partial_result,
    verify_work_assignment, AnnouncementOutcome, SessionVerifyOutcome,
};
pub use signing::{ContributorSigner, CoordinatorSigner, DispatcherSigner};
pub use supersession::{
    SupersessionReason, WorkAssignmentSupersession,
    SUPERSESSION_REASON_CUSTOM_LABEL_MAX, SUPERSESSION_SCHEMA_VERSION,
};
pub use supersession_verify::verify_assignment_supersession;
pub use state::{
    ContributorStateStore, PruneReport, StateNamespace, StateObjectKind,
    StateVersionMeta, STATE_VERSION,
};
pub use verify::{verify_result, VerifyOutcome};
pub use watch::{
    process_result_announcement, publish_result_link_for, run_watch_loop, AcceptFilters,
    CostCaps, EventEmitter, PublishedResultLink, ResultAnnouncementOutcome,
    ResultBroadcaster, SkipReason, StdoutEmitter, WatchEvent, WatchOptions,
};
