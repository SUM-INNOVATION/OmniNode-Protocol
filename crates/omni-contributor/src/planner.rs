//! Stage 12.8 — local coordinator-side assignment planner.
//!
//! Given a verified `ExecutionSession`, the set of verified
//! `ContributorJoin`s for that session, and (optionally) the set of
//! verified `ContributorPeerAdvertisement`s for that session,
//! produce a deterministic `AssignmentPlan` JSON that lists which
//! contributor handles which slice of work.
//!
//! ## What the planner is NOT
//!
//! - **NOT a marketplace.** The planner does not bid, score, rank by
//!   resources beyond an eligibility floor, weight by reputation,
//!   compute pricing, or pick winners. It produces a local
//!   coordination hint, period.
//! - **NOT a scheduler.** No RAM-weighted ranking, no
//!   historical-performance scoring, no global state.
//! - **NOT a network protocol.** `AssignmentPlan` is local-only and
//!   **unsigned** in v1; the signed trust artifact remains
//!   `WorkAssignment` at publish time. Adding a
//!   `coordinator_signature_hex` here is a `schema_version: 2`
//!   migration.
//! - **NOT a chain authority.** Coordinator is a process role.
//!
//! ## Determinism contract
//!
//! After eligibility filtering, contributors are sorted by
//! `contributor_pubkey_hex` (lexicographic ASCII order on the
//! lowercase hex string). All strategies consume that sorted list
//! in order — no tie-break randomness, no clock-dependent picks.
//! Re-running `plan_assignments` with the same inputs (and the same
//! `now_utc` if any advert is on the eligibility boundary) yields
//! byte-identical output, including `plan_hash`.
//!
//! ## Eligibility
//!
//! A `ContributorJoin` is eligible if:
//!   1. Its `available_ram_bytes >= min_available_ram_bytes`
//!      (operator floor, default 0 — i.e. no filter).
//!   2. If `require_live_routing` is set: a non-expired
//!      `ContributorPeerAdvertisement` for the same
//!      `(session_id, contributor_pubkey_hex)` exists AND its
//!      `capabilities.supports_live_handoff` is true AND
//!      `capabilities.supported_dtypes.contains(required_dtype)`.
//!
//! Filtering is pass/fail; there is no ranking. The minimum-RAM
//! filter is an "is this contributor able to participate at all"
//! gate, not a quality signal — by design, two contributors that
//! both clear the floor are interchangeable to the planner.

use serde::{Deserialize, Serialize};

use crate::error::PlannerError;
use crate::handoff::TensorDtype;
use crate::peer_advert::ContributorPeerAdvertisement;
use crate::peer_routing::verify_peer_advertisement_body;
use crate::result::WorkUnitKind;
use crate::session::{
    ContributorJoin, ExecutionSession, WorkAssignment, WorkKind, SESSION_SCHEMA_VERSION,
};
use crate::session_verify::{
    check_not_expired, verify_contributor_join, verify_execution_session, SessionVerifyOutcome,
};

/// Pinned planner schema version. A future stage that needs e.g. a
/// signed `AssignmentPlan` field bumps this to `2` and forks the
/// shape; today's binaries refuse plans they don't know how to read.
pub const PLANNER_SCHEMA_VERSION: u32 = 1;

/// Pinned model-plan schema version. Separate from the planner
/// schema because operators may edit the model-plan independently.
pub const MODEL_PLAN_SCHEMA_VERSION: u32 = 1;

// ── PlannerStrategy ──────────────────────────────────────────────

/// Closed enum of v1 strategies. Adding a strategy is a
/// `schema_version: 2` migration (or a new
/// `PlannerStrategy::Custom { label }` escape hatch, mirroring
/// `WorkKind::Custom`, in a future stage).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub enum PlannerStrategy {
    /// Equal layer split across eligible contributors (or
    /// model-plan-driven split when a `ModelPlan` is supplied).
    SequentialLayers,
    /// One contributor handles the entire work envelope.
    SingleContributor,
    /// Stage_index N → eligible[N % eligible.len()] in deterministic
    /// pubkey-sorted order.
    RoundRobin,
}

// ── ModelPlan (operator-supplied work shape) ──────────────────────

/// Operator-supplied JSON describing the *work shape* of a pooled
/// inference session. The planner consumes this; contributors never
/// see it directly. Not signed, not SNIP-published, not canonical
/// bytes anywhere.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPlan {
    pub schema_version: u32,
    pub stages: Vec<ModelPlanStage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPlanStage {
    pub stage_index: u32,
    pub work_kind: WorkKind,
    pub expected_work_units: u64,
    pub expected_work_unit_kind: WorkUnitKind,
}

impl ModelPlan {
    /// Structural validation independent of any session. Called by
    /// `plan_assignments` before the strategy runs.
    pub fn validate(&self) -> Result<(), PlannerError> {
        if self.schema_version != MODEL_PLAN_SCHEMA_VERSION {
            return Err(PlannerError::UnsupportedModelPlanVersion {
                got: self.schema_version,
                expected: MODEL_PLAN_SCHEMA_VERSION,
            });
        }
        if self.stages.is_empty() {
            return Err(PlannerError::InconsistentInputs {
                reason: "model-plan has zero stages".into(),
            });
        }
        for (i, stage) in self.stages.iter().enumerate() {
            if stage.stage_index as usize != i {
                return Err(PlannerError::InvalidModelPlanStage {
                    stage_index: stage.stage_index,
                    reason: format!(
                        "stage_index must equal its position in the stages array \
                         (expected {i}, got {})",
                        stage.stage_index
                    ),
                });
            }
            if stage.expected_work_units == 0 {
                return Err(PlannerError::InvalidModelPlanStage {
                    stage_index: stage.stage_index,
                    reason: "expected_work_units must be > 0".into(),
                });
            }
            if let WorkKind::Layers { start, end } = &stage.work_kind {
                if start >= end {
                    return Err(PlannerError::InvalidModelPlanStage {
                        stage_index: stage.stage_index,
                        reason: format!(
                            "WorkKind::Layers requires start < end (got start={start}, end={end})"
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

// ── PlannerInputs / outputs ──────────────────────────────────────

/// All-borrows input bundle. Construction is a separate concern from
/// the planner — the caller (CLI or test) loads and re-verifies
/// envelopes from disk or memory first.
#[derive(Debug)]
pub struct PlannerInputs<'a> {
    pub session: &'a ExecutionSession,
    pub joins: &'a [ContributorJoin],
    /// Empty unless `require_live_routing` is true.
    pub peer_adverts: &'a [ContributorPeerAdvertisement],
    pub required_dtype: TensorDtype,
    pub min_available_ram_bytes: u64,
    pub max_assignments: Option<u32>,
    pub require_live_routing: bool,
    /// `Some(N)` when the caller passed `--layer-count`; used by
    /// `sequential-layers` (equal split) and `round-robin` (defines
    /// the layer envelope for a single-stage-per-contributor plan)
    /// when no `ModelPlan` is supplied.
    pub layer_count: Option<u32>,
}

/// Local, unsigned plan artifact. Carries everything an operator
/// needs to review the assignment shape before `assign-session-plan`
/// publishes them as signed `WorkAssignment` envelopes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssignmentPlan {
    pub schema_version: u32,
    /// Copy of `ExecutionSession.session_id`. Drift-checked at
    /// publish time.
    pub session_id: String,
    pub planner_version: String,
    pub strategy: PlannerStrategy,
    pub required_dtype: TensorDtype,
    pub created_at_utc: String,
    /// Whose plan this is. Optional — the field exists so dry-run
    /// reviewers can confirm the right operator signed off, but the
    /// PLAN itself is unsigned in v1; the signed artifact is the
    /// WorkAssignment at publish time.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub coordinator_pubkey_hex: Option<String>,
    pub assignments: Vec<PlannedAssignment>,
    /// BLAKE3 hex over the canonical JSON body with `plan_hash`
    /// itself set to the empty string. Computed by
    /// `plan_hash_hex`. Re-checked by `assign-session-plan` to
    /// guard against accidental edits between dry-run and publish.
    pub plan_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedAssignment {
    pub stage_index: u32,
    pub contributor_pubkey_hex: String,
    pub work_kind: WorkKind,
    pub expected_work_units: u64,
    pub expected_work_unit_kind: WorkUnitKind,
}

impl PlannedAssignment {
    /// Build a `WorkAssignment` body (unsigned) from a planned entry.
    /// Filling `assignment_id` + `coordinator_signature_hex` is the
    /// caller's job — the publish helper in `omni-node` does that.
    pub fn to_unsigned_work_assignment(
        &self,
        session_id: &str,
        assigned_at_utc: &str,
    ) -> WorkAssignment {
        WorkAssignment {
            schema_version: SESSION_SCHEMA_VERSION,
            session_id: session_id.to_string(),
            assignment_id: String::new(),
            stage_index: self.stage_index,
            contributor_pubkey_hex: self.contributor_pubkey_hex.clone(),
            work_kind: self.work_kind.clone(),
            expected_work_units: self.expected_work_units,
            expected_work_unit_kind: self.expected_work_unit_kind,
            assigned_at_utc: assigned_at_utc.to_string(),
            coordinator_signature_hex: String::new(),
        }
    }
}

// ── plan_hash_hex ────────────────────────────────────────────────

/// BLAKE3-hex digest of the canonical JSON body with `plan_hash`
/// cleared. Deterministic across serde round-trips because we
/// re-serialize the cleared body through `serde_json::to_vec`.
pub fn plan_hash_hex(plan: &AssignmentPlan) -> String {
    let mut cloned = plan.clone();
    cloned.plan_hash = String::new();
    let bytes = serde_json::to_vec(&cloned).expect("serialize plan");
    let h = blake3::hash(&bytes);
    let mut s = String::with_capacity(64);
    for b in h.as_bytes() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ── plan_assignments (the entry point) ───────────────────────────

/// Validate inputs + run the requested strategy + assemble the plan.
/// `now_utc` is RFC 3339 with `Z` suffix; used only when
/// `require_live_routing` is set, to evaluate advert expiry.
pub fn plan_assignments(
    inputs: PlannerInputs<'_>,
    strategy: PlannerStrategy,
    model_plan: Option<&ModelPlan>,
    now_utc: &str,
    created_at_utc: &str,
) -> Result<AssignmentPlan, PlannerError> {
    // Defense-in-depth: re-verify the session and each join. The
    // CLI already re-verified at load time (Stage 12.7 trust
    // boundary lives in the caller), but the planner is a public
    // library entry too — a test or alternate caller might forget.
    if !verify_execution_session(inputs.session).is_ok() {
        return Err(PlannerError::InconsistentInputs {
            reason: "supplied session failed verify_execution_session".into(),
        });
    }
    // Expired sessions must not produce new assignments — even
    // through the `--no-prune-state-on-start` path. The legacy
    // pre-12.7 publish path already refused expired sessions via
    // the inner verifier; this check matches that posture for the
    // state-dir + planner flow.
    if let SessionVerifyOutcome::ExpiredAtCheck { now, expires_at } =
        check_not_expired(now_utc, &inputs.session.expires_at_utc)
    {
        return Err(PlannerError::SessionExpired {
            now_utc: now,
            expires_at_utc: expires_at,
        });
    }
    let joins_for_session: Vec<&ContributorJoin> = inputs
        .joins
        .iter()
        .filter(|j| {
            j.session_id == inputs.session.session_id
                && verify_contributor_join(inputs.session, j).is_ok()
        })
        .collect();
    if joins_for_session.is_empty() {
        return Err(PlannerError::NoEligibleContributors {
            joins_total: inputs.joins.len(),
            filtered_out: inputs.joins.len(),
            reason: "no joins survived session-binding + signature re-verification"
                .into(),
        });
    }

    if let Some(mp) = model_plan {
        mp.validate()?;
    }

    // Build the verified-advert map (session_id, pubkey) → advert
    // only when live routing is required. The CLI re-verified each
    // advert at load time too, but defense-in-depth re-checks here
    // are cheap and keep the library entry trustworthy.
    let live_lookup: std::collections::HashMap<String, &ContributorPeerAdvertisement> =
        if inputs.require_live_routing {
            inputs
                .peer_adverts
                .iter()
                .filter(|a| a.session_id == inputs.session.session_id)
                .filter(|a| {
                    matches!(
                        verify_peer_advertisement_body(
                            a,
                            inputs.joins,
                            Some(now_utc),
                        ),
                        crate::peer_routing::PeerAdvertisementOutcome::Verified { .. }
                    )
                })
                .map(|a| (a.contributor_pubkey_hex.clone(), a))
                .collect()
        } else {
            std::collections::HashMap::new()
        };

    // Eligibility filter. Pass/fail, no ranking.
    let mut filtered_out: usize = 0;
    let mut eligible: Vec<&ContributorJoin> = joins_for_session
        .into_iter()
        .filter(|j| {
            if j.available_ram_bytes < inputs.min_available_ram_bytes {
                filtered_out += 1;
                return false;
            }
            if inputs.require_live_routing {
                let Some(advert) = live_lookup.get(&j.contributor_pubkey_hex) else {
                    filtered_out += 1;
                    return false;
                };
                if !advert.capabilities.supports_live_handoff {
                    filtered_out += 1;
                    return false;
                }
                if !advert
                    .capabilities
                    .supported_dtypes
                    .contains(&inputs.required_dtype)
                {
                    filtered_out += 1;
                    return false;
                }
            }
            true
        })
        .collect();

    if eligible.is_empty() {
        return Err(PlannerError::NoEligibleContributors {
            joins_total: inputs.joins.len(),
            filtered_out,
            reason: if inputs.require_live_routing {
                "joins filtered by min_available_ram + live-routing peer-advert constraints".into()
            } else {
                "joins filtered by min_available_ram_bytes floor".into()
            },
        });
    }

    // ── Deterministic ordering (the entire ranking story) ──
    eligible.sort_by(|a, b| {
        a.contributor_pubkey_hex.cmp(&b.contributor_pubkey_hex)
    });

    // ── max_assignments policy ──
    //
    // Some strategies are "stage-count-driven": the number of
    // assignments they emit is fixed by the model-plan or the
    // `--layer-count` envelope, NOT by the eligible-contributor
    // count. For those, a `--max-assignments` cap below that fixed
    // count would silently drop layers / stages — incomplete plan,
    // valid `plan_hash`. Refuse instead with a typed error so the
    // operator can either raise the cap or pick a different
    // strategy.
    //
    // The one contributor-count-driven path is
    // `sequential-layers` WITHOUT a model-plan: equal layer split
    // across `eligible.len()` contributors. There, the operator
    // intent of `--max-assignments` is "use at most N contributors,"
    // so we truncate the eligible list BEFORE the strategy splits,
    // and the resulting layer ranges still cover the full envelope.
    let required: Option<u32> = required_assignment_count(
        strategy,
        model_plan,
        inputs.layer_count,
    );
    if let (Some(max), Some(req)) = (inputs.max_assignments, required) {
        if max < req {
            return Err(PlannerError::MaxAssignmentsTooSmall {
                strategy: strategy_name(strategy),
                required: req,
                max,
            });
        }
    }
    if required.is_none() {
        if let Some(max) = inputs.max_assignments {
            if (max as usize) < eligible.len() {
                eligible.truncate(max as usize);
            }
        }
    }

    // ── Strategy dispatch ──
    let assignments = match strategy {
        PlannerStrategy::SingleContributor => {
            plan_single_contributor(&eligible, model_plan, inputs.layer_count)?
        }
        PlannerStrategy::SequentialLayers => {
            plan_sequential_layers(&eligible, model_plan, inputs.layer_count)?
        }
        PlannerStrategy::RoundRobin => {
            plan_round_robin(&eligible, model_plan, inputs.layer_count)?
        }
    };

    let mut plan = AssignmentPlan {
        schema_version: PLANNER_SCHEMA_VERSION,
        session_id: inputs.session.session_id.clone(),
        planner_version: format!("omni-contributor v{}", env!("CARGO_PKG_VERSION")),
        strategy,
        required_dtype: inputs.required_dtype,
        created_at_utc: created_at_utc.to_string(),
        coordinator_pubkey_hex: Some(inputs.session.coordinator_pubkey_hex.clone()),
        assignments,
        plan_hash: String::new(),
    };
    plan.plan_hash = plan_hash_hex(&plan);
    Ok(plan)
}

// ── required-count helper ─────────────────────────────────────────

/// How many assignments will the strategy emit, given the supplied
/// inputs? Returns `None` for "contributor-count-driven" strategies
/// where the operator can legitimately cap eligible contributors.
fn required_assignment_count(
    strategy: PlannerStrategy,
    model_plan: Option<&ModelPlan>,
    layer_count: Option<u32>,
) -> Option<u32> {
    match strategy {
        PlannerStrategy::SingleContributor => {
            if let Some(mp) = model_plan {
                Some(mp.stages.len() as u32)
            } else {
                // Without a model-plan the strategy emits one
                // assignment regardless of layer_count.
                Some(1)
            }
        }
        PlannerStrategy::SequentialLayers => {
            if let Some(mp) = model_plan {
                Some(mp.stages.len() as u32)
            } else {
                // Equal-split across eligible contributors —
                // contributor-count-driven, so `--max-assignments`
                // is a contributor cap. Return None to signal "cap
                // eligible up-front instead of refusing."
                let _ = layer_count;
                None
            }
        }
        PlannerStrategy::RoundRobin => {
            if let Some(mp) = model_plan {
                Some(mp.stages.len() as u32)
            } else {
                layer_count
            }
        }
    }
}

fn strategy_name(s: PlannerStrategy) -> &'static str {
    match s {
        PlannerStrategy::SingleContributor => "single-contributor",
        PlannerStrategy::SequentialLayers => "sequential-layers",
        PlannerStrategy::RoundRobin => "round-robin",
    }
}

// ── strategy implementations ─────────────────────────────────────

fn plan_single_contributor(
    eligible: &[&ContributorJoin],
    model_plan: Option<&ModelPlan>,
    layer_count: Option<u32>,
) -> Result<Vec<PlannedAssignment>, PlannerError> {
    let picked = eligible
        .first()
        .ok_or_else(|| PlannerError::InconsistentInputs {
            reason: "single_contributor called with empty eligible set".into(),
        })?;
    if let Some(mp) = model_plan {
        // All model-plan stages go to the single picked contributor.
        Ok(mp
            .stages
            .iter()
            .map(|s| PlannedAssignment {
                stage_index: s.stage_index,
                contributor_pubkey_hex: picked.contributor_pubkey_hex.clone(),
                work_kind: s.work_kind.clone(),
                expected_work_units: s.expected_work_units,
                expected_work_unit_kind: s.expected_work_unit_kind,
            })
            .collect())
    } else {
        let n = layer_count.ok_or_else(|| PlannerError::InconsistentInputs {
            reason: "single_contributor without --model-plan requires --layer-count".into(),
        })?;
        if n == 0 {
            return Err(PlannerError::InconsistentInputs {
                reason: "layer_count must be > 0".into(),
            });
        }
        Ok(vec![PlannedAssignment {
            stage_index: 0,
            contributor_pubkey_hex: picked.contributor_pubkey_hex.clone(),
            work_kind: WorkKind::Layers { start: 0, end: n },
            expected_work_units: n as u64,
            expected_work_unit_kind: WorkUnitKind::Layers,
        }])
    }
}

fn plan_sequential_layers(
    eligible: &[&ContributorJoin],
    model_plan: Option<&ModelPlan>,
    layer_count: Option<u32>,
) -> Result<Vec<PlannedAssignment>, PlannerError> {
    if let Some(mp) = model_plan {
        // Model-plan-driven: one stage per stage entry, assigned
        // round-robin across the deterministically-sorted eligible
        // set (so a 3-stage plan over 2 contributors gives
        // contributor A: stages 0 and 2, contributor B: stage 1).
        // This keeps the model-plan path deterministic and
        // independent of contributor count.
        Ok(mp
            .stages
            .iter()
            .enumerate()
            .map(|(i, s)| PlannedAssignment {
                stage_index: s.stage_index,
                contributor_pubkey_hex: eligible[i % eligible.len()]
                    .contributor_pubkey_hex
                    .clone(),
                work_kind: s.work_kind.clone(),
                expected_work_units: s.expected_work_units,
                expected_work_unit_kind: s.expected_work_unit_kind,
            })
            .collect())
    } else {
        let total_layers = layer_count.ok_or_else(|| PlannerError::InconsistentInputs {
            reason: "sequential-layers without --model-plan requires --layer-count".into(),
        })?;
        if total_layers == 0 {
            return Err(PlannerError::InconsistentInputs {
                reason: "layer_count must be > 0".into(),
            });
        }
        let contrib_count = eligible.len() as u32;
        if total_layers < contrib_count {
            return Err(PlannerError::EqualSplitProducesEmptyStage {
                layer_count: total_layers,
                contributor_count: contrib_count,
            });
        }
        // Equal split with remainder absorbed by the LAST stage so
        // total layers always equals total_layers exactly and every
        // earlier stage gets `floor(N/M)` layers (deterministic).
        let base = total_layers / contrib_count;
        let mut out = Vec::with_capacity(eligible.len());
        let mut start: u32 = 0;
        for (i, contrib) in eligible.iter().enumerate() {
            let end = if i + 1 == eligible.len() {
                total_layers
            } else {
                start + base
            };
            out.push(PlannedAssignment {
                stage_index: i as u32,
                contributor_pubkey_hex: contrib.contributor_pubkey_hex.clone(),
                work_kind: WorkKind::Layers { start, end },
                expected_work_units: (end - start) as u64,
                expected_work_unit_kind: WorkUnitKind::Layers,
            });
            start = end;
        }
        Ok(out)
    }
}

fn plan_round_robin(
    eligible: &[&ContributorJoin],
    model_plan: Option<&ModelPlan>,
    layer_count: Option<u32>,
) -> Result<Vec<PlannedAssignment>, PlannerError> {
    if let Some(mp) = model_plan {
        // Stage_index N → eligible[N % len].
        Ok(mp
            .stages
            .iter()
            .map(|s| PlannedAssignment {
                stage_index: s.stage_index,
                contributor_pubkey_hex: eligible
                    [s.stage_index as usize % eligible.len()]
                .contributor_pubkey_hex
                .clone(),
                work_kind: s.work_kind.clone(),
                expected_work_units: s.expected_work_units,
                expected_work_unit_kind: s.expected_work_unit_kind,
            })
            .collect())
    } else {
        // Without a model-plan, round-robin needs --layer-count to
        // know how many single-layer-each stages to emit. Each
        // contributor gets a single-layer stage in deterministic
        // order; if `layer_count > eligible.len()`, the assignment
        // wraps around the eligible list.
        let n = layer_count.ok_or_else(|| PlannerError::InconsistentInputs {
            reason: "round-robin without --model-plan requires --layer-count".into(),
        })?;
        if n == 0 {
            return Err(PlannerError::InconsistentInputs {
                reason: "layer_count must be > 0".into(),
            });
        }
        Ok((0..n)
            .map(|i| PlannedAssignment {
                stage_index: i,
                contributor_pubkey_hex: eligible[i as usize % eligible.len()]
                    .contributor_pubkey_hex
                    .clone(),
                work_kind: WorkKind::Layers {
                    start: i,
                    end: i + 1,
                },
                expected_work_units: 1,
                expected_work_unit_kind: WorkUnitKind::Layers,
            })
            .collect())
    }
}
