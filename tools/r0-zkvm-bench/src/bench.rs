//! Reproducible benchmark / artifact schema — **structurally always NON-SELECTION**.
//!
//! ## What this artifact is (and is not)
//!
//! This crate emits exactly one JSON artifact, [`BenchArtifact`]. It carries **no
//! measured samples**, is **not a selection-valid artifact**, and contains **no
//! proof / finalization / protocol-hash data**. It records only: a
//! forced-non-selection status, the zero-spec-hash template hash, the exact
//! toolchain-ABSENT marker, descriptive host metadata, and per-candidate
//! *synthetic* fixture identities whose phase costs are all unmeasured.
//!
//! ## Unforgeable by construction AND by deserialization
//!
//! * Every invariant-bearing field of [`BenchArtifact`], [`CandidateArchResult`],
//!   [`CostBreakdown`], and [`PhaseCost`] is **private** with read-only
//!   accessors; there is no public setter.
//! * [`PhaseCost`] has exactly one constructible state —
//!   `NOT_MEASURED_TOOLCHAIN_ABSENT` with an empty sample list — and no public
//!   constructor accepts raw samples, a digest, a status, or a spec hash.
//! * The only constructor, [`BenchArtifact::research`], FORCES the run status,
//!   the zero template spec hash (read from a [`SyntheticJournal`]), the exact
//!   ABSENT toolchain marker, and unmeasured phases.
//! * Private fields alone do not constrain `serde`, so [`BenchArtifact`] uses a
//!   hand-written [`Deserialize`] that decodes into validated `Raw*` DTOs
//!   (`#[serde(deny_unknown_fields)]`) and REJECTS: an omitted/altered status, a
//!   nonzero spec hash, any non-empty sample array, any measurement state other
//!   than `NOT_MEASURED_TOOLCHAIN_ABSENT`, a non-ABSENT toolchain digest,
//!   inconsistent candidate/hex data, or any unknown field.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

use crate::b0::enums::Candidate;
use crate::classification::{RunClass, NON_SELECTION_STATUS};
use crate::statement::SyntheticJournal;

/// The exact frozen marker recorded because no zkVM toolchain/container exists.
pub const TOOLCHAIN_ABSENT_MARKER: &str = "ABSENT: no zkVM toolchain in this environment";

/// The eligibility status of a benchmark run. Exactly one variant exists, so no
/// selection-grade status can be represented, constructed, or deserialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    /// Not usable for R0 candidate selection / B0.
    #[serde(rename = "NON_SELECTION / INVALID_FOR_B0")]
    NonSelectionInvalidForB0,
}

impl RunStatus {
    /// Derive the status from the internal run classification. Because
    /// [`RunClass`] has only non-selection variants, this can only ever yield the
    /// non-selection status.
    pub const fn for_run(class: RunClass) -> Self {
        match class {
            RunClass::NonSelectionResearch => RunStatus::NonSelectionInvalidForB0,
        }
    }

    /// The exact visible status string.
    pub const fn as_str(self) -> &'static str {
        match self {
            RunStatus::NonSelectionInvalidForB0 => NON_SELECTION_STATUS,
        }
    }
}

/// Whether a phase cost was measured. Exactly one variant exists, so a "measured"
/// state cannot be represented, constructed, or deserialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementState {
    /// No measurement — no zkVM toolchain is present.
    #[serde(rename = "NOT_MEASURED_TOOLCHAIN_ABSENT")]
    NotMeasuredToolchainAbsent,
}

/// One phase's cost. Its only constructible state is unmeasured with an empty
/// sample list; the fields are private and there is no constructor that accepts
/// timing samples.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PhaseCost {
    state: MeasurementState,
    raw_samples_ns: Vec<u64>,
}

impl PhaseCost {
    /// The ONLY constructor: an unmeasured phase (toolchain absent, no samples).
    fn unmeasured() -> Self {
        Self {
            state: MeasurementState::NotMeasuredToolchainAbsent,
            raw_samples_ns: Vec::new(),
        }
    }

    /// The (forced) measurement state.
    pub fn state(&self) -> MeasurementState {
        self.state
    }

    /// The raw per-iteration nanosecond samples — always empty.
    pub fn raw_samples_ns(&self) -> &[u64] {
        &self.raw_samples_ns
    }
}

/// Per candidate×arch cost breakdown. All phases are unmeasured; fields private.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CostBreakdown {
    model_weight_auth: PhaseCost,
    transformer_unit: PhaseCost,
    state_manifest_hashing: PhaseCost,
    wrapping_groth16: PhaseCost,
    final_verify: PhaseCost,
}

impl CostBreakdown {
    /// The ONLY constructor: every phase unmeasured.
    fn unmeasured() -> Self {
        Self {
            model_weight_auth: PhaseCost::unmeasured(),
            transformer_unit: PhaseCost::unmeasured(),
            state_manifest_hashing: PhaseCost::unmeasured(),
            wrapping_groth16: PhaseCost::unmeasured(),
            final_verify: PhaseCost::unmeasured(),
        }
    }

    pub fn model_weight_auth(&self) -> &PhaseCost {
        &self.model_weight_auth
    }
    pub fn transformer_unit(&self) -> &PhaseCost {
        &self.transformer_unit
    }
    pub fn state_manifest_hashing(&self) -> &PhaseCost {
        &self.state_manifest_hashing
    }
    pub fn wrapping_groth16(&self) -> &PhaseCost {
        &self.wrapping_groth16
    }
    pub fn final_verify(&self) -> &PhaseCost {
        &self.final_verify
    }

    /// All five phases, for exhaustive iteration in tests / reporting.
    pub fn phases(&self) -> [&PhaseCost; 5] {
        [
            &self.model_weight_auth,
            &self.transformer_unit,
            &self.state_manifest_hashing,
            &self.wrapping_groth16,
            &self.final_verify,
        ]
    }
}

/// Hardware / OS metadata for reproducibility. Descriptive only (no invariant).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HwOsMetadata {
    pub os: String,
    pub arch: String,
    pub cpu_model: String,
    pub logical_cores: u32,
    pub total_memory_bytes: u64,
}

/// One candidate×arch result. The program/dep-lock ids are synthetic fixtures;
/// the cost is forced unmeasured. Fields private.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CandidateArchResult {
    candidate: String,
    arch: String,
    guest_program_id_hex: String,
    dep_lock_hash_hex: String,
    cost: CostBreakdown,
}

impl CandidateArchResult {
    pub fn candidate(&self) -> &str {
        &self.candidate
    }
    pub fn arch(&self) -> &str {
        &self.arch
    }
    pub fn guest_program_id_hex(&self) -> &str {
        &self.guest_program_id_hex
    }
    pub fn dep_lock_hash_hex(&self) -> &str {
        &self.dep_lock_hash_hex
    }
    pub fn cost(&self) -> &CostBreakdown {
        &self.cost
    }
}

/// A named artifact hash (hex). Descriptive only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedArtifactHash {
    pub label: String,
    pub hash_hex: String,
}

/// The top-level reproducible benchmark artifact — structurally NON-SELECTION.
/// Every invariant-bearing field is private with a read-only accessor; the type
/// deserializes only through validated `Raw*` DTOs (see the module docs).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BenchArtifact {
    schema_version: u16,
    run_status: RunStatus,
    commit_sha: String,
    b0_pre_spec_hash_hex: String,
    toolchain_container_digest: String,
    hw_os: HwOsMetadata,
    results: Vec<CandidateArchResult>,
    artifact_hashes: Vec<NamedArtifactHash>,
    notes: String,
}

impl BenchArtifact {
    /// Build a research (non-selection) artifact. FORCES: the run status, the
    /// zero template spec hash (read from `journal`), the exact ABSENT toolchain
    /// marker, and unmeasured phases in every supplied result. There is no way to
    /// pass a raw spec hash, a status, a toolchain digest, or timing samples.
    pub fn research(
        commit_sha: impl Into<String>,
        journal: &SyntheticJournal,
        hw_os: HwOsMetadata,
        results: Vec<CandidateArchResult>,
        artifact_hashes: Vec<NamedArtifactHash>,
    ) -> Self {
        Self {
            schema_version: crate::b0::consts::SCHEMA_VERSION,
            run_status: RunStatus::for_run(RunClass::research()),
            commit_sha: commit_sha.into(),
            b0_pre_spec_hash_hex: hex32(&journal.spec_hash()),
            toolchain_container_digest: TOOLCHAIN_ABSENT_MARKER.to_string(),
            hw_os,
            results,
            artifact_hashes,
            notes: "Non-selection research artifact: no zkVM toolchain is present, so \
                    no proving or measurement was run. It carries no measured samples, \
                    is not a selection-valid artifact, and contains no proof / \
                    finalization / protocol-hash data. All phase costs are \
                    NOT_MEASURED_TOOLCHAIN_ABSENT and the run is \
                    NON_SELECTION / INVALID_FOR_B0."
                .to_string(),
        }
    }

    // ── read-only accessors ──────────────────────────────────────────────────
    pub fn schema_version(&self) -> u16 {
        self.schema_version
    }
    pub fn run_status(&self) -> RunStatus {
        self.run_status
    }
    pub fn status_str(&self) -> &'static str {
        self.run_status.as_str()
    }
    pub fn commit_sha(&self) -> &str {
        &self.commit_sha
    }
    pub fn b0_pre_spec_hash_hex(&self) -> &str {
        &self.b0_pre_spec_hash_hex
    }
    pub fn toolchain_container_digest(&self) -> &str {
        &self.toolchain_container_digest
    }
    pub fn hw_os(&self) -> &HwOsMetadata {
        &self.hw_os
    }
    pub fn results(&self) -> &[CandidateArchResult] {
        &self.results
    }
    pub fn artifact_hashes(&self) -> &[NamedArtifactHash] {
        &self.artifact_hashes
    }
    pub fn notes(&self) -> &str {
        &self.notes
    }

    /// Serialize to pretty JSON (propagates any serializer error).
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Parse from JSON through the validated deserializer.
    pub fn from_json(s: &str) -> serde_json::Result<Self> {
        serde_json::from_str(s)
    }
}

// ── Validated deserialization (approach B: Raw DTOs + invariant checks) ───────

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPhaseCost {
    state: MeasurementState,
    raw_samples_ns: Vec<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCostBreakdown {
    model_weight_auth: RawPhaseCost,
    transformer_unit: RawPhaseCost,
    state_manifest_hashing: RawPhaseCost,
    wrapping_groth16: RawPhaseCost,
    final_verify: RawPhaseCost,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawResult {
    candidate: String,
    arch: String,
    guest_program_id_hex: String,
    dep_lock_hash_hex: String,
    cost: RawCostBreakdown,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArtifact {
    schema_version: u16,
    run_status: RunStatus,
    commit_sha: String,
    b0_pre_spec_hash_hex: String,
    toolchain_container_digest: String,
    hw_os: HwOsMetadata,
    results: Vec<RawResult>,
    artifact_hashes: Vec<NamedArtifactHash>,
    notes: String,
}

fn is_lower_hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_zero_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b == b'0')
}

impl<'de> Deserialize<'de> for BenchArtifact {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawArtifact::deserialize(deserializer)?;

        // Status: single variant, but assert explicitly for a clear error.
        if raw.run_status != RunStatus::NonSelectionInvalidForB0 {
            return Err(D::Error::custom(
                "run_status must be NON_SELECTION / INVALID_FOR_B0",
            ));
        }
        // Schema version pinned to the frozen B0 scalar.
        if raw.schema_version != crate::b0::consts::SCHEMA_VERSION {
            return Err(D::Error::custom("unexpected schema_version"));
        }
        // Spec hash must be the zero template hash — never a materialized one.
        if !is_zero_hex64(&raw.b0_pre_spec_hash_hex) {
            return Err(D::Error::custom(
                "b0_pre_spec_hash must be the zero template hash",
            ));
        }
        // Toolchain digest must be exactly the ABSENT marker.
        if raw.toolchain_container_digest != TOOLCHAIN_ABSENT_MARKER {
            return Err(D::Error::custom(
                "toolchain_container_digest must be the exact ABSENT marker",
            ));
        }
        // Every phase of every result must be unmeasured with no samples. The
        // MeasurementState enum already forbids a "measured" state; this also
        // forbids smuggling samples under the unmeasured state.
        let check_phase = |p: &RawPhaseCost| -> Result<(), D::Error> {
            // The state must be the single unmeasured variant. Deserialization
            // already rejects any other string, but matching it here keeps the
            // field load-bearing and forces a decision if a variant is ever added.
            match p.state {
                MeasurementState::NotMeasuredToolchainAbsent => {}
            }
            if !p.raw_samples_ns.is_empty() {
                return Err(D::Error::custom(
                    "phase raw_samples_ns must be empty (nothing is measured)",
                ));
            }
            Ok(())
        };
        for r in &raw.results {
            if r.candidate != "sp1" && r.candidate != "risc0" {
                return Err(D::Error::custom("unknown candidate label"));
            }
            if !is_lower_hex64(&r.guest_program_id_hex) || !is_lower_hex64(&r.dep_lock_hash_hex) {
                return Err(D::Error::custom(
                    "result hex fields must be 64 lowercase hex",
                ));
            }
            check_phase(&r.cost.model_weight_auth)?;
            check_phase(&r.cost.transformer_unit)?;
            check_phase(&r.cost.state_manifest_hashing)?;
            check_phase(&r.cost.wrapping_groth16)?;
            check_phase(&r.cost.final_verify)?;
        }
        for h in &raw.artifact_hashes {
            if !is_lower_hex64(&h.hash_hex) {
                return Err(D::Error::custom(
                    "artifact hash_hex must be 64 lowercase hex",
                ));
            }
        }

        // Reconstruct the public value with FORCED invariants (phase costs are
        // rebuilt as unmeasured regardless of the accepted-but-empty input).
        Ok(BenchArtifact {
            schema_version: raw.schema_version,
            run_status: RunStatus::NonSelectionInvalidForB0,
            commit_sha: raw.commit_sha,
            b0_pre_spec_hash_hex: raw.b0_pre_spec_hash_hex,
            toolchain_container_digest: raw.toolchain_container_digest,
            hw_os: raw.hw_os,
            results: raw
                .results
                .into_iter()
                .map(|r| CandidateArchResult {
                    candidate: r.candidate,
                    arch: r.arch,
                    guest_program_id_hex: r.guest_program_id_hex,
                    dep_lock_hash_hex: r.dep_lock_hash_hex,
                    cost: CostBreakdown::unmeasured(),
                })
                .collect(),
            artifact_hashes: raw.artifact_hashes,
            notes: raw.notes,
        })
    }
}

/// Lower-case hex of a 32-byte digest.
pub fn hex32(b: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for byte in b {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

/// The canonical `"sp1"` / `"risc0"` label for a candidate.
fn candidate_label(c: Candidate) -> &'static str {
    match c {
        Candidate::Sp1 => "sp1",
        Candidate::Risc0 => "risc0",
    }
}

/// Build an unmeasured result for `candidate`×`arch`. The cost is forced
/// unmeasured; there is no way to attach timing samples.
pub fn unmeasured_result(
    candidate: Candidate,
    arch: &str,
    guest_program_id: [u8; 32],
    dep_lock_hash: [u8; 32],
) -> CandidateArchResult {
    CandidateArchResult {
        candidate: candidate_label(candidate).to_string(),
        arch: arch.to_string(),
        guest_program_id_hex: hex32(&guest_program_id),
        dep_lock_hash_hex: hex32(&dep_lock_hash),
        cost: CostBreakdown::unmeasured(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;

    fn artifact() -> BenchArtifact {
        let j = fixture::journal().unwrap();
        BenchArtifact::research(
            "working-tree",
            &j,
            HwOsMetadata {
                os: "unspecified".into(),
                arch: "unspecified".into(),
                cpu_model: "unspecified".into(),
                logical_cores: 0,
                total_memory_bytes: 0,
            },
            vec![
                unmeasured_result(
                    Candidate::Sp1,
                    "groth16",
                    fixture::program_id(Candidate::Sp1),
                    fixture::dep_lock(Candidate::Sp1),
                ),
                unmeasured_result(
                    Candidate::Risc0,
                    "groth16",
                    fixture::program_id(Candidate::Risc0),
                    fixture::dep_lock(Candidate::Risc0),
                ),
            ],
            vec![NamedArtifactHash {
                label: "reference_executor".into(),
                hash_hex: hex32(&[0xEE; 32]),
            }],
        )
    }

    #[test]
    fn run_is_forced_non_selection_and_unmeasured() {
        let a = artifact();
        assert_eq!(a.run_status(), RunStatus::NonSelectionInvalidForB0);
        assert_eq!(a.status_str(), NON_SELECTION_STATUS);
        assert!(is_zero_hex64(a.b0_pre_spec_hash_hex()));
        assert_eq!(a.toolchain_container_digest(), TOOLCHAIN_ABSENT_MARKER);
        for r in a.results() {
            for phase in r.cost().phases() {
                assert_eq!(phase.state(), MeasurementState::NotMeasuredToolchainAbsent);
                assert!(phase.raw_samples_ns().is_empty());
            }
        }
    }

    #[test]
    fn valid_research_json_roundtrips() {
        let a = artifact();
        let json = a.to_json().unwrap();
        assert!(json.contains("NON_SELECTION / INVALID_FOR_B0"));
        assert!(json.contains("NOT_MEASURED_TOOLCHAIN_ABSENT"));
        let back = BenchArtifact::from_json(&json).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn deserializing_non_empty_samples_fails() {
        let json = artifact().to_json().unwrap();
        let tampered = json.replacen("\"raw_samples_ns\": []", "\"raw_samples_ns\": [123]", 1);
        assert!(BenchArtifact::from_json(&tampered).is_err());
    }

    #[test]
    fn deserializing_forged_measured_state_fails() {
        let json = artifact().to_json().unwrap();
        let tampered = json.replacen("NOT_MEASURED_TOOLCHAIN_ABSENT", "MEASURED", 1);
        assert!(BenchArtifact::from_json(&tampered).is_err());
    }

    #[test]
    fn deserializing_nonzero_spec_hash_fails() {
        let json = artifact().to_json().unwrap();
        let zeros = "0".repeat(64);
        let nonzero = format!("{}1", "0".repeat(63));
        let tampered = json.replacen(&zeros, &nonzero, 1);
        assert!(BenchArtifact::from_json(&tampered).is_err());
    }

    #[test]
    fn deserializing_non_absent_toolchain_digest_fails() {
        let json = artifact().to_json().unwrap();
        let tampered = json.replacen(TOOLCHAIN_ABSENT_MARKER, "sha256:deadbeef", 1);
        assert!(BenchArtifact::from_json(&tampered).is_err());
    }

    #[test]
    fn deserializing_altered_status_fails() {
        let json = artifact().to_json().unwrap();
        let tampered = json.replacen(
            "\"run_status\": \"NON_SELECTION / INVALID_FOR_B0\"",
            "\"run_status\": \"SELECTION / ELIGIBLE_FOR_B0\"",
            1,
        );
        assert!(BenchArtifact::from_json(&tampered).is_err());
    }

    #[test]
    fn deserializing_omitted_status_fails() {
        let json = artifact().to_json().unwrap();
        let tampered = json.replacen(
            "  \"run_status\": \"NON_SELECTION / INVALID_FOR_B0\",\n",
            "",
            1,
        );
        assert!(BenchArtifact::from_json(&tampered).is_err());
    }

    #[test]
    fn deserializing_unknown_field_fails() {
        let json = artifact().to_json().unwrap();
        let tampered = json.replacen("{\n", "{\n  \"malicious\": true,\n", 1);
        assert!(BenchArtifact::from_json(&tampered).is_err());
    }
}
