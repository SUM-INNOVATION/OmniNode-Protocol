//! O5 (issue #103) — extraction-boundary regression pins.
//!
//! Guards the invariants of extracting the seven operator observability
//! modules (`status`, `repair`, `resume`, `integrity`, `archive`,
//! `restore`, `cleanup`) out of `omni-contributor` into `omni-ops`:
//!
//!   1. The seven schema-version constants keep their pre-extraction
//!      values (a change is a persisted-report-format break, not a
//!      refactor). Values captured from `omni-contributor` @ `09763d5`.
//!   2. ONE-DIRECTION cross-crate smoke: state written through the
//!      RETAINED `omni-contributor` substrate (`ContributorStateStore`)
//!      is read by an `omni-ops` report generator (`omni-node →
//!      omni-ops → omni-contributor`); the report serializes across the
//!      boundary carrying `schema_version`. This is a
//!      serialization/schema-presence check — NOT a byte- or
//!      hash-stability guarantee, and NOT a reverse-direction test.
//!
//! The stronger, separate guarantees are proven by the moved
//! integration tests (which run unchanged after the move):
//!   - serialization stability: `report_json_roundtrip` (status_report),
//!     `plan_json_roundtrip_preserves_hash` (state_cleanup_plan),
//!     `repair_plan_hash_round_trips_through_json` (repair_plan).
//!   - plan/report hash stability + drift:
//!     `plan_hash_is_self_consistent_and_drift_aware` (state_cleanup_plan),
//!     `source_status_hash_*` + `apply_path_detects_plan_hash_drift`
//!     (repair_plan).
//!   - the REVERSE direction — omni-ops-written artifacts read back by
//!     their consumers through the shared store:
//!     `restore_round_trips_bytes_byte_for_byte` and
//!     `restored_session_is_accepted_by_load_verified_restart_snapshot`
//!     (restore_session_archive),
//!     `tampered_join_cleanup_then_restore_round_trips`
//!     (state_cleanup_quarantine_restore).

use omni_contributor::state::{ContributorStateStore, StateNamespace};
use omni_ops::{
    scan_state_integrity, ScanOptions, ARCHIVE_MANIFEST_SCHEMA_VERSION,
    CLEANUP_PLAN_SCHEMA_VERSION, QUARANTINE_MANIFEST_SCHEMA_VERSION,
    REPAIR_PLAN_SCHEMA_VERSION, STATE_INTEGRITY_DIFF_SCHEMA_VERSION,
    STATE_INTEGRITY_REPORT_SCHEMA_VERSION, STATUS_SCHEMA_VERSION,
};

const NOW_UTC: &str = "2026-06-01T00:00:00Z";

#[test]
fn schema_version_constants_are_frozen_across_extraction() {
    assert_eq!(STATUS_SCHEMA_VERSION, 3);
    assert_eq!(REPAIR_PLAN_SCHEMA_VERSION, 2);
    assert_eq!(ARCHIVE_MANIFEST_SCHEMA_VERSION, 1);
    assert_eq!(STATE_INTEGRITY_REPORT_SCHEMA_VERSION, 1);
    assert_eq!(STATE_INTEGRITY_DIFF_SCHEMA_VERSION, 1);
    assert_eq!(CLEANUP_PLAN_SCHEMA_VERSION, 1);
    assert_eq!(QUARANTINE_MANIFEST_SCHEMA_VERSION, 1);
}

#[test]
fn state_written_via_retained_substrate_is_read_by_omni_ops() {
    // ONE direction: write through the RETAINED omni-contributor store,
    // then produce an omni-ops report over it. This proves the
    // omni-contributor-writes → omni-ops-reads path across the crate
    // boundary. (The reverse path — omni-ops writes read back by
    // consumers — is covered by the restore/cleanup round-trip tests
    // named in the module doc above.)
    let dir = tempfile::TempDir::new().expect("tempdir");
    let (store, _) = ContributorStateStore::open(dir.path(), false, NOW_UTC)
        .expect("open retained store");
    let phantom_session = "aa".repeat(32);
    store
        .mark_seen(StateNamespace::Sessions, &phantom_session)
        .expect("write state via retained substrate");

    let report = scan_state_integrity(
        &store,
        &ScanOptions {
            session_id_filter: None,
            archive_dir: None,
            now_utc: NOW_UTC,
        },
    )
    .expect("omni-ops scan over retained store");

    // The report serializes across the crate boundary and carries the
    // frozen report schema_version. This is a serialization + schema
    // presence check (NOT byte/hash stability — see the module doc).
    let json = serde_json::to_string(&report).expect("serialize report");
    assert!(
        json.contains("\"schema_version\":1"),
        "omni-ops report must serialize with schema_version=1; got: {json}"
    );
}
