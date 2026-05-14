//! Stage 7a — status mapping unit tests.
//!
//! Cover every chain status variant (`submitted`, `included`,
//! `finalized`, `failed`, `unknown`) plus the negative cases that pin
//! Stage 5.1's contracts: `Unknown` maps to the non-terminal local
//! variant; `failed` without `reason` is a typed error; unknown
//! variants are rejected (not silently masked as `Unknown`).

use omni_sumchain::{map_status_info, InferenceAttestationStatusInfo};
use omni_zkml::AttestationStatus;

fn status_info(status: &str, height: Option<u64>, reason: Option<&str>) -> InferenceAttestationStatusInfo {
    InferenceAttestationStatusInfo {
        status: status.to_string(),
        included_at_height: height,
        reason: reason.map(str::to_string),
    }
}

#[test]
fn map_status_info_submitted() {
    let info = status_info("submitted", None, None);
    assert_eq!(map_status_info(info).unwrap(), AttestationStatus::Submitted);
}

#[test]
fn map_status_info_included() {
    let info = status_info("included", Some(100), None);
    assert_eq!(map_status_info(info).unwrap(), AttestationStatus::Included);
}

#[test]
fn map_status_info_finalized() {
    let info = status_info("finalized", Some(150), None);
    assert_eq!(map_status_info(info).unwrap(), AttestationStatus::Finalized);
}

#[test]
fn map_status_info_failed_with_reason() {
    let info = status_info("failed", Some(120), Some("execution reverted"));
    assert_eq!(
        map_status_info(info).unwrap(),
        AttestationStatus::Failed {
            reason: "execution reverted".into()
        }
    );
}

#[test]
fn map_status_info_failed_without_reason_is_error() {
    let info = status_info("failed", None, None);
    let err = map_status_info(info).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("failed") && msg.contains("reason"),
        "error message should mention failed and reason, got: {msg}"
    );
}

#[test]
fn map_status_info_unknown_maps_to_non_terminal_variant() {
    let info = status_info("unknown", None, None);
    // Stage 5.1 contract: chain `Unknown` -> local `Unknown` (NOT
    // `Failed`, NOT auto-translated to local `Dropped`).
    assert_eq!(map_status_info(info).unwrap(), AttestationStatus::Unknown);
}

#[test]
fn map_status_info_unexpected_variant_is_error() {
    let info = status_info("pending", None, None);
    let err = map_status_info(info).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("pending") || msg.contains("unexpected"),
        "error should mention unexpected variant, got: {msg}"
    );
}

#[test]
fn map_status_info_uppercase_is_rejected() {
    // Chain emits lowercase; uppercase is treated as unexpected so a
    // protocol drift is surfaced loudly rather than silently masked.
    let info = status_info("Submitted", None, None);
    assert!(
        map_status_info(info).is_err(),
        "uppercase status must be rejected, not coerced"
    );
}

#[test]
fn map_status_info_empty_string_is_rejected() {
    let info = status_info("", None, None);
    assert!(map_status_info(info).is_err());
}

#[test]
fn map_status_info_chain_never_returns_dropped() {
    // Negative: the chain v1 enum has no `dropped` variant. If the
    // mapper ever silently accepted one, this test would fail.
    let info = status_info("dropped", None, None);
    let err = map_status_info(info).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("dropped") || msg.contains("unexpected"));
}
