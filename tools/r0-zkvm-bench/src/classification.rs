//! Internal, **non-wire** run classification that structurally forbids selection.
//!
//! This module encodes the crate's central safety property: nothing it produces
//! can ever be mistaken for a B0-selection-valid run. Two independent mechanisms
//! enforce it, neither of which depends on a zero `b0_pre_spec_hash`:
//!
//! 1. [`RunClass`] is an enum whose **only** variant is
//!    [`RunClass::NonSelectionResearch`]. There is deliberately no
//!    `Final` / `Official` / `Selection` / `Eligible` variant, so no value of the
//!    type can represent a selection-valid run, and no `match` can route to a
//!    selection branch (none exists to write).
//! 2. Every emitted artifact carries the exact visible status string
//!    [`NON_SELECTION_STATUS`], stamped by a forcing constructor with no public
//!    setter (see [`crate::bench::BenchArtifact`]).

/// The exact, frozen, human-visible status string stamped into every emitted
/// artifact. Its bytes are part of the crate's contract; tests assert it appears
/// verbatim in serialized output and cannot be altered or omitted.
pub const NON_SELECTION_STATUS: &str = "NON_SELECTION / INVALID_FOR_B0";

/// Internal run classification (never serialized to any B0 wire structure).
///
/// The type has a single non-selection variant by construction. Because Rust
/// enums are closed, a caller cannot introduce a selection-valid value, and this
/// crate contains no code that constructs, matches, or returns any other kind of
/// run — there is no "official"/"final"/"selection"/"eligible" mode anywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunClass {
    /// A host-side research run with **no** zkVM toolchain, **no** proof, and
    /// **no** measurement. Never selection-grade.
    NonSelectionResearch,
}

impl RunClass {
    /// The only classification this crate can produce.
    pub const fn research() -> Self {
        RunClass::NonSelectionResearch
    }

    /// The frozen visible status string. Identical for every variant — and there
    /// is only the one non-selection variant.
    pub const fn status(self) -> &'static str {
        match self {
            RunClass::NonSelectionResearch => NON_SELECTION_STATUS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_classification_is_non_selection() {
        assert_eq!(RunClass::research(), RunClass::NonSelectionResearch);
        assert_eq!(RunClass::research().status(), NON_SELECTION_STATUS);
        assert_eq!(NON_SELECTION_STATUS, "NON_SELECTION / INVALID_FOR_B0");
    }
}
