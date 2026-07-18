//! API-surface boundary (OmniNode #101): prove that NO public path through
//! `r0-zkvm-bench` exposes B0 final-statement materialization, the raw `Writer` /
//! `Reader`, or the complete B0 statement module.
//!
//! Two mechanisms:
//!  * a compile-time positive check that the focused, SAFE adopted types are
//!    reachable through the public modules;
//!  * a source-surface (grep-style) check that the crate re-exports `b0`
//!    crate-privately and never publicly re-exports or calls the unsafe
//!    finalization/raw-writer surface.

use std::path::PathBuf;

// Positive: the focused safe surface is public and nameable. (If any of these
// were accidentally hidden, this test file would fail to compile.)
#[allow(unused_imports)]
use r0_zkvm_bench::envelope::{
    allowlist_membership, shared_binding_ok, Candidate, MembershipError, PartialComputeProofV1,
    ProductionProofEnvelopeV1,
};
#[allow(unused_imports)]
use r0_zkvm_bench::manifest::{InputManifestV1, OutputManifestV1, SlotDescriptorV1};
#[allow(unused_imports)]
use r0_zkvm_bench::object::{ObjectCommitmentV1, ObjectKind};
// The raw statement type is deliberately NOT imported here — it is not public.
#[allow(unused_imports)]
use r0_zkvm_bench::statement::{DerivedInputV1, SyntheticJournal, UnitKind};

fn src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn src_files() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(src_dir()).expect("read src/") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let text =
                std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));
            out.push((name, text));
        }
    }
    assert!(out.len() >= 10, "expected the full src module set");
    out
}

/// Trimmed non-comment code lines (ignores `//` and `//!` prose).
fn code_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.starts_with("//") && !l.starts_with('*'))
}

#[test]
fn b0_namespace_is_crate_private_not_public() {
    let lib = std::fs::read_to_string(src_dir().join("lib.rs")).unwrap();
    assert!(
        lib.contains("pub(crate) use sumchain_wire::b0;"),
        "lib.rs must re-export the b0 namespace crate-privately"
    );
    for line in code_lines(&lib) {
        assert!(
            line != "pub use sumchain_wire::b0;",
            "lib.rs must NOT publicly re-export the whole b0 namespace"
        );
    }
}

#[test]
fn no_public_reexport_of_finalization_or_raw_writer() {
    for (name, text) in src_files() {
        for line in code_lines(&text) {
            if line.starts_with("pub use") {
                for forbidden in [
                    "materialize_final",
                    "template_bytes",
                    "template_hash",
                    "Writer",
                    "Reader",
                    "::statement::{", // no glob re-export of the whole statement module
                ] {
                    assert!(
                        !line.contains(forbidden),
                        "{name}: public re-export must not expose `{forbidden}`: {line}"
                    );
                }
            }
        }
    }
}

#[test]
fn no_code_path_calls_materialize_final() {
    for (name, text) in src_files() {
        for line in code_lines(&text) {
            assert!(
                !line.contains("materialize_final("),
                "{name}: no code path may call materialize_final: {line}"
            );
        }
    }
}

#[test]
fn no_public_signature_mentions_raw_statement() {
    // Any `pub ` item (fn / use / struct / const / type / trait) — but NOT
    // `pub(crate)` — must not mention the raw statement type in name or signature.
    for (name, text) in src_files() {
        for line in code_lines(&text) {
            if line.starts_with("pub ") {
                assert!(
                    !line.contains("R0ComputationStatementV2"),
                    "{name}: public API must not mention R0ComputationStatementV2: {line}"
                );
            }
        }
    }
}
