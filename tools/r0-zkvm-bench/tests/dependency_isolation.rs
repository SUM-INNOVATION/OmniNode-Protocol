//! Dependency-isolation assertion (OmniNode #101, criterion 11).
//!
//! No production crate may depend on this research crate, and this crate must
//! not be a workspace member. This test reads the workspace manifests directly
//! (no `cargo` invocation required) so it runs under plain `cargo test`. A
//! companion `scripts/check_no_prod_dep.sh` performs the same check via
//! `cargo metadata` for CI.

use std::path::{Path, PathBuf};

/// This crate's own name, in both hyphen and underscore spellings.
const CRATE_HYPHEN: &str = "r0-zkvm-bench";
const CRATE_UNDERSCORE: &str = "r0_zkvm_bench";

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/tools/r0-zkvm-bench
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("repo root is two levels above the crate manifest")
        .to_path_buf()
}

#[test]
fn crate_lives_under_excluded_tools_dir() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    assert!(
        manifest_dir.ends_with("tools/r0-zkvm-bench"),
        "crate must live under tools/: {manifest_dir:?}"
    );
}

#[test]
fn workspace_excludes_tools_and_does_not_list_this_crate() {
    let root_manifest = repo_root().join("Cargo.toml");
    let text = std::fs::read_to_string(&root_manifest)
        .unwrap_or_else(|e| panic!("read {root_manifest:?}: {e}"));
    // tools/ is excluded from the workspace.
    assert!(
        text.contains(r#"exclude = ["tools"]"#),
        "root Cargo.toml must exclude tools/"
    );
    // This crate is not a workspace member.
    assert!(
        !text.contains(CRATE_HYPHEN),
        "root Cargo.toml must not reference {CRATE_HYPHEN}"
    );
}

#[test]
fn no_production_crate_depends_on_this_crate() {
    let crates_dir = repo_root().join("crates");
    let entries =
        std::fs::read_dir(&crates_dir).unwrap_or_else(|e| panic!("read_dir {crates_dir:?}: {e}"));

    let mut checked = 0usize;
    for entry in entries {
        let entry = entry.expect("dir entry");
        let manifest = entry.path().join("Cargo.toml");
        if !manifest.is_file() {
            continue;
        }
        let text =
            std::fs::read_to_string(&manifest).unwrap_or_else(|e| panic!("read {manifest:?}: {e}"));
        assert!(
            !text.contains(CRATE_HYPHEN) && !text.contains(CRATE_UNDERSCORE),
            "production crate {manifest:?} must not depend on the R0 research crate"
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "expected to check at least one production crate"
    );
}

#[test]
fn crate_exact_pins_sumchain_wire_from_cratesio() {
    // The B0 wire types are adopted via an EXACT crates.io pin — never a path or
    // git source.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let text =
        std::fs::read_to_string(&manifest).unwrap_or_else(|e| panic!("read {manifest:?}: {e}"));
    assert!(
        text.contains("sumchain-wire = \"=0.2.1\""),
        "Cargo.toml must exact-pin sumchain-wire = \"=0.2.1\""
    );
    assert!(
        !text.contains("path =") && !text.contains("git ="),
        "no path/git dependency is permitted in this crate"
    );
    assert!(text.contains("publish     = false") || text.contains("publish = false"));
}

#[test]
fn crate_lock_exact_pins_sumchain_wire_0_2_1() {
    let lock = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    let text = std::fs::read_to_string(&lock).unwrap_or_else(|e| panic!("read {lock:?}: {e}"));
    assert!(
        text.contains("name = \"sumchain-wire\"") && text.contains("version = \"0.2.1\""),
        "Cargo.lock must pin sumchain-wire 0.2.1"
    );
}
