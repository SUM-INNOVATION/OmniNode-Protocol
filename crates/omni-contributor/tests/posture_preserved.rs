//! Stage 12.0 — posture preservation tests.
//!
//! Restates the Stage 11d.3 reframe invariant: the chain remains
//! neutral; proof acceptance is a local verifier policy decision; no
//! chain-side allowlist exists. Stage 12.0 does not change any of
//! this — these tests guard against accidental drift.

/// The literal `i` + `p` + `f` + `s` would be flagged by this very
/// test if it appeared in the source. Build the needles at runtime
/// from char arrays so the test source is itself clean.
fn needles() -> [String; 3] {
    let lower: String = ['i', 'p', 'f', 's'].into_iter().collect();
    let upper: String = ['I', 'P', 'F', 'S'].into_iter().collect();
    let title: String = ['I', 'p', 'f', 's'].into_iter().collect();
    [lower, upper, title]
}

#[test]
fn snip_replaces_legacy_p2p_terminology_in_omni_contributor_source() {
    // SNIP replaces the legacy decentralized-storage term (4 chars,
    // built at runtime in `needles()`). The contributor crate's
    // source / fixtures / protocol doc must not contain the string
    // anywhere.
    let crate_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut offending = Vec::new();
    walk(&crate_dir.join("src"), &mut offending);
    walk(&crate_dir.join("tests"), &mut offending);
    // Also walk the Cargo.toml and the protocol doc if present.
    check_file(&crate_dir.join("Cargo.toml"), &mut offending);
    let proto_doc = crate_dir
        .parent()
        .and_then(std::path::Path::parent)
        .map(|repo| repo.join("docs/stage12-contributor-protocol.md"));
    if let Some(p) = proto_doc {
        if p.is_file() {
            check_file(&p, &mut offending);
        }
    }
    assert!(
        offending.is_empty(),
        "Stage 12.0 invariant violated: legacy term found in:\n  {}",
        offending.join("\n  "),
    );
}

fn walk(dir: &std::path::Path, offending: &mut Vec<String>) {
    if !dir.is_dir() {
        return;
    }
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            // Skip the test fixtures dir's HashMap entries etc.
            walk(&path, offending);
        } else {
            check_file(&path, offending);
        }
    }
}

fn check_file(path: &std::path::Path, offending: &mut Vec<String>) {
    // Skip binary files; only scan UTF-8 text files.
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    let n = needles();
    for needle in &n {
        if text.contains(needle.as_str()) {
            offending.push(format!("{}: contains {:?}", path.display(), needle));
            break;
        }
    }
}
