//! Stage 12.6 — persistent libp2p mesh identity.
//!
//! Lets operators stabilize the node's libp2p `PeerId` across
//! restarts so Stage 12.5 peer advertisements remain valid for
//! their full ≤24h freshness window. The identity is an Ed25519
//! libp2p `Keypair` encoded in the libp2p-standard protobuf form
//! (interoperable with go-libp2p / py-libp2p / kubo identity
//! files); the swarm builder decodes once at construction time.
//!
//! ## Roles are disjoint
//!
//! The libp2p mesh transport identity is **NOT** a contributor
//! signing identity. Stage 12.0–12.5 uses separate Ed25519 seeds
//! for `ContributorSigner` / `CoordinatorSigner` / `DispatcherSigner`.
//! Operators should never reuse a contributor seed file as a
//! `--net-identity-file` and vice versa.
//!
//! ## Failure posture
//!
//! - Missing file → auto-create with `0o600` (Unix). Operators
//!   get a deterministic identity on first run without a separate
//!   `init` command.
//! - Existing file with malformed bytes → typed `IdentityError::Decode`.
//!   **No silent fallback** to a fresh identity. Operators rotating
//!   identity delete the file explicitly.
//! - Unix file mode permitting non-owner read → typed
//!   `IdentityError::PermissionsTooBroad`. Stops loading a key
//!   that may have leaked through a stray `chmod 644`.
//! - On non-Unix targets, the permission check is `#[cfg(unix)]`-gated
//!   and behavior degrades to "create with default perms". Same
//!   trade-off Stage 12.0's contributor-seed loader makes.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use libp2p::identity::{Keypair, DecodingError};
use thiserror::Error;

/// Typed identity errors. Distinct from generic transport errors so
/// CLI callers can surface them clearly to operators.
#[derive(Debug, Error)]
pub enum IdentityError {
    #[error("identity file io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    /// File exists but its bytes did NOT decode as a libp2p
    /// `Keypair` protobuf. **No silent fallback** to a fresh
    /// identity — fail loud so corruption is visible.
    #[error("identity file at {path} did not decode as libp2p Keypair protobuf: {source}")]
    Decode {
        path: PathBuf,
        #[source]
        source: DecodingError,
    },

    /// On Unix only: the file's mode permits group/other read.
    /// Refuse to load a key that may have leaked.
    #[cfg(unix)]
    #[error(
        "identity file at {path} mode {mode:#o} permits group/other access; expected 0o600"
    )]
    PermissionsTooBroad { path: PathBuf, mode: u32 },

    #[error("identity file path {path} is not a regular file")]
    NotARegularFile { path: PathBuf },

    /// `path` is a symlink. Stage 12.6 refuses to follow symlinks
    /// for identity files: a symlink could point at a file with
    /// different permissions / owner than the operator audited.
    /// Operators wanting to use a non-standard location should
    /// pass that location directly.
    #[error("identity file path {path} is a symlink; refused (Stage 12.6 rejects symlinks for key material)")]
    IsSymlink { path: PathBuf },

    #[error("failed to encode freshly-generated keypair to protobuf: {0}")]
    EncodeFresh(DecodingError),
}

/// Load an existing identity file, or create one with `0o600`
/// (Unix) containing a freshly-generated Ed25519 keypair in libp2p
/// protobuf encoding.
///
/// Returns the protobuf bytes — the caller stashes them in
/// [`omni_types::config::NetIdentity::KeypairProtobufBytes`] and
/// the swarm builder decodes once at `OmniNet::new`.
///
/// Failure modes (see [`IdentityError`]):
/// - Missing file → auto-create + return new bytes.
/// - Existing file is not a regular file → `NotARegularFile`.
/// - Existing file is too permissive on Unix → `PermissionsTooBroad`
///   (the file is **not** overwritten — operator must fix it).
/// - Existing file bytes don't decode → `Decode` (no fallback).
/// - I/O error reading or writing → `Io`.
pub fn load_or_create_keypair_file_bytes(
    path: &Path,
) -> Result<Vec<u8>, IdentityError> {
    // Use `symlink_metadata` so we see the symlink itself rather
    // than its target. Following the symlink would let an attacker
    // point us at a file with different perms / owner than what
    // the operator audited.
    let symlink_meta = match fs::symlink_metadata(path) {
        Ok(m) => Some(m),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(source) => {
            return Err(IdentityError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    if let Some(meta) = symlink_meta {
        // Symlinks rejected outright — even if the target is a
        // regular file with sane perms, we don't follow.
        if meta.file_type().is_symlink() {
            return Err(IdentityError::IsSymlink {
                path: path.to_path_buf(),
            });
        }
        if !meta.is_file() {
            return Err(IdentityError::NotARegularFile {
                path: path.to_path_buf(),
            });
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = meta.permissions().mode() & 0o777;
            if mode & 0o077 != 0 {
                return Err(IdentityError::PermissionsTooBroad {
                    path: path.to_path_buf(),
                    mode,
                });
            }
        }
        let bytes = fs::read(path).map_err(|source| IdentityError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        // Decode once to verify the file is valid. We discard the
        // decoded keypair and return the raw bytes — the swarm
        // builder will re-decode them. Round-tripping protobuf is
        // cheap.
        Keypair::from_protobuf_encoding(&bytes).map_err(|source| {
            IdentityError::Decode {
                path: path.to_path_buf(),
                source,
            }
        })?;
        return Ok(bytes);
    }

    // Missing — generate + write + return.
    let kp = Keypair::generate_ed25519();
    let bytes = kp
        .to_protobuf_encoding()
        .map_err(IdentityError::EncodeFresh)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|source| IdentityError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
    }
    write_with_owner_only_perms(path, &bytes)?;
    Ok(bytes)
}

/// Decode protobuf bytes into a libp2p `Keypair`. Called by
/// `OmniSwarm::build` when `NetConfig.identity ==
/// KeypairProtobufBytes(_)`.
pub fn decode_keypair_protobuf(
    bytes: &[u8],
) -> Result<Keypair, DecodingError> {
    Keypair::from_protobuf_encoding(bytes)
}

#[cfg(unix)]
fn write_with_owner_only_perms(path: &Path, bytes: &[u8]) -> Result<(), IdentityError> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)
        .map_err(|source| IdentityError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    f.write_all(bytes).map_err(|source| IdentityError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

#[cfg(not(unix))]
fn write_with_owner_only_perms(path: &Path, bytes: &[u8]) -> Result<(), IdentityError> {
    // Non-Unix targets: write with platform-default permissions.
    // Same posture Stage 12.0's contributor-seed loader takes.
    fs::write(path, bytes).map_err(|source| IdentityError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::PublicKey;

    fn tempdir() -> tempfile::TempDir {
        tempfile::TempDir::new().expect("tempdir")
    }

    fn peer_id_from_bytes(bytes: &[u8]) -> String {
        decode_keypair_protobuf(bytes)
            .expect("valid keypair")
            .public()
            .to_peer_id()
            .to_base58()
    }

    #[test]
    fn load_or_create_creates_file_when_missing() {
        let d = tempdir();
        let p = d.path().join("id.key");
        assert!(!p.exists());
        let bytes = load_or_create_keypair_file_bytes(&p).expect("create");
        assert!(p.is_file(), "identity file must be created");
        assert!(!bytes.is_empty(), "protobuf bytes must be non-empty");
        // Reload returns same bytes → same PeerId.
        let bytes2 = load_or_create_keypair_file_bytes(&p).expect("reload");
        assert_eq!(
            peer_id_from_bytes(&bytes),
            peer_id_from_bytes(&bytes2),
            "second load must produce the same PeerId"
        );
    }

    #[cfg(unix)]
    #[test]
    fn load_or_create_sets_0600_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let d = tempdir();
        let p = d.path().join("id.key");
        load_or_create_keypair_file_bytes(&p).expect("create");
        let mode = fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "fresh identity file must be 0o600");
    }

    #[cfg(unix)]
    #[test]
    fn load_or_create_refuses_world_readable_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let d = tempdir();
        let p = d.path().join("id.key");
        // Plant a real keypair file but with 0o644 perms.
        let kp = Keypair::generate_ed25519();
        let bytes = kp.to_protobuf_encoding().unwrap();
        fs::write(&p, &bytes).unwrap();
        let mut perm = fs::metadata(&p).unwrap().permissions();
        perm.set_mode(0o644);
        fs::set_permissions(&p, perm).unwrap();
        let err = load_or_create_keypair_file_bytes(&p).unwrap_err();
        assert!(
            matches!(err, IdentityError::PermissionsTooBroad { .. }),
            "expected PermissionsTooBroad, got {err:?}"
        );
        // File must NOT have been overwritten by the rejection path.
        let after = fs::read(&p).unwrap();
        assert_eq!(after, bytes, "rejected file must be untouched");
    }

    #[test]
    fn load_or_create_refuses_malformed_existing_file_no_fallback() {
        let d = tempdir();
        let p = d.path().join("id.key");
        // Write garbage that does NOT decode as a libp2p Keypair.
        fs::write(&p, b"this is not a libp2p Keypair protobuf").unwrap();
        // On Unix, set 0o600 so the perms gate doesn't fire first.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perm = fs::metadata(&p).unwrap().permissions();
            perm.set_mode(0o600);
            fs::set_permissions(&p, perm).unwrap();
        }
        let err = load_or_create_keypair_file_bytes(&p).unwrap_err();
        assert!(
            matches!(err, IdentityError::Decode { .. }),
            "expected Decode, got {err:?}"
        );
        // Bytes must NOT have been replaced by the rejection path.
        let after = fs::read(&p).unwrap();
        assert_eq!(after, b"this is not a libp2p Keypair protobuf");
    }

    #[test]
    fn round_trip_preserves_peer_id() {
        let d = tempdir();
        let p = d.path().join("id.key");
        let bytes1 = load_or_create_keypair_file_bytes(&p).expect("create");
        // Independently decode and verify PeerId matches.
        let kp = decode_keypair_protobuf(&bytes1).unwrap();
        let peer_id_a = kp.public().to_peer_id().to_base58();
        let bytes2 = load_or_create_keypair_file_bytes(&p).expect("reload");
        let peer_id_b = decode_keypair_protobuf(&bytes2)
            .unwrap()
            .public()
            .to_peer_id()
            .to_base58();
        assert_eq!(peer_id_a, peer_id_b);
        // The public-key roundtrip also recovers correctly.
        let _: PublicKey = decode_keypair_protobuf(&bytes2).unwrap().public();
    }

    #[test]
    fn two_distinct_files_yield_distinct_peer_ids() {
        let d = tempdir();
        let p1 = d.path().join("a.key");
        let p2 = d.path().join("b.key");
        let bytes1 = load_or_create_keypair_file_bytes(&p1).expect("a");
        let bytes2 = load_or_create_keypair_file_bytes(&p2).expect("b");
        assert_ne!(
            peer_id_from_bytes(&bytes1),
            peer_id_from_bytes(&bytes2),
            "two distinct identity files must produce distinct PeerIds"
        );
    }

    #[test]
    fn rejects_non_regular_file() {
        let d = tempdir();
        // A directory path passes existence but is not a file.
        let p = d.path().to_path_buf();
        let err = load_or_create_keypair_file_bytes(&p).unwrap_err();
        assert!(
            matches!(err, IdentityError::NotARegularFile { .. }),
            "expected NotARegularFile, got {err:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_even_if_target_is_clean() {
        // Stage 12.6 review fix #2: `fs::metadata` follows
        // symlinks, so the prior implementation accepted a
        // symlink to a regular 0o600 keypair file. Now we
        // refuse symlinks outright via `symlink_metadata`.
        use std::os::unix::fs::symlink;
        let d = tempdir();
        // Real, well-formed keypair at one path...
        let real = d.path().join("real.key");
        let bytes = load_or_create_keypair_file_bytes(&real)
            .expect("baseline create succeeds");
        let _ = bytes;
        // ...with a symlink to it at the path we ask the loader for.
        let link = d.path().join("link.key");
        symlink(&real, &link).unwrap();
        let err = load_or_create_keypair_file_bytes(&link).unwrap_err();
        assert!(
            matches!(err, IdentityError::IsSymlink { .. }),
            "expected IsSymlink, got {err:?}"
        );
    }
}

#[cfg(test)]
mod debug_redaction_tests {
    //! Stage 12.6 review fix #1: `NetIdentity` derived `Debug`,
    //! so formatting a `NetConfig` printed the raw libp2p private
    //! keypair bytes. The manual `Debug` impl in
    //! `omni_types::config` redacts them — verified here from
    //! omni-net's vantage so the regression test lives next to
    //! the surface that actually constructs `NetIdentity` values.

    use omni_types::config::{NetConfig, NetIdentity};

    #[test]
    fn debug_redacts_protobuf_bytes() {
        let bytes = b"a hypothetical libp2p secret key payload".to_vec();
        let id = NetIdentity::KeypairProtobufBytes(bytes.clone());
        let s = format!("{:?}", id);
        assert!(
            s.contains("<redacted>") && s.contains("len:"),
            "expected redacted Debug output, got: {s}"
        );
        // Hard guard: the raw bytes must not appear in any form
        // a casual `eprintln!("{:?}", net_config)` could leak.
        assert!(
            !s.contains("payload") && !s.contains("hypothetical"),
            "raw byte payload leaked through Debug: {s}"
        );
    }

    #[test]
    fn debug_on_netconfig_does_not_leak_identity_bytes() {
        let bytes = b"another secret payload that must not leak".to_vec();
        let cfg = NetConfig {
            identity: NetIdentity::KeypairProtobufBytes(bytes),
            ..NetConfig::default()
        };
        let s = format!("{:?}", cfg);
        assert!(
            !s.contains("secret payload") && !s.contains("must not leak"),
            "NetConfig Debug leaked identity bytes: {s}"
        );
        assert!(
            s.contains("<redacted>"),
            "NetConfig Debug must surface the redaction marker: {s}"
        );
    }

    #[test]
    fn debug_ephemeral_is_unit_variant_tag_only() {
        let s = format!("{:?}", NetIdentity::Ephemeral);
        assert_eq!(s, "Ephemeral");
    }
}
