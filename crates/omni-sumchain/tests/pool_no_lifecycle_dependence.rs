//! #106: no `getFileInfoV2.lifecycle` dependence.
//!
//! #106 forbids treating SNIP activation as retrieval: an input counts only
//! when its retrieved bytes recompute to the committed roots. These tests hold
//! the `pool-client` surface to that:
//!
//! * **Behaviour.** Every pool-client decision (the unit and offer reads, the
//!   accept path, the one-active-offer precheck and root-checked retrieval) is
//!   run against a transport that also serves `storage_getFileInfoV2` with a
//!   lifecycle that is Pending, Active or Abandoned, absent, `null`, out of
//!   range, of the wrong type, or contradicting its own heights. Every decision
//!   must equal the one made when the server has no such method at all, and
//!   the client must never ask for it.
//! * **Source.** A narrow scan of `src/pool/` for the names through which a
//!   lifecycle could enter.
//!
//! The pool reads are the recorded sum-chain server output in
//! `fixtures/pool_reads/` (see `PROVENANCE.md` there). There is no recorded
//! `storage_getFileInfoV2` answer, and a server never produces the absent,
//! `null` or contradictory cases, so those bodies are built here in the wire
//! shape of sum-chain's `StorageFileInfoV2` (`crates/rpc/src/types.rs`,
//! main 2c275242), whose `lifecycle` is `0` = Pending, `1` = Active,
//! `2` = Abandoned.
#![cfg(feature = "pool-client")]

use omni_sumchain::pool::accept::{build_accept, ensure_no_active_offer};
use omni_sumchain::pool::ids::OwnOffer;
use omni_sumchain::pool::manifest::{layer_group_output, verify_retrieved_output};
use omni_sumchain::pool::reads::{METHOD_GET_OFFER, METHOD_GET_UNIT};
use omni_sumchain::{FakeJsonRpcTransport, SumChainClient};

const SEED: [u8; 32] = [42u8; 32];
const GET_FILE_INFO_V2: &str = "storage_getFileInfoV2";

fn id(b: u8) -> [u8; 32] {
    [b; 32]
}

/// The `result` of a recorded pool read.
fn recorded_result(name: &str) -> serde_json::Value {
    let path = format!(
        "{}/tests/fixtures/pool_reads/{name}.response.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let body = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    v["result"].clone()
}

/// A `storage_getFileInfoV2` answer for an Active public file.
fn file_info_active() -> serde_json::Value {
    serde_json::json!({
        "merkle_root": format!("0x{}", "ab".repeat(32)),
        "owner": "8KfGeEPJUAa16cX1Sq95Gvra4Va4M3o2a",
        "plaintext_size_bytes": 96,
        "stored_size_bytes": 96,
        "chunk_count": 1,
        "fee_pool": 1000,
        "created_at": 100,
        "activated_at_height": 120,
        "abandoned_at_height": null,
        "assignment_height": 100,
        "visibility": 0,
        "lifecycle": 1,
        "access_list": [],
        "access_total": 0,
        "access_offset": 0,
        "predecessor_root": null
    })
}

/// Every lifecycle a server could send, and several it never would.
fn lifecycle_variants() -> Vec<(&'static str, serde_json::Value)> {
    let with = |f: &dyn Fn(&mut serde_json::Map<String, serde_json::Value>)| {
        let mut v = file_info_active();
        f(v.as_object_mut().unwrap());
        v
    };
    vec![
        ("active", file_info_active()),
        (
            "pending",
            with(&|m| {
                m.insert("lifecycle".into(), 0.into());
                m.insert("activated_at_height".into(), serde_json::Value::Null);
            }),
        ),
        (
            "abandoned",
            with(&|m| {
                m.insert("lifecycle".into(), 2.into());
                m.insert("activated_at_height".into(), serde_json::Value::Null);
                m.insert("abandoned_at_height".into(), 150.into());
            }),
        ),
        (
            "absent",
            with(&|m| {
                m.remove("lifecycle");
            }),
        ),
        (
            "null",
            with(&|m| {
                m.insert("lifecycle".into(), serde_json::Value::Null);
            }),
        ),
        (
            "out of range",
            with(&|m| {
                m.insert("lifecycle".into(), 7.into());
            }),
        ),
        (
            "wrong type",
            with(&|m| {
                m.insert("lifecycle".into(), "Active".into());
            }),
        ),
        (
            "active but never activated, and abandoned",
            with(&|m| {
                m.insert("activated_at_height".into(), serde_json::Value::Null);
                m.insert("abandoned_at_height".into(), 150.into());
            }),
        ),
        (
            "pending but activated",
            with(&|m| {
                m.insert("lifecycle".into(), 0.into());
            }),
        ),
        ("not registered", serde_json::Value::Null),
    ]
}

/// Every pool-client decision, as comparable text.
#[derive(Debug, PartialEq, Eq)]
struct Decisions {
    unit: String,
    offer: String,
    accept: String,
    over_capacity: String,
    publish_precheck: String,
    honest_retrieval: String,
    tampered_retrieval: String,
    methods_called: Vec<String>,
}

/// Run every pool-client decision against `fake`, with `file_info` (if any)
/// served for `storage_getFileInfoV2`.
fn decide(file_info: Option<serde_json::Value>) -> Decisions {
    let fake = FakeJsonRpcTransport::new();
    fake.set_response(METHOD_GET_UNIT, Ok(recorded_result("unit_assigned")));
    fake.set_response(
        METHOD_GET_OFFER,
        Ok(recorded_result("offer_active_reserved")),
    );
    if let Some(info) = file_info {
        fake.set_response(GET_FILE_INFO_V2, Ok(info));
    }
    let client = SumChainClient::with_transport(SEED, fake.clone());

    let unit = client.pool_get_unit(&id(1), &id(3));
    let offer = client.pool_get_offer(&id(3));
    let (mut u, mut o) = (
        unit.clone().unwrap().expect("held"),
        offer.clone().unwrap().expect("held"),
    );
    // The recorded offer predates the ratified offer_bond_id derivation (G1-1).
    // Rebind it, and the unit's references to it, to the derived id of its own
    // recorded identity and offer_seq, as pool_reads_chain_fixtures does.
    let own = OwnOffer {
        chain_id: 1,
        identity: o.identity,
        offer_seq: o.offer_seq,
    };
    let (old, new) = (o.offer_bond_id, own.offer_bond_id());
    o.offer_bond_id = new;
    if let Some(a) = u
        .assignment
        .as_mut()
        .filter(|a| a.winner_offer_bond_id == old)
    {
        a.winner_offer_bond_id = new;
    }
    if let Some(l) = u.accepted_leaf.as_mut().filter(|l| l.offer_bond_id == old) {
        l.offer_bond_id = new;
    }
    let accept = build_accept(Some(&u), Some(&o), &own, [0x44; 32], 840);
    let over_capacity = build_accept(Some(&u), Some(&o), &own, [0x44; 32], 841);
    let publish_precheck = ensure_no_active_offer(&o.identity, Some(&o));

    let out = layer_group_output(5, &[1, -2, 3, -4, 5, -6, 7, -8], &[([1; 8], [-1; 8])]).unwrap();
    let honest =
        verify_retrieved_output(&out.manifest_commitment, &out.manifest_bytes, &out.objects);
    let mut tampered = out.objects.clone();
    tampered[0].2[0] ^= 0x01;
    let tampered =
        verify_retrieved_output(&out.manifest_commitment, &out.manifest_bytes, &tampered);

    Decisions {
        unit: format!("{unit:?}"),
        offer: format!("{offer:?}"),
        accept: format!("{accept:?}"),
        over_capacity: format!("{over_capacity:?}"),
        publish_precheck: format!("{publish_precheck:?}"),
        honest_retrieval: format!("{honest:?}"),
        tampered_retrieval: format!("{tampered:?}"),
        methods_called: fake.calls().into_iter().map(|(m, _)| m).collect(),
    }
}

#[test]
fn pool_client_decisions_do_not_depend_on_any_file_lifecycle() {
    // The server has no `storage_getFileInfoV2` at all: a call would fail.
    let baseline = decide(None);
    assert!(baseline.accept.starts_with("Ok("), "{}", baseline.accept);
    assert!(baseline.over_capacity.contains("CapacityExceeded"));
    assert!(baseline.honest_retrieval.starts_with("Ok("));
    assert!(baseline.tampered_retrieval.contains("SlotMismatch"));
    assert_eq!(
        baseline.methods_called,
        vec![METHOD_GET_UNIT, METHOD_GET_OFFER]
    );

    for (name, info) in lifecycle_variants() {
        assert_eq!(decide(Some(info)), baseline, "lifecycle {name}");
    }
}

#[test]
fn the_pool_client_never_asks_for_file_info() {
    for (name, info) in lifecycle_variants() {
        let calls = decide(Some(info)).methods_called;
        assert!(
            calls.iter().all(|m| !m.starts_with("storage_")),
            "lifecycle {name}: called {calls:?}"
        );
    }
}

#[test]
fn activation_is_not_retrieval() {
    // An Active file whose retrieved bytes do not recompute is refused, and a
    // Pending, Abandoned or unregistered one whose bytes do recompute verifies.
    for (name, info) in lifecycle_variants() {
        let d = decide(Some(info));
        assert!(
            d.tampered_retrieval.contains("SlotMismatch"),
            "lifecycle {name}"
        );
        assert!(d.honest_retrieval.starts_with("Ok("), "lifecycle {name}");
    }
}

/// Names through which a SNIP file lifecycle could reach pool-client code.
/// Comments are skipped: the module docs explain why none is used.
const FORBIDDEN: [&str; 7] = [
    "getFileInfo",
    "storage_",
    "lifecycle",
    "SnipV2Lifecycle",
    "activated_at_height",
    "abandoned_at_height",
    "phase5",
];

#[test]
fn pool_client_source_names_no_file_lifecycle() {
    let dir = format!("{}/src/pool", env!("CARGO_MANIFEST_DIR"));
    let mut scanned = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap();
        for (n, line) in src.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            for name in FORBIDDEN {
                assert!(
                    !code.contains(name),
                    "{}:{}: `{name}` in `{}`",
                    path.display(),
                    n + 1,
                    line.trim()
                );
            }
        }
        scanned += 1;
    }
    assert!(scanned >= 6, "scanned only {scanned} files in {dir}");
}
