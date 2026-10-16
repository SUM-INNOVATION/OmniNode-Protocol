//! `pool-client` reads and accept path against REAL sum-chain server output.
//!
//! `tests/fixtures/pool_reads/` holds the verbatim HTTP response bodies of
//! sum-chain's `pool_getUnit` / `pool_getOffer` server at `90c49da1`, for pool
//! state built with the chain's real model and store operations (see
//! `PROVENANCE.md` there). Each test serves one recorded body over a real
//! socket to the production `UreqTransport`, through `SumChainClient`, and,
//! where the client chose the request, requires the request it sent to equal
//! the one the server answered. Malformed-response tests mutate a real body.
//!
//! `live_server_*` runs the same reads against a live sum-chain RPC server
//! (`SUMCHAIN_POOL_RPC_URL`); it is ignored by default.
#![cfg(feature = "pool-client")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread::JoinHandle;

use omni_sumchain::pool::accept::{
    build_accept, ensure_no_active_offer, AcceptRefusal, PublishRefusal,
};
use omni_sumchain::pool::reads::{PoolOffer, PoolReadError, PoolUnit, Reservation, UnitState};
use omni_sumchain::{SumChainClient, UreqTransport};
use sumchain_wire::address::Address;
use sumchain_wire::compute_pool_wire::{AcceptWorkUnitV1, ComputePoolOperation, WorkItemRef};

const SEED: [u8; 32] = [42u8; 32];

fn fixture(name: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/pool_reads/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Serve `body` once over HTTP/1.1 on 127.0.0.1 and hand back the JSON-RPC
/// request the client sent.
fn serve_once(body: String) -> (String, JoinHandle<serde_json::Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        let (header_end, content_length) = loop {
            let n = stream.read(&mut chunk).unwrap();
            assert!(n > 0, "client closed before sending a request");
            buf.extend_from_slice(&chunk[..n]);
            if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..i]).to_ascii_lowercase();
                let len = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .map(|v| v.trim().parse::<usize>().unwrap())
                    .expect("a content-length header");
                break (i + 4, len);
            }
        };
        while buf.len() < header_end + content_length {
            let n = stream.read(&mut chunk).unwrap();
            assert!(n > 0, "client closed mid-body");
            buf.extend_from_slice(&chunk[..n]);
        }
        let request: serde_json::Value =
            serde_json::from_slice(&buf[header_end..header_end + content_length]).unwrap();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
        request
    });
    (url, handle)
}

fn client(url: String) -> SumChainClient<UreqTransport> {
    SumChainClient::with_transport(SEED, UreqTransport::new(url))
}

/// The client's request must be the one the server answered: same method and
/// parameters (the JSON-RPC id is the transport's).
fn assert_same_request(sent: &serde_json::Value, recorded: &str) {
    let recorded: serde_json::Value = serde_json::from_str(recorded).unwrap();
    assert_eq!(sent["jsonrpc"], "2.0");
    assert_eq!(sent["method"], recorded["method"], "method");
    assert_eq!(sent["params"], recorded["params"], "params");
}

fn id(b: u8) -> [u8; 32] {
    [b; 32]
}

fn read_unit(name: &str, job: u8, unit: u8) -> Result<Option<PoolUnit>, PoolReadError> {
    let (url, server) = serve_once(fixture(&format!("{name}.response.json")));
    let got = client(url).pool_get_unit(&id(job), &id(unit));
    let sent = server.join().unwrap();
    assert_same_request(&sent, &fixture(&format!("{name}.request.json")));
    got
}

fn read_offer(name: &str, offer: u8) -> Result<Option<PoolOffer>, PoolReadError> {
    let (url, server) = serve_once(fixture(&format!("{name}.response.json")));
    let got = client(url).pool_get_offer(&id(offer));
    let sent = server.join().unwrap();
    assert_same_request(&sent, &fixture(&format!("{name}.request.json")));
    got
}

/// Serve a (mutated) body for a unit read, without the request check.
fn unit_from_body(body: String, job: u8, unit: u8) -> Result<Option<PoolUnit>, PoolReadError> {
    let (url, server) = serve_once(body);
    let got = client(url).pool_get_unit(&id(job), &id(unit));
    server.join().unwrap();
    got
}

fn offer_from_body(body: String, offer: u8) -> Result<Option<PoolOffer>, PoolReadError> {
    let (url, server) = serve_once(body);
    let got = client(url).pool_get_offer(&id(offer));
    server.join().unwrap();
    got
}

fn addr(b: u8) -> Address {
    Address::new([b; 20])
}

// ── Decoding real answers ─────────────────────────────────────────────────────

#[test]
fn an_assigned_unit_decodes_to_its_work_item_and_assignment() {
    let u = read_unit("unit_assigned", 1, 3).unwrap().expect("held");
    assert_eq!(u.job_id, id(1));
    assert_eq!(u.unit_id, id(3));
    assert_eq!(u.generation, 0);
    assert_eq!(u.state, UnitState::Assigned);
    assert!(u.predecessors.is_empty() && u.required_inputs.is_empty());
    let a = u.assignment.as_ref().expect("assignment");
    assert_eq!(a.winner_offer_bond_id, id(3));
    assert_eq!(a.winner_payment_addr, addr(0x50));
    assert_eq!(u.accepted_leaf, None);
    assert_eq!(
        u.work_item(),
        WorkItemRef {
            job_id: id(1),
            unit_id: id(3),
            generation: 0
        }
    );
}

#[test]
fn an_accepted_unit_decodes_its_write_once_leaf() {
    let u = read_unit("unit_accepted", 1, 2).unwrap().expect("held");
    assert_eq!(u.state, UnitState::Accepted);
    let l = u.accepted_leaf.expect("leaf");
    assert_eq!(l.offer_bond_id, id(3));
    assert_eq!(l.commit_bond_id, id(4));
    assert_eq!(l.accepted_bytes, 100);
}

#[test]
fn a_blocked_unit_decodes_its_predecessor_and_required_input() {
    let u = read_unit("unit_blocked_with_input", 1, 4)
        .unwrap()
        .expect("held");
    assert_eq!(u.state, UnitState::Blocked);
    assert_eq!(u.predecessors, vec![id(2)]);
    assert_eq!(u.required_inputs.len(), 1);
    let ri = &u.required_inputs[0];
    assert_eq!(ri.predecessor, id(2));
    assert_eq!(ri.required_output_slot_id, id(0x21));
    assert_eq!(ri.pred_output_manifest_root, id(0x22));
    assert_eq!(ri.required_slot_state_object_root, id(0x23));
    assert_eq!((u.assignment, u.accepted_leaf), (None, None));
}

#[test]
fn null_is_not_held_including_every_read_while_the_gate_is_closed() {
    assert_eq!(read_unit("unit_not_held", 1, 9).unwrap(), None);
    assert_eq!(read_unit("unit_gate_closed", 1, 2).unwrap(), None);
    assert_eq!(read_offer("offer_not_held", 6).unwrap(), None);
    assert_eq!(read_offer("offer_gate_closed", 3).unwrap(), None);
}

#[test]
fn offers_decode_identity_amounts_activity_and_reservation() {
    let o = read_offer("offer_active_reserved", 3)
        .unwrap()
        .expect("held");
    assert_eq!(o.offer_bond_id, id(3));
    assert_eq!(o.identity, addr(8));
    assert_eq!(o.payment_addr, addr(0x50));
    assert_eq!(
        (o.offered_bytes, o.offer_seq, o.bond_locked),
        (1_000, 3, 500)
    );
    assert!(o.active);
    assert_eq!(
        o.reservation,
        Some(Reservation {
            offered: 1_000,
            reserved: 160
        })
    );

    let o = read_offer("offer_inactive", 5).unwrap().expect("held");
    assert_eq!(o.identity, addr(7));
    assert!(!o.active);
    assert_eq!(o.reservation, None);
}

#[test]
fn u128_amounts_survive_at_the_top_of_the_range() {
    let o = read_offer("offer_u128_max", 0x0B).unwrap().expect("held");
    assert_eq!(o.offered_bytes, u128::MAX);
    assert_eq!(o.bond_locked, u128::MAX);
    assert_eq!(
        o.reservation,
        Some(Reservation {
            offered: u128::MAX,
            reserved: u128::MAX - 5
        })
    );
}

#[test]
fn a_json_rpc_error_is_an_error_never_not_held() {
    // -32603: committed state the server itself found inconsistent.
    match unit_from_body(fixture("unit_corrupt_row.response.json"), 1, 3) {
        Err(PoolReadError::Rpc { method, message }) => {
            assert_eq!(method, "pool_getUnit");
            assert!(message.contains("-32603"), "{message}");
        }
        other => panic!("expected an RPC error, got {other:?}"),
    }
    // -32602: an id the server could not parse.
    match unit_from_body(fixture("unit_invalid_params.response.json"), 1, 3) {
        Err(PoolReadError::Rpc { message, .. }) => assert!(message.contains("-32602"), "{message}"),
        other => panic!("expected an RPC error, got {other:?}"),
    }
}

// ── Malformed answers are refused (mutations of real bodies) ──────────────────

/// One named edit of a real response's `result`.
type Mutation = Box<dyn FnOnce(&mut serde_json::Value)>;

fn mutate(name: &str, f: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(&fixture(name)).unwrap();
    f(&mut v["result"]);
    v.to_string()
}

fn assert_malformed<T: std::fmt::Debug>(got: Result<T, PoolReadError>, what: &str) {
    match got {
        Err(PoolReadError::Malformed { .. }) => {}
        other => panic!("{what}: expected Malformed, got {other:?}"),
    }
}

#[test]
fn malformed_unit_answers_are_refused() {
    let base = "unit_accepted.response.json";
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "upper-case 0X prefix",
            Box::new(|r| {
                r["job_id"] = r["job_id"].as_str().unwrap().replacen("0x", "0X", 1).into()
            }),
        ),
        ("short id", Box::new(|r| r["unit_id"] = "0x0202".into())),
        (
            "missing field",
            Box::new(|r| {
                r.as_object_mut().unwrap().remove("accepted_leaf");
            }),
        ),
        ("extra field", Box::new(|r| r["extra"] = true.into())),
        (
            "generation as string",
            Box::new(|r| r["generation"] = "0".into()),
        ),
        (
            "negative generation",
            Box::new(|r| r["generation"] = (-1).into()),
        ),
        (
            "unknown state",
            Box::new(|r| r["state"] = "finished".into()),
        ),
        (
            "amount with a leading zero",
            Box::new(|r| r["accepted_leaf"]["accepted_bytes"] = "0100".into()),
        ),
        (
            "amount above u128::MAX",
            Box::new(|r| {
                r["accepted_leaf"]["accepted_bytes"] =
                    "340282366920938463463374607431768211456".into()
            }),
        ),
        (
            "amount as a number",
            Box::new(|r| r["accepted_leaf"]["accepted_bytes"] = 100.into()),
        ),
        (
            "bad address checksum",
            Box::new(|r| {
                let s = r["assignment"]["winner_payment_addr"]
                    .as_str()
                    .unwrap()
                    .to_string();
                let mut c: Vec<char> = s.chars().collect();
                let last = c.len() - 1;
                c[last] = if c[last] == '2' { '3' } else { '2' };
                r["assignment"]["winner_payment_addr"] = c.into_iter().collect::<String>().into();
            }),
        ),
        (
            "an answer for another unit",
            Box::new(|r| r["unit_id"] = format!("0x{}", "09".repeat(32)).into()),
        ),
        (
            "a non-object result",
            Box::new(|r| *r = serde_json::json!(["not", "a", "unit"])),
        ),
    ];
    for (what, f) in cases {
        assert_malformed(unit_from_body(mutate(base, f), 1, 2), what);
    }
}

#[test]
fn malformed_offer_answers_are_refused() {
    let base = "offer_active_reserved.response.json";
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "reserved above offered",
            Box::new(|r| r["reservation"]["reserved"] = "1001".into()),
        ),
        (
            "missing reservation key",
            Box::new(|r| {
                r.as_object_mut().unwrap().remove("reservation");
            }),
        ),
        (
            "active as a string",
            Box::new(|r| r["active"] = "true".into()),
        ),
        (
            "offer_seq above u64",
            Box::new(|r| {
                r["offer_seq"] =
                    serde_json::from_str::<serde_json::Value>("18446744073709551616").unwrap()
            }),
        ),
        (
            "identity not base58",
            Box::new(|r| r["identity"] = "0OIl".into()),
        ),
        (
            "an answer for another offer",
            Box::new(|r| r["offer_bond_id"] = format!("0x{}", "05".repeat(32)).into()),
        ),
        (
            "bare-hex id",
            Box::new(|r| r["offer_bond_id"] = "03".repeat(32).into()),
        ),
    ];
    for (what, f) in cases {
        assert_malformed(offer_from_body(mutate(base, f), 3), what);
    }
    // Upper-case hex digits, on the one fixture whose id has letters.
    let upper = mutate("offer_u128_max.response.json", |r| {
        r["offer_bond_id"] = format!("0x{}", "0B".repeat(32)).into()
    });
    assert_malformed(offer_from_body(upper, 0x0B), "upper-case hex digits");
}

// ── The accept path, on decoded real answers ─────────────────────────────────

fn assigned() -> PoolUnit {
    read_unit("unit_assigned", 1, 3).unwrap().unwrap()
}
fn offer3() -> PoolOffer {
    read_offer("offer_active_reserved", 3).unwrap().unwrap()
}

#[test]
fn an_assigned_unit_is_accepted_up_to_the_offers_remaining_capacity() {
    let (u, o) = (assigned(), offer3());
    // Offer 3 has reserved 160 of 1,000: 840 more fit, 841 do not.
    let data = build_accept(Some(&u), Some(&o), &id(3), [0x44; 32], 840).unwrap();
    assert_eq!(
        ComputePoolOperation::decode_exact(&data.op_bytes).unwrap(),
        ComputePoolOperation::AcceptUnit(AcceptWorkUnitV1 {
            work_item: WorkItemRef {
                job_id: id(1),
                unit_id: id(3),
                generation: 0
            },
            commit_bond_id: [0x44; 32],
            accepted_bytes: 840,
        })
    );
    assert_eq!(
        build_accept(Some(&u), Some(&o), &id(3), [0x44; 32], 841),
        Err(AcceptRefusal::CapacityExceeded {
            offered: 1_000,
            reserved: 160,
            requested: 841,
            total: 1_001
        })
    );
}

#[test]
fn every_case_the_chain_refuses_is_refused_before_building() {
    let (u, o) = (assigned(), offer3());
    assert_eq!(
        build_accept(None, Some(&o), &id(3), [0x44; 32], 1),
        Err(AcceptRefusal::UnitNotHeld)
    );
    let accepted = read_unit("unit_accepted", 1, 2).unwrap().unwrap();
    assert_eq!(
        build_accept(Some(&accepted), Some(&o), &id(3), [0x44; 32], 1),
        Err(AcceptRefusal::AlreadyAccepted)
    );
    let blocked = read_unit("unit_blocked_with_input", 1, 4).unwrap().unwrap();
    assert_eq!(
        build_accept(Some(&blocked), Some(&o), &id(3), [0x44; 32], 1),
        Err(AcceptRefusal::NotAssigned(UnitState::Blocked))
    );
    // Offer 5 is not the assignment's winner.
    let o5 = read_offer("offer_inactive", 5).unwrap().unwrap();
    assert!(matches!(
        build_accept(Some(&u), Some(&o5), &id(5), [0x44; 32], 1),
        Err(AcceptRefusal::AssignedToAnotherOffer { .. })
    ));
    assert!(matches!(
        build_accept(Some(&u), None, &id(3), [0x44; 32], 1),
        Err(AcceptRefusal::OfferNotHeld(_))
    ));
    // The offer read must be this node's own offer.
    assert!(matches!(
        build_accept(Some(&u), Some(&o5), &id(3), [0x44; 32], 1),
        Err(AcceptRefusal::OfferMismatch { .. })
    ));
    let mut no_row = u.clone();
    no_row.assignment = None;
    assert_eq!(
        build_accept(Some(&no_row), Some(&o), &id(3), [0x44; 32], 1),
        Err(AcceptRefusal::AssignmentMissing)
    );
}

#[test]
fn a_reservation_at_the_top_of_the_range_refuses_on_overflow_not_wrap() {
    let big = read_offer("offer_u128_max", 0x0B).unwrap().unwrap();
    let mut u = assigned();
    u.assignment.as_mut().unwrap().winner_offer_bond_id = id(0x0B);
    // reserved = MAX - 5: 5 more fit exactly, 6 exceed, and 10 overflow.
    assert!(build_accept(Some(&u), Some(&big), &id(0x0B), [0x44; 32], 5).is_ok());
    assert!(matches!(
        build_accept(Some(&u), Some(&big), &id(0x0B), [0x44; 32], 6),
        Err(AcceptRefusal::Overflow { .. })
    ));
}

#[test]
fn one_active_offer_per_identity_is_checked_before_publishing() {
    let o3 = offer3();
    assert_eq!(
        ensure_no_active_offer(&addr(8), Some(&o3)),
        Err(PublishRefusal::ActiveOfferExists(format!(
            "0x{}",
            "03".repeat(32)
        )))
    );
    let o5 = read_offer("offer_inactive", 5).unwrap().unwrap();
    assert_eq!(ensure_no_active_offer(&addr(7), Some(&o5)), Ok(()));
    assert!(matches!(
        ensure_no_active_offer(&addr(8), Some(&o5)),
        Err(PublishRefusal::NotOwnOffer { .. })
    ));
    assert_eq!(ensure_no_active_offer(&addr(8), None), Ok(()));
}

// ── Live server (opt-in) ─────────────────────────────────────────────────────

/// Against a live sum-chain RPC server serving the generator's seeded state
/// (`tests/fixtures/pool_reads/PROVENANCE.md`): set `SUMCHAIN_POOL_RPC_URL` and
/// run with `--ignored`.
#[test]
#[ignore = "needs a live sum-chain RPC server; see tests/fixtures/pool_reads/PROVENANCE.md"]
fn live_server_reads_and_accept_path() {
    let url = std::env::var("SUMCHAIN_POOL_RPC_URL").expect("SUMCHAIN_POOL_RPC_URL");
    let c = client(url);
    let u = c
        .pool_get_unit(&id(1), &id(3))
        .unwrap()
        .expect("unit 3 held");
    assert_eq!(u.state, UnitState::Assigned);
    let o = c.pool_get_offer(&id(3)).unwrap().expect("offer 3 held");
    assert_eq!(
        o.reservation,
        Some(Reservation {
            offered: 1_000,
            reserved: 160
        })
    );
    assert!(build_accept(Some(&u), Some(&o), &id(3), [0x44; 32], 840).is_ok());
    assert!(matches!(
        build_accept(Some(&u), Some(&o), &id(3), [0x44; 32], 841),
        Err(AcceptRefusal::CapacityExceeded { .. })
    ));
    let accepted = c
        .pool_get_unit(&id(1), &id(2))
        .unwrap()
        .expect("unit 2 held");
    assert_eq!(accepted.state, UnitState::Accepted);
    assert_eq!(c.pool_get_unit(&id(1), &id(9)).unwrap(), None);
    assert_eq!(c.pool_get_offer(&id(6)).unwrap(), None);
    let big = c
        .pool_get_offer(&id(0x0B))
        .unwrap()
        .expect("offer 0x0b held");
    assert_eq!(big.offered_bytes, u128::MAX);
}
