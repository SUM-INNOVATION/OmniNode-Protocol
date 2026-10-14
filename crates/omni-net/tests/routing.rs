//! Two consumers of one node, over a real QUIC connection.
//!
//! The unit tests in `omni_net::router` pin the fan-out against a synthetic
//! event lane. These close the gap the old design actually failed in: a single
//! node with more than one interested consumer, receiving real traffic of both
//! kinds at once.
//!
//! Before the router there was one `mpsc::Receiver` per node and consumers
//! took turns on it behind `Arc<tokio::sync::Mutex<OmniNet>>`. Taking turns on
//! a receiver is not sharing — `recv()` removes the event — so
//! `omni-contributor`'s relay consumed the tensor transport's events and
//! dropped them, and the transport did the same to the relay's. Both consumers
//! below would have been fed by the same receiver; either one draining would
//! have starved the other.
//!
//! Gossip fan-out is exercised in the unit tests rather than here: forming a
//! gossipsub mesh takes a heartbeat, and a test that waits on one is a test
//! that fails on a loaded machine for reasons unrelated to routing. Nodes are
//! wired by an explicit dial to a loopback address, and run with mDNS off: with
//! it on, every node of every concurrently running test on the host discovers
//! and dials every other, so a node under test receives traffic and
//! connections from nodes it has nothing to do with.
//!
//! Every wait is on the condition the next step needs (a listen address from
//! the swarm, a connection to the expected peer, a router count), never on a
//! fixed sleep. `DEADLINE` only bounds how long a broken run takes to fail.

use std::time::Duration;

use libp2p::{multiaddr::Protocol, Multiaddr, PeerId};
use omni_net::{
    Interests, OmniNet, OmniNetEvent, RequestError, Subscription, TensorRequest,
};
use omni_types::config::NetConfig;

/// Loopback QUIC on a warm machine connects in milliseconds; this is a
/// failure deadline, not an expected duration.
const DEADLINE: Duration = Duration::from_secs(20);

// ── Harness ───────────────────────────────────────────────────────────────────

/// A node with mDNS off, so it connects only to what the test dials.
async fn new_node() -> OmniNet {
    OmniNet::new(NetConfig {
        mdns: false,
        ..NetConfig::default()
    })
    .await
    .expect("OmniNet::new on an OS-assigned port")
}

/// The node's loopback dial address, with its peer id appended.
///
/// The address comes from [`OmniNet::listen_addrs`], which answers from the
/// swarm's own state. Waiting for an `OmniNetEvent::Listening` on a
/// subscription instead races the swarm: the event is sent once, as soon as the
/// listener binds, and a subscription registered after that never sees it.
/// The node listens on `0.0.0.0`, so its port is reachable on loopback whichever
/// interface was reported first.
async fn dialable_addr(net: &OmniNet) -> Multiaddr {
    let addrs = tokio::time::timeout(DEADLINE, net.listen_addrs())
        .await
        .expect("timed out waiting for the swarm to report a listen address")
        .expect("the swarm reports its listen addresses");
    let port = addrs
        .iter()
        .find_map(|a| {
            a.iter().find_map(|p| match p {
                Protocol::Udp(port) => Some(port),
                _ => None,
            })
        })
        .expect("a QUIC listen address carries a UDP port");
    format!("/ip4/127.0.0.1/udp/{port}/quic-v1")
        .parse::<Multiaddr>()
        .expect("loopback QUIC multiaddr")
        .with(Protocol::P2p(net.local_peer_id()))
}

/// Wait until `condition` holds, checking it every few milliseconds. For
/// conditions with no event to wait on, such as a router count: the wait ends
/// as soon as the condition holds, and `DEADLINE` only bounds a broken run.
async fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(DEADLINE, async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting until {what}"));
}

/// Wait until `control` reports a connection to `expected`.
///
/// It must be that specific peer: a connection to anything else would let a
/// request go out before the dialled connection exists.
async fn wait_connected(control: &mut Subscription, expected: PeerId) {
    tokio::time::timeout(DEADLINE, async {
        while let Some(event) = control.recv().await {
            if let OmniNetEvent::PeerConnected { peer_id } = event {
                if peer_id == expected {
                    return Some(());
                }
            }
        }
        None
    })
    .await
    .expect("timed out waiting for the dialled connection")
    .expect("event stream closed before connecting")
}

fn tensor_for(session: &str) -> TensorRequest {
    TensorRequest {
        session_id: session.to_string(),
        micro_batch_index: 0,
        from_stage: 0,
        to_stage: 1,
        seq_len: 1,
        hidden_dim: 2,
        dtype: 0,
        data: vec![7, 7, 7, 7],
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn a_shard_consumer_and_a_tensor_consumer_do_not_steal_from_each_other() {
    // The shape of the bug, live: one node, two consumers, both kinds of
    // traffic arriving. Each consumer must see its own and only its own.
    let server = new_node().await;
    let server_net = server.handle();
    let server_addr = dialable_addr(&server).await;
    let server_peer = server.local_peer_id();

    // The two consumers that used to fight over one receiver.
    let mut store = server_net
        .subscribe(Interests::none().shard())
        .expect("shard consumer");
    let mut transport = server_net
        .subscribe(Interests::none().tensor())
        .expect("tensor consumer");

    let client = new_node().await;
    let client_net = client.handle();
    let mut client_control = client_net
        .subscribe(Interests::none().control())
        .expect("client control subscription");
    client_net.dial(server_addr).await.expect("dial");
    wait_connected(&mut client_control, server_peer).await;

    // Both kinds of traffic, interleaved, from the same peer.
    client_net
        .request_shard_chunk(server_peer, "cid-alpha".into(), None, None)
        .await
        .expect("shard request");
    client_net
        .request_tensor(server_peer, tensor_for("session-alpha"))
        .await
        .expect("tensor request");
    client_net
        .request_shard_chunk(server_peer, "cid-beta".into(), None, None)
        .await
        .expect("second shard request");

    // The tensor consumer gets the tensor — the event a draining shard
    // consumer would have swallowed.
    let tensor_event = tokio::time::timeout(DEADLINE, transport.recv())
        .await
        .expect("timed out waiting for the tensor")
        .expect("tensor stream closed");
    match tensor_event {
        OmniNetEvent::TensorReceived { request, .. } => {
            assert_eq!(request.session_id, "session-alpha");
        }
        other => panic!("the tensor consumer received {other:?}"),
    }

    // ...and the shard consumer still has both of its requests.
    let mut cids = Vec::new();
    for _ in 0..2 {
        let event = tokio::time::timeout(DEADLINE, store.recv())
            .await
            .expect("timed out waiting for a shard request")
            .expect("shard stream closed");
        match event {
            OmniNetEvent::ShardRequested { request, .. } => cids.push(request.cid),
            other => panic!("the shard consumer received {other:?}"),
        }
    }
    cids.sort();
    assert_eq!(cids, vec!["cid-alpha".to_string(), "cid-beta".to_string()]);

    // Neither consumer was handed the other's traffic to discard.
    assert!(transport.try_recv().is_none());
    assert!(store.try_recv().is_none());

    // And nothing that did arrive went uncounted.
    let counts = server_net.router_counts();
    assert!(counts.delivered >= 3, "counts: {counts:?}");
    assert_eq!(counts.dropped_backlogged, 0, "counts: {counts:?}");
    assert_eq!(counts.dropped_departed, 0, "counts: {counts:?}");

    let _ = server_net.shutdown().await;
    let _ = client_net.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_consumer_deregisters_and_the_survivor_keeps_receiving() {
    let server = new_node().await;
    let server_net = server.handle();
    let server_addr = dialable_addr(&server).await;
    let server_peer = server.local_peer_id();

    // Release the `OmniNet` value itself: it carries an all-interest
    // subscription backing `next_event`, and a node whose consumers hold
    // `NetHandle`s — which is what `omni-node` now does — has no such
    // catch-all registered. Keeping it would mean every event below had a
    // taker, and "nobody asked for this" could never be observed. The swarm
    // and the router stay alive; `server_net` holds a command sender clone.
    drop(server);

    let store = server_net.subscribe(Interests::none().shard()).unwrap();
    let mut transport = server_net.subscribe(Interests::none().tensor()).unwrap();
    let before = server_net.router().subscriber_count();
    assert_eq!(before, 2, "only the two consumers under test are registered");

    // One consumer walks away mid-session.
    drop(store);
    assert_eq!(
        server_net.router().subscriber_count(),
        before - 1,
        "a dropped consumer must release its slot without a sweep"
    );

    let client = new_node().await;
    let client_net = client.handle();
    let mut client_control = client_net
        .subscribe(Interests::none().control())
        .expect("client control subscription");
    client_net.dial(server_addr).await.expect("dial");
    wait_connected(&mut client_control, server_peer).await;

    // Traffic for the departed consumer, then traffic for the survivor. The
    // two travel on separate streams, so nothing orders the server's handling
    // of one before the other: wait until the router has counted the shard
    // request as unwanted before asserting anything about it.
    client_net
        .request_shard_chunk(server_peer, "cid-orphan".into(), None, None)
        .await
        .expect("shard request");
    wait_until("the router counts the orphaned shard request", || {
        server_net.router_counts().unwanted_shard >= 1
    })
    .await;
    client_net
        .request_tensor(server_peer, tensor_for("session-survivor"))
        .await
        .expect("tensor request");

    let event = tokio::time::timeout(DEADLINE, transport.recv())
        .await
        .expect("timed out waiting for the tensor")
        .expect("tensor stream closed");
    match event {
        OmniNetEvent::TensorReceived { request, .. } => {
            assert_eq!(request.session_id, "session-survivor");
        }
        other => panic!("the survivor received {other:?}"),
    }

    // The shard request went to nobody — counted as unsubscribed, which is a
    // different fault from failing a consumer that did ask.
    let counts = server_net.router_counts();
    assert!(counts.unwanted_shard >= 1, "counts: {counts:?}");
    assert_eq!(counts.dropped_departed, 0, "counts: {counts:?}");

    let _ = server_net.shutdown().await;
    let _ = client_net.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn shutdown_fails_in_flight_requests_and_ends_every_subscription() {
    // The "no caller waits forever" guarantee, across both delivery paths at
    // once: a solicited request waiting on its private completion channel, and
    // three consumers waiting on their router subscriptions.
    let server = new_node().await;
    let server_net = server.handle();
    let server_addr = dialable_addr(&server).await;
    let server_peer = server.local_peer_id();
    // Deliberately no responder task: the server accepts the requests and
    // never answers, so both requests are still in flight at shutdown.

    let client = new_node().await;
    let client_net = client.handle();
    let mut control = client_net
        .subscribe(Interests::none().control())
        .expect("control consumer");
    let mut shard_events = client_net
        .subscribe(Interests::none().shard())
        .expect("shard consumer");
    let mut tensor_events = client_net
        .subscribe(Interests::none().tensor())
        .expect("tensor consumer");
    client_net.dial(server_addr).await.expect("dial");
    wait_connected(&mut control, server_peer).await;

    let first = client_net
        .fetch_shard_chunk(server_peer, "never-answered".into(), None, None)
        .await;
    let second = client_net
        .send_tensor(server_peer, tensor_for("never-acked"))
        .await;

    client_net.shutdown().await.expect("shutdown command");

    // Every waiting caller is completed, with the reason, rather than left on
    // a channel that can no longer produce.
    assert_eq!(
        tokio::time::timeout(DEADLINE, first.response())
            .await
            .expect("the shard caller must be completed, not left waiting")
            .unwrap_err(),
        RequestError::RouterGone
    );
    assert_eq!(
        tokio::time::timeout(DEADLINE, second.response())
            .await
            .expect("the tensor caller must be completed, not left waiting")
            .unwrap_err(),
        RequestError::RouterGone
    );

    // And every consumer learns the stream has ended.
    for (name, sub) in [
        ("control", &mut control),
        ("shard", &mut shard_events),
        ("tensor", &mut tensor_events),
    ] {
        let ended = tokio::time::timeout(DEADLINE, async {
            while sub.recv().await.is_some() {}
        })
        .await;
        assert!(
            ended.is_ok(),
            "the {name} consumer was left awaiting a stream that had stopped"
        );
    }

    assert!(
        !client_net.router().is_running(),
        "the router must stop when its lane closes"
    );
    assert!(
        client_net.subscribe(Interests::everything()).is_err(),
        "a subscription taken out after shutdown would never yield"
    );

    let _ = server_net.shutdown().await;
}
