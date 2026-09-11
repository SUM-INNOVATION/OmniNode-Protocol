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
//! that fails on a loaded machine for reasons unrelated to routing. Both nodes
//! bind `127.0.0.1` on an OS-assigned port and are wired by an explicit dial,
//! so nothing here depends on mDNS, the DHT, or a heartbeat.

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

async fn new_node() -> OmniNet {
    OmniNet::new(NetConfig::default())
        .await
        .expect("OmniNet::new on an OS-assigned port")
}

fn is_loopback(addr: &Multiaddr) -> bool {
    addr.iter()
        .any(|p| matches!(p, Protocol::Ip4(ip) if ip.is_loopback()))
}

/// Wait on a *control* subscription until the node reports a loopback listen
/// address, and return it with the node's peer id appended so it can be
/// dialled directly.
async fn dialable_addr(net: &OmniNet, control: &mut Subscription) -> Multiaddr {
    let peer_id = net.local_peer_id();
    let found = tokio::time::timeout(DEADLINE, async {
        while let Some(event) = control.recv().await {
            if let OmniNetEvent::Listening { addr } = event {
                if is_loopback(&addr) {
                    return Some(addr);
                }
            }
        }
        None
    })
    .await
    .expect("timed out waiting for a listen address")
    .expect("event stream closed before a listen address appeared");

    found.with(Protocol::P2p(peer_id))
}

/// Wait until `control` reports a connection to `expected`.
///
/// It must be that specific peer: mDNS is on by default, so a node in this
/// suite will also connect to the other tests' nodes running concurrently on
/// the same host.
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
    let mut server_control = server_net
        .subscribe(Interests::none().control())
        .expect("server control subscription");
    let server_addr = dialable_addr(&server, &mut server_control).await;
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
    let mut server_control = server_net
        .subscribe(Interests::none().control())
        .expect("server control subscription");
    let server_addr = dialable_addr(&server, &mut server_control).await;
    let server_peer = server.local_peer_id();

    // Release the `OmniNet` value itself: it carries an all-interest
    // subscription backing `next_event`, and a node whose consumers hold
    // `NetHandle`s — which is what `omni-node` now does — has no such
    // catch-all registered. Keeping it would mean every event below had a
    // taker, and "nobody asked for this" could never be observed. The swarm
    // and the router stay alive; `server_net` holds a command sender clone.
    drop(server_control);
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

    // Traffic for the departed consumer, then traffic for the survivor.
    client_net
        .request_shard_chunk(server_peer, "cid-orphan".into(), None, None)
        .await
        .expect("shard request");
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
    let mut server_control = server_net
        .subscribe(Interests::none().control())
        .expect("server control subscription");
    let server_addr = dialable_addr(&server, &mut server_control).await;
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
