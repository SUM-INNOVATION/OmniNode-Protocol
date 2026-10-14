//! End-to-end correlation over a real QUIC connection between two nodes.
//!
//! The unit tests in `omni_net::request` pin the table's behaviour with a
//! stand-in key, because libp2p does not let anyone construct an
//! `OutboundRequestId`. These tests close that gap by driving real requests
//! through a real swarm, where the ids are the ones libp2p actually issued.
//!
//! Nodes are wired together by an explicit dial to a loopback address and run
//! with mDNS off, so nothing here depends on mDNS, the DHT, or a gossipsub
//! heartbeat. With mDNS on, every node of every concurrently running test on
//! the host discovers and dials every other, and a request can then travel on
//! one of those extra connections and fail when it closes.
//!
//! Every wait is on the condition the next step needs (a listen address from
//! the swarm, a connection, a request held by the server, a full event lane),
//! never on a fixed sleep. `DEADLINE` only bounds how long a broken run takes
//! to fail.

use std::time::Duration;

use libp2p::{Multiaddr, PeerId, multiaddr::Protocol};
use omni_net::{OmniNet, OmniNetEvent, RequestError, ShardResponse};
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
/// swarm's own state rather than from a `Listening` event on the best-effort
/// event lane. The node listens on `0.0.0.0`, so its port is reachable on
/// loopback whichever interface was reported first.
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

/// Drain `net` until it reports a connection to `expected`.
///
/// It must be that specific peer: returning on the first `PeerConnected` from
/// anything else would let a request go out before the dialled connection
/// exists.
async fn wait_connected(net: &mut OmniNet, expected: PeerId) {
    tokio::time::timeout(DEADLINE, async {
        while let Some(event) = net.next_event().await {
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

/// A node that answers every shard request by echoing the requested cid back
/// as the response payload, so a response can be attributed to its request.
///
/// `delay_first` holds the first response back until the second has been sent,
/// which is what makes the ordering in the correlation test genuinely
/// out-of-order rather than incidentally so.
fn spawn_echo_responder(mut net: OmniNet, delay_first: Option<Duration>) {
    tokio::spawn(async move {
        let mut answered = 0usize;
        while let Some(event) = net.next_event().await {
            if let OmniNetEvent::ShardRequested {
                request,
                channel_id,
                ..
            } = event
            {
                if answered == 0 {
                    if let Some(delay) = delay_first {
                        tokio::time::sleep(delay).await;
                    }
                }
                answered += 1;
                let response = ShardResponse {
                    cid: request.cid.clone(),
                    offset: request.offset.unwrap_or(0),
                    total_bytes: request.cid.len() as u64,
                    data: request.cid.into_bytes(),
                    error: None,
                };
                let _ = net.respond_shard(channel_id, response).await;
            }
        }
    });
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn two_requests_to_one_peer_complete_independently() {
    // The load-bearing correlation property, over the wire. Both requests go
    // to the same peer and differ only in their cid; only the per-request
    // `OutboundRequestId` can tell their responses apart. Keying the pending
    // table by PeerId makes the second request's response overwrite — or be
    // dropped by — the first.
    let mut server = new_node().await;
    let server_addr = dialable_addr(&server).await;
    let server_peer = server.local_peer_id();
    // Hold the first response back so the responses land in the opposite
    // order from the requests.
    spawn_echo_responder(server, Some(Duration::from_millis(300)));

    let mut client = new_node().await;
    client.dial(server_addr).await.expect("dial");
    wait_connected(&mut client, server_peer).await;

    let first = client
        .fetch_shard_chunk(server_peer, "cid-first".into(), Some(0), Some(64))
        .await;
    let second = client
        .fetch_shard_chunk(server_peer, "cid-second".into(), Some(0), Some(64))
        .await;

    let (a, b) = tokio::join!(
        first.response_within(DEADLINE),
        second.response_within(DEADLINE),
    );

    let a = a.expect("first request must complete");
    let b = b.expect("second request must complete");
    assert_eq!(
        a.cid, "cid-first",
        "first caller received another request's response"
    );
    assert_eq!(
        b.cid, "cid-second",
        "second caller received another request's response"
    );
    assert_eq!(a.data, b"cid-first".to_vec());
    assert_eq!(b.data, b"cid-second".to_vec());
}

/// Concurrent request streams one connection carries: libp2p
/// request-response's default `max_concurrent_streams` (0.28), which omni-net
/// does not override. Each flooder can have this many requests outstanding.
const STREAMS_PER_CONNECTION: usize = 100;

/// The event lane's depth (`omni_net`'s per-subscriber capacity).
const LANE_CAPACITY: usize = 256;

/// Flood `victim` with inbound tensor requests it will never answer.
///
/// One flooder holds at most [`STREAMS_PER_CONNECTION`] requests open at once,
/// so filling a [`LANE_CAPACITY`]-deep lane takes
/// `LANE_CAPACITY / STREAMS_PER_CONNECTION + 1` of them.
fn spawn_flooder(victim_addr: Multiaddr, victim_peer: PeerId, tag: &'static str) {
    tokio::spawn(async move {
        let mut net = new_node().await;
        if net.dial(victim_addr).await.is_err() {
            return;
        }
        wait_connected(&mut net, victim_peer).await;
        for i in 0..400u32 {
            let request = omni_net::TensorRequest {
                session_id: format!("{tag}-{i}"),
                micro_batch_index: i,
                from_stage: 0,
                to_stage: 0,
                seq_len: 1,
                hidden_dim: 1,
                dtype: 2,
                data: vec![0u8; 8],
            };
            let _ = net.request_tensor(victim_peer, request).await;
        }
        // Hold the node — and therefore the connection — open.
        while net.next_event().await.is_some() {}
    });
}

#[tokio::test(flavor = "multi_thread")]
async fn a_response_arrives_while_the_ordinary_event_lane_is_saturated() {
    // The point of a per-request completion channel: a solicited response is
    // not queued behind unsolicited traffic, and is not lost when that traffic
    // overflows. Here the client's shared event lane is filled to capacity and
    // never drained, and the response still arrives.
    let mut client = new_node().await;
    let client_addr = dialable_addr(&client).await;
    let client_peer = client.local_peer_id();

    let mut server = new_node().await;
    let server_addr = dialable_addr(&server).await;
    let server_peer = server.local_peer_id();
    spawn_echo_responder(server, None);

    client.dial(server_addr).await.expect("dial");
    wait_connected(&mut client, server_peer).await;

    // Drain whatever else has queued so the flood below is the only thing in
    // the client's lane. Nothing drains it from here on.
    while client.try_next_event().is_some() {}

    // From here every event the router delivers to this node stays queued in
    // its only subscription, so new deliveries are the backlog. Wait until the
    // flood has actually filled the lane rather than for a fixed time: a slow
    // runner needs longer to connect two flooders and open their streams.
    let delivered_before = client.router_counts().delivered;
    let tags = ["flood-a", "flood-b", "flood-c"];
    let flooders = LANE_CAPACITY / STREAMS_PER_CONNECTION + 1;
    assert!(flooders <= tags.len(), "{flooders} flooders needed");
    for tag in tags.into_iter().take(flooders) {
        spawn_flooder(client_addr.clone(), client_peer, tag);
    }
    tokio::time::timeout(DEADLINE, async {
        while client.router_counts().delivered - delivered_before < 250 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the flood never filled the client's event lane");

    let response = client
        .fetch_shard_chunk(server_peer, "cid-under-flood".into(), Some(0), Some(64))
        .await
        .response_within(DEADLINE)
        .await
        .expect("a solicited response must not queue behind unsolicited traffic");
    assert_eq!(response.cid, "cid-under-flood");

    // Now show the lane really was saturated, and that the response did not
    // travel through it.
    let mut banked = 0usize;
    let mut solicited_on_the_lane = 0usize;
    while let Some(event) = client.try_next_event() {
        banked += 1;
        if matches!(event, OmniNetEvent::ShardReceived { .. }) {
            solicited_on_the_lane += 1;
        }
    }
    assert!(
        banked >= 250,
        "the event lane was not actually saturated ({banked} events banked), \
         so this test did not exercise what it claims"
    );
    assert_eq!(
        solicited_on_the_lane, 0,
        "the solicited response was published on the shared lane, which is \
         exactly what the completion channel exists to prevent"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn shutting_the_swarm_down_completes_every_in_flight_request() {
    // The server receives the request and never answers. The client then
    // shuts down, and must release its caller rather than leave it waiting on
    // a response that can no longer arrive.
    let mut server = new_node().await;
    let server_addr = dialable_addr(&server).await;
    let server_peer = server.local_peer_id();

    let mut client = new_node().await;
    client.dial(server_addr).await.expect("dial");
    wait_connected(&mut client, server_peer).await;

    let pending = client
        .fetch_shard_chunk(server_peer, "cid-never-answered".into(), Some(0), Some(16))
        .await;

    // Shut down only once the server holds the request: it has been handed to
    // the server's consumer, so the server keeps its response channel open and
    // nothing but the client's own shutdown can complete the caller.
    tokio::time::timeout(DEADLINE, async {
        while let Some(event) = server.next_event().await {
            if let OmniNetEvent::ShardRequested { request, .. } = event {
                if request.cid == "cid-never-answered" {
                    return;
                }
            }
        }
        panic!("the server's event stream closed before the request arrived");
    })
    .await
    .expect("the server never received the request");
    client.shutdown().await.expect("shutdown");

    let err = pending
        .response_within(DEADLINE)
        .await
        .expect_err("shutdown must not deliver a response");
    assert_eq!(
        err,
        RequestError::RouterGone,
        "a shutdown caller must be told why, not merely time out"
    );

    drop(server);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_undialable_peer_completes_the_caller_with_a_failure() {
    // No address is known for this peer, so libp2p reports an outbound
    // failure. The caller must learn that through its own completion channel.
    let client = new_node().await;
    let nowhere = PeerId::random();

    let err = client
        .fetch_shard_chunk(nowhere, "cid-nowhere".into(), Some(0), Some(16))
        .await
        .response_within(DEADLINE)
        .await
        .expect_err("a request to an unreachable peer cannot succeed");

    match err {
        RequestError::Outbound { peer, .. } => {
            assert_eq!(
                peer,
                nowhere.to_string(),
                "failure attributed to the wrong peer"
            );
        }
        other => panic!("expected an outbound failure, got {other:?}"),
    }

    let _ = client.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_caller_does_not_wedge_the_swarm() {
    // Dropping the handle is cancellation. The swarm must keep serving other
    // requests afterwards — the cancelled entry is released, not leaked into
    // the path of the next caller.
    let mut server = new_node().await;
    let server_addr = dialable_addr(&server).await;
    let server_peer = server.local_peer_id();
    spawn_echo_responder(server, None);

    let mut client = new_node().await;
    client.dial(server_addr).await.expect("dial");
    wait_connected(&mut client, server_peer).await;

    let abandoned = client
        .fetch_shard_chunk(server_peer, "cid-abandoned".into(), Some(0), Some(16))
        .await;
    drop(abandoned);

    let response = client
        .fetch_shard_chunk(server_peer, "cid-after-cancel".into(), Some(0), Some(16))
        .await
        .response_within(DEADLINE)
        .await
        .expect("a request issued after a cancellation must still complete");
    assert_eq!(response.cid, "cid-after-cancel");

    let _ = client.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_issued_after_shutdown_is_reported_as_never_sent() {
    let client = new_node().await;
    client.shutdown().await.expect("shutdown");
    // Wait for the loop to exit and drop its command receiver. Commands are
    // handled in order, so a query sent after the shutdown is never answered:
    // it fails exactly when the loop has gone, taking the receiver with it.
    tokio::time::timeout(DEADLINE, client.listen_addrs())
        .await
        .expect("the swarm loop never exited")
        .expect_err("a query queued behind the shutdown is never answered");

    let err = client
        .fetch_shard_chunk(PeerId::random(), "cid-too-late".into(), None, None)
        .await
        .response_within(DEADLINE)
        .await
        .expect_err("the swarm is gone; nothing can have been sent");
    assert_eq!(err, RequestError::NotSent);
}
