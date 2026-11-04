//! The drain's consequence on a live network, made deterministic.
//!
//! `tests/shutdown_drain.rs` shows that a message published just before exit
//! reaches a connected watcher under real scheduling. Whether the same
//! sequence *without* the drain loses the message, though, is a race: the
//! connection task may write the message out before the swarm is dropped, and
//! on some machines it nearly always does. A negative control that waits to
//! lose that race fails whenever it is not lost.
//!
//! These tests take the race out. The publisher runs with a
//! [`WriteHold`](super::hold) on its final message: the message's bytes are
//! held at the transport, unsent, until the swarm is polled after the stop
//! request. Each run checks that the bytes really were held when the stop was
//! accepted, so every run is the worst case the drain exists for — the
//! message still queued in the connection when shutdown is requested — and
//! then:
//!
//! * with the drain, the swarm keeps running, the hold is released, and the
//!   watcher receives the message;
//! * without it, the swarm is dropped without being polled again, the hold is
//!   never released, and the message is lost. This follows from the order of
//!   operations in the swarm loop, not from timing.
//!
//! The harness is otherwise the one in `tests/shutdown_drain.rs`: each run is
//! a fresh publisher on its own runtime and thread, shut down through
//! [`OmniNet::shutdown_with_drain`] and then torn down with
//! `shutdown_background` as a process exit would, and a pass is the watcher's
//! own receipt of the bytes. Loopback QUIC, mDNS off.

use std::time::{Duration, Instant};

use libp2p::{Multiaddr, PeerId, multiaddr::Protocol};
use omni_types::config::NetConfig;
use tokio::sync::{mpsc, oneshot};

use super::hold::{HoldReport, WriteHold};
use crate::{
    DEFAULT_SHUTDOWN_DRAIN, DrainOutcome, DrainReport, Interests, OmniNet, OmniNetEvent,
    TOPIC_CONTRIBUTOR_RESULT,
};

/// Failure deadline for connecting, warming up, and the final message
/// reaching the transport; loopback takes milliseconds.
const DEADLINE: Duration = Duration::from_secs(20);

/// How long the watcher keeps looking for the final message after the
/// publisher has gone. Once the publisher's runtime is gone nothing more can
/// arrive; this only absorbs scheduling delay on the watcher's side.
const RECEIPT_WINDOW: Duration = Duration::from_secs(1);

/// Runs per test. The outcome of a run does not depend on scheduling, so a
/// few runs suffice; more than one shows it does not drift across fresh
/// publishers against one watcher.
const RUNS: usize = 5;

const TOPIC: &str = TOPIC_CONTRIBUTOR_RESULT;

/// The final message of `run`, which is also the hold's marker. Probes never
/// contain it.
fn final_payload(run: usize) -> String {
    format!("held-final-message-{run}")
}

// ── Harness ───────────────────────────────────────────────────────────────────

async fn new_node() -> OmniNet {
    OmniNet::new(NetConfig {
        mdns: false,
        ..NetConfig::default()
    })
    .await
    .expect("OmniNet::new on an OS-assigned port")
}

async fn dialable_addr(net: &OmniNet) -> Multiaddr {
    let addrs = tokio::time::timeout(DEADLINE, net.listen_addrs())
        .await
        .expect("timed out waiting for a listen address")
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

async fn wait_connected(net: &mut OmniNet, expected: PeerId) {
    tokio::time::timeout(DEADLINE, async {
        while let Some(event) = net.next_event().await {
            if matches!(event, OmniNetEvent::PeerConnected { peer_id } if peer_id == expected) {
                return;
            }
        }
        panic!("event stream closed before connecting");
    })
    .await
    .expect("timed out waiting for the dialled connection")
}

/// One publisher "process": connect, warm up, publish the final message, wait
/// until its bytes are held at the transport, shut down with `drain`, then
/// stop the runtime the way a process exit does.
fn run_publisher(
    watcher: Multiaddr,
    watcher_peer: PeerId,
    run: usize,
    drain: Duration,
    probe_seen: oneshot::Receiver<()>,
) -> (DrainReport, HoldReport) {
    // `#[tokio::main]`'s default flavour, as the CLI uses.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("publisher runtime");
    let payload = final_payload(run);
    let hold = WriteHold::new(payload.as_bytes());
    let node_hold = hold.clone();
    let report = rt.block_on(async move {
        node_hold.attach_to_next_build();
        let mut node = new_node().await;
        assert!(
            node_hold.is_attached(),
            "run {run}: the publisher's swarm did not take the hold"
        );
        node.dial(watcher).await.expect("dial the watcher");
        wait_connected(&mut node, watcher_peer).await;

        // Probe until the watcher has one, so the publisher knows the watcher
        // subscribes: the state of the reported failure.
        let probes = async {
            let mut k = 0u64;
            loop {
                node.publish(TOPIC, format!("probe-{run}-{k}").into_bytes())
                    .await
                    .expect("publish probe");
                k += 1;
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        };
        tokio::select! {
            _ = probes => unreachable!("the probe loop does not end"),
            seen = tokio::time::timeout(DEADLINE, probe_seen) => {
                seen.expect("the watcher never received a probe")
                    .expect("the test stopped waiting for probes");
            }
        }

        node.publish(TOPIC, payload.into_bytes())
            .await
            .expect("publish the final message");
        // The connection has taken the message and is trying to write it.
        tokio::time::timeout(DEADLINE, node_hold.wait_held())
            .await
            .unwrap_or_else(|_| panic!("run {run}: the final message never reached the transport"));
        node.shutdown_with_drain(drain).await.expect("shutdown")
    });
    // Process exit: nothing of this node runs past this line.
    rt.shutdown_background();
    (report, hold.report())
}

/// The watcher's view of one topic, as payload strings.
struct Watcher {
    addr: Multiaddr,
    peer: PeerId,
    received: mpsc::UnboundedReceiver<String>,
    _node: OmniNet,
}

async fn start_watcher() -> Watcher {
    let node = new_node().await;
    let addr = dialable_addr(&node).await;
    let peer = node.local_peer_id();
    let mut sub = node
        .subscribe(Interests::none().topic(TOPIC))
        .expect("subscribe to the result topic");
    let (tx, received) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(event) = sub.recv().await {
            match event {
                OmniNetEvent::MessageReceived { topic, data, .. } if topic == TOPIC => {
                    let _ = tx.send(String::from_utf8_lossy(&data).into_owned());
                }
                _ => {}
            }
        }
    });
    Watcher {
        addr,
        peer,
        received,
        _node: node,
    }
}

impl Watcher {
    /// Wait up to `within` for a payload matching `want`, discarding others.
    async fn wait_for(&mut self, within: Duration, want: impl Fn(&str) -> bool) -> bool {
        tokio::time::timeout(within, async {
            while let Some(m) = self.received.recv().await {
                if want(&m) {
                    return true;
                }
            }
            false
        })
        .await
        .unwrap_or(false)
    }
}

#[derive(Debug)]
struct RunResult {
    received: bool,
    report: DrainReport,
    hold: HoldReport,
}

/// One publisher run against `watcher`.
async fn one_run(watcher: &mut Watcher, run: usize, drain: Duration) -> RunResult {
    let (probe_tx, probe_rx) = oneshot::channel();
    let (addr, peer) = (watcher.addr.clone(), watcher.peer);
    let publisher = std::thread::spawn(move || run_publisher(addr, peer, run, drain, probe_rx));

    let probe_prefix = format!("probe-{run}-");
    assert!(
        watcher
            .wait_for(DEADLINE, |m| m.starts_with(&probe_prefix))
            .await,
        "run {run}: no probe reached the watcher"
    );
    let _ = probe_tx.send(());

    // Join without blocking the watcher's runtime: its swarm has to keep
    // running while the publisher drains.
    let (report, hold) = tokio::task::spawn_blocking(move || publisher.join())
        .await
        .expect("join task")
        .expect("publisher thread panicked");

    // The publisher is gone. Whatever the watcher has not received by now
    // was lost with it.
    let final_msg = final_payload(run);
    let received = watcher.wait_for(RECEIPT_WINDOW, |m| m == final_msg).await;
    RunResult {
        received,
        report,
        hold,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// With the default drain, a final message that is still queued in the
/// connection when shutdown is requested reaches the watcher, every run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_message_still_queued_at_shutdown_is_delivered_by_the_drain() {
    let mut watcher = start_watcher().await;
    for run in 0..RUNS {
        let started = Instant::now();
        let r = one_run(&mut watcher, run, DEFAULT_SHUTDOWN_DRAIN).await;
        eprintln!(
            "held drain run {run}: received={} outcome={} waited_ms={} hold={:?} total_ms={}",
            r.received,
            r.report.outcome.as_str(),
            r.report.waited.as_millis(),
            r.hold,
            started.elapsed().as_millis()
        );
        assert!(
            r.hold.held_at_stop,
            "run {run}: the final message was queued, unsent, when the stop was accepted: {r:?}"
        );
        assert_eq!(
            r.report.outcome,
            DrainOutcome::Settled,
            "run {run}: the drain must have waited out the window: {r:?}"
        );
        assert!(
            r.report.publishes_in_window >= 1 && r.report.recipients_connected >= 1,
            "run {run}: the final message was queued for the connected watcher: {r:?}"
        );
        assert!(
            r.hold.released && r.hold.written,
            "run {run}: the drain drove the swarm after the stop request, and the \
             connection wrote the message out: {r:?}"
        );
        assert!(
            r.received,
            "run {run}: the final message was lost despite the drain: {r:?}"
        );
    }
}

/// Negative control: the same sequence with the drain disabled — the loop
/// drops the swarm as soon as it is asked to stop, as it did before the drain
/// — loses the final message, every run. Together with the test above this
/// shows that the drain, and nothing else in the harness, is what delivers it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_the_drain_the_same_sequence_loses_the_message() {
    let mut watcher = start_watcher().await;
    for run in 0..RUNS {
        let r = one_run(&mut watcher, run, Duration::ZERO).await;
        eprintln!(
            "held no-drain run {run}: received={} outcome={} waited_ms={} hold={:?}",
            r.received,
            r.report.outcome.as_str(),
            r.report.waited.as_millis(),
            r.hold
        );
        assert!(
            r.hold.held_at_stop,
            "run {run}: the final message was queued, unsent, when the stop was accepted: {r:?}"
        );
        assert_eq!(r.report.outcome, DrainOutcome::Disabled, "run {run}: {r:?}");
        assert!(
            r.report.waited < Duration::from_millis(100),
            "run {run}: a disabled drain does not wait: {r:?}"
        );
        assert!(
            !r.hold.released && !r.hold.written,
            "run {run}: without the drain the swarm is not polled after the stop \
             request, so the message never leaves the connection: {r:?}"
        );
        assert!(
            !r.received,
            "run {run}: with the drain disabled the final message still arrived; the \
             test above would then pass whether or not the drain works: {r:?}"
        );
    }
}
