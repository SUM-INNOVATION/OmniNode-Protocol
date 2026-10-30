//! A node that publishes and then exits must not take its last message with it.
//!
//! `gossipsub` only queues a published message; a connection task writes it
//! out later. A command that broadcast its result and returned at once used to
//! drop the swarm — and then the process — before that happened, so a
//! connected watcher never received the announcement although the publisher
//! had logged it as published.
//!
//! Each run here starts a fresh publisher on its own runtime and thread, the
//! way a CLI command owns its process: it dials one long-lived watcher over
//! loopback QUIC, publishes a final message, shuts down, and then its runtime
//! is torn down with `shutdown_background` — every task, connection and QUIC
//! driver stops at that instant, as at process exit. Only then does the test
//! look for the message, so a pass is the watcher's own receipt of the bytes,
//! not the publisher's log line.
//!
//! Before the final message the publisher sends probes on the same topic until
//! the watcher has received one. That is the state of the reported failure —
//! connected long enough that the publisher knows the watcher subscribes — and
//! it rules out the unrelated loss of publishing to a peer whose subscription
//! has not arrived yet.
//!
//! All nodes run with mDNS off and connect only to what the test dials.

use std::time::{Duration, Instant};

use libp2p::{Multiaddr, PeerId, multiaddr::Protocol};
use omni_net::{
    DEFAULT_SHUTDOWN_DRAIN, DrainOutcome, DrainReport, Interests, OmniNet, OmniNetEvent,
    TOPIC_CONTRIBUTOR_RESULT,
};
use omni_types::config::NetConfig;
use tokio::sync::{mpsc, oneshot};

/// Failure deadline for connecting and warming up; loopback takes
/// milliseconds.
const DEADLINE: Duration = Duration::from_secs(20);

/// How long the watcher keeps looking for the final message after the
/// publisher has gone. Loopback delivery takes milliseconds, and once the
/// publisher's runtime is gone nothing more can arrive; this only absorbs
/// scheduling delay on the watcher's side.
const RECEIPT_WINDOW: Duration = Duration::from_secs(1);

/// Runs of the drain test. The reported exit lost its announcement in five of
/// six runs; twenty deliveries in a row is what "the race is gone" has to mean
/// here.
const RUNS: usize = 20;

/// Upper bound on negative-control runs. Without the drain the final message
/// was lost in every run when this test ran alone and in roughly a third to
/// half of the runs alongside the drain test, so the control stops at the
/// first loss and only a broken control runs this many times.
const CONTROL_RUNS: usize = 40;

const TOPIC: &str = TOPIC_CONTRIBUTOR_RESULT;

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

/// One publisher "process": connect, warm up, publish `final-<run>`, shut
/// down with `drain`, then stop the runtime the way a process exit does.
fn run_publisher(
    watcher: Multiaddr,
    watcher_peer: PeerId,
    run: usize,
    drain: Duration,
    probe_seen: oneshot::Receiver<()>,
) -> DrainReport {
    // `#[tokio::main]`'s default flavour, as the CLI uses.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("publisher runtime");
    let report = rt.block_on(async move {
        let mut node = new_node().await;
        node.dial(watcher).await.expect("dial the watcher");
        wait_connected(&mut node, watcher_peer).await;

        let probes = async {
            for k in 0u64.. {
                node.publish(TOPIC, format!("probe-{run}-{k}").into_bytes())
                    .await
                    .expect("publish probe");
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

        node.publish(TOPIC, format!("final-{run}").into_bytes())
            .await
            .expect("publish the final message");
        node.shutdown_with_drain(drain).await.expect("shutdown")
    });
    // Process exit: nothing of this node runs past this line.
    rt.shutdown_background();
    report
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
            if let OmniNetEvent::MessageReceived { topic, data, .. } = event {
                if topic == TOPIC {
                    let _ = tx.send(String::from_utf8_lossy(&data).into_owned());
                }
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

struct RunResult {
    received: bool,
    report: DrainReport,
}

/// One publisher run against `watcher`.
async fn one_run(watcher: &mut Watcher, run: usize, drain: Duration) -> RunResult {
    let (probe_tx, probe_rx) = oneshot::channel();
    let (addr, peer) = (watcher.addr.clone(), watcher.peer);
    let publisher =
        std::thread::spawn(move || run_publisher(addr, peer, run, drain, probe_rx));

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
    let report = tokio::task::spawn_blocking(move || publisher.join())
        .await
        .expect("join task")
        .expect("publisher thread panicked");

    // The publisher is gone. Whatever the watcher has not received by now
    // was lost with it.
    let final_msg = format!("final-{run}");
    let received = watcher
        .wait_for(RECEIPT_WINDOW, |m| m == final_msg)
        .await;
    RunResult { received, report }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// With the default drain, every run's final message reaches the watcher.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_message_published_just_before_exit_reaches_a_connected_watcher() {
    let mut watcher = start_watcher().await;
    let mut lost = Vec::new();
    for run in 0..RUNS {
        let started = Instant::now();
        let RunResult { received, report } =
            one_run(&mut watcher, run, DEFAULT_SHUTDOWN_DRAIN).await;
        assert_eq!(
            report.outcome,
            DrainOutcome::Settled,
            "run {run}: the drain must have waited out the window: {report:?}"
        );
        assert!(
            report.publishes_in_window >= 1 && report.recipients_connected >= 1,
            "run {run}: the final message was queued for the connected watcher: {report:?}"
        );
        assert!(
            report.waited <= DEFAULT_SHUTDOWN_DRAIN + Duration::from_secs(1),
            "run {run}: the drain is bounded by its grace: {report:?}"
        );
        if !received {
            lost.push(run);
        }
        eprintln!(
            "drain run {run}: received={received} outcome={} waited_ms={} total_ms={}",
            report.outcome.as_str(),
            report.waited.as_millis(),
            started.elapsed().as_millis()
        );
    }
    assert!(
        lost.is_empty(),
        "final message lost in runs {lost:?} of {RUNS} despite the drain"
    );
}

/// Negative control: with the drain disabled — the loop drops the swarm as
/// soon as it is asked to stop, as it did before — the same sequence loses
/// the final message. This is what shows the test above can fail.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_the_drain_the_same_sequence_loses_the_message() {
    let mut watcher = start_watcher().await;
    let mut first_loss = None;
    for run in 0..CONTROL_RUNS {
        let RunResult { received, report } = one_run(&mut watcher, run, Duration::ZERO).await;
        assert_eq!(report.outcome, DrainOutcome::Disabled, "run {run}: {report:?}");
        assert!(
            report.waited < Duration::from_millis(100),
            "run {run}: a disabled drain does not wait: {report:?}"
        );
        eprintln!("no-drain run {run}: received={received}");
        if !received {
            first_loss = Some(run);
            break;
        }
    }
    assert!(
        first_loss.is_some(),
        "with the drain disabled all {CONTROL_RUNS} final messages still arrived; \
         the test above would then pass whether or not the drain works"
    );
}

// ── Drain bounds on a live swarm ──────────────────────────────────────────────

/// Connect `publisher` to `watcher` and publish probes until the watcher has
/// received one, so the publisher has a recent publish queued for it.
async fn connect_and_publish(publisher: &mut OmniNet, watcher: &OmniNet) {
    let mut seen = watcher
        .subscribe(Interests::none().topic(TOPIC))
        .expect("subscribe");
    publisher
        .dial(dialable_addr(watcher).await)
        .await
        .expect("dial");
    wait_connected(publisher, watcher.local_peer_id()).await;
    let received = async {
        while let Some(event) = seen.recv().await {
            if matches!(event, OmniNetEvent::MessageReceived { .. }) {
                return;
            }
        }
        panic!("watcher subscription closed");
    };
    let probes = async {
        for k in 0u64.. {
            publisher
                .publish(TOPIC, format!("probe-{k}").into_bytes())
                .await
                .expect("publish probe");
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    };
    tokio::time::timeout(DEADLINE, async {
        tokio::select! {
            _ = received => {}
            _ = probes => unreachable!("the probe loop does not end"),
        }
    })
    .await
    .expect("the watcher never received a probe");
}

/// A node that published nothing stops at once, however long a grace it is
/// given: the drain is not a sleep.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nothing_published_means_shutdown_does_not_wait() {
    let node = new_node().await;
    let started = Instant::now();
    let report = node
        .shutdown_with_drain(Duration::from_secs(10))
        .await
        .expect("shutdown");
    assert_eq!(report.outcome, DrainOutcome::Idle, "{report:?}");
    assert_eq!(report.publishes_in_window, 0);
    assert!(started.elapsed() < Duration::from_secs(2), "{report:?}");
}

/// When every peer a recent publish was queued for has gone, nothing more can
/// be delivered, and the drain ends without waiting out its grace.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_drain_ends_early_once_no_recipient_is_connected() {
    let mut publisher = new_node().await;
    let watcher = new_node().await;
    connect_and_publish(&mut publisher, &watcher).await;
    watcher.shutdown().await.expect("watcher shutdown");

    // Far longer than the connection takes to close; reaching it would mean
    // the drain ignored the disconnect.
    let grace = Duration::from_secs(25);
    let report = publisher
        .shutdown_with_drain(grace)
        .await
        .expect("publisher shutdown");
    assert_eq!(report.outcome, DrainOutcome::RecipientsGone, "{report:?}");
    assert!(report.publishes_in_window >= 1, "{report:?}");
    assert_eq!(report.recipients_connected, 0, "{report:?}");
    assert!(report.waited < Duration::from_secs(15), "{report:?}");
}

/// A second shutdown while the first is draining waits for the same drain and
/// gets the same answer, instead of failing or ending the drain early.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_concurrent_shutdown_shares_the_drain() {
    let mut publisher = new_node().await;
    let watcher = new_node().await;
    connect_and_publish(&mut publisher, &watcher).await;

    let grace = Duration::from_millis(300);
    let other = publisher.handle();
    let (first, second) = tokio::join!(
        publisher.shutdown_with_drain(grace),
        async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            other.shutdown_with_drain(Duration::ZERO).await
        }
    );
    let first = first.expect("first shutdown");
    let second = second.expect("second shutdown");
    assert_eq!(first.outcome, DrainOutcome::Settled, "{first:?}");
    assert_eq!(first, second);
    // The loop is gone once either call returns.
    assert!(publisher.listen_addrs().await.is_err());
    let _ = watcher.shutdown().await;
}
