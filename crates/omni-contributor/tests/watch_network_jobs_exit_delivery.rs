//! `watch-network-jobs --publish-result-link --max-jobs 1`: the result
//! announcement broadcast for the last job must reach a connected
//! `watch-network-results` peer although the contributor exits right after
//! broadcasting it.
//!
//! The contributor side reproduces the command's own sequence: one OmniNet
//! node, `run_watch_loop` on a blocking thread with `max_jobs = Some(1)` and a
//! broadcaster that publishes a signed `NetworkPostedResultAnnouncement`
//! through `OmniNetRelay::publish_result`, then `shutdown_with_drain` with the
//! command's `--propagation-wait-ms` default, then process exit. The process is
//! emulated by giving each contributor run its own runtime on its own thread
//! and tearing it down with `shutdown_background`, which stops every task,
//! connection and QUIC driver at once.
//!
//! The watcher is a real second node reading the result topic through
//! `OmniNetRelay::poll_results`, the consumer `watch-network-results` uses. A
//! run passes only when that relay hands back the announcement for this run's
//! job, with a verifying announcer signature and the link root the contributor
//! published — receipt, not the contributor's log line.
//!
//! Jobs come from an in-memory relay and SNIP store (as in
//! `watch_network_jobs_integration.rs`), so no SNIP binary is needed. Only the
//! result announcement crosses the network. Synthetic seeds; loopback only;
//! mDNS off.

#![cfg(feature = "network")]

mod common;

use std::time::Duration;

use common::MockSnipStore;
use libp2p::{Multiaddr, PeerId, multiaddr::Protocol};
use omni_contributor::{
    canonical::{
        hex_lower, job_hash_hex, network_job_announcement_signing_input,
        network_result_announcement_signing_input, posted_id_hex,
    },
    posted::POSTED_SCHEMA_VERSION,
    run_watch_loop,
    signing::verify_signature_hex,
    AcceptFilters, BaseUnitRewardPolicy, ContributorJob, ContributorRelay, ContributorSigner,
    CostCaps, DispatcherSigner, EventEmitter, InMemoryRelay, JobAccounting,
    NetworkPostedJobAnnouncement, NetworkPostedResultAnnouncement, NetworkSource, OmniNetRelay,
    PostedJob, PublishedResultLink, ResultBroadcaster, StubRunner, VerificationRequirement,
    WatchEvent, WatchOptions, NET_SCHEMA_VERSION,
};
use omni_net::{
    DrainOutcome, DrainReport, Interests, OmniNet, OmniNetEvent, TOPIC_CONTRIBUTOR_RESULT,
};
use omni_types::config::NetConfig;
use tokio::sync::oneshot;

const CONTRIBUTOR_SEED: [u8; 32] = *b"exit-delivery-contrib-seed-32b!!";
const ANNOUNCER_SEED: [u8; 32] = *b"exit-delivery-announcer-seed-32!";

/// `watch-network-jobs --propagation-wait-ms` default.
const PROPAGATION_WAIT: Duration = Duration::from_millis(200);

const DEADLINE: Duration = Duration::from_secs(20);
/// How long the watcher keeps polling after the contributor has exited.
const RECEIPT_WINDOW: Duration = Duration::from_secs(1);
const RUNS: usize = 20;
/// The old exit path lost the announcement in two of three reported runs and
/// in some runs of this harness, so the opt-in control runs until it sees a
/// loss, up to this many times.
const CONTROL_RUNS: usize = 20;
/// Whole-test bound for the 20-run tests below. A clean run of the gating test
/// took 7.2 s (debug build, macOS arm64, 2026-11-16); past this bound the test
/// binary fails with status 124 instead of hanging.
const TEST_DEADLINE: Duration = Duration::from_secs(90);

// ── Deadlines ─────────────────────────────────────────────────────────────────

/// Fails the process instead of letting it hang. Unless [`Watchdog::disarm`]
/// is called first, it prints what overran and exits with status 124 once
/// `limit` of wall-clock time has passed. Wall clock, so time the machine
/// spends asleep counts: a run that spans a sleep fails rather than passing.
/// A panic does not disarm it, so a test whose runtime then hangs on shutdown
/// (waiting for a stuck blocking task) still ends.
struct Watchdog {
    disarm: Option<std::sync::mpsc::Sender<()>>,
}

impl Watchdog {
    fn arm(what: String, limit: Duration) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let started = std::time::SystemTime::now();
        std::thread::spawn(move || loop {
            match rx.recv_timeout(Duration::from_millis(250)) {
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if started.elapsed().unwrap_or_default() >= limit {
                        eprintln!(
                            "DEADLINE EXCEEDED: {what}: no result within {limit:?} of wall-clock \
                             time; failing instead of hanging"
                        );
                        std::process::exit(124);
                    }
                }
                _ => return,
            }
        });
        Self { disarm: Some(tx) }
    }

    fn disarm(mut self) {
        if let Some(tx) = self.disarm.take() {
            let _ = tx.send(());
        }
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        // Dropped without `disarm` (a panic): stay armed.
        if let Some(tx) = self.disarm.take() {
            std::mem::forget(tx);
        }
    }
}

// ── Network harness ───────────────────────────────────────────────────────────

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
                    return;
                }
            }
        }
        panic!("event stream closed before connecting");
    })
    .await
    .expect("timed out waiting for the dialled connection")
}

// ── Job fixtures (as in watch_network_jobs_integration.rs) ───────────────────

struct CollectEmitter(Vec<WatchEvent>);
impl EventEmitter for CollectEmitter {
    fn emit(&mut self, e: WatchEvent) {
        self.0.push(e);
    }
}

/// A job unique to `run`, announced on the in-memory relay. Returns its
/// `posted_id`. `run` is any label that makes the job unique (the runner
/// roles below use `<session>-<run>`).
fn announce_job(
    snip: &MockSnipStore,
    relay: &mut InMemoryRelay,
    run: impl std::fmt::Display,
) -> String {
    let manifest_bytes = b"exit-delivery-manifest".to_vec();
    let input_bytes = format!("exit-delivery-input-{run}").into_bytes();
    let manifest_root = snip.insert_bytes(&manifest_bytes);
    let input_root = snip.insert_bytes(&input_bytes);
    let mut job = ContributorJob {
        schema_version: 1,
        job_id: String::new(),
        model_hash: hex_lower(blake3::hash(&manifest_bytes).as_bytes()),
        manifest_snip_root: manifest_root.to_hex(),
        input_snip_root: input_root.to_hex(),
        input_hash: hex_lower(blake3::hash(&input_bytes).as_bytes()),
        verification_requirement: VerificationRequirement::AttestationOnly,
        accounting: JobAccounting {
            tokenizer_hash: hex_lower(blake3::hash(b"exit-delivery-tok").as_bytes()),
            tokenizer_id: "net/tok".into(),
            input_token_count: 9,
            max_output_token_count: 75,
            base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
        },
        dispatched_at_utc: "2026-05-26T00:00:00Z".into(),
        expires_at_utc: None,
        dispatcher_pubkey_hex: None,
        dispatcher_signature_hex: None,
        notes: None,
    };
    job.job_id = job_hash_hex(&job).unwrap();

    let job_root = snip.insert_bytes(serde_json::to_string_pretty(&job).unwrap().as_bytes());
    let mut posted = PostedJob {
        schema_version: POSTED_SCHEMA_VERSION,
        posted_id: String::new(),
        job_snip_root: job_root.to_hex(),
        job_hash: job_hash_hex(&job).unwrap(),
        model_hash: job.model_hash.clone(),
        posted_at_utc: "2026-05-26T00:00:01Z".into(),
        expires_at_utc: None,
        poster_pubkey_hex: None,
        poster_signature_hex: None,
        notes: None,
    };
    posted.posted_id = posted_id_hex(&posted).unwrap();
    let posted_root =
        snip.insert_bytes(serde_json::to_string_pretty(&posted).unwrap().as_bytes());

    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let mut ann = NetworkPostedJobAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        posted_job_snip_root: posted_root.to_hex(),
        posted_id: posted.posted_id.clone(),
        job_hash: posted.job_hash.clone(),
        model_hash: posted.model_hash.clone(),
        tokenizer_hash: None,
        announced_at_utc: "2026-05-26T00:00:02Z".into(),
        announcer_pubkey_hex: announcer.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    ann.announcer_signature_hex =
        announcer.sign_hex(&network_job_announcement_signing_input(&ann).unwrap());
    relay.publish_job(&ann).unwrap();
    posted.posted_id
}

/// The command's broadcaster: sign a result announcement for the published
/// link and publish it on the mesh.
struct NetResultBroadcaster {
    relay: OmniNetRelay,
    signer: ContributorSigner,
}

impl ResultBroadcaster for NetResultBroadcaster {
    fn broadcast(&mut self, published: &PublishedResultLink) -> Result<(), String> {
        let mut ann = NetworkPostedResultAnnouncement {
            schema_version: NET_SCHEMA_VERSION,
            posted_id: published.link.posted_id.clone(),
            posted_result_link_snip_root: format!(
                "0x{}",
                hex_lower(published.link_snip_root.as_bytes())
            ),
            result_canonical_hash: published.link.result_canonical_hash.clone(),
            contributor_pubkey_hex: published.link.contributor_pubkey_hex.clone(),
            announced_at_utc: "2026-05-26T00:00:42Z".into(),
            announcer_pubkey_hex: self.signer.pubkey_hex(),
            announcer_signature_hex: String::new(),
        };
        let signing_input =
            network_result_announcement_signing_input(&ann).map_err(|e| e.to_string())?;
        ann.announcer_signature_hex = self.signer.sign_hex(&signing_input);
        ann.validate_schema().map_err(|e| e.to_string())?;
        self.relay.publish_result(&ann).map_err(|e| e.to_string())
    }
}

/// The `watch-network-jobs --publish-result-link --max-jobs 1` loop: take the
/// job announced on `jobs`, run it, publish its result link and broadcast the
/// result announcement on `net`, then return on `max_jobs`. Blocking; call it
/// from a blocking thread. Returns the loop's events.
fn watch_one_job(
    snip: &MockSnipStore,
    jobs: &mut InMemoryRelay,
    net: omni_net::NetHandle,
    handle: tokio::runtime::Handle,
) -> Vec<WatchEvent> {
    let out = tempfile::tempdir().unwrap();
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = StubRunner::new(signer.pubkey_hex(), b"exit-delivery-resp".to_vec(), 9, 11);
    let mut source = NetworkSource::new(jobs, snip);
    let mut emitter = CollectEmitter(Vec::new());
    let mut broadcaster = NetResultBroadcaster {
        relay: OmniNetRelay::new(net, handle),
        signer: ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap(),
    };
    let opts = WatchOptions {
        poll_interval: Duration::ZERO,
        max_jobs: Some(1),
        max_polls: Some(5),
        filters: AcceptFilters::default(),
        caps: CostCaps {
            max_input_tokens: 1_000_000,
            max_output_tokens: 1_000_000,
            max_total_base_units: 2_000_000,
        },
        runner: &runner,
        signer: &signer,
        result_out_dir: out.path().to_path_buf(),
        publish_link: true,
        emit: &mut emitter,
        result_broadcaster: Some(&mut broadcaster),
        state_store: None,
    };
    run_watch_loop(snip, &mut source, opts).expect("watch loop");
    emitter.0
}

// ── One contributor "process" ─────────────────────────────────────────────────

struct ContributorRun {
    posted_id: String,
    link_snip_root: String,
    events: Vec<WatchEvent>,
    drain: DrainReport,
}

fn run_contributor(
    watcher: Multiaddr,
    watcher_peer: PeerId,
    run: usize,
    propagation_wait: Duration,
    probe_seen: oneshot::Receiver<()>,
) -> ContributorRun {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("contributor runtime");
    let outcome = rt.block_on(async move {
        let mut node = new_node().await;
        node.dial(watcher).await.expect("dial the watcher");
        wait_connected(&mut node, watcher_peer).await;
        // Until the watcher has received one of these, this node may not yet
        // know the watcher subscribes to the result topic. The watcher's
        // relay ignores them: they are not announcements.
        let probes = async {
            for k in 0u64.. {
                node.publish(TOPIC_CONTRIBUTOR_RESULT, format!("probe-{run}-{k}").into_bytes())
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

        // From here on, the `watch-network-jobs` sequence.
        let net = node.handle();
        let handle = tokio::runtime::Handle::current();
        let broadcaster_net = net.clone();
        let (posted_id, events) = tokio::task::spawn_blocking(move || {
            let snip = MockSnipStore::new();
            let mut jobs = InMemoryRelay::new();
            let posted_id = announce_job(&snip, &mut jobs, run);
            let events = watch_one_job(&snip, &mut jobs, broadcaster_net, handle);
            (posted_id, events)
        })
        .await
        .expect("watch loop task");

        let drain = net
            .shutdown_with_drain(propagation_wait)
            .await
            .expect("shutdown");
        (posted_id, events, drain)
    });
    // Process exit.
    rt.shutdown_background();

    let (posted_id, events, drain) = outcome;
    let link_snip_root = events
        .iter()
        .find_map(|e| match e {
            WatchEvent::ResultLinkPublished { link_snip_root, .. } => Some(link_snip_root.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("run {run}: no result link published: {events:?}"));
    ContributorRun {
        posted_id,
        link_snip_root,
        events,
        drain,
    }
}

// ── Watcher ───────────────────────────────────────────────────────────────────

struct Watcher {
    addr: Multiaddr,
    peer: PeerId,
    relay: OmniNetRelay,
    probes: omni_net::Subscription,
    _node: OmniNet,
}

impl Watcher {
    async fn start() -> Self {
        let node = new_node().await;
        let addr = dialable_addr(&node).await;
        let peer = node.local_peer_id();
        let relay = OmniNetRelay::new(node.handle(), tokio::runtime::Handle::current());
        let probes = node
            .subscribe(Interests::none().topic(TOPIC_CONTRIBUTOR_RESULT))
            .expect("subscribe");
        Self {
            addr,
            peer,
            relay,
            probes,
            _node: node,
        }
    }

    async fn wait_for_probe(&mut self, run: usize) {
        let prefix = format!("probe-{run}-");
        tokio::time::timeout(DEADLINE, async {
            while let Some(event) = self.probes.recv().await {
                if let OmniNetEvent::MessageReceived { data, .. } = event {
                    if data.starts_with(prefix.as_bytes()) {
                        return;
                    }
                }
            }
            panic!("watcher subscription closed");
        })
        .await
        .unwrap_or_else(|_| panic!("run {run}: no probe reached the watcher"))
    }

    /// Poll the relay, as `watch-network-results` does, for the announcement
    /// of `posted_id`.
    async fn wait_for_result(&mut self, posted_id: &str) -> Option<NetworkPostedResultAnnouncement> {
        let until = tokio::time::Instant::now() + RECEIPT_WINDOW;
        loop {
            for ann in self.relay.poll_results().expect("poll_results") {
                if ann.posted_id == posted_id {
                    return Some(ann);
                }
            }
            if tokio::time::Instant::now() >= until {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

async fn one_run(
    watcher: &mut Watcher,
    run: usize,
    propagation_wait: Duration,
) -> (ContributorRun, Option<NetworkPostedResultAnnouncement>) {
    let (probe_tx, probe_rx) = oneshot::channel();
    let (addr, peer) = (watcher.addr.clone(), watcher.peer);
    let contributor = std::thread::spawn(move || {
        run_contributor(addr, peer, run, propagation_wait, probe_rx)
    });
    watcher.wait_for_probe(run).await;
    let _ = probe_tx.send(());
    let contributor = tokio::task::spawn_blocking(move || contributor.join())
        .await
        .expect("join task")
        .expect("contributor thread panicked");
    // The contributor has exited; look for its announcement only now.
    let received = watcher.wait_for_result(&contributor.posted_id).await;
    (contributor, received)
}

fn exited_on_max_jobs(events: &[WatchEvent]) -> bool {
    matches!(
        events.last(),
        Some(WatchEvent::Exit { reason, jobs_picked: 1 }) if reason == "max_jobs_reached"
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn result_announcement_reaches_the_watcher_when_the_contributor_exits_on_max_jobs() {
    let watchdog = Watchdog::arm("result_announcement_reaches_the_watcher…".into(), TEST_DEADLINE);
    every_run_delivers().await;
    watchdog.disarm();
}

async fn every_run_delivers() {
    let mut watcher = Watcher::start().await;
    let mut lost = Vec::new();
    for run in 0..RUNS {
        let (contributor, received) = one_run(&mut watcher, run, PROPAGATION_WAIT).await;
        assert!(
            exited_on_max_jobs(&contributor.events),
            "run {run}: the loop must have returned on max_jobs straight after broadcasting: {:?}",
            contributor.events
        );
        assert_eq!(
            contributor.drain.outcome,
            DrainOutcome::Settled,
            "run {run}: {:?}",
            contributor.drain
        );
        let Some(ann) = received else {
            lost.push(run);
            continue;
        };
        // What arrived is this run's announcement, intact.
        assert_eq!(ann.posted_result_link_snip_root, contributor.link_snip_root, "run {run}");
        let signing_input = network_result_announcement_signing_input(&ann).unwrap();
        assert!(
            verify_signature_hex(
                &ann.announcer_pubkey_hex,
                &signing_input,
                &ann.announcer_signature_hex
            )
            .unwrap(),
            "run {run}: announcer signature must verify"
        );
    }
    assert!(
        lost.is_empty(),
        "result announcement lost in runs {lost:?} of {RUNS}"
    );
}

/// Negative control: `--propagation-wait-ms 0` is the old exit path — the
/// node is dropped as soon as the loop returns — and loses the announcement.
///
/// Opt-in (`cargo test … -- --ignored`): in this end-to-end harness the old
/// path loses the announcement only sometimes, so the control cannot gate CI.
/// On a GitHub runner all 20 announcements arrived without the wait in one
/// attempt of post-merge run 40993101696 and one was lost in the next, and run
/// on its own on macOS it saw no loss in three runs of 20; the drain test above
/// passed every time. That the drain itself is what delivers the last message
/// is shown by `omni-net`'s `without_the_drain_the_same_sequence_loses_the_message`,
/// which stays in CI.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "end-to-end negative control that does not reproduce the loss reliably; run with --ignored"]
async fn with_no_propagation_wait_the_announcement_is_lost() {
    let watchdog = Watchdog::arm("with_no_propagation_wait…".into(), TEST_DEADLINE);
    a_run_loses_its_announcement().await;
    watchdog.disarm();
}

async fn a_run_loses_its_announcement() {
    let mut watcher = Watcher::start().await;
    let mut first_loss = None;
    for run in 0..CONTROL_RUNS {
        let (contributor, received) = one_run(&mut watcher, run, Duration::ZERO).await;
        assert!(exited_on_max_jobs(&contributor.events), "run {run}");
        assert_eq!(contributor.drain.outcome, DrainOutcome::Disabled, "run {run}");
        if received.is_none() {
            first_loss = Some(run);
            break;
        }
    }
    eprintln!("no propagation wait: first loss at run {first_loss:?}");
    assert!(
        first_loss.is_some(),
        "all {CONTROL_RUNS} announcements arrived without a propagation wait; the \
         test above would pass whether or not the wait works"
    );
}

// ── Acceptance runner roles (issue #150) ─────────────────────────────────────
//
// The tests above run both sides in one process, on loopback. The ignored
// tests in `runner` split the same sequence into separate OS processes, so
// that the contributor's exit is a real process exit and the two sides can run
// on separate hosts, optionally through an emulated slow link:
//
// * `runner::watcher` — the receiving side. It reads the result topic through
//   `OmniNetRelay::poll_results`, the consumer `watch-network-results` uses,
//   records for every contributor run whether that run's announcement arrived,
//   and decides the verdict.
// * `runner::contributor` — ONE contributor run per process: connect to the
//   watcher(s), handshake, then the `watch-network-jobs --publish-result-link
//   --max-jobs 1` sequence (`watch_one_job`), the drain, and process exit.
// * `runner::link_proxy` — a userspace UDP forwarder adding one-way delay,
//   jitter and loss in both directions: an emulated slow link without root.
//
// `scripts/acceptance/result-delivery.sh` drives them, and
// `docs/acceptance-result-delivery.md` documents the parameters, the records
// and the pass criteria. Every endpoint comes from an `OMNI_DELIVERY_*`
// environment variable. Without its required variable a role returns at once,
// so an `--ignored` run of this file does not hang on them.

mod runner {
    use super::*;

    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::io::Write;
    use std::net::{Ipv4Addr, SocketAddr};
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use omni_net::TOPIC_TEST;
    use omni_types::config::NetIdentity;
    use serde_json::json;
    use tokio::net::UdpSocket;

    // ── Environment (defaults: see the script and the doc) ──
    const RECORD: &str = "OMNI_DELIVERY_RECORD";
    const SESSION: &str = "OMNI_DELIVERY_SESSION";
    const RUNS_EXPECTED: &str = "OMNI_DELIVERY_RUNS";
    const RUN: &str = "OMNI_DELIVERY_RUN";
    const LISTEN_PORT: &str = "OMNI_DELIVERY_LISTEN_PORT";
    const ADVERTISE_IP: &str = "OMNI_DELIVERY_ADVERTISE_IP";
    const ADDR_FILE: &str = "OMNI_DELIVERY_ADDR_FILE";
    const IDENTITY_FILE: &str = "OMNI_DELIVERY_IDENTITY_FILE";
    const MDNS: &str = "OMNI_DELIVERY_MDNS";
    const OUTBOUND_ALLOW: &str = "OMNI_DELIVERY_OUTBOUND_ALLOW";
    const WATCHER_ADDRS: &str = "OMNI_DELIVERY_WATCHER_ADDRS";
    const PROPAGATION_WAIT_MS: &str = "OMNI_DELIVERY_PROPAGATION_WAIT_MS";
    const CONNECT_TIMEOUT_SECS: &str = "OMNI_DELIVERY_CONNECT_TIMEOUT_SECS";
    const SETTLE_MS: &str = "OMNI_DELIVERY_SETTLE_MS";
    const PROBE_INTERVAL_MS: &str = "OMNI_DELIVERY_PROBE_INTERVAL_MS";
    const RECEIPT_TIMEOUT_MS: &str = "OMNI_DELIVERY_RECEIPT_TIMEOUT_MS";
    const MAX_RUN_SECS: &str = "OMNI_DELIVERY_MAX_RUN_SECS";
    const IDLE_TIMEOUT_SECS: &str = "OMNI_DELIVERY_IDLE_TIMEOUT_SECS";
    const EVIDENCE_CLASS: &str = "OMNI_DELIVERY_EVIDENCE_CLASS";
    const PROXY_LISTEN: &str = "OMNI_DELIVERY_PROXY_LISTEN";
    const PROXY_UPSTREAM: &str = "OMNI_DELIVERY_PROXY_UPSTREAM";
    const PROXY_ADDR_FILE: &str = "OMNI_DELIVERY_PROXY_ADDR_FILE";
    const PROXY_STATS_FILE: &str = "OMNI_DELIVERY_PROXY_STATS_FILE";
    const PROXY_LIFETIME_SECS: &str = "OMNI_DELIVERY_PROXY_LIFETIME_SECS";
    const LINK_DELAY_MS: &str = "OMNI_DELIVERY_LINK_DELAY_MS";
    const LINK_JITTER_MS: &str = "OMNI_DELIVERY_LINK_JITTER_MS";
    const LINK_LOSS_PCT: &str = "OMNI_DELIVERY_LINK_LOSS_PCT";
    const LINK_SEED: &str = "OMNI_DELIVERY_LINK_SEED";
    const DEADLINE_SECS: &str = "OMNI_DELIVERY_DEADLINE_SECS";

    /// Handshake payloads. A probe goes out on the result topic, so its
    /// arrival shows the result path to the watcher works; the watcher's relay
    /// ignores it (it is not an announcement). The ack comes back on
    /// `omni/test/v1`.
    const PROBE_TAG: &str = "omni-delivery-probe/v1";
    const ACK_TAG: &str = "omni-delivery-ack/v1";

    const UNATTESTED: &str =
        "unattested: the operator must state whether the hosts and the link were real";

    fn env_str(name: &str) -> Option<String> {
        std::env::var(name)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    }

    fn env_or<T>(name: &str, default: T) -> T
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        match env_str(name) {
            None => default,
            Some(v) => v.parse().unwrap_or_else(|e| panic!("{name}={v}: {e}")),
        }
    }

    fn env_ms(name: &str, default: u64) -> Duration {
        Duration::from_millis(env_or(name, default))
    }

    fn now_utc() -> String {
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
    }

    fn ms(d: Duration) -> u64 {
        d.as_millis() as u64
    }

    /// An append-only JSON-lines record.
    struct Record(std::fs::File);

    impl Record {
        fn open(path: &str) -> Self {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .unwrap_or_else(|e| panic!("open record {path}: {e}"));
            Self(f)
        }

        fn write(&mut self, v: serde_json::Value) {
            writeln!(self.0, "{v}").expect("write record");
            self.0.flush().expect("flush record");
        }
    }

    /// Write `contents` to `path` so that a reader polling for the file never
    /// sees it half-written.
    fn write_atomically(path: &str, contents: &str) {
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, contents).unwrap_or_else(|e| panic!("write {tmp}: {e}"));
        std::fs::rename(&tmp, path).unwrap_or_else(|e| panic!("rename {tmp}: {e}"));
    }

    /// An OmniNet node from the environment: listen port (default 0), an
    /// optional persistent identity, mDNS (default off) and an optional
    /// outbound allow-list.
    async fn node_from_env() -> OmniNet {
        let identity = match env_str(IDENTITY_FILE) {
            None => NetIdentity::Ephemeral,
            Some(p) => NetIdentity::KeypairProtobufBytes(
                omni_net::load_or_create_keypair_file_bytes(Path::new(&p))
                    .unwrap_or_else(|e| panic!("{IDENTITY_FILE}={p}: {e}")),
            ),
        };
        let mdns = match env_str(MDNS).as_deref() {
            None | Some("off") => false,
            Some("on") => true,
            Some(other) => panic!("{MDNS}={other}: expected on or off"),
        };
        let outbound_allow = env_str(OUTBOUND_ALLOW)
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        OmniNet::new(NetConfig {
            listen_port: env_or(LISTEN_PORT, 0u16),
            mdns,
            outbound_allow,
            identity,
            ..NetConfig::default()
        })
        .await
        .expect("OmniNet::new")
    }

    async fn udp_listen_port(net: &OmniNet) -> u16 {
        let addrs = tokio::time::timeout(DEADLINE, net.listen_addrs())
            .await
            .expect("timed out waiting for a listen address")
            .expect("the swarm reports its listen addresses");
        addrs
            .iter()
            .find_map(|a| {
                a.iter().find_map(|p| match p {
                    Protocol::Udp(port) => Some(port),
                    _ => None,
                })
            })
            .expect("a QUIC listen address carries a UDP port")
    }

    fn peer_of(addr: &Multiaddr) -> Option<PeerId> {
        addr.iter().find_map(|p| match p {
            Protocol::P2p(id) => Some(id),
            _ => None,
        })
    }

    /// `<tag> <session> <run> <k> <last>` → `(session, run, k, last)`.
    fn parse_handshake(tag: &str, data: &[u8]) -> Option<(String, u64, u64, String)> {
        let text = std::str::from_utf8(data).ok()?;
        let mut it = text.split(' ');
        if it.next()? != tag {
            return None;
        }
        let session = it.next()?.to_string();
        let run = it.next()?.parse().ok()?;
        let k = it.next()?.parse().ok()?;
        let last = it.next()?.to_string();
        if it.next().is_some() {
            return None;
        }
        Some((session, run, k, last))
    }

    // ── watcher ──────────────────────────────────────────────────────────────

    struct RunState {
        session: String,
        run: u64,
        posted_id: String,
        contributor: PeerId,
        first_probe: Instant,
        disconnected: Option<Instant>,
        /// `delivered`, `bad_signature` or `lost`, once decided.
        outcome: Option<&'static str>,
        received_before_disconnect: Option<bool>,
    }

    /// The receiving side. Records every run it sees and fails unless
    /// `OMNI_DELIVERY_RUNS` runs were seen and each one's announcement arrived
    /// intact within the receipt timeout.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "issue #150 acceptance runner role; run by scripts/acceptance/result-delivery.sh"]
    async fn watcher() {
        let Some(record_path) = env_str(RECORD) else {
            eprintln!("runner::watcher: {RECORD} is not set; nothing to do");
            return;
        };
        let runs: u64 = env_or(RUNS_EXPECTED, 20);
        // Default bound: 30 s per run (a clean loopback run takes about 1.5 s) plus
        // 60 s; the script passes one derived from its own parameters.
        let watchdog = Watchdog::arm(
            "runner::watcher".into(),
            Duration::from_secs(env_or(DEADLINE_SECS, runs * 30 + 60)),
        );
        let receipt_timeout = env_ms(RECEIPT_TIMEOUT_MS, 10_000);
        let max_run = Duration::from_secs(env_or(MAX_RUN_SECS, 120));
        let idle_timeout = Duration::from_secs(env_or(IDLE_TIMEOUT_SECS, 120));
        let evidence = env_str(EVIDENCE_CLASS).unwrap_or_else(|| UNATTESTED.into());
        let advertise_ip: Ipv4Addr = env_or(ADVERTISE_IP, Ipv4Addr::LOCALHOST);

        let node = node_from_env().await;
        let me = node.local_peer_id();
        let port = udp_listen_port(&node).await;
        let addr = format!("/ip4/{advertise_ip}/udp/{port}/quic-v1/p2p/{me}");
        let mut relay = OmniNetRelay::new(node.handle(), tokio::runtime::Handle::current());
        let mut events = node
            .subscribe(Interests::none().topic(TOPIC_CONTRIBUTOR_RESULT).control())
            .expect("subscribe");

        let mut rec = Record::open(&record_path);
        rec.write(json!({
            "kind": "watcher_start", "at_utc": now_utc(), "peer_id": me.to_string(),
            "addr": addr, "runs_expected": runs,
            "receipt_timeout_ms": ms(receipt_timeout), "max_run_secs": max_run.as_secs(),
            "idle_timeout_secs": idle_timeout.as_secs(), "evidence_class": evidence,
            "policy": node.network_policy().describe(),
        }));
        println!("delivery-runner role=watcher peer_id={me} addr={addr}");
        if let Some(f) = env_str(ADDR_FILE) {
            write_atomically(&f, &format!("{addr}\n"));
        }

        let mut by_key: BTreeMap<(String, u64), RunState> = BTreeMap::new();
        let mut by_posted: HashMap<String, (String, u64)> = HashMap::new();
        let (mut late, mut duplicates, mut unmatched) = (0u64, 0u64, 0u64);
        let mut last_activity = Instant::now();
        let ended_by;

        loop {
            // 1. Probes (answered with an ack) and peer lifecycle.
            let tick = tokio::time::sleep(Duration::from_millis(10));
            tokio::pin!(tick);
            loop {
                let event = tokio::select! {
                    e = events.recv() => e.expect("watcher subscription closed"),
                    _ = &mut tick => break,
                };
                match event {
                    OmniNetEvent::MessageReceived { data, from, .. } => {
                        let Some((session, run, k, posted_id)) =
                            parse_handshake(PROBE_TAG, &data)
                        else {
                            continue;
                        };
                        last_activity = Instant::now();
                        let key = (session.clone(), run);
                        if !by_key.contains_key(&key) {
                            rec.write(json!({
                                "kind": "probe", "at_utc": now_utc(), "session": session,
                                "run": run, "posted_id": posted_id,
                                "contributor": from.to_string(),
                            }));
                            println!("delivery-runner watcher: run {run} probing from {from}");
                            by_posted.insert(posted_id.clone(), key.clone());
                            by_key.insert(key, RunState {
                                session: session.clone(),
                                run,
                                posted_id,
                                contributor: from,
                                first_probe: Instant::now(),
                                disconnected: None,
                                outcome: None,
                                received_before_disconnect: None,
                            });
                        }
                        let ack = format!("{ACK_TAG} {session} {run} {k} {me}");
                        let _ = node.publish(TOPIC_TEST, ack.into_bytes()).await;
                    }
                    OmniNetEvent::PeerDisconnected { peer_id } => {
                        for st in by_key.values_mut() {
                            if st.contributor == peer_id && st.disconnected.is_none() {
                                st.disconnected = Some(Instant::now());
                                if st.received_before_disconnect.is_none() {
                                    st.received_before_disconnect = Some(false);
                                }
                                rec.write(json!({
                                    "kind": "contributor_disconnected", "at_utc": now_utc(),
                                    "session": st.session, "run": st.run,
                                    "ms_since_first_probe": ms(st.first_probe.elapsed()),
                                }));
                            }
                        }
                    }
                    _ => {}
                }
            }

            // 2. Result announcements, through `watch-network-results`'s consumer.
            for ann in relay.poll_results().expect("poll_results") {
                last_activity = Instant::now();
                let Some(key) = by_posted.get(&ann.posted_id).cloned() else {
                    unmatched += 1;
                    rec.write(json!({
                        "kind": "unmatched_announcement", "at_utc": now_utc(),
                        "posted_id": ann.posted_id,
                    }));
                    continue;
                };
                let st = by_key.get_mut(&key).expect("indexed run");
                let signature_ok = network_result_announcement_signing_input(&ann)
                    .ok()
                    .and_then(|input| {
                        verify_signature_hex(
                            &ann.announcer_pubkey_hex,
                            &input,
                            &ann.announcer_signature_hex,
                        )
                        .ok()
                    })
                    .unwrap_or(false);
                match st.outcome {
                    Some("lost") => {
                        late += 1;
                        rec.write(json!({
                            "kind": "late_receipt", "at_utc": now_utc(), "session": st.session,
                            "run": st.run, "posted_id": ann.posted_id,
                            "ms_since_first_probe": ms(st.first_probe.elapsed()),
                        }));
                    }
                    Some(_) => duplicates += 1,
                    None => {
                        let outcome = if signature_ok { "delivered" } else { "bad_signature" };
                        st.outcome = Some(outcome);
                        if st.disconnected.is_none() {
                            st.received_before_disconnect = Some(true);
                        }
                        rec.write(json!({
                            "kind": "receipt", "at_utc": now_utc(), "session": st.session,
                            "run": st.run, "outcome": outcome, "posted_id": ann.posted_id,
                            "link_snip_root": ann.posted_result_link_snip_root,
                            "announcer_signature_ok": signature_ok,
                            "ms_since_first_probe": ms(st.first_probe.elapsed()),
                            "contributor_disconnected_before_receipt": st.disconnected.is_some(),
                            "ms_after_contributor_disconnect": st.disconnected.map(|d| ms(d.elapsed())),
                        }));
                        println!("delivery-runner watcher: run {} {outcome}", st.run);
                    }
                }
            }

            // 3. Runs whose announcement did not come in time.
            for st in by_key.values_mut().filter(|s| s.outcome.is_none()) {
                let (expired, reason) = match st.disconnected {
                    Some(d) => (
                        d.elapsed() >= receipt_timeout,
                        "no receipt within the receipt timeout after the contributor disconnected",
                    ),
                    None => (
                        st.first_probe.elapsed() >= max_run,
                        "no receipt and no disconnect within the maximum run time",
                    ),
                };
                if expired {
                    st.outcome = Some("lost");
                    rec.write(json!({
                        "kind": "lost", "at_utc": now_utc(), "session": st.session,
                        "run": st.run, "posted_id": st.posted_id, "reason": reason,
                    }));
                    println!("delivery-runner watcher: run {} LOST ({reason})", st.run);
                }
            }

            // 4. The end.
            let resolved = by_key.values().filter(|s| s.outcome.is_some()).count() as u64;
            if resolved >= runs {
                ended_by = "all_runs_resolved";
                break;
            }
            if last_activity.elapsed() >= idle_timeout {
                ended_by = "idle_timeout";
                break;
            }
        }

        // Stay up until the last contributor has gone (up to 10 s), so the
        // watcher leaving does not end that contributor's drain early.
        let linger_until = Instant::now() + Duration::from_secs(10);
        while by_key.values().any(|s| s.disconnected.is_none()) && Instant::now() < linger_until {
            if let Ok(Some(OmniNetEvent::PeerDisconnected { peer_id })) =
                tokio::time::timeout(Duration::from_millis(50), events.recv()).await
            {
                for st in by_key.values_mut().filter(|s| s.contributor == peer_id) {
                    st.disconnected.get_or_insert_with(Instant::now);
                }
            }
        }

        let count = |o: &str| by_key.values().filter(|s| s.outcome == Some(o)).count() as u64;
        let (delivered, lost, bad_signature) =
            (count("delivered"), count("lost"), count("bad_signature"));
        let seen = by_key.len() as u64;
        let before_disconnect = by_key
            .values()
            .filter(|s| s.received_before_disconnect == Some(true))
            .count();
        let lost_runs: Vec<u64> = by_key
            .values()
            .filter(|s| s.outcome != Some("delivered"))
            .map(|s| s.run)
            .collect();
        let verdict = if seen == runs && delivered == runs { "PASS" } else { "FAIL" };
        let summary = json!({
            "kind": "summary", "at_utc": now_utc(), "verdict": verdict,
            "runs_expected": runs, "runs_seen": seen, "delivered": delivered, "lost": lost,
            "bad_signature": bad_signature, "late_receipts": late,
            "duplicate_receipts": duplicates, "unmatched_announcements": unmatched,
            "received_before_contributor_disconnect": before_disconnect,
            "runs_not_delivered": lost_runs, "ended_by": ended_by,
            "evidence_class": evidence,
        });
        rec.write(summary.clone());
        println!(
            "delivery-runner verdict={verdict} delivered={delivered}/{runs} lost={lost} \
             bad_signature={bad_signature} late={late} runs_seen={seen} ended_by={ended_by}"
        );
        let _ = node.shutdown().await;
        watchdog.disarm();
        assert_eq!(verdict, "PASS", "{summary}");
    }

    // ── contributor ──────────────────────────────────────────────────────────

    /// One contributor run, then process exit. Records how the run went; the
    /// watcher decides whether its announcement was delivered.
    #[test]
    #[ignore = "issue #150 acceptance runner role; run by scripts/acceptance/result-delivery.sh"]
    fn contributor() {
        let Some(watchers) = env_str(WATCHER_ADDRS) else {
            eprintln!("runner::contributor: {WATCHER_ADDRS} is not set; nothing to do");
            return;
        };
        // Default bound: 60 s for one run (a clean loopback run takes about 1.5 s,
        // one through a 150 ms emulated link about 3.6 s).
        let watchdog = Watchdog::arm(
            "runner::contributor".into(),
            Duration::from_secs(env_or(DEADLINE_SECS, 60)),
        );
        let record_path = env_str(RECORD).unwrap_or_else(|| panic!("{RECORD} must be set"));
        let session = env_str(SESSION).unwrap_or_else(|| "adhoc".into());
        assert!(
            !session.contains(char::is_whitespace),
            "{SESSION} must not contain whitespace"
        );
        let run: u64 = env_or(RUN, 0);
        let propagation_wait = env_ms(PROPAGATION_WAIT_MS, 200);
        let connect_timeout = Duration::from_secs(env_or(CONNECT_TIMEOUT_SECS, 60));
        let settle = env_ms(SETTLE_MS, 0);
        let probe_interval = env_ms(PROBE_INTERVAL_MS, 100);
        let watcher_addrs: Vec<Multiaddr> = watchers
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap_or_else(|e| panic!("{WATCHER_ADDRS}: {s}: {e}")))
            .collect();
        let expected: BTreeSet<PeerId> = watcher_addrs
            .iter()
            .map(|a| {
                peer_of(a).unwrap_or_else(|| panic!("{WATCHER_ADDRS}: {a} has no /p2p/<peer-id>"))
            })
            .collect();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("contributor runtime");
        let session_in = session.clone();
        let outcome = rt.block_on(async move {
            let session = session_in;
            let started = Instant::now();
            let mut node = node_from_env().await;
            let me = node.local_peer_id();
            let policy = node.network_policy().describe();
            let snip = MockSnipStore::new();
            let mut jobs = InMemoryRelay::new();
            let posted_id = announce_job(&snip, &mut jobs, format!("{session}-{run}"));
            println!("delivery-runner role=contributor run={run} peer_id={me} posted_id={posted_id}");

            for a in &watcher_addrs {
                node.dial(a.clone())
                    .await
                    .unwrap_or_else(|e| panic!("run {run}: dial {a}: {e}"));
            }
            let mut connected = BTreeSet::new();
            tokio::time::timeout(connect_timeout, async {
                while connected.len() < expected.len() {
                    match node.next_event().await {
                        Some(OmniNetEvent::PeerConnected { peer_id }) if expected.contains(&peer_id) => {
                            connected.insert(peer_id);
                        }
                        Some(_) => {}
                        None => panic!("event stream closed"),
                    }
                }
            })
            .await
            .unwrap_or_else(|_| panic!("run {run}: not connected to every watcher in {connect_timeout:?}"));
            let connected_ms = ms(started.elapsed());

            // Handshake: probe on the result topic until every watcher acks.
            let net = node.handle();
            let mut acked: BTreeSet<PeerId> = BTreeSet::new();
            let mut sent: BTreeMap<u64, Instant> = BTreeMap::new();
            let mut ack_rtt_ms: Option<u64> = None;
            let mut probes = 0u64;
            let mut next_probe = tokio::time::Instant::now();
            tokio::time::timeout(connect_timeout, async {
                while acked.len() < expected.len() {
                    tokio::select! {
                        _ = tokio::time::sleep_until(next_probe) => {
                            let probe = format!("{PROBE_TAG} {session} {run} {probes} {posted_id}");
                            let _ = net.publish(TOPIC_CONTRIBUTOR_RESULT, probe.into_bytes()).await;
                            sent.insert(probes, Instant::now());
                            probes += 1;
                            next_probe += probe_interval;
                        }
                        event = node.next_event() => match event {
                            Some(OmniNetEvent::MessageReceived { data, .. }) => {
                                let Some((s, r, k, watcher)) = parse_handshake(ACK_TAG, &data) else {
                                    continue;
                                };
                                if s != session || r != run {
                                    continue;
                                }
                                if let Some(peer) = expected.iter().find(|p| p.to_string() == watcher) {
                                    if acked.insert(*peer) && ack_rtt_ms.is_none() {
                                        ack_rtt_ms = sent.get(&k).map(|t| ms(t.elapsed()));
                                    }
                                }
                            }
                            Some(_) => {}
                            None => panic!("event stream closed"),
                        },
                    }
                }
            })
            .await
            .unwrap_or_else(|_| panic!("run {run}: not every watcher acked a probe in {connect_timeout:?}"));
            if !settle.is_zero() {
                tokio::time::sleep(settle).await;
            }

            // From here on, the `watch-network-jobs` sequence, then exit.
            let handle = tokio::runtime::Handle::current();
            let broadcaster_net = net.clone();
            let loop_started_utc = now_utc();
            let events = tokio::task::spawn_blocking(move || {
                let mut jobs = jobs;
                watch_one_job(&snip, &mut jobs, broadcaster_net, handle)
            })
            .await
            .expect("watch loop task");
            let loop_returned_utc = now_utc();
            let drain = net
                .shutdown_with_drain(propagation_wait)
                .await
                .expect("shutdown");
            let drain_ended_utc = now_utc();
            json!({
                "peer_id": me.to_string(), "policy": policy, "posted_id": posted_id,
                "connected_ms": connected_ms, "probes_sent": probes, "probe_ack_rtt_ms": ack_rtt_ms,
                "exited_on_max_jobs": exited_on_max_jobs(&events),
                "link_snip_root": events.iter().find_map(|e| match e {
                    WatchEvent::ResultLinkPublished { link_snip_root, .. } => Some(link_snip_root.clone()),
                    _ => None,
                }),
                "loop_started_utc": loop_started_utc, "loop_returned_utc": loop_returned_utc,
                "drain_ended_utc": drain_ended_utc,
                "drain": {
                    "outcome": drain.outcome.as_str(), "grace_ms": ms(drain.grace),
                    "waited_ms": ms(drain.waited), "publishes_in_window": drain.publishes_in_window,
                    "recipients_connected": drain.recipients_connected,
                },
            })
        });
        // Process exit, as in `run_contributor`.
        rt.shutdown_background();

        let mut record = json!({
            "kind": "contributor_run", "session": session, "run": run,
            "propagation_wait_ms": ms(propagation_wait), "watchers": watchers,
        });
        if let (Some(r), Some(o)) = (record.as_object_mut(), outcome.as_object()) {
            r.extend(o.clone());
        }
        Record::open(&record_path).write(record.clone());
        watchdog.disarm();
        println!(
            "delivery-runner role=contributor run={run} exited_on_max_jobs={} drain={} \
             drain_waited_ms={} recipients_connected={}",
            record["exited_on_max_jobs"], record["drain"]["outcome"],
            record["drain"]["waited_ms"], record["drain"]["recipients_connected"]
        );
        assert_eq!(record["exited_on_max_jobs"], json!(true), "{record}");
        assert!(record["link_snip_root"].is_string(), "{record}");
    }

    // ── link proxy ───────────────────────────────────────────────────────────

    /// xorshift64*: the proxy's loss and jitter draws, reproducible from a seed.
    struct Rng(Mutex<u64>);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(Mutex::new(seed.max(1)))
        }

        /// Uniform in [0, 1).
        fn unit(&self) -> f64 {
            let mut s = self.0.lock().unwrap();
            let mut x = *s;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            *s = x;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[derive(Clone, Copy)]
    struct Link {
        delay: Duration,
        jitter: Duration,
        loss: f64,
    }

    impl Link {
        /// `None`: drop the datagram. `Some(d)`: deliver it after `d`.
        fn decide(&self, rng: &Rng) -> Option<Duration> {
            if self.loss > 0.0 && rng.unit() < self.loss {
                return None;
            }
            Some(self.delay + self.jitter.mul_f64(rng.unit()))
        }
    }

    #[derive(Default)]
    struct Counters {
        up_forwarded: AtomicU64,
        up_dropped: AtomicU64,
        down_forwarded: AtomicU64,
        down_dropped: AtomicU64,
    }

    impl Counters {
        fn json(&self) -> serde_json::Value {
            json!({
                "at_utc": now_utc(),
                "to_upstream": {"forwarded": self.up_forwarded.load(Relaxed), "dropped": self.up_dropped.load(Relaxed)},
                "to_client": {"forwarded": self.down_forwarded.load(Relaxed), "dropped": self.down_dropped.load(Relaxed)},
            })
        }
    }

    fn send_later(sock: Arc<UdpSocket>, datagram: Vec<u8>, to: SocketAddr, after: Duration) {
        tokio::spawn(async move {
            if !after.is_zero() {
                tokio::time::sleep(after).await;
            }
            let _ = sock.send_to(&datagram, to).await;
        });
    }

    /// Forward UDP between clients and `OMNI_DELIVERY_PROXY_UPSTREAM`, each
    /// datagram in each direction delayed by `OMNI_DELIVERY_LINK_DELAY_MS`
    /// plus up to `OMNI_DELIVERY_LINK_JITTER_MS`, or dropped with probability
    /// `OMNI_DELIVERY_LINK_LOSS_PCT`. QUIC runs through it unchanged; the
    /// watcher sees the proxy as the client.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "issue #150 acceptance runner role; run by scripts/acceptance/result-delivery.sh"]
    async fn link_proxy() {
        let Some(upstream) = env_str(PROXY_UPSTREAM) else {
            eprintln!("runner::link_proxy: {PROXY_UPSTREAM} is not set; nothing to do");
            return;
        };
        let upstream: SocketAddr = tokio::net::lookup_host(&upstream)
            .await
            .unwrap_or_else(|e| panic!("{PROXY_UPSTREAM}={upstream}: {e}"))
            .next()
            .unwrap_or_else(|| panic!("{PROXY_UPSTREAM}={upstream}: no address"));
        let link = Link {
            delay: env_ms(LINK_DELAY_MS, 0),
            jitter: env_ms(LINK_JITTER_MS, 0),
            loss: env_or(LINK_LOSS_PCT, 0.0f64) / 100.0,
        };
        assert!((0.0..=1.0).contains(&link.loss), "{LINK_LOSS_PCT} must be 0..=100");
        let seed: u64 = env_or(LINK_SEED, 150);
        let rng = Arc::new(Rng::new(seed));
        // The proxy stops on its own after its lifetime (default: its deadline), and
        // the watchdog ends it if that does not happen, so an orphaned proxy cannot
        // outlive its session.
        let deadline_secs: u64 = env_or(DEADLINE_SECS, 900);
        let lifetime = Duration::from_secs(env_or(PROXY_LIFETIME_SECS, deadline_secs));
        let watchdog = Watchdog::arm(
            "runner::link_proxy".into(),
            lifetime + Duration::from_secs(10),
        );
        let listen = env_str(PROXY_LISTEN).unwrap_or_else(|| "127.0.0.1:0".into());
        let front = Arc::new(
            UdpSocket::bind(&listen)
                .await
                .unwrap_or_else(|e| panic!("{PROXY_LISTEN}={listen}: {e}")),
        );
        let bound = front.local_addr().expect("proxy address");
        println!(
            "delivery-runner role=link_proxy listen={bound} upstream={upstream} delay_ms={} \
             jitter_ms={} loss_pct={} seed={seed}",
            ms(link.delay),
            ms(link.jitter),
            link.loss * 100.0
        );
        if let Some(f) = env_str(PROXY_ADDR_FILE) {
            write_atomically(&f, &format!("{bound}\n"));
        }

        let counters = Arc::new(Counters::default());
        let stats_file = env_str(PROXY_STATS_FILE);
        if let Some(path) = stats_file.clone() {
            let counters = counters.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    write_atomically(&path, &format!("{}\n", counters.json()));
                }
            });
        }

        let bind_any = if upstream.is_ipv4() { "0.0.0.0:0" } else { "[::]:0" };
        let mut backs: HashMap<SocketAddr, Arc<UdpSocket>> = HashMap::new();
        let deadline = tokio::time::Instant::now() + lifetime;
        let mut buf = vec![0u8; 65_535];
        loop {
            let (n, client) = tokio::select! {
                _ = tokio::time::sleep_until(deadline) => break,
                r = front.recv_from(&mut buf) => match r {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("link_proxy: recv: {e}");
                        continue;
                    }
                },
            };
            let back = match backs.get(&client) {
                Some(b) => b.clone(),
                None => {
                    let b = Arc::new(UdpSocket::bind(bind_any).await.expect("bind upstream socket"));
                    let (b2, front2, counters2, rng2) =
                        (b.clone(), front.clone(), counters.clone(), rng.clone());
                    tokio::spawn(async move {
                        let mut buf = vec![0u8; 65_535];
                        while let Ok((n, _)) = b2.recv_from(&mut buf).await {
                            match link.decide(&rng2) {
                                None => {
                                    counters2.down_dropped.fetch_add(1, Relaxed);
                                }
                                Some(d) => {
                                    counters2.down_forwarded.fetch_add(1, Relaxed);
                                    send_later(front2.clone(), buf[..n].to_vec(), client, d);
                                }
                            }
                        }
                    });
                    backs.insert(client, b.clone());
                    b
                }
            };
            match link.decide(&rng) {
                None => {
                    counters.up_dropped.fetch_add(1, Relaxed);
                }
                Some(d) => {
                    counters.up_forwarded.fetch_add(1, Relaxed);
                    send_later(back, buf[..n].to_vec(), upstream, d);
                }
            }
        }
        let stats = counters.json();
        if let Some(path) = stats_file {
            write_atomically(&path, &format!("{stats}\n"));
        }
        println!("delivery-runner role=link_proxy stats={stats}");
        watchdog.disarm();
    }
}
