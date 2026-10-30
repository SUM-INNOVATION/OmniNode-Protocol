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
/// `posted_id`.
fn announce_job(snip: &MockSnipStore, relay: &mut InMemoryRelay, run: usize) -> String {
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
            let out = tempfile::tempdir().unwrap();
            let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
            let runner = StubRunner::new(signer.pubkey_hex(), b"exit-delivery-resp".to_vec(), 9, 11);
            let mut source = NetworkSource::new(&mut jobs, &snip);
            let mut emitter = CollectEmitter(Vec::new());
            let mut broadcaster = NetResultBroadcaster {
                relay: OmniNetRelay::new(broadcaster_net, handle),
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
            run_watch_loop(&snip, &mut source, opts).expect("watch loop");
            (posted_id, emitter.0)
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
