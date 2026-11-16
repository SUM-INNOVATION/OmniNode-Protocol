# Acceptance runner: result delivery at `--max-jobs` exit (issue #150)

Issue #150: `watch-network-jobs --publish-result-link --max-jobs 1` broadcast its
result announcement and exited before the message left the node, so a connected
`watch-network-results` never received it. #152 added the bounded shutdown drain
(`--propagation-wait-ms`, default 200). The evidence so far is loopback with one
recipient. This runner is the reproducible procedure for the rest: separate
processes, separate hosts, and a slow or lossy link. It records what happened and
gives a verdict. It does not close #150; the external acceptance does.

**The limit.** Bounded shutdown draining does not guarantee remote delivery.
`--propagation-wait-ms` bounds how long the node keeps driving the network after its
last broadcast. It is not an acknowledgement from the recipient. If a packet is lost
and QUIC would resend it after the drain ends, the message is lost. The runner saw
exactly that once on an emulated lossy link (see **Observed results**).

**Scope: test tooling only.** This runner adds:
- test roles in `crates/omni-contributor/tests/watch_network_jobs_exit_delivery.rs`;
- `scripts/acceptance/result-delivery.sh`;
- this document.

No product code changes. Every file under `crates/*/src/` is the same as on `main`,
including the delivery path:
- `crates/omni-contributor/src/watch.rs` and `crates/omni-contributor/src/relay.rs`;
- `crates/omni-net/src/drain.rs` and `crates/omni-net/src/swarm.rs`;
- `crates/omni-node/src/contributor_cli.rs`.

The deadline fix (no hang) and the redial change below are fixes to the runner, not
to `omni-node`.

## What it runs

`scripts/acceptance/result-delivery.sh` drives three ignored test roles in
`crates/omni-contributor/tests/watch_network_jobs_exit_delivery.rs`, next to the
in-process regression tests from #152:

| Role | What it is |
|---|---|
| `runner::contributor` | One contributor run per OS process. It connects to the watcher(s), handshakes (below), then runs the `watch-network-jobs --max-jobs 1` sequence: `run_watch_loop` with `max_jobs = Some(1)` and the command's broadcaster (`OmniNetRelay::publish_result`), then `shutdown_with_drain(--propagation-wait-ms)`, then process exit. |
| `runner::watcher` | The receiving side. It reads the result topic through `OmniNetRelay::poll_results`, the consumer `watch-network-results` uses. It records each run and decides the verdict. |
| `runner::link_proxy` | A userspace UDP forwarder between the contributor and the watcher. It delays each datagram in each direction by a fixed one-way delay plus uniform jitter, or drops it with a given probability. QUIC runs through it unchanged. It needs no root. |

**What it does not cover.** The job comes from an in-memory SNIP stand-in, as in
`watch_network_jobs_integration.rs`, so no SUM Chain or SNIP is needed. Only the
result announcement and the handshake cross the network. The SNIP fetch that
`watch-network-results` does after receipt is not exercised. The SwUM lab's
`lab-l3 --cases netjob-local` covers the whole binary path with SNIP.

**Connecting.** The contributor dials every watcher. It dials again any watcher still
unconnected every 5 s (`OMNI_DELIVERY_REDIAL_MS`), because on a lossy link a single
QUIC handshake can fail outright.

**Handshake.** Before running the job, the contributor publishes probes on the result
topic every 100 ms (`OMNI_DELIVERY_PROBE_INTERVAL_MS`). Each watcher answers every
probe with an ack on `omni/test/v1`. The job starts once every watcher has acked
and a further `--settle-ms` (default 1000) has passed. So each watcher is known to
receive the result topic from this contributor, and the probes are outside the drain
window when the announcement goes out. The watcher's relay ignores probes, because
they are not announcements.

## Modes

```text
result-delivery.sh loopback     [options]   # every role on this machine
result-delivery.sh watcher      [options]   # receiving host; prints its address
result-delivery.sh contributor  --watcher-addr /ip4/<ip>/udp/<port>/quic-v1/p2p/<peer> [options]
result-delivery.sh proxy        --proxy-upstream <host:port> [options]
result-delivery.sh build                    # build or locate the test executable
```

Without `--test-bin`, the script builds the test executable with
`cargo test --locked -p omni-contributor --features network --test watch_network_jobs_exit_delivery --no-run`.
The executable is self-contained. A host without a toolchain can use one built on a
host with the same OS, architecture and glibc.

## Parameters

Each parameter is a flag or an environment variable; the flag wins. No endpoint is
compiled in.

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--runs` | `OMNI_DELIVERY_RUNS` | 20 | Contributor runs (processes), and what the watcher waits for |
| `--propagation-wait-ms` | `OMNI_DELIVERY_PROPAGATION_WAIT_MS` | 200 | The drain bound, as `watch-network-jobs --propagation-wait-ms`; 0 is the pre-#152 exit |
| `--settle-ms` | `OMNI_DELIVERY_SETTLE_MS` | 1000 | Connected time after the handshake, before the job |
| `--listen-port` | `OMNI_DELIVERY_LISTEN_PORT` | 0 (OS-assigned) | QUIC listen port on `0.0.0.0`. Fix it on the watcher host for firewalls |
| `--advertise-ip` | `OMNI_DELIVERY_ADVERTISE_IP` | 127.0.0.1 | The IPv4 address the watcher prints for dialling |
| `--watcher-addr` | `OMNI_DELIVERY_WATCHER_ADDRS` | none | Contributor mode: the watcher multiaddr(s), comma-separated, each with `/p2p/<peer-id>` |
| `--identity-file` | `OMNI_DELIVERY_IDENTITY_FILE` | ephemeral | Persistent libp2p identity (created 0600), so the watcher's peer id stays stable |
| `--outbound-allow` | `OMNI_DELIVERY_OUTBOUND_ALLOW` | `loopback`: `127.0.0.1,::1`; otherwise none | OmniNet outbound allow-list (#158) for the watcher and the contributor; across hosts, set it to the other host's address |
| `--link-delay-ms` | `OMNI_DELIVERY_LINK_DELAY_MS` | 0 | Emulated one-way delay, each direction (the round trip adds 2×) |
| `--link-jitter-ms` | `OMNI_DELIVERY_LINK_JITTER_MS` | 0 | Uniform 0..N ms added per datagram (can reorder) |
| `--link-loss-pct` | `OMNI_DELIVERY_LINK_LOSS_PCT` | 0 | Datagram loss, each direction |
| `--link-seed` | `OMNI_DELIVERY_LINK_SEED` | 150 | Seed for the loss and jitter draws |
| `--proxy-listen` | `OMNI_DELIVERY_PROXY_LISTEN` | 127.0.0.1:0 | The proxy's own UDP address |
| `--proxy-upstream` | `OMNI_DELIVERY_PROXY_UPSTREAM` | none | Proxy mode: the watcher's `host:port` |
| `--receipt-timeout-ms` | `OMNI_DELIVERY_RECEIPT_TIMEOUT_MS` | 10000 | How long after a contributor disconnects its announcement may still arrive |
| `--run-deadline-secs` | `OMNI_DELIVERY_RUN_DEADLINE_SECS` | derived (see Deadlines) | Bound on one contributor process; also its connect and handshake bound, and how long a run may go without receipt or disconnect |
| `--start-grace-secs` | `OMNI_DELIVERY_START_GRACE_SECS` | 600 | Watcher mode: how long to wait for the first contributor |
| `--evidence-class` | `OMNI_DELIVERY_EVIDENCE_CLASS` | see below | Recorded with the result |
| `--link-note` | `OMNI_DELIVERY_LINK_NOTE` | empty | Free text: the real path, the OS shaping applied, the measured RTT |
| `--session` | `OMNI_DELIVERY_SESSION` | `s<UTC>-<pid>` | Run label; makes every job unique |
| `--out` | `OMNI_DELIVERY_OUT` | `./delivery-runs/<session>-<mode>` | Record directory; must be new or empty |
| `--test-bin` | `OMNI_DELIVERY_TEST_BIN` | built by cargo | The test executable |

With a delay, jitter or loss set, `loopback` and `contributor` start the proxy on
the contributor's host and dial the watcher through it. They also set the
contributor's OmniNet `--outbound-allow` to the proxy's address only (#158), so no
dial can bypass the emulated link. An emulated link takes exactly one watcher.

mDNS is off in every role, and `loopback` keeps every OmniNet dial on loopback
(`127.0.0.1,::1`), so a local run never reaches another host.

## Deadlines

Nothing in the runner waits without a bound; a stuck process fails instead of
hanging. The bounds come from a clean baseline (macOS arm64, debug build): 1.45 s
per run on loopback, and 3.6 s per run through a 150 ms + 0..20 ms emulated link.

- **Expected run time:** `250 ms + --settle-ms + --propagation-wait-ms + 14 × (--link-delay-ms + --link-jitter-ms)`.
- **Per contributor process:** `max(20 s, 10 × expected)`, unless `--run-deadline-secs` is given.
- **Watcher and proxy:** `runs × run deadline + 60 s`. Watcher mode adds `--start-grace-secs`.
- **Watcher idle bound:** twice the run deadline without a probe (watcher mode: at least `--start-grace-secs`).

Each role enforces its own bound on the wall clock and exits with status 124 when it
overruns. The script also kills any process still running at its bound, so a role
never outlives its session. Background roles are started with `exec`, so the
script's kill reaches the test executable itself. The two in-process tests in the
same file have a 90 s whole-test bound (clean run: 7.2 s).

**Sleep.** On macOS the script reads `pmset -g log`. A session that overlaps a
system sleep is `RESULT=VOID`, whatever it measured. Elsewhere this check reports
`unavailable`, and the operator must make sure the hosts stay awake.

## Records

All records go in `--out`:

- **`session.env`:** mode, session, start time, evidence class, link note, host
  (`hostname`, `uname -sm`), commit and number of uncommitted changes, test
  executable, and every parameter, with the derived deadlines.
- **`watcher.jsonl`:** JSON lines:
  - `watcher_start` (its address and OmniNet policy);
  - per run, `probe` (first probe, contributor peer id, posted id);
  - `contributor_disconnected`;
  - `receipt` (outcome, link root, announcer signature check, ms since the first
    probe, whether the contributor had already disconnected, and by how long);
  - `lost` (and why), `late_receipt`, `unmatched_announcement`;
  - a final `summary` with the verdict.
- **`contributor.jsonl`:** one `contributor_run` line per process:
  - posted id, link root, `exited_on_max_jobs`;
  - connect time, probes sent, probe-to-ack round trip;
  - the loop's start and return times and the drain's end time (UTC);
  - the drain report (`outcome`, `grace_ms`, `waited_ms`, `publishes_in_window`,
    `recipients_connected`);
  - the OmniNet policy.
- **`proxy.stats`:** datagrams forwarded and dropped, per direction.
- **Logs:** `watcher.log`, `contributor-<run>.log`, `proxy.log`.
- **`summary.txt`:** `session.env`, the contributor tally (records, exits, drain
  outcomes, probe-to-ack round trip min/median/max), proxy counters, the watcher's
  verdict line, the elapsed time, the sleep check and `RESULT=` (`PASS`, `FAIL` or `VOID`).

The watcher uses only its own clock, so its timings need no clock sync between
hosts.

## Pass and fail

**PASS** needs all of these:

1. **The watcher's verdict.** The watcher saw `--runs` runs, and for each run the
   announcement with that run's posted id arrived with a verifying announcer
   signature. It arrived before the contributor disconnected, or within
   `--receipt-timeout-ms` after.
2. **The contributor processes**, where the contributor runs on this host
   (`loopback`, `contributor`):
   - every one exited 0;
   - every one wrote a record;
   - every record has `exited_on_max_jobs: true` and a published link root.

Anything else is a **FAIL**: a lost run, a late receipt, a bad signature, a run
never seen, a contributor that failed, or any process past its deadline. The
watcher's exit status is its verdict. `loopback` and `contributor` exit non-zero
unless `RESULT=PASS`. **VOID** (macOS: the machine slept during the session)
overrides both. The summary tells a setup failure from a delivery loss: a contributor that never connected shows as `processes_failed` and `deadline_exceeded`, and as a run the watcher never saw (`runs_seen` below `--runs`). A delivery loss shows as `lost`.

With `--propagation-wait-ms` above zero, every drain outcome is expected to be
`settled`. `recipients_gone` means the watcher went away first. Investigate it,
but it is not a delivery failure if the receipt is there.

**Negative control.** `--propagation-wait-ms 0` is the exit path before #152. Run
it once per setup to show that this setup can see a loss. A setup that delivers
everything without the drain cannot show that the drain matters.

## Observed results (local evidence)

All rows below are local evidence: one macOS arm64 machine, loopback, separate OS
processes. "Emulated" means through the userspace link proxy. None is external
acceptance. Every session's pmset sleep check was clean. Runs are 20 per session.

| Commit | Session (UTC, 2026-11-16) | Link | `--propagation-wait-ms` | Result |
|---|---|---|---|---|
| `80857da0` | 21:26–21:27 | loopback | 200 | PASS 20/20, all drains `settled` |
| `80857da0` | 21:27–21:28 | emulated 150 ms + 0..20 ms | 200 | PASS 20/20 |
| `80857da0` | 21:28–21:31 | emulated 150 ms, 5 % loss | 200 | **FAIL**: run 12 lost (see below); run 9 never connected |
| `44cd7487` | 21:37 | loopback | 200 | PASS 20/20 |
| `44cd7487` | 21:37–21:39 | emulated 150 ms + 0..20 ms | 200 | PASS 20/20 |
| `44cd7487` | 21:39–21:40 | emulated 150 ms, 5 % loss | 200 | PASS 20/20 |
| `44cd7487` | 21:40–21:42 | emulated 150 ms, 5 % loss | 1000 | PASS 20/20 |
| `44cd7487` | 21:42–21:45 | emulated 400 ms, 5 % loss | 200 | PASS 20/20 |
| `44cd7487` | 21:45 | loopback | 0 (control) | FAIL 16/20, 4 lost |
| `44cd7487` | 21:45–21:46 | emulated 150 ms | 0 (control) | FAIL 8/20, 12 lost |
| `44cd7487` | 21:47 | loopback, two watchers | 200 | PASS 20/20 at each watcher |

**The loss at `80857da0`.** Through the emulated 150 ms, 5 % loss link with the
default 200 ms drain, run 12's announcement never reached the watcher. Its drain
report says `outcome: settled`, `waited_ms: 200`, `recipients_connected: 1`. The drain
ran its full bound with the recipient still connected, and the message was still
lost. The session ran `80857da0`. The contributor sequence has not changed since,
because `407e93ac` touched only how the runner connects.

In the same session, run 9 never connected: its only QUIC handshake failed, and the
runner did not dial again. That was a setup failure in the runner, not a delivery
loss. `407e93ac` fixed it by redialling.

The next session of the same scenario, at `44cd7487`, delivered 20 of 20. One loss in
20 runs, then none in the next 20, says the loss is intermittent. It does not say the
link is safe.

**Outstanding.** External acceptance is NOT done. It includes repeating the lossy
scenarios on the two-host setup over several sessions, not one.

## Evidence classes

- **`loopback` is local evidence** (this machine, loopback). With a delay or loss
  it is an emulated slow link through the userspace proxy. The record says so, and
  says it is not external acceptance.
- **`watcher` and `contributor`** record what `--evidence-class` says, by default
  `unattested: …`. The runner cannot tell whether the hosts are separate machines
  or the link is real. The operator states that, and the path, in
  `--evidence-class` and `--link-note`.
- **External acceptance** for #150 means the watcher and the contributor on
  separate hosts, over a real network path. For the slow-link case, that is a real
  slow path, or OS-level shaping on a real path. It is still to be done.

## Which scenarios use a real network

| Scenario | Network |
|---|---|
| `loopback`, with or without an emulated link | **None.** Loopback only, every OmniNet dial fenced to `127.0.0.1,::1`, mDNS off. Nothing leaves the machine. |
| `watcher` and `contributor` on two hosts | **Real network.** Two test hosts, over a LAN or WAN path. |
| The same with OS-level shaping (`tc netem`, dummynet) | **Real network, shaped** by the operator on the test hosts. |

No production host is used or needed: every host is a test host. Fence the OmniNet
dials of each host to the other with `--outbound-allow`.

## Running it across hosts

Host B, the watcher. Fix the port so it can be opened in the firewall:

```sh
scripts/acceptance/result-delivery.sh watcher \
  --listen-port 4710 --advertise-ip <B's IPv4> --identity-file ./delivery-watcher.key \
  --outbound-allow <A's IPv4> \
  --runs 20 --evidence-class "external: A=<host>, B=<host>, path=<LAN|WAN|...>" \
  --link-note "<shaping applied, measured RTT>"
# prints: watcher address: /ip4/<B>/udp/4710/quic-v1/p2p/<peer-id>
```

Host A, the contributor:

```sh
scripts/acceptance/result-delivery.sh contributor \
  --watcher-addr /ip4/<B>/udp/4710/quic-v1/p2p/<peer-id> --runs 20 \
  --outbound-allow <B's IPv4> \
  --evidence-class "external: A=<host>, B=<host>, path=<...>"
```

Host B's `summary.txt` has the verdict. Host A's has the contributor tally and
drain outcomes. Keep both directories.

**Suggested matrix.** The maintainers decide what the external acceptance must
include. A useful set, each at the default `--propagation-wait-ms 200` with 20 runs,
plus one `--propagation-wait-ms 0` control each:

1. **Real network, no shaping:** two hosts on one LAN.
2. **Real network, slow:** two hosts over a real slow path, or with OS shaping of
   about 150 ms one way.
3. **Real network, slow and lossy:** as 2, with 1–5 % loss. Repeat it over several
   sessions, because the local loss above was intermittent.
4. **Real network, two recipients:** two watcher hosts at once
   (`--watcher-addr B1,B2`, no emulated link).

**Host requirements:**
- UDP reachability from A to B on the watcher's port.
- The test executable for each host, or a Rust toolchain to build it.
- `bash`, `awk`, `sed` and `grep`.
- Nothing else for the emulated link.
- **Real shaping needs root** and the OS's own tool, applied by the operator
  outside this script and recorded with `--link-note`. This runner does not apply
  it.
  - **Linux:** `tc` (iproute2) with `sch_netem`. Example:
    `tc qdisc add dev <if> root netem delay 150ms 20ms loss 1%`, and later
    `tc qdisc del dev <if> root`.
  - **macOS:** `dnctl` and `pfctl` (dummynet).
- Neither shaping tool is exercised by this repository's checks.
