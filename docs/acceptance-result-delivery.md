# Acceptance runner: result delivery at `--max-jobs` exit (issue #150)

Issue #150: `watch-network-jobs --publish-result-link --max-jobs 1` broadcast its
result announcement and exited before the message left the node, so a connected
`watch-network-results` never received it. #152 added the bounded shutdown drain
(`--propagation-wait-ms`, default 200). The evidence so far is loopback with one
recipient. This runner is the reproducible procedure for the rest: separate
processes, separate hosts, and a slow or lossy link. It records what happened and
gives a verdict. It does not close #150; the external acceptance does.

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

**Handshake.** Before running the job, the contributor publishes probes on the
result topic every 100 ms (`OMNI_DELIVERY_PROBE_INTERVAL_MS`). Each watcher answers every probe
with an ack on `omni/test/v1`. The job starts once every watcher has acked and a further
`--settle-ms` (default 1000) has passed. So each watcher is known to receive the
result topic from this contributor, and the probes are outside the drain window
when the announcement goes out. The watcher's relay ignores probes, because they
are not announcements.

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
| `--link-delay-ms` | `OMNI_DELIVERY_LINK_DELAY_MS` | 0 | Emulated one-way delay, each direction (the round trip adds 2×) |
| `--link-jitter-ms` | `OMNI_DELIVERY_LINK_JITTER_MS` | 0 | Uniform 0..N ms added per datagram (can reorder) |
| `--link-loss-pct` | `OMNI_DELIVERY_LINK_LOSS_PCT` | 0 | Datagram loss, each direction |
| `--link-seed` | `OMNI_DELIVERY_LINK_SEED` | 150 | Seed for the loss and jitter draws |
| `--proxy-listen` | `OMNI_DELIVERY_PROXY_LISTEN` | 127.0.0.1:0 | The proxy's own UDP address |
| `--proxy-upstream` | `OMNI_DELIVERY_PROXY_UPSTREAM` | none | Proxy mode: the watcher's `host:port` |
| `--receipt-timeout-ms` | `OMNI_DELIVERY_RECEIPT_TIMEOUT_MS` | 10000 | How long after a contributor disconnects its announcement may still arrive |
| `--max-run-secs` | `OMNI_DELIVERY_MAX_RUN_SECS` | 120 | A run with no receipt and no disconnect is lost after this |
| `--connect-timeout-secs` | `OMNI_DELIVERY_CONNECT_TIMEOUT_SECS` | 60 | Contributor: connect and handshake bound |
| `--idle-timeout-secs` | `OMNI_DELIVERY_IDLE_TIMEOUT_SECS` | 600 | Watcher: give up after this long without a probe |
| `--evidence-class` | `OMNI_DELIVERY_EVIDENCE_CLASS` | see below | Recorded with the result |
| `--link-note` | `OMNI_DELIVERY_LINK_NOTE` | empty | Free text: the real path, the OS shaping applied, the measured RTT |
| `--session` | `OMNI_DELIVERY_SESSION` | `s<UTC>-<pid>` | Run label; makes every job unique |
| `--out` | `OMNI_DELIVERY_OUT` | `./delivery-runs/<session>-<mode>` | Record directory; must be new or empty |
| `--test-bin` | `OMNI_DELIVERY_TEST_BIN` | built by cargo | The test executable |

With a delay, jitter or loss set, `loopback` and `contributor` start the proxy on
the contributor's host and dial the watcher through it. They also set the
contributor's OmniNet `--outbound-allow` to the proxy's address only (#158), so no
dial can bypass the emulated link. An emulated link takes exactly one watcher.

## Records

All records go in `--out`:

- **`session.env`:** mode, session, start time, evidence class, link note, host
  (`hostname`, `uname -sm`), commit and number of uncommitted changes, test
  executable, and every parameter.
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
  verdict line and `RESULT=`.

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
never seen, or a contributor that failed. The watcher's exit status is its verdict;
`loopback` and `contributor` exit non-zero on a FAIL.

With `--propagation-wait-ms` above zero, every drain outcome is expected to be
`settled`. `recipients_gone` means the watcher went away first. Investigate it,
but it is not a delivery failure if the receipt is there.

**Negative control.** `--propagation-wait-ms 0` is the exit path before #152. Run
it once per setup to show that this setup can see a loss. A setup that delivers
everything without the drain cannot show that the drain matters. As local evidence (macOS
arm64, loopback, separate processes), this control lost announcements on plain
loopback and lost more through a 150 ms emulated link. With the 200 ms default,
none were lost.

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

## Running it across hosts

Host B, the watcher. Fix the port so it can be opened in the firewall:

```sh
scripts/acceptance/result-delivery.sh watcher \
  --listen-port 4710 --advertise-ip <B's IPv4> --identity-file ./delivery-watcher.key \
  --runs 20 --evidence-class "external: A=<host>, B=<host>, path=<LAN|WAN|...>" \
  --link-note "<shaping applied, measured RTT>"
# prints: watcher address: /ip4/<B>/udp/4710/quic-v1/p2p/<peer-id>
```

Host A, the contributor:

```sh
scripts/acceptance/result-delivery.sh contributor \
  --watcher-addr /ip4/<B>/udp/4710/quic-v1/p2p/<peer-id> --runs 20 \
  --evidence-class "external: A=<host>, B=<host>, path=<...>"
```

Host B's `summary.txt` has the verdict. Host A's has the contributor tally and
drain outcomes. Keep both directories.

**Suggested matrix.** The maintainers decide what the external acceptance must
include. A useful set, each at the default `--propagation-wait-ms 200` with 20 runs,
plus one `--propagation-wait-ms 0` control each:

1. Two hosts on one LAN, no shaping.
2. Two hosts over a real slow path, or with OS shaping of about 150 ms one way.
3. As 2, with 1–5 % loss.
4. Two watcher hosts at once (`--watcher-addr B1,B2`, no emulated link), for more
   than one recipient.

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
