# Acceptance runner: OmniNet routing controls (issue #158)

#158 asked for a way to turn off OmniNet's mDNS and to keep `omni-node` on a set of
addresses. #160 added both, merged as `710a5ab095a49a3b2d54a5639d1dbd470face2b1`:

- **`--mdns on|off`** (environment `OMNI_MDNS`): `off` builds no mDNS behaviour.
- **`--outbound-allow <entry>[,…]`** (environment `OMNI_OUTBOUND_ALLOW`, SNIP's entry
  syntax):
  - Every OmniNet dial outside the list is refused before a packet is sent:
    `--peer`, mDNS, Identify, Kademlia, AutoNAT, relay and DCUtR.
  - Refused addresses are kept out of the peer stores.
  - Inbound connections are kept only from listed addresses.
- **Visibility:**
  - a startup `network policy: …` line;
  - one refusal line per source and address, with the source tagged (`bootstrap`,
    `mdns`, `identify`, `kad`, `inbound` or `dial`);
  - per-source refusal counters.

Defaults are unchanged: mDNS on, no restriction. The unit and loopback-swarm tests
from #160 cover parsing, flag-over-environment precedence and the swarm-level
behaviour. This runner checks the same promises on the real `omni-node` binary and
records the result. It changes no code.

## What it runs

`scripts/acceptance/routing-controls.sh run` starts real
`omni-node operator contributor watch-network-results` processes. That command opens
OmniNet with the controls and needs no SUM Chain or SNIP until an announcement
arrives; none does.

It judges each node by its own log:
- `network policy: …` (the effective policy);
- `local_peer_id=`;
- `listening on address`;
- `connection established peer_id=`;
- `mDNS discovered peer`;
- libp2p-mdns's `creating instance on iface`, which shows mDNS actually started;
- `network policy: address not in --outbound-allow, refused source=…`.

On this machine every node runs with `--mdns off` and an outbound allow-list limited
to loopback (`--fence`, default `127.0.0.1,::1`), so no run reaches another host.
The single exception is `mdns-on-instances`; see below.

| Scenario | Setup | PASS when |
|---|---|---|
| `mdns-off-flag` | A: `--mdns off` | A's policy says `mdns=off`; A starts no mDNS instance; A logs no mDNS discovery and no connection |
| `mdns-off-env` | A: `OMNI_MDNS=off` from the environment | Same as `mdns-off-flag` |
| `dial-peer` | B and C plain. A: `--outbound-allow <C's address> --peer B --peer C` | A's policy lists C's address. A connects to C and never to B, and refuses B at admission (`source=bootstrap`). Every peer A connects to is C |
| `dial-peer-env` | As `dial-peer`, with the list from `OMNI_OUTBOUND_ALLOW` | Same as `dial-peer` |
| `inbound` | C plain. A: `--outbound-allow <C's address>`. B: `--peer A` | A refuses B's inbound connection (`source=inbound`); A has no connection at all |
| `kad-identify` | B plain. C: `--peer B`. A: `--outbound-allow <C's address> --peer C` | C is connected to B. A connects to C and never to B, and refuses B's address learned through C (`source` `kad`, `identify` or `dial`). With no such refusal the learned-address path was not exercised, and the result is **INCONCLUSIVE**, not PASS |
| `mdns-on-instances` | A: default mDNS, outbound limited to the fence. Needs `--allow-multicast` | A's policy says `mdns=on`; mDNS instances started; no connection |

**`mdns-on-instances`** is the positive control for the "no mDNS instance" check in
`mdns-off-*`: it shows that the log line appears when mDNS is on.
- It multicasts on this host's LAN interfaces.
- Its record says so (`MULTICAST-NOTICE.txt`, with the interfaces).
- It discovers nothing by design, and its outbound dials stay on the fence.
- It runs only with `--allow-multicast`.

**Verdicts.** Each scenario gets one:

- **PASS**
- **FAIL**
- **INCONCLUSIVE:** the path under test was not exercised, so the absence of a
  connection proves nothing.
- **SKIPPED**

The run's `RESULT=PASS` needs at least one PASS and no FAIL or INCONCLUSIVE.

**Bounds.** Every node has a wall-clock bound:
- it exits on its own after `--observe-secs + 20` polls of 1 s;
- the script kills it at the end of its scenario;
- startup waits are bounded at 20 s.

On macOS, a run that overlaps a system sleep in `pmset -g log` is `RESULT=VOID`.

## Parameters

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--omni-node` | `OMNI_ROUTING_OMNI_NODE` | built with `cargo build -p omni-node` | The binary under test |
| `--ip` | `OMNI_ROUTING_IP` | 127.0.0.1 | The IPv4 address local nodes are dialled on and allowed by |
| `--fence` | `OMNI_ROUTING_FENCE` | `127.0.0.1,::1` | `--outbound-allow` for every helper node |
| `--observe-secs` | `OMNI_ROUTING_OBSERVE_SECS` | 12 | How long each scenario watches |
| `--scenarios` | `OMNI_ROUTING_SCENARIOS` | all of the above | Comma-separated |
| `--allow-multicast` | `OMNI_ROUTING_ALLOW_MULTICAST=1` | off | Run `mdns-on-instances` |
| `--evidence-class` | `OMNI_ROUTING_EVIDENCE_CLASS` | local evidence (this machine, loopback) | Recorded with the result |
| `--out` | `OMNI_ROUTING_OUT` | `./routing-runs/<UTC>-<pid>` | Record directory; must be new or empty |
| `--rust-log` | `OMNI_ROUTING_RUST_LOG` | `info` | The nodes' `RUST_LOG` (the checks need `info`) |

## Records

All records go in `--out`:

- **`run.env`:** start time, evidence class, host, commit and uncommitted-change
  count, the binary and its `--version`, the parameters.
- **One directory per scenario:**
  - each node's full log (`A.log`, `B.log`, `C.log`);
  - `commands.txt` (each node's flags and environment);
  - `notes.txt` on setup trouble;
  - `MULTICAST-NOTICE.txt` where it applies.
- **`results.tsv`:** scenario, verdict, and each check with `ok:` or `FAILED:`, plus
  refusal counts by source.
- **`summary.txt`:** the above, the elapsed time, the sleep check and `RESULT=`.

## Local evidence vs external acceptance

`run` on one machine is **local evidence (this machine, loopback)**.

It does not show:
- mDNS discovery between two hosts, with the control on or off;
- an mDNS discovery refused by the allow-list;
- the controls on a real network path;
- relay and DCUtR.

Those are the external acceptance #160 left open. It is **NOT yet done**.

### Two-host procedure

**Host requirement:** an isolated network segment with two test hosts, H1 and H2,
that carries nothing else that answers mDNS. Multicast and the discoveries happen
on that segment only.

1. **mDNS on — positive control.**
   - On H1, with the node's routing flags after `--` (mDNS is on by default):
     `routing-controls.sh node --observe-secs 60 -- --outbound-allow <H2's IP>`.
   - On H2, within the minute:
     `routing-controls.sh node --observe-secs 60 -- --outbound-allow <H1's IP>`.
   - Then `routing-controls.sh scan --log <node log> --peer <other id>` on each.
   - Expect `discovered=yes` and `connected=yes` on both. Without this, the next
     checks prove nothing.
2. **mDNS off.**
   - As 1, but H2's node also gets `--mdns off`.
   - Expect H2: no mDNS instance, no discovery. H1: no discovery of H2's peer id,
     because H2 does not announce itself.
3. **mDNS discovery refused.**
   - As 1, but H2's node gets `--outbound-allow 127.0.0.1` instead.
   - Expect H2: a `source=mdns` refusal for H1's address, no `mDNS discovered peer`
     for H1, no connection to H1. Any inbound attempt from H1 is refused with
     `source=inbound`.
4. **Dial and inbound across hosts.**
   - Repeat `dial-peer` and `inbound` with B on H1 (`node`) and A on H2: the node
     command with `--peer` and `--outbound-allow` as in the table.
   - Judge them with `scan`.

Record the segment, both hosts and the commands with the logs. The operator states
the evidence class; the runner cannot tell whether hosts are separate.

**Relay and DCUtR.** Nothing here covers them. `omni-node` has no command-line
option to run a relay server (`NetConfig::relay_server` is not exposed). These paths
need a lab with a relay host and a host behind NAT.
