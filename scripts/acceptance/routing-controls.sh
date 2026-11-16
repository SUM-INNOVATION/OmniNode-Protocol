#!/usr/bin/env bash
# Issue #158 acceptance runner: do omni-node's OmniNet routing controls (#160)
# do what they say, on the real binary?
#
#   --mdns off            builds no mDNS: no multicast, no mDNS discovery
#   --outbound-allow L    no OmniNet connection outside L, whichever part of
#                         OmniNet asks (--peer, Identify, Kademlia, inbound)
#   OMNI_MDNS / OMNI_OUTBOUND_ALLOW   the same from the environment
#
# Each node is a real `omni-node operator contributor watch-network-results`
# (it opens OmniNet with the controls and needs no SUM Chain or SNIP until an
# announcement arrives; none does). The runner reads each node's own log:
# `network policy: ...` (the effective policy), `local_peer_id=`, `listening on
# address`, `connection established peer_id=`, `mDNS discovered peer`,
# libp2p-mdns `creating instance on iface` (mDNS actually started), and
# `network policy: address not in --outbound-allow, refused source=...`.
#
# Usage:
#   routing-controls.sh run   [options]   run scenarios on this host (default set below)
#   routing-controls.sh node  [options] -- <omni-node routing flags...>
#                                         one node, for a second host; prints its address
#   routing-controls.sh scan  --log FILE [--peer ID]...
#                                         what a node's log shows (policy, connections,
#                                         discoveries, refusals by source)
#
# Scenarios (run --scenarios a,b,...; default: every local one):
#   mdns-off-flag      A --mdns off: policy mdns=off, no mDNS instance, no discovery
#   mdns-off-env       the same from OMNI_MDNS=off
#   dial-peer          A --outbound-allow <C> --peer B --peer C: connects to C only;
#                      B refused at admission (source=bootstrap)
#   dial-peer-env      the same from OMNI_OUTBOUND_ALLOW
#   inbound            B (unrestricted by A's list) dials A: A refuses it (source=inbound)
#   kad-identify       C knows B; A (allow C only) bootstraps through C: never connects to B,
#                      and refuses B's address learned from C (source kad/identify/dial)
#   mdns-on-instances  (needs --allow-multicast) A with the default mDNS: policy mdns=on and
#                      mDNS instances created. Sends mDNS multicast on this host's LAN
#                      interfaces; outbound stays on the fence. The positive control for the
#                      "no mDNS instance" check above. Discovers nothing by design.
#   two-host scenarios need a second host and are not run by `run` alone; see
#   docs/acceptance-routing-controls.md (mDNS discovery on and off, mDNS refusal,
#   dial/inbound across hosts). Relay and DCUtR need a relay host; not covered.
#
# Options (flag, environment variable, default):
#   --omni-node PATH       OMNI_ROUTING_OMNI_NODE     (none: cargo build -p omni-node, target/debug)
#   --ip IP                OMNI_ROUTING_IP            127.0.0.1 (IPv4 the local nodes are dialled on)
#   --fence LIST           OMNI_ROUTING_FENCE         127.0.0.1,::1 (--outbound-allow for every helper
#                                                     node, so no node here can reach another host)
#   --observe-secs N       OMNI_ROUTING_OBSERVE_SECS  12 (how long each scenario watches)
#   --scenarios LIST       OMNI_ROUTING_SCENARIOS     all local scenarios
#   --allow-multicast      OMNI_ROUTING_ALLOW_MULTICAST=1   run mdns-on-instances
#   --evidence-class TEXT  OMNI_ROUTING_EVIDENCE_CLASS  "local evidence (this machine, loopback)"
#   --out DIR              OMNI_ROUTING_OUT           ./routing-runs/<UTC>
#   --rust-log FILTER      OMNI_ROUTING_RUST_LOG      info
#
# Verdicts per scenario: PASS, FAIL, INCONCLUSIVE (the path under test was not
# exercised, so its absence proves nothing), SKIPPED. The run passes only if
# no scenario FAILs and none is INCONCLUSIVE. Every node and every scenario
# has a wall-clock bound; nothing waits without one. On macOS a run that
# overlaps a system sleep (pmset log) is VOID.

# No pipefail: the log readers stop reading early (head, grep -q) by design.
set -eu

die() { echo "routing-controls: $*" >&2; exit 2; }
usage() { sed -n "2,/^# No pipefail/p" "$0" | sed -e "\$d" -e "s/^# \\{0,1\\}//"; }

CMD=${1:-}
[ -n "$CMD" ] || { usage; exit 2; }
shift
case "$CMD" in
  -h|--help|help) usage; exit 0 ;;
  run|node|scan) ;;
  *) die "unknown command: $CMD (run|node|scan)" ;;
esac

ALL_SCENARIOS="mdns-off-flag mdns-off-env dial-peer dial-peer-env inbound kad-identify mdns-on-instances"
OMNI_NODE=${OMNI_ROUTING_OMNI_NODE:-}
IP=${OMNI_ROUTING_IP:-127.0.0.1}
FENCE=${OMNI_ROUTING_FENCE:-127.0.0.1,::1}
OBSERVE=${OMNI_ROUTING_OBSERVE_SECS:-12}
SCENARIOS=${OMNI_ROUTING_SCENARIOS:-$ALL_SCENARIOS}
ALLOW_MULTICAST=${OMNI_ROUTING_ALLOW_MULTICAST:-0}
EVIDENCE_CLASS=${OMNI_ROUTING_EVIDENCE_CLASS:-}
OUT=${OMNI_ROUTING_OUT:-}
NODE_RUST_LOG=${OMNI_ROUTING_RUST_LOG:-info}
SCAN_LOG=""; SCAN_PEERS=""; NODE_FLAGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; NODE_FLAGS=("$@"); break ;;
    --allow-multicast) ALLOW_MULTICAST=1; shift; continue ;;
  esac
  [ $# -ge 2 ] || die "$1 needs a value"
  case "$1" in
    --omni-node) OMNI_NODE=$2 ;;
    --ip) IP=$2 ;;
    --fence) FENCE=$2 ;;
    --observe-secs) OBSERVE=$2 ;;
    --scenarios) SCENARIOS=$(echo "$2" | tr ',' ' ') ;;
    --evidence-class) EVIDENCE_CLASS=$2 ;;
    --out) OUT=$2 ;;
    --rust-log) NODE_RUST_LOG=$2 ;;
    --log) SCAN_LOG=$2 ;;
    --peer) SCAN_PEERS="$SCAN_PEERS $2" ;;
    *) die "unknown option: $1" ;;
  esac
  shift 2
done
case "$OBSERVE" in ''|*[!0-9]*) die "--observe-secs: not a number" ;; esac

REPO=$(cd "$(dirname "$0")/../.." && pwd)
strip() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }

# ── Log reading (shared by run and scan) ─────────────────────────────────────
peer_of()      { strip "$1" | grep -o 'local_peer_id=[1-9A-HJ-NP-Za-km-z]*' | head -1 | cut -d= -f2; }
port_of()      { strip "$1" | grep 'listening on address' | grep -o '/udp/[0-9]*/' | head -1 | tr -dc '0-9'; }
policy_of()    { strip "$1" | grep -o 'network policy: mdns=.*' | head -1; }
policy_has()   { policy_of "$1" | grep -qF -- "$2"; }
connected()    { strip "$1" | grep -q "connection established peer_id=$2"; }
connected_peers() { strip "$1" | grep -o 'connection established peer_id=[1-9A-HJ-NP-Za-km-z]*' | cut -d= -f2 | sort -u; }
discovered_any() { strip "$1" | grep -q 'mDNS discovered peer'; }
discovered()   { strip "$1" | grep -q "mDNS discovered peer peer_id=$2"; }
mdns_started() { strip "$1" | grep -q 'libp2p_mdns.*creating instance'; }
refusals()     { strip "$1" | grep 'address not in --outbound-allow, refused'; }
refused()      { # refused <log> <source-regex> <port>
  refusals "$1" | grep -E "source=\"?($2)\"?" | grep -q "/udp/$3/"
}
refused_any()  { refusals "$1" | grep -qE "source=\"?($2)\"?"; }
refusal_tally() { refusals "$1" | grep -o 'source="\{0,1\}[a-z]*' | tr -d '"' | sort | uniq -c | awk '{printf "%s=%s ", $2, $1}'; }

scan_report() { # scan_report <log> [peer...]
  local log=$1; shift
  echo "log: $log"
  echo "local_peer_id: $(peer_of "$log")"
  echo "listen port: $(port_of "$log")"
  echo "policy: $(policy_of "$log")"
  echo "mdns instances started: $(mdns_started "$log" && echo yes || echo no)"
  echo "mdns discoveries: $(strip "$log" | grep -c 'mDNS discovered peer' || true)"
  echo "connected peers: $(connected_peers "$log" | tr '\n' ' ')"
  echo "refusals by source: $(refusal_tally "$log")"
  local p
  for p in "$@"; do
    echo "peer $p: connected=$(connected "$log" "$p" && echo yes || echo no) discovered=$(discovered "$log" "$p" && echo yes || echo no)"
  done
}

if [ "$CMD" = scan ]; then
  [ -n "$SCAN_LOG" ] && [ -f "$SCAN_LOG" ] || die "scan needs --log FILE"
  # shellcheck disable=SC2086
  scan_report "$SCAN_LOG" $SCAN_PEERS
  exit 0
fi

# ── The binary ───────────────────────────────────────────────────────────────
if [ -z "$OMNI_NODE" ]; then
  (cd "$REPO" && cargo build --locked -p omni-node --bin omni-node) >&2 || die "cargo build failed"
  OMNI_NODE=${CARGO_TARGET_DIR:-$REPO/target}/debug/omni-node
fi
[ -x "$OMNI_NODE" ] || die "omni-node not executable: $OMNI_NODE"

[ -n "$OUT" ] || OUT="$PWD/routing-runs/$(date -u +%Y%m%dT%H%M%SZ)-$$"
[ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && die "--out $OUT exists and is not empty"
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
[ -n "$EVIDENCE_CLASS" ] || EVIDENCE_CLASS="local evidence (this machine, loopback); NOT external acceptance"

PIDS=""
kill_hard() {
  kill -TERM "$1" 2>/dev/null || return 0
  local i=0
  while kill -0 "$1" 2>/dev/null && [ $i -lt 20 ]; do sleep 0.1; i=$((i + 1)); done
  kill -KILL "$1" 2>/dev/null || true
}
cleanup() { for p in $PIDS; do kill_hard "$p"; done; return 0; }
trap cleanup EXIT
trap 'exit 130' INT TERM

# start_node <dir> <name> <env assignments ("" for none)> [omni-node routing flags...]
# One watch-network-results node in the background, bounded by --max-polls
# (it exits on its own after OBSERVE + 20 s) and killed at the scenario's end.
start_node() {
  local dir=$1 name=$2 envs=$3; shift 3
  mkdir -p "$dir/out-$name"
  # shellcheck disable=SC2086
  ( exec env -u OMNI_MDNS -u OMNI_OUTBOUND_ALLOW NO_COLOR=1 RUST_LOG="$NODE_RUST_LOG" $envs \
      "$OMNI_NODE" operator contributor watch-network-results \
      --result-out-dir "$dir/out-$name" --listen-port 0 --poll-interval-secs 1 \
      --max-polls $((OBSERVE + 20)) "$@" ) > "$dir/$name.log" 2>&1 &
  local pid=$!
  PIDS="$PIDS $pid"
  echo "$pid" > "$dir/$name.pid"
  echo "$name: $envs omni-node operator contributor watch-network-results ... $*" >> "$dir/commands.txt"
  # Ready once it has logged its peer id and a listen port (bounded: 20 s).
  local i=0
  while [ -z "$(peer_of "$dir/$name.log")" ] || [ -z "$(port_of "$dir/$name.log")" ]; do
    kill -0 "$pid" 2>/dev/null || { echo "$name exited during startup" >> "$dir/notes.txt"; return 1; }
    i=$((i + 1)); [ $i -le 200 ] || { echo "$name not ready in 20 s" >> "$dir/notes.txt"; return 1; }
    sleep 0.1
  done
}

addr_of() { echo "/ip4/$IP/udp/$(port_of "$1/$2.log")/quic-v1/p2p/$(peer_of "$1/$2.log")"; }
prefix_of() { echo "/ip4/$IP/udp/$(port_of "$1/$2.log")/quic-v1"; }

stop_nodes() { # stop_nodes <dir>
  local f
  for f in "$1"/*.pid; do [ -f "$f" ] && kill_hard "$(cat "$f")"; done
  return 0
}

observe() { sleep "$OBSERVE"; }

RESULTS="$OUT/results.tsv"
: > "$RESULTS"
record() { # record <scenario> <verdict> <details>
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$RESULTS"
  printf '%-18s %-12s %s\n' "$1" "$2" "$3"
}

# check <description> <command...>: run a check, note it, return its status
CHECKS=""
check() {
  local what=$1; shift
  if "$@"; then CHECKS="$CHECKS ok:$what;"; return 0; fi
  CHECKS="$CHECKS FAILED:$what;"; return 1
}
not() { ! "$@"; }

# ── Scenarios ────────────────────────────────────────────────────────────────

sc_mdns_off() { # sc_mdns_off <scenario> <via: flag|env>
  local sc=$1 via=$2 d="$OUT/$1" ok=1
  mkdir -p "$d"; CHECKS=""
  if [ "$via" = flag ]; then
    start_node "$d" A "" --mdns off --outbound-allow "$FENCE" || { record "$sc" FAIL "setup: $(cat "$d/notes.txt")"; return; }
  else
    start_node "$d" A "OMNI_MDNS=off" --outbound-allow "$FENCE" || { record "$sc" FAIL "setup: $(cat "$d/notes.txt")"; return; }
  fi
  observe; stop_nodes "$d"
  check "policy says mdns=off" policy_has "$d/A.log" "mdns=off" || ok=0
  check "no mDNS instance started" not mdns_started "$d/A.log" || ok=0
  check "no mDNS discovery" not discovered_any "$d/A.log" || ok=0
  check "no connection" test -z "$(connected_peers "$d/A.log")" || ok=0
  [ $ok = 1 ] && record "$sc" PASS "$CHECKS" || record "$sc" FAIL "$CHECKS"
}

sc_dial_peer() { # sc_dial_peer <scenario> <via: flag|env>
  local sc=$1 via=$2 d="$OUT/$1" ok=1 allow
  mkdir -p "$d"; CHECKS=""
  start_node "$d" B "" --mdns off --outbound-allow "$FENCE" &&
  start_node "$d" C "" --mdns off --outbound-allow "$FENCE" || { record "$sc" FAIL "setup: $(cat "$d/notes.txt")"; return; }
  allow=$(prefix_of "$d" C)
  if [ "$via" = flag ]; then
    start_node "$d" A "" --mdns off --outbound-allow "$allow" --peer "$(addr_of "$d" B)" --peer "$(addr_of "$d" C)" \
      || { record "$sc" FAIL "setup: $(cat "$d/notes.txt")"; return; }
  else
    start_node "$d" A "OMNI_OUTBOUND_ALLOW=$allow" --mdns off --peer "$(addr_of "$d" B)" --peer "$(addr_of "$d" C)" \
      || { record "$sc" FAIL "setup: $(cat "$d/notes.txt")"; return; }
  fi
  observe; stop_nodes "$d"
  local B C pb; B=$(peer_of "$d/B.log"); C=$(peer_of "$d/C.log"); pb=$(port_of "$d/B.log")
  check "policy lists C ($allow)" policy_has "$d/A.log" "$allow" || ok=0
  check "A connected to C" connected "$d/A.log" "$C" || ok=0
  check "A never connected to B" not connected "$d/A.log" "$B" || ok=0
  check "B refused at admission (source=bootstrap)" refused "$d/A.log" bootstrap "$pb" || ok=0
  check "A's connections within {C}" test -z "$(connected_peers "$d/A.log" | grep -vx "$C")" || ok=0
  [ $ok = 1 ] && record "$sc" PASS "$CHECKS refusals: $(refusal_tally "$d/A.log")" || record "$sc" FAIL "$CHECKS refusals: $(refusal_tally "$d/A.log")"
}

sc_inbound() {
  local sc=inbound d="$OUT/inbound" ok=1 allow
  mkdir -p "$d"; CHECKS=""
  start_node "$d" C "" --mdns off --outbound-allow "$FENCE" || { record "$sc" FAIL "setup: $(cat "$d/notes.txt")"; return; }
  allow=$(prefix_of "$d" C)
  start_node "$d" A "" --mdns off --outbound-allow "$allow" || { record "$sc" FAIL "setup: $(cat "$d/notes.txt")"; return; }
  start_node "$d" B "" --mdns off --outbound-allow "$FENCE" --peer "$(addr_of "$d" A)" || { record "$sc" FAIL "setup: $(cat "$d/notes.txt")"; return; }
  observe; stop_nodes "$d"
  local B A; B=$(peer_of "$d/B.log"); A=$(peer_of "$d/A.log")
  check "A refused an inbound connection (source=inbound)" refused_any "$d/A.log" inbound || ok=0
  check "A never connected to B" not connected "$d/A.log" "$B" || ok=0
  check "A has no connection at all" test -z "$(connected_peers "$d/A.log")" || ok=0
  local seen="B's own log: connected to A=$(connected "$d/B.log" "$A" && echo yes || echo no)"
  seen="$seen; inbound refusals at: $(refusals "$d/A.log" | grep -E "source=\"?inbound" | grep -o "addr=[^ ]*" | sort -u | tr "\n" " ")"
  [ $ok = 1 ] && record "$sc" PASS "$CHECKS $seen" || record "$sc" FAIL "$CHECKS $seen"
}

sc_kad_identify() {
  local sc=kad-identify d="$OUT/kad-identify" ok=1 allow
  mkdir -p "$d"; CHECKS=""
  start_node "$d" B "" --mdns off --outbound-allow "$FENCE" || { record "$sc" FAIL "setup: $(cat "$d/notes.txt")"; return; }
  start_node "$d" C "" --mdns off --outbound-allow "$FENCE" --peer "$(addr_of "$d" B)" || { record "$sc" FAIL "setup: $(cat "$d/notes.txt")"; return; }
  # Let C connect to B first, so C has B to hand on.
  local i=0; while ! connected "$d/C.log" "$(peer_of "$d/B.log")" && [ $i -lt 100 ]; do sleep 0.1; i=$((i + 1)); done
  allow=$(prefix_of "$d" C)
  start_node "$d" A "" --mdns off --outbound-allow "$allow" --peer "$(addr_of "$d" C)" || { record "$sc" FAIL "setup: $(cat "$d/notes.txt")"; return; }
  observe; stop_nodes "$d"
  local B C pb; B=$(peer_of "$d/B.log"); C=$(peer_of "$d/C.log"); pb=$(port_of "$d/B.log")
  check "C connected to B (precondition)" connected "$d/C.log" "$B" || ok=0
  check "A connected to C" connected "$d/A.log" "$C" || ok=0
  check "A never connected to B" not connected "$d/A.log" "$B" || ok=0
  if [ $ok = 1 ] && ! refused "$d/A.log" 'kad|identify|dial' "$pb"; then
    record "$sc" INCONCLUSIVE "$CHECKS but A logged no refusal of B's address from kad/identify/dial, so the learned-address path was not exercised; refusals: $(refusal_tally "$d/A.log")"
    return
  fi
  check "B's learned address refused (source kad/identify/dial)" refused "$d/A.log" 'kad|identify|dial' "$pb" || ok=0
  [ $ok = 1 ] && record "$sc" PASS "$CHECKS refusals: $(refusal_tally "$d/A.log")" || record "$sc" FAIL "$CHECKS refusals: $(refusal_tally "$d/A.log")"
}

sc_mdns_on_instances() {
  local sc=mdns-on-instances d="$OUT/mdns-on-instances" ok=1
  if [ "$ALLOW_MULTICAST" != 1 ]; then
    record "$sc" SKIPPED "needs --allow-multicast (sends mDNS multicast on this host's LAN interfaces)"; return
  fi
  mkdir -p "$d"; CHECKS=""
  echo "MULTICAST: this scenario ran omni-node with mDNS on (the default); libp2p-mdns sent multicast on this host's LAN interfaces. Outbound dials stayed on --outbound-allow $FENCE." > "$d/MULTICAST-NOTICE.txt"
  start_node "$d" A "" --outbound-allow "$FENCE" || { record "$sc" FAIL "setup: $(cat "$d/notes.txt")"; return; }
  observe; stop_nodes "$d"
  local ifaces; ifaces=$(strip "$d/A.log" | grep 'libp2p_mdns.*creating instance' | grep -o 'address=[0-9a-f.:]*' | tr '\n' ' ')
  echo "multicast interfaces: $ifaces" >> "$d/MULTICAST-NOTICE.txt"
  check "policy says mdns=on (default, no flag)" policy_has "$d/A.log" "mdns=on" || ok=0
  check "mDNS instances started" mdns_started "$d/A.log" || ok=0
  check "no connection outside the fence" test -z "$(connected_peers "$d/A.log")" || ok=0
  [ $ok = 1 ] && record "$sc" PASS "$CHECKS multicast on: $ifaces" || record "$sc" FAIL "$CHECKS multicast on: $ifaces"
}

utc_of() { date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ; }

sleep_check() { # macOS only: a pmset Sleep entry inside the run makes it VOID
  local start=$1 end hits="" d t z e from to
  end=$(date +%s)
  if ! command -v pmset >/dev/null 2>&1; then echo "unavailable (no pmset)"; return; fi
  from=$(date -r "$start" +%Y-%m-%d); to=$(date -r "$end" +%Y-%m-%d)
  while read -r d t z; do
    e=$(date -j -f '%Y-%m-%d %H:%M:%S %z' "$d $t $z" +%s 2>/dev/null) || continue
    if [ "$e" -ge "$start" ] && [ "$e" -le "$end" ]; then hits="$hits $d $t $z;"; fi
  done <<EOF
$(pmset -g log 2>/dev/null | awk -v f="$from" -v t="$to" '$1>=f && $1<=t && $4=="Sleep" {print $1, $2, $3}')
EOF
  if [ -n "$hits" ]; then echo "VOID (system sleep at$hits)"; else echo "clean (no pmset Sleep entry during the run)"; fi
}

if [ "$CMD" = node ]; then
  d="$OUT/node"; mkdir -p "$d"
  start_node "$d" N "" "${NODE_FLAGS[@]+"${NODE_FLAGS[@]}"}" || die "node did not start: $(cat "$d/notes.txt")"
  echo "node address: /ip4/<this host's IP>/udp/$(port_of "$d/N.log")/quic-v1/p2p/$(peer_of "$d/N.log")"
  echo "policy: $(policy_of "$d/N.log")"
  echo "log: $d/N.log (runs for about $((OBSERVE + 20)) s; scan it afterwards with: $0 scan --log $d/N.log --peer <id>)"
  pid=$(cat "$d/N.pid"); i=0
  while kill -0 "$pid" 2>/dev/null && [ $i -lt $(( (OBSERVE + 40) * 5 )) ]; do sleep 0.2; i=$((i + 1)); done
  kill_hard "$pid"
  scan_report "$d/N.log"
  exit 0
fi

# ── run ──────────────────────────────────────────────────────────────────────
STARTED=$(date +%s)
{
  echo "started_utc=$(utc_of "$STARTED")"
  echo "evidence_class=$EVIDENCE_CLASS"
  echo "host=$(hostname 2>/dev/null || echo unknown) $(uname -sm)"
  echo "commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "worktree_changes=$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  echo "omni_node=$OMNI_NODE ($("$OMNI_NODE" --version 2>/dev/null))"
  echo "ip=$IP fence=$FENCE observe_secs=$OBSERVE allow_multicast=$ALLOW_MULTICAST rust_log=$NODE_RUST_LOG"
  echo "scenarios=$SCENARIOS"
} > "$OUT/run.env"

for sc in $SCENARIOS; do
  t0=$(date +%s)
  case "$sc" in
    mdns-off-flag) sc_mdns_off mdns-off-flag flag ;;
    mdns-off-env) sc_mdns_off mdns-off-env env ;;
    dial-peer) sc_dial_peer dial-peer flag ;;
    dial-peer-env) sc_dial_peer dial-peer-env env ;;
    inbound) sc_inbound ;;
    kad-identify) sc_kad_identify ;;
    mdns-on-instances) sc_mdns_on_instances ;;
    *) die "unknown scenario: $sc" ;;
  esac
  # Every scenario is bounded: observe + node startups (20 s each) + 30 s.
  [ $(( $(date +%s) - t0 )) -le $((OBSERVE + 90)) ] || echo "warning: $sc took $(( $(date +%s) - t0 )) s" >&2
done

fails=$(awk -F'\t' '$2=="FAIL"' "$RESULTS" | wc -l | tr -d ' ')
inconcl=$(awk -F'\t' '$2=="INCONCLUSIVE"' "$RESULTS" | wc -l | tr -d ' ')
passes=$(awk -F'\t' '$2=="PASS"' "$RESULTS" | wc -l | tr -d ' ')
verdict=PASS; { [ "$fails" != 0 ] || [ "$inconcl" != 0 ] || [ "$passes" = 0 ]; } && verdict=FAIL
sc_line=$(sleep_check "$STARTED"); case "$sc_line" in VOID*) verdict=VOID ;; esac
{
  cat "$OUT/run.env"
  echo "ended_utc=$(utc_of "$(date +%s)") elapsed_secs=$(( $(date +%s) - STARTED ))"
  awk -F'\t' '{printf "%-18s %s\n", $1, $2}' "$RESULTS"
  echo "pass=$passes fail=$fails inconclusive=$inconcl"
  echo "sleep_check=$sc_line"
  echo "RESULT=$verdict ($EVIDENCE_CLASS)"
} > "$OUT/summary.txt"
cat "$OUT/summary.txt"
[ "$verdict" = PASS ]
