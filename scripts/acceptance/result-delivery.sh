#!/usr/bin/env bash
# Issue #150 acceptance runner: does the result announcement that
# `watch-network-jobs --publish-result-link --max-jobs 1` broadcasts just
# before it exits reach a connected `watch-network-results` peer, when the two
# run as separate processes, on separate hosts, or over a slow link?
#
# It drives the three ignored runner roles in
# crates/omni-contributor/tests/watch_network_jobs_exit_delivery.rs:
#
#   runner::watcher      receiving side; records every run and decides PASS/FAIL
#   runner::contributor  ONE contributor run per process: connect, handshake,
#                        the watch-network-jobs --max-jobs 1 sequence, the
#                        drain (--propagation-wait-ms), process exit
#   runner::link_proxy   userspace UDP forwarder adding one-way delay, jitter
#                        and loss in both directions (emulated slow link)
#
# The contributor side is the command's own library sequence (run_watch_loop,
# OmniNetRelay::publish_result, shutdown_with_drain), with the job taken from
# an in-memory SNIP stand-in, so no SUM Chain or SNIP is needed. Only the
# result announcement (and the handshake) crosses the network. mDNS is off in
# every role.
#
# Modes:
#   loopback     every role on this machine, as separate processes
#   watcher      the watcher only (the receiving host); prints its address
#   contributor  N contributor runs against --watcher-addr (the sending host),
#                through a local link proxy when a link delay/loss is set
#   proxy        the link proxy only
#   build        build (or locate) the test executable and print its path
#
# Parameters (flag, environment variable, default):
#   --runs N                  OMNI_DELIVERY_RUNS                 20
#   --propagation-wait-ms N   OMNI_DELIVERY_PROPAGATION_WAIT_MS  200 (the command's default)
#   --settle-ms N             OMNI_DELIVERY_SETTLE_MS            1000 (connected time before the job)
#   --listen-port P           OMNI_DELIVERY_LISTEN_PORT          0 (OS-assigned; QUIC on 0.0.0.0)
#   --advertise-ip IP         OMNI_DELIVERY_ADVERTISE_IP         127.0.0.1 (watcher: IPv4 it is dialled on)
#   --watcher-addr MA[,MA]    OMNI_DELIVERY_WATCHER_ADDRS        (contributor mode; /ip4/../udp/../quic-v1/p2p/..)
#   --outbound-allow E[,E]    OMNI_DELIVERY_OUTBOUND_ALLOW       loopback: 127.0.0.1,::1; else none
#   --identity-file PATH      OMNI_DELIVERY_IDENTITY_FILE        (none: ephemeral; watcher: stable peer id)
#   --link-delay-ms N         OMNI_DELIVERY_LINK_DELAY_MS        0 (one-way, each direction)
#   --link-jitter-ms N        OMNI_DELIVERY_LINK_JITTER_MS       0 (uniform 0..N added per datagram)
#   --link-loss-pct X         OMNI_DELIVERY_LINK_LOSS_PCT        0 (per datagram, each direction)
#   --link-seed N             OMNI_DELIVERY_LINK_SEED            150
#   --proxy-listen HOST:PORT  OMNI_DELIVERY_PROXY_LISTEN         127.0.0.1:0
#   --proxy-upstream H:P      OMNI_DELIVERY_PROXY_UPSTREAM       (proxy mode)
#   --receipt-timeout-ms N    OMNI_DELIVERY_RECEIPT_TIMEOUT_MS   10000 (after the contributor disconnects)
#   --run-deadline-secs N     OMNI_DELIVERY_RUN_DEADLINE_SECS    derived (below)
#   --start-grace-secs N      OMNI_DELIVERY_START_GRACE_SECS     600 (watcher mode: wait for the first probe)
#   --evidence-class TEXT     OMNI_DELIVERY_EVIDENCE_CLASS       loopback: fixed; else "unattested: ..."
#   --link-note TEXT          OMNI_DELIVERY_LINK_NOTE            (free text: OS shaping applied, path, RTT)
#   --session ID              OMNI_DELIVERY_SESSION              s<UTC>-<pid>
#   --out DIR                 OMNI_DELIVERY_OUT                  ./delivery-runs/<session>-<mode>
#   --test-bin PATH           OMNI_DELIVERY_TEST_BIN             (none: cargo builds it)
#
# Deadlines: nothing here waits without a bound. From a clean baseline (macOS
# arm64, debug build: 1.45 s per run on loopback, 3.6 s per run through a
# 150 ms + 0..20 ms emulated link), one run is expected to take
#   expected_ms = 250 + settle + propagation_wait + 14 x (link delay + jitter)
# and each contributor process gets run_deadline = max(20 s, 10 x expected).
# The watcher and the proxy get runs x run_deadline + 60 s (watcher mode adds
# --start-grace-secs). Each role enforces its own deadline on the wall clock
# (exit 124), and this script kills anything still running at its deadline.
#
# Records (in --out): session.env (parameters, deadlines, commit, host),
# watcher.jsonl, contributor.jsonl, proxy.stats, one log per process,
# summary.txt. PASS: the watcher saw --runs runs and every run's announcement
# arrived with a verifying signature within the receipt timeout, and (where the
# contributor runs here) every contributor process exited 0 after returning on
# max_jobs. On macOS a session that overlaps a system sleep (pmset log) is
# VOID whatever its result. See docs/acceptance-result-delivery.md.
#
# Evidence classes: `loopback` is local evidence (this machine, loopback),
# with an emulated slow link if a delay/loss is set. Separate hosts over a
# real link are external acceptance only when the operator says so
# (--evidence-class); the runner cannot tell.
#
# It never contacts a host it is not given, and needs no root. Real latency
# shaping (tc netem on Linux, dnctl/pfctl on macOS) is the operator's, applied
# outside this script; record it with --link-note.

set -euo pipefail

die() { echo "result-delivery: $*" >&2; exit 2; }

usage() { sed -n '2,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

MODE=${1:-}
[ -n "$MODE" ] || { usage; exit 2; }
shift
case "$MODE" in
  -h|--help|help) usage; exit 0 ;;
  loopback|watcher|contributor|proxy|build) ;;
  *) die "unknown mode: $MODE (loopback|watcher|contributor|proxy|build)" ;;
esac

RUNS=${OMNI_DELIVERY_RUNS:-20}
PROPAGATION_WAIT_MS=${OMNI_DELIVERY_PROPAGATION_WAIT_MS:-200}
SETTLE_MS=${OMNI_DELIVERY_SETTLE_MS:-1000}
LISTEN_PORT=${OMNI_DELIVERY_LISTEN_PORT:-0}
ADVERTISE_IP=${OMNI_DELIVERY_ADVERTISE_IP:-127.0.0.1}
WATCHER_ADDRS=${OMNI_DELIVERY_WATCHER_ADDRS:-}
OUTBOUND_ALLOW=${OMNI_DELIVERY_OUTBOUND_ALLOW:-}
OUTBOUND_ALLOW_SET=${OMNI_DELIVERY_OUTBOUND_ALLOW+x}
IDENTITY_FILE=${OMNI_DELIVERY_IDENTITY_FILE:-}
LINK_DELAY_MS=${OMNI_DELIVERY_LINK_DELAY_MS:-0}
LINK_JITTER_MS=${OMNI_DELIVERY_LINK_JITTER_MS:-0}
LINK_LOSS_PCT=${OMNI_DELIVERY_LINK_LOSS_PCT:-0}
LINK_SEED=${OMNI_DELIVERY_LINK_SEED:-150}
PROXY_LISTEN=${OMNI_DELIVERY_PROXY_LISTEN:-127.0.0.1:0}
PROXY_UPSTREAM=${OMNI_DELIVERY_PROXY_UPSTREAM:-}
RECEIPT_TIMEOUT_MS=${OMNI_DELIVERY_RECEIPT_TIMEOUT_MS:-10000}
RUN_DEADLINE_SECS=${OMNI_DELIVERY_RUN_DEADLINE_SECS:-}
START_GRACE_SECS=${OMNI_DELIVERY_START_GRACE_SECS:-600}
EVIDENCE_CLASS=${OMNI_DELIVERY_EVIDENCE_CLASS:-}
LINK_NOTE=${OMNI_DELIVERY_LINK_NOTE:-}
SESSION=${OMNI_DELIVERY_SESSION:-s$(date -u +%Y%m%dT%H%M%SZ)-$$}
OUT=${OMNI_DELIVERY_OUT:-}
TEST_BIN=${OMNI_DELIVERY_TEST_BIN:-}

while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || die "$1 needs a value"
  case "$1" in
    --runs) RUNS=$2 ;;
    --propagation-wait-ms) PROPAGATION_WAIT_MS=$2 ;;
    --settle-ms) SETTLE_MS=$2 ;;
    --listen-port) LISTEN_PORT=$2 ;;
    --advertise-ip) ADVERTISE_IP=$2 ;;
    --watcher-addr) WATCHER_ADDRS=$2 ;;
    --outbound-allow) OUTBOUND_ALLOW=$2; OUTBOUND_ALLOW_SET=x ;;
    --identity-file) IDENTITY_FILE=$2 ;;
    --link-delay-ms) LINK_DELAY_MS=$2 ;;
    --link-jitter-ms) LINK_JITTER_MS=$2 ;;
    --link-loss-pct) LINK_LOSS_PCT=$2 ;;
    --link-seed) LINK_SEED=$2 ;;
    --proxy-listen) PROXY_LISTEN=$2 ;;
    --proxy-upstream) PROXY_UPSTREAM=$2 ;;
    --receipt-timeout-ms) RECEIPT_TIMEOUT_MS=$2 ;;
    --run-deadline-secs) RUN_DEADLINE_SECS=$2 ;;
    --start-grace-secs) START_GRACE_SECS=$2 ;;
    --evidence-class) EVIDENCE_CLASS=$2 ;;
    --link-note) LINK_NOTE=$2 ;;
    --session) SESSION=$2 ;;
    --out) OUT=$2 ;;
    --test-bin) TEST_BIN=$2 ;;
    *) die "unknown option: $1" ;;
  esac
  shift 2
done

case "$SESSION" in *[[:space:]]*) die "--session must not contain whitespace" ;; esac
for n in "$RUNS" "$PROPAGATION_WAIT_MS" "$SETTLE_MS" "$LISTEN_PORT" "$LINK_DELAY_MS" \
         "$LINK_JITTER_MS" "$LINK_SEED" "$RECEIPT_TIMEOUT_MS" "$START_GRACE_SECS" \
         ${RUN_DEADLINE_SECS:+"$RUN_DEADLINE_SECS"}; do
  case "$n" in ''|*[!0-9]*) die "not a non-negative integer: '$n'" ;; esac
done
case "$LINK_LOSS_PCT" in ''|*[!0-9.]*) die "--link-loss-pct: not a number: '$LINK_LOSS_PCT'" ;; esac
[ "$RUNS" -ge 1 ] || die "--runs must be at least 1"

REPO=$(cd "$(dirname "$0")/../.." && pwd)
SHAPED=0
if [ "$LINK_DELAY_MS" != 0 ] || [ "$LINK_JITTER_MS" != 0 ] || ! awk -v x="$LINK_LOSS_PCT" 'BEGIN{exit !(x==0)}'; then
  SHAPED=1
fi

# ── Deadlines (see the header) ───────────────────────────────────────────────
EXPECTED_RUN_MS=$((250 + SETTLE_MS + PROPAGATION_WAIT_MS + 14 * (LINK_DELAY_MS + LINK_JITTER_MS)))
if [ -z "$RUN_DEADLINE_SECS" ]; then
  RUN_DEADLINE_SECS=$(( (10 * EXPECTED_RUN_MS + 999) / 1000 ))
  [ "$RUN_DEADLINE_SECS" -ge 20 ] || RUN_DEADLINE_SECS=20
fi
SESSION_DEADLINE_SECS=$((RUNS * RUN_DEADLINE_SECS + 60))
IDLE_TIMEOUT_SECS=$((2 * RUN_DEADLINE_SECS))
if [ "$MODE" = watcher ]; then
  SESSION_DEADLINE_SECS=$((SESSION_DEADLINE_SECS + START_GRACE_SECS))
  [ "$START_GRACE_SECS" -gt "$IDLE_TIMEOUT_SECS" ] && IDLE_TIMEOUT_SECS=$START_GRACE_SECS
fi
# A run with no receipt and no disconnect is lost once its deadline has passed.
MAX_RUN_SECS=$RUN_DEADLINE_SECS
CONNECT_TIMEOUT_SECS=$RUN_DEADLINE_SECS

# loopback mode is local by definition: keep every OmniNet dial on loopback.
if [ "$MODE" = loopback ] && [ -z "$OUTBOUND_ALLOW_SET" ]; then
  OUTBOUND_ALLOW="127.0.0.1,::1"
fi

# ── The test executable ──────────────────────────────────────────────────────
locate_test_bin() {
  if [ -n "$TEST_BIN" ]; then
    [ -x "$TEST_BIN" ] || die "--test-bin $TEST_BIN is not executable"
    return
  fi
  local log
  log=$(cd "$REPO" && cargo test --locked -p omni-contributor --features network \
          --test watch_network_jobs_exit_delivery --no-run 2>&1) || { echo "$log" >&2; die "cargo build failed"; }
  TEST_BIN=$(printf '%s\n' "$log" | sed -n 's/.*Executable .*watch_network_jobs_exit_delivery.rs (\(.*\))$/\1/p' | tail -1)
  [ -n "$TEST_BIN" ] || { echo "$log" >&2; die "could not find the test executable in cargo's output"; }
  case "$TEST_BIN" in /*) ;; *) TEST_BIN="$REPO/$TEST_BIN" ;; esac
}

locate_test_bin
if [ "$MODE" = build ]; then echo "$TEST_BIN"; exit 0; fi

# ── Session directory and record header ──────────────────────────────────────
[ -n "$OUT" ] || OUT="$PWD/delivery-runs/$SESSION-$MODE"
[ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && die "--out $OUT exists and is not empty"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

if [ "$MODE" = loopback ]; then
  EVIDENCE_CLASS="local evidence (this machine, loopback)"
  [ "$SHAPED" = 1 ] && EVIDENCE_CLASS="$EVIDENCE_CLASS; emulated slow link (userspace proxy: ${LINK_DELAY_MS} ms one-way + 0..${LINK_JITTER_MS} ms jitter, ${LINK_LOSS_PCT}% loss each direction)"
  EVIDENCE_CLASS="$EVIDENCE_CLASS; NOT external acceptance"
elif [ -z "$EVIDENCE_CLASS" ]; then
  EVIDENCE_CLASS="unattested: the operator must state whether the hosts and the link were real"
fi

utc_of() { date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ; }

STARTED_EPOCH=$(date +%s)
{
  echo "mode=$MODE"
  echo "session=$SESSION"
  echo "started_utc=$(utc_of "$STARTED_EPOCH")"
  echo "evidence_class=$EVIDENCE_CLASS"
  echo "link_note=$LINK_NOTE"
  echo "host=$(hostname 2>/dev/null || echo unknown) $(uname -sm)"
  echo "commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "worktree_changes=$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  echo "test_bin=$TEST_BIN"
  echo "runs=$RUNS propagation_wait_ms=$PROPAGATION_WAIT_MS settle_ms=$SETTLE_MS listen_port=$LISTEN_PORT mdns=off"
  echo "advertise_ip=$ADVERTISE_IP watcher_addrs=$WATCHER_ADDRS outbound_allow=$OUTBOUND_ALLOW identity_file=$IDENTITY_FILE"
  echo "link_delay_ms=$LINK_DELAY_MS link_jitter_ms=$LINK_JITTER_MS link_loss_pct=$LINK_LOSS_PCT link_seed=$LINK_SEED"
  echo "proxy_listen=$PROXY_LISTEN proxy_upstream=$PROXY_UPSTREAM receipt_timeout_ms=$RECEIPT_TIMEOUT_MS"
  echo "expected_run_ms=$EXPECTED_RUN_MS run_deadline_secs=$RUN_DEADLINE_SECS session_deadline_secs=$SESSION_DEADLINE_SECS idle_timeout_secs=$IDLE_TIMEOUT_SECS"
} > "$OUT/session.env"

export RUST_LOG=${RUST_LOG:-warn}
PIDS=""

kill_hard() { # kill_hard <pid>: TERM, then KILL if it is still there 2 s later
  kill -TERM "$1" 2>/dev/null || return 0
  local i=0
  while kill -0 "$1" 2>/dev/null && [ $i -lt 20 ]; do sleep 0.1; i=$((i + 1)); done
  kill -KILL "$1" 2>/dev/null || true
}

cleanup() { for p in $PIDS; do kill_hard "$p"; done; }
trap cleanup EXIT
trap 'exit 130' INT TERM

# start_role <name> <log> [VAR=value ...]: the role in its own background
# process. `exec` makes $! the test executable itself, so a kill reaches it.
start_role() {
  local name=$1 log=$2; shift 2
  ( exec env "$@" "$TEST_BIN" --exact "runner::$name" --ignored --nocapture --test-threads=1 ) > "$log" 2>&1 &
  LAST_PID=$!
  PIDS="$PIDS $LAST_PID"
}

# wait_with_deadline <pid> <secs> <what>: the process's exit status, or 124
# after killing it at the deadline (wall clock).
wait_with_deadline() {
  local pid=$1 limit=$2 what=$3 start rc=0
  start=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    if [ $(( $(date +%s) - start )) -ge "$limit" ]; then
      echo "result-delivery: DEADLINE EXCEEDED: $what (pid $pid) still running after ${limit}s; killed" >&2
      kill_hard "$pid"
      wait "$pid" 2>/dev/null || true
      return 124
    fi
    sleep 0.2
  done
  wait "$pid" || rc=$?
  return $rc
}

wait_for_file() { # wait_for_file <path> <what> <pid>
  local i=0
  while [ ! -s "$1" ]; do
    kill -0 "$3" 2>/dev/null || die "$2 exited before it was ready (see $OUT)"
    i=$((i + 1)); [ $i -le 600 ] || die "$2 not ready after 60 s"
    sleep 0.1
  done
}

start_watcher() {
  start_role watcher "$OUT/watcher.log" \
    OMNI_DELIVERY_RECORD="$OUT/watcher.jsonl" OMNI_DELIVERY_RUNS="$RUNS" \
    OMNI_DELIVERY_ADDR_FILE="$OUT/watcher.addr" OMNI_DELIVERY_LISTEN_PORT="$LISTEN_PORT" \
    OMNI_DELIVERY_ADVERTISE_IP="$ADVERTISE_IP" OMNI_DELIVERY_IDENTITY_FILE="$IDENTITY_FILE" \
    OMNI_DELIVERY_OUTBOUND_ALLOW="$OUTBOUND_ALLOW" OMNI_DELIVERY_MDNS=off \
    OMNI_DELIVERY_RECEIPT_TIMEOUT_MS="$RECEIPT_TIMEOUT_MS" OMNI_DELIVERY_MAX_RUN_SECS="$MAX_RUN_SECS" \
    OMNI_DELIVERY_IDLE_TIMEOUT_SECS="$IDLE_TIMEOUT_SECS" OMNI_DELIVERY_EVIDENCE_CLASS="$EVIDENCE_CLASS" \
    OMNI_DELIVERY_DEADLINE_SECS="$SESSION_DEADLINE_SECS"
  WATCHER_PID=$LAST_PID
  wait_for_file "$OUT/watcher.addr" watcher "$WATCHER_PID"
}

start_proxy() { # start_proxy <upstream host:port>
  start_role link_proxy "$OUT/proxy.log" \
    OMNI_DELIVERY_PROXY_UPSTREAM="$1" OMNI_DELIVERY_PROXY_LISTEN="$PROXY_LISTEN" \
    OMNI_DELIVERY_PROXY_ADDR_FILE="$OUT/proxy.addr" OMNI_DELIVERY_PROXY_STATS_FILE="$OUT/proxy.stats" \
    OMNI_DELIVERY_LINK_DELAY_MS="$LINK_DELAY_MS" OMNI_DELIVERY_LINK_JITTER_MS="$LINK_JITTER_MS" \
    OMNI_DELIVERY_LINK_LOSS_PCT="$LINK_LOSS_PCT" OMNI_DELIVERY_LINK_SEED="$LINK_SEED" \
    OMNI_DELIVERY_DEADLINE_SECS="$SESSION_DEADLINE_SECS"
  PROXY_PID=$LAST_PID
  wait_for_file "$OUT/proxy.addr" "link proxy" "$PROXY_PID"
}

stop_proxy() {
  if [ -n "${PROXY_PID:-}" ]; then kill_hard "$PROXY_PID"; wait "$PROXY_PID" 2>/dev/null || true; fi
}

# /ip4/<ip>/udp/<port>/quic-v1/p2p/<peer> → "<ip> <port> <peer>"
split_addr() {
  echo "$1" | awk -F/ '$2=="ip4" && $4=="udp" && $6=="quic-v1" && $7=="p2p" {print $3, $5, $8; ok=1} END {exit !ok}' \
    || die "not an /ip4/<ip>/udp/<port>/quic-v1/p2p/<peer-id> address: $1"
}

# Run the contributor processes, one at a time, each within its deadline.
# Sets CONTRIB_FAILS and CONTRIB_DEADLINES.
run_contributors() { # run_contributors <dial addrs> <outbound allow (may be empty)>
  CONTRIB_FAILS=0; CONTRIB_DEADLINES=0
  local r=0 rc
  while [ $r -lt "$RUNS" ]; do
    start_role contributor "$OUT/contributor-$r.log" \
      OMNI_DELIVERY_WATCHER_ADDRS="$1" OMNI_DELIVERY_OUTBOUND_ALLOW="$2" OMNI_DELIVERY_MDNS=off \
      OMNI_DELIVERY_RECORD="$OUT/contributor.jsonl" OMNI_DELIVERY_SESSION="$SESSION" \
      OMNI_DELIVERY_RUN="$r" OMNI_DELIVERY_PROPAGATION_WAIT_MS="$PROPAGATION_WAIT_MS" \
      OMNI_DELIVERY_SETTLE_MS="$SETTLE_MS" OMNI_DELIVERY_CONNECT_TIMEOUT_SECS="$CONNECT_TIMEOUT_SECS" \
      OMNI_DELIVERY_LISTEN_PORT=0 OMNI_DELIVERY_DEADLINE_SECS="$RUN_DEADLINE_SECS"
    rc=0
    wait_with_deadline "$LAST_PID" $((RUN_DEADLINE_SECS + 5)) "contributor run $r" || rc=$?
    if [ $rc = 0 ]; then
      echo "run $r: contributor exited 0"
    else
      echo "run $r: contributor FAILED with status $rc (see $OUT/contributor-$r.log)"
      CONTRIB_FAILS=$((CONTRIB_FAILS + 1))
      [ $rc = 124 ] && CONTRIB_DEADLINES=$((CONTRIB_DEADLINES + 1))
    fi
    r=$((r + 1))
  done
}

# The dial address and allow-list for the contributor side: direct, or through
# a local link proxy when the link is shaped (one watcher only).
contributor_route() { # sets DIAL and ALLOW
  DIAL=$WATCHER_ADDRS; ALLOW=$OUTBOUND_ALLOW
  if [ "$SHAPED" = 1 ]; then
    case "$WATCHER_ADDRS" in *,*) die "an emulated slow link takes exactly one --watcher-addr" ;; esac
    local parts
    parts=$(split_addr "$WATCHER_ADDRS") || exit 2
    set -- $parts
    start_proxy "$1:$2"
    local pport; pport=$(sed 's/.*://' "$OUT/proxy.addr")
    DIAL="/ip4/127.0.0.1/udp/$pport/quic-v1/p2p/$3"
    # Keep the contributor on the proxy: any other dial is refused (#158).
    ALLOW="/ip4/127.0.0.1/udp/$pport/quic-v1"
  fi
}

contributor_summary() {
  local f="$OUT/contributor.jsonl" n=0 exited=0
  [ -f "$f" ] && n=$(grep -c '"kind":"contributor_run"' "$f" || true)
  [ -f "$f" ] && exited=$(grep -c '"exited_on_max_jobs":true' "$f" || true)
  echo "contributor_records=$n exited_on_max_jobs=$exited processes_failed=$CONTRIB_FAILS deadline_exceeded=$CONTRIB_DEADLINES"
  if [ -f "$f" ]; then
    echo "drain_outcomes: $(grep -o '"outcome":"[a-z_]*"' "$f" | sed 's/.*:"\(.*\)"/\1/' | sort | uniq -c | awk '{printf "%s=%s ", $2, $1}')"
    echo "probe_ack_rtt_ms: $(grep -o '"probe_ack_rtt_ms":[0-9]*' "$f" | sed 's/.*://' | sort -n | awk '{a[NR]=$1} END {if (NR) printf "min=%s median=%s max=%s", a[1], a[int((NR+1)/2)], a[NR]}')"
  fi
  [ "$n" = "$RUNS" ] && [ "$exited" = "$RUNS" ] && [ "$CONTRIB_FAILS" = 0 ]
}

watcher_verdict() {
  local v; v=$(grep -h 'delivery-runner verdict=' "$OUT/watcher.log" | tail -1 || true)
  echo "${v:-no verdict (see watcher.log)}"
}

# macOS: was the machine asleep at any point of the session? A session that
# overlaps a sleep is VOID. Elsewhere this check is unavailable.
sleep_check() {
  local end hits="" d t z e from to
  end=$(date +%s)
  if ! command -v pmset >/dev/null 2>&1; then echo "unavailable (no pmset)"; return; fi
  from=$(date -r "$STARTED_EPOCH" +%Y-%m-%d); to=$(date -r "$end" +%Y-%m-%d)
  while read -r d t z; do
    e=$(date -j -f '%Y-%m-%d %H:%M:%S %z' "$d $t $z" +%s 2>/dev/null) || continue
    if [ "$e" -ge "$STARTED_EPOCH" ] && [ "$e" -le "$end" ]; then hits="$hits $d $t $z;"; fi
  done <<EOF
$(pmset -g log 2>/dev/null | awk -v f="$from" -v t="$to" '$1>=f && $1<=t && $4=="Sleep" {print $1, $2, $3}')
EOF
  if [ -n "$hits" ]; then echo "VOID (system sleep at$hits)"; else echo "clean (no pmset Sleep entry during the session)"; fi
}

finish() { # finish <PASS|FAIL>: the summary's last lines; a sleep makes it VOID
  local verdict=$1 sc
  sc=$(sleep_check)
  case "$sc" in VOID*) verdict=VOID ;; esac
  echo "ended_utc=$(utc_of "$(date +%s)") elapsed_secs=$(( $(date +%s) - STARTED_EPOCH ))"
  echo "sleep_check=$sc"
  echo "RESULT=$verdict ($EVIDENCE_CLASS)"
}

case "$MODE" in
  proxy)
    [ -n "$PROXY_UPSTREAM" ] || die "proxy mode needs --proxy-upstream HOST:PORT"
    start_proxy "$PROXY_UPSTREAM"
    echo "link proxy listening on $(cat "$OUT/proxy.addr") -> $PROXY_UPSTREAM (deadline ${SESSION_DEADLINE_SECS}s)"
    rc=0; wait_with_deadline "$PROXY_PID" $((SESSION_DEADLINE_SECS + 15)) "link proxy" || rc=$?
    exit $rc
    ;;

  watcher)
    start_watcher
    echo "watcher address: $(cat "$OUT/watcher.addr")"
    echo "on the contributor host: $0 contributor --watcher-addr $(cat "$OUT/watcher.addr") --runs $RUNS"
    rc=0; wait_with_deadline "$WATCHER_PID" $((SESSION_DEADLINE_SECS + 15)) watcher || rc=$?
    verdict=FAIL; [ $rc = 0 ] && verdict=PASS
    { cat "$OUT/session.env"; echo "watcher_exit=$rc"; echo "watcher: $(watcher_verdict)"; finish $verdict; } > "$OUT/summary.txt"
    cat "$OUT/summary.txt"
    grep -q '^RESULT=PASS' "$OUT/summary.txt"
    ;;

  contributor)
    [ -n "$WATCHER_ADDRS" ] || die "contributor mode needs --watcher-addr"
    contributor_route
    run_contributors "$DIAL" "$ALLOW"
    stop_proxy
    ok=0; contributor_summary > "$OUT/contributors.txt" || ok=1
    verdict=FAIL; [ $ok = 0 ] && verdict=PASS
    {
      cat "$OUT/session.env"; cat "$OUT/contributors.txt"
      [ -f "$OUT/proxy.stats" ] && echo "proxy: $(cat "$OUT/proxy.stats")"
      echo "delivery verdict: decided by the watcher host (its summary.txt); RESULT below covers this side only"
      finish $verdict
    } > "$OUT/summary.txt"
    cat "$OUT/summary.txt"
    grep -q '^RESULT=PASS' "$OUT/summary.txt"
    ;;

  loopback)
    start_watcher
    WATCHER_ADDRS=$(cat "$OUT/watcher.addr")
    contributor_route
    run_contributors "$DIAL" "$ALLOW"
    wrc=0; wait_with_deadline "$WATCHER_PID" $((SESSION_DEADLINE_SECS + 15)) watcher || wrc=$?
    stop_proxy
    crc=0; contributor_summary > "$OUT/contributors.txt" || crc=1
    verdict=FAIL; [ $wrc = 0 ] && [ $crc = 0 ] && verdict=PASS
    {
      cat "$OUT/session.env"
      cat "$OUT/contributors.txt"
      [ -f "$OUT/proxy.stats" ] && echo "proxy: $(cat "$OUT/proxy.stats")"
      echo "watcher_exit=$wrc"
      echo "watcher: $(watcher_verdict)"
      finish $verdict
    } > "$OUT/summary.txt"
    cat "$OUT/summary.txt"
    grep -q '^RESULT=PASS' "$OUT/summary.txt"
    ;;
esac
