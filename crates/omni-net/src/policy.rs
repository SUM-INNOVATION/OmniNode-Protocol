//! OmniNet's outbound address allow-list (issue #158).
//!
//! The design and the entry syntax are SNIP's (`sum_net::policy`, SNIP #79 /
//! #80), so one allow-list reads the same in both stacks. Like SNIP's, this is
//! an **application-level routing control for cooperative networks**: it
//! keeps an honest `omni-node` on the endpoints its operator configured. It is
//! **not an operating-system security boundary**: a modified client can ignore
//! it, and nothing here constrains another process on the host.
//!
//! # Where it is enforced
//!
//! With no allow-list (the default) nothing is checked and the node behaves as
//! before. With one:
//!
//! 1. **The transport.** [`crate::gate::OutboundGate`] wraps the whole
//!    transport stack (relay client and QUIC) and judges every address handed
//!    to `Transport::dial`, after the swarm has merged the addresses every
//!    behaviour contributed. So no dial reaches a socket unless it passes,
//!    whichever part of the node asked for it: an explicit `--peer` dial,
//!    mDNS, Identify, Kademlia, request-response, gossipsub, AutoNAT, DCUtR
//!    hole-punching or a relay-client dial to a relay. A refused dial fails
//!    with `PermissionDenied` before any packet is sent. This is the decisive
//!    control.
//! 2. **The peer stores.** A refused address is dropped where it is learned,
//!    so it is never stored and never offered for a dial again: `--peer`
//!    bootstrap entries, mDNS discoveries (a peer with no allowed address is
//!    not made a gossipsub explicit peer or reported as discovered), Identify
//!    listen addresses (not fed to Kademlia) and Kademlia routing updates
//!    (removed from the routing table). The address-supplying behaviours
//!    (mDNS, Identify, Kademlia) are wrapped in [`crate::gate::Gated`], which
//!    filters the addresses they contribute to a dial and withholds refused
//!    peer addresses they publish to other behaviours.
//! 3. **Inbound connections.** [`crate::gate::PolicyBehaviour`] keeps a direct
//!    inbound connection only from an allowed address, and a relayed one only
//!    through an allowed relay. Without it an allowed peer could open the
//!    second route itself by dialling in from an address that is not allowed.
//!
//! Every refusal is counted by source ([`PolicyRefusals`]) and logged: at
//! `warn` the first time a source refuses an address, at `debug` after that.
//! The effective policy is logged once when the swarm is built
//! ([`NetPolicy::describe`]).
//!
//! # Matching rules
//!
//! An entry is either an IP network or a multiaddr prefix.
//!
//! * **IP network**: `127.0.0.1`, `10.0.0.0/8`, `::1`, `2001:db8::/32`. A
//!   bare address is a single host (`/32` or `/128`). It matches an address
//!   whose host is `/ip4` or `/ip6` inside the network, on any transport and
//!   port, for any peer. An IPv4-mapped IPv6 host (`::ffff:a.b.c.d`) also
//!   matches IPv4 networks.
//! * **Multiaddr prefix**: a host (`/ip4`, `/ip6`, `/dns`, `/dns4`, `/dns6`,
//!   `/dnsaddr`), then optionally `/tcp/<port>`, `/udp/<port>` or
//!   `/udp/<port>/quic-v1`, then optionally `/p2p/<peer-id>`. It matches an
//!   address that begins with the same components (DNS names compare
//!   case-insensitively). An entry ending in `/p2p/<peer-id>` also requires
//!   the dial to name that peer; libp2p then refuses the connection if the
//!   remote proves to be someone else.
//!
//! A relayed address (`<relay address>/p2p-circuit/p2p/<peer>`) is judged by
//! its relay part alone, because that is where the connection goes; the dial
//! to the relay is judged again on its own. Entries cannot contain
//! `/p2p-circuit`, and an entry that names only a peer (`/p2p/<peer-id>`) is
//! refused: an allowed peer can advertise any address. An address that does
//! not start with a host (`/unix`, `/memory`, ...) is never allowed while an
//! allow-list is set.

use std::collections::HashSet;
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use libp2p::multiaddr::Protocol;
use libp2p::{Multiaddr, PeerId};
use omni_types::config::NetConfig;
use tracing::{debug, info, warn};

/// The CLI flag that carries the allow-list, named in errors and refusals so
/// an operator can find the knob that produced them.
pub const OUTBOUND_ALLOW_FLAG: &str = "--outbound-allow";

// ── Parse errors ─────────────────────────────────────────────────────────────

/// A malformed allow-list entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyParseError {
    entry: String,
    reason: String,
}

impl PolicyParseError {
    fn new(entry: &str, reason: impl Into<String>) -> Self {
        Self {
            entry: entry.to_string(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for PolicyParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid {OUTBOUND_ALLOW_FLAG} entry `{}`: {}",
            self.entry, self.reason
        )
    }
}

impl std::error::Error for PolicyParseError {}

// ── Entries ──────────────────────────────────────────────────────────────────

/// One allow-list entry. See the module documentation for the matching rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllowEntry {
    /// An IP network. `addr` is stored with the host bits cleared.
    Network { addr: IpAddr, prefix: u8 },
    /// A multiaddr prefix: a host, optionally a port (and `quic-v1`), and an
    /// optional peer pin taken from a trailing `/p2p/<peer-id>`.
    Prefix {
        components: Vec<Protocol<'static>>,
        peer: Option<PeerId>,
    },
}

impl FromStr for AllowEntry {
    type Err = PolicyParseError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let s = raw.trim();
        if s.is_empty() {
            return Err(PolicyParseError::new(raw, "empty entry"));
        }
        let parsed = if s.starts_with('/') {
            parse_prefix(s)
        } else {
            parse_network(s)
        };
        parsed.map_err(|reason| PolicyParseError::new(s, reason))
    }
}

impl fmt::Display for AllowEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllowEntry::Network { addr, prefix } => write!(f, "{addr}/{prefix}"),
            AllowEntry::Prefix { components, peer } => {
                let mut ma: Multiaddr = components.iter().cloned().collect();
                if let Some(p) = peer {
                    ma.push(Protocol::P2p(*p));
                }
                write!(f, "{ma}")
            }
        }
    }
}

fn parse_network(s: &str) -> Result<AllowEntry, String> {
    let (ip_part, prefix_part) = match s.split_once('/') {
        Some((ip, prefix)) => (ip, Some(prefix)),
        None => (s, None),
    };
    let addr: IpAddr = ip_part.parse().map_err(|_| {
        format!(
            "`{ip_part}` is not an IP address or CIDR (multiaddr entries start with `/`, \
             e.g. /ip4/127.0.0.1/udp/4001/quic-v1)"
        )
    })?;
    let max = if addr.is_ipv4() { 32 } else { 128 };
    let prefix = match prefix_part {
        None => max,
        Some(p) => {
            let n: u8 = p
                .parse()
                .map_err(|_| format!("`{p}` is not a prefix length"))?;
            if n > max {
                return Err(format!("prefix length {n} exceeds {max}"));
            }
            n
        }
    };
    Ok(AllowEntry::Network {
        addr: mask(addr, prefix),
        prefix,
    })
}

fn parse_prefix(s: &str) -> Result<AllowEntry, String> {
    let ma: Multiaddr = s
        .parse()
        .map_err(|e| format!("not a valid multiaddr ({e})"))?;
    let all: Vec<Protocol<'static>> = ma.iter().map(|p| p.acquire()).collect();

    if all.iter().any(|p| matches!(p, Protocol::P2pCircuit)) {
        return Err(
            "relayed (/p2p-circuit) addresses cannot be listed; list the relay's own address \
             instead (a relayed address is allowed when its relay part is)"
                .into(),
        );
    }

    let (body, peer) = match all.split_last() {
        Some((Protocol::P2p(p), rest)) => (rest.to_vec(), Some(*p)),
        _ => (all.clone(), None),
    };

    let Some(host) = body.first() else {
        return Err(
            "a peer ID alone does not restrict addresses (an allowed peer can advertise any \
             address); name the host too, e.g. /ip4/127.0.0.1/udp/4001/quic-v1/p2p/<peer-id>"
                .into(),
        );
    };
    if !is_host(host) {
        return Err(format!(
            "must start with /ip4, /ip6, /dns, /dns4, /dns6 or /dnsaddr, not /{}",
            host.tag()
        ));
    }

    match &body[1..] {
        [] => {}
        [Protocol::Tcp(port)] | [Protocol::Udp(port)] | [Protocol::Udp(port), Protocol::QuicV1] => {
            if *port == 0 {
                return Err(
                    "port 0 is not a wildcard; leave the port out to allow every port on the host"
                        .into(),
                );
            }
        }
        _ => {
            return Err(
                "unsupported form after the host: an entry may continue with /tcp/<port>, \
                 /udp/<port> or /udp/<port>/quic-v1, then an optional final /p2p/<peer-id>"
                    .into(),
            );
        }
    }

    Ok(AllowEntry::Prefix {
        components: body,
        peer,
    })
}

fn is_host(p: &Protocol<'_>) -> bool {
    matches!(
        p,
        Protocol::Ip4(_)
            | Protocol::Ip6(_)
            | Protocol::Dns(_)
            | Protocol::Dns4(_)
            | Protocol::Dns6(_)
            | Protocol::Dnsaddr(_)
    )
}

fn mask(addr: IpAddr, prefix: u8) -> IpAddr {
    match addr {
        IpAddr::V4(v4) => {
            let m = if prefix == 0 {
                0
            } else {
                u32::MAX << (32 - u32::from(prefix))
            };
            IpAddr::V4(Ipv4Addr::from(u32::from(v4) & m))
        }
        IpAddr::V6(v6) => {
            let m = if prefix == 0 {
                0
            } else {
                u128::MAX << (128 - u32::from(prefix))
            };
            IpAddr::V6(Ipv6Addr::from(u128::from(v6) & m))
        }
    }
}

fn network_contains(net: IpAddr, prefix: u8, ip: IpAddr) -> bool {
    match (net, ip) {
        (IpAddr::V4(_), IpAddr::V4(_)) | (IpAddr::V6(_), IpAddr::V6(_)) => mask(ip, prefix) == net,
        (IpAddr::V4(_), IpAddr::V6(v6)) => v6
            .to_ipv4_mapped()
            .is_some_and(|v4| mask(IpAddr::V4(v4), prefix) == net),
        (IpAddr::V6(_), IpAddr::V4(_)) => false,
    }
}

fn dns_eq(a: &str, b: &str) -> bool {
    a.trim_end_matches('.')
        .eq_ignore_ascii_case(b.trim_end_matches('.'))
}

fn same_component(entry: &Protocol<'_>, addr: &Protocol<'_>) -> bool {
    match (entry, addr) {
        (Protocol::Dns(a), Protocol::Dns(b))
        | (Protocol::Dns4(a), Protocol::Dns4(b))
        | (Protocol::Dns6(a), Protocol::Dns6(b))
        | (Protocol::Dnsaddr(a), Protocol::Dnsaddr(b)) => dns_eq(a, b),
        _ => entry == addr,
    }
}

/// How a peer pin on an entry is treated when the remote peer is unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PeerPin {
    /// The address (or the dial) must name the pinned peer.
    Enforce,
    /// The peer is not known yet (an inbound connection before its
    /// handshake). A pinned entry may still match once it is, so only what no
    /// entry could ever allow is refused now.
    Defer,
}

impl AllowEntry {
    /// Does this entry allow `transport` (an address without its `/p2p`
    /// suffix) for `peer`, the peer the address names, if any?
    fn matches(&self, transport: &[Protocol<'_>], peer: Option<PeerId>, pin: PeerPin) -> bool {
        match self {
            AllowEntry::Network { addr, prefix } => match transport.first() {
                Some(Protocol::Ip4(ip)) => network_contains(*addr, *prefix, IpAddr::V4(*ip)),
                Some(Protocol::Ip6(ip)) => network_contains(*addr, *prefix, IpAddr::V6(*ip)),
                _ => false,
            },
            AllowEntry::Prefix {
                components,
                peer: wanted,
            } => {
                let prefix_ok = components.len() <= transport.len()
                    && components
                        .iter()
                        .zip(transport)
                        .all(|(e, a)| same_component(e, a));
                let peer_ok = match (wanted, peer) {
                    (None, _) => true,
                    (Some(w), Some(p)) => *w == p,
                    (Some(_), None) => pin == PeerPin::Defer,
                };
                prefix_ok && peer_ok
            }
        }
    }
}

// ── The allow-list ───────────────────────────────────────────────────────────

/// A parsed, non-empty outbound allow-list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutboundAllowList {
    entries: Vec<AllowEntry>,
}

impl OutboundAllowList {
    /// Parse a list of entries, such as [`omni_types::config::NetConfig::outbound_allow`].
    /// Blank items (from a trailing comma, say) are skipped; a list with no
    /// entries left is "no restriction" (`None`), the default.
    pub fn parse<I, S>(items: I) -> Result<Option<Self>, PolicyParseError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut entries = Vec::new();
        for item in items {
            let item = item.as_ref();
            if item.trim().is_empty() {
                continue;
            }
            entries.push(item.parse::<AllowEntry>()?);
        }
        Ok((!entries.is_empty()).then_some(Self { entries }))
    }

    /// The parsed entries, in the order given.
    pub fn entries(&self) -> &[AllowEntry] {
        &self.entries
    }

    /// May the node dial `addr`?
    ///
    /// A relayed address is judged by its relay part. Otherwise the address
    /// is judged as given, with the peer named by a trailing `/p2p/<peer-id>`
    /// (the swarm appends it before the transport sees a dial to a known
    /// peer).
    pub fn allows(&self, addr: &Multiaddr) -> bool {
        self.decide(addr, None, PeerPin::Enforce)
    }

    /// Like [`Self::allows`], for an address learned for `peer` (which the
    /// address itself may not name).
    pub fn allows_for(&self, addr: &Multiaddr, peer: Option<PeerId>) -> bool {
        self.decide(addr, peer, PeerPin::Enforce)
    }

    fn decide(&self, addr: &Multiaddr, peer_hint: Option<PeerId>, pin: PeerPin) -> bool {
        let comps: Vec<Protocol<'_>> = addr.iter().collect();
        let (segment, relayed) = match comps.iter().position(|p| matches!(p, Protocol::P2pCircuit))
        {
            Some(i) => (&comps[..i], true),
            None => (&comps[..], false),
        };
        let (transport, peer) = match segment.split_last() {
            Some((Protocol::P2p(p), rest)) => (rest, Some(*p)),
            // The hint names the destination; a relayed address goes to the
            // relay, so the hint says nothing about it.
            _ => (segment, if relayed { None } else { peer_hint }),
        };
        let Some(host) = transport.first() else {
            return false;
        };
        if !is_host(host) || transport.iter().any(|p| matches!(p, Protocol::P2p(_))) {
            return false;
        }
        self.entries.iter().any(|e| e.matches(transport, peer, pin))
    }
}

impl fmt::Display for OutboundAllowList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, e) in self.entries.iter().enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            write!(f, "{e}")?;
        }
        Ok(())
    }
}

// ── Sources and counters ─────────────────────────────────────────────────────

/// Where a refused address or connection came from. A fixed set, so the
/// counters stay bounded whatever remote peers send.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolicySource {
    /// A `--peer` (bootstrap) entry.
    Bootstrap,
    /// mDNS discovery.
    Mdns,
    /// Listen addresses a peer advertised through Identify.
    Identify,
    /// The Kademlia routing table or its query results.
    Kad,
    /// A connection a remote peer opened to this node.
    Inbound,
    /// A dial refused at the transport: an address no earlier filter judged,
    /// such as an explicit dial, an AutoNAT dial-back, a DCUtR hole-punch or
    /// a relay-client dial.
    Dial,
}

impl PolicySource {
    /// Every source, in counter order.
    pub const ALL: [PolicySource; 6] = [
        PolicySource::Bootstrap,
        PolicySource::Mdns,
        PolicySource::Identify,
        PolicySource::Kad,
        PolicySource::Inbound,
        PolicySource::Dial,
    ];

    /// Stable lower-case label used in logs.
    pub fn label(self) -> &'static str {
        match self {
            PolicySource::Bootstrap => "bootstrap",
            PolicySource::Mdns => "mdns",
            PolicySource::Identify => "identify",
            PolicySource::Kad => "kad",
            PolicySource::Inbound => "inbound",
            PolicySource::Dial => "dial",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// At which point a refusal happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefusalStage {
    /// An address was learned and dropped before it was stored or offered
    /// for any dial.
    Admission,
    /// An address was refused as a dial target.
    Dial,
    /// A connection opened by a remote peer was refused.
    Inbound,
}

impl RefusalStage {
    /// Stable lower-case label used in logs.
    pub fn label(self) -> &'static str {
        match self {
            RefusalStage::Admission => "admission",
            RefusalStage::Dial => "dial",
            RefusalStage::Inbound => "inbound",
        }
    }
}

/// Refusal counts by source. Counts refusals, not distinct addresses: an
/// address offered again is counted again.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PolicyRefusals {
    pub bootstrap: u64,
    pub mdns: u64,
    pub identify: u64,
    pub kad: u64,
    pub inbound: u64,
    pub dial: u64,
}

impl PolicyRefusals {
    /// Sum over every source.
    pub fn total(&self) -> u64 {
        PolicySource::ALL.iter().map(|s| self.get(*s)).sum()
    }

    /// The count for one source.
    pub fn get(&self, source: PolicySource) -> u64 {
        match source {
            PolicySource::Bootstrap => self.bootstrap,
            PolicySource::Mdns => self.mdns,
            PolicySource::Identify => self.identify,
            PolicySource::Kad => self.kad,
            PolicySource::Inbound => self.inbound,
            PolicySource::Dial => self.dial,
        }
    }
}

/// How many distinct `(source, address)` refusals are logged at `warn` before
/// the rest are logged at `debug` only. Counters are never capped.
const WARN_DEDUP_CAP: usize = 1024;

// ── The policy a swarm runs under ────────────────────────────────────────────

/// The network policy one OmniNet swarm runs under, shared by the transport
/// gate, the policy behaviours and the swarm's event handling.
#[derive(Debug)]
pub struct NetPolicy {
    mdns: bool,
    allow: Option<OutboundAllowList>,
    refusals: [AtomicU64; PolicySource::ALL.len()],
    warned: Mutex<HashSet<(PolicySource, Multiaddr)>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl NetPolicy {
    /// A policy from its parts.
    pub fn new(mdns: bool, allow: Option<OutboundAllowList>) -> Self {
        Self {
            mdns,
            allow,
            refusals: Default::default(),
            warned: Mutex::new(HashSet::new()),
        }
    }

    /// The default: mDNS on, no restriction.
    pub fn unrestricted() -> Self {
        Self::new(true, None)
    }

    /// The policy in a [`NetConfig`]. Fails on a malformed allow-list entry.
    pub fn from_config(config: &NetConfig) -> Result<Self, PolicyParseError> {
        Ok(Self::new(
            config.mdns,
            OutboundAllowList::parse(&config.outbound_allow)?,
        ))
    }

    /// Whether mDNS discovery runs.
    pub fn mdns_enabled(&self) -> bool {
        self.mdns
    }

    /// The allow-list, if one is set.
    pub fn allow_list(&self) -> Option<&OutboundAllowList> {
        self.allow.as_ref()
    }

    /// Whether an allow-list is in force.
    pub fn is_restricted(&self) -> bool {
        self.allow.is_some()
    }

    /// One-line description of the effective policy. It holds the configured
    /// addresses and nothing else.
    pub fn describe(&self) -> String {
        let mdns = if self.mdns { "on" } else { "off" };
        match &self.allow {
            None => format!("mdns={mdns} outbound_allow=unrestricted"),
            Some(list) => {
                let entries: Vec<String> = list.entries().iter().map(ToString::to_string).collect();
                format!(
                    "mdns={mdns} outbound_allow=[{}] (application-level routing control, \
                     not an OS security boundary)",
                    entries.join(", ")
                )
            }
        }
    }

    /// Log the effective policy. Called once, when the swarm is built.
    pub(crate) fn log_effective(&self) {
        info!("network policy: {}", self.describe());
    }

    /// Snapshot of the refusal counters.
    pub fn refusals(&self) -> PolicyRefusals {
        let get = |s: PolicySource| self.refusals[s.index()].load(Ordering::Relaxed);
        PolicyRefusals {
            bootstrap: get(PolicySource::Bootstrap),
            mdns: get(PolicySource::Mdns),
            identify: get(PolicySource::Identify),
            kad: get(PolicySource::Kad),
            inbound: get(PolicySource::Inbound),
            dial: get(PolicySource::Dial),
        }
    }

    fn record(&self, source: PolicySource, stage: RefusalStage, addr: &Multiaddr) {
        self.refusals[source.index()].fetch_add(1, Ordering::Relaxed);
        let first = {
            let mut warned = lock(&self.warned);
            warned.len() < WARN_DEDUP_CAP && warned.insert((source, addr.clone()))
        };
        if first {
            warn!(
                source = source.label(),
                stage = stage.label(),
                %addr,
                "network policy: address not in {OUTBOUND_ALLOW_FLAG}, refused"
            );
        } else {
            debug!(
                source = source.label(),
                stage = stage.label(),
                %addr,
                "network policy: address not in {OUTBOUND_ALLOW_FLAG}, refused (repeat)"
            );
        }
    }

    fn judge(
        &self,
        source: PolicySource,
        stage: RefusalStage,
        addr: &Multiaddr,
        peer: Option<PeerId>,
    ) -> bool {
        match &self.allow {
            None => true,
            Some(list) if list.allows_for(addr, peer) => true,
            Some(_) => {
                self.record(source, stage, addr);
                false
            }
        }
    }

    /// Admit an address learned from `source` for `peer` into a peer store
    /// or event. A refused one is counted and logged; the caller drops it.
    pub fn admit(&self, source: PolicySource, addr: &Multiaddr, peer: Option<PeerId>) -> bool {
        self.judge(source, RefusalStage::Admission, addr, peer)
    }

    /// Admit an address `source` offers as a dial target.
    pub(crate) fn admit_dial(
        &self,
        source: PolicySource,
        addr: &Multiaddr,
        peer: Option<PeerId>,
    ) -> bool {
        self.judge(source, RefusalStage::Dial, addr, peer)
    }

    /// Like [`Self::admit`] but uncounted: for echoes of addresses already
    /// judged where they were first learned.
    pub(crate) fn allows_quietly(&self, addr: &Multiaddr, peer: Option<PeerId>) -> bool {
        self.allow
            .as_ref()
            .is_none_or(|list| list.allows_for(addr, peer))
    }

    /// The transport gate's check, counted under [`PolicySource::Dial`].
    pub(crate) fn check_transport_dial(&self, addr: &Multiaddr) -> bool {
        self.judge(PolicySource::Dial, RefusalStage::Dial, addr, None)
    }

    /// Judge a connection a remote peer opened. A relayed connection is
    /// judged by the relay part of `local_addr` (the circuit it arrived on), a
    /// direct one by `remote_addr`. `peer` is `None` before the handshake,
    /// when a peer-pinned entry is given the benefit of the doubt; it is
    /// judged again once the peer is known.
    pub(crate) fn inbound_allowed(
        &self,
        local_addr: &Multiaddr,
        remote_addr: &Multiaddr,
        peer: Option<PeerId>,
    ) -> bool {
        let Some(list) = &self.allow else {
            return true;
        };
        let relayed = local_addr.iter().any(|p| matches!(p, Protocol::P2pCircuit));
        let (judged, ok) = if relayed {
            (local_addr, list.decide(local_addr, None, PeerPin::Enforce))
        } else {
            let pin = if peer.is_some() {
                PeerPin::Enforce
            } else {
                PeerPin::Defer
            };
            (remote_addr, list.decide(remote_addr, peer, pin))
        };
        if !ok {
            self.record(PolicySource::Inbound, RefusalStage::Inbound, judged);
        }
        ok
    }
}

/// Run the test `name` (its full path in this test binary) alone, in a child
/// process, and assert that it passed. Returns `true` in that child, where
/// the caller runs its body, and `false` in the parent, which should return.
///
/// For tests that capture logs: `tracing` keeps process-wide state (callsite
/// interest, the max-level hint) that other tests' threads update while they
/// run, which can drop events from a scoped subscriber. Alone in a process,
/// nothing else touches it.
#[cfg(test)]
pub(crate) fn isolated(name: &str) -> bool {
    const MARK: &str = "OMNI_NET_ISOLATED_TEST";
    if std::env::var(MARK).as_deref() == Ok(name) {
        return true;
    }
    let out = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", name, "--nocapture", "--test-threads=1"])
        .env(MARK, name)
        .output()
        .expect("spawn the isolated test");
    assert!(
        out.status.success(),
        "isolated test {name} failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("1 passed"),
        "isolated test {name} did not run:\n{stdout}"
    );
    false
}

/// Run `f` with a `tracing` subscriber that writes, without colour, into a
/// buffer; returns `f`'s result and everything logged at `debug` or above.
/// Use it inside [`isolated`].
#[cfg(test)]
pub(crate) fn capture_logs<T>(f: impl FnOnce() -> T) -> (T, String) {
    use std::sync::Arc;

    #[derive(Clone, Default)]
    struct Buffer(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Buffer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            lock(&self.0).extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Buffer {
        type Writer = Buffer;
        fn make_writer(&'a self) -> Buffer {
            self.clone()
        }
    }

    // `tracing` keeps global state (callsite interest, the max-level hint)
    // that every registered subscriber feeds. Two tests registering and
    // dropping scoped subscribers at once can leave a callsite disabled for
    // the other one, so log captures run one at a time.
    static CAPTURE: Mutex<()> = Mutex::new(());
    let _one_at_a_time = lock(&CAPTURE);

    let buffer = Buffer::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(buffer.clone())
        .with_ansi(false)
        .with_max_level(tracing::Level::DEBUG)
        .finish();
    let out = tracing::subscriber::with_default(subscriber, || {
        // Another test thread may have cached a callsite's interest while
        // this subscriber was being registered; recompute it now that the
        // subscriber is in place, so no event of `f` is filtered out.
        tracing::callsite::rebuild_interest_cache();
        f()
    });
    let logs = String::from_utf8_lossy(&lock(&buffer.0)).into_owned();
    (out, logs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(items: &[&str]) -> OutboundAllowList {
        OutboundAllowList::parse(items).unwrap().expect("non-empty")
    }

    fn ma(s: &str) -> Multiaddr {
        s.parse().unwrap()
    }

    fn peer() -> PeerId {
        PeerId::random()
    }

    #[test]
    fn an_empty_or_blank_list_is_no_restriction() {
        assert_eq!(
            OutboundAllowList::parse(Vec::<String>::new()).unwrap(),
            None
        );
        assert_eq!(OutboundAllowList::parse(["", "  "]).unwrap(), None);
    }

    #[test]
    fn networks_match_any_port_transport_and_peer_inside_them() {
        let l = list(&["127.0.0.1", "10.0.0.0/8", "::1"]);
        let p = peer();
        for ok in [
            "/ip4/127.0.0.1/udp/4001/quic-v1".to_string(),
            format!("/ip4/127.0.0.1/udp/9/quic-v1/p2p/{p}"),
            "/ip4/10.20.30.40/tcp/1".to_string(),
            "/ip6/::1/udp/4001/quic-v1".to_string(),
            // IPv4-mapped IPv6 matches the IPv4 network.
            "/ip6/::ffff:10.1.2.3/udp/4001/quic-v1".to_string(),
        ] {
            assert!(l.allows(&ma(&ok)), "{ok} must be allowed");
        }
        for refused in [
            "/ip4/127.0.0.2/udp/4001/quic-v1",
            "/ip4/192.168.1.10/udp/4001/quic-v1",
            "/ip6/::2/udp/4001/quic-v1",
        ] {
            assert!(!l.allows(&ma(refused)), "{refused} must be refused");
        }
    }

    #[test]
    fn prefixes_match_their_own_components_only() {
        let l = list(&[
            "/ip4/192.0.2.7/udp/4001/quic-v1",
            "/dns4/Gateway.Example/udp/7",
        ]);
        assert!(l.allows(&ma("/ip4/192.0.2.7/udp/4001/quic-v1")));
        assert!(l.allows(&ma(&format!(
            "/ip4/192.0.2.7/udp/4001/quic-v1/p2p/{}",
            peer()
        ))));
        assert!(
            !l.allows(&ma("/ip4/192.0.2.7/udp/4002/quic-v1")),
            "another port"
        );
        assert!(
            !l.allows(&ma("/ip4/192.0.2.8/udp/4001/quic-v1")),
            "another host"
        );
        assert!(
            l.allows(&ma("/dns4/gateway.example./udp/7/quic-v1")),
            "DNS names compare case-insensitively"
        );
    }

    #[test]
    fn a_peer_pinned_entry_requires_the_dial_to_name_that_peer() {
        let (a, b) = (peer(), peer());
        let l = list(&[&format!("/ip4/127.0.0.1/udp/4001/quic-v1/p2p/{a}")]);
        assert!(l.allows(&ma(&format!("/ip4/127.0.0.1/udp/4001/quic-v1/p2p/{a}"))));
        assert!(!l.allows(&ma(&format!("/ip4/127.0.0.1/udp/4001/quic-v1/p2p/{b}"))));
        assert!(
            !l.allows(&ma("/ip4/127.0.0.1/udp/4001/quic-v1")),
            "no peer named"
        );
    }

    #[test]
    fn a_relayed_address_is_judged_by_its_relay_part() {
        let (relay, target) = (peer(), peer());
        let l = list(&["198.51.100.1"]);
        assert!(l.allows(&ma(&format!(
            "/ip4/198.51.100.1/udp/4001/quic-v1/p2p/{relay}/p2p-circuit/p2p/{target}"
        ))));
        assert!(!l.allows(&ma(&format!(
            "/ip4/203.0.113.9/udp/4001/quic-v1/p2p/{relay}/p2p-circuit/p2p/{target}"
        ))));
        assert!(
            !l.allows(&ma(&format!("/p2p/{relay}/p2p-circuit/p2p/{target}"))),
            "a relay part without a host matches nothing"
        );
    }

    #[test]
    fn addresses_without_a_host_are_never_allowed() {
        let l = list(&["0.0.0.0/0", "::/0"]);
        assert!(l.allows(&ma("/ip4/203.0.113.1/udp/1/quic-v1")));
        assert!(!l.allows(&ma("/memory/1")));
        assert!(!l.allows(&ma(&format!("/p2p/{}", peer()))));
    }

    #[test]
    fn malformed_entries_are_refused_with_the_flag_named() {
        for bad in [
            "not-an-ip",
            "10.0.0.0/33",
            "::/129",
            "/ip4/1.2.3.4/udp/0/quic-v1",
            "/memory/1",
            "/ip4/1.2.3.4/udp/1/quic-v1/webtransport",
            &format!("/p2p/{}", peer()),
            &format!("/ip4/1.2.3.4/udp/1/quic-v1/p2p/{}/p2p-circuit", peer()),
        ] {
            let err = OutboundAllowList::parse([bad]).unwrap_err();
            assert!(
                err.to_string().contains(OUTBOUND_ALLOW_FLAG),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn display_round_trips_through_parse() {
        let p = peer();
        let l = list(&[
            "10.1.2.3/8",
            &format!("/ip4/127.0.0.1/udp/4001/quic-v1/p2p/{p}"),
        ]);
        assert_eq!(
            l.to_string(),
            format!("10.0.0.0/8,/ip4/127.0.0.1/udp/4001/quic-v1/p2p/{p}")
        );
        let again = list(&l.to_string().split(',').collect::<Vec<_>>());
        assert_eq!(again, l);
    }

    // ── The runtime policy (items 2 and 3) ──────────────────────────────

    fn restricted(items: &[&str]) -> NetPolicy {
        NetPolicy::new(false, Some(list(items)))
    }

    #[test]
    fn the_effective_policy_is_described_with_the_boundary_caveat() {
        assert_eq!(
            NetPolicy::unrestricted().describe(),
            "mdns=on outbound_allow=unrestricted"
        );
        assert_eq!(
            restricted(&["127.0.0.1", "/ip4/192.0.2.7/udp/4001/quic-v1"]).describe(),
            "mdns=off outbound_allow=[127.0.0.1/32, /ip4/192.0.2.7/udp/4001/quic-v1] \
             (application-level routing control, not an OS security boundary)"
        );
    }

    /// Every source counts its own refusals; the first refusal of an address
    /// by a source is logged at `warn`, repeats at `debug`.
    #[test]
    fn refusals_are_counted_and_logged_by_source() {
        if !isolated("policy::tests::refusals_are_counted_and_logged_by_source") {
            return;
        }
        let policy = restricted(&["127.0.0.1"]);
        let refused = ma("/ip4/192.0.2.1/udp/4001/quic-v1");
        let allowed = ma("/ip4/127.0.0.1/udp/4001/quic-v1");
        let ((), logs) = capture_logs(|| {
            for (i, source) in PolicySource::ALL.iter().enumerate() {
                for _ in 0..=i {
                    assert!(!policy.admit(*source, &refused, None));
                }
                assert!(
                    policy.admit(*source, &allowed, None),
                    "allowed is never counted"
                );
            }
        });
        let r = policy.refusals();
        assert_eq!(
            (r.bootstrap, r.mdns, r.identify, r.kad, r.inbound, r.dial),
            (1, 2, 3, 4, 5, 6)
        );
        assert_eq!(r.total(), 21);
        for source in PolicySource::ALL {
            let label = format!("source=\"{}\"", source.label());
            let warned = logs
                .lines()
                .filter(|l| l.contains(" WARN ") && l.contains(&label))
                .count();
            assert_eq!(warned, 1, "{label} warns once:\n{logs}");
        }
        assert!(logs.contains("network policy: address not in --outbound-allow, refused"));
        assert!(
            logs.contains("(repeat)"),
            "repeats are logged at debug:\n{logs}"
        );
    }

    #[test]
    fn an_unrestricted_policy_refuses_and_counts_nothing() {
        let policy = NetPolicy::unrestricted();
        let addr = ma("/ip4/192.0.2.1/udp/4001/quic-v1");
        for source in PolicySource::ALL {
            assert!(policy.admit(source, &addr, None));
        }
        assert!(policy.inbound_allowed(&ma("/ip4/0.0.0.0/udp/1/quic-v1"), &addr, None));
        assert_eq!(policy.refusals(), PolicyRefusals::default());
    }

    /// Inbound: a direct connection is judged by the remote address, a
    /// relayed one by the relay part of the local address; a peer pin is
    /// deferred until the peer is known.
    #[test]
    fn inbound_connections_are_judged_by_their_route() {
        let (pinned, other, relay) = (peer(), peer(), peer());
        let policy = restricted(&[
            "10.0.0.0/8",
            &format!("/ip4/127.0.0.1/p2p/{pinned}"),
            "198.51.100.1",
        ]);
        let local = ma("/ip4/0.0.0.0/udp/4001/quic-v1");
        assert!(policy.inbound_allowed(&local, &ma("/ip4/10.1.2.3/udp/9/quic-v1"), None));
        assert!(!policy.inbound_allowed(&local, &ma("/ip4/192.0.2.1/udp/9/quic-v1"), None));
        let loopback = ma("/ip4/127.0.0.1/udp/9/quic-v1");
        assert!(policy.inbound_allowed(&local, &loopback, None), "deferred");
        assert!(policy.inbound_allowed(&local, &loopback, Some(pinned)));
        assert!(!policy.inbound_allowed(&local, &loopback, Some(other)));
        let circuit = |r: &str| {
            ma(&format!(
                "/ip4/{r}/udp/4001/quic-v1/p2p/{relay}/p2p-circuit"
            ))
        };
        let source = ma(&format!("/p2p/{other}"));
        assert!(policy.inbound_allowed(&circuit("198.51.100.1"), &source, Some(other)));
        assert!(!policy.inbound_allowed(&circuit("203.0.113.1"), &source, Some(other)));
        assert_eq!(policy.refusals().inbound, 3);
    }
}
