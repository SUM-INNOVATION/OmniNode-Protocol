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
//! before. With one, [`crate::gate::OutboundGate`] wraps the whole transport
//! stack (relay client and QUIC) and judges every address handed to
//! `Transport::dial`, after the swarm has merged the addresses every behaviour
//! contributed. So no dial reaches a socket unless it passes, whichever part
//! of the node asked for it: an explicit `--peer` dial, mDNS, Identify,
//! Kademlia, request-response, gossipsub, AutoNAT, DCUtR hole-punching or a
//! relay-client dial to a relay. A refused dial fails with `PermissionDenied`
//! before any packet is sent.
//!
//! Not covered here (issue #158, items 2 and 3): refused addresses are not yet
//! kept out of the peer stores (Kademlia, gossipsub explicit peers), inbound
//! connections are not judged, and refusals are not counted.
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

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::str::FromStr;

use libp2p::multiaddr::Protocol;
use libp2p::{Multiaddr, PeerId};

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

impl AllowEntry {
    /// Does this entry allow `transport` (an address without its `/p2p`
    /// suffix) for `peer`, the peer the address names, if any?
    fn matches(&self, transport: &[Protocol<'_>], peer: Option<PeerId>) -> bool {
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
                let peer_ok = match wanted {
                    None => true,
                    Some(w) => peer == Some(*w),
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
        let comps: Vec<Protocol<'_>> = addr.iter().collect();
        let segment = match comps.iter().position(|p| matches!(p, Protocol::P2pCircuit)) {
            Some(i) => &comps[..i],
            None => &comps[..],
        };
        let (transport, peer) = match segment.split_last() {
            Some((Protocol::P2p(p), rest)) => (rest, Some(*p)),
            _ => (segment, None),
        };
        let Some(host) = transport.first() else {
            return false;
        };
        if !is_host(host) || transport.iter().any(|p| matches!(p, Protocol::P2p(_))) {
            return false;
        }
        self.entries.iter().any(|e| e.matches(transport, peer))
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
}
