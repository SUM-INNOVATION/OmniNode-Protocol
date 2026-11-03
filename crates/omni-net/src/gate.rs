//! Enforcement of the outbound allow-list (issue #158). See [`crate::policy`]
//! for the rules and for why each piece sits where it does.
//!
//! * [`OutboundGate`]: the outermost layer of OmniNet's transport stack. Every
//!   `Transport::dial` passes through it after the swarm has merged the
//!   addresses every behaviour contributed, and before any packet is sent.
//!   This is the decisive check.
//! * [`PolicyBehaviour`]: the first field of
//!   [`crate::behaviour::OmniNodeBehaviour`]. Judges inbound connections.
//! * [`Gated`]: wraps a behaviour that stores peer addresses (mDNS, Identify,
//!   Kademlia). Filters the addresses it contributes to a dial and withholds
//!   refused peer addresses it publishes to other behaviours.
//!
//! With no allow-list every one of these is a pass-through.

use std::convert::Infallible;
use std::io;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use libp2p::core::muxing::StreamMuxerBox;
use libp2p::core::transport::{
    Boxed, DialOpts, ListenerId, PortUse, TransportError, TransportEvent,
};
use libp2p::core::{Endpoint, Transport};
use libp2p::swarm::{
    ConnectionDenied, ConnectionId, FromSwarm, NetworkBehaviour, THandler, THandlerInEvent,
    THandlerOutEvent, ToSwarm, dummy,
};
use libp2p::{Multiaddr, PeerId};
use tracing::debug;

use crate::policy::{NetPolicy, OUTBOUND_ALLOW_FLAG, PolicySource};

// ── Transport gate ───────────────────────────────────────────────────────────

/// The transport every OmniNet connection goes through, gated by the
/// outbound allow-list.
///
/// A refused dial fails with `io::ErrorKind::PermissionDenied` before a
/// socket is touched, and is counted under [`PolicySource::Dial`]; the swarm
/// reports it per address in `SwarmEvent::OutgoingConnectionError` and moves
/// on to the next address of the dial. Listening is not affected.
pub struct OutboundGate {
    inner: Boxed<(PeerId, StreamMuxerBox)>,
    policy: Arc<NetPolicy>,
}

impl OutboundGate {
    /// Gate `inner` with `policy`; an unrestricted policy gates nothing.
    pub fn new(inner: Boxed<(PeerId, StreamMuxerBox)>, policy: Arc<NetPolicy>) -> Self {
        Self { inner, policy }
    }
}

impl Transport for OutboundGate {
    type Output = (PeerId, StreamMuxerBox);
    type Error = io::Error;
    type ListenerUpgrade = <Boxed<(PeerId, StreamMuxerBox)> as Transport>::ListenerUpgrade;
    type Dial = <Boxed<(PeerId, StreamMuxerBox)> as Transport>::Dial;

    fn listen_on(
        &mut self,
        id: ListenerId,
        addr: Multiaddr,
    ) -> Result<(), TransportError<Self::Error>> {
        self.inner.listen_on(id, addr)
    }

    fn remove_listener(&mut self, id: ListenerId) -> bool {
        self.inner.remove_listener(id)
    }

    fn dial(
        &mut self,
        addr: Multiaddr,
        opts: DialOpts,
    ) -> Result<Self::Dial, TransportError<Self::Error>> {
        if !self.policy.check_transport_dial(&addr) {
            debug!(%addr, role = ?opts.role, "dial refused by the outbound allow-list");
            return Err(TransportError::Other(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("refused by the {OUTBOUND_ALLOW_FLAG} policy: {addr}"),
            )));
        }
        self.inner.dial(addr, opts)
    }

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<TransportEvent<Self::ListenerUpgrade, Self::Error>> {
        Pin::new(&mut self.inner).poll(cx)
    }
}

// ── Inbound connections ──────────────────────────────────────────────────────

/// The reason carried by a connection the policy refused.
#[derive(Debug)]
pub struct PolicyDenied {
    addr: Multiaddr,
}

impl std::fmt::Display for PolicyDenied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "refused by the {OUTBOUND_ALLOW_FLAG} policy: {}",
            self.addr
        )
    }
}

impl std::error::Error for PolicyDenied {}

fn denied(addr: &Multiaddr) -> ConnectionDenied {
    ConnectionDenied::new(PolicyDenied { addr: addr.clone() })
}

/// Judges inbound connections. Carries no protocol; its handler is `dummy`.
///
/// It must be the **first** field of the composed behaviour: the derived
/// behaviour asks its fields in declaration order, so a connection refused
/// here is refused before any other behaviour builds a handler or records
/// state for it. The swarm builds it only when an allow-list is set.
pub struct PolicyBehaviour {
    policy: Arc<NetPolicy>,
}

impl PolicyBehaviour {
    pub fn new(policy: Arc<NetPolicy>) -> Self {
        Self { policy }
    }
}

impl NetworkBehaviour for PolicyBehaviour {
    type ConnectionHandler = dummy::ConnectionHandler;
    type ToSwarm = Infallible;

    fn handle_pending_inbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        local_addr: &Multiaddr,
        remote_addr: &Multiaddr,
    ) -> Result<(), ConnectionDenied> {
        // Before the handshake: refuse what no entry could allow, whoever the
        // peer turns out to be.
        if self.policy.inbound_allowed(local_addr, remote_addr, None) {
            Ok(())
        } else {
            Err(denied(remote_addr))
        }
    }

    fn handle_established_inbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        peer: PeerId,
        local_addr: &Multiaddr,
        remote_addr: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        // After the handshake the peer is known, so peer-pinned entries are
        // enforced.
        if self
            .policy
            .inbound_allowed(local_addr, remote_addr, Some(peer))
        {
            Ok(dummy::ConnectionHandler)
        } else {
            Err(denied(remote_addr))
        }
    }

    fn handle_established_outbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        _peer: PeerId,
        _addr: &Multiaddr,
        _role_override: Endpoint,
        _port_use: PortUse,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        // Outbound connections were judged at the transport.
        Ok(dummy::ConnectionHandler)
    }

    fn on_swarm_event(&mut self, _event: FromSwarm) {}

    fn on_connection_handler_event(
        &mut self,
        _peer_id: PeerId,
        _connection_id: ConnectionId,
        event: THandlerOutEvent<Self>,
    ) {
        match event {}
    }

    fn poll(
        &mut self,
        _cx: &mut Context<'_>,
    ) -> Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
        Poll::Pending
    }
}

// ── Address-source wrapper ───────────────────────────────────────────────────

/// A behaviour whose peer addresses are filtered by the policy and
/// attributed to `source`.
///
/// It dereferences to the inner behaviour, so call sites such as
/// `behaviour_mut().kademlia.add_address(..)` are unchanged, and its
/// `ToSwarm` is the inner one, so the derived event enum keeps its variants.
pub struct Gated<B> {
    inner: B,
    source: PolicySource,
    policy: Arc<NetPolicy>,
}

impl<B> Gated<B> {
    pub fn new(inner: B, source: PolicySource, policy: Arc<NetPolicy>) -> Self {
        Self {
            inner,
            source,
            policy,
        }
    }

    /// The source this wrapper reports refusals under.
    pub fn source(&self) -> PolicySource {
        self.source
    }
}

impl<B> Deref for Gated<B> {
    type Target = B;
    fn deref(&self) -> &B {
        &self.inner
    }
}

impl<B> DerefMut for Gated<B> {
    fn deref_mut(&mut self) -> &mut B {
        &mut self.inner
    }
}

impl<B: NetworkBehaviour> NetworkBehaviour for Gated<B> {
    type ConnectionHandler = B::ConnectionHandler;
    type ToSwarm = B::ToSwarm;

    fn handle_pending_inbound_connection(
        &mut self,
        connection_id: ConnectionId,
        local_addr: &Multiaddr,
        remote_addr: &Multiaddr,
    ) -> Result<(), ConnectionDenied> {
        self.inner
            .handle_pending_inbound_connection(connection_id, local_addr, remote_addr)
    }

    fn handle_established_inbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        local_addr: &Multiaddr,
        remote_addr: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        self.inner.handle_established_inbound_connection(
            connection_id,
            peer,
            local_addr,
            remote_addr,
        )
    }

    fn handle_pending_outbound_connection(
        &mut self,
        connection_id: ConnectionId,
        maybe_peer: Option<PeerId>,
        addresses: &[Multiaddr],
        effective_role: Endpoint,
    ) -> Result<Vec<Multiaddr>, ConnectionDenied> {
        let mut offered = self.inner.handle_pending_outbound_connection(
            connection_id,
            maybe_peer,
            addresses,
            effective_role,
        )?;
        if self.policy.is_restricted() {
            offered.retain(|a| self.policy.admit_dial(self.source, a, maybe_peer));
        }
        Ok(offered)
    }

    fn handle_established_outbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        addr: &Multiaddr,
        role_override: Endpoint,
        port_use: PortUse,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        self.inner.handle_established_outbound_connection(
            connection_id,
            peer,
            addr,
            role_override,
            port_use,
        )
    }

    fn on_swarm_event(&mut self, event: FromSwarm) {
        self.inner.on_swarm_event(event)
    }

    fn on_connection_handler_event(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        event: THandlerOutEvent<Self>,
    ) {
        self.inner
            .on_connection_handler_event(peer_id, connection_id, event)
    }

    fn poll(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
        loop {
            match self.inner.poll(cx) {
                // A peer address this behaviour learned is published to every
                // other behaviour (request-response keeps a cache of them and
                // dials from it). A refused one is withheld, so it cannot come
                // back through a behaviour that is not wrapped. Not counted:
                // it was already judged where it was learned.
                Poll::Ready(ToSwarm::NewExternalAddrOfPeer { peer_id, address })
                    if !self.policy.allows_quietly(&address, Some(peer_id)) =>
                {
                    debug!(
                        source = self.source.label(),
                        %peer_id,
                        %address,
                        "network policy: peer address withheld from other behaviours"
                    );
                    continue;
                }
                other => return other,
            }
        }
    }
}
