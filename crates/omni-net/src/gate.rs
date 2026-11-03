//! Enforcement of the outbound allow-list (issue #158). See [`crate::policy`]
//! for the rules.
//!
//! [`OutboundGate`] is the outermost layer of OmniNet's transport stack, so
//! every `Transport::dial` passes through it after the swarm has merged the
//! addresses every behaviour contributed, and before any packet is sent.
//! With no allow-list it passes every call straight through.

use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use libp2p::core::muxing::StreamMuxerBox;
use libp2p::core::transport::{Boxed, DialOpts, ListenerId, TransportError, TransportEvent};
use libp2p::core::Transport;
use libp2p::{Multiaddr, PeerId};
use tracing::debug;

use crate::policy::{OutboundAllowList, OUTBOUND_ALLOW_FLAG};

/// The transport every OmniNet connection goes through, gated by the
/// outbound allow-list.
///
/// A refused dial fails with `io::ErrorKind::PermissionDenied` before a
/// socket is touched; the swarm reports it per address in
/// `SwarmEvent::OutgoingConnectionError` and moves on to the next address of
/// the dial. Listening and inbound connections are not affected.
pub struct OutboundGate {
    inner: Boxed<(PeerId, StreamMuxerBox)>,
    allow: Option<Arc<OutboundAllowList>>,
}

impl OutboundGate {
    /// Gate `inner` with `allow`; `None` (no allow-list) gates nothing.
    pub fn new(
        inner: Boxed<(PeerId, StreamMuxerBox)>,
        allow: Option<Arc<OutboundAllowList>>,
    ) -> Self {
        Self { inner, allow }
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
        if let Some(allow) = &self.allow {
            if !allow.allows(&addr) {
                debug!(%addr, role = ?opts.role, "dial refused by the outbound allow-list");
                return Err(TransportError::Other(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!("refused by the {OUTBOUND_ALLOW_FLAG} policy: {addr}"),
                )));
            }
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
