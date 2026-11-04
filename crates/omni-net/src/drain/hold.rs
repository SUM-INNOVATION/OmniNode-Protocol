//! Test-only seam: hold one message at the transport until the swarm is driven
//! after a stop request.
//!
//! Whether a message published just before shutdown is lost without the drain
//! depends on a race: the connection task that writes it out may or may not
//! run before the swarm is dropped. A test that relies on losing that race is
//! probabilistic. This seam fixes the race's outcome to the worst case instead.
//!
//! A [`WriteHold`] wraps every substream of every connection of one node (QUIC
//! and relayed alike, inbound and outbound). A write whose bytes contain the
//! hold's marker — the payload of the message under test — returns `Pending`
//! and leaves those bytes queued in the connection, unsent. Every other write
//! passes through untouched. The hold is released at the first `poll` of the
//! node's transport after the swarm loop has accepted a stop request. The swarm
//! polls its transport from inside `Swarm::poll`, which only the swarm task
//! calls, so the release means exactly "the swarm was still being driven after
//! the stop was requested":
//!
//! * with the drain, the loop keeps polling the swarm, the first poll releases
//!   the hold, and the connection writes the message out while the drain holds
//!   the swarm up;
//! * without it, the loop drops the swarm in the same poll that accepted the
//!   stop, so the transport is never polled again and the hold is never
//!   released. A connection task that runs afterwards sees its command channel
//!   closed before it polls the connection, and drops the connection with the
//!   message still in it.
//!
//! The message therefore leaves the node if and only if the swarm is driven
//! after the stop request — the property the drain exists to provide — and the
//! outcome does not depend on how the runtime schedules the connection task.
//!
//! Compiled only into this crate's unit tests. [`crate::swarm::OmniSwarm`]
//! takes a hold only through [`WriteHold::attach_to_next_build`], and calls it
//! only from `#[cfg(test)]` lines, so production builds contain none of this.

use std::cell::RefCell;
use std::io;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use futures::{AsyncRead, AsyncWrite};
use libp2p::core::muxing::{StreamMuxer, StreamMuxerBox, StreamMuxerEvent, SubstreamBox};
use libp2p::core::transport::{Boxed, DialOpts, ListenerId, TransportError, TransportEvent};
use libp2p::core::Transport;
use libp2p::{Multiaddr, PeerId};
use tokio::sync::Notify;

thread_local! {
    /// The hold the next `OmniSwarm::build` on this thread takes.
    static NEXT_BUILD: RefCell<Option<WriteHold>> = const { RefCell::new(None) };
}

/// Called by `OmniSwarm::build`: the hold attached on this thread, if any.
/// Taking it detaches it, so it applies to exactly one node.
pub(crate) fn take_for_build() -> Option<WriteHold> {
    let hold = NEXT_BUILD.with(|next| next.borrow_mut().take());
    if let Some(hold) = &hold {
        hold.state().attached = true;
    }
    hold
}

/// What a hold saw, for the test's assertions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HoldReport {
    /// A write carrying the marker was being held when the swarm loop accepted
    /// the stop request: the message was queued in the connection, unsent.
    pub held_at_stop: bool,
    /// The swarm was polled after the stop request, which released the hold.
    pub released: bool,
    /// Bytes carrying the marker were handed to the connection's stream.
    pub written: bool,
}

/// Holds writes carrying one marker. Cheap to clone; clones share state.
#[derive(Clone)]
pub(crate) struct WriteHold {
    shared: Arc<Shared>,
}

struct Shared {
    marker: Vec<u8>,
    state: Mutex<State>,
    /// Signalled the first time a write carrying the marker is held.
    held: Notify,
}

#[derive(Default)]
struct State {
    attached: bool,
    held: bool,
    stop_requested: bool,
    held_at_stop: bool,
    released: bool,
    written: bool,
    /// Tasks whose held write is waiting for the release.
    wakers: Vec<Waker>,
}

impl WriteHold {
    /// Hold every write whose bytes contain `marker`.
    pub(crate) fn new(marker: &[u8]) -> Self {
        assert!(!marker.is_empty(), "an empty marker would hold every write");
        Self {
            shared: Arc::new(Shared {
                marker: marker.to_vec(),
                state: Mutex::new(State::default()),
                held: Notify::new(),
            }),
        }
    }

    /// Give this hold to the next `OmniSwarm::build` on the calling thread.
    /// `OmniNet::new` builds its swarm before its first await, so calling this
    /// just before `OmniNet::new(..).await` attaches it to that node; check
    /// [`WriteHold::is_attached`] afterwards.
    pub(crate) fn attach_to_next_build(&self) {
        NEXT_BUILD.with(|next| *next.borrow_mut() = Some(self.clone()));
    }

    /// Whether a swarm took this hold.
    pub(crate) fn is_attached(&self) -> bool {
        self.state().attached
    }

    /// Wrap `transport`: every connection's substreams consult this hold, and
    /// every poll of the transport by the swarm is reported to it.
    pub(crate) fn wrap(
        &self,
        transport: Boxed<(PeerId, StreamMuxerBox)>,
    ) -> Boxed<(PeerId, StreamMuxerBox)> {
        let hold = self.clone();
        let held = transport
            .map(move |(peer, muxer), _| {
                (
                    peer,
                    StreamMuxerBox::new(HeldMuxer {
                        inner: muxer,
                        hold: hold.clone(),
                    }),
                )
            })
            .boxed();
        ObservedTransport {
            inner: held,
            hold: self.clone(),
        }
        .boxed()
    }

    /// Called by the swarm loop once it has accepted a stop request, before
    /// the drain.
    pub(crate) fn stop_requested(&self) {
        let mut state = self.state();
        state.stop_requested = true;
        state.held_at_stop = state.held;
    }

    /// Wait until a write carrying the marker is being held.
    pub(crate) async fn wait_held(&self) {
        if self.state().held {
            return;
        }
        // `notify_one` stores a permit when nobody is waiting yet, so a hold
        // that lands between the check above and this await is not missed.
        self.shared.held.notified().await;
    }

    pub(crate) fn report(&self) -> HoldReport {
        let state = self.state();
        HoldReport {
            held_at_stop: state.held_at_stop,
            released: state.released,
            written: state.written,
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn carries_marker(&self, buf: &[u8]) -> bool {
        buf.windows(self.shared.marker.len())
            .any(|window| window == self.shared.marker.as_slice())
    }

    /// Whether a write of `buf` must wait. If so, `waker` is woken on release.
    fn holds(&self, buf: &[u8], waker: &Waker) -> bool {
        if !self.carries_marker(buf) {
            return false;
        }
        let mut state = self.state();
        if state.released {
            return false;
        }
        if !state.wakers.iter().any(|w| w.will_wake(waker)) {
            state.wakers.push(waker.clone());
        }
        if !state.held {
            state.held = true;
            self.shared.held.notify_one();
        }
        true
    }

    fn note_written(&self, buf: &[u8]) {
        if self.carries_marker(buf) {
            self.state().written = true;
        }
    }

    /// The swarm polled its transport. After a stop request, that releases
    /// the hold.
    fn swarm_polled(&self) {
        let wakers = {
            let mut state = self.state();
            if !state.stop_requested || state.released {
                return;
            }
            state.released = true;
            std::mem::take(&mut state.wakers)
        };
        for waker in wakers {
            waker.wake();
        }
    }
}

// ── Transport: report the swarm's polls ──────────────────────────────────────

/// Outermost transport layer. `Swarm::poll` is its only caller of `poll`.
struct ObservedTransport {
    inner: Boxed<(PeerId, StreamMuxerBox)>,
    hold: WriteHold,
}

impl Transport for ObservedTransport {
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
        self.inner.dial(addr, opts)
    }

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<TransportEvent<Self::ListenerUpgrade, Self::Error>> {
        self.hold.swarm_polled();
        Pin::new(&mut self.inner).poll(cx)
    }
}

// ── Muxer and substreams: hold the marked write ──────────────────────────────

struct HeldMuxer {
    inner: StreamMuxerBox,
    hold: WriteHold,
}

impl StreamMuxer for HeldMuxer {
    type Substream = HeldSubstream;
    type Error = io::Error;

    fn poll_inbound(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Self::Substream, Self::Error>> {
        let hold = self.hold.clone();
        Pin::new(&mut self.inner)
            .poll_inbound(cx)
            .map_ok(|inner| HeldSubstream { inner, hold })
    }

    fn poll_outbound(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Self::Substream, Self::Error>> {
        let hold = self.hold.clone();
        Pin::new(&mut self.inner)
            .poll_outbound(cx)
            .map_ok(|inner| HeldSubstream { inner, hold })
    }

    fn poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Pin::new(&mut self.inner).poll_close(cx)
    }

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<StreamMuxerEvent, Self::Error>> {
        Pin::new(&mut self.inner).poll(cx)
    }
}

/// A substream whose writes consult the hold. Vectored writes use the default
/// `poll_write_vectored`, which goes through `poll_write`.
struct HeldSubstream {
    inner: SubstreamBox,
    hold: WriteHold,
}

impl AsyncRead for HeldSubstream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for HeldSubstream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        if self.hold.holds(buf, cx.waker()) {
            return Poll::Pending;
        }
        let written = Pin::new(&mut self.inner).poll_write(cx, buf);
        if let Poll::Ready(Ok(n)) = written {
            self.hold.note_written(&buf[..n]);
        }
        written
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_close(cx)
    }
}
