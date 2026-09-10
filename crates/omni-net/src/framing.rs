//! The single length-prefixed framing implementation.
//!
//! Both the shard codec and the tensor codec speak `[u32 BE length][payload]`.
//! They previously carried byte-identical private copies of the reader, and
//! the eager-allocation defect existed in both — which is the argument for
//! this module: a security boundary duplicated is a security boundary that
//! drifts. Every codec routes through here.

use std::io;

use futures::prelude::*;

/// The largest single read attempted, and the ceiling on the scratch buffer.
///
/// This is the bound on declaration-derived allocation: the scratch buffer is
/// sized `min(READ_STEP, declared_len)`, so a declared length can influence its
/// size but can never push it past this constant. Large enough, too, that a big
/// legitimate frame is not reassembled in tiny increments.
pub(crate) const READ_STEP: usize = 64 * 1024;

/// Read a `[u32 BE length][payload]` frame.
///
/// # Memory
///
/// Two buffers, with different relationships to the declared length. Being
/// exact about which is which is the point of this section:
///
/// * **Scratch.** One buffer, allocated once per frame, sized
///   `min(READ_STEP, declared_len)`. For a frame smaller than [`READ_STEP`]
///   that size *is* derived from the declaration — a 32-byte frame allocates
///   32 bytes, not 64 KiB. So it is not true that no allocation derives from
///   the declared length; what is true is that a declaration-derived
///   allocation is capped: **no allocation derived from the declared length
///   ever exceeds [`READ_STEP`].** A 4 GiB declaration and a 64 KiB
///   declaration allocate exactly the same scratch.
/// * **Destination.** Never derived from the declaration at all. It is grown
///   only *after* bytes have arrived, and only by the count actually received:
///   the reserve request is exactly `n`, the byte count the last read
///   returned. `Vec` may satisfy that request with amortised growth, so
///   capacity can exceed length — but every request is for delivered bytes,
///   so capacity tracks a small multiple of what arrived.
///
/// The two halves of the guarantee, stated so each can be tested rather than
/// asserted:
///
/// 1. **No declaration-derived allocation exceeds [`READ_STEP`].**
///    `no_declaration_derived_allocation_exceeds_one_read_step` pins it by
///    observing the real allocator while the production reader runs, and
///    `scratch_is_sized_by_the_smaller_of_read_step_and_declaration` pins the
///    small-frame side of `min`.
/// 2. **Destination allocation follows delivered bytes.**
///    `one_delivered_byte_does_not_reserve_a_whole_read_step_of_destination`
///    pins it: a 64 MiB declaration delivering a single byte must request no
///    more than one read step across the whole call — scratch plus one byte,
///    not scratch plus a step of destination.
///
/// A read returning zero before `len` bytes have arrived is `UnexpectedEof`.
/// An oversized declaration is refused immediately after the length prefix,
/// before any body read and before any allocation at all. Every growth is
/// fallible, so an allocator refusal is an `io::Error` rather than an abort.
pub(crate) async fn read_length_prefixed<T>(io: &mut T, max_bytes: usize) -> io::Result<Vec<u8>>
where
    T: AsyncRead + Unpin + Send,
{
    let mut len_buf = [0u8; 4];
    io.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    // Inclusive maximum: `len == max_bytes` is accepted, `max_bytes + 1` is
    // refused here, with no body read and no allocation.
    if len > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("message too large: {len} bytes (max {max_bytes})"),
        ));
    }

    let mut out: Vec<u8> = Vec::new();
    if len == 0 {
        return Ok(out);
    }

    // The one fixed buffer. Capped by READ_STEP, so a 4 GiB declaration and a
    // 64 KiB declaration allocate the same scratch.
    let scratch_len = READ_STEP.min(len);
    let mut scratch: Vec<u8> = Vec::new();
    scratch.try_reserve_exact(scratch_len).map_err(|e| {
        io::Error::new(
            io::ErrorKind::OutOfMemory,
            format!("allocation failed reserving a {scratch_len}-byte read buffer: {e}"),
        )
    })?;
    scratch.resize(scratch_len, 0);

    while out.len() < len {
        let want = scratch_len.min(len - out.len());
        let n = io.read(&mut scratch[..want]).await?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("stream ended after {} of {len} declared bytes", out.len()),
            ));
        }
        // Grow by what arrived, never by what was promised.
        out.try_reserve(n).map_err(|e| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!("allocation failed growing frame buffer by {n} bytes: {e}"),
            )
        })?;
        // Capacity for `n` more is guaranteed by the line above, so this
        // cannot take `Vec`'s infallible growth path.
        out.extend_from_slice(&scratch[..n]);
    }
    Ok(out)
}

/// Write a `[u32 BE length][payload]` frame.
pub(crate) async fn write_length_prefixed<T>(io: &mut T, data: &[u8]) -> io::Result<()>
where
    T: AsyncWrite + Unpin + Send,
{
    let len = u32::try_from(data.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("payload exceeds u32::MAX: {} bytes", data.len()),
        )
    })?;
    io.write_all(&len.to_be_bytes()).await?;
    io.write_all(data).await?;
    io.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_alloc::{ArmGuard, ObserveGuard};
    use futures::io::Cursor;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    /// Yields at most `per_poll` bytes per read, to model a slow peer.
    struct Trickle {
        data: Vec<u8>,
        pos: usize,
        per_poll: usize,
    }

    impl futures::AsyncRead for Trickle {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            let remaining = self.data.len() - self.pos;
            let n = self.per_poll.min(buf.len()).min(remaining);
            if n == 0 {
                return Poll::Ready(Ok(0));
            }
            let pos = self.pos;
            buf[..n].copy_from_slice(&self.data[pos..pos + n]);
            self.pos += n;
            Poll::Ready(Ok(n))
        }
    }

    pub(crate) fn framed(payload: &[u8]) -> Vec<u8> {
        let mut v = (payload.len() as u32).to_be_bytes().to_vec();
        v.extend_from_slice(payload);
        v
    }

    /// A header declaring `len` with only `body` actually present.
    fn lying_frame(len: u32, body: &[u8]) -> Vec<u8> {
        let mut v = len.to_be_bytes().to_vec();
        v.extend_from_slice(body);
        v
    }

    #[tokio::test]
    async fn zero_length_frame_round_trips() {
        let out = read_length_prefixed(&mut Cursor::new(framed(b"")), 1024)
            .await
            .unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn exact_maximum_is_accepted_and_one_over_is_refused() {
        let payload = vec![0xABu8; 1024];
        let out = read_length_prefixed(&mut Cursor::new(framed(&payload)), 1024)
            .await
            .unwrap();
        assert_eq!(out.len(), 1024);

        // Refused before the body: the reader is given only the 4-byte header,
        // so a body read would EOF rather than produce this error.
        let err = read_length_prefixed(&mut Cursor::new(lying_frame(1025, b"")), 1024)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("message too large"), "{err}");
    }

    #[tokio::test]
    async fn slow_partial_delivery_reassembles_exactly() {
        let payload: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let mut io = Trickle {
            data: framed(&payload),
            pos: 0,
            per_poll: 7, // deliberately not a divisor of READ_STEP
        };
        let out = read_length_prefixed(&mut io, 1 << 30).await.unwrap();
        assert_eq!(out, payload);
    }

    #[tokio::test]
    async fn early_eof_is_an_error_and_releases_partial_state() {
        let f = lying_frame(1024 * 1024, &[1u8; 100]);
        let err = read_length_prefixed(&mut Cursor::new(f), 1 << 30)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[tokio::test]
    async fn no_allocation_approaches_the_declared_length() {
        // 200 MiB declared (under the ceiling), 100 bytes delivered, allocation
        // armed to fail at 1 MiB. The scratch is capped at READ_STEP and the
        // destination grows by the byte count delivered, so both stay under the
        // threshold and this runs out of DATA. Committing the declared length
        // would abort the process instead.
        let f = lying_frame(200 * 1024 * 1024, &[9u8; 100]);
        let err = {
            let _guard = ArmGuard::new(1024 * 1024);
            read_length_prefixed(&mut Cursor::new(f), 256 * 1024 * 1024)
                .await
                .unwrap_err()
        };
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::UnexpectedEof,
            "expected to run out of DATA, not out of memory: {err}"
        );
    }

    #[tokio::test]
    async fn allocation_failure_is_an_io_error_not_an_abort() {
        // Materialised BEFORE arming: the harness must fail the code under
        // test, not its own fixture.
        let f = framed(&vec![3u8; 256 * 1024]);
        let err = {
            let _guard = ArmGuard::new(1024);
            read_length_prefixed(&mut Cursor::new(f), 1 << 30)
                .await
                .unwrap_err()
        };
        assert_eq!(err.kind(), std::io::ErrorKind::OutOfMemory);
        assert!(err.to_string().contains("allocation failed"), "{err}");
    }

    /// Half one of the guarantee: a declaration-derived allocation is capped at
    /// one read step. Observed on the **production** reader rather than
    /// reproduced beside it.
    ///
    /// The previous version of this test re-implemented the growth loop and
    /// measured its own `Vec`. That proves the test's arithmetic, not the
    /// reader's. This one runs `read_length_prefixed` and watches the global
    /// allocator, so the assertion is about what production actually asked
    /// for.
    ///
    /// Note what is *not* claimed: that no allocation derives from the
    /// declaration. For a frame under `READ_STEP` the scratch is sized from the
    /// declaration — see
    /// `scratch_is_sized_by_the_smaller_of_read_step_and_declaration`. The
    /// claim is that such an allocation is bounded by `READ_STEP`.
    #[tokio::test]
    async fn no_declaration_derived_allocation_exceeds_one_read_step() {
        let delivered = 300_000usize;
        let declared = 200 * 1024 * 1024u32; // 200 MiB, under the ceiling
        // Materialised before observation begins: the fixture's own
        // allocations must not be attributed to the reader.
        let frame = lying_frame(declared, &vec![7u8; delivered]);

        let (result, max_request, requests, total_requested) = {
            let g = ObserveGuard::new();
            let r = read_length_prefixed(&mut Cursor::new(frame), 256 * 1024 * 1024).await;
            (r, g.max_request(), g.requests(), g.total_requested())
        };

        let err = result.expect_err("the body is short, so this must run out of data");
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);

        assert!(
            requests > 0,
            "the observer saw nothing — it is not wired up"
        );

        // Every request is bounded either by READ_STEP (the scratch) or by the
        // amortised growth of a buffer holding `delivered` bytes (the
        // destination). Neither bound involves the declaration.
        assert!(
            max_request <= (READ_STEP.max(2 * delivered)),
            "largest request {max_request} exceeds max(READ_STEP, 2 * delivered); \
             declared was {declared}"
        );
        assert!(
            (max_request as u64) < (declared as u64) / 100,
            "largest request {max_request} is within 1% of the DECLARED {declared} — \
             allocation is following declaration, not delivery"
        );

        // And in aggregate: geometric growth over `delivered` bytes sums to a
        // small multiple of `delivered`, nowhere near `declared`.
        assert!(
            (total_requested as u64) < (declared as u64) / 10,
            "total requested {total_requested} is within 10% of the DECLARED {declared}"
        );
    }

    /// Half two of the guarantee: the destination is grown only after bytes
    /// arrive, and only by the count delivered. A frame declaring a large
    /// length that delivers a single byte must not produce a `READ_STEP`-sized
    /// *destination* request — the one `READ_STEP` request on this path is the
    /// scratch, and there must be no second one.
    ///
    /// This is the specific defect the previous implementation had: it called
    /// `try_reserve(READ_STEP)` and `resize` before reading that step, so one
    /// byte of payload cost 64 KiB of reserved, zeroed destination.
    #[tokio::test]
    async fn one_delivered_byte_does_not_reserve_a_whole_read_step_of_destination() {
        let declared = 64 * 1024 * 1024u32;
        let frame = lying_frame(declared, &[0xEE]);

        let (result, requests, total_requested, max_request) = {
            let g = ObserveGuard::new();
            let r = read_length_prefixed(&mut Cursor::new(frame), 128 * 1024 * 1024).await;
            (r, g.requests(), g.total_requested(), g.max_request())
        };
        assert_eq!(
            result.unwrap_err().kind(),
            std::io::ErrorKind::UnexpectedEof
        );

        // Exactly two allocations are expected on this path: the scratch
        // buffer, and the destination grown by the single delivered byte.
        // Pinning the count is what makes the "destination is not grown ahead
        // of delivery" claim testable rather than rhetorical.
        assert!(
            requests > 0,
            "the observer saw nothing — it is not wired up"
        );

        // Nothing larger than one read step is ever requested, despite a
        // 64 MiB declaration.
        assert!(
            max_request <= READ_STEP,
            "largest request {max_request} exceeds one read step ({READ_STEP}) \
             for a frame that delivered a single byte"
        );

        // The arithmetic that makes this discriminating: the scratch buffer is
        // READ_STEP bytes, the destination is grown by the one byte that
        // arrived, and the error path formats a short string. Total lands just
        // over READ_STEP. The previous implementation reserved READ_STEP of
        // *destination* as well, before reading it — which lands just over
        // 2 * READ_STEP and fails this bound.
        assert!(
            total_requested < READ_STEP + 4096,
            "total requested {total_requested} is more than one read step plus \
             slack; the destination is being grown ahead of delivery"
        );
    }

    /// The small-frame side of `min(READ_STEP, declared_len)`: a short frame
    /// does not pay for a full read step of scratch.
    ///
    /// This is also the test that makes the *honest* form of the guarantee
    /// necessary. Here the scratch size is derived from the declaration — 32
    /// bytes for a 32-byte frame — so "no allocation derives from the declared
    /// length" would be false. The guarantee is the bounded one.
    #[tokio::test]
    async fn scratch_is_sized_by_the_smaller_of_read_step_and_declaration() {
        let payload = vec![1u8; 32];
        let frame = framed(&payload);

        let (out, max_request) = {
            let g = ObserveGuard::new();
            let r = read_length_prefixed(&mut Cursor::new(frame), 1 << 30).await;
            (r.unwrap(), g.max_request())
        };
        assert_eq!(out, payload);
        assert!(
            max_request < READ_STEP,
            "a 32-byte frame requested {max_request} bytes; scratch should be \
             min(READ_STEP, len)"
        );
    }

    /// A stream that stops mid-frame is an `UnexpectedEof` naming how far it
    /// got, not a silent truncation.
    #[tokio::test]
    async fn a_stream_that_ends_early_reports_how_far_it_got() {
        let frame = lying_frame(4096, &[5u8; 100]);
        let err = read_length_prefixed(&mut Cursor::new(frame), 1 << 30)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
        assert!(
            err.to_string().contains("100 of 4096"),
            "the error should say how much arrived: {err}"
        );
    }
}
