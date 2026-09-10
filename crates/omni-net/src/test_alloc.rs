//! Test-only instrumented allocator, shared by the framing tests.
//!
//! Two capabilities, both per-thread and both off by default, so the other
//! tests in this binary are unaffected:
//!
//! * **Failing** — refuse allocations at or above a threshold, so the reader's
//!   fallible-growth path can be driven for real ([`ArmGuard`]).
//! * **Observing** — record the size of every allocation and reallocation
//!   request, so a test can assert what the production code *asked the
//!   allocator for* rather than reproducing its arithmetic ([`ObserveGuard`]).
//!
//! One `#[global_allocator]` exists per test binary, so the harness lives here
//! rather than inside a single test module.
//!
//! The observation path must not allocate — it runs inside the allocator, and
//! a `Vec` here would recurse — so it is `Cell<usize>` arithmetic only, and it
//! records aggregates rather than a log of sizes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    /// Allocations of at least this size fail while armed.
    static FAIL_AT_OR_ABOVE: Cell<usize> = const { Cell::new(usize::MAX) };
    /// Whether request sizes are being recorded on this thread.
    static OBSERVING: Cell<bool> = const { Cell::new(false) };
    /// Largest single request seen while observing.
    static MAX_REQUEST: Cell<usize> = const { Cell::new(0) };
    /// How many requests were seen while observing.
    static REQUEST_COUNT: Cell<usize> = const { Cell::new(0) };
    /// Sum of all request sizes seen while observing.
    static TOTAL_REQUESTED: Cell<usize> = const { Cell::new(0) };
}

/// Record one request size. Allocation-free by construction.
fn note(size: usize) {
    if OBSERVING.try_with(|c| c.get()).unwrap_or(false) {
        let _ = MAX_REQUEST.try_with(|c| c.set(c.get().max(size)));
        let _ = REQUEST_COUNT.try_with(|c| c.set(c.get() + 1));
        let _ = TOTAL_REQUESTED.try_with(|c| c.set(c.get().saturating_add(size)));
    }
}

pub(crate) struct Failing;

unsafe impl GlobalAlloc for Failing {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        let armed = FAIL_AT_OR_ABOVE.try_with(|c| c.get()).unwrap_or(usize::MAX);
        if layout.size() >= armed {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        let armed = FAIL_AT_OR_ABOVE.try_with(|c| c.get()).unwrap_or(usize::MAX);
        if new_size >= armed {
            return std::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Arms the failing allocator for as long as it is held.
///
/// RAII rather than manual arm/disarm: a panic between the two would leave
/// allocation failure enabled for every later test on this thread, turning one
/// failure into a cascade of unrelated ones.
pub(crate) struct ArmGuard(usize);

impl ArmGuard {
    pub(crate) fn new(threshold: usize) -> Self {
        ArmGuard(FAIL_AT_OR_ABOVE.with(|c| c.replace(threshold)))
    }
}

impl Drop for ArmGuard {
    fn drop(&mut self) {
        FAIL_AT_OR_ABOVE.with(|c| c.set(self.0));
    }
}

/// Records allocation and reallocation request sizes for as long as it is held.
///
/// RAII for the same reason as [`ArmGuard`]: a panic between manual start and
/// stop would leave every later test on this thread paying for observation and
/// reading someone else's totals.
pub(crate) struct ObserveGuard {
    was_observing: bool,
    prev_max: usize,
    prev_count: usize,
    prev_total: usize,
}

impl ObserveGuard {
    pub(crate) fn new() -> Self {
        let g = Self {
            was_observing: OBSERVING.with(|c| c.get()),
            prev_max: MAX_REQUEST.with(|c| c.get()),
            prev_count: REQUEST_COUNT.with(|c| c.get()),
            prev_total: TOTAL_REQUESTED.with(|c| c.get()),
        };
        MAX_REQUEST.with(|c| c.set(0));
        REQUEST_COUNT.with(|c| c.set(0));
        TOTAL_REQUESTED.with(|c| c.set(0));
        OBSERVING.with(|c| c.set(true));
        g
    }

    /// Largest single request seen so far.
    pub(crate) fn max_request(&self) -> usize {
        MAX_REQUEST.with(|c| c.get())
    }

    /// How many requests were seen so far.
    pub(crate) fn requests(&self) -> usize {
        REQUEST_COUNT.with(|c| c.get())
    }

    /// Sum of every request size seen so far.
    pub(crate) fn total_requested(&self) -> usize {
        TOTAL_REQUESTED.with(|c| c.get())
    }
}

impl Drop for ObserveGuard {
    fn drop(&mut self) {
        OBSERVING.with(|c| c.set(self.was_observing));
        MAX_REQUEST.with(|c| c.set(self.prev_max));
        REQUEST_COUNT.with(|c| c.set(self.prev_count));
        TOTAL_REQUESTED.with(|c| c.set(self.prev_total));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_harness_is_not_vacuous_and_disarms_on_drop() {
        let refused = {
            let _g = ArmGuard::new(1 << 20);
            Vec::<u8>::new().try_reserve(4 << 20).is_err()
        };
        let allowed = {
            let _g = ArmGuard::new(1 << 20);
            Vec::<u8>::new().try_reserve(1024).is_ok()
        };
        assert!(refused, "a large reservation must be refused while armed");
        assert!(allowed, "a small one must still succeed");
        assert!(
            Vec::<u8>::new().try_reserve(4 << 20).is_ok(),
            "disarms on drop"
        );
    }

    #[test]
    fn the_observer_is_not_vacuous_and_restores_on_drop() {
        let (max, count) = {
            let g = ObserveGuard::new();
            let mut v: Vec<u8> = Vec::new();
            v.try_reserve_exact(4096).unwrap();
            v.resize(4096, 0);
            (g.max_request(), g.requests())
        };
        assert!(count >= 1, "the observer saw no requests at all");
        assert!(
            max >= 4096,
            "a 4096-byte reservation must be visible, saw max {max}"
        );

        // Outside any guard nothing is recorded, so a later guard starts clean.
        let mut v: Vec<u8> = Vec::new();
        v.try_reserve_exact(1 << 20).unwrap();
        let g = ObserveGuard::new();
        assert_eq!(g.max_request(), 0, "a fresh guard starts at zero");
        assert_eq!(g.requests(), 0);
    }

    #[test]
    fn the_guard_restores_the_threshold_on_panic() {
        let before = std::panic::catch_unwind(|| {
            let _g = ArmGuard::new(1024);
            panic!("boom");
        });
        assert!(before.is_err());
        assert!(Vec::<u8>::new().try_reserve(4 << 20).is_ok());
    }
}
