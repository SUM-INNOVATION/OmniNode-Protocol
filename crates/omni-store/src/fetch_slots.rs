//! The fetch-concurrency bound for blocking shard downloads (issue #132).
//!
//! [`FetchSlots`] is a counting limiter with first-in, first-out admission. A
//! thread that wants to download a shard takes a [`FetchPermit`]; while the
//! limit is reached it waits in line, and it is admitted in the order it
//! arrived. The permit gives its slot back when it is dropped, so a download
//! that completes, fails or unwinds from a panic always frees its slot.
//!
//! The peer-to-peer path ([`crate::FetchManager`]) is driven by events and
//! never blocks, so it keeps its own queue instead of waiting here.

use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};

use crate::error::{Result, StoreError};

/// A first-in, first-out counting limiter.
pub(crate) struct FetchSlots {
    state: Mutex<State>,
    changed: Condvar,
}

struct State {
    limit: usize,
    in_use: usize,
    /// The ticket the next arrival gets.
    next_ticket: u64,
    /// The oldest ticket not yet admitted. Only its holder may take a slot,
    /// so a later arrival cannot overtake an earlier one.
    next_admitted: u64,
}

/// One held slot. Dropping it frees the slot.
#[must_use = "the slot is freed as soon as the permit is dropped"]
pub(crate) struct FetchPermit<'a> {
    slots: &'a FetchSlots,
}

impl FetchSlots {
    /// A limiter admitting at most `limit` holders at once. `limit` must be
    /// at least 1, by the same rule as `StoreConfig::max_concurrent_fetches`.
    pub(crate) fn new(limit: usize) -> Result<Self> {
        omni_types::config::validate_max_concurrent_fetches(limit).map_err(StoreError::Other)?;
        Ok(Self {
            state: Mutex::new(State {
                limit,
                in_use: 0,
                next_ticket: 0,
                next_admitted: 0,
            }),
            changed: Condvar::new(),
        })
    }

    pub(crate) fn limit(&self) -> usize {
        self.lock().limit
    }

    /// Take a slot, waiting in line while all of them are held.
    pub(crate) fn acquire(&self) -> FetchPermit<'_> {
        let mut state = self.lock();
        let ticket = state.next_ticket;
        state.next_ticket += 1;
        while state.next_admitted != ticket || state.in_use >= state.limit {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.next_admitted += 1;
        state.in_use += 1;
        drop(state);
        // The next ticket in line may fit as well.
        self.changed.notify_all();
        FetchPermit { slots: self }
    }

    fn release(&self) {
        let mut state = self.lock();
        state.in_use -= 1;
        drop(state);
        self.changed.notify_all();
    }

    /// The counters stay consistent under every critical section here, none
    /// of which can panic, so a poisoned lock is still safe to use.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    #[cfg(test)]
    pub(crate) fn in_use(&self) -> usize {
        self.lock().in_use
    }

    #[cfg(test)]
    pub(crate) fn waiting(&self) -> usize {
        let state = self.lock();
        (state.next_ticket - state.next_admitted) as usize
    }
}

impl Drop for FetchPermit<'_> {
    fn drop(&mut self) {
        self.slots.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out waiting until {what}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn a_zero_limit_is_refused() {
        assert!(FetchSlots::new(0).is_err());
        assert_eq!(FetchSlots::new(1).unwrap().limit(), 1);
    }

    #[test]
    fn at_most_the_limit_is_held_and_dropping_a_permit_frees_its_slot() {
        let slots = FetchSlots::new(2).unwrap();
        let a = slots.acquire();
        let b = slots.acquire();
        assert_eq!(slots.in_use(), 2);

        std::thread::scope(|s| {
            let third = s.spawn(|| {
                let _c = slots.acquire();
                slots.in_use()
            });
            wait_until("the third caller waits", || slots.waiting() == 1);
            std::thread::sleep(Duration::from_millis(50));
            assert_eq!(slots.in_use(), 2, "a third holder must wait");
            drop(a);
            assert_eq!(third.join().unwrap(), 2, "admitted into the freed slot");
        });
        drop(b);
        assert_eq!((slots.in_use(), slots.waiting()), (0, 0));
    }

    #[test]
    fn waiting_callers_are_admitted_in_arrival_order() {
        let slots = FetchSlots::new(1).unwrap();
        let order = Mutex::new(Vec::new());
        let held = slots.acquire();
        std::thread::scope(|s| {
            for name in ["first", "second", "third"] {
                let before = slots.waiting();
                let (slots, order) = (&slots, &order);
                s.spawn(move || {
                    let _p = slots.acquire();
                    order.lock().unwrap().push(name);
                });
                // Each caller is in line before the next one arrives.
                wait_until("the caller is in line", || slots.waiting() == before + 1);
            }
            drop(held);
        });
        assert_eq!(*order.lock().unwrap(), vec!["first", "second", "third"]);
    }

    #[test]
    fn a_permit_dropped_by_a_panic_frees_its_slot() {
        let slots = FetchSlots::new(1).unwrap();
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _p = slots.acquire();
            panic!("download panicked");
        }));
        assert!(unwound.is_err());
        assert_eq!(slots.in_use(), 0);
        let _again = slots.acquire();
    }
}
