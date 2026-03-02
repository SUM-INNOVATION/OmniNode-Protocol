//! Liveness detection for pipeline stages.
//!
//! Default: 3-second heartbeat interval, 3× timeout (9 seconds).
//! If a stage misses 3 consecutive heartbeats, it is declared dead.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Tracks heartbeat timestamps for active pipeline stages.
pub struct HeartbeatMonitor {
    interval: Duration,
    timeout: Duration,
    /// `(session_id, stage_index) → last heartbeat time`
    last_seen: HashMap<(String, u32), Instant>,
}

impl HeartbeatMonitor {
    pub fn new(interval_secs: u64, timeout_factor: u32) -> Self {
        let interval = Duration::from_secs(interval_secs);
        let timeout = interval * timeout_factor;
        Self {
            interval,
            timeout,
            last_seen: HashMap::new(),
        }
    }

    /// Record a heartbeat for a stage. Resets the timeout clock.
    pub fn record_heartbeat(&mut self, session_id: &str, stage_index: u32) {
        self.last_seen
            .insert((session_id.to_string(), stage_index), Instant::now());
    }

    /// Initialize all stages for a session with the current timestamp.
    pub fn init_session(&mut self, session_id: &str, num_stages: u32) {
        let now = Instant::now();
        for stage in 0..num_stages {
            self.last_seen
                .insert((session_id.to_string(), stage), now);
        }
    }

    /// Return stage indices that have exceeded the timeout threshold.
    pub fn check_timeouts(&self, session_id: &str, num_stages: u32) -> Vec<u32> {
        let now = Instant::now();
        let mut timed_out = Vec::new();

        for stage in 0..num_stages {
            let key = (session_id.to_string(), stage);
            if let Some(last) = self.last_seen.get(&key) {
                if now.duration_since(*last) > self.timeout {
                    timed_out.push(stage);
                }
            }
        }

        timed_out
    }

    /// Check if a specific stage is still alive.
    pub fn is_alive(&self, session_id: &str, stage_index: u32) -> bool {
        let key = (session_id.to_string(), stage_index);
        match self.last_seen.get(&key) {
            Some(last) => Instant::now().duration_since(*last) <= self.timeout,
            None => false,
        }
    }

    pub fn interval(&self) -> Duration {
        self.interval
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Remove all heartbeat state for a session.
    pub fn clear_session(&mut self, session_id: &str) {
        self.last_seen.retain(|k, _| k.0 != session_id);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_and_alive() {
        let mut hb = HeartbeatMonitor::new(3, 3);
        hb.init_session("sess-1", 3);

        assert!(hb.is_alive("sess-1", 0));
        assert!(hb.is_alive("sess-1", 1));
        assert!(hb.is_alive("sess-1", 2));
        assert!(!hb.is_alive("sess-1", 3)); // out of range
        assert!(!hb.is_alive("sess-2", 0)); // wrong session
    }

    #[test]
    fn no_timeouts_initially() {
        let mut hb = HeartbeatMonitor::new(3, 3);
        hb.init_session("sess-1", 3);
        assert!(hb.check_timeouts("sess-1", 3).is_empty());
    }

    #[test]
    fn clear_removes_session() {
        let mut hb = HeartbeatMonitor::new(3, 3);
        hb.init_session("sess-1", 2);
        hb.clear_session("sess-1");
        assert!(!hb.is_alive("sess-1", 0));
    }

    #[test]
    fn durations() {
        let hb = HeartbeatMonitor::new(3, 3);
        assert_eq!(hb.interval(), Duration::from_secs(3));
        assert_eq!(hb.timeout(), Duration::from_secs(9));
    }
}
