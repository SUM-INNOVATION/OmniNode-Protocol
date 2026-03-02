//! GPipe-style micro-batch scheduling.
//!
//! ```text
//! Time →  0    1    2    3    4    5    6    7
//! S0:    [m0] [m1] [m2] [m3]
//! S1:         [m0] [m1] [m2] [m3]
//! S2:              [m0] [m1] [m2] [m3]
//!
//! Pipeline bubble = S-1 time slots at start + end
//! Efficiency = M / (M + S - 1)
//! ```

/// A single cell in the GPipe execution grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleCell {
    /// Which pipeline stage executes this cell.
    pub stage_index: u32,
    /// Which micro-batch is being processed.
    pub micro_batch_index: u32,
    /// Discrete time slot (0-indexed).
    pub time_slot: u32,
}

/// Deterministic GPipe micro-batch schedule.
///
/// Each node knows exactly when to expect input and when to send output
/// by consulting the cells for its stage index.
#[derive(Debug, Clone)]
pub struct MicroBatchSchedule {
    num_stages: u32,
    num_micro_batches: u32,
    cells: Vec<ScheduleCell>,
}

impl MicroBatchSchedule {
    /// Generate the forward-pass schedule for GPipe.
    ///
    /// Stage `s` processes micro-batch `m` at time slot `s + m`.
    pub fn new(num_stages: u32, num_micro_batches: u32) -> Self {
        let mut cells = Vec::with_capacity((num_stages * num_micro_batches) as usize);

        for m in 0..num_micro_batches {
            for s in 0..num_stages {
                cells.push(ScheduleCell {
                    stage_index: s,
                    micro_batch_index: m,
                    time_slot: s + m,
                });
            }
        }

        cells.sort_by_key(|c| (c.time_slot, c.stage_index));

        Self {
            num_stages,
            num_micro_batches,
            cells,
        }
    }

    pub fn num_stages(&self) -> u32 {
        self.num_stages
    }

    pub fn num_micro_batches(&self) -> u32 {
        self.num_micro_batches
    }

    /// All cells assigned to a specific stage, in time order.
    pub fn cells_for_stage(&self, stage_index: u32) -> Vec<&ScheduleCell> {
        self.cells
            .iter()
            .filter(|c| c.stage_index == stage_index)
            .collect()
    }

    /// All cells executing at a specific time slot.
    pub fn cells_at_time(&self, time_slot: u32) -> Vec<&ScheduleCell> {
        self.cells
            .iter()
            .filter(|c| c.time_slot == time_slot)
            .collect()
    }

    /// Total number of discrete time slots in the schedule.
    pub fn total_time_slots(&self) -> u32 {
        self.num_micro_batches + self.num_stages - 1
    }

    /// Pipeline efficiency: ratio of useful work to total grid cells.
    ///
    /// `M / (M + S - 1)` where M = micro-batches, S = stages.
    pub fn efficiency(&self) -> f64 {
        let useful = self.num_micro_batches as f64;
        let total = self.total_time_slots() as f64;
        if total == 0.0 {
            0.0
        } else {
            useful / total
        }
    }

    /// Access the raw cell list.
    pub fn all_cells(&self) -> &[ScheduleCell] {
        &self.cells
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_stages_four_batches() {
        let sched = MicroBatchSchedule::new(3, 4);

        assert_eq!(sched.num_stages(), 3);
        assert_eq!(sched.num_micro_batches(), 4);
        assert_eq!(sched.total_time_slots(), 6); // 4 + 3 - 1
        assert_eq!(sched.all_cells().len(), 12); // 3 × 4

        // Stage 0 runs at time slots 0, 1, 2, 3
        let s0 = sched.cells_for_stage(0);
        assert_eq!(s0.len(), 4);
        assert_eq!(s0[0].time_slot, 0);
        assert_eq!(s0[3].time_slot, 3);

        // Stage 2 runs at time slots 2, 3, 4, 5
        let s2 = sched.cells_for_stage(2);
        assert_eq!(s2.len(), 4);
        assert_eq!(s2[0].time_slot, 2);
        assert_eq!(s2[3].time_slot, 5);
    }

    #[test]
    fn efficiency_calculation() {
        // 2 stages, 4 micro-batches → efficiency = 4 / (4 + 2 - 1) = 4/5 = 0.8
        let sched = MicroBatchSchedule::new(2, 4);
        let eff = sched.efficiency();
        assert!((eff - 0.8).abs() < 1e-10);
    }

    #[test]
    fn single_stage_full_efficiency() {
        let sched = MicroBatchSchedule::new(1, 4);
        let eff = sched.efficiency();
        assert!((eff - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cells_at_time_concurrent() {
        let sched = MicroBatchSchedule::new(3, 4);
        // At time slot 2, all 3 stages are active:
        // S0:m2, S1:m1, S2:m0
        let t2 = sched.cells_at_time(2);
        assert_eq!(t2.len(), 3);
    }
}
