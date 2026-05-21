//! Self-modification meta-controller — Phase 5.
//!
//! Per docs/ARCHITECTURE.md §7: the meta-controller observes per-operation
//! performance and decides which ops to reweight, remove, or add.
//!
//! Wave 1 ships a deterministic greedy controller:
//!
//! - Per-operation rolling-mean reward tracker (online updates).
//! - Decisions exposed via [`MetaController::should_remove`] +
//!   [`MetaController::weight_suggestions`].
//! - Mutation primitives [`Stack::insert_operation`] /
//!   [`Stack::remove_operation`] / [`Stack::replace_operation`] are
//!   re-used from Phase 1.
//!
//! Meta-learning loops (RL / MAML / evolutionary) are Phase 5 wave 2; this
//! wave gives the GUI + REPL a deterministic baseline to drive.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::op::Operation;
use crate::stack::Stack;

/// Errors raised by the meta-controller.
#[derive(Debug, Error)]
pub enum MetaError {
    /// observed operation index out of range.
    #[error("operation index {index} out of range (Stack has {len} ops)")]
    IndexOutOfRange {
        /// The bad index.
        index: usize,
        /// Stack length at observation time.
        len: usize,
    },
}

/// Greedy meta-controller for op-level self-modification.
///
/// Tracks a rolling mean of per-operation rewards (any scalar; convention
/// is "higher is better, e.g. cos_sim(expected, actual)").
///
/// `min_observations` is the minimum samples before a removal suggestion
/// is reliable; defaults to 5. `removal_threshold` is the mean-reward
/// floor below which an op is flagged for removal (defaults to 0.0 —
/// i.e. ops that hurt more than help).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaController {
    rewards: Vec<RewardTracker>,
    min_observations: usize,
    removal_threshold: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RewardTracker {
    sum: f64,
    count: u64,
}

impl RewardTracker {
    fn observe(&mut self, reward: f64) {
        self.sum += reward;
        self.count = self.count.saturating_add(1);
    }

    #[allow(clippy::cast_precision_loss)]
    fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.sum / self.count as f64)
        }
    }
}

impl Default for MetaController {
    fn default() -> Self {
        Self::new(5, 0.0)
    }
}

impl MetaController {
    /// Construct a fresh controller.
    #[must_use]
    pub fn new(min_observations: usize, removal_threshold: f64) -> Self {
        Self {
            rewards: Vec::new(),
            min_observations,
            removal_threshold,
        }
    }

    /// Resize the per-op tracking vector to match the Stack's op count.
    /// Call once after attaching the controller and again after any
    /// add/remove that changes the op count.
    pub fn sync_to(&mut self, stack: &Stack) {
        self.rewards.resize(stack.len(), RewardTracker::default());
    }

    /// Record an observed reward for the operation at `index`.
    pub fn observe(&mut self, index: usize, reward: f64) -> Result<(), MetaError> {
        if index >= self.rewards.len() {
            return Err(MetaError::IndexOutOfRange {
                index,
                len: self.rewards.len(),
            });
        }
        self.rewards[index].observe(reward);
        Ok(())
    }

    /// Mean reward observed for operation at `index`. None if no
    /// observations yet OR if index is out of range.
    pub fn mean_reward(&self, index: usize) -> Option<f64> {
        self.rewards.get(index).and_then(RewardTracker::mean)
    }

    /// Total observations seen for operation at `index`.
    pub fn observation_count(&self, index: usize) -> u64 {
        self.rewards.get(index).map_or(0, |r| r.count)
    }

    /// Should the operation at `index` be removed?
    ///
    /// Returns true iff:
    /// 1. At least `min_observations` samples have been seen, AND
    /// 2. mean_reward is below `removal_threshold`.
    pub fn should_remove(&self, index: usize) -> bool {
        let Some(r) = self.rewards.get(index) else {
            return false;
        };
        if r.count < self.min_observations as u64 {
            return false;
        }
        r.mean().is_some_and(|m| m < self.removal_threshold)
    }

    /// Normalised weight suggestions for the operations (Σ = 1).
    ///
    /// Ops with positive mean reward get weight proportional to their
    /// reward; ops with non-positive reward get zero (let the bundler
    /// down-weight them). Returns uniform weights if no positives.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn weight_suggestions(&self) -> Vec<f64> {
        if self.rewards.is_empty() {
            return Vec::new();
        }
        let positives: Vec<f64> = self
            .rewards
            .iter()
            .map(|r| r.mean().unwrap_or(0.0).max(0.0))
            .collect();
        let total: f64 = positives.iter().sum();
        if total <= 0.0 {
            let n = self.rewards.len() as f64;
            return vec![1.0 / n; self.rewards.len()];
        }
        positives.into_iter().map(|p| p / total).collect()
    }

    /// Apply the suggested removals to a Stack in-place.
    ///
    /// Removes ops back-to-front so indices stay valid mid-loop. Updates
    /// the per-op tracking vector to match. Returns the list of indices
    /// that were removed.
    pub fn apply_removals(&mut self, stack: &mut Stack) -> Vec<usize> {
        let mut removed = Vec::new();
        let n = self.rewards.len().min(stack.len());
        for i in (0..n).rev() {
            if self.should_remove(i) {
                stack.remove_operation(i);
                self.rewards.remove(i);
                removed.push(i);
            }
        }
        removed
    }

    /// Suggest adding a new operation: returns true if total rewards are
    /// trending high enough that an extra op might help. Returns false
    /// when the Stack already has at least one op with very high reward
    /// (no need to add more) or no observations yet.
    ///
    /// This is a rough heuristic — caller is expected to make the final
    /// add decision based on richer context.
    pub fn should_add(&self) -> bool {
        if self.rewards.is_empty() {
            return true; // empty Stack always benefits from a first op
        }
        let total_obs: u64 = self.rewards.iter().map(|r| r.count).sum();
        if total_obs < self.min_observations as u64 {
            return false;
        }
        let max_mean = self
            .rewards
            .iter()
            .filter_map(RewardTracker::mean)
            .fold(f64::NEG_INFINITY, f64::max);
        // Already have a star performer — don't add more clutter.
        max_mean < 0.8
    }

    /// Builder helper: add an operation to a Stack AND sync the controller.
    pub fn add_operation(&mut self, stack: &mut Stack, op: Operation) {
        stack.add_operation(op);
        self.rewards.push(RewardTracker::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plausiden_hdc::Hypervector;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    #[test]
    fn default_controller_has_reasonable_thresholds() {
        let m = MetaController::default();
        assert_eq!(m.min_observations, 5);
        assert_eq!(m.removal_threshold, 0.0);
    }

    #[test]
    fn sync_to_resizes_rewards_to_match() {
        let s = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Identity);
        let mut m = MetaController::default();
        m.sync_to(&s);
        assert_eq!(m.rewards.len(), 2);
    }

    #[test]
    fn observe_rejects_out_of_range_index() {
        let mut m = MetaController::default();
        let err = m.observe(0, 1.0).expect_err("err");
        assert!(matches!(
            err,
            MetaError::IndexOutOfRange { index: 0, len: 0 }
        ));
    }

    #[test]
    fn mean_reward_after_observations() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let mut m = MetaController::default();
        m.sync_to(&s);
        m.observe(0, 1.0).expect("ok");
        m.observe(0, 3.0).expect("ok");
        m.observe(0, 5.0).expect("ok");
        assert!((m.mean_reward(0).expect("set") - 3.0).abs() < 1e-9);
        assert_eq!(m.observation_count(0), 3);
    }

    #[test]
    fn should_remove_requires_min_observations() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let mut m = MetaController::new(5, 0.0);
        m.sync_to(&s);
        // 3 observations of low reward — not enough to trigger removal.
        for _ in 0..3 {
            m.observe(0, -1.0).expect("ok");
        }
        assert!(!m.should_remove(0));
        // 2 more observations — now at min_observations threshold.
        for _ in 0..2 {
            m.observe(0, -1.0).expect("ok");
        }
        assert!(m.should_remove(0));
    }

    #[test]
    fn should_remove_respects_threshold() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let mut m = MetaController::new(3, 0.5);
        m.sync_to(&s);
        // Mean = 0.6, threshold = 0.5 → don't remove.
        m.observe(0, 0.5).expect("ok");
        m.observe(0, 0.6).expect("ok");
        m.observe(0, 0.7).expect("ok");
        assert!(!m.should_remove(0));
    }

    #[test]
    fn weight_suggestions_proportional_to_reward() {
        let s = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Dense { key: hv(1) });
        let mut m = MetaController::default();
        m.sync_to(&s);
        // op0: high reward; op1: low reward.
        m.observe(0, 0.9).expect("ok");
        m.observe(1, 0.1).expect("ok");
        let w = m.weight_suggestions();
        let total: f64 = w.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
        assert!(w[0] > w[1], "high-reward op should outweigh low-reward");
    }

    #[test]
    fn weight_suggestions_uniform_when_all_non_positive() {
        let s = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Identity);
        let mut m = MetaController::default();
        m.sync_to(&s);
        m.observe(0, -1.0).expect("ok");
        m.observe(1, -2.0).expect("ok");
        let w = m.weight_suggestions();
        for wi in &w {
            assert!((wi - 0.5).abs() < 1e-9, "uniform expected, got {wi}");
        }
    }

    #[test]
    fn apply_removals_strips_low_reward_ops() {
        let mut s = Stack::new(1_000)
            .with_operation(Operation::Identity) // op 0 — bad
            .with_operation(Operation::Dense { key: hv(1) }) // op 1 — good
            .with_operation(Operation::Identity); // op 2 — bad
        let mut m = MetaController::new(3, 0.0);
        m.sync_to(&s);
        for _ in 0..3 {
            m.observe(0, -1.0).expect("ok");
            m.observe(1, 0.5).expect("ok");
            m.observe(2, -1.0).expect("ok");
        }
        let removed = m.apply_removals(&mut s);
        // Removed indices reported back-to-front.
        assert_eq!(removed, vec![2, 0]);
        // op1 (Dense) survives as the only remaining op.
        assert_eq!(s.len(), 1);
        assert_eq!(s.operations()[0].tag(), "dense");
        // Reward tracker stays in sync.
        assert_eq!(m.rewards.len(), 1);
    }

    #[test]
    fn should_add_yes_when_empty() {
        let m = MetaController::default();
        assert!(m.should_add());
    }

    #[test]
    fn should_add_no_before_enough_observations() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let mut m = MetaController::new(10, 0.0);
        m.sync_to(&s);
        m.observe(0, 0.5).expect("ok");
        // Only 1 observation, below min_observations=10.
        assert!(!m.should_add());
    }

    #[test]
    fn should_add_no_if_star_performer_exists() {
        let s = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Identity);
        let mut m = MetaController::new(2, 0.0);
        m.sync_to(&s);
        m.observe(0, 0.9).expect("ok");
        m.observe(0, 0.95).expect("ok");
        m.observe(1, 0.5).expect("ok");
        m.observe(1, 0.5).expect("ok");
        // op0 has mean 0.925 > 0.8 star threshold → don't add.
        assert!(!m.should_add());
    }

    #[test]
    fn add_operation_keeps_tracker_in_sync() {
        let mut s = Stack::new(1_000);
        let mut m = MetaController::default();
        m.sync_to(&s);
        m.add_operation(&mut s, Operation::Identity);
        m.add_operation(&mut s, Operation::Identity);
        assert_eq!(s.len(), 2);
        assert_eq!(m.rewards.len(), 2);
    }
}
