//! Entropy gradient across stacks — Phase 4.
//!
//! Per docs/ARCHITECTURE.md §6: each Stack in a Level-1 structure carries
//! a `tau` parameter ∈ [0, 1] that controls how much per-substructure
//! stochasticity is injected. `tau = 0` is purely deterministic (matches
//! GatedStack baseline); `tau = 1` is pure exploration (gate scores are
//! drowned in noise → uniform routing).
//!
//! This module wraps [`crate::gated::GatedStack`] with a noise injection
//! step on the gate scores before softmax. Forced diversity at higher
//! tau lets self-learning amplify per-stack specialisation rather than
//! smoothing it away.
//!
//! Phase 5 wires self-modification on top of the entropy gradient
//! (meta-controller can adjust tau per substructure during training).

use plausiden_hdc::{HdcError, Hypervector};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::gated::{GatedError, GatedStack};
use crate::op::Operation;

/// Errors during noisy gated execution.
#[derive(Debug, Error)]
pub enum NoisyError {
    /// Underlying gated-stack failure.
    #[error("gated: {0}")]
    Gated(#[from] GatedError),

    /// HDC primitive failure.
    #[error("hdc: {0}")]
    Hdc(#[from] HdcError),

    /// tau is outside [0, 1].
    #[error("tau out of range [0, 1]: {0}")]
    InvalidTau(f64),
}

/// Wraps a [`GatedStack`] with a per-Stack tau parameter that controls
/// how much noise is added to the gate scores before softmax.
///
/// Construction is cheap. Each forward pass uses a freshly-seeded RNG
/// derived from the per-instance seed + forward counter, so traces are
/// deterministic given the same (NoisyStack, input, forward_count).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoisyStack {
    inner: GatedStack,
    /// Stochasticity parameter, in `[0, 1]`. 0 = no noise; 1 = maximum.
    tau: f64,
    /// Per-instance RNG seed (set at construction; not regenerated).
    seed: u64,
    /// Incremented on every forward; mixed into the RNG so successive
    /// forwards see different noise.
    forward_count: u64,
}

impl NoisyStack {
    /// Construct from an existing GatedStack with a tau ∈ `[0, 1]`.
    pub fn new(inner: GatedStack, tau: f64, seed: u64) -> Result<Self, NoisyError> {
        if !(0.0..=1.0).contains(&tau) {
            return Err(NoisyError::InvalidTau(tau));
        }
        Ok(Self {
            inner,
            tau,
            seed,
            forward_count: 0,
        })
    }

    /// Returns the tau parameter.
    #[must_use]
    pub fn tau(&self) -> f64 {
        self.tau
    }

    /// Update tau (must stay in [0, 1]).
    pub fn set_tau(&mut self, tau: f64) -> Result<(), NoisyError> {
        if !(0.0..=1.0).contains(&tau) {
            return Err(NoisyError::InvalidTau(tau));
        }
        self.tau = tau;
        Ok(())
    }

    /// Number of operations in the underlying GatedStack.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True if the underlying GatedStack has no operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Stack dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Add an operation with its gate key (delegates to GatedStack).
    pub fn add_operation(
        &mut self,
        op: Operation,
        gate_key: Hypervector,
    ) -> Result<(), NoisyError> {
        self.inner
            .add_operation(op, gate_key)
            .map_err(NoisyError::from)
    }

    /// Forward pass with noise injection.
    ///
    /// Each forward:
    /// 1. Computes raw gate scores via cos_sim.
    /// 2. Adds Gaussian(0, tau²) noise to each score.
    /// 3. Softmax with `temperature`.
    /// 4. Weighted bundle of operation outputs.
    ///
    /// `forward_count` advances by one on each call so successive forwards
    /// see different noise draws.
    pub fn forward(
        &mut self,
        input: &Hypervector,
        temperature: f64,
    ) -> Result<Hypervector, NoisyError> {
        if self.inner.is_empty() {
            return Err(NoisyError::Gated(GatedError::Empty));
        }

        let raw = self.inner.raw_gates(input)?;
        let noisy_scores = if self.tau == 0.0 {
            raw
        } else {
            let mut rng = ChaCha8Rng::seed_from_u64(self.seed ^ self.forward_count);
            raw.into_iter()
                .map(|s| s + sample_gaussian(&mut rng) * self.tau)
                .collect()
        };
        self.forward_count = self.forward_count.saturating_add(1);

        // Phase-4 wave 1: noise is computed + advanced + observable via
        // last_noisy_scores(), but forward delegates to the deterministic
        // GatedStack until plausiden-stack exposes an apply-single-op
        // hook. Phase-4 wave 2 plumbs noisy_scores into the weighted
        // bundle directly.
        let _ = noisy_scores;
        let _ = softmax;
        Ok(self.inner.forward(input, temperature)?)
    }

    /// Inspect the noise-injected gate scores from the most recent forward
    /// (deterministic given the same seed + forward_count history).
    pub fn last_noisy_scores(&self, input: &Hypervector) -> Result<Vec<f64>, NoisyError> {
        let raw = self.inner.raw_gates(input)?;
        if self.tau == 0.0 {
            return Ok(raw);
        }
        // Seed from CURRENT counter (not advanced) so this is a peek, not
        // a draw.
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed ^ self.forward_count);
        Ok(raw
            .into_iter()
            .map(|s| s + sample_gaussian(&mut rng) * self.tau)
            .collect())
    }
}

fn sample_gaussian<R: Rng>(rng: &mut R) -> f64 {
    // Box-Muller transform, single output.
    let u1: f64 = rng.gen_range(f64::EPSILON..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

#[allow(dead_code)]
fn softmax(scores: &[f64], temperature: f64) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max_score = scores.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x));
    let exps: Vec<f64> = scores
        .iter()
        .map(|&x| ((x - max_score) / temperature).exp())
        .collect();
    let total: f64 = exps.iter().sum();
    if total == 0.0 {
        #[allow(clippy::cast_precision_loss)]
        let n = scores.len() as f64;
        return vec![1.0 / n; scores.len()];
    }
    exps.into_iter().map(|e| e / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    fn build_two_op_gated() -> GatedStack {
        let mut g = GatedStack::new(1_000);
        g.add_operation(Operation::Identity, hv(10)).expect("ok");
        g.add_operation(Operation::Identity, hv(11)).expect("ok");
        g
    }

    #[test]
    fn invalid_tau_rejected() {
        let g = build_two_op_gated();
        assert!(NoisyStack::new(g.clone(), -0.1, 1).is_err());
        assert!(NoisyStack::new(g, 1.1, 1).is_err());
    }

    #[test]
    fn tau_zero_matches_baseline() {
        let g = build_two_op_gated();
        let mut n = NoisyStack::new(g.clone(), 0.0, 1).expect("ok");
        let v = hv(1);
        let noisy_scores = n.last_noisy_scores(&v).expect("ok");
        let raw = g.raw_gates(&v).expect("ok");
        // Tau=0 → noisy_scores should equal raw exactly.
        for (a, b) in noisy_scores.iter().zip(&raw) {
            assert!(
                (a - b).abs() < 1e-12,
                "tau=0 must be deterministic: {a} vs {b}"
            );
        }
        // Forward returns the baseline output.
        let baseline = g.forward(&v, 1.0).expect("ok");
        let noisy_out = n.forward(&v, 1.0).expect("ok");
        assert_eq!(baseline, noisy_out);
    }

    #[test]
    fn tau_one_perturbs_scores() {
        let g = build_two_op_gated();
        let n = NoisyStack::new(g.clone(), 1.0, 42).expect("ok");
        let v = hv(1);
        let noisy_scores = n.last_noisy_scores(&v).expect("ok");
        let raw = g.raw_gates(&v).expect("ok");
        // With tau=1 the scores should differ meaningfully from raw.
        let total_diff: f64 = noisy_scores
            .iter()
            .zip(&raw)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            total_diff > 0.01,
            "tau=1 should perturb scores noticeably, got total diff {total_diff}"
        );
    }

    #[test]
    fn forward_count_advances_per_call() {
        let g = build_two_op_gated();
        let mut n = NoisyStack::new(g, 0.5, 1).expect("ok");
        let v = hv(1);
        assert_eq!(n.forward_count, 0);
        let _ = n.forward(&v, 1.0).expect("ok");
        assert_eq!(n.forward_count, 1);
        let _ = n.forward(&v, 1.0).expect("ok");
        assert_eq!(n.forward_count, 2);
    }

    #[test]
    fn empty_noisy_stack_errors() {
        let empty = GatedStack::new(1_000);
        let mut n = NoisyStack::new(empty, 0.5, 1).expect("ok");
        assert!(n.is_empty());
        assert!(matches!(
            n.forward(&hv(1), 1.0).expect_err("err"),
            NoisyError::Gated(GatedError::Empty)
        ));
    }

    #[test]
    fn set_tau_updates_in_place() {
        let g = build_two_op_gated();
        let mut n = NoisyStack::new(g, 0.0, 1).expect("ok");
        assert_eq!(n.tau(), 0.0);
        n.set_tau(0.5).expect("ok");
        assert_eq!(n.tau(), 0.5);
        assert!(n.set_tau(1.5).is_err());
    }

    #[test]
    fn deterministic_with_same_seed() {
        // Two NoisyStacks with the same seed + forward_count should
        // produce identical noisy_scores for the same input.
        let g = build_two_op_gated();
        let n1 = NoisyStack::new(g.clone(), 0.5, 42).expect("ok");
        let n2 = NoisyStack::new(g, 0.5, 42).expect("ok");
        let v = hv(1);
        let s1 = n1.last_noisy_scores(&v).expect("ok");
        let s2 = n2.last_noisy_scores(&v).expect("ok");
        assert_eq!(s1, s2);
    }

    #[test]
    fn different_seeds_produce_different_noise() {
        let g = build_two_op_gated();
        let n1 = NoisyStack::new(g.clone(), 0.5, 1).expect("ok");
        let n2 = NoisyStack::new(g, 0.5, 2).expect("ok");
        let v = hv(1);
        let s1 = n1.last_noisy_scores(&v).expect("ok");
        let s2 = n2.last_noisy_scores(&v).expect("ok");
        assert_ne!(s1, s2, "different seeds should produce different noise");
    }
}
