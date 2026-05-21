//! Soft-gated routing — Phase 2.
//!
//! Where [`crate::stack::Stack::forward`] bundles every operation's output
//! with equal weight, the gated variant computes a per-operation weight
//! (the "gate" value) from a learned linear projection of the input and
//! uses a *soft weighted bundle* to combine the outputs.
//!
//! This is NOT Mixture-of-Experts: the operations all share the same
//! input AND all contribute to the same bundled output. The gate just
//! says how much each contributes. A gate ≈ 1 keeps that operation's
//! full contribution; a gate ≈ 0 nearly suppresses it. Soft because the
//! gates are smoothed via softmax rather than top-K selected.
//!
//! Phase 3 extends this with a stack-of-stacks recursion. Phase 4 adds
//! the entropy gradient that perturbs gates with noise per-stack.

use plausiden_hdc::{HdcError, Hypervector};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::op::{Operation, OperationError};

/// Errors during gated-stack execution.
#[derive(Debug, Error)]
pub enum GatedError {
    /// Nested operation failed.
    #[error("op: {0}")]
    Op(#[from] OperationError),

    /// HDC primitive failure.
    #[error("hdc: {0}")]
    Hdc(#[from] HdcError),

    /// `forward` was called on an empty Stack.
    #[error("forward on empty GatedStack (need at least one operation)")]
    Empty,

    /// The gating keys aren't aligned to the operation list.
    #[error("gate length {gates} does not match op count {ops}")]
    GateMismatch {
        /// Number of gates supplied.
        gates: usize,
        /// Number of operations in the Stack.
        ops: usize,
    },
}

/// A Stack with learned per-operation soft gates.
///
/// The gating function is a simple linear scoring: dot(input_features,
/// gate_key) for each operation, then softmax across operations. Each
/// operation's output is scaled by its softmax weight before bundling.
///
/// "input_features" is the bipolar hypervector itself (cast to f64); this
/// is the simplest possible learnable projection — Phase 4+ can swap in
/// a richer gating MLP without changing this trait's surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatedStack {
    dim: usize,
    operations: Vec<Operation>,
    /// One gate key per operation. Each key is a hypervector of the same
    /// dim as the input; the scalar gate value is `cos_sim(input, key)`.
    gate_keys: Vec<Hypervector>,
}

impl GatedStack {
    /// Create an empty gated Stack at dimensionality `dim`.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            operations: Vec::new(),
            gate_keys: Vec::new(),
        }
    }

    /// Append `op` with `gate_key`.
    ///
    /// BUG ASSUMPTION: `gate_key.dim()` must equal the Stack's `dim`.
    pub fn add_operation(
        &mut self,
        op: Operation,
        gate_key: Hypervector,
    ) -> Result<(), GatedError> {
        if gate_key.dim() != self.dim {
            return Err(GatedError::Hdc(HdcError::DimMismatch {
                a: gate_key.dim(),
                b: self.dim,
            }));
        }
        self.operations.push(op);
        self.gate_keys.push(gate_key);
        Ok(())
    }

    /// Stack dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// True if no operations have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Compute the raw gate scores (before softmax) for a given input.
    ///
    /// Returns one score per operation. Useful for inspection / debugging
    /// the gating decisions.
    pub fn raw_gates(&self, input: &Hypervector) -> Result<Vec<f64>, GatedError> {
        if input.dim() != self.dim {
            return Err(GatedError::Hdc(HdcError::DimMismatch {
                a: input.dim(),
                b: self.dim,
            }));
        }
        self.gate_keys
            .iter()
            .map(|k| plausiden_hdc::cos_sim(input, k).map_err(GatedError::Hdc))
            .collect()
    }

    /// Compute the softmax-normalised gate weights for a given input.
    ///
    /// `temperature` scales the logits (higher T = more uniform; lower T =
    /// closer to one-hot). Default 1.0.
    pub fn softmax_gates(
        &self,
        input: &Hypervector,
        temperature: f64,
    ) -> Result<Vec<f64>, GatedError> {
        if temperature <= 0.0 {
            return Err(GatedError::Hdc(HdcError::DimMismatch { a: 0, b: 0 }));
        }
        let scores = self.raw_gates(input)?;
        Ok(softmax(&scores, temperature))
    }

    /// Apply all operations and combine via softmax-weighted bundle.
    ///
    /// Soft weighted bundle: sum_i(weight_i * sign-preserved op_i(input)),
    /// then threshold back to bipolar. `temperature` controls gate sharpness.
    pub fn forward(
        &self,
        input: &Hypervector,
        temperature: f64,
    ) -> Result<Hypervector, GatedError> {
        if self.operations.is_empty() {
            return Err(GatedError::Empty);
        }
        let weights = self.softmax_gates(input, temperature)?;

        let outs: Result<Vec<Hypervector>, OperationError> =
            self.operations.iter().map(|op| op.apply(input)).collect();
        let outs = outs?;

        // Weighted sum in f64, then sign-threshold back to bipolar.
        let dim = self.dim;
        let mut acc = vec![0.0_f64; dim];
        for (w, out) in weights.iter().zip(&outs) {
            for (i, &x) in out.as_slice().iter().enumerate() {
                acc[i] += w * f64::from(x);
            }
        }
        let data: Vec<i8> = acc
            .into_iter()
            .map(|s| if s >= 0.0 { 1i8 } else { -1 })
            .collect();
        Hypervector::from_bipolar(data)
            .ok_or(GatedError::Hdc(HdcError::DimMismatch { a: dim, b: dim }))
    }
}

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
        // Degenerate: all -inf. Return uniform.
        return vec![1.0 / scores.len() as f64; scores.len()];
    }
    exps.into_iter().map(|e| e / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    #[test]
    fn empty_gated_stack_errors_on_forward() {
        let g = GatedStack::new(1_000);
        assert!(g.is_empty());
        let err = g.forward(&hv(1), 1.0).expect_err("err");
        assert!(matches!(err, GatedError::Empty));
    }

    #[test]
    fn add_operation_rejects_wrong_dim_key() {
        let mut g = GatedStack::new(1_000);
        let wrong = Hypervector::random_seeded(500, 1);
        let err = g
            .add_operation(Operation::Identity, wrong)
            .expect_err("err");
        assert!(matches!(err, GatedError::Hdc(_)));
    }

    #[test]
    fn raw_gates_returns_one_per_op() {
        let mut g = GatedStack::new(1_000);
        g.add_operation(Operation::Identity, hv(10)).expect("ok");
        g.add_operation(Operation::Identity, hv(11)).expect("ok");
        g.add_operation(Operation::Identity, hv(12)).expect("ok");
        let scores = g.raw_gates(&hv(1)).expect("ok");
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn softmax_gates_sum_to_one() {
        let mut g = GatedStack::new(1_000);
        g.add_operation(Operation::Identity, hv(10)).expect("ok");
        g.add_operation(Operation::Identity, hv(11)).expect("ok");
        let weights = g.softmax_gates(&hv(1), 1.0).expect("ok");
        let total: f64 = weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "softmax sum = {total}");
        assert!(weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn matched_gate_key_dominates_with_low_temperature() {
        // If one gate key IS the input, that op's softmax weight should be near 1
        // at low temperature.
        let v = hv(42);
        let mut g = GatedStack::new(1_000);
        g.add_operation(Operation::Identity, v.clone()).expect("ok"); // perfect match
        g.add_operation(Operation::Identity, hv(99)).expect("ok"); // random
        g.add_operation(Operation::Identity, hv(100)).expect("ok"); // random
        let weights = g.softmax_gates(&v, 0.01).expect("ok");
        assert!(
            weights[0] > 0.99,
            "matched key should dominate at low T, got {:?}",
            weights
        );
    }

    #[test]
    fn forward_single_identity_op_returns_input() {
        let mut g = GatedStack::new(1_000);
        g.add_operation(Operation::Identity, hv(10)).expect("ok");
        let v = hv(1);
        let out = g.forward(&v, 1.0).expect("ok");
        // Single op + soft weight 1.0 + identity = input recovered.
        assert_eq!(out, v);
    }

    #[test]
    fn forward_two_identity_ops_bundle_back_to_input() {
        let mut g = GatedStack::new(1_000);
        g.add_operation(Operation::Identity, hv(10)).expect("ok");
        g.add_operation(Operation::Identity, hv(11)).expect("ok");
        let v = hv(1);
        let out = g.forward(&v, 1.0).expect("ok");
        // weighted bundle of (v, v) is v regardless of weights.
        assert_eq!(out, v);
    }

    #[test]
    fn softmax_with_uniform_scores_is_uniform() {
        let s = softmax(&[0.0, 0.0, 0.0, 0.0], 1.0);
        for w in &s {
            assert!((w - 0.25).abs() < 1e-9, "uniform expected, got {w}");
        }
    }
}
