//! Stack — heterogeneous bundle of operations on a shared hypervector.

use plausiden_hdc::{bundle, Hypervector};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::op::{Operation, OperationError};

/// Errors during Stack execution.
#[derive(Debug, Error)]
pub enum StackError {
    /// A nested operation failed.
    #[error("op: {0}")]
    Op(#[from] OperationError),

    /// HDC primitive failure.
    #[error("hdc: {0}")]
    Hdc(#[from] plausiden_hdc::HdcError),

    /// `forward` was called on an empty Stack.
    #[error("forward on empty Stack (need at least one operation)")]
    Empty,

    /// An operation produced output of incompatible dimensionality.
    #[error("op `{op}` produced dim {got}, expected {expected}")]
    DimMismatch {
        /// Operation tag.
        op: &'static str,
        /// Actual output dimensionality.
        got: usize,
        /// Expected dimensionality (the Stack's `dim`).
        expected: usize,
    },
}

/// A Stack — N operations sharing a hypervector input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stack {
    dim: usize,
    operations: Vec<Operation>,
}

impl Stack {
    /// Create an empty Stack at dimensionality `dim`.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            operations: Vec::new(),
        }
    }

    /// Builder: append `op` to this Stack.
    #[must_use]
    pub fn with_operation(mut self, op: Operation) -> Self {
        self.operations.push(op);
        self
    }

    /// Add `op` in place.
    pub fn add_operation(&mut self, op: Operation) -> &mut Self {
        self.operations.push(op);
        self
    }

    /// Insert `op` at `index`, shifting later ops back. Panics if
    /// `index > self.len()`.
    pub fn insert_operation(&mut self, index: usize, op: Operation) {
        self.operations.insert(index, op);
    }

    /// Remove and return the op at `index`. Panics if `index >= self.len()`.
    pub fn remove_operation(&mut self, index: usize) -> Operation {
        self.operations.remove(index)
    }

    /// Replace the op at `index`, returning the previous occupant. Panics
    /// if `index >= self.len()`.
    pub fn replace_operation(&mut self, index: usize, op: Operation) -> Operation {
        std::mem::replace(&mut self.operations[index], op)
    }

    /// Returns the Stack's dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of operations in the Stack.
    #[must_use]
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Returns true if no operations have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Read-only access to operations.
    #[must_use]
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Apply all operations to `input` in parallel, then bundle outputs.
    pub fn forward(&self, input: &Hypervector) -> Result<Hypervector, StackError> {
        if self.operations.is_empty() {
            return Err(StackError::Empty);
        }
        if input.dim() != self.dim {
            return Err(StackError::DimMismatch {
                op: "stack.input",
                got: input.dim(),
                expected: self.dim,
            });
        }

        let outs: Result<Vec<Hypervector>, OperationError> =
            self.operations.iter().map(|op| op.apply(input)).collect();
        let outs = outs?;
        for (op, out) in self.operations.iter().zip(&outs) {
            if out.dim() != self.dim {
                return Err(StackError::DimMismatch {
                    op: op.tag(),
                    got: out.dim(),
                    expected: self.dim,
                });
            }
        }
        let refs: Vec<&Hypervector> = outs.iter().collect();
        Ok(bundle(&refs)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plausiden_hdc::cos_sim;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    #[test]
    fn empty_stack_errors() {
        let s = Stack::new(1_000);
        assert!(s.is_empty());
        assert!(matches!(
            s.forward(&hv(1)).expect_err("err"),
            StackError::Empty
        ));
    }

    #[test]
    fn identity_stack_is_identity() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let v = hv(1);
        assert_eq!(s.forward(&v).expect("ok"), v);
    }

    #[test]
    fn two_identity_ops_bundle_back_to_input() {
        let s = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Identity);
        let v = hv(1);
        assert_eq!(s.forward(&v).expect("ok"), v);
    }

    #[test]
    fn dense_plus_identity_partial_recovery() {
        let v = hv(1);
        let k = hv(2);
        let s = Stack::new(1_000)
            .with_operation(Operation::Dense { key: k })
            .with_operation(Operation::Identity);
        let out = s.forward(&v).expect("ok");
        let sim = cos_sim(&out, &v).expect("ok");
        assert!(sim > 0.3, "identity contribution should remain: {sim}");
        assert!(sim < 0.95, "should be diluted by dense binding: {sim}");
    }

    #[test]
    fn input_dim_mismatch_errors() {
        let s = Stack::new(1_000).with_operation(Operation::Identity);
        let wrong = Hypervector::random_seeded(500, 1);
        let err = s.forward(&wrong).expect_err("err");
        assert!(matches!(err, StackError::DimMismatch { .. }));
    }

    #[test]
    fn len_and_is_empty_track_operations() {
        let mut s = Stack::new(1_000);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        s.add_operation(Operation::Identity);
        assert!(!s.is_empty());
        assert_eq!(s.len(), 1);
    }
}
