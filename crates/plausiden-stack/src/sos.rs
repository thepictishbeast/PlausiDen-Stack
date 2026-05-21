//! Stack-of-Stacks — Phase 3 recursion via a shared blackboard hypervector.
//!
//! Per docs/ARCHITECTURE.md §4.2: at level 1, N Level-0 stacks share a
//! single HDC vector as a blackboard. Each substructure writes its
//! contribution by binding to a role key and bundling into the
//! blackboard; readers extract by unbinding.
//!
//! For Phase 3 we model the simplest case: every substructure produces a
//! hypervector contribution from the shared input, and the level-1
//! output is the bundle of (role_key ⊗ contribution) across substructures.
//! Phase 4 adds entropy gradients on per-substructure stochasticity;
//! Phase 5 wires self-modification.

use plausiden_hdc::{bind, bundle, unbind, HdcError, Hypervector};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::stack::{Stack, StackError};

/// Errors during Stack-of-Stacks execution.
#[derive(Debug, Error)]
pub enum SosError {
    /// A nested Stack execution failed.
    #[error("stack: {0}")]
    Stack(#[from] StackError),

    /// HDC primitive failure.
    #[error("hdc: {0}")]
    Hdc(#[from] HdcError),

    /// `forward` was called on an empty StackOfStacks.
    #[error("forward on empty StackOfStacks")]
    Empty,

    /// Role key dim doesn't match the blackboard dim.
    #[error("role key dim {got} doesn't match blackboard dim {expected}")]
    KeyDimMismatch {
        /// Role key dim.
        got: usize,
        /// Expected dim.
        expected: usize,
    },

    /// Substructure produced output of wrong dim.
    #[error("substructure #{index} produced dim {got}, expected {expected}")]
    SubstructureDimMismatch {
        /// Index of the substructure.
        index: usize,
        /// Actual output dim.
        got: usize,
        /// Expected dim.
        expected: usize,
    },
}

/// One substructure attached to a Stack-of-Stacks: a Stack + its role key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Substructure {
    /// The nested Stack.
    pub stack: Stack,
    /// The role key this substructure binds its output to before
    /// contributing to the blackboard.
    pub role_key: Hypervector,
}

/// Level-1 Stack composition: N substructures sharing a blackboard hypervector.
///
/// All substructures see the same input and contribute to one composite
/// output. Readers extract a specific substructure's contribution by
/// unbinding with the corresponding role key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackOfStacks {
    dim: usize,
    substructures: Vec<Substructure>,
}

impl StackOfStacks {
    /// Construct an empty Level-1 structure at dimensionality `dim`.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            substructures: Vec::new(),
        }
    }

    /// Attach a Stack with its role key.
    ///
    /// Returns an error if the Stack's dim or role key's dim doesn't
    /// match the blackboard dim.
    pub fn attach(&mut self, stack: Stack, role_key: Hypervector) -> Result<(), SosError> {
        if stack.dim() != self.dim {
            return Err(SosError::KeyDimMismatch {
                got: stack.dim(),
                expected: self.dim,
            });
        }
        if role_key.dim() != self.dim {
            return Err(SosError::KeyDimMismatch {
                got: role_key.dim(),
                expected: self.dim,
            });
        }
        self.substructures.push(Substructure { stack, role_key });
        Ok(())
    }

    /// Blackboard dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of substructures attached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.substructures.len()
    }

    /// True if no substructures attached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.substructures.is_empty()
    }

    /// Read-only access to attached substructures.
    #[must_use]
    pub fn substructures(&self) -> &[Substructure] {
        &self.substructures
    }

    /// Compute the blackboard hypervector for the given input.
    ///
    /// Each substructure produces `contribution_i = stack_i.forward(input)`,
    /// which is then bound to its role key. All `bind(role_key, contribution)`
    /// vectors are bundled into the blackboard.
    ///
    /// Returns Err(Empty) on an empty structure.
    pub fn forward(&self, input: &Hypervector) -> Result<Hypervector, SosError> {
        if self.substructures.is_empty() {
            return Err(SosError::Empty);
        }
        if input.dim() != self.dim {
            return Err(SosError::KeyDimMismatch {
                got: input.dim(),
                expected: self.dim,
            });
        }

        let mut bound: Vec<Hypervector> = Vec::with_capacity(self.substructures.len());
        for (i, sub) in self.substructures.iter().enumerate() {
            let contribution = sub.stack.forward(input)?;
            if contribution.dim() != self.dim {
                return Err(SosError::SubstructureDimMismatch {
                    index: i,
                    got: contribution.dim(),
                    expected: self.dim,
                });
            }
            bound.push(bind(&sub.role_key, &contribution)?);
        }
        let refs: Vec<&Hypervector> = bound.iter().collect();
        Ok(bundle(&refs)?)
    }

    /// Read one substructure's contribution back out of a blackboard
    /// hypervector by unbinding with the role key.
    ///
    /// Returns the noisy reconstruction. For random role keys at high dim,
    /// the recovered vector retains substantial similarity to the original
    /// contribution (cosine >> 0.5 typically) and can be cleaned up
    /// against a codebook.
    pub fn read_role(
        &self,
        blackboard: &Hypervector,
        index: usize,
    ) -> Result<Hypervector, SosError> {
        let sub = self
            .substructures
            .get(index)
            .ok_or(SosError::SubstructureDimMismatch {
                index,
                got: 0,
                expected: self.substructures.len(),
            })?;
        Ok(unbind(blackboard, &sub.role_key)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::Operation;
    use plausiden_hdc::cos_sim;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    fn identity_stack() -> Stack {
        Stack::new(1_000).with_operation(Operation::Identity)
    }

    #[test]
    fn empty_sos_errors_on_forward() {
        let s = StackOfStacks::new(1_000);
        assert!(s.is_empty());
        let err = s.forward(&hv(1)).expect_err("err");
        assert!(matches!(err, SosError::Empty));
    }

    #[test]
    fn attach_rejects_wrong_dim_stack() {
        let mut sos = StackOfStacks::new(1_000);
        let wrong_stack = Stack::new(500).with_operation(Operation::Identity);
        let err = sos.attach(wrong_stack, hv(1)).expect_err("err");
        assert!(matches!(
            err,
            SosError::KeyDimMismatch {
                got: 500,
                expected: 1_000
            }
        ));
    }

    #[test]
    fn attach_rejects_wrong_dim_role_key() {
        let mut sos = StackOfStacks::new(1_000);
        let wrong_key = Hypervector::random_seeded(500, 1);
        let err = sos.attach(identity_stack(), wrong_key).expect_err("err");
        assert!(matches!(
            err,
            SosError::KeyDimMismatch {
                got: 500,
                expected: 1_000
            }
        ));
    }

    #[test]
    fn single_substructure_blackboard_recoverable_via_unbind() {
        // For one substructure with identity Stack: contribution = input.
        // blackboard = bind(role_key, input).
        // unbind(blackboard, role_key) = input (bipolar self-inverse).
        let mut sos = StackOfStacks::new(1_000);
        sos.attach(identity_stack(), hv(10)).expect("ok");
        let v = hv(1);
        let blackboard = sos.forward(&v).expect("ok");
        let recovered = sos.read_role(&blackboard, 0).expect("ok");
        assert_eq!(recovered, v);
    }

    #[test]
    fn three_substructure_contributions_partial_recovery() {
        // For three substructures, each contribution is partially recoverable
        // via unbind. Recovered ≈ contribution + noise from the other terms;
        // cos_sim(recovered, contribution) should be substantially > 0.
        let mut sos = StackOfStacks::new(10_000);
        sos.attach(
            Stack::new(10_000).with_operation(Operation::Identity),
            Hypervector::random_seeded(10_000, 10),
        )
        .expect("ok");
        sos.attach(
            Stack::new(10_000).with_operation(Operation::Identity),
            Hypervector::random_seeded(10_000, 11),
        )
        .expect("ok");
        sos.attach(
            Stack::new(10_000).with_operation(Operation::Identity),
            Hypervector::random_seeded(10_000, 12),
        )
        .expect("ok");
        let v = Hypervector::random_seeded(10_000, 1);
        let blackboard = sos.forward(&v).expect("ok");
        let recovered = sos.read_role(&blackboard, 0).expect("ok");
        let sim = cos_sim(&recovered, &v).expect("ok");
        // With 3 sources at D=10_000, cos_sim of recovered to the true
        // contribution should be ~0.5 (1/sqrt(N) attenuation in bipolar HDC).
        assert!(sim > 0.3, "recovery via unbind too weak: {sim}");
    }

    #[test]
    fn len_and_is_empty_track_substructures() {
        let mut sos = StackOfStacks::new(1_000);
        assert!(sos.is_empty());
        assert_eq!(sos.len(), 0);
        sos.attach(identity_stack(), hv(10)).expect("ok");
        assert!(!sos.is_empty());
        assert_eq!(sos.len(), 1);
    }

    #[test]
    fn input_dim_mismatch_errors() {
        let mut sos = StackOfStacks::new(1_000);
        sos.attach(identity_stack(), hv(10)).expect("ok");
        let wrong = Hypervector::random_seeded(500, 1);
        let err = sos.forward(&wrong).expect_err("err");
        assert!(matches!(err, SosError::KeyDimMismatch { .. }));
    }

    #[test]
    fn read_role_out_of_range_errors() {
        let mut sos = StackOfStacks::new(1_000);
        sos.attach(identity_stack(), hv(10)).expect("ok");
        let blackboard = sos.forward(&hv(1)).expect("ok");
        let err = sos.read_role(&blackboard, 99).expect_err("err");
        assert!(matches!(err, SosError::SubstructureDimMismatch { .. }));
    }
}
