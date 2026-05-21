//! Hyperdimensional Computing primitives for PlausiDen-Stack.
//!
//! BUG ASSUMPTION: bipolar `{-1, +1}` only in v0. Binary `{0, 1}` HRR variant
//! is a future option behind a feature flag.
//!
//! The fundamental data type is `Hypervector` — a fixed-length vector of bipolar
//! integers. The default dimensionality is 10,000 (Kanerva's recommended baseline
//! for symbolic robustness).
//!
//! Five canonical operations:
//!
//! - `random(D)`             — draw a fresh hypervector from uniform bipolar
//! - `bind(a, b)`            — elementwise multiplication; self-inverse for bipolar
//! - `bundle(a, b, ...)`     — additive superposition with majority threshold
//! - `permute(v, n)`         — circular shift (used for position-tagging)
//! - `unbind(c, k)`          — `bind(c, k)` since bipolar binding is self-inverse
//!
//! Plus similarity probes:
//!
//! - `cos_sim(a, b)`         — cosine similarity in `[-1, 1]`
//! - `hamming(a, b)`         — fraction of differing dimensions in `[0, 1]`
//!
//! Algebraic laws (asserted in property tests):
//!
//! - `bind(bind(a, k), k) ≈ a`    (binding is self-inverse)
//! - `bundle(a, b) = bundle(b, a)` (bundling is commutative)
//! - `permute(permute(v, n), m) = permute(v, n + m)` (permutation composes)

#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Default hypervector dimensionality (Kanerva 1988 recommended baseline).
pub const DEFAULT_DIM: usize = 10_000;

/// Errors that can occur in HDC operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum HdcError {
    /// Two hypervectors have incompatible dimensionality for the requested op.
    #[error("dim mismatch: {a} vs {b}")]
    DimMismatch {
        /// Left operand dimensionality.
        a: usize,
        /// Right operand dimensionality.
        b: usize,
    },

    /// `bundle` was given an empty input slice.
    #[error("bundle requires at least one operand")]
    EmptyBundle,
}

/// A bipolar hypervector: each element is `+1` or `-1`.
///
/// BUG ASSUMPTION: invariant maintained at construction. Operations may
/// produce intermediates outside `{-1, +1}` (e.g. raw `bundle` sum) but
/// public API restores the invariant before returning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hypervector {
    data: Vec<i8>,
}

impl Hypervector {
    /// Construct a fresh random hypervector of dimensionality `dim`.
    ///
    /// Each element is drawn uniformly from `{-1, +1}`.
    ///
    /// BUG ASSUMPTION: `dim == 0` produces an empty vector, which subsequently
    /// breaks `bind` and `bundle`. Callers must use `dim ≥ 1`.
    #[must_use]
    pub fn random(dim: usize, rng: &mut impl Rng) -> Self {
        let mut data = Vec::with_capacity(dim);
        for _ in 0..dim {
            data.push(if rng.gen::<bool>() { 1 } else { -1 });
        }
        Self { data }
    }

    /// Construct a random hypervector with a deterministic seed.
    ///
    /// Useful for reproducible tests + benchmarks.
    #[must_use]
    pub fn random_seeded(dim: usize, seed: u64) -> Self {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        Self::random(dim, &mut rng)
    }

    /// Construct from raw bipolar data. Returns `None` if any element is not
    /// in `{-1, +1}`.
    #[must_use]
    pub fn from_bipolar(data: Vec<i8>) -> Option<Self> {
        if data.iter().all(|&x| x == 1 || x == -1) {
            Some(Self { data })
        } else {
            None
        }
    }

    /// Returns the dimensionality of this hypervector.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Returns a view of the underlying bipolar data.
    #[must_use]
    pub fn as_slice(&self) -> &[i8] {
        &self.data
    }
}

/// Bind two hypervectors via elementwise multiplication.
///
/// For bipolar vectors, `bind` is **self-inverse**: `bind(bind(a, k), k) = a`.
/// Use this property to "unbind" — see [`unbind`].
///
/// BUG ASSUMPTION: returns `Err(DimMismatch)` if dimensions differ. Both
/// operands must have equal `dim()`.
pub fn bind(a: &Hypervector, b: &Hypervector) -> Result<Hypervector, HdcError> {
    if a.dim() != b.dim() {
        return Err(HdcError::DimMismatch {
            a: a.dim(),
            b: b.dim(),
        });
    }
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect();
    Ok(Hypervector { data })
}

/// Unbind: given a bound `c = bind(a, k)`, recover `a` by binding with `k`.
///
/// For bipolar vectors this is just `bind` (self-inverse property).
/// Kept as a separate function for code-reading clarity at call sites.
pub fn unbind(c: &Hypervector, k: &Hypervector) -> Result<Hypervector, HdcError> {
    bind(c, k)
}

/// Bundle: additive superposition of N hypervectors with majority threshold.
///
/// Sums elementwise, then `sign()` to restore bipolar invariant.
/// Ties (sum == 0) resolve to `+1` deterministically.
///
/// BUG ASSUMPTION: empty slice returns `Err(EmptyBundle)`. All operands must
/// have equal `dim()`.
pub fn bundle(vectors: &[&Hypervector]) -> Result<Hypervector, HdcError> {
    let Some(first) = vectors.first() else {
        return Err(HdcError::EmptyBundle);
    };
    let dim = first.dim();
    for v in vectors.iter().skip(1) {
        if v.dim() != dim {
            return Err(HdcError::DimMismatch { a: dim, b: v.dim() });
        }
    }
    let mut sums = vec![0i32; dim];
    for v in vectors {
        for (i, &x) in v.data.iter().enumerate() {
            sums[i] += i32::from(x);
        }
    }
    let data = sums
        .into_iter()
        .map(|s| if s >= 0 { 1i8 } else { -1 })
        .collect();
    Ok(Hypervector { data })
}

/// Permute a hypervector by circular shift of `shift` positions.
///
/// Used for role-tagging in sequence + set encodings. Permutation composes:
/// `permute(permute(v, n), m) == permute(v, n + m mod D)`.
#[must_use]
pub fn permute(v: &Hypervector, shift: usize) -> Hypervector {
    let dim = v.dim();
    if dim == 0 {
        return v.clone();
    }
    let shift = shift % dim;
    if shift == 0 {
        return v.clone();
    }
    let mut data = vec![0i8; dim];
    for i in 0..dim {
        data[(i + shift) % dim] = v.data[i];
    }
    Hypervector { data }
}

/// Cosine similarity between two hypervectors. Returns a value in `[-1, 1]`.
///
/// For random bipolar hypervectors of dim `D ≥ 1000`, independence implies
/// `cos_sim ≈ 0` with stddev `≈ 1/√D`. Similar vectors have `cos_sim → 1`;
/// negated vectors have `cos_sim → -1`.
///
/// BUG ASSUMPTION: returns 0.0 if either operand is all-zero (cannot happen
/// for properly constructed bipolar vectors but defensive).
pub fn cos_sim(a: &Hypervector, b: &Hypervector) -> Result<f64, HdcError> {
    if a.dim() != b.dim() {
        return Err(HdcError::DimMismatch {
            a: a.dim(),
            b: b.dim(),
        });
    }
    if a.data.is_empty() {
        return Ok(0.0);
    }
    let dot: i64 = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| i64::from(*x) * i64::from(*y))
        .sum();
    // For bipolar vectors of equal dim D, |a| = |b| = √D, so |a||b| = D.
    #[allow(clippy::cast_precision_loss)]
    let dim_f = a.dim() as f64;
    #[allow(clippy::cast_precision_loss)]
    let dot_f = dot as f64;
    Ok(dot_f / dim_f)
}

/// Hamming distance: fraction of dimensions where two hypervectors differ.
///
/// Returns a value in `[0.0, 1.0]`. `0.0` = identical, `1.0` = fully negated.
/// For random bipolar vectors of high dim, expected hamming ≈ 0.5.
pub fn hamming(a: &Hypervector, b: &Hypervector) -> Result<f64, HdcError> {
    if a.dim() != b.dim() {
        return Err(HdcError::DimMismatch {
            a: a.dim(),
            b: b.dim(),
        });
    }
    if a.data.is_empty() {
        return Ok(0.0);
    }
    let diffs = a.data.iter().zip(&b.data).filter(|(x, y)| x != y).count();
    #[allow(clippy::cast_precision_loss)]
    let total = a.dim() as f64;
    #[allow(clippy::cast_precision_loss)]
    let diffs_f = diffs as f64;
    Ok(diffs_f / total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_seeded_is_deterministic() {
        let a = Hypervector::random_seeded(100, 42);
        let b = Hypervector::random_seeded(100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn random_seeded_diff_seed_diff_vector() {
        let a = Hypervector::random_seeded(100, 42);
        let b = Hypervector::random_seeded(100, 43);
        assert_ne!(a, b);
    }

    #[test]
    fn from_bipolar_validates() {
        assert!(Hypervector::from_bipolar(vec![1, -1, 1, -1]).is_some());
        assert!(Hypervector::from_bipolar(vec![1, 0, 1]).is_none()); // 0 not bipolar
        assert!(Hypervector::from_bipolar(vec![1, 2, 1]).is_none()); // 2 not bipolar
    }

    #[test]
    fn bind_self_inverse() {
        let a = Hypervector::random_seeded(1000, 1);
        let k = Hypervector::random_seeded(1000, 2);
        let bound = bind(&a, &k).expect("dim match");
        let recovered = unbind(&bound, &k).expect("dim match");
        assert_eq!(a, recovered);
    }

    #[test]
    fn bundle_commutative() {
        let a = Hypervector::random_seeded(1000, 1);
        let b = Hypervector::random_seeded(1000, 2);
        let ab = bundle(&[&a, &b]).expect("ok");
        let ba = bundle(&[&b, &a]).expect("ok");
        assert_eq!(ab, ba);
    }

    #[test]
    fn bundle_similar_to_operands() {
        // Bundle of two random vectors should be more similar to each operand
        // than two random vectors are to each other (cos_sim ≈ 0.5 vs ≈ 0).
        let a = Hypervector::random_seeded(10_000, 1);
        let b = Hypervector::random_seeded(10_000, 2);
        let bundled = bundle(&[&a, &b]).expect("ok");
        let sim_to_a = cos_sim(&bundled, &a).expect("ok");
        let sim_to_b = cos_sim(&bundled, &b).expect("ok");
        let sim_a_b = cos_sim(&a, &b).expect("ok");
        assert!(sim_to_a > 0.3, "bundled ↔ a: {sim_to_a}");
        assert!(sim_to_b > 0.3, "bundled ↔ b: {sim_to_b}");
        assert!(sim_a_b.abs() < 0.1, "a ↔ b: {sim_a_b}");
    }

    #[test]
    fn permute_composes() {
        let v = Hypervector::random_seeded(100, 1);
        let p1 = permute(&v, 3);
        let p2 = permute(&p1, 5);
        let p_combined = permute(&v, 8);
        assert_eq!(p2, p_combined);
    }

    #[test]
    fn permute_full_cycle_identity() {
        let v = Hypervector::random_seeded(100, 1);
        let cycled = permute(&v, 100);
        assert_eq!(v, cycled);
    }

    #[test]
    fn cos_sim_self_is_one() {
        let v = Hypervector::random_seeded(1000, 1);
        let s = cos_sim(&v, &v).expect("ok");
        assert!((s - 1.0).abs() < 1e-10, "cos_sim(v, v) = {s}, expected 1.0");
    }

    #[test]
    fn cos_sim_random_pair_near_zero() {
        // For D=10_000, stddev of cos_sim is ~0.01; bound to 0.05.
        let a = Hypervector::random_seeded(10_000, 1);
        let b = Hypervector::random_seeded(10_000, 2);
        let s = cos_sim(&a, &b).expect("ok");
        assert!(s.abs() < 0.05, "cos_sim of random pair = {s}");
    }

    #[test]
    fn hamming_self_is_zero() {
        let v = Hypervector::random_seeded(1000, 1);
        let h = hamming(&v, &v).expect("ok");
        assert_eq!(h, 0.0);
    }

    #[test]
    fn hamming_random_pair_near_half() {
        let a = Hypervector::random_seeded(10_000, 1);
        let b = Hypervector::random_seeded(10_000, 2);
        let h = hamming(&a, &b).expect("ok");
        assert!((h - 0.5).abs() < 0.05, "hamming of random pair = {h}");
    }

    #[test]
    fn bind_dim_mismatch_errors() {
        let a = Hypervector::random_seeded(100, 1);
        let b = Hypervector::random_seeded(200, 2);
        assert_eq!(bind(&a, &b), Err(HdcError::DimMismatch { a: 100, b: 200 }));
    }

    #[test]
    fn bundle_empty_errors() {
        let empty: &[&Hypervector] = &[];
        assert_eq!(bundle(empty), Err(HdcError::EmptyBundle));
    }
}
