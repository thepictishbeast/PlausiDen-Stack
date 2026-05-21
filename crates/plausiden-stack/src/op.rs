//! Stack operations — the primitive units a Stack composes.

use plausiden_hdc::{bind, Hypervector};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors during operation execution.
#[derive(Debug, Error)]
pub enum OperationError {
    /// HDC primitive error (dim mismatch).
    #[error("hdc: {0}")]
    Hdc(#[from] plausiden_hdc::HdcError),
}

/// One operation mode within a Stack.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Operation {
    /// Pass-through. Returns the input unchanged.
    Identity,

    /// HDC dense binding: `bind(input, key)`.
    Dense {
        /// The key hypervector that input is bound against.
        key: Hypervector,
    },

    /// HRR / FFT binding via circular convolution.
    HrrBind {
        /// The key hypervector that input is convolved with.
        key: Hypervector,
    },
}

impl Operation {
    /// Apply the operation to the given input hypervector.
    pub fn apply(&self, input: &Hypervector) -> Result<Hypervector, OperationError> {
        match self {
            Operation::Identity => Ok(input.clone()),
            Operation::Dense { key } => Ok(bind(input, key)?),
            Operation::HrrBind { key } => hrr_bind(input, key),
        }
    }

    /// A short tag for tracing / logging.
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            Operation::Identity => "identity",
            Operation::Dense { .. } => "dense",
            Operation::HrrBind { .. } => "hrr_bind",
        }
    }
}

/// HRR binding via circular convolution: O(D log D), spectral-preserving.
fn hrr_bind(a: &Hypervector, b: &Hypervector) -> Result<Hypervector, OperationError> {
    if a.dim() != b.dim() {
        return Err(OperationError::Hdc(plausiden_hdc::HdcError::DimMismatch {
            a: a.dim(),
            b: b.dim(),
        }));
    }
    let dim = a.dim();
    let mut a_c: Vec<Complex<f64>> = a
        .as_slice()
        .iter()
        .map(|&x| Complex::new(f64::from(x), 0.0))
        .collect();
    let mut b_c: Vec<Complex<f64>> = b
        .as_slice()
        .iter()
        .map(|&x| Complex::new(f64::from(x), 0.0))
        .collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(dim);
    let ifft = planner.plan_fft_inverse(dim);
    fft.process(&mut a_c);
    fft.process(&mut b_c);
    let mut prod: Vec<Complex<f64>> = a_c.iter().zip(&b_c).map(|(x, y)| x * y).collect();
    ifft.process(&mut prod);

    let data: Vec<i8> = prod
        .iter()
        .map(|c| if c.re >= 0.0 { 1i8 } else { -1 })
        .collect();
    Hypervector::from_bipolar(data).ok_or(OperationError::Hdc(
        plausiden_hdc::HdcError::DimMismatch { a: dim, b: dim },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use plausiden_hdc::cos_sim;

    fn hv(seed: u64) -> Hypervector {
        Hypervector::random_seeded(1_000, seed)
    }

    #[test]
    fn identity_returns_input_unchanged() {
        let v = hv(1);
        assert_eq!(Operation::Identity.apply(&v).expect("ok"), v);
    }

    #[test]
    fn dense_self_inverse_via_double_bind() {
        let v = hv(1);
        let k = hv(2);
        let op = Operation::Dense { key: k };
        let once = op.apply(&v).expect("ok");
        let twice = op.apply(&once).expect("ok");
        assert_eq!(v, twice);
    }

    #[test]
    fn dense_binds_decorrelates() {
        let v = hv(1);
        let k = hv(2);
        let out = Operation::Dense { key: k }.apply(&v).expect("ok");
        let s = cos_sim(&out, &v).expect("ok");
        assert!(s.abs() < 0.1, "decorrelation expected, got cos_sim={s}");
    }

    #[test]
    fn hrr_bind_returns_correct_dim() {
        let v = Hypervector::random_seeded(1_024, 1);
        let k = Hypervector::random_seeded(1_024, 2);
        let out = Operation::HrrBind { key: k }.apply(&v).expect("ok");
        assert_eq!(out.dim(), 1_024);
    }

    #[test]
    fn dim_mismatch_in_dense_errors() {
        let v = Hypervector::random_seeded(100, 1);
        let k = Hypervector::random_seeded(200, 2);
        let err = Operation::Dense { key: k }
            .apply(&v)
            .expect_err("should err");
        assert!(matches!(err, OperationError::Hdc(_)));
    }

    #[test]
    fn tag_is_stable() {
        assert_eq!(Operation::Identity.tag(), "identity");
        assert_eq!(Operation::Dense { key: hv(1) }.tag(), "dense");
        assert_eq!(Operation::HrrBind { key: hv(1) }.tag(), "hrr_bind");
    }
}
