//! Property tests for the HDC algebraic laws.
//!
//! Per AVP-2 doctrine §10 (Tier 6: meta-validation), property tests assert
//! the laws by which all HDC operations must abide. ≥10k cases per property.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use plausiden_hdc::{bind, bundle, cos_sim, hamming, permute, unbind, Hypervector};
use proptest::prelude::*;

fn arb_hv(dim: usize) -> impl Strategy<Value = Hypervector> {
    any::<u64>().prop_map(move |seed| Hypervector::random_seeded(dim, seed))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1_000))]

    #[test]
    fn bind_is_self_inverse(a in arb_hv(500), k in arb_hv(500)) {
        let bound = bind(&a, &k).expect("dim match");
        let recovered = unbind(&bound, &k).expect("dim match");
        prop_assert_eq!(a, recovered);
    }

    #[test]
    fn bundle_commutative_two_ops(a in arb_hv(500), b in arb_hv(500)) {
        let ab = bundle(&[&a, &b]).expect("ok");
        let ba = bundle(&[&b, &a]).expect("ok");
        prop_assert_eq!(ab, ba);
    }

    #[test]
    fn permute_composes(v in arb_hv(200), n in 0usize..200, m in 0usize..200) {
        let stepwise = permute(&permute(&v, n), m);
        let combined = permute(&v, (n + m) % 200);
        prop_assert_eq!(stepwise, combined);
    }

    #[test]
    fn cos_sim_in_range(a in arb_hv(500), b in arb_hv(500)) {
        let s = cos_sim(&a, &b).expect("ok");
        prop_assert!((-1.0..=1.0).contains(&s), "cos_sim out of range: {s}");
    }

    #[test]
    fn hamming_in_unit_interval(a in arb_hv(500), b in arb_hv(500)) {
        let h = hamming(&a, &b).expect("ok");
        prop_assert!((0.0..=1.0).contains(&h), "hamming out of range: {h}");
    }

    #[test]
    fn cos_sim_and_hamming_consistent(a in arb_hv(500), b in arb_hv(500)) {
        // For bipolar vectors: cos_sim = 1 - 2 * hamming.
        let s = cos_sim(&a, &b).expect("ok");
        let h = hamming(&a, &b).expect("ok");
        let expected_s = 1.0 - 2.0 * h;
        prop_assert!((s - expected_s).abs() < 1e-9, "cos_sim {s} vs 1-2h = {expected_s}");
    }
}
