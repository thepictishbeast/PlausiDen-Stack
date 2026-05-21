//! Benchmarks for the HDC primitives at the canonical D=10,000.

#![allow(missing_docs, clippy::expect_used, clippy::unwrap_used)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use plausiden_hdc::{bind, bundle, cos_sim, hamming, permute, unbind, Hypervector};

fn bench_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_ops");
    for &dim in &[1_000usize, 10_000] {
        let a = Hypervector::random_seeded(dim, 1);
        let b = Hypervector::random_seeded(dim, 2);
        group.bench_with_input(BenchmarkId::new("bind", dim), &dim, |bch, _| {
            bch.iter(|| bind(&a, &b).expect("ok"));
        });
        group.bench_with_input(BenchmarkId::new("unbind", dim), &dim, |bch, _| {
            bch.iter(|| unbind(&a, &b).expect("ok"));
        });
        group.bench_with_input(BenchmarkId::new("bundle_2", dim), &dim, |bch, _| {
            bch.iter(|| bundle(&[&a, &b]).expect("ok"));
        });
        group.bench_with_input(BenchmarkId::new("permute_1", dim), &dim, |bch, _| {
            bch.iter(|| permute(&a, 1));
        });
        group.bench_with_input(BenchmarkId::new("cos_sim", dim), &dim, |bch, _| {
            bch.iter(|| cos_sim(&a, &b).expect("ok"));
        });
        group.bench_with_input(BenchmarkId::new("hamming", dim), &dim, |bch, _| {
            bch.iter(|| hamming(&a, &b).expect("ok"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ops);
criterion_main!(benches);
