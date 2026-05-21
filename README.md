> # ⚠️ DO NOT USE — UNVERIFIED — UNSAFE ⚠️
>
> This software is **unverified and unsafe for any production use**.
> It is published publicly only for transparency, third-party audit,
> and reproducibility. Treat every commit as guilty until proven
> innocent.
>
> By using this code you accept:
> - **No warranty** of any kind, express or implied.
> - **No fitness** for any particular purpose.
> - **No guarantee** of correctness, safety, or freedom from defects.
> - **Zero liability** on the maintainer for any damages — data loss,
>   security compromise, financial loss, or any consequential damages.
>
> The code is under active engineering development per the
> [Adversarial Validation Protocol v2](https://github.com/thepictishbeast/PlausiDen-AVP-Doctrine/blob/main/AVP2_PROTOCOL.md).
> Every commit's default verdict is **STILL BROKEN**. AVP-2 requires
> a minimum of 36 verification passes before a `SHIP-DECISION:`
> annotation may be considered. **No commit in this repository has
> reached `SHIP-DECISION:` status.**

# PlausiDen-Stack

A heterogeneous compositional neural architecture on the **HDC / VSA
substrate** (Hyperdimensional Computing / Vector Symbolic Architecture).

The basic unit is a *Stack* — a tightly-coupled bundle of mixed
computation modes (dense binding, FFT-style mixing à la FNet, gated
routing, multi-perspective aggregation) all operating on a shared
hypervector. Stacks compose recursively into Stack-of-Stacks structures
with per-level entropy gradients, and the whole system is designed for
self-modification.

This is **not** a transformer. This is **not** a Mixture of Experts.
Either could be wrapped behind the same Stack abstraction; the
architecture is substrate-first, family-agnostic.

## Why this exists

Current LLM architecture is one local optimum, not THE solution. The
PlausiDen Stack architecture targets four structural weaknesses
directly:

1. **HDC substrate**: O(n) operations instead of O(n²) softmax attention.
2. **Heterogeneous composition**: cortical-column-like mixed compute, not
   committee-style independent experts.
3. **Designed specialization via entropy gradient**: forced diversity that
   self-learning amplifies rather than smooths away.
4. **Self-modification as first-class capability**: the system reshapes
   its own stack composition over time.

## Status — pre-1.0, AVP-2 in flight, NOT production-ready

| Crate | Purpose |
|---|---|
| `crates/plausiden-hdc` | HDC primitives: 10K-dim bipolar hypervectors + bind / bundle / unbind / permute / cosine-sim / Hamming distance. Property-tested. |
| `crates/plausiden-stack` | Stack + StackOfStacks impls (Phase 1+) |

## Plan

See [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — the full architecture
spec (Stack primitive, recursion, entropy gradient, self-modification,
build phases, references).

## Related repos

- [PlausiDen-AI](https://github.com/thepictishbeast/PlausiDen-AI) — LFI
  (the operational AI); the original `lfi_vsa_core` crate inspired this
  one but Stack now ships its own canonical HDC primitives.
- [PlausiDen-GraphNet](https://github.com/thepictishbeast/PlausiDen-GraphNet)
  — live REPL + visualisation environment that consumes both LFI and
  Stack via adapters.

## License

[FSL-1.1-MIT](./LICENSE).
