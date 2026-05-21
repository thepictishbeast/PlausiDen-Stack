# Stack Architecture

Heterogeneous compositional architecture built on HDC / VSA primitives. The
basic unit ("Stack") is a tightly-coupled bundle of mixed computation modes —
dense binding, FFT-style mixing, gated routing, multi-perspective aggregation —
all expressed in HDC operations on a shared hypervector. Stacks compose
recursively into stack-of-stacks structures with per-level entropy gradients,
and the whole system is designed for self-modification.

This doc captures the architecture vision from 2026-05-17 owner direction
and the formal grounding in published HDC / VSA / spectral-NN research.

## Status

| Section | State |
|---|---|
| Architecture vision | Specified |
| Stack primitive math | Specified (HDC operations canonical) |
| Stack-of-stacks recursion | Specified; collapse-resistance argued |
| Self-modification mechanism | Open — meta-learning loop unspecified |
| Implementation | Not started; build order in §9 |

## 1. Vision

Current LLM architecture is one local optimum, not THE solution. The dominant
transformer paradigm has known structural weaknesses:

- **Attention is O(n²)** — long context is expensive even when it nominally fits.
- **Quadratic attention is dense and undifferentiated** — every token attends to
  every other, with quality of attention only weakly correlated with task
  relevance.
- **Architectures are monolithic** — a transformer is the same operation at
  every layer; "specialization" is emergent rather than designed.
- **Self-modification is absent** — fine-tuning aside, a trained model cannot
  reshape its own architecture in response to encountered tasks.

The PlausiDen Stack architecture targets all four directly:

1. **HDC substrate** — replace O(n²) softmax attention with O(n) HDC operations.
2. **Heterogeneous composition** — each Stack contains multiple operation
   modes in parallel, tightly coupled on a shared hypervector. Not Mixture
   of Experts (independent committee members); think cortical columns or
   octopus / tentacles — heterogeneous compute units feeding a shared
   integration substrate.
3. **Designed specialization via entropy gradient** — across a stack-of-stacks
   array, individual stacks run at different entropy levels (from purely
   deterministic to purely random). Forced diversity that self-learning amplifies
   rather than smooths away.
4. **Self-modification as first-class capability** — the system reshapes its own
   stack composition over time. Initial uniform stacks differentiate into
   specialized stacks through training dynamics, then can be explicitly
   restructured by a meta-controller.

## 2. Why HDC / VSA

HDC is the substrate for this whole architecture because four properties of HDC
operations make compositional and recursive architectures cheap and well-behaved:

### 2.1 Compositional algebra

HDC operations are associative and compose without algebraic incompatibility.

| Operation | Symbol | Type | Result |
|---|---|---|---|
| **Binding** | `⊗` | mul-like (XOR / circular conv / componentwise) | new vector with operands "attached" |
| **Bundling** | `+` | superposition (additive) | new vector containing both operands |
| **Permutation** | `Π` | unitary rotation | sequence position / role tagging |
| **Unbinding** | `⊗⁻¹` | inverse binding | extract operand given other operand |

`Π(bind(bundle(a, b), KEY))` is a meaningful primitive operation. You can
recursively stack HDC operations without algebraic gotchas. Transformer
attention does not compose like this — `attention(attention(attention(x)))`
is structurally suspect even if it's syntactically valid in code.

### 2.2 Holographic / distributed representation

Vectors are typically 10,000-dim bipolar `{-1, +1}` (or binary). Information
is distributed across all dimensions rather than localized. Damage 10% of
dimensions → ~10% fidelity loss, not 100%.

Consequence for Stack: "dynamic dimension shutdown" is meaningful at the
substrate. Quantization isn't catastrophic loss — it's graceful degradation.
A Stack can run at full precision when compute is available and at quantized
precision when not, without architectural change.

### 2.3 Interpretable via decomposition

A bundle `c = a ⊗ KEY_A + b ⊗ KEY_B + ...` can be queried by unbinding:
`c ⊗ KEY_A⁻¹ ≈ a` (with noise). Components can literally be extracted.

This is the wave/FFT intuition formalized correctly: thoughts as bundled
components, retrievable by binding with the matching key. The underlying
operation doesn't need to be linear (the binding operation distributes over
bundling), so unlike literal FFT on raw signals, HDC unbinding works through
nonlinear post-bundle processing.

### 2.4 FFT-native binding implementation

Plate's **Holographic Reduced Representations** use circular convolution as
the binding operation. Circular convolution = FFT of pointwise multiplication
of FFTs. So the wave / spectral framing is literally one of the canonical
HDC binding implementations, and one of the most efficient.

## 3. The Stack primitive

A Stack is a heterogeneous bundle of operations acting on a shared HDC vector.

```
Stack input:  hypervector v ∈ ℝ^D  (typically D = 10,000)
Stack output: hypervector v' ∈ ℝ^D

Internal operations (a Stack contains multiple in parallel):
  - dense_bind(v, W)        : HDC dense layer (multi-key binding)
  - fft_bind(v, K)          : HRR circular-convolution binding (FFT-based)
  - gated_route(v, R)       : MoE-like routing within the same vector
  - aggregate(v, perspectives) : intra-stack multi-perspective combination
  - identity                : pass-through (skip-connection equivalent)

Stack composition: v' = Π(bundle(op_i(v) for op_i in active_ops))
  where active_ops is decided by a gating signal (learned + entropy-perturbed)
```

Key distinction from MoE:

| MoE | Stack |
|---|---|
| Each expert is an autonomous sub-model | Each operation is a substructure |
| Router selects 1-of-N experts | All active ops contribute, weighted |
| Outputs from different experts may diverge | Outputs combine into ONE shared vector |
| Loose coupling (committee) | Tight coupling (cortical column) |

Stack is closer to multi-head attention than to MoE — each "operation" is like
a "head," all working on the same input, all contributing to a single output.
The novelty over multi-head attention is that the operations are
*architecturally heterogeneous*, not just different parameter slices of the same
operation type.

## 4. Stack-of-stacks recursion

The full architecture composes stacks fractally:

```
Level 0: Stack         (basic unit)
Level 1: Stack-of-Stacks  (composes N Level-0 stacks)
Level 2: Stack-of-Stack-of-Stacks  (composes N Level-1 structures)
...
```

The composition rule is identical at every level: each level's "stack" is
itself a bundle of substructures. This is fractal architecture in the strict
sense — same operation pattern at every scale.

### 4.1 Collapse-resistance

Naive recursive composition has a known failure mode: representational collapse.
If every level applies the same operations to the output of the level below, the
system converges to a fixed point as a Markov chain and the upper levels add
nothing.

The Stack architecture avoids collapse via two mechanisms:

**1. Per-level information differentiation.** Each level either receives
distinct inputs (hierarchical feature extraction, standard practice in deep
nets) OR applies a distinct operation composition. Stack composition is
selected per-level so the recursion doesn't reduce to a fixed point.

**2. Entropy gradient across stacks at a given level.** Within a level,
individual stacks operate at different entropy levels. Some stacks are
purely deterministic (entropy = 0); others are purely random (entropy = 1).
Most are in between. This forced diversity prevents the level's
substructures from converging.

This mechanism is loosely related to:

- **Reservoir computing** (echo state networks, liquid state machines) —
  random fixed network + trained readout; the random part projects inputs
  into a high-D useful space.
- **Free energy principle / active inference** (Friston) — agents balance
  entropy reduction (exploitation) and entropy increase (exploration).
- **Diversity-regularized ensemble methods** — explicit penalty terms on
  ensemble members for being too similar.

The novel piece is using entropy as a *structural gradient across stacks*,
not just as a hyperparameter of one model.

### 4.2 Information bouncing

Within a stack-of-stacks, intermediate hypervectors are routed (or "bounced")
between substructures via the binding/unbinding operations. A stack can write
to a shared key, another stack reads from that key by unbinding, applies
its computation, and rebinds to a different key. The shared HDC vector
becomes a blackboard between substructures.

Concretely:

```
shared = bundle(
  KEY_PERCEPTION ⊗ perception_stack(v),
  KEY_MEMORY     ⊗ memory_stack(v),
  KEY_REASONING  ⊗ reasoning_stack(v),
  KEY_INTEGRATION ⊗ ZERO,
)

# Reasoning stack reads perception + memory by unbinding, integrates:
integration = reasoning_stack(
  shared ⊗ KEY_PERCEPTION⁻¹,
  shared ⊗ KEY_MEMORY⁻¹,
)

# Bind back into shared:
shared = shared + (KEY_INTEGRATION ⊗ integration) - (KEY_INTEGRATION ⊗ ZERO)
```

This pattern lets substructures communicate without explicit message passing;
they read and write to a shared HDC vector with role-keyed binding.

## 5. Operation modes within a Stack

The Stack primitive contains a heterogeneous set of operation modes. Each
mode is implemented in HDC primitives. Stacks may include any subset.

### 5.1 Dense binding (transformer-like)

`v' = Π(v ⊗ W)` where `W` is a learned weight hypervector.
Captures transformer "dense linear" semantics in HDC. O(D) cost.

### 5.2 HRR / FFT binding (spectral)

`v' = ifft(fft(v) * fft(K))` — circular convolution via FFT, Plate-style HRR.
Captures FNet-like frequency mixing. O(D log D) cost.

Where dense binding loses spectral structure, HRR preserves it. Stacks that
need to reason about periodicity, harmonics, or spectral features (audio, EEG,
oscillatory time-series, frequency-domain features of any kind) include HRR.

### 5.3 Gated routing (MoE-like, tightly coupled)

`v' = Σᵢ gᵢ(v) · opᵢ(v)` where `gᵢ(v)` is a learned gating function and `opᵢ`
are the operations in the Stack.

Unlike MoE, routing is *soft* (all operations contribute, weighted by gates)
and the operations share the same input vector + contribute to the same output
vector. Closer to a soft-attention mechanism over operation choice than to
expert selection.

### 5.4 Multi-perspective aggregation (debate-like, intra-stack)

`v' = aggregator(op_dense(v), op_fft(v), op_random(v), ...)`

The Stack applies each operation in parallel, then combines outputs through
a learned aggregator. Conceptually similar to multi-head attention's output
projection, but the "heads" are architecturally different.

This is the "1000 eyes on the output" intuition at the substack level.

### 5.5 Identity / skip

`v' = v`. Standard residual-connection equivalent. Lets the Stack pass
information through unmodified when no transformation is needed.

### 5.6 Wavelet / multi-resolution (future)

For signals with non-stationary frequency content (where FFT assumptions break),
wavelet binding offers multi-resolution decomposition. Not in v1 but kept on
the roadmap.

## 6. Entropy gradient across stacks

At a given level of the recursion, individual stacks operate at different
entropy levels. Concretely, each stack has a temperature parameter `τ ∈ [0, 1]`
controlling the stochasticity of:

- Gating decisions (which operations to activate)
- Routing decisions (where to bounce information)
- Initial conditions (noise added to the input)

`τ = 0` stacks are purely deterministic (always the same operation, no noise).
`τ = 1` stacks are pure exploration (random gates, random routing, max noise).
A typical level might have stacks spread across the full range.

The entropy gradient is set initially as a structural choice. Self-learning then
amplifies the differentiation: low-τ stacks specialize on stable patterns;
high-τ stacks specialize on novelty detection / exploration. Even if 1000
stacks start with identical weights, they differentiate naturally because they
see different effective input distributions (deterministic stacks see clean
input; stochastic stacks see noisy input).

## 7. Self-modification

The Stack architecture is designed to be self-modifying. A meta-controller
observes performance per-stack-per-operation and can:

1. **Reweight operations within a Stack** — increase gating toward operations
   that contribute to correct outputs, decrease toward those that don't.
2. **Add operations to a Stack** — instantiate a new operation mode with
   small initialization; meta-controller decides which mode to add based on
   error patterns.
3. **Remove operations from a Stack** — prune operations that consistently
   contribute below noise floor.
4. **Spawn new Stacks** — replicate a high-performing Stack at a different
   entropy level.
5. **Merge Stacks** — combine two Stacks whose function has converged.

The meta-controller itself is unspecified in v1; candidates:

- **Gradient-based meta-learning** (MAML-style) — meta-loss over architectural
  decisions, gradient through architecture choices via Gumbel-softmax
  relaxation.
- **Evolutionary search** — population of architectures, select for fitness.
- **Reinforcement learning** — architectural decisions as actions, task reward
  as signal.
- **Free energy minimization** — choose architecture changes that minimize
  expected free energy under a generative model.

Open research question; pick one when prototype demands force the choice.

## 8. Relationship to published research

The Stack architecture is novel in combination but each constituent piece has
research lineage:

| Stack feature | Closest published lineage |
|---|---|
| HDC substrate | Kanerva (Sparse Distributed Memory), Plate (HRR), Gayler (VSA), Neubert (HD) |
| FFT binding | Plate 1995 (HRR), FNet 2021 (Lee-Thorp et al.) |
| Heterogeneous ops | Mixture of Depths (Raposo et al. 2024), MoE (Switch / DeepSeek) |
| Recursive composition | Hierarchical Recurrent Networks; tensor networks (MERA) |
| Entropy gradient | Reservoir Computing; Free Energy Principle (Friston) |
| Self-modification | Neural Architecture Search; Meta-learning (MAML); Differentiable Architecture Search |
| Information bouncing | Global Workspace Theory (Baars); HDC blackboard binding |
| Multi-perspective intra-stack | Multi-Agent Debate (Du et al. 2023); ensemble methods |

The PlausiDen contribution is the unified HDC-substrate composition pattern,
not any individual mechanism.

## 9. Build order

Smallest-viable-thing first. Each level either works on top of the previous
or doesn't, and you find out cheaply.

### Phase 0: HDC foundation (already exists in `lfi_vsa_core`)

Bipolar 10,000-dim hypervectors + bind/bundle/permute/unbind primitives.
Rust crate. Confirm baseline + benchmarks.

### Phase 1: Single Stack with three operation modes

- `Stack::new(D, operations: Vec<Operation>)` where `Operation` is one of
  `Dense`, `HRRBind`, `Identity`.
- Forward: parallel application + bundling.
- Train on a toy task: associative recall (memorize `a → b` pairs, recall
  `b` given `a`). Baseline = plain HDC associative memory; Stack should match
  or exceed.

**Success criterion:** Single Stack with three operations learns the toy task
without instability. ~1-2 weeks.

### Phase 2: Gated Stack (soft routing)

Add `GatedRoute` operation mode. Learned gates select operation weights per
input. Train on a slightly harder task: classification with multi-modal input
features (e.g., text + categorical).

**Success criterion:** Gating selects different operations for different input
classes. Verifiable by inspecting gate activations. ~1-2 weeks.

### Phase 3: Stack-of-Stacks (level-1 recursion)

Compose N Level-0 Stacks into a Level-1 structure with shared blackboard
HDC vector. All Stacks read + write to the blackboard via key-binding.

Train on a task requiring multi-stage reasoning (e.g., compositional
question answering on a small synthetic dataset).

**Success criterion:** Level-1 structure outperforms a single Stack with
the same total operation count. ~2-3 weeks.

### Phase 4: Entropy gradient

Add per-Stack `τ` parameter. Initialize a Level-1 structure with N stacks
spanning `τ ∈ [0, 1]`. Train.

**Success criterion:** Stacks at different `τ` levels measurably specialize
on different input distributions. Verifiable by entropy-of-gate-distribution
metrics. ~1-2 weeks.

### Phase 5: Self-modification (minimal)

Implement operation-reweighting (Phase-5a) and operation-add/remove (Phase-5b)
under a simple performance-monitoring meta-controller. RL or gradient-based
meta-learning is Phase 6+.

**Success criterion:** Architecture converges to a stable composition that
differs from initialization, and the converged composition outperforms the
initial. ~2-4 weeks.

### Phase 6+: Open research

- Full meta-learning loop (RL or MAML)
- Wavelet binding
- Tensor-network operations (for FFT-resistant regimes)
- Stack-of-stack-of-stacks (level-2 recursion)
- Distributed training across nodes

## 10. Open questions

These are real unknowns. Tracking them so we don't pretend they're solved:

1. **Stability under self-modification.** A self-modifying architecture can
   in principle reshape itself into a non-functional configuration.
   What invariants must be preserved? Constraint-based meta-controllers?
2. **Base case for recursion.** Level-0 Stack is the obvious base case, but
   what's the right *depth* for the recursion? Does Level-5 add anything
   over Level-3?
3. **Entropy schedule.** Is entropy fixed per-stack, or does it anneal over
   training (start high, end low)? Both have precedent.
4. **Meta-controller architecture.** RL vs. evolutionary vs. gradient-based
   meta-learning. Pick when prototype forces choice; don't pre-decide.
5. **HDLM / LLM substitution at the leaves.** Does the leaf-level operation
   set need to include "tiny LLM" or can pure HDC ops suffice for language
   tasks? Open. The supersociety baseline favors HDC-only.

## 11. References

Core HDC / VSA:
- Kanerva, P. (1988). *Sparse Distributed Memory.* MIT Press.
- Plate, T. (1995). *Holographic Reduced Representations.* IEEE TNN.
- Gayler, R. (2003). *Vector Symbolic Architectures Answer Jackendoff's Challenges.*
- Kanerva, P. (2009). *Hyperdimensional Computing.*

Spectral neural networks:
- Lee-Thorp et al. (2021). *FNet: Mixing Tokens with Fourier Transforms.*
- Gu, A., Goel, K., Ré, C. (2021). *Efficiently Modeling Long Sequences with Structured State Spaces.* (S4)
- Gu, A., Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*

Heterogeneous / mixture architectures:
- Fedus, W. et al. (2021). *Switch Transformer.*
- Raposo et al. (2024). *Mixture of Depths.*
- DeepSeek-V3 technical report (2024).

Self-modification / meta-learning:
- Finn, C. et al. (2017). *Model-Agnostic Meta-Learning (MAML).*
- Liu, H. et al. (2019). *DARTS: Differentiable Architecture Search.*

Diversity / debate:
- Du, Y. et al. (2023). *Improving Factuality and Reasoning in Language Models through Multiagent Debate.*
- Wang, X. et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning.*

Free energy / active inference:
- Friston, K. (2010). *The free-energy principle: a unified brain theory?*
