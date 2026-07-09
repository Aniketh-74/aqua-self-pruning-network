# DESIGN

This document answers the four required design questions and records the key decisions and the one non-obvious bug this challenge is built around.

---

## 1. Derive the importance criterion and explain why it approximates the loss change from removing a connection

Removing a weight means setting it to zero. If a weight currently has value `w`, removing it is a perturbation `δw = -w`. Expand the loss around the current weights with a Taylor series:

```
L(w + δw) ≈ L(w) + g · δw + (1/2) · δw · H · δw
```

where `g = ∂L/∂w` is the gradient and `H` is the Hessian. Keeping the first-order term, the change in loss from removing the weight is:

```
ΔL ≈ g · δw = g · (-w) = -w · g
```

We care about how much removing the weight _disturbs_ the loss, in either direction, so we take the magnitude:

```
saliency = |w · g|
```

A connection is important if removing it would change the loss a lot. `|w · g|` is the first-order estimate of exactly that quantity. Magnitude pruning (`|w|`) is the special case that drops the gradient term — it assumes every weight sits on an equally steep part of the loss surface, which is why it can keep large-but-useless weights and cut small-but-load-bearing ones.

**The subtlety that matters in practice.** During SGD the gradient `g` is different for every minibatch — it is a one-sample estimate. If saliency is computed from whatever gradient happens to be in the buffer at prune time, the score is `|w · (one noisy gradient)|` and pruning decisions are partly driven by gradient noise. Measured directly, the single-batch saliency correlates only ~0.85 with the epoch-accumulated saliency, i.e. ~15% of the ranking is noise. The fix is `accum_saliency`: accumulate `|w · g|` across all batches in the epoch and prune on the average. This is the same reasoning behind the moving-average saliency used in the pruning literature (e.g. Molchanov et al., 2019).

**What the evidence shows.** On two-spirals, accumulating removes the noise but magnitude still wins at high sparsity (0.997 at 90/95% vs 0.993/0.991 for accumulated saliency). This is expected: two-spirals is small and heavily over-parameterized, so almost any subset of weights solves it and the stable signal (magnitude) beats the theoretically-smarter but higher-variance signal. Saliency is expected to overtake magnitude on harder tasks with a tighter parameter budget, where _which_ weights survive actually matters — that is the falsifiable prediction this result implies.

---

## 2. What does the engine compute as "the gradient of a masked weight," and why is that the right choice?

Masking is implemented in the forward pass as `masked_W = W ⊙ mask`, where `mask` is a constant 0/1 array. Because the mask enters the computation graph as a multiplicative constant, the chain rule gives:

```
∂L/∂W = (∂L/∂masked_W) ⊙ mask
```

So a masked weight (mask = 0) receives a gradient of exactly **zero**. That is the right choice: a pruned connection does not participate in the forward pass, so it has no effect on the loss, so its gradient should be zero. The engine does not special-case this — it falls out of representing the mask as part of the graph, which is what keeps the autodiff honest while the effective architecture changes.

Zero gradient alone is **not sufficient**, which is the core correctness requirement of this challenge. Even with zero gradient, Adam keeps per-parameter momentum buffers `m` and `v` that still hold residual values from before the weight was pruned. On subsequent steps:

```
m = β1 · m + (1-β1) · 0 = β1 · m      (decays, but non-zero)
update = lr · m_hat / (sqrt(v_hat) + ε)   (non-zero)
```

so a "pruned" weight silently drifts off zero and is no longer really pruned. This is handled in `AdamPrunable`:

- **After every step, re-zero pruned weights:** `p.data *= mask`. Momentum can no longer drift a pruned weight; it stays an exact zero. This is verified by `test_masked_weight_stays_zero_under_momentum` and by `train_prune.py`, which asserts the maximum absolute value among pruned weights is exactly 0.
- **On revival (regrowth), reset the moments:** a connection that comes back has stale `m` and `v` from many steps ago. `reset_moments` zeros **both** for the revived entries. Resetting both together is important: if only `v` were reset to zero while `m` stayed non-zero, the next update would be `lr · m_hat / (sqrt(0) + ε)` ≈ `m_hat · 1e8`, an enormous destructive step. `test_revival_resets_both_moments_no_explosion` guards this.

A real bug was caught during development and is worth recording: pruned weights were leaking to ~5e-6 despite the re-zeroing. The cause was that the pruner rebound `layer.mask = new_mask` (a new array) while the optimizer still referenced the original mask array from setup, so the optimizer re-zeroed with a stale mask. The fix is to update the mask **in place** (`layer.mask[:] = new_mask`) so the optimizer and the layer always share one array. After the fix, the maximum pruned weight is exactly 0.

---

## 3. Where does the autodiff engine bottleneck, and how would you optimize it?

- **Python-level graph traversal.** `backward()` builds a topological order with a recursive DFS and then calls a Python closure per node. For a graph with many small ops this is interpreter-bound — the per-node Python overhead dominates the actual NumPy math. Optimization: fuse composite ops (a single `linear` op instead of separate matmul + add), reducing node count; or flatten the graph into an iterative traversal to avoid recursion-depth limits and call overhead.
- **Recomputing masked matmuls densely.** The training forward pass computes `W ⊙ mask` and a full dense matmul even at 90% sparsity, so training does not get faster as the network is pruned. That is acceptable for correctness during training but wasteful. Optimization: a sparse representation (CSR) for inference once the mask stabilizes — done in `prune/cost.py` for measurement.
- **New closures per forward.** Every op allocates a new `out` tensor and a new `_backward` closure each forward pass. In a hot training loop this is allocation pressure. Optimization: preallocate gradient buffers and reuse graph structure across steps when the shapes are fixed.
- **No vectorized gradient checking.** Gradient checks perturb one scalar at a time — fine for tests, `O(params)` forward passes. Not a training bottleneck, but would be the bottleneck of the test suite if scaled; batching perturbations would help.

The single highest-impact change for this workload is reducing Python-per-node overhead (op fusion / iterative backward), because the matrices here are small enough that interpreter overhead, not FLOPs, is the limit.

---

## 4. How would you serve a self-pruned model in a real multi-tenant inference service at scale?

**Represent the sparsity honestly.** Once the mask is final, store each layer as a sparse matrix (CSR or a structured/2:4 pattern), not a dense matrix full of zeros. Dense-times-zero is not a speedup; the storage and compute must actually skip the zeros. Structured sparsity is preferable in production because hardware (sparse tensor cores, blocked kernels) can exploit it, whereas unstructured sparsity often needs large sizes before it beats dense BLAS — the same effect observed at small scale in this project.

**Separate the artifact from the runtime.** Ship an immutable, versioned model artifact (weights + mask + metadata: sparsity, eval numbers, seed, training commit). The serving layer loads artifacts by version; promotion/rollback is repointing a version alias, never editing weights in place. This gives reproducibility and instant rollback.

**Multi-tenant isolation and efficiency.** Tenants may run different pruned variants (different sparsity/accuracy trade-offs). Serve them behind one gateway that routes by tenant to the right artifact version, with per-tenant quotas and request isolation so one tenant cannot exhaust another's capacity. Batch requests across tenants where the same variant is targeted, to keep the accelerator well utilized.

**Measure what you claim.** Emit per-request traces (latency p50/p95/p99, tokens/rows, active-parameter path, cost) and validate that the sparse serving path matches the dense reference within tolerance (here ~1e-14) before trusting it. A pruned model that is cheaper on paper but not measured in production is not actually cheaper — the same standard of evidence used in Part 4 applies to the live service.

**Fit with the broader platform.** The importance/eval discipline generalizes: the same way pruning decisions are made on measured saliency and validated across seeds, model-variant promotion in production should be gated on measured eval and cost, not on the assumption that a smaller model is good enough.
