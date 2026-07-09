# AQUA - The Self-Pruning Network

A small neural-network library built from scratch in pure Python + NumPy: a reverse-mode autodiff engine, an MLP with an Adam optimizer written from scratch, and a network that prunes its own connections during training under a hard sparsity budget, with honest evidence that the pruned model is genuinely cheaper.

No deep-learning frameworks are used anywhere in the core (no PyTorch, TensorFlow, JAX, Keras, or any autodiff / pruning / NN library). NumPy is used throughout; Matplotlib is used only to render plots. Scikit-learn is **not** used at all - the dataset is a synthetic two-spirals generator written from scratch, so the project runs fully offline.

## Requirements

```
pip install -r requirements.txt
```

Python 3.10+ (developed on 3.12). Dependencies: `numpy`, `matplotlib`.

## Repository layout

```
engine/     autodiff engine (Tensor, ops, reverse-mode backward)
nn/         Linear / ReLU / Sequential layers with prunable masks
prune/      pruner (importance criteria + cubic schedule) and cost measurement
train/      optimizers, dataset, and the training / experiment entry points
tests/      gradient-check tests and masked-weight correctness tests
```

All commands are run from the repository root using module syntax (`python -m ...`), because the code is organized as a package.

## Reproduce everything

Run the gradient-check tests (Part 1):

```
python -m tests.test_engine
```

Run the masked-weight / pruning correctness tests (Part 1 + Part 3):

```
python -m tests.test_pruning
```

Reproduce Part 2 (dense MLP training on two-spirals, saves `learning_curve.png`):

```
python -m train.train
```

Reproduce Part 3 (self-pruning to 90% sparsity, verifies honest zeros):

```
python -m train.train_prune
```

Reproduce Part 4 (sparsity sweep, saves `pareto.png`, `results.json`, `claim.json`):

```
python -m train.experiment
```

## Reproducibility

All randomness is seeded. Model initialization is seeded per run; the Part 4 sweep averages over 5 fixed seeds (0–4). Cloning the repository and running the commands above reproduces the committed numbers in `results.json` and `claim.json` and the committed plots.

## What the code does

**Part 1 - autodiff engine (`engine/engine.py`).** A `Tensor` wraps a NumPy array, tracks its parents in a computation graph, and stores a gradient. It implements element-wise add / sub / mul / div, matmul, sum / mean, ReLU / sigmoid / tanh, a power op, and a numerically stable combined softmax + cross-entropy. `backward()` builds a reverse topological order and accumulates gradients, with broadcasting handled correctly (e.g. `(N, D) + (D,)`). Every op is verified against finite-difference numerical gradients in `tests/test_engine.py`.

**Part 2 - training (`nn/nn.py`, `train/optim.py`, `train/train.py`).** `Linear` layers use He initialization (variance `2 / fan_in`, matched to ReLU) so training is stable with no NaNs. `Adam` is implemented from scratch (first/second moment estimates with bias correction). The model trains on two-spirals - a non-linearly-separable task a linear model cannot solve - and reaches ~99% test accuracy.

**Part 3 - self-pruning (`prune/prune.py`, `train/train_prune.py`).** During training the network progressively removes its least-important connections following a cubic sparsity schedule up to a target (default 90%). Three importance criteria are provided: `magnitude` (|w|), `saliency` (|w·g| from the current batch), and `accum_saliency` (|w·g| accumulated over the epoch to reduce gradient noise). Masking is honest: a pruned weight is multiplied out of the forward pass, receives zero gradient, and is re-zeroed after every optimizer step so momentum cannot drift it. `train_prune.py` verifies that the maximum absolute value among pruned weights is exactly zero.

**Part 4 - evidence (`prune/cost.py`, `train/experiment.py`).** A sparsity/accuracy Pareto sweep over {0, 50, 75, 90, 95}% for all three criteria, averaged across 5 seeds. Cost is measured with a sparse-aware forward path (only non-zero weights participate), which agrees with the dense forward to ~1e-14, alongside FLOP-reduction and active-parameter counts. Results and a falsifiable claim are written to `results.json` and `claim.json`.

## result

On two-spirals, magnitude pruning is the strongest criterion at high sparsity (0.997 accuracy at 90% and 95%). Single-batch saliency underperforms it because a one-batch gradient is a noisy importance estimate; accumulating saliency over the epoch recovers part of that gap (e.g. at 95% sparsity, 0.988 → 0.991), but does not overtake magnitude on this easy, over-parameterized task. Differences are reported with per-seed standard deviations and a within-noise check, and the numbers are committed rather than asserted. See `DESIGN.md` for the reasoning and for where saliency would be expected to win.

## Note on cost

FLOP reduction tracks sparsity directly (0.90 at 90% sparsity). Wall-clock speedup on this small network is **not** reported as a win: on a 64-unit MLP, a Python-loop sparse forward cannot beat NumPy's optimized dense matmul until the matrices are much larger. The sparse path is included to measure real work removed and to prove correctness against the dense path, not to claim a wall-clock speedup that does not exist at this scale.
