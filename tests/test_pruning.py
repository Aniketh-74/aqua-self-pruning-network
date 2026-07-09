import numpy as np
from engine.engine import Tensor
from nn.nn import Linear, ReLU, Sequential
from train.optim import AdamPrunable
from prune.prune import Pruner, count_sparsity, cubic_schedule


def test_masked_weight_stays_zero_under_momentum():
    W = Tensor(np.array([[0.5, 0.3]]))
    b = Tensor(np.zeros(2))
    mask = np.array([[1.0, 1.0]])
    opt = AdamPrunable([W, b], [mask, np.ones(2)], lr=0.01)
    for _ in range(15):
        W.grad = np.random.randn(*W.data.shape) * 0.1
        b.grad = np.zeros(2)
        opt.step()
    mask[0, 0] = 0.0
    W.data *= mask
    for _ in range(30):
        W.grad = np.zeros_like(W.data)
        b.grad = np.zeros(2)
        opt.step()
    assert W.data[0, 0] == 0.0


def test_masked_weight_gets_zero_gradient():
    np.random.seed(1)
    layer = Linear(3, 2)
    layer.mask[0, 0] = 0.0
    x = Tensor(np.random.randn(4, 3))
    layer(x).sum().backward()
    assert abs(layer.W.grad[0, 0]) < 1e-12


def test_revival_resets_both_moments_no_explosion():
    W = Tensor(np.array([[1.0, 2.0]]))
    b = Tensor(np.zeros(2))
    mask = np.array([[1.0, 1.0]])
    opt = AdamPrunable([W, b], [mask, np.ones(2)], lr=0.01)
    for _ in range(10):
        W.grad = np.array([[0.5, 0.5]]); b.grad = np.zeros(2)
        opt.step()
    mask[0, 0] = 0.0
    W.data *= mask
    revived = np.array([[True, False]])
    opt.reset_moments(0, revived)
    assert opt.m[0][0, 0] == 0.0 and opt.v[0][0, 0] == 0.0
    mask[0, 0] = 1.0
    W.data[0, 0] = 0.01
    W.grad = np.array([[0.5, 0.5]]); b.grad = np.zeros(2)
    opt.step()
    assert abs(W.data[0, 0]) < 1.0


def test_honest_zeros_after_pruning():
    np.random.seed(0)
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 3))
    params = model.parameters()
    masks = []
    for l in model.layers:
        if isinstance(l, Linear):
            masks.append(l.mask); masks.append(np.ones_like(l.b.data))
    opt = AdamPrunable(params, masks, lr=0.01)
    pruner = Pruner(model, opt, criterion='accum_saliency', s_final=0.8,
                    total_steps=200, prune_start=0, prune_every=10)
    X = np.random.randn(32, 4)
    y = np.random.randint(0, 3, 32)
    for step in range(200):
        loss = model(Tensor(X)).softmax_cross_entropy(y)
        opt.zero_grad(); loss.backward()
        pruner.accumulate(); opt.step(); pruner.maybe_prune(step)
    max_pruned = 0.0
    for l in model.linear_layers():
        pr = l.W.data[l.mask == 0]
        if pr.size:
            max_pruned = max(max_pruned, np.abs(pr).max())
    assert max_pruned == 0.0


def test_cubic_schedule_monotonic():
    vals = [cubic_schedule(s, 100, 0.0, 0.9, 0, 75) for s in range(100)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-9
    assert abs(vals[-1] - 0.9) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"[PASS] {name}")
    print("All pruning tests passed.")