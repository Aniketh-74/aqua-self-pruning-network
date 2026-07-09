import numpy as np
from engine.engine import Tensor


def _grad_check(build_fn, tensors, eps=1e-5, tol=1e-4):
    for t in tensors:
        t.grad = np.zeros_like(t.data)
    loss = build_fn(*tensors)
    loss.backward()
    for t in tensors:
        for flat_i in range(min(5, t.data.size)):
            coord = np.unravel_index(
                np.random.randint(t.data.size), t.data.shape)
            analytical = t.grad[coord]
            old = t.data[coord]
            t.data[coord] = old + eps
            lp = build_fn(*tensors).data
            t.data[coord] = old - eps
            lm = build_fn(*tensors).data
            t.data[coord] = old
            numerical = (lp - lm) / (2 * eps)
            if abs(analytical) < 1e-7 and abs(numerical) < 1e-7:
                continue
            assert abs(analytical - numerical) < tol, \
                f"grad mismatch {coord}: {analytical} vs {numerical}"


def test_matmul():
    np.random.seed(0)
    _grad_check(lambda A, B: A.matmul(B).sum(),
                [Tensor(np.random.randn(3, 4)), Tensor(np.random.randn(4, 5))])


def test_matmul_relu():
    np.random.seed(1)
    _grad_check(lambda A, B: A.matmul(B).relu().sum(),
                [Tensor(np.random.randn(3, 4)), Tensor(np.random.randn(4, 5))])


def test_broadcast_add():
    np.random.seed(2)
    _grad_check(lambda X, b: (X + b).relu().sum(),
                [Tensor(np.random.randn(8, 5)), Tensor(np.random.randn(5))])


def test_broadcast_mul():
    np.random.seed(3)
    _grad_check(lambda X, w: (X * w).sum(),
                [Tensor(np.random.randn(4, 3)), Tensor(np.random.randn(3))])


def test_sub():
    np.random.seed(4)
    _grad_check(lambda A, B: (A - B).relu().sum(),
                [Tensor(np.random.randn(3, 3)), Tensor(np.random.randn(3, 3))])


def test_div():
    np.random.seed(5)
    _grad_check(lambda A, B: (A / B).sum(),
                [Tensor(np.random.randn(2, 3)), Tensor(np.random.randn(2, 3) * 2 + 3)])


def test_sigmoid():
    np.random.seed(6)
    _grad_check(lambda X: X.sigmoid().sum(), [Tensor(np.random.randn(4, 5))])


def test_tanh():
    np.random.seed(7)
    _grad_check(lambda X: X.tanh().sum(), [Tensor(np.random.randn(4, 5))])


def test_mean():
    np.random.seed(8)
    _grad_check(lambda X: X.mean(), [Tensor(np.random.randn(4, 5))])


def test_softmax_ce():
    np.random.seed(9)
    targets = np.array([1, 0, 2, 1])
    _grad_check(lambda X: X.softmax_cross_entropy(targets),
                [Tensor(np.random.randn(4, 4))])


def test_full_mlp():
    np.random.seed(10)
    targets = np.array([0, 1, 2, 0, 1])
    _grad_check(
        lambda X, W1, b1, W2: X.matmul(W1).__add__(b1).relu().matmul(W2)
                              .softmax_cross_entropy(targets),
        [Tensor(np.random.randn(5, 4)), Tensor(np.random.randn(4, 6)),
         Tensor(np.random.randn(6)), Tensor(np.random.randn(6, 3))])


def test_basic_accumulation():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b
    d = c + a
    L = d * d
    L.backward()
    assert abs(a.grad - 64.0) < 1e-6
    assert abs(b.grad - 32.0) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("All gradient-check tests passed.")
