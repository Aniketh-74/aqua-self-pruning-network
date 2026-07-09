import numpy as np


def make_spirals(n_points_per_class=500, n_classes=3, noise=0.2, seed=0):
    rng = np.random.RandomState(seed)
    N = n_points_per_class
    D = 2
    X = np.zeros((N * n_classes, D))
    y = np.zeros(N * n_classes, dtype=np.int64)
    for c in range(n_classes):
        ix = range(N * c, N * (c + 1))
        r = np.linspace(0.0, 1.0, N)
        t = np.linspace(c * 4, (c + 1) * 4, N) + rng.randn(N) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    perm = rng.permutation(N * n_classes)
    return X[perm], y[perm]


def train_test_split(X, y, test_frac=0.2, seed=0):
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
