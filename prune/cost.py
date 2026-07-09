import time
import numpy as np
from engine.engine import Tensor
from nn.nn import Linear


def active_params(model):
    active = 0
    total = 0
    for layer in model.linear_layers():
        active += int(layer.mask.sum())
        total += layer.mask.size
    return active, total


def dense_flops(model, batch_size=1):
    flops = 0
    for layer in model.linear_layers():
        flops += 2 * batch_size * layer.W.data.shape[0] * layer.W.data.shape[1]
    return flops


def sparse_flops(model, batch_size=1):
    flops = 0
    for layer in model.linear_layers():
        nnz = int(layer.mask.sum())
        flops += 2 * batch_size * nnz
    return flops


def _forward_dense(model, X):
    return model(Tensor(X)).data


def _build_sparse_layers(model):
    layers = []
    for layer in model.layers:
        if isinstance(layer, Linear):
            W = layer.W.data * layer.mask
            out_dim = W.shape[1]
            cols = []
            for j in range(out_dim):
                rows = np.nonzero(W[:, j])[0]
                vals = W[rows, j]
                cols.append((rows, vals))
            layers.append(('linear', cols, layer.b.data, out_dim))
        else:
            layers.append(('relu', None, None, None))
    return layers


def _forward_sparse(sparse_layers, X):
    h = X
    for kind, cols, b, out_dim in sparse_layers:
        if kind == 'linear':
            out = np.empty((h.shape[0], out_dim))
            for j, (rows, vals) in enumerate(cols):
                if rows.size == 0:
                    out[:, j] = b[j]
                else:
                    out[:, j] = h[:, rows] @ vals + b[j]
            h = out
        else:
            h = np.maximum(0, h)
    return h


def time_forward(fn, *args, repeats=30):
    fn(*args)
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def measure_cost(model, X, repeats=30):
    active, total = active_params(model)
    d_flops = dense_flops(model, batch_size=len(X))
    s_flops = sparse_flops(model, batch_size=len(X))

    dense_time = time_forward(_forward_dense, model, X, repeats=repeats)

    sparse_layers = _build_sparse_layers(model)
    dense_out = _forward_dense(model, X)
    sparse_out = _forward_sparse(sparse_layers, X)
    max_diff = float(np.abs(dense_out - sparse_out).max())

    sparse_time = time_forward(_forward_sparse, sparse_layers, X, repeats=repeats)

    return {
        'active_params': active,
        'total_params': total,
        'sparsity': 1.0 - active / total,
        'dense_flops': d_flops,
        'sparse_flops': s_flops,
        'flop_reduction': 1.0 - s_flops / d_flops,
        'dense_time_ms': dense_time * 1e3,
        'sparse_time_ms': sparse_time * 1e3,
        'speedup': dense_time / sparse_time,
        'output_max_diff': max_diff,
    }