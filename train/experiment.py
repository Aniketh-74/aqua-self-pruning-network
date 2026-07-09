import json
import numpy as np
from engine.engine import Tensor
from nn.nn import Linear, ReLU, Sequential
from train.optim import AdamPrunable
from train.data import make_spirals, train_test_split
from prune.prune import Pruner, count_sparsity
from prune.cost import measure_cost


def accuracy(model, X, y):
    logits = model(Tensor(X))
    return float(np.mean(np.argmax(logits.data, axis=1) == y))


def iterate_minibatches(X, y, batch_size, seed=None):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


def build_model(seed, in_features=2, hidden=64, n_classes=3):
    np.random.seed(seed)
    return Sequential(
        Linear(in_features, hidden), ReLU(),
        Linear(hidden, hidden), ReLU(),
        Linear(hidden, n_classes),
    )


def build_masks(model):
    masks = []
    for layer in model.layers:
        if isinstance(layer, Linear):
            masks.append(layer.mask)
            masks.append(np.ones_like(layer.b.data))
    return masks


def train_pruned(X_train, y_train, X_test, y_test, target_sparsity,
                 criterion, seed, epochs=150, batch_size=64, lr=0.01):
    model = build_model(seed)
    params = model.parameters()
    masks = build_masks(model)
    optimizer = AdamPrunable(params, masks, lr=lr)
    steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
    total_steps = epochs * steps_per_epoch

    pruner = None
    if target_sparsity > 0:
        pruner = Pruner(model, optimizer, criterion=criterion,
                        s_final=target_sparsity, total_steps=total_steps,
                        prune_start=steps_per_epoch * 5,
                        prune_end=int(total_steps * 0.75),
                        prune_every=steps_per_epoch, allow_regrowth=False)

    step = 0
    for epoch in range(epochs):
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, seed=epoch):
            logits = model(Tensor(xb))
            loss = logits.softmax_cross_entropy(yb)
            optimizer.zero_grad()
            loss.backward()
            if pruner is not None:
                pruner.accumulate()
            optimizer.step()
            if pruner is not None:
                pruner.maybe_prune(step)
            step += 1

    achieved_sparsity, _, _ = count_sparsity(model.linear_layers())
    test_acc = accuracy(model, X_test, y_test)
    return model, achieved_sparsity, test_acc


def run_sweep(seeds=(0, 1, 2, 3, 4),
              sparsities=(0.0, 0.5, 0.75, 0.9, 0.95),
              criteria=('magnitude', 'saliency', 'accum_saliency')):
    X, y = make_spirals(500, 3, noise=0.2, seed=0)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.2, seed=0)

    results = []
    for criterion in criteria:
        for target in sparsities:
            accs, sps = [], []
            last_model = None
            for seed in seeds:
                model, sp, acc = train_pruned(
                    X_train, y_train, X_test, y_test, target, criterion, seed)
                accs.append(acc)
                sps.append(sp)
                last_model = model
            cost = measure_cost(last_model, X_test)
            row = {
                'criterion': criterion,
                'target_sparsity': target,
                'achieved_sparsity_mean': float(np.mean(sps)),
                'acc_mean': float(np.mean(accs)),
                'acc_std': float(np.std(accs)),
                'acc_all_seeds': [float(a) for a in accs],
                'n_seeds': len(seeds),
                'active_params': cost['active_params'],
                'total_params': cost['total_params'],
                'flop_reduction': cost['flop_reduction'],
                'dense_time_ms': cost['dense_time_ms'],
                'sparse_time_ms': cost['sparse_time_ms'],
                'speedup': cost['speedup'],
                'output_max_diff': cost['output_max_diff'],
            }
            results.append(row)
            print(f"{criterion:15s} | target {target:.2f} | "
                  f"achieved {row['achieved_sparsity_mean']:.3f} | "
                  f"acc {row['acc_mean']:.3f} +/- {row['acc_std']:.3f} | "
                  f"FLOP-red {row['flop_reduction']:.3f} | "
                  f"out_diff {row['output_max_diff']:.1e}")
    return results


def make_claim(results):
    def find(crit, target):
        for r in results:
            if r['criterion'] == crit and abs(r['target_sparsity'] - target) < 1e-9:
                return r
        return None

    out = {}
    for target in (0.9, 0.95):
        mag = find('magnitude', target)
        sal = find('saliency', target)
        acc = find('accum_saliency', target)
        combined_std = (acc['acc_std'] ** 2 + mag['acc_std'] ** 2) ** 0.5
        diff = acc['acc_mean'] - mag['acc_mean']
        out[f'sparsity_{int(target*100)}'] = {
            'magnitude_acc': [mag['acc_mean'], mag['acc_std']],
            'single_batch_saliency_acc': [sal['acc_mean'], sal['acc_std']],
            'accum_saliency_acc': [acc['acc_mean'], acc['acc_std']],
            'accum_minus_magnitude': diff,
            'combined_std': combined_std,
            'difference_within_noise': bool(abs(diff) <= combined_std),
        }
    out['claim'] = ('Single-batch saliency (|w*grad| from one minibatch) is a noisy '
                    'importance estimate and underperforms magnitude pruning. '
                    'Accumulating saliency over an epoch reduces this noise. The '
                    'committed per-seed numbers show whether the accumulated version '
                    'closes the gap on two-spirals; differences within combined std '
                    'are treated as noise, not signal.')
    return out


def plot_pareto(results, path='pareto.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'magnitude': 'crimson', 'saliency': 'orange', 'accum_saliency': 'navy'}
    labels = {'magnitude': 'magnitude', 'saliency': 'single-batch saliency',
              'accum_saliency': 'accumulated saliency'}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for criterion in ('magnitude', 'saliency', 'accum_saliency'):
        rows = [r for r in results if r['criterion'] == criterion]
        rows.sort(key=lambda r: r['achieved_sparsity_mean'])
        xs = [r['achieved_sparsity_mean'] for r in rows]
        ys = [r['acc_mean'] for r in rows]
        es = [r['acc_std'] for r in rows]
        ax.errorbar(xs, ys, yerr=es, marker='o', capsize=4,
                    color=colors[criterion], label=labels[criterion])
    ax.set_xlabel('Achieved sparsity')
    ax.set_ylabel('Test accuracy (mean +/- std over seeds)')
    ax.set_title('Sparsity vs Accuracy: three importance criteria')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"Saved {path}")


if __name__ == '__main__':
    results = run_sweep()
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved results.json")
    claim = make_claim(results)
    with open('claim.json', 'w') as f:
        json.dump(claim, f, indent=2)
    print("Saved claim.json")
    plot_pareto(results)

    print("=" * 66)
    print("CLAIM SUMMARY")
    print("=" * 66)
    for k in ('sparsity_90', 'sparsity_95'):
        c = claim[k]
        print(f"\n@ {k}:")
        print(f"  magnitude:              {c['magnitude_acc'][0]:.4f} +/- {c['magnitude_acc'][1]:.4f}")
        print(f"  single-batch saliency:  {c['single_batch_saliency_acc'][0]:.4f} +/- {c['single_batch_saliency_acc'][1]:.4f}")
        print(f"  accumulated saliency:   {c['accum_saliency_acc'][0]:.4f} +/- {c['accum_saliency_acc'][1]:.4f}")
        print(f"  accum - magnitude:      {c['accum_minus_magnitude']:+.4f}  (within noise: {c['difference_within_noise']})")