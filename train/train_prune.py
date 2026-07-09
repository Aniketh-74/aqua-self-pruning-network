import numpy as np
from engine.engine import Tensor
from nn.nn import Linear, ReLU, Sequential
from train.optim import AdamPrunable
from train.data import make_spirals, train_test_split
from prune.prune import Pruner, count_sparsity


def accuracy(model, X, y):
    logits = model(Tensor(X))
    return np.mean(np.argmax(logits.data, axis=1) == y)


def iterate_minibatches(X, y, batch_size, seed=None):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


def build_model(in_features=2, hidden=64, n_classes=3):
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


def train_with_pruning(model, X_train, y_train, X_test, y_test,
                       target_sparsity=0.9, criterion='saliency',
                       epochs=150, batch_size=64, lr=0.01,
                       allow_regrowth=False, seed=42, verbose=True):
    np.random.seed(seed)
    params = model.parameters()
    masks = build_masks(model)
    optimizer = AdamPrunable(params, masks, lr=lr)

    steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
    total_steps = epochs * steps_per_epoch

    pruner = Pruner(model, optimizer, criterion=criterion,
                    s_final=target_sparsity, total_steps=total_steps,
                    prune_start=steps_per_epoch * 5,           
                    prune_end=int(total_steps * 0.75),
                    prune_every=steps_per_epoch,              
                    allow_regrowth=allow_regrowth)

    step = 0
    history = {'loss': [], 'test_acc': [], 'sparsity': []}
    for epoch in range(epochs):
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, seed=epoch):
            logits = model(Tensor(xb))
            loss = logits.softmax_cross_entropy(yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()          
            pruner.maybe_prune(step)   
            step += 1

        sp, _, _ = count_sparsity(model.linear_layers())
        te = accuracy(model, X_test, y_test)
        history['loss'].append(loss.data)
        history['test_acc'].append(te)
        history['sparsity'].append(sp)
        if verbose and (epoch % 15 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d} | loss {loss.data:.4f} | "
                  f"test_acc {te:.3f} | sparsity {sp:.3f}")

    return history, pruner


def verify_honest_masking(model):
    max_pruned_weight = 0.0
    for layer in model.linear_layers():
        pruned = layer.W.data[layer.mask == 0]
        if pruned.size > 0:
            max_pruned_weight = max(max_pruned_weight, np.abs(pruned).max())
    return max_pruned_weight


if __name__ == "__main__":
    X, y = make_spirals(500, 3, noise=0.2, seed=0)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.2, seed=0)

    print("=" * 60)
    print("PART 3: Self-pruning to 90% sparsity (saliency criterion)")
    print("=" * 60)
    model = build_model()
    dense_params = sum(p.data.size for p in model.parameters())
    hist, pruner = train_with_pruning(
        model, X_train, y_train, X_test, y_test,
        target_sparsity=0.9, criterion='saliency', epochs=150)

    print("=" * 60)
    final_sp, n_zero, n_total = count_sparsity(model.linear_layers())
    print(f"Final sparsity:   {final_sp:.3f}  ({n_zero}/{n_total} weights pruned)")
    print(f"Final test acc:   {hist['test_acc'][-1]:.3f}")

    max_pruned = verify_honest_masking(model)
    print(f"Max |pruned weight|: {max_pruned:.2e}  "
          f"({'real zeros' if max_pruned == 0.0 else 'LEAK!'})")
    print("=" * 60)