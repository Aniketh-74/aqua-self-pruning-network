import numpy as np
from engine.engine import Tensor
from nn.nn import Linear, ReLU, Sequential
from train.optim import Adam
from train.data import make_spirals, train_test_split


def accuracy(model, X, y):
    logits = model(Tensor(X))
    preds = np.argmax(logits.data, axis=1)
    return np.mean(preds == y)


def iterate_minibatches(X, y, batch_size, seed=None):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def build_model(in_features=2, hidden=64, n_classes=3):
    return Sequential(
        Linear(in_features, hidden),
        ReLU(),
        Linear(hidden, hidden),
        ReLU(),
        Linear(hidden, n_classes),
    )


def train(model, X_train, y_train, X_test, y_test,
          epochs=100, batch_size=64, lr=0.01, verbose=True):
    optimizer = Adam(model.parameters(), lr=lr)
    history = {'loss': [], 'train_acc': [], 'test_acc': []}
    for epoch in range(epochs):
        epoch_losses = []
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, seed=epoch):
            logits = model(Tensor(xb))
            loss = logits.softmax_cross_entropy(yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.data)
        mean_loss = np.mean(epoch_losses)
        tr_acc = accuracy(model, X_train, y_train)
        te_acc = accuracy(model, X_test, y_test)
        history['loss'].append(mean_loss)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        assert not np.isnan(mean_loss), f"NaN loss at epoch {epoch}"
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d} | loss {mean_loss:.4f} | "
                  f"train_acc {tr_acc:.3f} | test_acc {te_acc:.3f}")
    return history


if __name__ == "__main__":
    np.random.seed(42)
    X, y = make_spirals(n_points_per_class=500, n_classes=3, noise=0.2, seed=0)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_frac=0.2, seed=0)
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | classes: {len(np.unique(y))}")
    print("=" * 55)

    model = build_model()
    n_params = sum(p.data.size for p in model.parameters())
    print(f"Model: 2 -> 64 -> 64 -> 3  ({n_params} parameters)")
    print("=" * 55)

    history = train(model, X_train, y_train, X_test, y_test,
                    epochs=100, batch_size=64, lr=0.01)
    print("=" * 55)
    print(f"Final train acc: {history['train_acc'][-1]:.3f}")
    print(f"Final test acc:  {history['test_acc'][-1]:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(history['loss'], color='crimson')
        ax1.set_title("Training Loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
        ax2.plot(history['train_acc'], label='train', color='navy')
        ax2.plot(history['test_acc'], label='test', color='green')
        ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
        ax2.legend()
        plt.tight_layout()
        plt.savefig("learning_curve.png", dpi=120)
        print("Saved learning_curve.png")
    except ImportError:
        print("(matplotlib not installed)")
