import numpy as np

def _unbroadcast(grad, shape):
    ndims_added = grad.ndim - len(shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op='{self._op}')"


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self.__add__(other)


    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, _children=(self, other), _op='-')

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(-out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, _children=(self, other), _op='/')

        def _backward():
            grad_self = (1.0 / other.data) * out.grad
            grad_other = (-self.data / (other.data ** 2)) * out.grad
            self.grad += _unbroadcast(grad_self, self.data.shape)
            other.grad += _unbroadcast(grad_other, other.data.shape)
        out._backward = _backward

        return out

    def __neg__(self):
        out = Tensor(-self.data, _children=(self,), _op='neg')

        def _backward():
            self.grad += -out.grad
        out._backward = _backward

        return out
    
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), _children=(self,), _op='relu')

        def _backward():
            self.grad += (self.data > 0).astype(np.float64) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, _children=(self,), _op='sigmoid')

        def _backward():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward

        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims),
                     _children=(self,), _op='sum')

        def _backward():
            if axis is None:
                self.grad += np.ones_like(self.data) * out.grad
            else:
                # Expand the summed axis back
                self.grad += np.ones_like(self.data) * np.expand_dims(out.grad, axis=axis) \
                    if not keepdims else np.ones_like(self.data) * out.grad
        out._backward = _backward

        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims),
                     _children=(self,), _op='mean')

        def _backward():
            if axis is None:
                self.grad += np.ones_like(self.data) * out.grad / n
            else:
                grad_expanded = np.expand_dims(out.grad, axis=axis) \
                    if not keepdims else out.grad
                self.grad += np.ones_like(self.data) * grad_expanded / n
        out._backward = _backward

        return out

    def softmax_cross_entropy(self, targets):
        shifted = self.data - self.data.max(axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

        N = self.data.shape[0]
        log_probs = np.log(probs[np.arange(N), targets] + 1e-12)
        loss_val = -np.mean(log_probs)

        out = Tensor(loss_val, _children=(self,), _op='softmax_ce')

        def _backward():
            grad = probs.copy()
            grad[np.arange(N), targets] -= 1.0 
            self.grad += grad / N * out.grad    
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data, dtype=np.float64)

        for node in reversed(topo):
            node._backward()


def gradient_check(name, build_and_run, tensors, eps=1e-5, tol=1e-4):

    for t in tensors:
        t.grad = np.zeros_like(t.data)
    loss = build_and_run(*tensors)
    loss.backward()

    all_pass = True

    for t_idx, t in enumerate(tensors):
        flat_size = t.data.size
        num_checks = min(5, flat_size)
        indices = np.random.choice(flat_size, num_checks, replace=False)

        for flat_i in indices:
            coord = np.unravel_index(flat_i, t.data.shape)
            analytical = t.grad[coord]

            old_val = t.data[coord]

            t.data[coord] = old_val + eps
            loss_plus = build_and_run(*tensors).data

            t.data[coord] = old_val - eps
            loss_minus = build_and_run(*tensors).data

            t.data[coord] = old_val  # restore

            numerical = (loss_plus - loss_minus) / (2 * eps)

            diff = abs(analytical - numerical)
            if diff > tol:
                print(f"  FAIL tensor[{t_idx}]{coord}: "
                      f"analytical={analytical:.6f} numerical={numerical:.6f} diff={diff:.6f}")
                all_pass = False

    status = "PASS ✓" if all_pass else "FAIL ✗"
    print(f"[{status}] {name}")
    return all_pass


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("GRADIENT CHECKS — Part 1 verification")
    print("=" * 60)

    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b
    d = c + a
    L = d * d
    L.backward()
    assert abs(a.grad - 64.0) < 1e-6, f"Expected 64.0, got {a.grad}"
    assert abs(b.grad - 32.0) < 1e-6, f"Expected 32.0, got {b.grad}"
    print("[PASS ✓] basic add + mul (dL/da=64, dL/db=32)")

    gradient_check(
        "matmul + relu",
        lambda A, B: A.matmul(B).relu().sum(),
        [Tensor(np.random.randn(3, 4)), Tensor(np.random.randn(4, 5))]
    )

    gradient_check(
        "broadcast add (N,D) + (D,)",
        lambda X, b: (X + b).relu().sum(),
        [Tensor(np.random.randn(8, 5)), Tensor(np.random.randn(5))]
    )

    gradient_check(
        "broadcast mul (N,D) * (D,)",
        lambda X, w: (X * w).sum(),
        [Tensor(np.random.randn(4, 3)), Tensor(np.random.randn(3))]
    )

    gradient_check(
        "subtraction",
        lambda A, B: (A - B).sum(),
        [Tensor(np.random.randn(3, 3)), Tensor(np.random.randn(3, 3))]
    )

    gradient_check(
        "division",
        lambda A, B: (A / B).sum(),
        [Tensor(np.random.randn(3, 3)), Tensor(np.random.randn(3, 3) * 2 + 3)]
    )

    gradient_check(
        "sigmoid",
        lambda X: X.sigmoid().sum(),
        [Tensor(np.random.randn(4, 5))]
    )

    gradient_check(
        "tanh",
        lambda X: X.tanh().sum(),
        [Tensor(np.random.randn(4, 5))]
    )

    gradient_check(
        "mean",
        lambda X: X.mean(),
        [Tensor(np.random.randn(4, 5))]
    )

    targets_10 = np.array([1, 0, 2, 1])
    gradient_check(
        "softmax + cross-entropy",
        lambda X: X.softmax_cross_entropy(targets_10),
        [Tensor(np.random.randn(4, 4))]
    )

    targets_11 = np.array([0, 1, 2, 0, 1])
    gradient_check(
        "full MLP forward: matmul→bias→relu→matmul→softmax_ce",
        lambda X, W1, b1, W2: X.matmul(W1).__add__(b1).relu().matmul(W2)
                                .softmax_cross_entropy(targets_11),
        [
            Tensor(np.random.randn(5, 4)),   
            Tensor(np.random.randn(4, 6)),   
            Tensor(np.random.randn(6)),       
            Tensor(np.random.randn(6, 3)),    
        ]
    )

    print("=" * 60)
    print("All gradient checks complete.")
    print("=" * 60)