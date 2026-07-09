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

    def __pow__(self, exponent):
        out = Tensor(self.data ** exponent, _children=(self,), _op=f'**{exponent}')

        def _backward():
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad
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
                grad = np.expand_dims(out.grad, axis=axis) if not keepdims else out.grad
                self.grad += np.ones_like(self.data) * grad
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
                grad = np.expand_dims(out.grad, axis=axis) if not keepdims else out.grad
                self.grad += np.ones_like(self.data) * grad / n
        out._backward = _backward
        return out

    def softmax_cross_entropy(self, targets):
        """Combined softmax+CE. Gradient: dL/dz = softmax(z) - onehot(y), divided by N."""
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
