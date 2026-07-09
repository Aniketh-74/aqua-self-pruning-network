import numpy as np
from engine.engine import Tensor


class Linear:
    """y = x @ (W * mask) + b. He init. Mask multiplies W so pruned weights
    contribute exactly zero in forward AND receive zero gradient."""

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        scale = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.randn(in_features, out_features) * scale)
        self.b = Tensor(np.zeros(out_features))
        self.mask = np.ones_like(self.W.data)

    def __call__(self, x):
        masked_W = self.W * Tensor(self.mask)
        return x.matmul(masked_W) + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU:
    def __call__(self, x):
        return x.relu()

    def parameters(self):
        return []


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params += layer.parameters()
        return params

    def linear_layers(self):
        return [l for l in self.layers if isinstance(l, Linear)]
