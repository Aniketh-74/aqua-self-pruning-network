import numpy as np


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


class AdamPrunable(Adam):
    """Adam with correct masked-weight handling.
    FIX 1: after each step, p.data *= mask re-zeros pruned weights so residual
    momentum cannot drift them off zero.
    FIX 2: reset_moments() zeros m,v for revived connections so stale momentum
    does not corrupt their first updates."""

    def __init__(self, params, masks, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr, beta1, beta2, eps)
        self.masks = masks

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data *= self.masks[i]

    def reset_moments(self, param_index, revived_boolean_mask):
        self.m[param_index][revived_boolean_mask] = 0.0
        self.v[param_index][revived_boolean_mask] = 0.0


class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * p.grad
            p.data += self.velocity[i]

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)
