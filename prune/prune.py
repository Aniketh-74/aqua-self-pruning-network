import numpy as np


def count_sparsity(linear_layers):
    n_zero = 0
    n_total = 0
    for layer in linear_layers:
        n_zero += int((layer.mask == 0).sum())
        n_total += layer.mask.size
    return n_zero / n_total, n_zero, n_total


def cubic_schedule(step, total_steps, s_initial=0.0, s_final=0.9,
                   start_step=0, end_step=None):
    if end_step is None:
        end_step = total_steps
    if step < start_step:
        return s_initial
    if step >= end_step:
        return s_final
    progress = (step - start_step) / (end_step - start_step)
    return s_final + (s_initial - s_final) * (1 - progress) ** 3


class Pruner:

    def __init__(self, model, optimizer, criterion='saliency',
                 s_final=0.9, total_steps=1000, prune_start=0,
                 prune_end=None, prune_every=50, allow_regrowth=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.s_final = s_final
        self.total_steps = total_steps
        self.prune_start = prune_start
        self.prune_end = prune_end if prune_end is not None else int(total_steps * 0.75)
        self.prune_every = prune_every
        self.allow_regrowth = allow_regrowth

        self.linear_layers = model.linear_layers()
        self.w_param_index = {}
        for li, layer in enumerate(self.linear_layers):
            for pi, p in enumerate(optimizer.params):
                if p is layer.W:
                    self.w_param_index[li] = pi
                    break

    def _importance(self, layer):
        if self.criterion == 'magnitude':
            return np.abs(layer.W.data)
        elif self.criterion == 'saliency':
            return np.abs(layer.W.data * layer.W.grad)
        else:
            raise ValueError(f"unknown criterion {self.criterion}")

    def maybe_prune(self, step):
        """Call after optimizer.step(). Applies the schedule and updates masks."""
        if step < self.prune_start or (step % self.prune_every != 0):
            return
        target_sparsity = cubic_schedule(
            step, self.total_steps, 0.0, self.s_final,
            self.prune_start, self.prune_end)
        self._apply_sparsity(target_sparsity)

    def _apply_sparsity(self, target_sparsity):
        all_scores = []
        for layer in self.linear_layers:
            all_scores.append(self._importance(layer).flatten())
        flat = np.concatenate(all_scores)
        n_total = flat.size
        n_prune = int(round(target_sparsity * n_total))
        if n_prune <= 0:
            return

        threshold = np.partition(flat, n_prune - 1)[n_prune - 1]

        for li, layer in enumerate(self.linear_layers):
            scores = self._importance(layer)
            new_mask = (scores > threshold).astype(np.float64)

            if self.allow_regrowth:
                revived = (new_mask == 1) & (layer.mask == 0)
                if revived.any():
                    pi = self.w_param_index[li]
                    self.optimizer.reset_moments(pi, revived)
                    layer.W.data[revived] = np.random.randn(int(revived.sum())) * 0.01
            else:
                new_mask = np.minimum(new_mask, layer.mask)

            layer.mask[:] = new_mask
            layer.W.data *= layer.mask