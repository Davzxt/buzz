"""Adam optimizer that updates numpy arrays in place."""
import numpy as np


class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        bias1 = 1 - self.beta1 ** self.t
        bias2 = 1 - self.beta2 ** self.t
        for k, g in grads.items():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g * g
            update = (self.m[k] / bias1) / (np.sqrt(self.v[k] / bias2) + self.eps)
            self.params[k] -= (self.lr * update).astype(self.params[k].dtype)


def log_softmax(z):
    shifted = z - z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
