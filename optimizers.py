"""Gradient-based optimizers for neural networks."""

import numpy as np
from typing import List


class SGD:
    """Stochastic gradient descent with fixed learning rate."""

    def __init__(self, parameters: List, lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad.fill(0)


class Adam:
    """Adam optimizer with bias-corrected moment estimates.

    Reference: Kingma & Ba, 'Adam: A Method for Stochastic Optimization' (2014)
    """

    def __init__(self, parameters: List, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            param.grad.fill(0)
