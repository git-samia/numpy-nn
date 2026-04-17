"""Loss functions with forward (compute loss) and backward (compute gradient)."""

import numpy as np


class MSELoss:
    """Mean squared error: (1/2n) * sum((pred - target)^2)"""

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        self._pred = predictions
        self._target = targets
        return 0.5 * np.mean((predictions - targets) ** 2)

    def backward(self) -> np.ndarray:
        return (self._pred - self._target) / self._target.shape[0]


class CrossEntropyLoss:
    """Cross-entropy loss with built-in numerically stable softmax."""

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        self._n = logits.shape[0]
        self._targets = targets.flatten().astype(int)

        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        self._probs = exp_z / exp_z.sum(axis=1, keepdims=True)

        log_probs = -np.log(self._probs[np.arange(self._n), self._targets] + 1e-12)
        return float(np.mean(log_probs))

    def backward(self) -> np.ndarray:
        grad = self._probs.copy()
        grad[np.arange(self._n), self._targets] -= 1
        return grad / self._n
