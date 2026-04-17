"""Training loop for neural networks."""

import numpy as np
from typing import Optional, Dict, List, Callable


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Classification accuracy from raw logits and integer labels."""
    return float(np.mean(predictions.argmax(axis=1) == targets.flatten()))


class Trainer:
    """Mini-batch training loop with per-epoch loss and metric tracking."""

    def __init__(self, model, optimizer, loss_fn, metric_fn=None,
                 epochs: int = 100, batch_size: int = 32, seed: int = 0):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_metric": [], "val_metric": [],
        }

    def fit(self, X, Y, X_val=None, Y_val=None):
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]

        for _ in range(self.epochs):
            idx = rng.permutation(n)
            epoch_loss = []

            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                Xb, Yb = X[batch], Y[batch]

                pred = self.model.forward(Xb)
                loss = self.loss_fn.forward(pred, Yb)
                epoch_loss.append(loss)

                self.optimizer.zero_grad()
                grad = self.loss_fn.backward()
                self.model.backward(grad)
                self.optimizer.step()

            self.history["train_loss"].append(float(np.mean(epoch_loss)))

            if self.metric_fn:
                self.history["train_metric"].append(
                    self.metric_fn(self.model.forward(X), Y))

            if X_val is not None and Y_val is not None:
                val_pred = self.model.forward(X_val)
                self.history["val_loss"].append(
                    float(self.loss_fn.forward(val_pred, Y_val)))
                if self.metric_fn:
                    self.history["val_metric"].append(
                        self.metric_fn(val_pred, Y_val))

        return self
