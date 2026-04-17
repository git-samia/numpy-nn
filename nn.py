"""Neural network building blocks with manual backpropagation."""

import numpy as np
from typing import List, Dict


class Parameter:
    """Wraps a NumPy array with its accumulated gradient."""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)

    def __repr__(self):
        return f"Parameter(shape={self.data.shape})"


class Module:
    """Base class for all neural network components."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> List[Parameter]:
        return []

    def named_parameters(self, prefix: str = "") -> Dict[str, Parameter]:
        return {}


class Linear(Module):
    """Fully connected layer: y = xW + b, with He initialization."""

    def __init__(self, in_features: int, out_features: int, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        scale = np.sqrt(2.0 / in_features)
        self.W = Parameter(rng.standard_normal((in_features, out_features)) * scale)
        self.b = Parameter(np.zeros((1, out_features)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return x @ self.W.data + self.b.data

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.W.grad += self._input.T @ grad
        self.b.grad += grad.sum(axis=0, keepdims=True)
        return grad @ self.W.data.T

    def parameters(self) -> List[Parameter]:
        return [self.W, self.b]

    def named_parameters(self, prefix: str = "") -> Dict[str, Parameter]:
        return {f"{prefix}W": self.W, f"{prefix}b": self.b}


class ReLU(Module):
    """ReLU activation: max(0, x)"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0).astype(float)
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask


class Sigmoid(Module):
    """Sigmoid activation: 1 / (1 + exp(-x))"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = 1.0 / (1.0 + np.exp(-x))
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._out * (1.0 - self._out)


class Sequential(Module):
    """Chains modules: forward in order, backward in reverse."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> List[Parameter]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def named_parameters(self, prefix: str = "") -> Dict[str, Parameter]:
        named = {}
        for i, layer in enumerate(self.layers):
            named.update(layer.named_parameters(f"{prefix}layer{i}."))
        return named


class SkipBlock(Module):
    """Residual block: output = f(x) + x

    Applies a two-layer sub-network with an activation in between,
    then adds the original input back (skip connection).
    Input and output dimensions must match.
    """

    def __init__(self, dim: int, activation_cls=ReLU, rng=None):
        self.block = Sequential(
            Linear(dim, dim, rng),
            activation_cls(),
            Linear(dim, dim, rng),
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return self.block.forward(x) + x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.block.backward(grad) + grad

    def parameters(self) -> List[Parameter]:
        return self.block.parameters()

    def named_parameters(self, prefix: str = "") -> Dict[str, Parameter]:
        return self.block.named_parameters(f"{prefix}skip.")


class Autoencoder(Module):
    """Autoencoder: encoder compresses, decoder reconstructs."""

    def __init__(self, encoder: Module, decoder: Module):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._encoded = self.encoder.forward(x)
        return self.decoder.forward(self._encoded)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.encoder.forward(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = self.decoder.backward(grad)
        return self.encoder.backward(grad)

    def parameters(self) -> List[Parameter]:
        return self.encoder.parameters() + self.decoder.parameters()

    def named_parameters(self, prefix: str = "") -> Dict[str, Parameter]:
        named = {}
        named.update(self.encoder.named_parameters(f"{prefix}enc."))
        named.update(self.decoder.named_parameters(f"{prefix}dec."))
        return named


def build_mlp(layer_sizes, activation_cls=ReLU, rng=None):
    """Build an MLP from a list of layer sizes (no activation after the last layer)."""
    if rng is None:
        rng = np.random.default_rng()
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(Linear(layer_sizes[i], layer_sizes[i + 1], rng))
        if i < len(layer_sizes) - 2:
            layers.append(activation_cls())
    return Sequential(*layers)
