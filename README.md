# numpy-nn

A neural network framework built from scratch using only NumPy. Implements backpropagation through a modular computational graph — no PyTorch, TensorFlow, or autograd.

## What's Implemented

**Layers & Activations**
- Linear (fully connected) with He initialization
- ReLU and Sigmoid activations
- Sequential container for chaining layers
- Residual (skip connection) blocks

**Loss Functions**
- Mean Squared Error
- Cross-Entropy with numerically stable softmax

**Optimizers**
- SGD — stochastic gradient descent
- Adam — adaptive learning rates with bias-corrected moments

**Architectures**
- MLP — configurable depth and width
- MLP with skip connections
- Autoencoder — encoder-decoder for compression and reconstruction

## Demo Notebook

`demo.ipynb` runs three experiments on handwritten digits:

1. **MLP Classification** — trains a 3-layer MLP to classify digits, visualizes loss and accuracy curves
2. **Skip Connections** — compares a deep MLP with and without residual blocks to show improved convergence
3. **Autoencoder** — compresses 64-dimensional digit images to 16 dimensions and reconstructs them, with side-by-side visualization

## Getting Started

```bash
git clone https://github.com/git-samia/numpy-nn.git
cd numpy-nn
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

## Project Structure

```
numpy-nn/
├── nn.py             # Layers, activations, skip blocks, autoencoder
├── losses.py         # MSE and cross-entropy losses
├── optimizers.py     # SGD and Adam
├── trainer.py        # Training loop and metrics
├── demo.ipynb        # Interactive experiments
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python 3** + **NumPy** — all neural network implementations
- **Matplotlib** — visualizations
- **scikit-learn** — dataset loading only
