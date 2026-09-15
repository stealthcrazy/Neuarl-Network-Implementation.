# Neural Network from Scratch

A feedforward neural network in NumPy. The purpose of this project was to take a deep dive into the mathematics behind Backpropgation and learn how Neural Networks are implemented.

Trained on MNIST (784 → 16 → 16 → 10, sigmoid throughout).

## What's implemented

- **Forward pass** — affine transform plus sigmoid at each layer, weights and
  biases held in dictionaries keyed by layer.
- **Backpropagation** — the chain rule written out layer by layer. Each step
  computes the error signal `Sₗ = (Wₗ₊₁ᵀ Sₗ₊₁) ⊙ σ'(Zₗ)`, then the weight
  gradient as the outer product `Sₗ Aₗ₋₁ᵀ`.
- **Minibatch SGD** — gradients summed across a batch, then applied once.
- **Sum of squared residuals** as the cost, with its derivative.

## References
- https://www.3blue1brown.com/?topic=neural-networks&lesson=neural-networks
