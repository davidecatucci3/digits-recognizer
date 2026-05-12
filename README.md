# digits-recognizer

> A fully connected feedforward neural network built in pure NumPy — no PyTorch, no TensorFlow — trained with mini-batch SGD and the Adam optimizer.

---

## Overview

This project implements a multi-layer neural network from the ground up using only NumPy. It supports configurable layer sizes, mini-batch stochastic gradient descent, and an Adam-style adaptive optimizer. Trained weights are saved to JSON when a target accuracy threshold is reached.

---

## How It Works

1. **Feedforward** — Input passes through each layer: `z = Wx + b`, then `a = sigmoid(z)`.
2. **Backpropagation** — Gradients of the MSE loss are computed layer by layer using the chain rule.
3. **Mini-batch SGD + Adam** — Weights are updated using momentum and adaptive per-parameter learning rates.
4. **Evaluation** — After each epoch, accuracy is measured on the test set.
5. **Checkpoint** — If final accuracy ≥ 96%, weights and biases are saved to `parameters.json`.

---

## Project Structure

```
.
├── network.py              # Neural network definition, training loop, optimizer
├── hyperparameters.py      # All training hyperparameters in one place
├── dataset.py              # train_data and test_data loaders
└── parameters.json         # Saved model weights (generated after training)
```

---

## Architecture

The network is defined by a list of layer sizes passed to `NeuralNetwork`:

```python
NeuralNetwork([input_size, 128, 128, 128, 128, 10])
```

| Component | Detail |
|---|---|
| Activation function | Sigmoid |
| Loss function | Mean Squared Error (MSE) |
| Optimizer | Adam (momentum + RMSProp) |
| Output size | 10 classes |
| Hidden layers | 4 × 128 neurons |

---

## Optimizer

The SGD method implements **Adam**-style updates:

- **Momentum** (`beta`) — exponential moving average of gradients
- **RMSProp** (`gamma`) — exponential moving average of squared gradients
- **Epsilon** — small constant for numerical stability

Update rule per parameter:

```
m = β·m + (1 - β)·∇w          # momentum
v = γ·v + (1 - γ)·∇w²         # velocity
w = w - lr · m / (√v + ε)     # weight update
```

---

## Hyperparameters

Defined in `hyperparameters.py` and read at runtime:

| Key | Description |
|---|---|
| `epochs` | Number of full passes over training data |
| `mini batch size` | Samples per gradient update step |
| `learning rate` | Step size (`lr`) |
| `momentum` | Adam `beta` — gradient moving average decay |
| `gamma` | Adam `gamma` — squared gradient decay |
| `weight decay` | Adam `epsilon` — denominator stability term |

---

## Requirements

- Python 3.8+
- NumPy

```bash
pip install numpy
```

---

## Usage

```bash
python network.py
```

Training prints progress each epoch:

```
epoch 1/50  | loss: 0.312  accuracy: 81.45%
epoch 2/50  | loss: 0.278  accuracy: 85.10%
...
epoch 50/50 | loss: 0.041  accuracy: 96.73%
```

If final accuracy reaches **96% or above**, the trained parameters are saved:

```
parameters.json   ← weights + biases serialized as JSON
```

---

## Saving & Loading

The `save()` method serializes the model to JSON:

```python
nn.save('parameters.json')
```

The file contains:

```json
{
  "layers": [784, 128, 128, 128, 128, 10],
  "weights": [[...]],
  "biases": [[...]]
}
```

---

## Known Issues

There is a bug in the `SGD` method — the momentum update for biases overwrites the weights momentum instead of updating `momentum_b`:

```python
# Bug (line 2 should reference momentum_b and sum_dldb):
self.momentum_w = [beta * mw + (1 - beta) * dw for mw, dw in zip(self.momentum_w, sum_dldw)]
self.momentum_w = [beta * mb + (1 - beta) * db for mb, db in zip(self.momentum_w, sum_dldw)]  # ← wrong

# Fix:
self.momentum_b = [beta * mb + (1 - beta) * db for mb, db in zip(self.momentum_b, sum_dldb)]
```

This causes bias momentum to never update, and weight momentum to be overwritten each step.

---

## License

MIT License. See `LICENSE` for details.
