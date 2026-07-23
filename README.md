
# Neural Networks From Scratch

A neural network library built entirely from raw `numpy` — no PyTorch, TensorFlow, JAX, or autograd. Every forward pass, every backward pass, and every gradient was derived by hand (chain rule, on paper) before being implemented and numerically verified.

The goal wasn't just "make something that works" — it was to actually understand what's happening inside a neural network, rather than trusting a framework's autograd to do it for me.

## What's implemented

**Core network (`nn/layers.py`, `nn/model.py`, `nn/losses.py`)**
- `Neuron` — single logistic regression unit, hand-derived gradients
- `Dense` — fully-connected layer with He initialization, ReLU/softmax activations, manual forward + backward
- `Dropout` — inverted dropout with correct train/test scaling
- `Sequential` — chains layers together, propagates gradients end-to-end
- Numerically stable softmax + cross-entropy loss, with the softmax+CE gradient cancellation derived and exploited directly

**Optimizers (`nn/optim.py`)**
- `SGD`
- `Momentum` — exponential moving average of gradients
- `Adam` — adaptive per-parameter learning rates with bias correction
- L2 weight decay (implemented at the optimizer level, not the layer level)

**Convolutions (`nn/conv.py`)**
- Naive nested-loop convolution (for correctness verification)
- Vectorized convolution via the `im2col` trick — patches flattened and extracted as one big matrix multiply instead of Python loops
- Multi-filter, multi-channel forward pass
- Full backward pass: `dBias`, `dFilters`, and `dImage` — the last derived from raw chain-rule expansion, generalizing into the "flip the filter 180°, pad, and convolve" result, verified against numerical gradient checking

**Data (`nn/data.py`)**
- Raw MNIST IDX-format binary parser, written from the file spec (no helper libraries)

**Verification (`test/`)**
- Numerical gradient checking for both the dense network and the convolution backward pass — every custom gradient is checked against finite-difference approximation before being trusted in training

## Results

| Model | Optimizer | Test Accuracy |
|---|---|---|
| MLP (784→128→64→10) | SGD | 97.75% |
| MLP (784→128→64→10) | Momentum | 97.46% |
| MLP (784→128→64→10) | Adam | 97.94% |

All trained on the full 60,000-example MNIST training set, evaluated on the 10,000-example test set.

## What I learned along the way

Some of this project's most useful moments came from things *not* working as expected:
- Dropout and L2 regularization don't universally help — on a small-data overfitting experiment, both underperformed doing nothing under certain hyperparameter choices (dropout too aggressive on a narrow layer; L2 interacting poorly with Adam's adaptive scaling, which is part of the real-world motivation for AdamW).
- Gradient checking caught real bugs before they wasted hours of debugging a training run.
- The convolution backward pass ("flip and convolve") isn't just a rule to memorize — it falls directly out of the chain rule once you track which filter weight touches which pixel across all sliding-window positions.

## Project structure

```
nn/
  layers.py     # Neuron, Dense, Dropout
  losses.py     # softmax, cross-entropy
  model.py      # Sequential
  optim.py      # SGD, Momentum, Adam
  conv.py       # convolution forward + backward
  data.py       # MNIST IDX parser
test/
  test_gradients.py   # numerical gradient checking
train.py                  # MLP training on full MNIST
train_overfit_demo.py     # regularization experiments
```

## Running it

```bash
python train.py
```

Requires only `numpy` (and `matplotlib` if you want to plot loss/accuracy curves). MNIST files should be placed in the working directory in raw IDX format (`train-images.idx3-ubyte`, etc.).
