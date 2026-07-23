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
- `Conv2D` — a full layer class (batched, He-initialized, plugs directly into `Sequential` and every optimizer with no special-casing)

**Pooling & reshaping (`nn/layers.py`)**
- `MaxPool2D` — forward takes the max over non-overlapping windows; backward routes gradient only to the position that was the max, verified with gradient checking
- `Flatten` — bridges the `(m, C, H, W)` convolutional convention and the `(n_in, m)` dense-layer convention, both directions

**A complete, working CNN**
- `Conv2D → MaxPool2D → Flatten → Dense → Dense(softmax)`, trained end-to-end on real MNIST images, every gradient flowing through every layer type derived and verified by hand

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
| CNN (Conv→Pool→Dense→Dense) | Adam | 93.30% |

The MLP results are trained on the full 60,000-example MNIST training set. The CNN result is from just 5,000 training examples and 5 epochs (the from-scratch `Conv2D` layer loops over each image in Python rather than being fully vectorized across the batch, so full-dataset training is slow) — getting within ~4.5 points of the fully-trained MLP on a fraction of the data and epochs is a good sign the convolutional structure is learning efficiently. Training the CNN on the full dataset for more epochs is a natural next step.

## What I learned along the way

Some of this project's most useful moments came from things *not* working as expected:
- Dropout and L2 regularization don't universally help — on a small-data overfitting experiment, both underperformed doing nothing under certain hyperparameter choices (dropout too aggressive on a narrow layer; L2 interacting poorly with Adam's adaptive scaling, which is part of the real-world motivation for AdamW).
- Gradient checking caught real bugs before they wasted hours of debugging a training run.
- The convolution backward pass ("flip and convolve") isn't just a rule to memorize — it falls directly out of the chain rule once you track which filter weight touches which pixel across all sliding-window positions.
- A loop-based, per-image convolution is correct but genuinely slow — a real lesson in why frameworks invest heavily in fully vectorized (or GPU-parallelized) convolution implementations rather than looping over the batch dimension in Python.

## Project structure

```
nn/
  layers.py     # Neuron, Dense, Dropout, MaxPool2D, Flatten
  losses.py     # softmax, cross-entropy
  model.py      # Sequential
  optim.py      # SGD, Momentum, Adam (+ L2 weight decay)
  conv.py       # convolution forward + backward, Conv2D layer
  data.py       # MNIST IDX parser
test/
  test_gradients.py            # numerical gradient checking (dense network)
  test_conv_backward_multi.py  # numerical gradient checking (convolution)
train.py                  # MLP training on full MNIST
train_overfit_demo.py     # regularization experiments
train_cnn.py               # CNN training on MNIST subset
```

## Running it

```bash
python train.py
```

Requires only `numpy` (and `matplotlib` if you want to plot loss/accuracy curves). MNIST files should be placed in the working directory in raw IDX format (`train-images.idx3-ubyte`, etc.).
