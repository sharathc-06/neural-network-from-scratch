import sys
import os
# Hack to make Python look in the parent 'data' directory for the 'nn' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from nn.layers import Dense
from nn.model import Sequential
from nn.losses import cross_entropy_loss

def compute_loss(model, X, Y):
    A_last = model.forward(X)
    loss, _ = cross_entropy_loss(A_last, Y)
    return loss

def gradient_check():
    # Small deterministic network
    np.random.seed(42)
    X = np.random.randn(3, 1) # 3 features, 1 example
    Y = np.array([[1], [0]])  # 2 classes, one-hot

    model = Sequential([
        Dense(n_in=3, n_out=2, activation="relu"),
        Dense(n_in=2, n_out=2, activation="softmax")
    ])

    # 1. Get analytical gradients
    A_last = model.forward(X)
    _, dA_last = cross_entropy_loss(A_last, Y)
    model.backward(dA_last)

    # Let's test the first layer's weights
    layer = model.layers[0]
    analytical_dW = layer.dW.copy()
    numerical_dW = np.zeros_like(analytical_dW)

    epsilon = 1e-7

    # 2. Compute numerical gradients
    # Iterate over every single weight in the layer
    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            original_weight = layer.W[i, j]

            layer.W[i, j] = original_weight + epsilon
            loss_plus = compute_loss(model, X, Y)
        
            layer.W[i, j] = original_weight - epsilon
            loss_minus = compute_loss(model, X, Y)
        
            layer.W[i, j] = original_weight
        
            numerical_dW[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            

    # 3. Compare using relative error
    numerator = np.linalg.norm(analytical_dW - numerical_dW)
    denominator = np.linalg.norm(analytical_dW) + np.linalg.norm(numerical_dW)
    difference = numerator / denominator
    
    print(f"Relative Difference: {difference}")
    if difference < 1e-7:
        print("Gradient check PASSED. Your calculus is flawless.")
    else:
        print("Gradient check FAILED. You have a bug in your backward pass.")

if __name__ == "__main__":
    gradient_check()