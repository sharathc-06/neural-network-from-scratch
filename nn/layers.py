import numpy as np
class Neuron:
    def __init__(self, n_inputs):
        # initialize self.w (shape n_inputs,) and self.b (scalar)
        self.w = np.random.randn(n_inputs) * 0.01
        self.b=0  

    def forward(self, x):
        """x: shape (n_inputs,) -> returns scalar activation"""
        z = np.dot(self.w, x) +self.b
        if z >= 0:
            a = 1.0 / (1.0 + np.exp(-z))
        else:
            a = np.exp(z) / (1.0 + np.exp(z))
        return a

    def backward(self, x, y, a, lr):
        """Compute gradients and update self.w, self.b in place."""
        dw = (a - y) * x
        db = a - y
        self.w = self.w - lr * dw
        self.b = self.b - lr * db

class Dense:
    def __init__(self, n_in, n_out, activation="relu"):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        # He initialization: scales variance for ReLU networks
        self.W = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
        self.b = np.zeros((n_out, 1))

        # caches, filled in during forward(), used during backward()
        self.A_prev = None
        self.Z = None

        # gradients, filled in during backward(), used by the optimizer
        self.dW = None
        self.db = None
    def forward(self, A_prev):
        """A_prev: shape (n_in, m) -> returns A, shape (n_out, m)"""
        self.A_prev = A_prev
        self.Z = self.W @ A_prev + self.b

        if self.activation == "relu":
            self.A = np.maximum(0, self.Z)
        elif self.activation == "softmax":
            # numerically stable softmax: subtract max before exp
            shift = self.Z - np.max(self.Z, axis=0, keepdims=True)
            exp = np.exp(shift)
            self.A = exp / np.sum(exp, axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        return self.A
    def backward(self, dA):
        """dA: shape (n_out, m) -> returns dA_prev, shape (n_in, m)"""
        m = self.A_prev.shape[1]

        if self.activation == "relu":
            dZ = dA * (self.Z > 0).astype(float)
        elif self.activation == "softmax":
            # The derivative was already computed by the loss function shortcut (A - Y).
            # So dA is already exactly dZ. We just pass it through.
            dZ = dA
        else:
            raise ValueError(f"backward() doesn't support activation: {self.activation}")
        
        self.dW = (1/m) * dZ @ self.A_prev.T
        self.db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = self.W.T @ dZ

        return dA_prev
class Dropout:
    def __init__(self, p=0.5):
        """p: probability of dropping a neuron"""
        self.p = p
        self.mask = None
        self.training = True  

    def forward(self, A_prev):
        if not self.training:
            return A_prev
            
        keep_prob = 1.0 - self.p
        self.mask = (np.random.rand(*A_prev.shape) < keep_prob).astype(float)
        
        self.mask /= keep_prob
        
        return A_prev * self.mask

    def backward(self, dA):
        if not self.training:
            return dA
            
        return dA * self.mask