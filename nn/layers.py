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

class MaxPool2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.X_batch = None  # Cache for the backward pass

    def forward(self, X_batch):
        """
        X_batch: shape (m, C, H, W)
        Returns: shape (m, C, out_H, out_W)
        """
        # 1. Cache the input for the backward pass!
        self.X_batch = X_batch
        
        # 2. Extract dimensions
        m, C, H, W = X_batch.shape
        out_H = H // self.pool_size
        out_W = W // self.pool_size
        
        # 3. Initialize the output volume
        output = np.zeros((m, C, out_H, out_W))
        
        # 4. Loop over every image in the batch
        for b in range(m):
            # Loop over every channel in the image
            for c in range(C):
                
                # EXACTLY your logic, just applied to X_batch[b, c]
                for i in range(out_H):
                    for j in range(out_W):
                        r_start = i * self.pool_size
                        r_end = (i + 1) * self.pool_size
                        c_start = j * self.pool_size
                        c_end = (j + 1) * self.pool_size
                        
                        # Extract the window
                        window = X_batch[b, c, r_start:r_end, c_start:c_end]
                        
                        # Find the max and store it
                        output[b, c, i, j] = np.max(window)
                        
        return output

    def backward(self, dOut_batch):
        """
        dOut_batch: shape (m, C, out_H, out_W)
        Returns: dX_batch, shape (m, C, H, W)
        """
        # 1. Extract dimensions from our cached input
        m, C, H, W = self.X_batch.shape
        out_H, out_W = dOut_batch.shape[2], dOut_batch.shape[3]
        
        # 2. Initialize the gradient volume for the input
        dX_batch = np.zeros((m, C, H, W))
        
        # 3. Loop over every image and channel
        for b in range(m):
            for c in range(C):
                
                # EXACTLY your backward logic
                for i in range(out_H):
                    for j in range(out_W):
                        r_start = i * self.pool_size
                        r_end = (i + 1) * self.pool_size
                        c_start = j * self.pool_size
                        c_end = (j + 1) * self.pool_size
                        
                        # Look at the original window from the forward pass
                        window = self.X_batch[b, c, r_start:r_end, c_start:c_end]
                        
                        # Create the mask (True where the max was, False elsewhere)
                        mask = (window == np.max(window))
                        
                        # Route the incoming gradient ONLY to the max pixel(s)
                        dX_batch[b, c, r_start:r_end, c_start:c_end] = mask * dOut_batch[b, c, i, j]
                        
        return dX_batch

class Flatten:
    def __init__(self):
        # We must cache the original shape so we know exactly how to 
        # "un-flatten" the gradients during the backward pass!
        self.orig_shape = None  

    def forward(self, X_batch):
        """
        X_batch: shape (m, C, H, W)
        Returns: shape (C*H*W, m) - perfectly formatted for a Dense layer!
        """
        # 1. Cache the original shape (m, C, H, W)
        self.orig_shape = X_batch.shape
        m, C, H, W = self.orig_shape
        
        # 2. Flatten the spatial and channel dimensions for each image 
        # Using -1 tells numpy to automatically compute C * H * W
        # Shape becomes: (m, C*H*W)
        X_reshaped = X_batch.reshape(m, -1)
        
        # 3. Transpose so the batch dimension is last
        # Shape becomes: (C*H*W, m)
        X_flat = X_reshaped.T
        
        return X_flat

    def backward(self, dOut):
        """
        dOut: shape (C*H*W, m) - gradient flowing backward from the Dense layer
        Returns: shape (m, C, H, W) - gradient mapped back to physical space
        """
        # 1. Transpose back to put the batch dimension first
        # Shape becomes: (m, C*H*W)
        dX_reshaped = dOut.T
        
        # 2. Reshape the flat lists back into the original 4D volumes!
        # Shape becomes: (m, C, H, W)
        dX_batch = dX_reshaped.reshape(self.orig_shape)
        
        return dX_batch