import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layers):
        
        for layer in layers:
            if hasattr(layer, 'W') and layer.W is not None:
                layer.W = layer.W - (self.lr * layer.dW)
                layer.b = layer.b - (self.lr * layer.db)

class Momentum:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocities = {}

    def update(self, layers):
        for layer in layers:
            if hasattr(layer, 'W') and layer.W is not None:
                if self.weight_decay > 0:
                    layer.dW += self.weight_decay * layer.W
                    
                if layer not in self.velocities:
                    self.velocities[layer] = {
                        'W': np.zeros_like(layer.W),
                        'b': np.zeros_like(layer.b)
                    }
                
                v = self.velocities[layer]
                v['W'] = (self.beta * v['W']) + ((1 - self.beta) * layer.dW)
                v['b'] = (self.beta * v['b']) + ((1 - self.beta) * layer.db)
                
                layer.W = layer.W - (self.lr * v['W'])
                layer.b = layer.b - (self.lr * v['b'])

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layers):
        self.t += 1
        
        for layer in layers:
            if hasattr(layer, 'W') and layer.W is not None:
                if self.weight_decay>0:
                    layer.dW+=self.weight_decay*layer.W
                if layer not in self.m:
                    self.m[layer] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                    self.v[layer] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                
                m = self.m[layer]
                v = self.v[layer]
                
                m['W'] = (self.beta1 * m['W']) + ((1 - self.beta1) * layer.dW)
                m['b'] = (self.beta1 * m['b']) + ((1 - self.beta1) * layer.db)
                
                v['W'] = (self.beta2 * v['W']) + ((1 - self.beta2) * np.square(layer.dW))
                v['b'] = (self.beta2 * v['b']) + ((1 - self.beta2) * np.square(layer.db))
                
                m_hat_W = m['W'] / (1 - (self.beta1 ** self.t))
                m_hat_b = m['b'] / (1 - (self.beta1 ** self.t))
                
                v_hat_W = v['W'] / (1 - (self.beta2 ** self.t))
                v_hat_b = v['b'] / (1 - (self.beta2 ** self.t))
                
                layer.W = layer.W - (self.lr * m_hat_W) / (np.sqrt(v_hat_W) + self.epsilon)
                layer.b = layer.b - (self.lr * m_hat_b) / (np.sqrt(v_hat_b) + self.epsilon)