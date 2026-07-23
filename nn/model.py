from nn.losses import cross_entropy_loss
class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, X):
        current_data = X
        for layer in self.layers:
            current_data = layer.forward(current_data)
        return current_data
    def backward(self, dA):
        current_blame = dA
        for layer in reversed(self.layers):
            current_blame=layer.backward(current_blame)
        return current_blame
    def train_step(self, X, Y, optimizer):
        A_last = self.forward(X)
        loss, dA_last = cross_entropy_loss(A_last, Y)
        self.backward(dA_last)
        optimizer.update(self.layers)
        return loss
    def set_training(self, mode):
        for layer in self.layers:
           if hasattr(layer, 'training'):
              layer.training = mode