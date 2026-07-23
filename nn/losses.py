import numpy as np

def softmax(Z):
    shift_Z = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(shift_Z)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return A

def cross_entropy_loss(A_last, Y):
    epsilon = 1e-8
    A_clipped = np.clip(A_last, epsilon, 1.0 - epsilon)
    loss = -np.mean(np.sum(Y * np.log(A_clipped), axis=0))
    dA_last = A_last - Y 
    return loss, dA_last