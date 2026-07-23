import numpy as np
from nn.data import load_mnist_images, load_mnist_labels
from nn.model import Sequential
from nn.layers import Dense, MaxPool2D, Flatten
from nn.conv import Conv2D
from nn.optim import Adam

X_train_flat = load_mnist_images('train-images.idx3-ubyte')  # (784, 60000)
Y_train = load_mnist_labels('train-labels.idx1-ubyte')
X_test_flat = load_mnist_images('t10k-images.idx3-ubyte')
Y_test = load_mnist_labels('t10k-labels.idx1-ubyte')

# Reshape from (784, m) back into (m, 1, 28, 28) for Conv2D
def to_images(X_flat):
    return X_flat.T.reshape(-1, 1, 28, 28)

# Use a smaller subset for speed
n_train = 5000
X_train = to_images(X_train_flat[:, :n_train])
Y_train = Y_train[:, :n_train]
X_test = to_images(X_test_flat[:, :1000])   # smaller test set too, for speed
Y_test = Y_test[:, :1000]

model = Sequential([
    Conv2D(in_channels=1, n_filters=8, f=3),
    MaxPool2D(pool_size=2),
    Flatten(),
    Dense(1352, 64, activation='relu'),
    Dense(64, 10, activation='softmax'),
])
optimizer = Adam(lr=0.001)

batch_size = 32
m = X_train.shape[0]

for epoch in range(5):
    perm = np.random.permutation(m)
    X_shuf, Y_shuf = X_train[perm], Y_train[:, perm]

    epoch_loss = 0
    for start in range(0, m, batch_size):
        X_batch = X_shuf[start:start+batch_size]
        Y_batch = Y_shuf[:, start:start+batch_size]
        loss = model.train_step(X_batch, Y_batch, optimizer)
        epoch_loss += loss * X_batch.shape[0]
        batch_num = start // batch_size
        if batch_num % 10 == 0:
            print(f"  batch {batch_num}/{m//batch_size} | loss: {loss:.4f}")
    epoch_loss /= m

    preds = np.argmax(model.forward(X_test), axis=0)
    truth = np.argmax(Y_test, axis=0)
    acc = np.mean(preds == truth) * 100
    print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Test Acc: {acc:.2f}%")