import numpy as np
from nn.data import load_mnist_images, load_mnist_labels
from nn.model import Sequential
from nn.optim import SGD,Momentum,Adam
from nn.layers import Dense, Dropout

X_train = load_mnist_images('train-images.idx3-ubyte')
Y_train = load_mnist_labels('train-labels.idx1-ubyte')
X_test = load_mnist_images('t10k-images.idx3-ubyte')
Y_test = load_mnist_labels('t10k-labels.idx1-ubyte')

model = Sequential([
    Dense(784, 128, activation='relu'),
    Dropout(p=0.2), 
    Dense(128, 64, activation='relu'),
    Dense(64, 10, activation='softmax'),
])
optimizer = Adam(lr=0.001)

batch_size = 128
m = X_train.shape[1]

for epoch in range(20):
    model.set_training(True)
    # 1. Create the bucket to accumulate total loss for this epoch
    epoch_loss = 0
    
    # shuffle each epoch
    perm = np.random.permutation(m)
    X_shuf, Y_shuf = X_train[:, perm], Y_train[:, perm]

    for start in range(0, m, batch_size):
        X_batch = X_shuf[:, start:start+batch_size]
        Y_batch = Y_shuf[:, start:start+batch_size]
        
        # 2. Get the mean loss of THIS specific batch
        batch_loss = model.train_step(X_batch, Y_batch, optimizer)
        
        # 3. Multiply by the actual number of items in this batch to get raw error, then accumulate
        epoch_loss += batch_loss * X_batch.shape[1]

    # 4. Divide by total dataset size to get the true average epoch loss
    epoch_loss /= m

    model.set_training(False)
    
    preds = np.argmax(model.forward(X_test), axis=0)
    truth = np.argmax(Y_test, axis=0)
    acc = np.mean(preds == truth) * 100
    
    print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | Test Accuracy: {acc:.2f}%")