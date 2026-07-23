import numpy as np
from nn.data import load_mnist_images, load_mnist_labels
from nn.layers import Dense, Dropout
from nn.model import Sequential
from nn.optim import Adam

# 1. Load Data
X_train = load_mnist_images('train-images.idx3-ubyte')
Y_train = load_mnist_labels('train-labels.idx1-ubyte')
X_test = load_mnist_images('t10k-images.idx3-ubyte')
Y_test = load_mnist_labels('t10k-labels.idx1-ubyte')

# 2. Slice a tiny subset to force extreme overfitting
X_small = X_train[:, :2000]
Y_small = Y_train[:, :2000]
print(f"Dataset sliced. Training on {X_small.shape[1]} images. Testing on {X_test.shape[1]} images.")

def train_and_evaluate(model, name, epochs=40, weight_decay = 0.0):
    print(f"\n{'='*40}")
    print(f"--- Training {name} ---")
    print(f"{'='*40}")
    
    # Using Adam for rapid convergence on this tiny dataset
    optimizer = Adam(lr=0.005, weight_decay=weight_decay)
    batch_size = 32
    m = X_small.shape[1]
    
    for epoch in range(1, epochs + 1):
        # 1. TURN DROPOUT ON
        model.set_training(True)
        
        # Shuffle
        perm = np.random.permutation(m)
        X_shuf, Y_shuf = X_small[:, perm], Y_small[:, perm]
        
        for start in range(0, m, batch_size):
            X_batch = X_shuf[:, start:start+batch_size]
            Y_batch = Y_shuf[:, start:start+batch_size]
            model.train_step(X_batch, Y_batch, optimizer)
            
        if epoch <= 20 or epoch % 20 == 0:
            # 2. TURN DROPOUT OFF FOR FAIR TESTING
            model.set_training(False)
            
            # Train Accuracy (Did we memorize the 200 images?)
            train_preds = np.argmax(model.forward(X_small), axis=0)
            train_truth = np.argmax(Y_small, axis=0)
            train_acc = np.mean(train_preds == train_truth) * 100
            
            # Test Accuracy (Did we actually learn what a digit looks like?)
            test_preds = np.argmax(model.forward(X_test), axis=0)
            test_truth = np.argmax(Y_test, axis=0)
            test_acc = np.mean(test_preds == test_truth) * 100
            
            gap = train_acc - test_acc
            print(f"Epoch {epoch:03d} | Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}% | Gap: {gap:6.2f}%")

model_a = Sequential([Dense(784, 128, 'relu'), Dense(128, 10, 'softmax')])
train_and_evaluate(model_a, "Model A (No Regularization)", weight_decay=0.0)

model_b = Sequential([Dense(784, 128, 'relu'), Dropout(p=0.3), Dense(128, 10, 'softmax')])
train_and_evaluate(model_b, "Model B (Dropout only)", weight_decay=0.0)

model_c = Sequential([Dense(784, 128, 'relu'), Dense(128, 10, 'softmax')])
train_and_evaluate(model_c, "Model C (L2 only)", weight_decay=0.01)

