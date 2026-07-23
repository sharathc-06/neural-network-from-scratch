import numpy as np
import struct

def load_mnist_images(path):
    with open(path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num_images, rows * cols)
        images = images.astype(np.float32) / 255.0
        return images.T

def load_mnist_labels(path):
    with open(path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        one_hot = np.zeros((10, num_labels), dtype=np.float32)
        one_hot[labels, np.arange(num_labels)] = 1.0
        
        return one_hot