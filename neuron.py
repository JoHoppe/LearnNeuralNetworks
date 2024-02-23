import nnfs
import numpy as np
from nnfs.datasets import spiral_data

nnfs.init()


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases with random values
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Calculate output from input, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# Create dataset
x, y = spiral_data(samples=100, classes=3)

# Create layer
layer1 = LayerDense(2, 3)
layer1.forward(x)
layer2 = LayerDense(3, 1)
layer2.forward((layer1.output))
print(layer1.output[:5])
print(layer2.output[:5])


def relu(inputs):
    return np.maximum(0,inputs)

print(relu(layer1.output[:5]))