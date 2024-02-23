import numpy as np

inputs = [[1, 3, 6, 1], [1, 3, 6, 7], [15, 3, 6, 1]]
weights_1 = [[-1, 4, 2, 3], [3, 1, 7, 2], [3, 35, 1, 0]]
biases_1 = [2, 5, 1]

weights_2 = [[-1, 4, 2], [ 1, 7, 2], [ 35, 1, 0]]
biases_2 = [2, 5, 1]

layer_1_output = np.dot(inputs, np.array(weights_1).T) + biases_1
print(layer_1_output)
layer_2_output = np.dot(layer_1_output, np.array(weights_2).T) + biases_2
print((layer_2_output))
