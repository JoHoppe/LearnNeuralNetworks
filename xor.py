import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Mean squared error loss function
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Forward pass through the network
def forward_pass(input_data, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    # Hidden layer computation
    hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Output layer computation
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)

    return hidden_layer_output, output

# Backpropagation algorithm to compute gradients
def backpropagation(input_data, target, hidden_output, output, weights_hidden_output):
    # Compute gradients for output layer
    output_error = (output - target) * sigmoid_derivative(output)
    hidden_output_transpose = hidden_output.reshape(-1, 1)
    gradient_weights_hidden_output = np.dot(hidden_output_transpose, output_error)
    gradient_bias_output = output_error

    # Compute gradients for hidden layer
    hidden_error = np.dot(output_error, weights_hidden_output.T) * sigmoid_derivative(hidden_output)
    input_data_transpose = input_data.reshape(-1, 1)
    gradient_weights_input_hidden = np.dot(input_data_transpose, hidden_error)
    gradient_bias_hidden = hidden_error

    return gradient_weights_input_hidden, gradient_bias_hidden, gradient_weights_hidden_output, gradient_bias_output

# Update weights and biases using gradient descent
def update_parameters(weights_input_hidden, bias_hidden, weights_hidden_output, bias_output,
                      gradient_weights_input_hidden, gradient_bias_hidden,
                      gradient_weights_hidden_output, gradient_bias_output,
                      learning_rate):
    weights_input_hidden -= learning_rate * gradient_weights_input_hidden
    bias_hidden -= learning_rate * gradient_bias_hidden
    weights_hidden_output -= learning_rate * gradient_weights_hidden_output
    bias_output -= learning_rate * gradient_bias_output

    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

# Define neural network architecture
input_size = 2
hidden_size = 3
output_size = 1

# Initialize weights and biases
weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.random.randn(hidden_size)
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_output = np.random.randn(output_size)

# Define input data and target
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0], [1], [1], [0]])

# Training parameters
epochs = 10000
learning_rate = 0.1

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_output, output = forward_pass(input_data, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

    # Compute loss
    loss = mean_squared_error(target, output)

    # Backpropagation
    gradient_weights_input_hidden, gradient_bias_hidden, gradient_weights_hidden_output, gradient_bias_output = backpropagation(
        input_data, target, hidden_output, output, weights_hidden_output)

    # Update parameters
    weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = update_parameters(
        weights_input_hidden, bias_hidden, weights_hidden_output, bias_output,
        gradient_weights_input_hidden, gradient_bias_hidden,
        gradient_weights_hidden_output, gradient_bias_output,
        learning_rate)

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Final predictions
_, predictions = forward_pass(input_data, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
print("Final predictions:")
print(predictions)
