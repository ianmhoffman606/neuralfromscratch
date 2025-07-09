import numpy as np

class Layer:
    def __init__(self, input_size: int, output_size: int, activation_type: str = 'tanh'):
        # Instead of a list of neurons, store weights as a matrix and biases as a vector.
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros(output_size)
        self.output_size = output_size

        # Store last inputs and outputs for backpropagation
        self.last_inputs = None
        self.last_raw_output = None # To store the output before activation

        # Set activation functions
        if activation_type == 'leaky_relu':
            self.activation_function = self.leaky_relu
            self.activation_function_derivative = self.leaky_relu_derivative
        elif activation_type == 'tanh':
            self.activation_function = self.tanh
            self.activation_function_derivative = self.tanh_derivative
        elif activation_type == 'linear':
            self.activation_function = self.linear
            self.activation_function_derivative = self.linear_derivative
        
    def layer_output(self, inputs: np.ndarray) -> np.ndarray:
        self.last_inputs = inputs

        # Perform the entire layer calculation with one matrix operation
        raw_outputs = np.dot(self.weights, inputs) + self.biases
        self.last_raw_output = raw_outputs # Store for backpropagation

        # Apply the activation function to the entire vector at once
        return self.activation_function(raw_outputs)

    # --- Add activation functions directly to the Layer class ---
    def tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x)**2

    def leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.maximum(alpha * x, x)

    def leaky_relu_derivative(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, 1.0, alpha)

    def linear(self, x: np.ndarray) -> np.ndarray:
        return x

    def linear_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)