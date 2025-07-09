import numpy as np

class Neuron:

    def __init__(self, input_size: int, activation_type: str = 'tanh'):
        self.weights = np.random.randn(input_size) * np.sqrt(2.0 / input_size)  # initialize weights randomly
        self.bias = 0.0 # initialize bias to zero

        self.last_inputs = None # store last inputs for backpropagation
        self.last_raw_output = None # store raw output before activation for backpropagation

        # Set activation function based on type
        # In Neuron.__init__
        if activation_type == 'leaky_relu':
            self.activation_function = self.leaky_relu
            self.activation_function_derivative = self.leaky_relu_derivative
        elif activation_type == 'tanh':
            self.activation_function = self.tanh
            self.activation_function_derivative = self.tanh_derivative
        elif activation_type == 'linear':
            self.activation_function = self.linear
            self.activation_function_derivative = self.linear_derivative
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")

    def neuron_output(self, inputs: np.ndarray) -> float:
        self.last_inputs = np.array(inputs) # store inputs

        total = np.dot(self.weights, inputs) + self.bias # calculate weighted sum of inputs plus bias

        self.last_raw_output = total # store raw output

        return self.activation_function(total) # apply activation function

    def tanh(self, x: float) -> float:
        return np.tanh(x) # tanh activation function

    def tanh_derivative(self, x: float) -> float:
        return 1 - np.tanh(x)**2 # derivative of tanh(x)

    def leaky_relu(self, x: float, alpha: float = 0.01) -> float:
        return np.maximum(alpha * x, x) # Leaky ReLU activation function

    def leaky_relu_derivative(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, 1.0, alpha)
    
    def linear(self, x: float) -> float:
        return x

    def linear_derivative(self, x: float) -> float:
        return 1.0