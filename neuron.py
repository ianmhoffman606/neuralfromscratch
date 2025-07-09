import numpy as np

class Neuron:

    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = 0.0

        self.last_inputs = None
        self.last_raw_output = None

    def neuron_output(self, inputs: np.ndarray) -> float:
        self.last_inputs = np.array(inputs)

        total = np.dot(self.weights, inputs) + self.bias

        self.last_raw_output = total

        return self.activation_function(total)
    
    def activation_function(self, x: float) -> float:
        return np.tanh(x)
    
    def activation_function_derivative(self, x: float) -> float:
        return 1 - np.tanh(x)**2