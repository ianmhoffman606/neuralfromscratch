import numpy as np

class Neuron:

    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = 0.0

    def neuron_output(self, inputs: np.ndarray) -> float:
        total = np.dot(self.weights, inputs) + self.bias
        return self.activation_function(total)
    
    def activation_function(self, x: float) -> float:
        return np.tanh(x)