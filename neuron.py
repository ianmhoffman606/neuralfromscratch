import numpy as np

class Neuron:

    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.01  # initialize weights randomly
        self.bias = 0.0 # initialize bias to zero

        self.last_inputs = None # store last inputs for backpropagation
        self.last_raw_output = None # store raw output before activation for backpropagation

    def neuron_output(self, inputs: np.ndarray) -> float:
        self.last_inputs = np.array(inputs) # store inputs

        total = np.dot(self.weights, inputs) + self.bias # calculate weighted sum of inputs plus bias

        self.last_raw_output = total # store raw output

        return self.activation_function(total) # apply activation function
    
    def activation_function(self, x: float) -> float:
        return np.tanh(x) # tanh activation function
    
    def activation_function_derivative(self, x: float) -> float:
        return 1 - np.tanh(x)**2 # derivative of tanh(x)