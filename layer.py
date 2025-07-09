import numpy as np

from neuron import Neuron

class Layer:

    def __init__(self, input_size: int, output_size: int, activation_type: str = 'tanh'):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = []
        # initialize neurons for the layer with specified activation type
        for _ in range(output_size):
            self.neurons.append(Neuron(input_size, activation_type)) # Pass activation_type

        # store last inputs and outputs for backpropagation
        self.last_inputs = None
        self.last_outputs = None

    def layer_output(self, inputs: np.ndarray) -> np.ndarray:
        self.last_inputs = np.array(inputs)

        weights_matrix = np.array([neuron.weights for neuron in self.neurons])
        biases_vector = np.array([neuron.bias for neuron in self.neurons])

        raw_outputs = np.dot(weights_matrix, inputs) + biases_vector

        # --- FIX: Store the raw output in each neuron for backpropagation ---
        for i, neuron in enumerate(self.neurons):
            neuron.last_raw_output = raw_outputs[i]
        # --------------------------------------------------------------------

        self.last_outputs = [self.neurons[i].activation_function(raw_outputs[i]) for i in range(self.output_size)]

        return np.array(self.last_outputs)