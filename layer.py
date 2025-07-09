import numpy as np

from neuron import Neuron

class Layer:

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = []
        # initialize neurons for the layer
        for _ in range(output_size):
            self.neurons.append(Neuron(input_size))

        # store last inputs and outputs for backpropagation
        self.last_inputs = None
        self.last_outputs = None

    def layer_output(self, inputs: np.ndarray) -> np.ndarray:
        # store inputs for backpropagation
        self.last_inputs = np.array(inputs)

        outputs: list[float] = []
        # calculate output for each neuron in the layer
        for neuron in self.neurons: 
            outputs.append(neuron.neuron_output(inputs))

        # store outputs for backpropagation
        self.last_outputs = outputs
        
        return np.array(outputs)