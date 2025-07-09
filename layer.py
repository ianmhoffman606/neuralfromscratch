import numpy as np

from neuron import Neuron

class Layer:

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = []
        for _ in range(output_size):
            self.neurons.append(Neuron(input_size))

    def layer_output(self, inputs: np.ndarray) -> np.ndarray:
        outputs: list[float] = []
        for neuron in self.neurons: 
            outputs.append(neuron.neuron_output(inputs)) 
        return np.array(outputs)