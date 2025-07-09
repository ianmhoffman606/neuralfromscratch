import numpy as np
from layer import Layer

class Network:

    def __init__(self, input_size: int, hidden_layers_count: int, neurons_per_hidden_layer: int, output_size: int):

        self.input_size = input_size
        self.hidden_layers_count = hidden_layers_count
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.output_size = output_size

        self.input_layer = Layer(input_size, neurons_per_hidden_layer)

        self.hidden_layers = []
        for _ in range(hidden_layers_count):
            self.hidden_layers.append(Layer(neurons_per_hidden_layer, neurons_per_hidden_layer))

        self.output_layer = Layer(neurons_per_hidden_layer, output_size)

    def forward_pass(self, initial_input: np.ndarray) -> np.ndarray:

        current_output = self.input_layer.layer_output(initial_input)

        for layer in self.hidden_layers:
            current_output = layer.layer_output(current_output)

        final_output = self.output_layer.layer_output(current_output)
        return final_output