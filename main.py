import numpy as np
from layer import Layer

input = np.array(0.0)
hiddens = 2
neuron_per_hidden = 25

input_layer = Layer(1, neuron_per_hidden)

hidden_layers = []
for _ in range(hiddens - 1):
    hidden_layers.append(Layer(neuron_per_hidden, neuron_per_hidden))

for layer in hidden_layers:
    input = layer.layer_output(input)

output_layer = Layer(neuron_per_hidden, 1)

print(output_layer.layer_output(input))