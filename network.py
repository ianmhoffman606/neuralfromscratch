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
    
    def calculate_mse_loss_derivative(self, predicted_output: np.ndarray, target_output: np.ndarray) -> np.ndarray:

        return 2 * (predicted_output - target_output) / len(predicted_output)

    def back_pass(self, initial_input: np.ndarray, target_output: np.ndarray, learning_rate: float):

        predicted_output_array = self.forward_pass(initial_input)

        error_signal = self.calculate_mse_loss_derivative(predicted_output_array, target_output)
        
        deltas_raw_output_output_layer = []
        for i, neuron in enumerate(self.output_layer.neurons):

            neuron_output_error = error_signal[i]

            delta = neuron_output_error * neuron.activation_function_derivative(neuron.last_raw_output)
            deltas_raw_output_output_layer.append(delta)

            neuron.weights -= learning_rate * delta * neuron.last_inputs
            neuron.bias -= learning_rate * delta

        new_error_signal_for_prev_layer = np.zeros(self.output_layer.input_size)
        for i, neuron in enumerate(self.output_layer.neurons):
            new_error_signal_for_prev_layer += deltas_raw_output_output_layer[i] * neuron.weights
        error_signal = new_error_signal_for_prev_layer

        for layer_index in reversed(range(self.hidden_layers_count)): 
            current_layer = self.hidden_layers[layer_index] 

            deltas_raw_output_hidden_layer = []

            for i, neuron in enumerate(current_layer.neurons):

                neuron_output_error = error_signal[i] 

                delta = neuron_output_error * neuron.activation_function_derivative(neuron.last_raw_output)
                deltas_raw_output_hidden_layer.append(delta)

                neuron.weights -= learning_rate * delta * neuron.last_inputs
                neuron.bias -= learning_rate * delta
            
            if layer_index > 0:

                new_error_signal_for_prev_layer = np.zeros(current_layer.input_size)

                for i, neuron in enumerate(current_layer.neurons):

                    new_error_signal_for_prev_layer += deltas_raw_output_hidden_layer[i] * neuron.weights

                error_signal = new_error_signal_for_prev_layer 

        deltas_raw_output_input_layer = []
        for i, neuron in enumerate(self.input_layer.neurons):
            neuron_output_error = error_signal[i]
            delta = neuron_output_error * neuron.activation_function_derivative(neuron.last_raw_output)
            deltas_raw_output_input_layer.append(delta)

            neuron.weights -= learning_rate * delta * neuron.last_inputs
            neuron.bias -= learning_rate * delta