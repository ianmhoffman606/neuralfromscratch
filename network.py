import numpy as np
from layer import Layer

class Network:

    def __init__(self, input_size: int, hidden_layers_count: int, neurons_per_hidden_layer: int, output_size: int):

        self.input_size = input_size
        self.hidden_layers_count = hidden_layers_count
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.output_size = output_size

        # initialize input layer with Leaky ReLU
        self.input_layer = Layer(input_size, neurons_per_hidden_layer, activation_type='leaky_relu')

        # initialize hidden layers with Leaky ReLU
        self.hidden_layers = []
        for _ in range(hidden_layers_count):
            self.hidden_layers.append(Layer(neurons_per_hidden_layer, neurons_per_hidden_layer, activation_type='leaky_relu'))

        # initialize output layer with tanh
        self.output_layer = Layer(neurons_per_hidden_layer, output_size, activation_type='linear')

    # perform a forward pass through the network
    def forward_pass(self, initial_input: np.ndarray) -> np.ndarray:
        # Ensure input is at least 1-dimensional
        initial_input = np.atleast_1d(initial_input)

        current_output = self.input_layer.layer_output(initial_input)

        for layer in self.hidden_layers:
            current_output = layer.layer_output(current_output)

        final_output = self.output_layer.layer_output(current_output)
        return final_output

    # calculate the derivative of the mean squared error loss
    def calculate_mse_loss_derivative(self, predicted_output: np.ndarray, target_output: np.ndarray, ) -> np.ndarray:

        return 2 * (predicted_output - target_output) / len(predicted_output)

    # perform a backward pass (backpropagation) to update weights and biases
    def back_pass(self, predicted_output: np.ndarray, target_output: np.ndarray, learning_rate: float):
        target_output = np.atleast_1d(target_output)
        error = self.calculate_mse_loss_derivative(predicted_output, target_output)

        all_layers = [self.input_layer] + self.hidden_layers + [self.output_layer]

        for i in reversed(range(len(all_layers))):
            current_layer = all_layers[i]

            # Vectorized delta calculation
            if current_layer.last_raw_output is None:
                raise ValueError("last_raw_output is None. Ensure forward_pass is called before back_pass.")
            deltas = error * current_layer.activation_function_derivative(current_layer.last_raw_output)

            # Apply gradient clipping
            np.clip(deltas, -1.0, 1.0, out=deltas)

            # Get the output of the previous layer (or the initial input)
            if i == 0:
                prev_layer_output = current_layer.last_inputs
            else:
                raw_output = all_layers[i-1].last_raw_output
                if raw_output is None:
                    raise ValueError("last_raw_output is None. Cannot pass None to activation_function.")
                prev_layer_output = all_layers[i-1].activation_function(raw_output)

            if prev_layer_output is None:
                raise ValueError("prev_layer_output is None. Ensure forward_pass is called before back_pass.")

            # Calculate error for the previous layer
            # This must be done *before* updating the current layer's weights
            error = np.dot(current_layer.weights.T, deltas)

            # Vectorized weight and bias updates
            # Use np.outer to update the weight matrix
            current_layer.weights -= learning_rate * np.outer(deltas, prev_layer_output)
            current_layer.biases -= learning_rate * deltas