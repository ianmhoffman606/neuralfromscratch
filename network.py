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
    def calculate_mse_loss_derivative(self, predicted_output: np.ndarray, target_output: np.ndarray) -> np.ndarray:

        return 2 * (predicted_output - target_output) / len(predicted_output)

    # perform a backward pass (backpropagation) to update weights and biases
    def back_pass(self, predicted_output: np.ndarray, target_output: np.ndarray, learning_rate: float):
        # Ensure target_output is at least 1-dimensional
        target_output = np.atleast_1d(target_output)

        # --- backward pass ---
        # Start with the error at the output.
        # `predicted_output` is the result from the forward_pass call in the main loop.
        error = self.calculate_mse_loss_derivative(predicted_output, target_output)

        all_layers = [self.input_layer] + self.hidden_layers + [self.output_layer]

        # Propagate the error backward through the layers
        for i in reversed(range(len(all_layers))):
            current_layer = all_layers[i]
            
            # The input to the current layer was stored during the forward pass
            prev_layer_output = current_layer.last_inputs

            # Store the original weights before updating them
            original_weights = np.array([neuron.weights for neuron in current_layer.neurons])

            deltas = []
            for j, neuron in enumerate(current_layer.neurons):
                neuron_error = error[j]
                
                # The neuron's state (last_raw_output) is now correctly set from the forward_pass
                derivative = neuron.activation_function_derivative(neuron.last_raw_output)
                delta = neuron_error * derivative
                deltas.append(delta)
            
            deltas = np.array(deltas)

            # Calculate the error for the previous layer using the transposed original weights
            if i > 0:
                error = np.dot(original_weights.T, deltas)

            # Now, update the weights and biases for the current layer's neurons
            for j, neuron in enumerate(current_layer.neurons):
                delta = deltas[j]
                neuron.weights -= learning_rate * delta * prev_layer_output
                neuron.bias -= learning_rate * delta