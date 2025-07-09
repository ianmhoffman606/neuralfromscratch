import numpy as np
import matplotlib.pyplot as plt

from network import Network

# initialize the neural network
net = Network(input_size=1, hidden_layers_count=3, neurons_per_hidden_layer=50, output_size=1)

# set the number of training epochs
epochs = 800000
learning_rate = 0.001 # define learning rate

# train the network using backpropagation
for _ in range(epochs): # iterate over epochs
    input_val = (np.random.rand() * 6.0 * np.pi) - (3.0 * np.pi) # generate random input
    input_data = np.array(input_val)
    target_data = np.array([np.sin(input_val)])

    # Step 1: Perform a forward pass to get the prediction and set the internal state of all neurons
    predicted_output = net.forward_pass(input_data)

    # Step 2: Perform a backward pass using the prediction to update the weights
    net.back_pass(predicted_output, target_data, learning_rate, grad_clip_threshold=1.0)

def fix_sine(x):
    return [net.forward_pass(val)[0] for val in x]


# perform a forward pass with the input data to get the final output
x_values = np.linspace(-2.0 * np.pi, 2.0 * np.pi, 100) # 100 points between -2pi and 2pi
y_values = fix_sine(x_values)

real_sine_values = [np.sin(val) for val in x_values]

plt.plot(x_values, real_sine_values)
plt.plot(x_values, y_values)
plt.show()