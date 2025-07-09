import numpy as np

from network import Network

# initialize the neural network
net = Network(input_size=1, hidden_layers_count=2, neurons_per_hidden_layer=25, output_size=1)

# set the number of training epochs
epochs = 10000
learning_rate = 0.05 # define learning rate

# train the network using backpropagation
for _ in range(epochs): # iterate over epochs
    input = (np.random.rand() *  4.0 * np.pi) - (2.0 * np.pi) # generate random input
    input_data = np.array(input)
    net.back_pass(input_data, np.sin(input), learning_rate) # perform backpropagation


# perform a forward pass with the input data to get the final output
x = np.pi / 6  # input value to predict what sin(x) is equal to
output = net.forward_pass(np.array(x))

print(f"input: {x}, predicted output: {output}, target output: {np.sin(x)}")