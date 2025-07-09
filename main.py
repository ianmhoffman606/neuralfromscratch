import numpy as np

from network import Network

# initialize the neural network
net = Network(input_size=1, hidden_layers_count=2, neurons_per_hidden_layer=25, output_size=1)

# define the input for the network
input_data = np.array(0.0)

# set the number of training epochs
epochs = 5000
learning_rate = 0.01 # define learning rate

# train the network using backpropagation
for _ in range(epochs): # iterate over epochs
    net.back_pass(input_data, np.sin(input_data), learning_rate) # perform backpropagation

# perform a forward pass with the input data to get the final output
output = net.forward_pass(input_data)

print(f"input: {input_data}, predicted output: {output}, target output: {np.sin(input_data)}")