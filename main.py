import numpy as np

from network import Network

# initialize the neural network
net = Network(input_size=1, hidden_layers_count=2, neurons_per_hidden_layer=25, output_size=1)

# set the number of training epochs
epochs = 5000
learning_rate = 0.25 # define learning rate

# train the network using backpropagation
input = -2.0 * np.pi
for _ in range(epochs): # iterate over epochs
    input += (4.0 * np.pi) / epochs 
    input_data = np.array(input)
    net.back_pass(input_data, np.sin(input), learning_rate) # perform backpropagation


# perform a forward pass with the input data to get the final output
x = np.pi / 2  # input value to predict what sin(x) is equal to
output = net.forward_pass(np.array(x))

print(f"input: {x}, predicted output: {output}, target output: {np.sin(x)}")