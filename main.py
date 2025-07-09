import numpy as np

from network import Network

net = Network(1, 2, 25, 1)

input = np.array(0.0)

epochs = 500

for _ in range(epochs):
    net.back_pass(input, np.sin(0.0), 1.0)

output = net.forward_pass(input)

print(output)