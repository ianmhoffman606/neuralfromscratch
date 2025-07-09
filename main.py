import numpy as np

from network import Network

net = Network(1, 2, 25, 1)

input = np.array(0.0)

output = net.forward_pass(input)

print(output)