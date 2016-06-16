"""
    Creator     : Jayrese Heslop
    Created on  : 5/26/2016 (11:13 P.M.)
    Last Editted: 6/16/2016 (01:38 P.M.)
"""

from layer import Layer
from neuron import Neuron
from network import Network

try:
    # Attempt to load the network from a file
    network = Network.load("test_binary.json")
except Exception as e:
    # On failure, recreate the network from scratch
    network = Network()
    network.add_layer(10, 20, Network.ACTIVATION_SIGMOID) # Hidden Layer, 10 Neurons, 20 inputs
    network.add_layer(2,  10, Network.ACTIVATION_SIGMOID) # Output Layer,  2 Neurons, 10 inputs

# Simulate black and white images
# 0 - Black
# 1 - White
zero = [
    0, 1, 1, 0,
    1, 0, 0, 1,
    1, 0, 0, 1,
    1, 0, 0, 1,
    0, 1, 1, 0
]

one = [
    0, 0, 1, 0,
    0, 0, 1, 0,
    0, 0, 1, 0,
    0, 0, 1, 0,
    0, 0, 1, 0
]

two = [
    0, 1, 1, 0,
    1, 0, 0, 1,
    0, 0, 1, 0,
    0, 1, 0, 0,
    1, 1, 1, 1
]

three = [
    1, 1, 1, 1,
    0, 0, 0, 1,
    0, 1, 1, 1,
    0, 0, 0, 1,
    1, 1, 1, 1
]

# Set some quick properties for the upcoming training session
ITERATIONS = 1000  # Number of iterations per training session
LEARN_RATE = 0.03  # The rate the network learns on each iteration
THRESHOLD  = 0.001 # If this precision is reached, the training session is instantly complete

# Perform a quick training session
for i in range(0, ITERATIONS, 1):
    error = 0
    
    # Example Data         Inputs   Outputs Learn Rate
    error += network.train(zero,    [0, 0], LEARN_RATE)
    error += network.train(one,     [0, 1], LEARN_RATE)
    error += network.train(two,     [1, 0], LEARN_RATE)
    error += network.train(three,   [1, 1], LEARN_RATE)

    # Check the error
    if error < THRESHOLD:
        break

# Process the outputs
outputs = network.process([
    0, 1, 1, 0,
    1, 0, 0, 1,
    0, 0, 1, 0,
    0, 1, 0, 0,
    1, 1, 1, 1
])

# Save the neural network to a file
Network.save("test_binary.json", network)

# Print the decimal equivalent of the network's output (Expected: ~2)
print(sum([(2 ** i) * outputs[-i - 1] for i in range(0, len(outputs), 1)]))
