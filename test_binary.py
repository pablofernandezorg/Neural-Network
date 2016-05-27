"""
    Creator     : Jayrese Heslop
    Created on  : 5/26/2016 (11:13 P.M.)
    Last Editted: 5/26/2016 (11:50 P.M.)
"""

from layer import Layer
from neuron import Neuron
from network import Network

import math

network = Network()
network.add_layer(10, 20) # Hidden Layer, 10 Neurons, 20 inputs
network.add_layer(2)      # Output Layer,  2 Neurons

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
Network.ITERATIONS = 1000
Network.LEARN_RATE = 1.0

# Perform a quick training session
network.train([
    # Inputs | Outputs
    [zero,     [0, 0]],
    [one,      [0, 1]],
    [two,      [1, 0]],
    [three,    [1, 1]]
])

# Process the outputs
outputs = network.process([
    1, 1, 1, 1,
    1, 0, 0, 1,
    1, 0, 0, 1,
    1, 0, 0, 1,
    1, 1, 1, 0
])

print(list(map(math.floor, outputs)))
