#Neural Network in Python (tested with version 3.4.0)
##Usage
```python
from layer import Layer
from neuron import Neuron
from network import Network

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
ITERATIONS = 500  # Number of iterations per training session
LEARN_RATE = 0.1  # The rate the network learns on each iteration
THRESHOLD  = 0.01 # If this precision is reached, the training session is instantly complete

# Perform a quick training session
for i in range(0, ITERATIONS, 1):
    error = 0
    
    #        Examples      Inputs   Outputs Learn Rate
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

print(outputs)
```
##Credits
Made by Jayrese Heslop