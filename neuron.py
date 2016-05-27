"""
    Creator     : Jayrese Heslop
    Created on  : 5/26/2016 (10:57 P.M.)
    Last Editted: 5/26/2016 (11:07 P.M.)
"""

import random
import math

class Neuron(object):
    WEIGHT_RANGE = 0.2
    BIAS_RANGE = 0.2
    
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-Neuron.WEIGHT_RANGE, Neuron.WEIGHT_RANGE) for i in range(0, num_inputs, 1)]
        self.bias = random.uniform(-Neuron.BIAS_RANGE, Neuron.BIAS_RANGE)
        self.last_output = None
        self.last_input = None
        self.error = None
        self.delta = None

    def process(self, inputs):
        self.last_input = inputs
        self.last_output = sum([self.weights[i] * inputs[i] for i in range(0, len(inputs), 1)])
        self.last_output += self.bias
        self.last_output = 1.0 / (1.0 + pow(math.e, -self.last_output))
        
        return self.last_output
