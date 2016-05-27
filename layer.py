"""
    Creator     : Jayrese Heslop
    Created on  : 5/26/2016 (11:08 P.M.)
    Last Editted: 5/26/2016 (11:12 P.M.)
"""

from neuron import Neuron

class Layer(object):
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for i in range(0, num_neurons, 1)]

    def process(self, inputs):
        return [neuron.process(inputs) for neuron in self.neurons]
