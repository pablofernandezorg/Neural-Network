"""
    Creator     : Jayrese Heslop
    Created on  : 5/26/2016 (11:13 P.M.)
    Last Editted: 5/26/2016 (12:40 A.M.)
"""

from layer import Layer
from neuron import Neuron

class Network(object):
    # Number of iterations per training session
    ITERATIONS = 1000

    # The rate the network learns on each iteration
    LEARN_RATE = 0.5
    
    def __init__(self):
        self.layers = []

    def process(self, _inputs):
        # TODO: Come up with elegant solution using reduce
        outputs = None
        inputs = _inputs
        
        for layer in self.layers:
            outputs = layer.process(inputs)
            inputs = outputs

        return outputs

    def add_layer(self, num_neurons, num_inputs=None):
        if num_inputs == None:
            last_layer = self.layers[-1]
            num_inputs = len(last_layer.neurons)

        self.layers.append(Layer(num_neurons, num_inputs))

    def train(self, examples):
        output_layer = self.layers[-1]
        
        # For every training session ...
        for i in range(0, Network.ITERATIONS, 1):
            # For every example given ...
            for j in range(0, len(examples), 1):
                example = examples[j]
                inputs = example[0]
                targets = example[1]

                # Run the example through the neural network
                outputs = self.process(inputs)

                # For every neuron in the output layer ...
                for k in range(0, len(output_layer.neurons), 1):
                    # Calculate the error of the neuron
                    neuron = output_layer.neurons[k]
                    neuron.error = targets[k] - outputs[k]

                    # Use newtons method for later improvement of the result
                    neuron.delta = neuron.last_output * (1.0 - neuron.last_output) * neuron.error
                
                # For every other layer (backwards) ...
                for k in range(len(self.layers) - 2, 0 - 1, -1):
                    layer = self.layers[k]
                    layer_next = self.layers[k + 1]
                    
                    # For every neuron in this layer ...
                    for l in range(0, len(layer.neurons), 1):
                        # Calculate the error on the neuron
                        neuron_l = layer.neurons[l]
                        neuron_l.error = sum([neuron.weights[l] * neuron.delta for neuron in layer_next.neurons])

                        # Use newtons method for later improvement of the result
                        neuron_l.delta = neuron_l.last_output * (1.0 - neuron_l.last_output) * neuron_l.error

                        # For every neuron in the next layer
                        for m in range(0, len(layer_next.neurons), 1):
                            neuron_m = layer_next.neurons[m]

                            # For all the weights of the neuron
                            for n in range(0, len(neuron_m.weights), 1):
                                # Improve the neuron by a factor of the learning rate
                                neuron_m.weights[n] += Network.LEARN_RATE * neuron_m.last_input[n] * neuron_m.delta

                            # Improve this neuron using the previous newtons method information
                            neuron_m.bias += Network.LEARN_RATE * neuron_m.delta
