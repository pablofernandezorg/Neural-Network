"""
    Creator     : Jayrese Heslop
    Created on  : 5/26/2016 (11:13 P.M.)
    Last Editted: 6/11/2016 (09:05 A.M.)
"""

from layer import Layer
from neuron import Neuron

import json

class Network(object):
    """
        A neural network which contains multiple layers
        Note:
            The first layer is called the input layer
            The last layer is called the output layer
            All the layers inbetween are called hidden layers
        
        @property (Array<Layer>) layers - The layers of the neural network
    """
    def __init__(self):
        """
            Initializes the neural network
            @param (Network) self - The neural network to initialize
        """
        self.layers = []

    def process(self, _inputs):
        """
            Processes a set of inputs (the length of inputs must be the same as the number of inputs for the input layer)
            @param (Network) self - The neural network to process the inputs with
            @param (Array<Number>) _inputs - The inputs for the input layer
            @returns (Array<Number>) - The output of the output layer
        """
        outputs = None
        inputs = _inputs
        
        for layer in self.layers:
            outputs = layer.process(inputs)
            inputs = outputs

        return outputs

    def add_layer(self, num_neurons, num_inputs=None):
        """
            Adds a layer to the neural network
            @param (Network) self - The network to add the layer to
            @param (Number) - The number of neurons of the layer
            @param (Number) [num_inputs=None] - The number of inputs of the layer

            Note:
                If num_inputs is not specified it is the same as the number of inputs of the last layer added
        """
        if num_inputs == None:
            last_layer = self.layers[-1]
            num_inputs = len(last_layer.neurons)

        self.layers.append(Layer(num_neurons, num_inputs))
    
    def train(self, inputs, targets, learn_rate):
        """
            Trains the neural network with an example
            @param (Network) self - The network to train
            @param (Array<Number>) inputs - The inputs of the example
            @param (Array<Number>) targets - The target values of the example
            @param (Number) learn_rate - The rate to learn at
            @returns (Number) - The mean squared error
        """
        output_layer = self.layers[-1]
        error = 0
        
        # Run the example through the neural network
        outputs = self.process(inputs)

        # For every neuron in the output layer ...
        for k in range(0, len(output_layer.neurons), 1):
            # Calculate the error of the neuron
            neuron = output_layer.neurons[k]
            neuron.error = targets[k] - outputs[k]
            error += 0.5 * neuron.error ** 2

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
                        neuron_m.weights[n] += learn_rate * neuron_m.last_input[n] * neuron_m.error # neuron_l.delta

                    # Improve this neuron using the previous newtons method information
                    neuron_m.bias += learn_rate * neuron_m.delta

        return error
    
    @staticmethod
    def save(path, network):
        """
            Saves a neural network to a json file
            @param (String) path - The path to save the network to
            @param (Network) network - The network to save to a file
            @returns
        """
        network_data = dict()

        # For every layer in the network ...
        layers_data = []
        for layer in network.layers:
            layer_data = dict()
            layer_neurons = []
            
            # For every neuron in the layer ...
            for neuron in layer.neurons:
                weights = neuron.weights
                bias = neuron.bias

                # Save the neuron data to a dictionary
                neuron_data = dict()
                neuron_data["weights"] = weights
                neuron_data["bias"] = bias

                # Add the neuron to the collection
                layer_neurons.append(neuron_data)

            # Add the layer to the collection
            layer_data["neurons"] = layer_neurons
            layers_data.append(layer_data)

        # Add the layer data to the network data
        network_data["layers"] = layers_data

        # Convert the object to json (TODO: Formatting)
        network_json = json.dumps(network_data, sort_keys=False, indent=4)
        
        # Write the json data to the file
        with open(path, "w+") as f:
            f.write(network_json)

    @staticmethod
    def load(path):
        """
            Loads a neural network from a json file
            @param (String) path - The path to load the neural network from
            @returns (Network) - The neural network that was loaded
        """
        network = Network()

        try:
            with open(path, "r+") as f:
                network_data = "\n".join(f.readlines())
                network_json = json.loads(network_data)
                layers = network_json["layers"]
                
                # For every layer in the network ...
                for layer in layers:
                    neurons = []

                    # For every neuron in the layer ...
                    for neuron in layer["neurons"]:
                        weights = neuron["weights"]
                        bias = neuron["bias"]

                        # Add the processed neuron to the collection
                        neurons.append(Neuron(weights, bias))

                    # Create a new layer with this set of neurons
                    network.add_layer(Layer(neurons))
        except:
            raise Exception("Invalid Neural Network File @ {}!".format(path))

        return network
