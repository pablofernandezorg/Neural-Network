"""
    Creator      : Jayrese Heslop
    Created on   : 5/25/2016 (11:45 A.M.)
    Last Editted : 5/26/2016 (05:24 P.M.)
"""
import random

class Genome(object):
    OPERATIONS = [lambda a, b: a + b,      # Addition
                  lambda a, b: a - b,      # Subtraction
                  lambda a, b: a * b,      # Multiplication
                  lambda a, b: a ** 2 * b] # Powers of 2 (Weighted)

    """
        Creates a new genome using certain weights

        @param weights (Array<Number>) - The weights of the genome
    """
    def __init__(self, weights):
        self.weights = weights

    """
        Returns the weights used to compute the answer

        @param self (Genome) - The genome to get the weights of
        @returns (Array<Number>) - The weights of the genome
    """
    def get_weights(self):
        return self.weights

    """
        Returns a child from this genome

        @param self (Genome) - The father genome
        @returns (Genome) - The child genome
    """
    def get_child(self, sensitivity):
        return Genome(list(map(lambda weight: weight + random.uniform(-sensitivity, sensitivity), self.weights)))

    """
        Predicts the value by performing all desired options, weighting them differently then summing

        @param self (Genome) - The genome to use for prediction
        @param val (Number) - The input value to the function
        @returns (Number) - The value predicted
    """
    def predict(self, val):
        return sum(Genome.OPERATIONS[i](val, self.weights[i]) for i in range(len(Genome.OPERATIONS)))
