"""
    Creator      : J Heslop
    Created on   : 5/25/2016 (11:45 A.M.)
    Last Editted : 5/25/2016 (1:05 A.M.) Should probably go to sleep soon...
"""
import operator
import random
import functools

DEBUG = False

def safe_pow(a, b):
    if a == 0.0 and b <= 0.0:
        return 0.0
    else:
        return a ** b

class Genome(object):
    OPERATIONS = [lambda a, b: a + b,      # Addition
                  lambda a, b: a - b,      # Subtraction
                  lambda a, b: a * b,      # Multiplication
                  safe_pow]                # Powers (no errors)

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

"""
    Predicts the outputs of a function

    @param f (Function) - The function to predict
    @param tests (Array<Number>) - The numbers to use to test the genomes with
    @param num_gens (Number) - The number of generations to go through
    @param num_genomes (Number) - The number of genomes per generation
    @param sensitivity (Number) - The sensitivity of the controls for each operation
    @returns (Array<Number>(num_genomes)) - The best genomes of the generation
"""
def predict(f, tests, num_gens = 100, num_genomes = 25, sensitivity = 0.01):
    best_genomes = [Genome([random.uniform(-sensitivity, sensitivity) for i in range(0, len(Genome.OPERATIONS), 1)]) for i in range(0, num_genomes, 1)] # Stores the best genomes of the current generation
    new_genomes = [None, ] * num_genomes  # Stores the children of the best genomes of the current generation
    f_predictions = [f(tests[i]) for i in range(0, len(tests), 1)] # The predictions that a function makes
    
    """
        Calculates the total miscalculation of a genome

        @param g (Genome) - The genome to calculate the total miscalculation of
        @returns (Number) - The error margin
    """
    calculate_error_margin = lambda g: sum([abs(g.predict(tests[i]) - f_predictions[i]) for i in range(0, len(tests), 1)])

    # Loop through the number of desired generations
    for i in range(0, num_gens, 1):
        # Check if debug mode is enabled
        if DEBUG:
            print("Testing Generation #{}".format(i + 1))

        # Loop through all genomes creating children
        for j in range(0, num_genomes, 1):
            # Create a genome from a parent
            father = best_genomes[j]
            child = father.get_child(sensitivity)

            # Add the new genome to a collection
            new_genomes[j] = child
        
        # Only keep the best genomes of this generation
        best_genomes = sorted(best_genomes + new_genomes, key=calculate_error_margin)[:num_genomes]
    
    return best_genomes

def main():
    # TODO: Fix code for trinomials and higher degree polynomials
    # f(n) = 5(n ^ 2) + 3n + 7
    # f = lambda n: 5 * (n ** 2) + 3 * n + 7
    # f = lambda n: 5 * n + 5
    f = lambda n: n + 5
    tests = [2 ** i for i in range(0, 10, 1)]
    
    predictors = predict(f, tests, num_gens=100, num_genomes=100, sensitivity=0.5)
    predictor = predictors[0]

    n = 10.0
    approximate = predictor.predict(n)
    actual = f(n)
    error_margin = abs(approximate - actual)
    error_percentage = error_margin / (abs(approximate) + abs(actual)) * 100
    
    # Output to the user all of the crap
    print("Approximate : ~{:.02f}".format(approximate))
    print("Actual      : ~{:.02f}".format(actual))
    print("Error       : ~{:.02f} ({:.02f}%)".format(error_margin, error_percentage))
    
    return 0

# Check if this file was called directly
if __name__ == "__main__":
    main()
