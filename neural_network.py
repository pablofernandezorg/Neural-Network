"""
    Creator      : J Heslop
    Created on   : 5/25/2016 (11:45 A.M.)
    Last Editted : 5/26/2016 (05:22 P.M.)
"""

from genome import Genome
import random

DEBUG = False

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
