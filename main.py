"""
    Creator      : J Heslop
    Created on   : 5/25/2016 (11:45 A.M.)
    Last Editted : 5/26/2016 (05:26 P.M.)
"""

import neural_network

def main():
    # Uncomment any of the functions below to test
    # f = lambda n: 5 * (n ** 2) + 3 * n + 7 # Quadratic
    # f = lambda n: 5 * n + 5 # Linear
    f = lambda n: n + 5 # Simple
    tests = [2 ** i for i in range(0, 10, 1)]
    
    predictors = neural_network.predict(f, tests, num_gens=100, num_genomes=100, sensitivity=0.5)
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
