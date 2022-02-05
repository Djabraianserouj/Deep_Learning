import numpy as np

class Constant:
    # Very bad for weights, typically for biases

    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        # Fan in: input dimension of the weights
        # Fan_out: output dimension of the weights
        return np.full(weights_shape, self.value) # Makes an array of weights_shape size filled with the same value

class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.random.random_sample(weights_shape) # Makes an array of weights_shape size filled with values
                                                     # sampled from a uniform distributions in [0,1).

class Xavier: # Also known as Glorot initializer, typically for weights
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_in+fan_out))
        return np.random.normal(0, sigma, weights_shape) # Makes an array of weights shape size filled with values
                                                            #sampled from a normal N(0, σ) distribution

class He: # STD of weights determined by size of previous layer only
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/fan_in)
        return np.random.normal(0, sigma, weights_shape)