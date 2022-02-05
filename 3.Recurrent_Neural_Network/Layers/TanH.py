import numpy as np
from Layers.Base import BaseLayer

class TanH(BaseLayer):
    '''Hyperbolic Tangent Activation'''
    def __init__(self):
        super().__init__()
        self.activations = None # f(x)

    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor) # f(x)
        return self.activations

    def backward(self, error_tensor):
        return error_tensor * ( 1 - np.square(self.activations)) # error tensor * f'(x)