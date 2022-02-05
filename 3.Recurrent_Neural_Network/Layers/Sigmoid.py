import numpy as np
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations = None # f(x)

    def forward(self, input_tensor):
        self.activations = 1/(1 + np.exp(-1 * input_tensor)) # f(x)
        return self.activations

    def backward(self, error_tensor):
        return error_tensor*self.activations * (1 - self.activations) # errortensor * f'(x)