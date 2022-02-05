import numpy as np
from .Base import *
# from Layers import *

class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()
        self.input_tensor = None 
        self.error_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor # Save the last input tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        self.error_tensor = error_tensor # Save the last error tensor
        return self.error_tensor*(self.input_tensor>0)
