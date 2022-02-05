import numpy as np
from Layers import Base, Initializers

class Flatten(Base.BaseLayer):
    '''Flatten layers reshapes the multi-dimensional input to a one dimensional feature vector.
    Useful between conv and pool layer.'''

    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_tensor): # reshapes and returns the input tensor
        self.input_shape = input_tensor.shape
        batch, w, h, rgb = input_tensor.shape
        return np.reshape(input_tensor, (batch, w*h*rgb))

    def backward(self, error_tensor): # reshapes and returns the error tensor
        return np.reshape(error_tensor, self.input_shape)


#Shape input tensor: (batch, width, height, 3)
#                  Forward:    (10, 30, 30, 3) --> (10, 2700)
#                   Backward:  (10, 30, 30, 3) <-- (10, 2700)