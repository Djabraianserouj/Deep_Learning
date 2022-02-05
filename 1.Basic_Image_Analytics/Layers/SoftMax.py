import numpy as np
from Layers import Base


class SoftMax(Base.BaseLayer): #SoftMax inherits the BaseLayer but it is NOT trainable
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        x_max = np.tile(input_tensor.max(1), (input_tensor.shape[1], 1)).T # Max of the whole Matrix
        input_minus_max = np.exp(input_tensor - x_max) #we substract the max to pbtain numerical stabilty
        exp_sum = np.tile(np.sum(input_minus_max, axis=1), (input_tensor.shape[1], 1)).T
        self.output_tensor = np.divide(input_minus_max, exp_sum)
        return self.output_tensor

    def backward(self, error_tensor):
        tmp = np.multiply(error_tensor, self.output_tensor)
        tmp2 = np.tile(np.sum(tmp, axis=1), (error_tensor.shape[1], 1)).T
        return np.multiply(self.output_tensor, (error_tensor - tmp2))
