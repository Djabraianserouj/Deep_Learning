# from .Base import *
# import numpy as np

# class SoftMax(BaseLayer):

#     def __init__(self):

#         super().__init__()
#         self.batch_size = None
#         self.predictions = None
#         self.input_tensor = None
#         self.error_tensor = None

#     def forward(self, input_tensor):
#         self.input_tensor = input_tensor
#         self.batch_size = self.input_tensor.shape[1]

#         sumexp = np.sum(np.exp(self.input_tensor))
#         x_max =np.amax(self.input_tensor)

#         calculate_softmax = lambda x: np.exp(x - x_max)/sumexp
#         output_tensor = np.vectorize(calculate_softmax)(self.input_tensor)

#         self.predictions = output_tensor #Last output of the NN is the predictions - or ŷ 

#         return output_tensor

#     # def backward(self, error_tensor):
#     #     self.error_tensor = error_tensor

#     #     interm = np.dot(self.error_tensor, self.predictions)
#     #     new_error_tensor = np.dot(self.predictions, self.error_tensor - interm)

#     #     return new_error_tensor

#     def backward(self, error_tensor):
#         tmp = np.multiply(error_tensor, self.output_tensor)
#         tmp2 = np.tile(np.sum(tmp, axis=1), (error_tensor.shape[1], 1)).T
#         return np.multiply(self.output_tensor, (error_tensor - tmp2))


import numpy as np
from Layers import Base


class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        x_max = np.tile(input_tensor.max(1), (input_tensor.shape[1], 1)).T # Make an array with the max of the arrayja
        input_minus_max = np.exp(input_tensor - x_max)
        exp_sum = np.tile(np.sum(input_minus_max, axis=1), (input_tensor.shape[1], 1)).T
        self.output_tensor = np.divide(input_minus_max, exp_sum)
        return self.output_tensor

    def backward(self, error_tensor):
        tmp = np.multiply(error_tensor, self.output_tensor)
        tmp2 = np.tile(np.sum(tmp, axis=1), (error_tensor.shape[1], 1)).T
        return np.multiply(self.output_tensor, (error_tensor - tmp2))
