import numpy as np
from Layers import Base
import math

class FullyConnected(Base.BaseLayer): #FullyConnected inherits the BaseLayer

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self._optimizer = None #protected
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.random([self.input_size + 1, self.output_size])
        self.input_tensor = None
        self.input_with_bias = None
        self.error_tensor = None
        self._gradient_weights = None #protected, gradient with respect to weights after they have calculated to the
                                     # backward pass

    def forward(self, input_tensor):
        bias = np.ones(np.size(input_tensor, 0))
        input_with_bias = np.column_stack((input_tensor, bias)) # stacks the bias next to input tensor as columns
        self.input_tensor = input_tensor
        self.input_with_bias = input_with_bias
        return np.dot(input_with_bias, self.weights) #result input tensor for the next layer

    @property # takes the protected variable with getter and it changes the input with setter
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @optimizer.getter
    def optimizer(self):
        return self._optimizer

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        new_error_tensor = np.dot(error_tensor, self.weights.T) #transposed weights, error tensor from previous layer
        self.gradient_weights = np.dot(self.input_with_bias.T, error_tensor)
        if self._optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return new_error_tensor[:, :-1]

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @gradient_weights.getter
    def gradient_weights(self):
        return self._gradient_weights

