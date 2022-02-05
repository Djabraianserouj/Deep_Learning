import numpy as np
from Layers import Base, Initializers

class FullyConnected(Base.BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.random([self.input_size + 1, self.output_size])
        self._optimizer = None
        self._gradient_weights = None

    def forward(self, input_tensor):
        self.input_tensor = np.concatenate((input_tensor, np.ones([input_tensor.shape[0], 1])), axis = 1)
        return np.dot(self.input_tensor, self.weights) 

    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self.optimizer: 
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)        
        return np.dot(error_tensor, self.weights[:-1,:].T)

    def initialize(self, weights_initializer, bias_initializer):
        fan_in  = np.prod(self.input_size)
        fan_out = np.prod(self.output_size)
        self.weights[:-1,:] = weights_initializer.initialize(self.weights[:-1,:].shape, fan_in, fan_out)
        self.weights[-1, :] = bias_initializer.initialize(self.weights[-1].shape, fan_in, fan_out)

    
    #Getters
    @property
    def optimizer(self):
        return self._optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights


    #Setters
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer


    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
