import numpy as np
from Layers import Base
from Layers import Helpers
import copy

class BatchNormalization(Base.BaseLayer):
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.mean = 0
        self.var = 0
        self._optimizer = None
        self.bias_optimizer = None # Seperate optimizer for the bias
        self.initialize('weights_initialization', 'bias_initialization')

    def initialize(self, weights_initializer, bias_initializer): # We need to make the initialize method "universal" (because other types of layers have it as well)
        # so there must be something in the place of weights and bias initializers.
        '''Initializes always the weights with ones and the biases with zeros,
        ince you do not want the weights  and bias to have an impact at the beginning of the training.'''

        weights_initializer = None
        bias_initializer = None
        self.weights = np.ones((1, self.channels)) # creates a matrix with 1
        self.bias = np.zeros((1, self.channels)) # creates a matrix with 0

    def reformat(self, tensor): # a method which receives the tensor that must be reshaped.
        if len(tensor.shape) == 4: # Depending on the shape of the tensor, the method reformats the image-like tensor with 4D)
            B, C, H, W = tensor.shape # Batches, channels, height, width
            
            output_tensor = np.reshape(tensor, [B, C, H*W])
            output_tensor = np.transpose(output_tensor, [0, 2, 1]) #  just swaps the shape and stride information for each axis
            output_tensor = np.reshape(output_tensor, [B*H*W, C])
        
        else:
            B, C, H, W  = self.input_tensor.shape

            output_tensor = np.reshape(tensor, [B, H * W, C])
            output_tensor = np.transpose(output_tensor, [0, 2, 1])
            output_tensor = np.reshape(output_tensor, [B, C, H, W])          

        return output_tensor
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        epsilon = np.finfo(float).eps
        alpha = 0.8 # As slide 9

        input_tensor_shape = input_tensor.shape
        CNN = len(input_tensor_shape) == 4 # Check shape of input

        if CNN==True: 
            self.ref_input_tensor = self.reformat(self.input_tensor) # page 9 of slides
        else:          
            self.ref_input_tensor = self.input_tensor

        if not self.testing_phase:
            self.mean_k = np.mean(self.ref_input_tensor, axis = 0)
            self.var_k = np.std(self.ref_input_tensor, axis = 0)

            self.X = (self.ref_input_tensor - self.mean_k) / (np.sqrt(self.var_k**2 + epsilon))
            self.Y = self.weights * self.X + self.bias

            self.mean = alpha * self.mean + (1 - alpha) * self.mean_k
            self.var = alpha * self.var + (1 - alpha) * self.var_k
        
        else: # Change for testing
            self.X = (self.ref_input_tensor - self.mean) / np.sqrt(self.var**2 + epsilon)
            self.Y = self.weights * self.X + self.bias

        if CNN: 
            self.Y = self.reformat(self.Y)

        return self.Y

    def backward(self, error_tensor):
        CNN = len(error_tensor.shape) == 4

        if CNN: self.error_tensor = self.reformat(error_tensor)
        else: self.error_tensor = np.reshape(error_tensor,self.X.shape)

        gradient_weights = np.sum(self.error_tensor * self.X, axis = 0)
        self.gradient_weights = np.reshape(gradient_weights, [1, self.channels])

        gradient_bias = np.sum(self.error_tensor, axis = 0)
        self.gradient_bias = np.reshape(gradient_bias, [1, self.channels])

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        if self.bias_optimizer is not None:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        self.gradient_input = Helpers.compute_bn_gradients(
            self.error_tensor,
            self.ref_input_tensor,
            self.weights,
            self.mean_k,
            self.var_k**2,
            np.finfo(float).eps)

        if CNN:
            self.gradient_input = self.reformat(self.gradient_input)

        return self.gradient_input

# All properties

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)
        self.bias_optimizer = copy.deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
    
    @property
    def gradient_bias(self):
        return self._gradient_bias
    
    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias

