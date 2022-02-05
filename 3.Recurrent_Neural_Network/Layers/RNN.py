import numpy as np
import NeuralNetwork
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Base import BaseLayer
from Layers.Sigmoid import Sigmoid
from copy import deepcopy
from Optimization import Optimizers, Constraints

class RNN(BaseLayer):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size #input size denotes the dimension of the input vector
        self.hidden_size = hidden_size # hidden size denotes the dimension of the hidden state
        self.output_size = output_size

        self.states = [] #store each state of input tensor
        self.hidden_state = np.zeros(hidden_size) # Initialize hidden state with 0s
        self.memorize = False # which sets the boolean state representing whether the RNN regards subsequent sequences as a belonging to the same long sequence.
        self._weights = None
        self.optimizer = None
        
        
        self.layers = [
            FullyConnected(input_size + hidden_size, hidden_size),
            TanH(),
            FullyConnected(hidden_size, output_size),
            Sigmoid()] #recurrent

    def forward(self, input_tensor): # which returns a tensor that serves as the input tensor for the next layer.
        self.input_tensor = input_tensor
        B = input_tensor.shape[0] # Batch size
        self.states = [] # List holding all the states

        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)

        output_tensor = np.zeros([B, self.output_size])
        for t in range(B): # We interpret the batch dimension as time dimension. t --> time 
            local_states = []
            input_vector = input_tensor[t]
            
            tensor = np.concatenate([input_vector, self.hidden_state])
            tensor = np.expand_dims(tensor, axis=0)

            loc = self.layers[0].forward(tensor)
            local_states.append(self.layers[0].input_tensor)

            self.hidden_state = self.layers[1].forward(loc)
            local_states.append(self.layers[1].activations)

            loc = self.layers[2].forward(self.hidden_state)
            local_states.append(self.layers[2].input_tensor)

            output = self.layers[3].forward(loc)
            local_states.append(self.layers[3].activations)

            self.hidden_state = self.hidden_state.flatten()
            self.states.append(local_states)

            output_tensor[t] = output
        return output_tensor

    def backward(self, error_tensor): #which updates the parameters and re-turns the error tensor for the next layer.
        B = error_tensor.shape[0] # Batch size
        output_tensor = np.zeros([B, self.input_size])
        hid_error = 0 # Error from hidden

        FC2_weights = np.zeros_like(self.layers[2].weights) # Gradient weights of second fully connected (FC) layer
        FC1_weights = np.zeros_like(self.layers[0].weights) # Same as above for first FC layer

        for t in reversed(range(B)):
            self.layers[3].activations = self.states[t][3]
            self.layers[2].input_tensor = self.states[t][2]
            self.layers[1].activations = self.states[t][1]
            self.layers[0].input_tensor = self.states[t][0]

            error = error_tensor[t]
            error = self.layers[3].backward(error)
            error = self.layers[2].backward(error)
            error += hid_error # Accumulate error

            error = self.layers[1].backward(error)
            error = self.layers[0].backward(error)
            hid_error = error[:,self.input_size:]
            
            FC1_weights += self.layers[0].gradient_weights
            FC2_weights += self.layers[2].gradient_weights
            output_tensor[t] = error[0, :self.input_size]
            
        self.gradient_weights = FC1_weights

        if self.optimizer is not None:
            self.layers[2].weights = self.optimizer.calculate_update(self.layers[2].weights, FC2_weights)
            self.layers[0].weights = self.optimizer.calculate_update(self.layers[0].weights, FC1_weights)
        
        return output_tensor
    
    def initialize(self, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.layers[0].initialize(weights_initializer, bias_initializer)
        self.layers[2].initialize(weights_initializer, bias_initializer)


# All properties 

    @property
    def weights(self):
        return self.layers[0].weights

    @weights.setter
    def weights(self, weights):
        self.layers[0].weights = weights
   
    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = deepcopy(optimizer)
        self._optimizerbias = deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self.layers[0].gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.layers[0].gradient_weights = gradient_weights
        self._gradient_weights = gradient_weights
    
