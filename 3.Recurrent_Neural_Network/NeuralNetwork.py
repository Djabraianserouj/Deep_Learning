import copy
import numpy as np
from Optimization import Constraints, Loss, Optimizers

#self.optimizer.regularizer.norm(weight_tensor)

class NeuralNetwork():

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = [] # This list will store all the layers
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = None

    def forward(self):
        reg_loss = 0

        self.input_tensor, self.label_tensor = self.data_layer.next()

        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)
            if self.optimizer.regularizer and (layer.trainable == True):
                reg_loss += self.optimizer.regularizer.norm(layer.weights) # Get regularization loss inside all layers
                                                                            # and sum it up

        output = self.loss_layer.forward(self.input_tensor, self.label_tensor)

        return output + reg_loss

    def backward(self):

        error_tensor = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):

        if (layer.trainable == True): #Only if the layer is trainable (has trainable parameters)
            optimizer = copy.deepcopy(self.optimizer) #Copy optimizer, set and initialize it
            layer.optimizer = optimizer 
            layer.initialize(self.weights_initializer, self.bias_initializer) # Initialize trainable layer with the stored initializers
            
        self.layers.append(layer) # Appends the layer to the layers list, whether or not it's trainable

    def train(self, iterations):
        self.phase=False
        for layer in self.layers:
            layer.testing_phase = self.phase

        for iteration in range(iterations):
            self.loss.append(self.forward())
            error = self.backward()

    def test(self, input_tensor):
        self.phase=True
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            layer.testing_phase = self.phase
        prediction = input_tensor

        return prediction

    @property # provides an interface to instance attributes.
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase
        for layer in self.layers:
            layer.testing_phase = phase