import copy
import numpy as np


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

    def forward(self):

        self.input_tensor, self.label_tensor = self.data_layer.next()

        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)

        output = self.loss_layer.forward(self.input_tensor, self.label_tensor)

        return output

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

        for iteration in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        prediction = input_tensor

        return prediction

