import copy
import numpy as np



class NeuralNetwork():

    def __init__(self, optimizer):

        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None

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

    def append_layer(self, layer): #we have the row of the layers saved in a list, and append_layers adds there

        if (layer.trainable == True):
            optimizer = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer

        self.layers.append(layer)

    def train(self, iterations): #takes the iteration parameter and the method train all the given iterations
                                # + saved loss for each iteration in a list

        for iteration in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        prediction = input_tensor
        return prediction

