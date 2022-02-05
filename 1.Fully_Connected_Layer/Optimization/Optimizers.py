import numpy as np


class Sgd():

    def __init__(self, learning_r: float):
        self.learning_rate = learning_r

    def calculate_update(self, weights, gradient):
        return weights - self.learning_rate * gradient # Returns updated weights according to the basic gradient descent
                                                        # update scheme
