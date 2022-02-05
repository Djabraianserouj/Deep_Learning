import numpy as np


class CrossEntropyLoss():

    def __init__(self):

        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor

        epsilon = np.finfo(float).eps # Smallest representable number + it increases stability for very wrong predictions
                                    #to prevent values close to log(0)
        temp = self.prediction_tensor[label_tensor==1]
        loss_func = lambda x: -np.log(x + epsilon)
        loss = np.sum(np.vectorize(loss_func)(temp)) #instead of for loop, we use vectorize

        return loss

    def backward(self, label_tensor):
        return -np.true_divide(label_tensor, self.prediction_tensor) #DOES NOT depend on previous error
        # it is the starting point of the recursive computation of gradients
