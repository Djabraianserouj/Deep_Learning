import numpy as np


class CrossEntropyLoss():

    def __init__(self):

        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor

        epsilon = np.finfo(float).eps

        temp = self.prediction_tensor[label_tensor==1]
        loss_func = lambda x: -np.log(x + epsilon)
        loss = np.sum(np.vectorize(loss_func)(temp))

        return loss

    def backward(self, label_tensor):

        return -np.true_divide(label_tensor, self.prediction_tensor)
