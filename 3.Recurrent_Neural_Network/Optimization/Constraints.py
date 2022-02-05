import numpy as np

class L2_Regularizer: # L2 regularization acts like a force that removes a small percentage of weights at each iteration.

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return weights * self.alpha

    def norm(self, weights):
        return self.alpha * np.square(np.linalg.norm(weights))

class L1_Regularizer: # L1 regularization adds a fixed gradient to the loss at every value other than 0,
                      # while the gradient added by L2 regularization decreases as we approach 0.

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return np.sign(weights) * self.alpha

    def norm(self, weights):
        # return self.alpha*np.linalg.norm(weights, 1)
        return self.alpha * np.sum(np.abs(weights))