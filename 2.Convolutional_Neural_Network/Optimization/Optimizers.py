import numpy as np

class Sgd: # is an iterative method for optimizing an objective function with suitable smoothness properties

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor

class SgdWithMomentum: # is method which helps accelerate gradients vectors in the right directions,
                       # thus leading to faster converging
    # Momentum 
    # Commonly: μ = {0.9, 0.95, 0.99}

    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        self.v = v
        return weight_tensor + v

class Adam: # requires little memory, and is well suited for problems that are large in terms of data or parameters or both.
            # is a popular extension to stochastic gradient descent.

    def __init__(self, learning_rate, mu, rho):
        #mu: β1
        #rho: β2
        # Commonly:  μ = 0.9, ρ = 0.999,  η = 0.001
        
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):

        v = self.mu * self.v + (1-self.mu)*gradient_tensor
        r = self.rho * self.r + (1-self.rho) * gradient_tensor * gradient_tensor

        v_hat = v / (1 - np.power(self.mu, self.k))
        r_hat = r / (1 - np.power(self.rho, self.k))

        self.v = v
        self.r = r
        self.k += 1

        epsilon = np.finfo(float).eps

        return weight_tensor - self.learning_rate*(v_hat/(np.sqrt(r_hat) + epsilon))

