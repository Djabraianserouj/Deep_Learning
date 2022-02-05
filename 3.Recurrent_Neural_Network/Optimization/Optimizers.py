import numpy as np

class Optimizer: #Parent Class
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer): #method
        self.regularizer = regularizer

# Make all optimizers inherit from this base-optimizer".
class Sgd(Optimizer):

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            gradients = self.regularizer.calculate_gradient(weight_tensor)
            output = weight_tensor - self.learning_rate*gradients - self.learning_rate * gradient_tensor
        else:
            output =  weight_tensor - self.learning_rate * gradient_tensor

        return output

class SgdWithMomentum(Optimizer):
    # Momentum 
    # Commonly: μ = {0.9, 0.95, 0.99}

    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        self.v = v

        if self.regularizer:
            output = weight_tensor - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor) + v
        else:
            output = weight_tensor + v
        return output

class Adam(Optimizer):

    def __init__(self, learning_rate, mu, rho):
        #mu: β1
        #rho: β2
        # Commonly:  μ = 0.9, ρ = 0.999,  η = 0.001
        super().__init__()
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

        if self.regularizer:
            output = weight_tensor - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor) - self.learning_rate*(v_hat/(np.sqrt(r_hat) + epsilon))
        else:
            output = weight_tensor - self.learning_rate*(v_hat/(np.sqrt(r_hat) + epsilon))
        return output

