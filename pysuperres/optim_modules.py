import numpy as np


class gd_optimizer():
    def __init__(self):
        return

    def optimize(self,x,grad,learning_rate):
        x -= learning_rate * grad
        
        return x

class momentum_optimizer():
    def __init__(self,x,optimizer_parameter=None):
        self.v = np.zeros(x.shape)
        if optimizer_parameter is not None:
            self.m = optimizer_parameter
        else:
            self.m = 0.9

        return

    def optimize(self,x,grad,learning_rate):
        self.v = self.m * self.v - learning_rate * grad
        x -= self.v
        
        return x

class adam_optimizer():
    def __init__(self,x,optimizer_parameter=None):
        self.v = np.zeros(x.shape)
        self.m = np.zeros(x.shape)

        if optimizer_parameter is not None:
            self.beta1 = optimizer_parameter[0]
            self.beta2 = optimizer_parameter[1]
        else:
            self.beta1 = 0.9
            self.beta2 = 0.999

        return

    def optimize(self,x,grad,learning_rate):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad * grad
        x -= learning_rate * self.m / (np.sqrt(self.v) + 10**-8) 
        
        return x


def select_optimizer(optimizer_name, est, optimizer_parameter=None):
    if optimizer_name == 'gd':
        optimizer = gd_optimizer()
    elif optimizer_name == 'momentum':
        optimizer = momentum_optimizer(est, optimizer_parameter)
    elif optimizer_name == 'adam':
        optimizer = adam_optimizer(est, optimizer_parameter)
    else:
        warnings.warn('No optimizer found within the specification. ADAM optimizer with default parameters selected.')
        optimizer = adam_optimizer(est)
        
    return optimizer

