import numpy as np

# Sigmoid functions for output layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(a):
    return a * (1 - a)

# Relu functions for hidden layers
def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(float)