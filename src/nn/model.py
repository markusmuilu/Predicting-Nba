import numpy as np
import matplotlib.pyplot as plt

from .losses import bce_loss
from .activations import sigmoid, relu, drelu
from .metrics import evaluate

class NeuralNetwork:
    """
    Simple fully-connected neural network with ReLU hidden layers
    and sigmoid output, trained with binary cross-entropy.
    Supports minibatch gradient descent.
    """
    def __init__(self, layers, lr=0.001, batch_size=1):
        self.layers = layers # As a vector of number of nodes in each layer [input, hidden1, hidden2, ..., output] in example
        self.lr = lr # Initialize learning rate
        self.batch_size = batch_size

        
        weights = [] # First only an empty vector, but we append weight matrices, one for each layer

        #Initialize the weights as random for each weight of every layer
        for i in range(1,len(layers)-1):
            weights.append(np.random.randn(layers[i],layers[i-1]) * np.sqrt(2/(layers[i-1])))
        last = len(layers) - 1
        weights.append(np.random.randn(layers[last],layers[last-1]) * np.sqrt(1/(layers[last-1])))
        self.weights = weights

        #Initialize biases as column for every layer, except input layer
        biases = [np.zeros(layers[layer]).reshape(-1, 1) for layer in range(1, len(layers))]
        self.biases = biases



    # Forward propagation
    def forward_prop(self, x):
        activations = [x]
        z = []

        weights = self.weights
        biases = self.biases

        for i in range(len(weights)-1):
            z.append(np.dot(weights[i], activations[i]) +  biases[i]) # Make so that the biases is added into columns of x
            activations.append(relu(z[i]))

        # Use sigmoid for output layer
        last = len(weights) - 1
        z.append(np.dot(weights[last], activations[last]) +  biases[last]) 
        activations.append(sigmoid(z[last]))
        self.activations = activations
        self.z = z


    # Back propagation
    def back_prop(self, y):
        activations = self.activations
        weights = self.weights
        z = self.z
        L = len(self.layers)
        dz = [None] * L
        dw = [None] * L
        db = [None] * L

        # Output layer
        dz[L-1] = activations[L-1]-y
        dw[L-1] = np.dot(dz[L-1], activations[L-2].T)/self.batch_size
        db[L-1] = (np.sum(dz[L-1], axis=1, keepdims=True)) / self.batch_size

        # Backprop through hidden layers
        for i in reversed(range(1, L-1)):
            dz[i] = np.dot(weights[i].T, dz[i+1]) * drelu(z[i-1])
            dw[i] = np.dot(dz[i], activations[i-1].T)/self.batch_size
            db[i] = np.sum(dz[i], axis=1, keepdims=True) / self.batch_size
        biases = self.biases
        weights = self.weights

        # Make the change of the amount gradient times lr
        for i in range(1, L):
            self.weights[i-1] -= self.lr * dw[i]
            self.biases[i-1] -= self.lr * db[i]



    # Fitting
    def fit(self, X, Y, epochs=100):
        batch_size = self.batch_size
        samples = X.shape[0]
        batches = samples // batch_size
        losses = []

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        for i in range(epochs):
            # Shuffle for every epoch
            indices = np.random.permutation(samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            for j in range(batches):
                start = j*batch_size
                end = start + batch_size
                batch = slice(start, end)
                self.forward_prop(X_shuffled[batch].T)
                self.back_prop(Y_shuffled[batch].T)

            self.forward_prop(X.T)
            y_pred = self.activations[-1]      # final output layer

            loss = bce_loss(Y.T, y_pred)
            losses.append(loss)

        
        # Lets print final loss
        print(f"Final loss {losses[-1]}")
       
    


    # Make prediction
    def predict(self, X):

        # Convert to NumPy array
        X = np.array(X)


        # If X is one sample of shape (n_features,), reshape (1, n_features)
        # If X is already (n_samples, n_features), do nothing
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.forward_prop(X.T)

        probs = self.activations[-1].flatten()       

        # Predicted classes
        predictions = (probs >= 0.5).astype(int)

        # Confidence for each prediction
        confidences = np.where(predictions == 1, probs, 1 - probs)

        # Build output list of dicts
        results = [
            {"result": int(predictions[i]), "confidence": float(confidences[i])}
            for i in range(len(predictions))
        ]

        # when given only one sample, return a single dict
        if len(results) == 1:
            return results[0]

        return results



    