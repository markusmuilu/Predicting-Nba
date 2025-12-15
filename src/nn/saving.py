import numpy as np
from .model import NeuralNetwork
"""
Need to implement model saving and loading to be used inside my NBA prediction project.
"""

def save_model(model: NeuralNetwork, path: str):
    """
    Save a NeuralNetwork model to a compressed .npz file.

    Stored fields:
    - layers
    - learning rate
    - batch size
    - weights (list of arrays)
    - biases  (list of arrays)
    """
    np.savez_compressed(
        path,
        layers=np.array(model.layers, dtype=object),
        lr=model.lr,
        batch_size=model.batch_size,
        weights=np.array(model.weights, dtype=object),
        biases=np.array(model.biases, dtype=object),
    )
    print(f"Model saved to {path}")


def load_model(path: str) -> NeuralNetwork:
    """
    Load a NeuralNetwork model from a .npz file and return a fully restored instance.
    """
    data = np.load(path, allow_pickle=True)

    layers = data["layers"].tolist()
    lr = float(data["lr"])
    batch_size = int(data["batch_size"])

    weights = data["weights"].tolist()
    biases = data["biases"].tolist()

    # Create a new model with the same architecture & hyperparameters
    model = NeuralNetwork(layers=layers, lr=lr, batch_size=batch_size)

    # Overwrite randomly-initialized values with saved values
    model.weights = weights
    model.biases = biases

    print(f"Model loaded from {path}")
    return model
