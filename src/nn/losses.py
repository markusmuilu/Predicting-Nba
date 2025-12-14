import numpy as np

def bce_loss(y_true, y_pred, eps=1e-8):
    a = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(a) + (1 - y_true) * np.log(1 - a))
