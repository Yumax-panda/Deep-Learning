import numpy as np

def sum_squared_error(y: np.ndarray, t: np.ndarray):
    return 0.5*np.sum((y-t)**2)


def cross_entropy_error(y: np.ndarray, t: np.ndarray):
    delta = 1e-7
    return (-1) * np.sum(t*np.log(y+delta)) # to prevent inf when zero