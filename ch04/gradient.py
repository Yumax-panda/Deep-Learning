import numpy as np
import typing as t

F = t.Callable[[np.ndarray], float]

def func(x: np.ndarray) -> float:
    return np.sum(x**2)

def gradient(f: F, x: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(x)

    for idx in range(x.size):
        h = np.zeros_like(x)
        h[idx] = 1e-4
        grad[idx] = (f(x+h) - f(x-h))/(2*1e-4)
    return grad



if __name__ == '__main__':
    print(gradient(func, np.array([3.0, 4.0])))