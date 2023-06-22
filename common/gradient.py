import numpy as np
import typing as t

F = t.Callable[[np.ndarray], float]


def func(x: np.ndarray) -> float:
    return np.sum(x**2)


def numerical_gradient(f: F, x: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(x)

    for idx in range(x.size):
        h = np.zeros_like(x)
        h[idx] = 1e-4
        grad[idx] = (f(x+h) - f(x-h))/(2*1e-4)
    return grad


def gradient_descent(
    f: F,
    x: np.ndarray,
    rate: float = 0.01,
    step: int = 100
) -> tuple[np.ndarray, list[list[float, float]]]:
    arr = x.copy()
    history = []

    for _ in range(step):
        arr -= rate*numerical_gradient(f, arr)
        history.append(arr.copy())

    return arr, np.array(history)