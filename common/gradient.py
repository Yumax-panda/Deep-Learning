import numpy as np
import typing as t

F = t.Callable[[np.ndarray], float]


def func(x: np.ndarray) -> float:
    return np.sum(x**2)


def _numerical_gradient_no_batch(f: F, x: np.ndarray) -> np.ndarray:
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 値を元に戻す

    return grad


def numerical_gradient(f: F, X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

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