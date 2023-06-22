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


def gradient_descent(
    f: F,
    x: np.ndarray,
    rate: float = 0.01,
    step: int = 100
) -> tuple[np.ndarray, list[list[float, float]]]:
    arr = x.copy()
    history = []

    for _ in range(step):
        arr -= rate*gradient(f, arr)
        history.append(arr.copy())

    return arr, np.array(history)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(gradient(func, np.array([3.0, 4.0])))
    minimum, history = gradient_descent(func, np.array([-3.0, 4.0]), rate=0.1)
    print(f"Minimum : {minimum}")
    plt.plot([-5, 5], [0,0], '--b')
    plt.plot([0,0], [-5, 5], '--b')
    plt.plot(history[:,0], history[:,1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()