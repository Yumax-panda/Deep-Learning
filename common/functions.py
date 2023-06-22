import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x :np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))


def softmax(a: np.ndarray) -> np.ndarray:
    c = np.max(a)
    total = np.sum(np.exp(a-c))
    return np.exp(a-c)/total

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)
    print(np.sum(y))