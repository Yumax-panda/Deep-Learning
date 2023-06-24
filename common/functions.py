import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x :np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def sum_squared_error(y: np.ndarray, t: np.ndarray):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y: np.ndarray, t: np.ndarray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    return (1.0 - sigmoid(x)) * sigmoid(x)


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