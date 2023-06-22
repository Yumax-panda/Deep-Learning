import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

from common.functions import (
    sigmoid,
    softmax,
    sigmoid_grad,
    cross_entropy_error
)
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        std: float = 0.01
    ):
        self.params: dict[str, np.ndarray] = {
            "W1": std*np.random.randn(input_size, hidden_size),
            "b1": np.zeros(hidden_size),
            "W2": std*np.random.randn(hidden_size, output_size),
            "b2": np.zeros(output_size),
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        a1 = np.dot(x, self.params["W1"]) + self.params["b1"]
        a2 = np.dot(sigmoid(a1), self.params["W2"]) + self.params["b2"]
        return softmax(a2)

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        return np.sum(np.argmax(x, axis=1)==np.argmax(t, axis=1))/float(x.shape[0])

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        f = lambda _: self.loss(x, t)
        grad = {}

        for k in ("W1", "W2", "b1", "b2"):
            grad[k] = numerical_gradient(f, self.params[k])

        return grad

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataset.mnist import load_mnist

    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, one_hot_label=True)
    network = TwoLayerNet(
        input_size=784,
        hidden_size=50,
        output_size=10
    )
    cost_history = []

    # Hyper parameters
    max_iter = 10000
    batch_size = 100
    train_size = X_train.shape[0]
    learning_rate = 0.1

    for _ in range(max_iter):
        batch_mask = np.random.choice(train_size, batch_size)
        grad = network.gradient(X_train[batch_mask], y_train[batch_mask])

        for k in ("W1", "W2", "b1", "b2"):
            network.params[k] -= learning_rate * grad[k]

        cost_history.append(network.loss(X_train[batch_mask], y_train[batch_mask]))

    plt.plot(
        np.arange(len(cost_history)),
        cost_history
    )
    plt.title("Cost Function")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.show()