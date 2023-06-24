import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import TYPE_CHECKING, Union
from common.layers import (
    ReLU,
    Affine,
    Sigmoid,
    SoftmaxWithLoss
)
from common.gradient import numerical_gradient
from collections import OrderedDict
Layer = Union[ReLU, Affine, Sigmoid, SoftmaxWithLoss]

class TwoLayerNet:

    if TYPE_CHECKING:
        params: dict[str, np.ndarray]
        layers: OrderedDict[str, Layer]
        last_layer: SoftmaxWithLoss

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        std: float = 0.01
    ) -> None:
        W1 = np.random.randn(input_size, hidden_size) * std
        b1 = np.zeros(hidden_size)
        W2 = np.random.randn(hidden_size, output_size) * std
        b2 = np.zeros(output_size)
        self.params = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

        layers = OrderedDict()
        layers["Affine1"] = Affine(W1, b1)
        layers["ReLU1"] = ReLU()
        layers["Affine2"] = Affine(W2, b2)
        layers["ReLU2"] = ReLU()

        self.layers = layers
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        _x = x.copy()
        for val in self.layers.values():
            _x = val.forward(_x)
        return _x

    def loss(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        idx = np.argmax(y, axis=1)
        if t.ndim == 1:
            _t = t
        else:
            _t = np.argmax(t, axis=1)
        return np.sum(idx==_t)/float(x.shape[0])

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
        self.loss(x, t)
        dout = self.last_layer.backward(1)

        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        return {
            "W1": self.layers["Affine1"].dW,
            "b1": self.layers["Affine1"].db,
            "W2": self.layers["Affine2"].dW,
            "b2": self.layers["Affine2"].db
        }

if __name__ == "__main__":
    from dataset.mnist import load_mnist
    import matplotlib.pyplot as plt

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    rate = 0.1

    train_loss = []
    train_accuracy = []
    test_accuracy = []
    epoch = max(train_size/batch_size, 1)

    for i in range(iters_num):
        mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[mask]
        t_batch = t_train[mask]

        grad = network.gradient(x_batch, t_batch)

        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= grad[key] * rate

        train_loss.append(network.loss(x_batch, t_batch))

        if i % epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)
            print(
                (f"{i+1} iterations:\n"
                f"train accuracy: {train_acc:.3f}\n"
                f"test accuracy: {test_acc:.3f}\n")
            )

    line = np.arange(len(train_accuracy))
    plt.plot(line, train_accuracy, label="train accuracy")
    plt.plot(line, test_accuracy, label="test accuracy")
    plt.legend()
    plt.show()