import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import typing
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class SimpleNet:

    __slots__ = ("W",)

    if typing.TYPE_CHECKING:
        W: np.ndarray

    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.W)

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        pred = self.predict(x)
        return cross_entropy_error(softmax(pred), t)


if __name__ == '__main__':
    net = SimpleNet()
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])
    print(numerical_gradient(lambda w: net.loss(x, t), net.W))