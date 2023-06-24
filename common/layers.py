import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import TYPE_CHECKING, Optional
from .functions import softmax, cross_entropy_error


class ReLU:

    if TYPE_CHECKING:
        mask: Optional[np.NDArray[np.bool_]]

    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout[self.mask] = 0
        dx = dout

        return dx



class Sigmoid:

    if TYPE_CHECKING:
        out: Optional[np.ndarray]

    def __init__(self):
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = 1 / (1+np.exp(-x))
        return self.out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.out *(1.0 - self.out)


class Affine:

    if TYPE_CHECKING:
        W: np.ndarray
        b: np.ndarray
        x: Optional[np.ndarray]
        original_shape: Optional[tuple[int, int]]
        dW: Optional[np.ndarray]
        db: Optional[np.ndarray]

    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = W
        self.b = b
        self.x = None
        self.original_shape = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_vector = x.reshape(x.shape[0], -1)
        self.original_shape = x.shape
        self.x = x_vector
        return np.dot(x_vector, self.W) + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:

    if TYPE_CHECKING:
        t: Optional[np.ndarray]
        y: Optional[np.ndarray]
        loss: Optional[float]

    def __init__(self) -> None:
        self.t = None
        self.y = None
        self.loss = None

    def forward(self, x: np.ndarray, t:np.ndarray) -> np.ndarray:
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


if __name__ == '__main__':
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    print(ReLU().forward(x))