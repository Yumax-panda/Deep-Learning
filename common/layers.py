import numpy as np
from typing import TYPE_CHECKING, Optional

class ReLU:

    if TYPE_CHECKING:
        mask: Optional[np.NDArray[np.bool_]]

    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask.astype(np.float_)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask.astype(np.float_)


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


if __name__ == '__main__':
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    print(ReLU().forward(x))