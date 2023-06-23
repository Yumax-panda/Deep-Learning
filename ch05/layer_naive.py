from typing import TYPE_CHECKING, Optional


class MulLayer:

    if TYPE_CHECKING:
        x: Optional[float]
        y: Optional[float]

    def __init__(self) -> None:
        self.x = None
        self.y = None

    def forward(self, x: float, y: float) -> float:
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout: float) -> tuple[float, float]:
        return dout*self.y, dout*self.x


class AddLayer:

    if TYPE_CHECKING:
        x: Optional[float]
        y: Optional[float]

    def __init__(self) -> None:
        self.x = None
        self.y = None

    def forward(self, x: float, y: float) -> float:
        self.x = x
        self.y = y
        return x + y

    def backward(self, dout: float) -> tuple[float, float]:
        return dout, dout