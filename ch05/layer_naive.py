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

    def __init__(self) -> None:
        pass

    def forward(self, x: float, y: float) -> float:
        return x + y

    def backward(self, dout: float) -> tuple[float, float]:
        return dout, dout


if __name__ == "__main__":
    apple = 100.0
    apple_num = 2.0
    tax = 1.1

    mul = MulLayer()
    mul_tax = MulLayer()
    add = AddLayer()
    row_price = mul.forward(apple, apple_num)
    price = mul_tax.forward(row_price, tax)
    dp, dtax = mul_tax.backward(1)
    dapple, dnum = mul.backward(dp)
    print(dp, dtax, dapple, dnum)