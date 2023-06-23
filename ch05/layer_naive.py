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
    orange = 150.0
    apple_num = 2.0
    orange_num = 3
    tax = 1.1

    apple_layer = MulLayer()
    orange_layer = MulLayer()
    total = AddLayer()
    tax_layer = MulLayer()

    # forward
    apple_price = apple_layer.forward(apple, apple_num)
    orange_price = orange_layer.forward(orange, orange_num)
    total_price = total.forward(apple_price, orange_price)
    tax_price = tax_layer.forward(tax, total_price)

    # backward
    dtax, dtotal_price = tax_layer.backward(1)
    dapple_price, dorange_price = total.backward(dtotal_price)
    dapple, dapple_num = apple_layer.backward(dapple_price)
    dorange, dorange_num = orange_layer.backward(dorange_price)

    print(apple_price, orange_price, total_price, tax_price)
    print(dapple_price, dorange_price, dtotal_price, dtax, dapple_num, dorange_num, dapple, dorange)