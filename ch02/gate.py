from typing import Callable, Literal

Signal = Literal[0, 1]
Gate = Callable[[Signal, Signal], Signal]

def gate(w1: float, w2: float, b: float) -> Gate:

    def perceptron(x1: Signal, x2: Signal) -> Signal:
        return int(w1*x1 + w2*x2 + b > 0)

    return perceptron


AND = gate(0.5, 0.5, -0.7)
NAND = gate(-0.5, -0.5, 0.7)
OR = gate(0.7, 0.7, -0.1)

def XOR(x1: Signal, x2: Signal) -> Signal:
    return AND(NAND(x1, x2), OR(x1, x2))