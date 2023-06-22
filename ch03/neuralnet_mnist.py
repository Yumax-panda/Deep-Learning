import sys, os
import numpy as np
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.mnist import load_mnist
from common.sigmoid import sigmoid
from common.softmax import softmax

DIR = f"{os.getcwd()}\\ch03"

def get_data() -> tuple[np.ndarray, np.ndarray]:
    (_, _), (x_test, t_test) = load_mnist(flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network() -> dict[str, np.ndarray]:
    with open(f"{DIR}\\sample_weight.pkl", "rb") as f:
        return pickle.load(f)


def predict(network: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    w1, w2, w3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, w1) + b1
    a2 = np.dot(sigmoid(a1), w2) + b2
    a3 = np.dot(sigmoid(a2), w3) + b3

    return softmax(a3)


if __name__ == "__main__":
    x, t = get_data()
    network = init_network()
    count = 0

    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        count += int(p==t[i])

    print(f"Accuracy: {float(count)/len(x):.4f}")