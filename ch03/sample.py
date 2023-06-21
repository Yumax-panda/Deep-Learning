import numpy as np

def sigmoid(x :np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))


def init_network() -> dict[str, np.ndarray]:
    return {
        "W1": np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
        "b1": np.array([0.1, 0.2, 0.3]),
        "W2": np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
        "b2": np.array([0.1, 0.2]),
        "W3": np.array([[0.1, 0.3], [0.2, 0.4]]),
        "b3": np.array([0.1, 0.2])
    }


def forward(network: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    a1 = np.dot(x, network["W1"]) + network["b1"]
    a2 = np.dot(sigmoid(a1), network["W2"]) + network["b2"]
    return np.dot(sigmoid(a2), network["W3"]) + network["b3"]


if __name__ == "__main__":
    print(forward(init_network(), np.array([1.0, 0.5])))