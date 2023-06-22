import numpy as np

def softmax(a: np.ndarray) -> np.ndarray:
    c = np.max(a)
    total = np.sum(np.exp(a-c))
    return np.exp(a-c)/total

if __name__ == '__main__':
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)
    print(np.sum(y))