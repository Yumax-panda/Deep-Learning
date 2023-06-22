import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.mnist import load_mnist

(X_train, t_train), (X_test, t_test) = load_mnist(flatten=True, normalize=False)
print(X_train.shape)
print(t_train.shape)
print(X_test.shape)
print(t_test.shape)