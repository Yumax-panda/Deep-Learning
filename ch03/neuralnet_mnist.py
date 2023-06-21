import os, sys
sys.path.append(os.pardir)

from dataset.mnist import load_mnist

(X_train, t_train), (X_test, t_test) = load_mnist(flatten=True, normalize=False)
print(X_train.shape)
print(t_train.shape)
print(X_test.shape)
print(t_test.shape)