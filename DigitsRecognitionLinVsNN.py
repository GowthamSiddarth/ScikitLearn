from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
digits_X, digits_y = digits.data, digits.target

random_state = np.random.RandomState(0)
indices = random_state.permutation(len(digits_X))

test_size = 0.1
partition = int(-test_size * len(digits_X))
digits_X_train, digits_y_train, digits_X_test, digits_y_test = \
    digits_X[indices[:partition]], digits_y[indices[:partition]], \
    digits_X[indices[partition:]], digits_y[indices[partition:]]
