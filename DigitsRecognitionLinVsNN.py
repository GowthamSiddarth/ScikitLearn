from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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

kNN_clf = KNeighborsClassifier()
kNN_clf.fit(digits_X_train, digits_y_train)

log_regr_clf = LogisticRegression()
log_regr_clf.fit(digits_X_train, digits_y_train)

print(log_regr_clf.score(digits_X_test, digits_y_test))
print(kNN_clf.score(digits_X_test, digits_y_test))
