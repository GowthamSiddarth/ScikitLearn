from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

digits = load_digits()
digits_X, digits_y = digits.data, digits.target

random_state = np.random.RandomState(0)
indices = random_state.permutation(len(digits_X))

test_size = 0.1
partition = int(test_size * len(digits_X))
digits_X_train, digits_y_train, digits_X_test, digits_y_test = \
    digits_X[indices[:-partition]], digits_y[indices[:-partition]], \
    digits_X[indices[-partition:]], digits_y[indices[-partition:]]

parameters_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
]

clf = GridSearchCV(SVC(), param_grid=parameters_grid, cv=3, n_jobs=-1)
clf.fit(digits_X_train, digits_y_train)

print("Best score: ", clf.best_score_)
print("Best params: ", clf.best_params_)
print("Score on test set: ", clf.score(digits_X_test, digits_y_test))
