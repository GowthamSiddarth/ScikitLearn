from sklearn import datasets, svm
import numpy as np

digits = datasets.load_digits()
digits_X, digits_y = digits.data, digits.target

digits_X_fold, digits_y_fold = np.array_split(digits_X, 3), np.array_split(digits_y, 3)
scores = []

for fold in range(3):
    digits_X_train, digits_y_train = list(digits_X_fold), list(digits_y_fold)
    digits_X_test, digits_y_test = digits_X_train.pop(fold), digits_y_train.pop(fold)
    digits_X_train, digits_y_train = np.concatenate(digits_X_train), np.concatenate(digits_y_train)
    clf = svm.SVC(kernel='linear').fit(digits_X_train, digits_y_train)
    scores.append(clf.score(digits_X_test, digits_y_test))

print(scores)
