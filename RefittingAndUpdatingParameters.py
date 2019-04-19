import numpy as np
from sklearn.svm import SVC

random_state = np.random.RandomState(0)
X = random_state.rand(10, 10)
y = random_state.binomial(1, 0.5, 10)
X_test = random_state.rand(5, 10)

clf = SVC(gamma='auto')
clf.set_params(kernel='linear').fit(X, y)
print(clf.predict(X_test))

clf.set_params(kernel='rbf').fit(X, y)
print(clf.predict(X_test))
