import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from sklearn import datasets, svm

random_state = np.random.RandomState(0)
X = random_state.randn(10, 10000)
print(X.dtype)

X = np.array(X, dtype='float32')
print(X.dtype)

transformer = GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.dtype)
print(X.shape, X_new.shape)

iris = datasets.load_iris()
clf = svm.SVC(gamma='auto')

clf.fit(iris.data, iris.target)
print(clf.predict(iris.data[:3]))

clf.fit(iris.data, iris.target_names[iris.target])
print(clf.predict(iris.data[:3]))
