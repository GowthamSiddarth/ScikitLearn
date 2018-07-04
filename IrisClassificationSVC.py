from sklearn import datasets, svm
from sklearn.externals import joblib

iris = datasets.load_iris()
X, y = iris.data[:-1], iris.target[:-1]

clf = svm.SVC()
clf.fit(X, y)

prediction = clf.predict(iris.data[-1:])
actual = iris.target[-1:]
print("prediction = " + str(prediction) + ", actual = " + str(actual))

joblib.dump(clf, "IrisClf.pkl")
