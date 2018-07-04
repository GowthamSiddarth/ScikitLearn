from sklearn import datasets, svm

digits = datasets.load_digits()
X, y = digits.data[:-1], digits.target[:-1]

clf = svm.SVC(gamma=0.1, C=100)
clf.fit(X, y)

prediction = clf.predict(digits.data[-1:])
actual = digits.target[-1:]
print("prediction = " + str(prediction) + ", actual = " + str(actual))
