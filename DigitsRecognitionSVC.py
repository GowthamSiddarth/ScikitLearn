from sklearn import datasets, svm

digits = datasets.load_digits()
X, y = digits.data[:-1], digits.target[:-1]

clf = svm.SVC(gamma=0.1, C=100)
clf.fit(X, y)

res = clf.predict(digits.data[-1:])
print(res)
