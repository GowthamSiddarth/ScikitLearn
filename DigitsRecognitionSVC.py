from sklearn import datasets, svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()
X, y = digits.data[:-1], digits.target[:-1]

clf = svm.SVC(gamma=0.1, C=100)
clf.fit(X, y)

prediction = clf.predict(digits.data[-1:])
actual = digits.target[-1:]
print("prediction = " + str(prediction) + ", actual = " + str(actual))

plt.matshow(digits.images[-1])
plt.show()
