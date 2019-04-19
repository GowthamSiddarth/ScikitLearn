from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

clf = OneVsRestClassifier(estimator=SVC(random_state=0, gamma='auto'))
clf.fit(X, y)
print(clf.predict(X))
