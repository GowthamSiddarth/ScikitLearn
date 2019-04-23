from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC

digits = load_digits()
digits_X, digits_y = digits.data, digits.target

k_fold, clf = KFold(n_splits=3), SVC(kernel='linear')
cross_val_scores = cross_val_score(estimator=clf, n_jobs=-1, X=digits_X, y=digits_y, cv=k_fold)
print(cross_val_scores)
