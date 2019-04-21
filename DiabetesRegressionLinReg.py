from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import numpy as np

diabetes = load_diabetes()
diabetes_X, diabetes_y = diabetes.data, diabetes.target

random_state = np.random.RandomState(0)
indices = random_state.permutation(len(diabetes_X))
diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test = \
    diabetes_X[indices[:-10]], diabetes_y[indices[:-10]], diabetes_X[indices[-10:]], diabetes_y[indices[-10:]]

regr = LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

print(regr.coef_)

print("RMSE = ", np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
print("Variance score = ", regr.score(diabetes_X_test, diabetes_y_test))
