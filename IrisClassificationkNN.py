from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
iris_X, iris_y = iris.data, iris.target

random_state = np.random.RandomState(0)
indices = random_state.permutation(len(iris_X))
iris_X_train, iris_y_train, iris_X_test, iris_y_test = iris_X[indices[:-10]], iris_y[indices[:-10]], \
                                                       iris_X[indices[-10:]], iris_y[indices[-10:]]
