import numpy as np
from sklearn.random_projection import GaussianRandomProjection

random_state = np.random.RandomState(0)
X = random_state.randn(10, 10000)
print(X.dtype)

X = np.array(X, dtype='float32')
print(X.dtype)

transformer = GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.dtype)
print(X.shape, X_new.shape)
