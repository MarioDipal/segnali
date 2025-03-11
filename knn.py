import numpy as np

def knn (x_train, y_train, x_test, k):
    distances = np.linalg.norm(x_train - x_test, axis=1)
    indices = np.argsort(distances)[:k]
    counts = np.bincount(y_train[indices].astype(int))
    return np.argmax(counts)

