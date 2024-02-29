from collections import Counter
from distances import euclidean_distance, manhattan_distance, chebyshev_distance
import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_neighbors_indices = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_neighbors_indices]
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
