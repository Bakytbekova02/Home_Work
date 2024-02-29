from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer
from distances import euclidean_distance, manhattan_distance, chebyshev_distance
from knn import KNNClassifier

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Iterate over different random_state values
for random_state in [42, 1]:
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Initialize KNN classifier
    for k in [3, 5, 7]:  # Iterate over different values of k
        knn_classifier = KNNClassifier(k=k)
        knn_classifier.fit(X_train, y_train)

        # Predictions
        y_pred = [knn_classifier.predict(x) for x in X_test]

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Random State: {random_state}, k: {k}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("====================")


