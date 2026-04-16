import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, filters, feature



################################## TASK 1: K-Nearest Neighbour (KNN) Algorithm. #########################################

def Q1_KNN(k, input_coordinates):
    training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
    training_labels = ['A', 'A', 'A', 'B', 'B']

    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

    def knn_predict(training_data, training_labels, test_point, k):
        distances = []
        for i in range(len(training_data)):
            distances.append(euclidean_distance(training_data[i], test_point))

        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = [training_labels[i] for i in nearest_indices]

        return nearest_labels[0]

    predictions = []
    for coordinate in input_coordinates:
        prediction = knn_predict(training_data, training_labels, coordinate, k)
        predictions.append(prediction)
        print(f"Point {coordinate} -> Predicted class: {prediction}")

    return predictions



################################## TASK 2: Softmax classification algorithm. #########################################

# Entropy loss function for multi-class classification
def cross_entropy_loss(predictions, actual):
    m = actual.shape[0]
    log_likelihood = -np.log(predictions[range(m), actual])
    loss = np.sum(log_likelihood) / m
    return loss

# Softmax function
def Softmax(input_vector):
    input_vector = np.asarray(input_vector)
    if input_vector.ndim == 1:
        shifted = input_vector - np.max(input_vector)
        exp_input = np.exp(shifted)
        return exp_input / np.sum(exp_input)

    shifted = input_vector - np.max(input_vector, axis=1, keepdims=True)
    exp_input = np.exp(shifted)
    return exp_input / np.sum(exp_input, axis=1, keepdims=True)

### Custom class for Softmax Classifier ###
class SoftmaxClassifier:
    """
    A simple implementation of a Softmax classifier for multi-class classification.

    Parameters:
        learning_rate: The learning rate for gradient descent.
        num_classes: The number of classes in the dataset.
        num_features: The number of features in the input data.

    Attributes:
        weights: The weight matrix for the classifier.
        bias: The bias vector for the classifier.
    """

    def __init__(self, learning_rate, num_classes, num_features):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(num_features, num_classes)
        self.bias = np.zeros((1, num_classes))
        self.classes = None

    def train(self, X, y, epochs=1000):
        self.classes = np.unique(y)
        y_indices = np.searchsorted(self.classes, y)
        loss_history = []
        accuracy_history = []
        previous_accuracy = None

        for epoch in range(epochs):
            # Forward pass
            logits = np.dot(X, self.weights) + self.bias
            probabilities = Softmax(logits)

            # Compute loss: call the function
            loss = cross_entropy_loss(probabilities, y_indices)

            # Backward pass (Gradient Descent)
            m = X.shape[0]
            grad_logits = probabilities.copy()
            grad_logits[range(m), y_indices] -= 1 # Gradient of loss with respect to logits
            grad_logits /= m # Average over the batch

            # Update weights with the dot product of input features and gradients
            self.weights -= self.learning_rate * np.dot(X.T, grad_logits)
            # Update bias with the sum of gradients across the batch
            self.bias -= self.learning_rate * np.sum(grad_logits, axis=0, keepdims=True)

            predictions = self.predict(X)
            accuracy = np.mean(predictions == y)

            loss_history.append(loss)
            accuracy_history.append(accuracy)

            if previous_accuracy is None or accuracy != previous_accuracy:
                print(f"Epoch {epoch} - Loss: {loss} - Accuracy: {accuracy:.3f}")
                previous_accuracy = accuracy

        return loss_history, accuracy_history
    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        probabilities = Softmax(logits)
        predicted_indices = np.argmax(probabilities, axis=1)

        if self.classes is None:
            return predicted_indices

        return self.classes[predicted_indices]

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

if __name__ == "__main__":

    ##### Input for Question 1: KNN Algorithm #####
    k = 3
    Q1_input = [[1, 1], [3, 7], [10,10]]
    Q1_KNN(k, Q1_input)

    # #### Input for Question 2: Softmax Function #####
    # input_coordinates = np.array([[1, 2], [2, 1], [3, 1], [1, 3], [2, 3], [3, 2]])
    # input_labels = np.array(['A', 'A', 'B', 'B', 'C', 'C']) # 3 classes

    # classifier = SoftmaxClassifier(learning_rate=0.2, num_classes=3, num_features=2)
    # loss_history, accuracy_history = classifier.train(input_coordinates, input_labels, epochs=2000)

    # print(f"Final training accuracy: {classifier.accuracy(input_coordinates, input_labels):.3f}")

    # plt.figure(figsize=(8, 4))
    # plt.plot(accuracy_history, label="Training Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.title("Softmax Classifier Accuracy Over Time")
    # plt.ylim(0, 1.05)
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # Test the classifier with new data points
    # test_points = np.array([[1, 1], [1, 1], [3, 3]])
    # predictions = classifier.predict(test_points)
    # print("\nPredictions for test points:")
    # print(predictions)


