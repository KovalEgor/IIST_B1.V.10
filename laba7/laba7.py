import numpy as np
import matplotlib.pyplot as plt

class Cognitron:
    def __init__(self, input_size, num_classes, learning_rate=0.1):
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size, num_classes)

    def predict(self, x):
        return np.dot(x, self.weights)

    def train(self, x, y, epochs=1000):
        for _ in range(epochs):
            for i in range(len(x)):
                output = self.predict(x[i])
                predicted_class = np.argmax(output)
                actual_class = np.argmax(y[i])
                if predicted_class != actual_class:
                    self.weights[:, predicted_class] -= self.learning_rate * x[i]
                    self.weights[:, actual_class] += self.learning_rate * x[i]

# Функция для генерации произвольных образов (например, квадратов и треугольников)
def generate_patterns(num_patterns=100):
    patterns = []
    labels = []
    for _ in range(num_patterns):
        shape_type = np.random.choice(['square', 'triangle'])
        if shape_type == 'square':
            patterns.append(np.random.rand(10) + 0.5)
            labels.append([1, 0])
        else:
            patterns.append(np.random.rand(10))
            labels.append([0, 1])
    return np.array(patterns), np.array(labels)

if __name__ == "__main__":
    X, y = generate_patterns(200)
    cognitron = Cognitron(input_size=10, num_classes=2)
    cognitron.train(X, y, epochs=1000)
    test_patterns, test_labels = generate_patterns(10)
    for test, label in zip(test_patterns, test_labels):
        prediction = cognitron.predict(test)
        predicted_class = np.argmax(prediction)
        print(f"Test: {test}, Predicted: {predicted_class}, Actual: {np.argmax(label)}")
