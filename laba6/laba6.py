import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Сеть с обратным распространением ошибки
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        limit_input_hidden = np.sqrt(6 / (self.input_size + self.hidden_size))
        self.weights_input_hidden = np.random.uniform(-limit_input_hidden, limit_input_hidden, (self.input_size, self.hidden_size))
        limit_hidden_output = np.sqrt(6 / (self.hidden_size + self.output_size))
        self.weights_hidden_output = np.random.uniform(-limit_hidden_output, limit_hidden_output, (self.hidden_size, self.output_size))

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, x):
        self.hidden_layer_input = np.dot(x, self.weights_input_hidden)
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.output = self.softmax(self.output_layer_input)
        return self.output

    def backward(self, x, y):
        output_error = y - self.output
        output_delta = output_error
        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += x.T.dot(hidden_layer_delta) * self.learning_rate

    def train(self, x, y, epochs=5000):
        for epoch in range(epochs):
            self.forward(x)
            self.backward(x, y)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.output))
                accuracy = accuracy_score(np.argmax(y, axis=1), np.argmax(self.output, axis=1))
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

# Однослойный перцептрон
class SingleLayerPerceptron:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = np.random.rand(self.input_size, self.output_size)

    def predict(self, x):
        return self.sigmoid(np.dot(x, self.weights))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, x, y, epochs=5000):
        for _ in range(epochs):
            predictions = self.predict(x)
            error = y - predictions
            adjustment = self.learning_rate * np.dot(x.T, error)
            self.weights += adjustment

digits = load_digits()
X = digits.data / 16.0
y = digits.target
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
nn = NeuralNetwork(input_size=64, hidden_size=30, output_size=10)
nn.train(X_train, y_train, epochs=5000)
y_pred_nn = nn.forward(X_test)
y_pred_nn_labels = np.argmax(y_pred_nn, axis=1)
slp = SingleLayerPerceptron(input_size=64, output_size=10)
slp.train(X_train, y_train, epochs=5000)
y_pred_slp = slp.predict(X_test)
y_pred_slp_labels = np.argmax(y_pred_slp, axis=1)
accuracy_nn = accuracy_score(np.argmax(y_test, axis=1), y_pred_nn_labels)
accuracy_slp = accuracy_score(np.argmax(y_test, axis=1), y_pred_slp_labels)
print(f'Accuracy Neural Network: {accuracy_nn * 100:.2f}%')
print(f'Accuracy Single Layer Perceptron: {accuracy_slp * 100:.2f}%')
