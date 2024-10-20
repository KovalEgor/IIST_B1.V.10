import numpy as np

# Функция активации (шаговая функция)
def activation_function(x):
    return 1 if x >= 0 else 0

# Однонейронный перцептрон
class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.weights = np.zeros(input_size + 1)
        self.lr = lr

    # Метод предсказания
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return activation_function(summation)

    # Метод обучения перцептрона
    def train(self, training_inputs, labels, epochs=10):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.lr * error * inputs
                self.weights[0] += self.lr * error

    # Метод проверки результатов
    def test(self, test_inputs, labels):
        print("\nПроверка перцептрона:")
        correct_predictions = 0
        for inputs, label in zip(test_inputs, labels):
            prediction = self.predict(inputs)
            print(f"Вход: {inputs} -> Предсказано: {prediction}, Ожидалось: {label}")
            if prediction == label:
                correct_predictions += 1
        accuracy = correct_predictions / len(labels) * 100
        print(f"Точность: {accuracy:.2f}%\n")

# Логическая функция И
def logic_and():
    print("Обучение логической функции И:")
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 0, 0, 1])
    perceptron = Perceptron(input_size=2)
    perceptron.train(training_inputs, labels)
    perceptron.test(training_inputs, labels)

# Логическая функция ИЛИ
def logic_or():
    print("Обучение логической функции ИЛИ:")
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 1])
    perceptron = Perceptron(input_size=2)
    perceptron.train(training_inputs, labels)
    perceptron.test(training_inputs, labels)

# Логическая функция НЕ
def logic_not():
    print("Обучение логической функции НЕ:")
    training_inputs = np.array([[0], [1]])
    labels = np.array([1, 0])
    perceptron = Perceptron(input_size=1)
    perceptron.train(training_inputs, labels)
    perceptron.test(training_inputs, labels)

# Обучаем и проверяем перцептрон логическим функциям
logic_and()
logic_or()
logic_not()
