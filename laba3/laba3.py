import numpy as np

# Функция активации (сигмоида)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная сигмоиды для обратного распространения
def sigmoid_derivative(x):
    return x * (1 - x)

# Многослойный перцептрон
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) - 0.5
        self.bias_hidden = np.random.rand(hidden_size) - 0.5
        self.bias_output = np.random.rand(output_size) - 0.5
        self.lr = lr

    # Прямой проход
    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    # Обратное распространение
    def backward(self, error):
        d_output = error * sigmoid_derivative(self.final_output)
        d_hidden = d_output.dot(self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_output)

        # Обновление весов и смещений (bias)
        self.weights_hidden_output += self.lr * np.dot(self.hidden_output.T, d_output)
        self.weights_input_hidden += self.lr * np.dot(self.inputs.T, d_hidden)
        self.bias_output += self.lr * d_output.sum(axis=0)
        self.bias_hidden += self.lr * d_hidden.sum(axis=0)

# Создаем обучающие данные для XOR
def create_xor_data():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])
    return X_train, y_train

# Обучаем многослойный перцептрон
def train_mlp(mlp, X_train, y_train, epochs=10000):
    for epoch in range(epochs):
        outputs = mlp.forward(X_train)
        error = y_train - outputs
        mlp.backward(error)
        if epoch % 1000 == 0:
            loss = np.mean(np.abs(error))
            print(f"Эпоха {epoch}/{epochs}, Ошибка: {loss:.4f}")

# Проверяем результаты на обученных данных
def test_mlp(mlp, X_train, y_train):
    outputs = mlp.forward(X_train)
    predictions = np.round(outputs)
    print("\nТестирование на обучающих данных:")
    for inputs, prediction, expected in zip(X_train, predictions, y_train):
        print(f"Вход: {inputs}, Предсказано: {int(prediction)}, Ожидалось: {int(expected)}")

# Основная программа
if __name__ == "__main__":
    # Создаем многослойный перцептрон с 2 входами, 2 нейронами в скрытом слое и 1 выходом
    mlp = MLP(input_size=2, hidden_size=2, output_size=1, lr=0.1)
    
    # Генерируем данные для XOR
    X_train, y_train = create_xor_data()
    
    # Обучаем многослойный перцептрон
    train_mlp(mlp, X_train, y_train, epochs=10000)
    
    # Проверка результатов
    test_mlp(mlp, X_train, y_train)
