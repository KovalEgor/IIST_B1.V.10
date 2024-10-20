import numpy as np
import matplotlib.pyplot as plt

# Класс для реализации сети Хопфилда
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    # Метод обучения
    def train(self, patterns):
        num_patterns = len(patterns)
        for p in patterns:
            self.weights += np.outer(p, p)
        self.weights /= num_patterns
        np.fill_diagonal(self.weights, 0)

    def update(self, pattern, steps=10):
        for _ in range(steps):
            for i in np.random.permutation(self.size):
                summation = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if summation >= 0 else -1
        return pattern

    # Метод восстановления поврежденного изображения
    def recover(self, damaged_pattern, steps=100):
        return self.update(damaged_pattern.copy(), steps)

# Функция для преобразования изображения в вектор
def image_to_vector(image):
    return np.where(image > 0, 1, -1).flatten()

# Функция для преобразования вектора обратно в изображение
def vector_to_image(vector, shape):
    return vector.reshape(shape)

# Функция для отображения изображений
def display_image(image, title):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis('off')
    plt.show()

def create_training_data():
    # Создание обучающих данных (цифры от 0 до 9)
    zero = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    one = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    two = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    three = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    four = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                     [0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    five = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    six = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    seven = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    eight = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    nine = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    return [
        image_to_vector(zero), image_to_vector(one), image_to_vector(two),
        image_to_vector(three), image_to_vector(four), image_to_vector(five),
        image_to_vector(six), image_to_vector(seven), image_to_vector(eight),
        image_to_vector(nine)
    ]

# Функция для порчи изображения (замена части пикселей на случайные значения)
def damage_image(image, damage_ratio=0.3):
    damaged_image = image.copy()
    num_pixels_to_flip = int(damage_ratio * len(damaged_image))
    flip_indices = np.random.choice(len(damaged_image), num_pixels_to_flip, replace=False)
    damaged_image[flip_indices] *= -1
    return damaged_image

# Основная программа
if __name__ == "__main__":
    hopfield = HopfieldNetwork(size=100)
    training_patterns = create_training_data()
    hopfield.train(training_patterns)

    # Выбор поврежденного изображения для восстановления
    original_image = training_patterns[3]  # Например, цифра "3"
    damaged_image = damage_image(original_image, damage_ratio=0.3)

    display_image(vector_to_image(original_image, (10, 10)), "Оригинальная цифра")
    display_image(vector_to_image(damaged_image, (10, 10)), "Поврежденная цифра")

    # Восстановление поврежденного изображения
    recovered_image = hopfield.recover(damaged_image, steps=10)

    # Отображение восстановленного изображения
    display_image(vector_to_image(recovered_image, (10, 10)), "Восстановленная цифра")
