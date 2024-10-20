import numpy as np

class HammingNetwork:
    def __init__(self, patterns):
        self.patterns = np.array(patterns)
        self.num_patterns = self.patterns.shape[0]
        self.num_neurons = self.patterns.shape[1]
        self.w1 = self.patterns
        self.b1 = self.num_neurons / 2
        self.w2 = np.ones((self.num_patterns, self.num_patterns)) - np.eye(self.num_patterns)
        self.y = np.zeros(self.num_patterns)

    def layer1(self, x):
        net = np.dot(self.w1, x)
        net = (net + 1) / (np.max(net) + 1)
        print(f"Нормализованные расстояния Хэмминга для каждого шаблона: {net}")
        return net

    def layer2(self, net1):
        self.y = net1.copy()
        prev_y = np.zeros_like(self.y)
        iterations = 0
        print(f"Начальные значения y: {self.y}")
        while not np.allclose(self.y, prev_y, atol=0.91) and iterations < 1000:
            prev_y = self.y.copy()
            for i in range(self.num_patterns):
                inhibition = np.dot(self.w2[i], self.y)
                self.y[i] = max(self.y[i] - inhibition * 0.015, 0)
            iterations += 1
            print(f"Iteration {iterations}, y: {self.y}")
        max_index = np.argmax(self.y)
        max_value = self.y[max_index]
        print(f"Максимальное значение y: {max_value} для индекса {max_index}")
        if max_value < 0.5:
            return -1
        return max_index

    def recognize(self, x):
        net1 = self.layer1(x)
        print(f"Расстояния Хэмминга: {net1}")
        return self.layer2(net1)

# Функция для преобразования изображения 10x10 в вектор длиной 100
def image_to_vector(image):
    return image.flatten()

def create_reference_patterns():
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

    three = np.array([  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
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

    nine = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                     1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                     1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                     1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                     1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                     0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    return [image_to_vector(zero), image_to_vector(one), image_to_vector(two), image_to_vector(three), image_to_vector(four), image_to_vector(five), 
            image_to_vector(six), image_to_vector(seven), image_to_vector(eight), image_to_vector(nine)]

if __name__ == "__main__":
    reference_patterns = create_reference_patterns()
    hamming_network = HammingNetwork(reference_patterns)
    recovered_image = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    recovered_image = image_to_vector(recovered_image)
    print(f"Размер восстановленного образа: {recovered_image.shape}")
    result = hamming_network.recognize(recovered_image)
    print(f"Распознанный образ: {result}")
    
