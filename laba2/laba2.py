import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Загружаем датасет MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Преобразуем размер изображений с 28x28 до 10x10
def resize_images(images, new_size=(10, 10)):
    images_resized = np.zeros((images.shape[0], new_size[0], new_size[1]))
    for i in range(images.shape[0]):
        img = array_to_img(images[i].reshape(28, 28, 1))
        img = img.resize(new_size)
        images_resized[i] = img_to_array(img).reshape(new_size[0], new_size[1])
    return images_resized

x_train_resized = resize_images(x_train)
x_test_resized = resize_images(x_test)

# Нормализация данных
x_train_resized = x_train_resized.astype('float32') / 255.0
x_test_resized = x_test_resized.astype('float32') / 255.0

# Преобразуем изображения в векторы длиной 100
x_train_flatten = x_train_resized.reshape(-1, 100)
x_test_flatten = x_test_resized.reshape(-1, 100)

# Преобразуем метки классов в категориальный формат (one-hot encoding)
y_train_categorical = to_categorical(y_train, num_classes=10)
y_test_categorical = to_categorical(y_test, num_classes=10)

# Создаем модель однослойного перцептрона
model = Sequential()
model.add(Dense(10, input_dim=100, activation='softmax'))

# Компилируем модель
model.compile(optimizer=SGD(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучаем модель
model.fit(x_train_flatten, y_train_categorical, epochs=10, batch_size=32, validation_data=(x_test_flatten, y_test_categorical))

# Оцениваем модель на тестовых данных
test_loss, test_accuracy = model.evaluate(x_test_flatten, y_test_categorical)
print(f"\nТочность на тестовых данных: {test_accuracy * 100:.2f}%")
