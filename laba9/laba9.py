import numpy as np
import pygame

# Константы
WIDTH, HEIGHT = 800, 600
FPS = 30
HERBIVORE_ENERGY = 50
PREDATOR_ENERGY = 100
ENERGY_COST_PER_MOVE = 1
REPRODUCTION_ENERGY_HERBIVORE = 80
REPRODUCTION_ENERGY_PREDATOR = 150

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = 100

    def update(self):
        # Обновление состояния агента
        if self.energy <= 0:
            return False  # Умер
        return True  # Жив

    def draw(self, screen):
        pass  # Реализация в дочерних классах

class Plant(Agent):
    def grow(self):
        # Логика роста растения
        if self.energy < 100:
            self.energy += 1  # Растение восстанавливает энергию

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), (self.x, self.y), 5)  # Зеленый цвет для растений


class Herbivore(Agent):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.perceptron = Perceptron(3)

    def eat(self, plants):
        for plant in plants:
            if np.hypot(self.x - plant.x, self.y - plant.y) < 10:  # Если близко к растению
                self.energy += 20  # Восстанавливает энергию
                plants.remove(plant)  # Съедаем растение
                break

    def reproduce(self):
        if self.energy >= REPRODUCTION_ENERGY_HERBIVORE:  # Если достаточно энергии
            self.energy -= REPRODUCTION_ENERGY_HERBIVORE  # Расходуем энергию на размножение
            return Herbivore(self.x, self.y)  # Создаем нового травоядного
        return None

    def move(self):
        inputs = np.array([self.energy, 0, 0])  # Заполняем входы
        action = self.perceptron.predict(inputs)

        if action == 1:  # Двигаемся
            self.x += np.random.randint(-5, 6)
            self.y += np.random.randint(-5, 6)
            # Удерживаем агента в пределах границ
            self.x = np.clip(self.x, 0, WIDTH)
            self.y = np.clip(self.y, 0, HEIGHT)

    def update(self, plants):
        self.move()
        self.eat(plants)
        new_herbivore = self.reproduce()  # Проверка на возможность размножения
        if new_herbivore:
            return new_herbivore
        return super().update()  # Проверка на смерть

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 0), (self.x, self.y), 7)  # Желтый цвет для травоядных


class Predator(Agent):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.perceptron = Perceptron(3)  # 3 входа: энергия, ближайшее травоядное, расстояние до ближайшего травоядного

    def hunt(self, herbivores):
        for herbivore in herbivores:
            if np.hypot(self.x - herbivore.x, self.y - herbivore.y) < 10:  # Если близко к травоядному
                self.energy += 30  # Восстанавливает энергию
                herbivores.remove(herbivore)  # Съедаем травоядное
                break

    def move(self):
        inputs = np.array([self.energy, 0, 0])  # Заполняем входы
        action = self.perceptron.predict(inputs)

        if action == 1:  # Двигаемся
            self.x += np.random.randint(-5, 6)
            self.y += np.random.randint(-5, 6)
            # Удерживаем агента в пределах границ
            self.x = np.clip(self.x, 0, WIDTH)
            self.y = np.clip(self.y, 0, HEIGHT)

    def update(self, herbivores):
        self.move()
        self.hunt(herbivores)
        new_predator = self.reproduce()  # Проверка на возможность размножения
        if new_predator:
            return new_predator
        return super().update()  # Проверка на смерть
    
    def reproduce(self):
        if self.energy >= REPRODUCTION_ENERGY_PREDATOR:  # Если достаточно энергии
            self.energy -= REPRODUCTION_ENERGY_PREDATOR  # Расходуем энергию на размножение
            return Predator(self.x, self.y)  # Создаем нового хищника
        return None

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), 9)  # Красный цвет для хищников

class Ecosystem:
    def __init__(self):
        self.plants = [Plant(np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)) for _ in range(50)]
        self.herbivores = [Herbivore(np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)) for _ in range(20)]
        self.predators = [Predator(np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)) for _ in range(10)]

    def update(self):
        # Обновление всех агентов
        self.plants = [plant for plant in self.plants if plant.update()]
        for herbivore in self.herbivores:
            herbivore.update(self.plants)
        self.herbivores = [herbivore for herbivore in self.herbivores if herbivore.update(self.plants)]
        for predator in self.predators:
            predator.update(self.herbivores)
        self.predators = [predator for predator in self.predators if predator.update(self.herbivores)]

    def draw(self, screen):
        for plant in self.plants:
            plant.draw(screen)
        for herbivore in self.herbivores:
            herbivore.draw(screen)
        for predator in self.predators:
            predator.draw(screen)

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.learning_rate = 0.1

    def predict(self, inputs):
        return 1 if np.dot(inputs, self.weights) > 0 else 0

    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = target - prediction
        self.weights += self.learning_rate * error * inputs
        
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    ecosystem = Ecosystem()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # Очистка экрана
        ecosystem.update()
        ecosystem.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
