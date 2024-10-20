import numpy as np
import matplotlib.pyplot as plt

# Функция для максимизации
def fitness_function(x, y):
    return 1 / (1 + x**2 + y**2)

# Генетический алгоритм
class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, selection_method):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.population = np.random.uniform(-10, 10, (population_size, 2))
        self.best_fitnesses = []

    def select(self):
        if self.selection_method == 'elitism':
            fitness_scores = np.array([fitness_function(ind[0], ind[1]) for ind in self.population])
            best_indices = np.argsort(fitness_scores)[-2:]
            return self.population[best_indices]
        elif self.selection_method == 'roulette':
            fitness_scores = np.array([fitness_function(ind[0], ind[1]) for ind in self.population])
            fitness_sum = np.sum(fitness_scores)
            probabilities = fitness_scores / fitness_sum
            selected_indices = np.random.choice(range(self.population_size), size=2, p=probabilities)
            return self.population[selected_indices]

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            individual += np.random.normal(0, 1, 2)
        return individual

    def evolve(self):
        for generation in range(self.generations):
            new_population = []
            for _ in range(self.population_size // 2):
                parents = self.select()
                child1 = np.mean(parents, axis=0)
                child2 = np.mean(parents[::-1], axis=0)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))
            self.population = np.array(new_population)
            best_fitness = max([fitness_function(ind[0], ind[1]) for ind in self.population])
            self.best_fitnesses.append(best_fitness)

    def run(self):
        self.evolve()
        best_index = np.argmax([fitness_function(ind[0], ind[1]) for ind in self.population])
        best_solution = self.population[best_index]
        best_fitness = fitness_function(best_solution[0], best_solution[1])
        return best_solution, best_fitness

population_size = 100
generations = 100
mutation_rate = 0.1
ga_elitism = GeneticAlgorithm(population_size, generations, mutation_rate, 'elitism')
best_solution_elitism, best_fitness_elitism = ga_elitism.run()
ga_roulette = GeneticAlgorithm(population_size, generations, mutation_rate, 'roulette')
best_solution_roulette, best_fitness_roulette = ga_roulette.run()

print(f'Элитный отбор: x = {best_solution_elitism[0]:.4f}, y = {best_solution_elitism[1]:.4f}, f(x, y) = {best_fitness_elitism:.4f}')
print(f'Рулеточный отбор: x = {best_solution_roulette[0]:.4f}, y = {best_solution_roulette[1]:.4f}, f(x, y) = {best_fitness_roulette:.4f}')

# Построение графиков
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(ga_elitism.best_fitnesses, label='Элитный отбор')
plt.title('Эволюция лучших значений (Элитный отбор)')
plt.xlabel('Поколения')
plt.ylabel('Лучшее значение')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(ga_roulette.best_fitnesses, label='Рулеточный отбор', color='orange')
plt.title('Эволюция лучших значений (Рулеточный отбор)')
plt.xlabel('Поколения')
plt.ylabel('Лучшее значение')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
