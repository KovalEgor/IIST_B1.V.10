import numpy as np

def objective_function(x, y):
    return 1 / (1 + x**2 + y**2)

def simulated_annealing(objective, bounds, iterations, step_size, temp, cooling_rate):
    best = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    best_eval = objective(best[0], best[1])
    current, current_eval = best, best_eval

    for i in range(iterations):

        candidate = current + np.random.randn(len(bounds)) * step_size
        candidate_eval = objective(candidate[0], candidate[1])

        if candidate_eval > best_eval:
            best, best_eval = candidate, candidate_eval

        diff = candidate_eval - current_eval
        metropolis = np.exp(diff / temp) if diff < 0 else 1

        if diff > 0 or np.random.rand() < metropolis:
            current, current_eval = candidate, candidate_eval

        temp *= cooling_rate

    return best, best_eval

bounds = np.array([[-10, 10], [-10, 10]])
iterations = 1000
step_size = 0.1
initial_temp = 10
cooling_rate = 0.99

best_solution, best_value = simulated_annealing(
    objective_function, bounds, iterations, step_size, initial_temp, cooling_rate
)
print(f"Лучшее решение: x = {best_solution[0]:.5f}, y = {best_solution[1]:.5f}")
print(f"Максимальное значение функции: {best_value:.5f}")
