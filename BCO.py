import numpy as np
import time


def BCO(initial_population, fitness_function, lb, ub, max_iterations):
    # Border Collie Optimization (BCO)

    search_range = [lb, ub]
    population_size, n_var = initial_population.shape

    fitness_values = np.zeros(population_size)
    personal_best_positions = np.copy(initial_population)
    personal_best_fitness_values = np.full(population_size, np.inf)

    convergence_curve = []
    start_time = time.time()

    # Main loop
    for iteration in range(max_iterations):
        # Calculate fitness values
        for i in range(population_size):
            fitness_values[i] = fitness_function(initial_population[i])

        # Update personal best positions
        for i in range(population_size):
            if fitness_values[i] < personal_best_fitness_values[i]:
                personal_best_positions[i] = initial_population[i]
                personal_best_fitness_values[i] = fitness_values[i]

        # Update global best position
        global_best_index = np.argmin(fitness_values)
        global_best_position = initial_population[global_best_index]

        # Update positions of search agents
        for i in range(population_size):
            if np.random.rand() < 0.5:
                # Herding behavior
                for j in range(n_var):
                    if np.random.rand() < 0.5:
                        personal_best_position = personal_best_positions[i]
                    else:
                        personal_best_position = personal_best_positions[np.random.randint(population_size)]

                    initial_population[i, j] = personal_best_position[j] + np.random.rand() * (
                            global_best_position[j] - personal_best_position[j]
                    )
            else:
                # Sheep behavior
                for j in range(n_var):
                    if np.random.rand() < 0.5:
                        neighbor_position = personal_best_positions[np.random.randint(population_size)]
                    else:
                        neighbor_position = global_best_position

                    initial_population[i, j] = neighbor_position[j] + np.random.rand() * (
                            neighbor_position[j] - initial_population[i, j]
                    )

            # Boundary checking
            initial_population[i] = np.clip(initial_population[i], search_range[0][i], search_range[1][i])

        # Record convergence
        optimal_fitness = np.min(fitness_values)
        convergence_curve.append(optimal_fitness)

        print(f"Iteration {iteration + 1}/{max_iterations}, Best Fitness: {optimal_fitness}")

    optimal_fitness = np.min(fitness_values)
    optimal_solution_index = np.argmin(fitness_values)
    optimal_solution = initial_population[optimal_solution_index]
    elapsed_time = time.time() - start_time
    return optimal_solution, convergence_curve, optimal_fitness, elapsed_time
