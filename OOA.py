import numpy as np
import time


# Orangutan Optimization Algorithm (OOA)
def OOA(X, fobj, lb, ub, max_iter):
    n_agents, dim = X.shape
    fitness = np.apply_along_axis(fobj, 1, X)
    best_idx = np.argmin(fitness)
    best_pos = X[best_idx].copy()
    best_fit = fitness[best_idx]

    convergence_curve = np.zeros((max_iter))
    ct = time.time()
    for t in range(1, max_iter + 1):
        for i in range(n_agents):
            # Phase 1: Foraging Strategy (Exploration)
            FSi = np.array([X[k] for k in range(n_agents) if fitness[k] < fitness[i] and k != i])
            if FSi.size > 0:
                SFS = FSi[np.random.randint(FSi.shape[0])]
                r = np.random.rand()
                I = np.random.choice([1, 2])
                X_P1 = X[i] + r * (SFS - I * X[i])
                X_P1 = np.clip(X_P1, lb[i], ub[i])
                fit_P1 = fobj(X_P1)
                if fit_P1 < fitness[i]:
                    X[i] = X_P1
                    fitness[i] = fit_P1

        for i in range(n_agents):
            # Phase 2: Nesting Skill (Exploitation)
            for j in range(dim):
                r = np.random.rand()
                X_P2 = X[i][j] + (1 - 2 * r) * ((ub[i, j] - lb[i, j]) / t)
                X_P2 = np.clip(X_P2, lb[i, j], ub[i, j])
                temp = X[i].copy()
                temp[j] = X_P2
                fit_P2 = fobj(temp)
                if fit_P2 < fitness[i]:
                    X[i][j] = X_P2
                    fitness[i] = fit_P2

        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fit:
            best_fit = fitness[best_idx]
            best_pos = X[best_idx].copy()

        convergence_curve[t - 1] = best_fit
        print(f"Iteration {t}: Best Fitness = {best_fit}")
    ct = time.time() - ct
    return best_fit, convergence_curve, best_pos, ct
