import numpy as np
import time


def BOA(X, fobj, lb, ub, max_iter):
    # Barber Optimization Algorithm (BOA)

    N, dim = X.shape
    F = np.array([fobj(ind) for ind in X])
    best_idx = np.argmin(F)
    best_X = X[best_idx].copy()
    best_score = F[best_idx]
    Convergence = np.zeros(max_iter)

    ct = time.time()
    for t in range(1, max_iter + 1):
        for i in range(N):
            # Phase 1: Exploration (choose better "hairstyles")
            hairstyles = [X[k] for k in range(N) if F[k] < F[i]]
            if hairstyles:
                hairstyle = hairstyles[np.random.randint(len(hairstyles))]
                Ii = np.random.choice([1, 2], size=dim)
                ri = np.random.rand(dim)
                new_pos = X[i] + ri * (hairstyle - Ii * X[i])
                new_pos = np.clip(new_pos, lb[i], ub[i])
                new_f = fobj(new_pos)
                if new_f < F[i]:
                    X[i] = new_pos
                    F[i] = new_f

            # Phase 2: Exploitation (refining hairstyle details)
            ri = np.random.rand(dim)
            new_pos = X[i] + (1 - 2 * ri) * ((ub[i] - lb[i]) / t)
            new_pos = np.clip(new_pos, lb[i], ub[i])
            new_f = fobj(new_pos)
            if new_f < F[i]:
                X[i] = new_pos
                F[i] = new_f

        # Update global best
        best_idx = np.argmin(F)
        if F[best_idx] < best_score:
            best_score = F[best_idx]
            best_X = X[best_idx].copy()
        Convergence[t - 1] = best_score
    ct = time.time() - ct
    return best_score, Convergence, best_X, ct
