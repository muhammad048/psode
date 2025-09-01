import numpy as np
import random
from typing import Callable, Tuple, Union, Sequence

class PSOOptimizer:
    """Simple PSO for 2-D hyperparam search (F, CR).
    Maximizes the scalar fitness returned by eval_func(particle).
    If eval_func returns a tuple (acc, prec), we combine as fitness = w_acc*acc + w_prec*prec.
    """
    def __init__(self,
                 pop_size: int,
                 F_bounds: Tuple[float, float],
                 CR_bounds: Tuple[float, float],
                 max_iters: int,
                 w_min: float = 0.4,
                 w_max: float = 0.9,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 w_acc: float = 1.0,
                 w_prec: float = 0.0,
                 seed: int = 0):
        self.pop_size = int(pop_size)
        self.F_bounds = (float(min(F_bounds)), float(max(F_bounds)))
        self.CR_bounds = (float(min(CR_bounds)), float(max(CR_bounds)))
        self.max_iters = int(max_iters)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.w_acc = float(w_acc)
        self.w_prec = float(w_prec)
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        # init
        self.population = [self._rand_particle() for _ in range(self.pop_size)]
        self.velocities = [np.zeros(2, dtype=np.float32) for _ in range(self.pop_size)]
        self.pbest = [p.copy() for p in self.population]
        self.pbest_fit = [-np.inf for _ in range(self.pop_size)]
        self.gbest = self.population[0].copy()
        self.gbest_fit = -np.inf

    def _rand_particle(self):
        F = np.random.uniform(self.F_bounds[0], self.F_bounds[1])
        CR = np.random.uniform(self.CR_bounds[0], self.CR_bounds[1])
        return np.array([F, CR], dtype=np.float32)

    def _combine(self, val: Union[float, Sequence[float]]) -> float:
        print("precision"+val[1])
        if isinstance(val, (tuple, list)) and len(val) >= 2:
            return self.w_acc * float(val[0]) + self.w_prec * float(val[1])
        return float(val)

    def optimize(self, eval_func: Callable[[np.ndarray], Union[float, Tuple[float,float]]]):
        for t in range(self.max_iters):
            # cosine inertia
            w = self.w_min + (self.w_max - self.w_min) * (1 + np.cos(np.pi * t / max(1, self.max_iters))) / 2.0
            fits = []
            for i, particle in enumerate(self.population):
                val = eval_func(particle)
                fit = self._combine(val)
                fits.append(fit)
                # update pbest
                if fit > self.pbest_fit[i]:
                    self.pbest_fit[i] = fit
                    self.pbest[i] = particle.copy()
            # update gbest
            best_idx = int(np.argmax(fits))
            if fits[best_idx] > self.gbest_fit:
                self.gbest_fit = fits[best_idx]
                self.gbest = self.population[best_idx].copy()

            # velocity & position update
            for i in range(self.pop_size):
                r1, r2 = random.random(), random.random()
                self.velocities[i] = (w * self.velocities[i]
                                      + self.c1 * r1 * (self.pbest[i] - self.population[i])
                                      + self.c2 * r2 * (self.gbest - self.population[i]))
                self.population[i] = self.population[i] + self.velocities[i]
                self.population[i][0] = np.clip(self.population[i][0], *self.F_bounds)
                self.population[i][1] = np.clip(self.population[i][1], *self.CR_bounds)
        return self.gbest, self.gbest_fit
