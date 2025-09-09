from __future__ import annotations
from typing import Callable, Optional, Tuple, List
import numpy as np
import random

Array = np.ndarray

class GeneticAlgorithm:
    def __init__(
        self,
        lb: Array,
        ub: Array,
        pop_size: int = 32,
        n_gen: int = 40,
        seed: int = 123,
        tournament_k: int = 3,
        elite_frac: float = 0.10,
        sbx_eta: float = 12.0,
        sbx_prob: float = 0.9,
        mut_rate: float = 0.25,
        mut_scale: float = 0.12,
        objective: str = "max",   
    ) -> None:
        assert lb.shape == ub.shape, "lb/ub shape mismatch"
        self.lb = lb.astype(float)
        self.ub = ub.astype(float)
        self.dim = lb.size

        self.pop_size = int(pop_size)
        self.n_gen = int(n_gen)
        self.k = int(tournament_k)
        self.elite_frac = float(elite_frac)

        self.sbx_eta = float(sbx_eta)
        self.sbx_prob = float(sbx_prob)
        self.mut_rate = float(mut_rate)
        self.mut_scale = float(mut_scale)

        assert objective in ("max", "min")
        self.objective = objective

        self.rng = np.random.default_rng(int(seed))
        random.seed(int(seed))

    def _clip(self, x: Array) -> Array:
        return np.minimum(self.ub, np.maximum(self.lb, x))

    def _init_population(self) -> Array:
        return self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))

    def _is_better(self, s_new: float, s_old: float) -> bool:
        if self.objective == "max":
            return s_new > s_old
        else:
            return s_new < s_old

    def _tournament_select(self, pop: Array, scores: Array) -> Array:
        idxs = self.rng.integers(0, self.pop_size, size=self.k)
        if self.objective == "max":
            best = idxs[np.argmax(scores[idxs])]
        else:
            best = idxs[np.argmin(scores[idxs])]
        return pop[best].copy()

    def _sbx(self, p1: Array, p2: Array) -> Tuple[Array, Array]:
        """
        SBX交叉
        """
        if random.random() > self.sbx_prob:
            return p1.copy(), p2.copy()
        c1, c2 = p1.copy(), p2.copy()
        for j in range(self.dim):
            u = random.random()
            if u <= 0.5:
                beta = (2*u)**(1.0/(self.sbx_eta+1.0))
            else:
                beta = (1/(2*(1-u)))**(1.0/(self.sbx_eta+1.0))
            c1[j] = 0.5*((1+beta)*p1[j] + (1-beta)*p2[j])
            c2[j] = 0.5*((1-beta)*p1[j] + (1+beta)*p2[j])
        return self._clip(c1), self._clip(c2)

    def _mutate(self, x: Array, sigma: Array) -> Array:
        """
        高斯变异
        """
        mask = self.rng.random(self.dim) < self.mut_rate
        noise = self.rng.normal(0.0, 1.0, size=self.dim) * sigma
        x2 = np.where(mask, x + noise, x)
        return self._clip(x2)

    def evolve(
        self,
        fitness_fn: Callable[[Array], float],
        reeval_fn: Optional[Callable[[Array], float]] = None,
        reeval_frac: float = 0.20,
        verbose_every: int = 5,
        log_fn: Optional[Callable[[int, float, Array], None]] = None,
        repair_fn: Optional[Callable[[Array], Array]] = None,
        seed_population: Optional[List[Array]] = None,
        sigma_fn: Optional[Callable[[int, int, Array, Array], Array]] = None,
        stats_hook: Optional[Callable[[int, float, float], None]] = None,
    ) -> Tuple[Array, float]:

        # init pop
        pop = self._init_population()
        if seed_population:
            k = min(len(seed_population), self.pop_size)
            for i in range(k):
                pop[i, :] = self._clip(seed_population[i])

        # initial repair (optional)
        if repair_fn is not None:
            for i in range(self.pop_size):
                pop[i, :] = self._clip(repair_fn(pop[i, :]))

        # evaluate
        scores = np.array([fitness_fn(ind) for ind in pop], dtype=float)

        # best so far
        best_idx = int(np.argmax(scores) if self.objective == "max" else np.argmin(scores))
        best_x = pop[best_idx].copy()
        best_s = float(scores[best_idx])

        elite_k = max(2, int(round(self.pop_size * self.elite_frac)))
        reeval_k = max(2, int(round(self.pop_size * reeval_frac))) if reeval_fn else 0

        for gen in range(1, self.n_gen + 1):
            # per-gen sigma (anneal by default)
            if sigma_fn is None:
                anneal = max(0.3, 1.0 - gen / self.n_gen)
                sigma = self.mut_scale * (self.ub - self.lb) * anneal
            else:
                sigma = sigma_fn(gen, self.n_gen, self.lb, self.ub)

            # optional: refine top-p% with reeval_fn (only improve)
            if reeval_k > 0:
                if self.objective == "max":
                    order_tmp = np.argsort(scores)[::-1]
                else:
                    order_tmp = np.argsort(scores)
                top_idx = order_tmp[:reeval_k]
                for i in top_idx:
                    s2 = reeval_fn(pop[i, :])
                    if self._is_better(s2, scores[i]):
                        scores[i] = s2

            # update best
            curr_best_idx = int(np.argmax(scores) if self.objective == "max" else np.argmin(scores))
            if self._is_better(scores[curr_best_idx], best_s):
                best_x = pop[curr_best_idx].copy()
                best_s = float(scores[curr_best_idx])

            mean_s = float(np.mean(scores))
            if stats_hook is not None:
                stats_hook(gen, best_s, mean_s)

            if verbose_every > 0 and (gen % verbose_every == 0):
                if log_fn:
                    log_fn(gen, best_s, best_x)

            # selection + create next population (elitism + SBX + mutation)
            if self.objective == "max":
                order = np.argsort(scores)[::-1]
            else:
                order = np.argsort(scores)
            new_pop = [pop[i, :].copy() for i in order[:elite_k]]

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, scores)
                p2 = self._tournament_select(pop, scores)
                c1, c2 = self._sbx(p1, p2)
                c1 = self._mutate(c1, sigma)
                c2 = self._mutate(c2, sigma)
                if repair_fn is not None:
                    c1 = self._clip(repair_fn(c1))
                    c2 = self._clip(repair_fn(c2))
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop = np.vstack(new_pop)
            scores = np.array([fitness_fn(ind) for ind in pop], dtype=float)

        # final refine (optional)
        if reeval_k > 0 and reeval_fn is not None:
            if self.objective == "max":
                order_tmp = np.argsort(scores)[::-1]
            else:
                order_tmp = np.argsort(scores)
            top_idx = order_tmp[:reeval_k]
            for i in top_idx:
                s2 = reeval_fn(pop[i, :])
                if self._is_better(s2, scores[i]):
                    scores[i] = s2

        # final best
        final_idx = int(np.argmax(scores) if self.objective == "max" else np.argmin(scores))
        if self._is_better(scores[final_idx], best_s):
            best_x = pop[final_idx].copy()
            best_s = float(scores[final_idx])

        return best_x, best_s
