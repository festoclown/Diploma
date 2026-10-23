"""Sampling simulator: only this layer knows hidden reserve labels."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


class InapplicableScenario(ValueError):
    pass


def reserve_test_split(groups: np.ndarray, seed: int, test_fraction: float = 0.5
                       ) -> tuple[np.ndarray, np.ndarray]:
    # Splitting whole groups keeps linked documents or duplicate groups from
    # appearing with labels on both sides of the evaluation split.
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two independent groups are needed")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    reserve, test = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return np.sort(reserve), np.sort(test)


def annotation_order(y_reserve: np.ndarray, scenario: str, max_budget: int,
                     rng: np.random.Generator, bias_weight: float = 10.0
                     ) -> tuple[np.ndarray, dict]:
    """A single order supplies nested budgets; returned indices are reserve-relative."""
    n = len(y_reserve)
    if max_budget > n:
        raise InapplicableScenario(f"Budget {max_budget} exceeds reserve size {n}")
    classes = np.unique(y_reserve)
    weights = np.ones(n)
    attempts = 0
    favored = np.array([], dtype=classes.dtype)
    if scenario in {"missing_classes", "biased"}:
        # The simulator uses hidden labels to create a controlled coverage
        # condition. Selectors receive only labels at the sampled indices.
        if len(classes) < 2:
            raise InapplicableScenario("Coverage scenarios need at least two reserve classes")
        while True:
            favored = rng.choice(classes, size=(len(classes) + 1) // 2, replace=False)
            mask = np.isin(y_reserve, favored)
            attempts += 1
            if scenario == "biased" or mask.sum() >= max_budget:
                break
            if attempts >= 1000:
                raise InapplicableScenario("No feasible half-class sample in 1000 attempts")
        weights = np.where(mask, bias_weight if scenario == "biased" else 1.0,
                           1.0 if scenario == "biased" else 0.0)
    elif scenario != "random":
        raise ValueError(f"Unknown annotation scenario: {scenario}")
    eligible = np.flatnonzero(weights > 0)
    # Exponential race implements successive weighted sampling without replacement.
    keys = rng.exponential(size=len(eligible)) / weights[eligible]
    # The same order supplies every budget in a repeat, so L_20 is contained
    # in L_50 and paired method comparisons use identical annotations.
    order = eligible[np.argsort(keys, kind="stable")]
    return order, {"favored_classes": favored.tolist(), "half_class_draw_attempts": attempts}


def bootstrap_weights(groups: np.ndarray, repeats: int, rng: np.random.Generator
                      ) -> np.ndarray:
    """Block bootstrap; with unique groups this is ordinary object bootstrap."""
    _, inverse = np.unique(groups, return_inverse=True)
    count = int(inverse.max()) + 1
    # Resample groups rather than individual object pairs; ARI's pairs are
    # dependent, and linked observations must move together.
    block_counts = rng.multinomial(count, np.full(count, 1 / count), size=repeats)
    return block_counts[:, inverse]
