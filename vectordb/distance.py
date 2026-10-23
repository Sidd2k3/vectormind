"""Distance metrics for comparing vectors.

Every metric returns a *distance*: smaller means more similar.
That keeps one rule ("lower is better") true for all three, so the
search algorithms never need to care which metric they were given.
"""

from typing import Callable

import numpy as np

# A distance function takes two vectors and returns one float.
DistanceFn = Callable[[np.ndarray, np.ndarray], float]

# Below this length a vector is treated as all-zeros (no direction).
_EPS = 1e-9


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Straight-line (L2) distance: sqrt(sum((a - b)^2))."""
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    """City-block (L1) distance: sum(|a - b|)."""
    return float(np.sum(np.abs(a - b)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance: 1 - cos(angle between a and b).

    Range is 0 (same direction) to 2 (opposite directions).
    A zero vector has no direction, so it is treated as unrelated (1.0)
    instead of dividing by zero.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < _EPS or norm_b < _EPS:
        return 1.0
    similarity = np.dot(a, b) / (norm_a * norm_b)
    # Floating-point rounding can push similarity slightly outside [-1, 1].
    similarity = float(np.clip(similarity, -1.0, 1.0))
    return 1.0 - similarity


METRICS: dict[str, DistanceFn] = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "cosine": cosine,
}


def get_distance_fn(name: str) -> DistanceFn:
    """Look up a metric by name. Unknown names raise instead of silently
    falling back, so a typo like "cosin" is caught immediately."""
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(
            f"Unknown metric '{name}'. Choose one of: {', '.join(METRICS)}"
        ) from None


def normalize(v: np.ndarray) -> np.ndarray:
    """Scale a vector to length 1. Zero vectors are returned unchanged.

    For unit vectors, euclidean^2 = 2 - 2 * cosine_similarity, so
    euclidean and cosine produce the same nearest-neighbour ranking.
    """
    norm = np.linalg.norm(v)
    if norm < _EPS:
        return v.copy()
    return v / norm


def to_vector(values) -> np.ndarray:
    """Convert any list of numbers into the float32 array format used
    everywhere in the database."""
    v = np.asarray(values, dtype=np.float32)
    if v.ndim != 1:
        raise ValueError("A vector must be a flat list of numbers.")
    if not np.all(np.isfinite(v)):
        raise ValueError("A vector cannot contain NaN or infinity.")
    return v
