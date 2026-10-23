"""Brute Force k-NN: compare the query against every stored vector.

Exact by definition, so it is the ground truth the other indexes are
checked against. Cost is O(N * d) per query.
"""

import heapq

import numpy as np

from .distance import get_distance_fn


class BruteForceIndex:
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self._dist = get_distance_fn(metric)
        self._vectors: dict[int, np.ndarray] = {}

    def add(self, item_id: int, vector: np.ndarray) -> None:
        if item_id in self._vectors:
            raise ValueError(f"id {item_id} already exists")
        self._vectors[item_id] = vector

    def remove(self, item_id: int) -> bool:
        return self._vectors.pop(item_id, None) is not None

    def knn(self, query: np.ndarray, k: int) -> list[tuple[float, int]]:
        """Return the k closest (distance, id) pairs, closest first.
        Ties are broken by id so results are deterministic."""
        if k <= 0:
            return []
        scored = ((self._dist(query, v), i) for i, v in self._vectors.items())
        # nsmallest keeps a heap of size k: O(N log k) instead of a full sort.
        return heapq.nsmallest(k, scored)

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, item_id: int) -> bool:
        return item_id in self._vectors
