"""KD-Tree k-NN: exact search that prunes regions of space.

Each node splits space along one axis (cycling through the dimensions).
During search, a whole subtree is skipped when the distance from the
query to the splitting plane is already worse than the current k-th
best result.

That pruning bound, |query[axis] - node[axis]|, is a valid lower bound
for euclidean and manhattan distance, but NOT for cosine distance.
For cosine, this index stores unit-normalized copies of the vectors and
searches them with euclidean distance. For unit vectors
    euclidean^2 = 2 * cosine_distance
so the ranking is identical and the pruning is correct. The euclidean
result is converted back to a cosine distance before it is returned.
"""

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .distance import euclidean, get_distance_fn, manhattan, normalize

_EPS = 1e-9


@dataclass
class _Node:
    item_id: int
    point: np.ndarray
    axis: int
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class KDTreeIndex:
    def __init__(self, metric: str = "euclidean"):
        get_distance_fn(metric)  # validates the name
        self.metric = metric
        # Distance used inside the tree (see module docstring).
        self._tree_dist = manhattan if metric == "manhattan" else euclidean
        self._vectors: dict[int, np.ndarray] = {}
        self._root: Optional[_Node] = None
        # Zero vectors have no direction, so under cosine they sit outside
        # the tree at a fixed distance of 1.0 from everything.
        self._zero_ids: list[int] = []
        self._dirty = False

    # ── updates ───────────────────────────────────────────────────────
    # Inserts and deletes only mark the tree stale. It is rebuilt (balanced,
    # median splits) on the next query, so it never degrades into a skewed
    # tree the way one-by-one insertion can.

    def add(self, item_id: int, vector: np.ndarray) -> None:
        if item_id in self._vectors:
            raise ValueError(f"id {item_id} already exists")
        self._vectors[item_id] = vector
        self._dirty = True

    def remove(self, item_id: int) -> bool:
        if self._vectors.pop(item_id, None) is None:
            return False
        self._dirty = True
        return True

    def _rebuild(self) -> None:
        points: list[tuple[int, np.ndarray]] = []
        self._zero_ids = []
        for item_id, v in self._vectors.items():
            if self.metric == "cosine":
                if np.linalg.norm(v) < _EPS:
                    self._zero_ids.append(item_id)
                    continue
                v = normalize(v)
            points.append((item_id, v))
        self._root = self._build(points, depth=0)
        self._dirty = False

    def _build(self, points: list[tuple[int, np.ndarray]], depth: int) -> Optional[_Node]:
        if not points:
            return None
        axis = depth % len(points[0][1])
        points.sort(key=lambda p: p[1][axis])
        mid = len(points) // 2
        item_id, point = points[mid]
        return _Node(
            item_id=item_id,
            point=point,
            axis=axis,
            left=self._build(points[:mid], depth + 1),
            right=self._build(points[mid + 1:], depth + 1),
        )

    # ── search ────────────────────────────────────────────────────────

    def knn(self, query: np.ndarray, k: int) -> list[tuple[float, int]]:
        if k <= 0 or not self._vectors:
            return []
        if self._dirty:
            self._rebuild()

        if self.metric == "cosine":
            if np.linalg.norm(query) < _EPS:
                # A zero query has no direction: everything is equally unrelated.
                return [(1.0, i) for i in sorted(self._vectors)[:k]]
            query = normalize(query)

        # Max-heap of the best k so far, stored as (-distance, -id) so the
        # root is the current worst result.
        heap: list[tuple[float, int]] = []
        self._search(self._root, query, k, heap)
        results = sorted((-nd, -ni) for nd, ni in heap)

        if self.metric == "cosine":
            results = [(min(2.0, d * d / 2.0), i) for d, i in results]
            results.extend((1.0, i) for i in self._zero_ids)
            results.sort()
            results = results[:k]
        return results

    def _search(self, node: Optional[_Node], q: np.ndarray, k: int,
                heap: list[tuple[float, int]]) -> None:
        if node is None:
            return
        d = self._tree_dist(q, node.point)
        entry = (-d, -node.item_id)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

        diff = float(q[node.axis] - node.point[node.axis])
        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._search(near, q, k, heap)

        # Prune: every point on the far side is at least |diff| away.
        worst = -heap[0][0] if heap else math.inf
        if len(heap) < k or abs(diff) <= worst:
            self._search(far, q, k, heap)

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, item_id: int) -> bool:
        return item_id in self._vectors
