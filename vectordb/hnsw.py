"""HNSW: Hierarchical Navigable Small World graph (approximate k-NN).

Layer 0 holds every vector. Each layer above holds roughly 1/M of the
layer below, with longer-range links. A search greedily descends from
the top-layer entry point, then runs a beam search of width `ef` on
layer 0.

Parameters
    M                max links per node on layers >= 1
    M0 = 2 * M       max links per node on layer 0 (denser, fine search)
    ef_construction  candidate list size while inserting (graph quality)
    ef_search        candidate list size while querying (speed vs recall)

Neighbour selection uses the heuristic from the HNSW paper: a candidate
is kept only if it is closer to the new node than to any neighbour
already chosen. This spreads links in different directions instead of
packing them into one tight cluster, which keeps the graph navigable.

Deletion is a soft delete: the node keeps routing searches but is never
returned. When deleted nodes exceed `rebuild_threshold` of the graph,
the graph is rebuilt from the live nodes.
"""

import heapq
import math
import random

import numpy as np

from .distance import get_distance_fn


class HNSWIndex:
    def __init__(
        self,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
        rebuild_threshold: float = 0.25,
    ):
        if M < 2:
            raise ValueError("M must be at least 2")
        self.metric = metric
        self._dist = get_distance_fn(metric)
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.rebuild_threshold = rebuild_threshold
        self._mL = 1.0 / math.log(M)
        self._rng = random.Random(seed)
        self._reset()

    def _reset(self) -> None:
        self._vectors: dict[int, np.ndarray] = {}
        self._levels: dict[int, int] = {}
        self._links: dict[int, list[list[int]]] = {}  # id -> one list per layer
        self._deleted: set[int] = set()
        self._entry: int | None = None
        self._max_level = -1

    # ── helpers ───────────────────────────────────────────────────────

    def _random_level(self) -> int:
        # P(level >= l) = (1/M)^l  →  exponentially fewer nodes per layer.
        return int(-math.log(1.0 - self._rng.random()) * self._mL)

    def _search_layer(self, q: np.ndarray, entry_points: list[tuple[float, int]],
                      ef: int, layer: int) -> list[tuple[float, int]]:
        """Beam search on one layer. Returns up to `ef` (distance, id) pairs,
        closest first."""
        visited = {i for _, i in entry_points}
        candidates = list(entry_points)           # min-heap: closest first
        heapq.heapify(candidates)
        results = [(-d, i) for d, i in entry_points]  # max-heap: furthest first
        heapq.heapify(results)

        while candidates:
            d, current = heapq.heappop(candidates)
            if d > -results[0][0]:
                break  # nothing left to explore can improve the results
            for nb in self._links[current][layer]:
                if nb in visited:
                    continue
                visited.add(nb)
                dn = self._dist(q, self._vectors[nb])
                if len(results) < ef or dn < -results[0][0]:
                    heapq.heappush(candidates, (dn, nb))
                    heapq.heappush(results, (-dn, nb))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted((-nd, i) for nd, i in results)

    def _select_neighbors(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Heuristic selection. `candidates` are (distance to base, id),
        sorted closest first."""
        selected: list[tuple[float, int]] = []
        skipped: list[tuple[float, int]] = []
        for d, c in candidates:
            if len(selected) >= m:
                break
            cv = self._vectors[c]
            if all(d < self._dist(cv, self._vectors[s]) for _, s in selected):
                selected.append((d, c))
            else:
                skipped.append((d, c))
        # Fill any free slots with the closest skipped candidates.
        for item in skipped:
            if len(selected) >= m:
                break
            selected.append(item)
        return [c for _, c in selected]

    # ── updates ───────────────────────────────────────────────────────

    def add(self, item_id: int, vector: np.ndarray) -> None:
        if item_id in self._vectors:
            raise ValueError(f"id {item_id} already exists")
        level = self._random_level()
        self._vectors[item_id] = vector
        self._levels[item_id] = level
        self._links[item_id] = [[] for _ in range(level + 1)]

        if self._entry is None:
            self._entry, self._max_level = item_id, level
            return

        ep = [(self._dist(vector, self._vectors[self._entry]), self._entry)]
        # Greedy descent through the layers above the new node's level.
        for layer in range(self._max_level, level, -1):
            ep = self._search_layer(vector, ep, 1, layer)[:1]

        # Connect the node on every layer from its level down to 0.
        for layer in range(min(level, self._max_level), -1, -1):
            found = self._search_layer(vector, ep, self.ef_construction, layer)
            neighbors = self._select_neighbors(found, self.M)
            self._links[item_id][layer] = neighbors

            max_links = self.M0 if layer == 0 else self.M
            for nb in neighbors:
                nb_links = self._links[nb][layer]
                nb_links.append(item_id)
                if len(nb_links) > max_links:  # too many links: shrink
                    nv = self._vectors[nb]
                    ranked = sorted((self._dist(nv, self._vectors[x]), x) for x in nb_links)
                    self._links[nb][layer] = self._select_neighbors(ranked, max_links)
            ep = found

        if level > self._max_level:
            self._entry, self._max_level = item_id, level

    def remove(self, item_id: int) -> bool:
        if item_id not in self._vectors or item_id in self._deleted:
            return False
        self._deleted.add(item_id)
        if len(self._deleted) > self.rebuild_threshold * len(self._vectors):
            self.rebuild()
        return True

    def rebuild(self) -> None:
        """Rebuild the graph from live nodes only, dropping soft-deleted ones."""
        live = [(i, v) for i, v in self._vectors.items() if i not in self._deleted]
        self._reset()
        for i, v in live:
            self.add(i, v)

    # ── search ────────────────────────────────────────────────────────

    def knn(self, query: np.ndarray, k: int, ef: int | None = None) -> list[tuple[float, int]]:
        if k <= 0 or self._entry is None or len(self) == 0:
            return []
        width = max(ef or self.ef_search, k)
        # Deleted nodes still occupy slots in the beam, so widen it a little.
        width += min(len(self._deleted), width)

        ep = [(self._dist(query, self._vectors[self._entry]), self._entry)]
        for layer in range(self._max_level, 0, -1):
            ep = self._search_layer(query, ep, 1, layer)[:1]
        found = self._search_layer(query, ep, width, 0)
        return [(d, i) for d, i in found if i not in self._deleted][:k]

    # ── introspection ─────────────────────────────────────────────────

    def graph_info(self) -> dict:
        layers = self._max_level + 1 if self._entry is not None else 0
        nodes_per_layer = [0] * layers
        edges: set[tuple[int, int, int]] = set()
        nodes = []
        for i, level in self._levels.items():
            if i in self._deleted:
                continue
            nodes.append({"id": i, "level": level})
            for layer in range(level + 1):
                nodes_per_layer[layer] += 1
                for nb in self._links[i][layer]:
                    if nb not in self._deleted:
                        edges.add((min(i, nb), max(i, nb), layer))
        edges_per_layer = [0] * layers
        for _, _, layer in edges:
            edges_per_layer[layer] += 1
        return {
            "top_layer": self._max_level,
            "entry_point": self._entry,
            "node_count": len(self),
            "nodes_per_layer": nodes_per_layer,
            "edges_per_layer": edges_per_layer,
            "nodes": nodes,
            "edges": [{"src": a, "dst": b, "layer": l} for a, b, l in sorted(edges)],
            "params": {"M": self.M, "M0": self.M0,
                       "ef_construction": self.ef_construction, "ef_search": self.ef_search},
        }

    def __len__(self) -> int:
        return len(self._vectors) - len(self._deleted)

    def __contains__(self, item_id: int) -> bool:
        return item_id in self._vectors and item_id not in self._deleted
