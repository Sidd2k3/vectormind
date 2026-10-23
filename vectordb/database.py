"""The two databases.

VectorDB     16D demo vectors. Every item is indexed by all three
             algorithms for all three metrics, so any (algorithm, metric)
             pair can be searched and compared on identical data.

DocumentDB   Real embeddings of document chunks (768D for
             nomic-embed-text). Cosine distance only, HNSW for search,
             brute force when the collection is tiny. Persisted to disk.
"""

import json
import os
import re
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .brute_force import BruteForceIndex
from .distance import METRICS, to_vector
from .hnsw import HNSWIndex
from .kdtree import KDTreeIndex

ALGORITHMS = ("bruteforce", "kdtree", "hnsw")


def _make_index(algo: str, metric: str, hnsw_m: int = 16):
    if algo == "bruteforce":
        return BruteForceIndex(metric)
    if algo == "kdtree":
        return KDTreeIndex(metric)
    if algo == "hnsw":
        return HNSWIndex(metric, M=hnsw_m, ef_construction=200, ef_search=50)
    raise ValueError(f"Unknown algorithm '{algo}'. Choose one of: {', '.join(ALGORITHMS)}")


# ══════════════════════════════════════════════════════════════════════
#  Demo database
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Item:
    id: int
    text: str
    category: str
    vector: np.ndarray


class VectorDB:
    def __init__(self, dims: int, hnsw_m: int = 16):
        """`hnsw_m` sets HNSW's M (about 1 in M nodes reaches each layer up)."""
        self.dims = dims
        self._items: dict[int, Item] = {}
        self._next_id = 1
        # One lock guards all indexes so a search never sees a half-finished insert.
        self._lock = threading.RLock()
        self._indexes = {a: {m: _make_index(a, m, hnsw_m) for m in METRICS} for a in ALGORITHMS}

    def _check(self, vector) -> np.ndarray:
        v = to_vector(vector)
        if v.shape[0] != self.dims:
            raise ValueError(f"Expected a {self.dims}D vector, got {v.shape[0]}D")
        return v

    def insert(self, text: str, category: str, vector) -> int:
        v = self._check(vector)
        with self._lock:
            item = Item(self._next_id, text, category, v)
            self._next_id += 1
            self._items[item.id] = item
            for by_metric in self._indexes.values():
                for index in by_metric.values():
                    index.add(item.id, v)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if self._items.pop(item_id, None) is None:
                return False
            for by_metric in self._indexes.values():
                for index in by_metric.values():
                    index.remove(item_id)
            return True

    def search(self, query, k: int, metric: str = "cosine", algo: str = "hnsw"):
        """Returns (hits, latency_us) where hits is a list of (distance, Item)."""
        q = self._check(query)
        index = self._index(algo, metric)
        with self._lock:
            start = time.perf_counter_ns()
            raw = index.knn(q, k)
            elapsed_us = (time.perf_counter_ns() - start) / 1000
            hits = [(d, self._items[i]) for d, i in raw if i in self._items]
        return hits, elapsed_us

    def benchmark(self, query, k: int, metric: str = "cosine", repeats: int = 25) -> dict:
        """Time every algorithm on the same query (median of `repeats` runs)
        and measure recall against brute force."""
        q = self._check(query)
        with self._lock:
            timings, results = {}, {}
            for algo in ALGORITHMS:
                index = self._index(algo, metric)
                index.knn(q, k)  # warm-up (also builds a stale KD-Tree)
                runs = []
                for _ in range(repeats):
                    start = time.perf_counter_ns()
                    res = index.knn(q, k)
                    runs.append((time.perf_counter_ns() - start) / 1000)
                timings[algo] = statistics.median(runs)
                results[algo] = {i for _, i in res}
            truth = results["bruteforce"]
            recall = {a: (len(results[a] & truth) / len(truth) if truth else 1.0) for a in ALGORITHMS}
            return {"timings_us": timings, "recall": recall,
                    "item_count": len(self._items), "repeats": repeats, "k": k, "metric": metric}

    def _index(self, algo: str, metric: str):
        if algo not in self._indexes:
            raise ValueError(f"Unknown algorithm '{algo}'. Choose one of: {', '.join(ALGORITHMS)}")
        if metric not in METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Choose one of: {', '.join(METRICS)}")
        return self._indexes[algo][metric]

    def get(self, item_id: int) -> Item | None:
        return self._items.get(item_id)

    def items(self) -> list[Item]:
        with self._lock:
            return list(self._items.values())

    def hnsw_info(self, metric: str = "cosine") -> dict:
        with self._lock:
            return self._index("hnsw", metric).graph_info()

    def __len__(self) -> int:
        return len(self._items)


# ══════════════════════════════════════════════════════════════════════
#  Document database
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    id: int
    title: str
    text: str
    vector: np.ndarray


class DocumentDB:
    # Below this many chunks, exact brute force is faster than walking a graph.
    BRUTE_FORCE_BELOW = 10

    def __init__(self, storage_dir: Path | None = None, model_name: str = "default"):
        self._chunks: dict[int, Chunk] = {}
        self._next_id = 1
        self.dims: int | None = None
        self._lock = threading.RLock()
        self._hnsw = HNSWIndex("cosine", M=16, ef_construction=200, ef_search=50)
        self._bf = BruteForceIndex("cosine")
        # One file per embedding model: vectors from different models live in
        # different spaces and must never be mixed in one index.
        self._path = None
        if storage_dir is not None:
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)
            self._path = Path(storage_dir) / f"documents_{safe}.json"
            self._load()

    # ── persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        for c in data["chunks"]:
            self._add(Chunk(c["id"], c["title"], c["text"], to_vector(c["vector"])))
        self._next_id = data.get("next_id", max(self._chunks, default=0) + 1)

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "dims": self.dims,
            "chunks": [{"id": c.id, "title": c.title, "text": c.text,
                        "vector": [round(float(x), 6) for x in c.vector]}
                       for c in self._chunks.values()],
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self._path)  # atomic: never leaves a half-written file

    # ── updates ───────────────────────────────────────────────────────

    def _add(self, chunk: Chunk) -> None:
        if self.dims is None:
            self.dims = int(chunk.vector.shape[0])
        elif chunk.vector.shape[0] != self.dims:
            raise ValueError(f"Embedding has {chunk.vector.shape[0]} dims, "
                             f"but this database stores {self.dims}D vectors")
        self._chunks[chunk.id] = chunk
        self._hnsw.add(chunk.id, chunk.vector)
        self._bf.add(chunk.id, chunk.vector)

    def insert_many(self, entries: list[tuple[str, str, np.ndarray]]) -> list[int]:
        """Insert (title, text, vector) entries. All or nothing."""
        vectors = [to_vector(v) for _, _, v in entries]
        with self._lock:
            dims = self.dims or (vectors[0].shape[0] if vectors else None)
            if any(v.shape[0] != dims for v in vectors):
                raise ValueError(f"All embeddings must have {dims} dims")
            ids = []
            for (title, text, _), v in zip(entries, vectors):
                chunk = Chunk(self._next_id, title, text, v)
                self._next_id += 1
                self._add(chunk)
                ids.append(chunk.id)
            self._save()
            return ids

    def remove(self, chunk_id: int) -> bool:
        with self._lock:
            if self._chunks.pop(chunk_id, None) is None:
                return False
            self._hnsw.remove(chunk_id)
            self._bf.remove(chunk_id)
            if not self._chunks:
                self.dims = None
            self._save()
            return True

    # ── search ────────────────────────────────────────────────────────

    def search(self, query, k: int, max_distance: float = 0.7) -> list[tuple[float, Chunk]]:
        """Top-k chunks by cosine distance, dropping any further than
        `max_distance` (nearest is not the same as relevant)."""
        q = to_vector(query)
        with self._lock:
            if not self._chunks:
                return []
            if q.shape[0] != self.dims:
                raise ValueError(f"Query has {q.shape[0]} dims, documents have {self.dims}")
            index = self._bf if len(self._chunks) < self.BRUTE_FORCE_BELOW else self._hnsw
            raw = index.knn(q, k)
            return [(d, self._chunks[i]) for d, i in raw if d <= max_distance and i in self._chunks]

    def all(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks.values())

    def __len__(self) -> int:
        return len(self._chunks)
