import numpy as np
import pytest

from vectordb.brute_force import BruteForceIndex
from vectordb.distance import cosine
from vectordb.hnsw import HNSWIndex
from vectordb.kdtree import KDTreeIndex

METRICS = ["euclidean", "manhattan", "cosine"]


def random_vectors(n, dims, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=dims).astype(np.float32) for _ in range(n)]


def build(index, vectors):
    for i, v in enumerate(vectors):
        index.add(i, v)
    return index


def ids(results):
    return [i for _, i in results]


# ── Brute Force ───────────────────────────────────────────────────────

def test_brute_force_exact_order():
    vecs = random_vectors(50, 8)
    bf = build(BruteForceIndex("cosine"), vecs)
    q = vecs[7]
    res = bf.knn(q, 5)
    assert res[0] == (pytest.approx(0.0, abs=1e-6), 7)
    expected = sorted(range(50), key=lambda i: cosine(q, vecs[i]))[:5]
    assert ids(res) == expected


def test_brute_force_remove_and_duplicates():
    bf = build(BruteForceIndex("euclidean"), random_vectors(5, 4))
    assert bf.remove(2) and not bf.remove(2)
    assert 2 not in ids(bf.knn(np.zeros(4, np.float32), 10))
    with pytest.raises(ValueError):
        bf.add(0, np.zeros(4, np.float32))


# ── KD-Tree: must match brute force exactly, for every metric ─────────

@pytest.mark.parametrize("metric", METRICS)
def test_kdtree_matches_brute_force(metric):
    vecs = random_vectors(300, 6, seed=1)
    bf = build(BruteForceIndex(metric), vecs)
    kd = build(KDTreeIndex(metric), vecs)
    for q in random_vectors(40, 6, seed=2):
        expected, got = bf.knn(q, 7), kd.knn(q, 7)
        assert ids(got) == ids(expected)
        assert [d for d, _ in got] == pytest.approx([d for d, _ in expected], abs=1e-5)


def test_kdtree_cosine_pruning_bug_is_fixed():
    """Naive pruning with cosine distance misses true neighbours. Vectors with
    different magnitudes make the axis gap a bad bound for the angle."""
    rng = np.random.default_rng(3)
    vecs = [(rng.normal(size=4) * rng.uniform(0.1, 50)).astype(np.float32) for _ in range(400)]
    bf = build(BruteForceIndex("cosine"), vecs)
    kd = build(KDTreeIndex("cosine"), vecs)
    for q in random_vectors(50, 4, seed=4):
        assert ids(kd.knn(q, 5)) == ids(bf.knn(q, 5))


def test_kdtree_updates_and_zero_vectors():
    kd = KDTreeIndex("cosine")
    vecs = random_vectors(20, 3)
    build(kd, vecs)
    kd.add(99, np.zeros(3, np.float32))
    kd.remove(0)
    res = kd.knn(vecs[0], 25)
    assert 0 not in ids(res) and 99 in ids(res)
    assert dict((i, d) for d, i in res)[99] == 1.0
    assert kd.knn(np.zeros(3, np.float32), 3) == [(1.0, 1), (1.0, 2), (1.0, 3)]


# ── HNSW: approximate, measured by recall against brute force ─────────

def recall(index, bf, queries, k):
    hits = 0
    for q in queries:
        hits += len(set(ids(index.knn(q, k))) & set(ids(bf.knn(q, k))))
    return hits / (k * len(queries))


@pytest.mark.parametrize("metric", METRICS)
def test_hnsw_recall(metric):
    vecs = random_vectors(1000, 32, seed=5)
    bf = build(BruteForceIndex(metric), vecs)
    hnsw = build(HNSWIndex(metric), vecs)
    queries = random_vectors(50, 32, seed=6)
    assert recall(hnsw, bf, queries, 10) >= 0.95


def test_hnsw_finds_itself_and_has_layers():
    vecs = random_vectors(500, 16, seed=7)
    hnsw = build(HNSWIndex("cosine"), vecs)
    for i in range(0, 500, 25):
        assert hnsw.knn(vecs[i], 1)[0][1] == i
    info = hnsw.graph_info()
    assert info["node_count"] == 500
    assert info["nodes_per_layer"][0] == 500
    # Each layer up should be much sparser than the one below.
    assert all(a > b for a, b in zip(info["nodes_per_layer"], info["nodes_per_layer"][1:]))


def test_hnsw_soft_delete_and_rebuild():
    vecs = random_vectors(200, 8, seed=8)
    hnsw = build(HNSWIndex("euclidean", rebuild_threshold=0.25), vecs)
    bf = build(BruteForceIndex("euclidean"), vecs)
    for i in range(0, 40):
        assert hnsw.remove(i)
        bf.remove(i)
    assert not hnsw.remove(0)
    assert len(hnsw) == 160
    queries = random_vectors(30, 8, seed=9)
    for q in queries:
        assert not set(ids(hnsw.knn(q, 10))) & set(range(40))
    assert recall(hnsw, bf, queries, 10) >= 0.95
    # Past the threshold the graph was rebuilt without the deleted nodes.
    for i in range(40, 80):
        hnsw.remove(i)
    assert len(hnsw._deleted) < 0.25 * len(hnsw._vectors) + 1


def test_empty_indexes():
    q = np.ones(4, np.float32)
    for index in (BruteForceIndex(), KDTreeIndex(), HNSWIndex()):
        assert index.knn(q, 5) == []
