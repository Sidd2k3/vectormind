import numpy as np
import pytest

from vectordb.database import ALGORITHMS, DocumentDB, VectorDB
from vectordb.demo_data import DIMS, demo_vectors, encode_text
from vectordb.distance import METRICS
from vectordb.pca import PCA2D


@pytest.fixture
def demo_db():
    db = VectorDB(DIMS, hnsw_m=16)
    for text, cat, v in demo_vectors():
        db.insert(text, cat, v)
    return db


@pytest.mark.parametrize("query,expected", [
    ("python code", "tech"),
    ("planets and stars", "space"),
    ("guitar song", "music"),
    ("cricket match", "sports"),
])
def test_keyword_queries_hit_right_category(demo_db, query, expected):
    q = encode_text(query)
    for algo in ALGORITHMS:
        for metric in METRICS:
            hits, latency = demo_db.search(q, 5, metric, algo)
            assert len(hits) == 5 and latency >= 0
            assert hits[0][1].category == expected, (algo, metric)


def test_all_algorithms_agree_on_demo_data(demo_db):
    for text, _, v in demo_vectors():
        for metric in METRICS:
            bf, _ = demo_db.search(v, 5, metric, "bruteforce")
            for algo in ("kdtree", "hnsw"):
                got, _ = demo_db.search(v, 5, metric, algo)
                assert [i.id for _, i in got] == [i.id for _, i in bf]


def test_unknown_words_and_bad_input(demo_db):
    assert encode_text("zzz qqq") is None
    with pytest.raises(ValueError):
        demo_db.search(np.ones(3), 5)
    with pytest.raises(ValueError):
        demo_db.search(np.ones(DIMS), 5, algo="annoy")


def test_insert_remove_benchmark(demo_db):
    new_id = demo_db.insert("Rust: memory-safe systems language", "tech", encode_text("code"))
    hits, _ = demo_db.search(encode_text("code"), 1, "cosine", "kdtree")
    assert hits[0][1].id == new_id
    assert demo_db.remove(new_id) and not demo_db.remove(new_id)
    b = demo_db.benchmark(encode_text("music"), 5)
    assert set(b["timings_us"]) == set(ALGORITHMS)
    assert b["recall"]["bruteforce"] == 1.0
    info = demo_db.hnsw_info()
    assert info["node_count"] == 20


def test_pca_separates_categories(demo_db):
    items = demo_db.items()
    pca = PCA2D([i.vector for i in items])
    pts = {}
    for i in items:
        pts.setdefault(i.category, []).append(pca.project(i.vector))
    centers = {c: np.mean(p, axis=0) for c, p in pts.items()}
    # Every item is closer to its own category's centre than to any other.
    for c, plist in pts.items():
        for p in plist:
            own = np.linalg.norm(p - centers[c])
            assert all(own < np.linalg.norm(p - centers[o]) for o in centers if o != c)


def test_document_db_threshold_and_persistence(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.normal(size=64)
    near = base + rng.normal(scale=0.1, size=64)
    far = -base
    db = DocumentDB(tmp_path, "nomic-embed-text")
    ids = db.insert_many([("Doc [1/2]", "about A", near), ("Doc [2/2]", "unrelated", far)])
    hits = db.search(base, 3, max_distance=0.7)
    assert [c.id for _, c in hits] == [ids[0]]  # the far chunk is filtered out

    # Reload from disk
    db2 = DocumentDB(tmp_path, "nomic-embed-text")
    assert len(db2) == 2 and db2.dims == 64
    assert db2.remove(ids[0])
    assert len(DocumentDB(tmp_path, "nomic-embed-text")) == 1
    # A different model gets its own separate store
    assert len(DocumentDB(tmp_path, "other-model")) == 0


def test_document_db_switches_to_hnsw_and_rejects_wrong_dims(tmp_path):
    rng = np.random.default_rng(1)
    db = DocumentDB(None)
    vecs = [rng.normal(size=32) for _ in range(30)]
    db.insert_many([(f"c{i}", "t", v) for i, v in enumerate(vecs)])
    hits = db.search(vecs[5], 1, max_distance=2.0)
    assert hits[0][1].title == "c5"
    with pytest.raises(ValueError):
        db.insert_many([("x", "y", np.ones(8))])
    with pytest.raises(ValueError):
        db.search(np.ones(8), 3)
