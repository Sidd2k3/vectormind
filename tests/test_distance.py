import numpy as np
import pytest

from vectordb.distance import (
    cosine,
    euclidean,
    get_distance_fn,
    manhattan,
    normalize,
    to_vector,
)

# The same three vectors used in the explanation.
A = to_vector([3, 4])
B = to_vector([6, 8])  # same direction as A, twice as long
C = to_vector([4, 3])  # close to A, slightly different direction


def test_euclidean():
    assert euclidean(A, B) == pytest.approx(5.0)
    assert euclidean(A, C) == pytest.approx(np.sqrt(2))


def test_manhattan():
    assert manhattan(A, B) == pytest.approx(7.0)
    assert manhattan(A, C) == pytest.approx(2.0)


def test_cosine():
    assert cosine(A, B) == pytest.approx(0.0, abs=1e-6)
    assert cosine(A, C) == pytest.approx(0.04, abs=1e-6)


def test_cosine_range_and_zero_vector():
    assert cosine(A, -A) == pytest.approx(2.0)
    assert cosine(A, to_vector([-4, 3])) == pytest.approx(1.0)  # 90 degrees
    assert cosine(A, to_vector([0, 0])) == 1.0


def test_metrics_disagree():
    # Euclidean says C is closer to A; cosine says B is.
    assert euclidean(A, C) < euclidean(A, B)
    assert cosine(A, B) < cosine(A, C)


def test_normalized_euclidean_matches_cosine_ranking():
    rng = np.random.default_rng(0)
    query = normalize(rng.normal(size=32).astype(np.float32))
    points = [normalize(rng.normal(size=32).astype(np.float32)) for _ in range(100)]

    by_cosine = sorted(range(100), key=lambda i: cosine(query, points[i]))
    by_euclid = sorted(range(100), key=lambda i: euclidean(query, points[i]))
    assert by_cosine == by_euclid

    # And the exact identity: euclidean^2 = 2 * cosine_distance
    for p in points:
        assert euclidean(query, p) ** 2 == pytest.approx(2 * cosine(query, p), abs=1e-5)


def test_get_distance_fn():
    assert get_distance_fn("cosine") is cosine
    with pytest.raises(ValueError):
        get_distance_fn("cosin")


def test_to_vector_validation():
    with pytest.raises(ValueError):
        to_vector([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        to_vector([1.0, float("nan")])
