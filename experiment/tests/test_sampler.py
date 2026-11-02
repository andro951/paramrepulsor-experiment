"""Focused contract tests for the sampler-only experiment patch."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_data(name, folder):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / folder / "src/parampacmap/utils/data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data():
    return load_data("modified_data", "modified")


def line_points(n):
    return np.column_stack((np.arange(n), np.zeros(n))).astype(np.float32)


def assert_contract(n, nn, mn, fp, n_nn=10, n_mn=5, n_fp=20):
    nn = nn[:, 1].reshape(n, n_nn)
    mn = mn[:, 1].reshape(n, n_mn)
    fp = fp[:, 1].reshape(n, n_fp)
    for i in range(n):
        excluded = set(nn[i]) | {i}
        assert not excluded.intersection(mn[i])
        assert not excluded.intersection(fp[i])
        assert len(set(mn[i])) == n_mn
        assert len(set(fp[i])) == n_fp
    return nn, mn, fp


@pytest.mark.parametrize("seed", [None, 0, 21, 42])
@pytest.mark.parametrize("n", [36, 50, 100])
def test_contract_on_actual_datasets(data, seed, n):
    x = np.genfromtxt(
        ROOT / "inputs/datasets" / f"paramrep_2d_{n}.csv",
        delimiter=",", skip_header=1, usecols=(1, 2), dtype=np.float32
    )
    nn, mn, fp, _ = data.generate_pair(x, 10, 5, 20, verbose=False, random_state=seed)
    assert_contract(n, nn, mn, fp)


@pytest.mark.parametrize("option", [0, 1, 2, 3])
def test_second_closest_preserves_metric(data, option):
    rng = np.random.RandomState(12)
    x = rng.rand(8, 5).astype(np.float32)
    if option == 3:
        x = (x > 0.5).astype(np.float32)
    sampled = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    distances = np.array([data.calculate_dist(x[0], x[j], option) for j in sampled])
    first = np.argmin(distances)
    expected = np.delete(sampled, first)[np.argmin(np.delete(distances, first))]
    assert data._second_closest(x, 0, sampled, option) == expected


def test_entire_duplicate_result_draw_retried_without_pool_removal(data, monkeypatch):
    original_choice = np.random.choice
    calls = []
    scripted = [
        np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
        np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
        np.array([1, 3, 4, 5, 6, 7], dtype=np.int32),
    ]

    def choice(pool, size, replace):
        assert size == 6
        assert replace is False
        if len(calls) < 3:
            calls.append(pool.copy())
            return scripted[len(calls) - 1].copy()
        return original_choice(pool, size, replace=replace)

    monkeypatch.setattr(np.random, "choice", choice)
    x = line_points(9)
    nn = np.empty((0, 2), dtype=np.int32)
    mn = data._sample_MN_pair.py_func(x, nn, 0, 2, -1, 0)
    assert list(mn[:2, 1]) == [2, 3]
    assert len(calls) == 3
    assert all(0 not in pool for pool in calls)
    assert all(2 in pool for pool in calls)  # Prior final MN is still eligible.
    assert all(len(pool) == len(set(pool)) for pool in calls)


def test_candidate_draws_are_six_distinct(data, monkeypatch):
    seen = []
    original_second = data._second_closest

    def second(x, i, sampled, option):
        assert len(sampled) == len(set(sampled)) == 6
        assert i not in sampled
        seen.append(sampled.copy())
        return original_second(x, i, sampled, option)

    monkeypatch.setattr(data, "_second_closest", second)
    data._sample_MN_pair.py_func(line_points(12), np.empty((0, 2), np.int32), 0, 2, -1, 0)
    assert len(seen) >= 24


def test_mn_fp_overlap_explicitly_allowed(data):
    x = line_points(12)
    nn = np.empty((0, 2), dtype=np.int32)
    mn = data.sample_MN_pair_deterministic(x, nn, 0, 5, 1, 0)
    fp = data.sample_FP_pair_deterministic(x, nn, 0, 11, 1)
    for i in range(12):
        assert set(mn[i * 5:(i + 1) * 5, 1]).issubset(fp[i * 11:(i + 1) * 11, 1])


def test_seeded_repeat_exact_and_matches_nn_baseline(data):
    baseline = load_data("baseline_data", "upstream")
    x = line_points(36)
    a = data.generate_pair(x, 10, 5, 20, verbose=False, random_state=42)
    b = data.generate_pair(x, 10, 5, 20, verbose=False, random_state=42)
    original = baseline.generate_pair(x, 10, 5, 20, verbose=False, random_state=42)
    for left, right in zip(a[:3], b[:3]):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(a[0], original[0])


@pytest.mark.parametrize("n,mn,fp", [(6, 1, 0), (7, 2, 0), (9, 4, 0), (12, 1, 12)])
def test_infeasible_requests_fail_explicitly(data, n, mn, fp):
    x = line_points(n)
    nn = np.empty((0, 2), dtype=np.int32)
    with pytest.raises(ValueError):
        data.sample_MN_pair_deterministic(x, nn, 0, mn, 1, 0)
        data.sample_FP_pair_deterministic(x, nn, 0, fp, 1)


def test_tied_distances_can_have_larger_second_closest_support(data):
    # Hamming distance to every other point is zero; draw order breaks ties.
    x = np.zeros((7, 2), dtype=np.float32)
    nn = np.empty((0, 2), dtype=np.int32)
    mn = data.sample_MN_pair_deterministic(x, nn, 0, 6, 1, 3)
    for i in range(7):
        assert set(mn[i * 6:(i + 1) * 6, 1]) == set(range(7)) - {i}


def test_tie_aware_support_matches_exhaustive_draw_orders(data):
    distances = np.array([0, 1, 1, 2, 3, 3, 4, 5], dtype=np.float32)
    x = np.column_stack((distances, np.zeros(8))).astype(np.float32)
    candidates = np.arange(1, 8, dtype=np.int32)
    possible = {
        data._second_closest(x, 0, np.array(draw, dtype=np.int32), 0)
        for draw in itertools.permutations(candidates, 6)
    }
    data._check_MN_support(x, 0, candidates, len(possible), 0)
    with pytest.raises(ValueError):
        data._check_MN_support(x, 0, candidates, len(possible) + 1, 0)


def test_retry_limit_raises_without_distribution_fallback(data, monkeypatch):
    monkeypatch.setattr(data, "_second_closest", lambda *args: 1)
    with pytest.raises(RuntimeError, match="retry limit"):
        data._sample_MN_pair.py_func(line_points(9), np.empty((0, 2), np.int32), 0, 2, -1, 0)


def test_zero_and_negative_counts(data):
    x = line_points(7)
    nn = np.empty((0, 2), dtype=np.int32)
    assert data.sample_MN_pair(x, nn, 0, 0).shape == (0, 2)
    assert data.sample_FP_pair(x, nn, 0, 0).shape == (0, 2)
    with pytest.raises(ValueError):
        data.sample_MN_pair(x, nn, 0, -1)
    with pytest.raises(ValueError):
        data.sample_FP_pair(x, nn, 0, -1)


@pytest.mark.parametrize("counts", [(-1, 5, 20), (10, -1, 20), (10, 5, -1), (36, 5, 20), (10, 1.5, 20)])
def test_generate_pair_validates_counts(data, counts):
    with pytest.raises(ValueError):
        data.generate_pair(line_points(36), *counts, verbose=False, random_state=1)
