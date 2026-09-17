from __future__ import annotations

import numpy as np

from validation.repro import capture_manifest, hash_array, hash_config, rng


def test_rng_is_deterministic():
    a = rng(42).standard_normal(100)
    b = rng(42).standard_normal(100)
    assert np.array_equal(a, b)


def test_rng_differs_by_seed():
    assert not np.array_equal(rng(1).standard_normal(50), rng(2).standard_normal(50))


def test_hash_array_stable_and_sensitive():
    x = np.arange(10, dtype=np.complex64)
    assert hash_array(x) == hash_array(x.copy())
    y = x.copy()
    y[0] += 1
    assert hash_array(x) != hash_array(y)


def test_hash_config_order_independent():
    assert hash_config({"a": 1, "b": 2}) == hash_config({"b": 2, "a": 1})


def test_capture_manifest_has_versions():
    m = capture_manifest(seed=42, dataset_hash="d", config_hash="c")
    assert m.seed == 42 and m.dataset_hash == "d"
    assert "numpy" in m.versions
    assert isinstance(m.timestamp, str) and m.timestamp
