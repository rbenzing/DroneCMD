from __future__ import annotations

import numpy as np

from validation.synth.scenarios import DatasetSpec, build_scenario
from validation.types import ModScheme


def _spec(seed=42):
    return DatasetSpec(
        protocols=["mavlink", "dji"],
        snr_grid_db=[-10.0, 0.0, 20.0],
        n_per_cell=2,
        sample_rate=2_048_000.0,
        seed=seed,
        scheme_by_protocol={"mavlink": ModScheme.FSK, "dji": ModScheme.QPSK},
        payload_len=16,
    )


def test_cardinality_matches_grid():
    caps = build_scenario(_spec())
    assert len(caps) == 2 * 3 * 2  # protocols * snr * n_per_cell


def test_truth_regions_and_provenance():
    caps = build_scenario(_spec())
    for c in caps:
        assert c.truth_regions is not None and len(c.truth_regions) == 1
        start, end, proto = c.truth_regions[0]
        assert 0 <= start < end <= len(c.iq)
        assert proto in ("mavlink", "dji")
        assert c.provenance["source"] == "synth"
        assert "snr_db" in c.provenance
        assert c.iq.dtype == np.complex64


def test_deterministic_by_seed():
    a = build_scenario(_spec(1))[0].iq
    b = build_scenario(_spec(1))[0].iq
    assert np.array_equal(a, b)
