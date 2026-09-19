from __future__ import annotations

import numpy as np

from validation import create_synth_dataset
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


def test_differential_defaults_false_in_provenance():
    caps = build_scenario(_spec())
    for c in caps:
        assert c.provenance["differential"] is False


def test_differential_bpsk_dataset_records_provenance():
    # Task 7 regression guard: a differential/BPSK synth dataset must build
    # successfully and record both the BPSK scheme and the differential flag
    # in every capture's provenance, end to end through create_synth_dataset
    # (validation/__init__.py) -> DatasetSpec -> build_scenario -> modulate.
    n_per_cell = 2
    snr_grid = [-10.0, 0.0, 20.0]
    ds = create_synth_dataset(
        protocols=["bpsk_link"],
        snr_grid_db=snr_grid,
        n_per_cell=n_per_cell,
        scheme_by_protocol={"bpsk_link": ModScheme.BPSK},
        differential=True,
    )
    assert len(ds) == len(snr_grid) * n_per_cell
    for c in ds:
        assert c.provenance["scheme"] == "bpsk"
        assert c.provenance["differential"] is True


def test_build_scenario_records_pilot_spacing() -> None:
    from validation.synth.scenarios import DatasetSpec, build_scenario
    from validation.types import ModScheme

    spec = DatasetSpec(
        protocols=["qpsk_link"],
        snr_grid_db=[30.0],
        n_per_cell=1,
        sample_rate=2_048_000.0,
        seed=1,
        scheme_by_protocol={"qpsk_link": ModScheme.QPSK},
        pilot_spacing=8,
    )
    caps = build_scenario(spec)
    assert caps[0].provenance["pilot_spacing"] == 8


def test_build_scenario_records_profile_and_uses_its_params() -> None:
    from validation.synth.scenarios import DatasetSpec, build_scenario

    spec = DatasetSpec(
        protocols=["p_ble2m"],
        snr_grid_db=[30.0],
        n_per_cell=1,
        sample_rate=2_048_000.0,
        seed=1,
        profile_by_protocol={"p_ble2m": "ble_2m"},
    )
    caps = build_scenario(spec)
    assert caps[0].provenance["profile"] == "ble_2m"
    assert caps[0].provenance["scheme"] == "gfsk"


def test_build_scenario_scheme_by_protocol_backcompat() -> None:
    from validation.synth.scenarios import DatasetSpec, build_scenario
    from validation.types import ModScheme

    # Legacy caller path: scheme_by_protocol still works, mapped to a
    # canonical catalog profile.
    spec = DatasetSpec(
        protocols=["p"],
        snr_grid_db=[30.0],
        n_per_cell=1,
        sample_rate=2_048_000.0,
        seed=1,
        scheme_by_protocol={"p": ModScheme.QPSK},
    )
    caps = build_scenario(spec)
    assert caps[0].provenance["profile"] == "qpsk_link"
    assert caps[0].provenance["scheme"] == "qpsk"


def test_build_scenario_ofdm_profile_provenance() -> None:
    from validation import create_synth_dataset

    ds = create_synth_dataset(
        protocols=["wide", "narrow"],
        snr_grid_db=[30.0],
        n_per_cell=1,
        profile_by_protocol={"wide": "wifi_40", "narrow": "ofdm_nb"},
        seed=2,
    )
    by_proto = {c.provenance["protocol"]: c.provenance for c in ds}
    assert by_proto["wide"]["profile"] == "wifi_40"
    assert by_proto["wide"]["scheme"] == "ofdm"
    assert by_proto["narrow"]["profile"] == "ofdm_nb"
    assert by_proto["narrow"]["scheme"] == "ofdm"
    # Prove the profile drives modulation: different profiles yield different spans.
    caps = {c.provenance["protocol"]: c for c in ds}
    wide_truth = caps["wide"].truth_regions[0]
    narrow_truth = caps["narrow"].truth_regions[0]
    wide_span = wide_truth[1] - wide_truth[0]
    narrow_span = narrow_truth[1] - narrow_truth[0]
    assert wide_span != narrow_span


def test_build_scenario_records_coding_provenance() -> None:
    from validation import create_synth_dataset

    ds = create_synth_dataset(
        protocols=["c", "u"],
        snr_grid_db=[30.0],
        n_per_cell=1,
        profile_by_protocol={"c": "rep_bpsk", "u": "sik_gfsk"},
        seed=4,
    )
    by_proto = {cap.provenance["protocol"]: cap.provenance for cap in ds}
    assert by_proto["c"]["coding"] == "rep3"
    assert by_proto["u"]["coding"] == "uncoded"
