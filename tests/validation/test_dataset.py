from __future__ import annotations

import numpy as np

from validation.dataset import LabeledDataset
from validation.types import LabeledCapture


def _caps():
    return [
        LabeledCapture(
            iq=np.ones(32, dtype=np.complex64),
            sample_rate=1e6,
            truth_regions=[(4, 20, "mavlink")],
            provenance={"source": "synth", "protocol": "mavlink"},
        ),
        LabeledCapture(
            iq=(np.arange(16).astype(np.complex64)),
            sample_rate=1e6,
            truth_regions=[(0, 16, "dji")],
            provenance={"source": "synth", "protocol": "dji"},
        ),
    ]


def test_len_and_iter():
    ds = LabeledDataset(_caps())
    assert len(ds) == 2
    assert [c.provenance["protocol"] for c in ds] == ["mavlink", "dji"]


def test_content_hash_stable_and_sensitive():
    assert (
        LabeledDataset(_caps()).content_hash() == LabeledDataset(_caps()).content_hash()
    )
    mutated = _caps()
    mutated[0].iq[0] += 1
    assert (
        LabeledDataset(mutated).content_hash() != LabeledDataset(_caps()).content_hash()
    )


def test_write_then_from_dir_roundtrip(tmp_path):
    LabeledDataset(_caps()).write(tmp_path)
    loaded = LabeledDataset.from_dir(tmp_path, sample_rate=1e6)
    assert len(loaded) == 2
    protos = sorted(c.provenance["protocol"] for c in loaded)
    assert protos == ["dji", "mavlink"]
