"""802.11n-style QC-LDPC: reproducible IRA construction + Tanner graph.

Dimensions match IEEE 802.11n n=648 (Z=27, 24 block-columns) but the shift
values are a deterministic, seeded construction (not the exact IEEE tables) —
interop is out of scope; see docs/design/0010-p3e-ldpc.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import numpy.typing as npt

Z = 27
NB = 24
MB: Dict[str, int] = {"1/2": 12, "2/3": 8, "3/4": 6}


def _creates_4cycle(B: npt.NDArray[np.int64], c: int, r: int, s: int) -> bool:
    mb, nb = B.shape
    for rp in range(mb):
        if rp == r or B[rp, c] == -1:
            continue
        sp = int(B[rp, c])
        for c2 in range(nb):
            if c2 == c or B[r, c2] == -1 or B[rp, c2] == -1:
                continue
            if (s - int(B[r, c2])) % Z == (sp - int(B[rp, c2])) % Z:
                return True
    return False


def build_base(rate: str, seed: int) -> npt.NDArray[np.int64]:
    mb = MB[rate]
    kb = NB - mb
    B = np.full((mb, NB), -1, dtype=np.int64)
    # dual-diagonal accumulator parity part (shift-0 identity blocks)
    for i in range(mb):
        B[i, kb + i] = 0
        if i >= 1:
            B[i, kb + i - 1] = 0
    # seeded, 4-cycle-free info part, column weight 3
    rng = np.random.default_rng(seed)
    for c in range(kb):
        placed = 0
        attempts = 0
        while placed < 3:
            attempts += 1
            if attempts > 20000:
                raise RuntimeError(
                    f"LDPC construction stuck (rate {rate}); try another seed"
                )
            r = int(rng.integers(0, mb))
            if B[r, c] != -1:
                continue
            s = int(rng.integers(0, Z))
            if _creates_4cycle(B, c, r, s):
                continue
            B[r, c] = s
            placed += 1
    return B


@dataclass
class LdpcCode:
    B: npt.NDArray[np.int64]
    checks: List[List[int]]
    vars: List[List[int]]
    m: int
    n: int
    k: int
    mb: int
    kb: int


def build_code(rate: str, seed: int) -> LdpcCode:
    B = build_base(rate, seed)
    mb, nb = B.shape
    n = nb * Z
    m = mb * Z
    checks: List[List[int]] = [[] for _ in range(m)]
    vars: List[List[int]] = [[] for _ in range(n)]
    for br in range(mb):
        for bc in range(nb):
            sh = int(B[br, bc])
            if sh == -1:
                continue
            for i in range(Z):
                v = bc * Z + i
                c = br * Z + ((i + sh) % Z)
                checks[c].append(v)
                vars[v].append(c)
    return LdpcCode(B=B, checks=checks, vars=vars, m=m, n=n, k=n - m, mb=mb, kb=nb - mb)
