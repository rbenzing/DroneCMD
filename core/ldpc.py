"""802.11n-style QC-LDPC: reproducible IRA construction + Tanner graph.

Dimensions match IEEE 802.11n n=648 (Z=27, 24 block-columns) but the shift
values are a deterministic, seeded construction (not the exact IEEE tables) —
interop is out of scope; see docs/design/0010-p3e-ldpc.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, cast

import numpy as np
import numpy.typing as npt

Z = 27
NB = 24
MB: Dict[str, int] = {"1/2": 12, "2/3": 8, "3/4": 6}

Bits = npt.NDArray[np.uint8]


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


def encode(code: LdpcCode, info_bits: Bits) -> Bits:
    info = np.asarray(info_bits, dtype=np.uint8)
    if info.size != code.k:
        raise ValueError("info length must equal code.k")
    # info-only syndrome per check
    synd = np.zeros(code.m, dtype=np.uint8)
    for c in range(code.m):
        acc = 0
        for v in code.checks[c]:
            if v < code.k:
                acc ^= int(info[v])
        synd[c] = acc
    # accumulator solve over parity blocks: p_i = synd_i XOR p_{i-1}
    parity = np.zeros(code.n - code.k, dtype=np.uint8)
    prev = np.zeros(Z, dtype=np.uint8)
    for i in range(code.mb):
        sblk = synd[i * Z : (i + 1) * Z]
        pblk = (sblk ^ prev) if i >= 1 else sblk
        parity[i * Z : (i + 1) * Z] = pblk
        prev = pblk
    return cast(Bits, np.concatenate([info, parity]).astype(np.uint8))


def decode_min_sum(
    code: LdpcCode,
    llr: "npt.NDArray[np.float64]",
    max_iters: int = 50,
    norm: float = 0.8,
) -> Bits:
    ch = np.asarray(llr, dtype=np.float64)
    # check->var messages, indexed [c][v]
    mcv: List[Dict[int, float]] = [
        {v: 0.0 for v in code.checks[c]} for c in range(code.m)
    ]
    hard = (ch < 0).astype(np.uint8)
    for _ in range(max_iters):
        # variable totals = channel + sum incoming check msgs
        total = ch.copy()
        for c in range(code.m):
            for v in code.checks[c]:
                total[v] += mcv[c][v]
        # check-node update (normalized min-sum)
        for c in range(code.m):
            vs = code.checks[c]
            inc = [total[v] - mcv[c][v] for v in vs]
            sign_prod = 1.0
            m1 = m2 = float("inf")
            arg = -1
            for idx, x in enumerate(inc):
                if x < 0:
                    sign_prod = -sign_prod
                a = abs(x)
                if a < m1:
                    m2 = m1
                    m1 = a
                    arg = idx
                elif a < m2:
                    m2 = a
            for idx, v in enumerate(vs):
                s_other = sign_prod * (-1.0 if inc[idx] < 0 else 1.0)
                mag = m2 if idx == arg else m1
                mcv[c][v] = norm * s_other * mag
        # hard decision + parity check
        total = ch.copy()
        for c in range(code.m):
            for v in code.checks[c]:
                total[v] += mcv[c][v]
        hard = (total < 0).astype(np.uint8)
        satisfied = True
        for c in range(code.m):
            acc = 0
            for v in code.checks[c]:
                acc ^= int(hard[v])
            if acc:
                satisfied = False
                break
        if satisfied:
            break
    return hard
