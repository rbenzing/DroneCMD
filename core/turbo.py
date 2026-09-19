"""LTE-style turbo code: RSC constituent, QPP interleaver, max-log-MAP decode.

Feedback g0 = 13 octal = 1 + D + D^3 ; feedforward g1 = 15 octal = 1 + D^2 + D^3.
State bits: m1=bit0 (most recent), m2=bit1, m3=bit2. Self-consistent (not
LTE-bit-exact); interop out of scope — see docs/design/0011-p3f-turbo.md.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.typing as npt

Bits = npt.NDArray[np.uint8]
LLRs = npt.NDArray[np.float64]

NSTATES = 8


def _rsc_step(state: int, u: int) -> Tuple[int, int]:
    m1 = state & 1
    m2 = (state >> 1) & 1
    m3 = (state >> 2) & 1
    a = u ^ m1 ^ m3  # feedback g0 = 1 + D + D^3
    z = a ^ m2 ^ m3  # output   g1 = 1 + D^2 + D^3
    nxt = a | (m1 << 1) | (m2 << 2)
    return nxt, z


NXT: List[List[int]] = [[0, 0] for _ in range(NSTATES)]
PAR: List[List[int]] = [[0, 0] for _ in range(NSTATES)]
for _s in range(NSTATES):
    for _u in (0, 1):
        _n, _z = _rsc_step(_s, _u)
        NXT[_s][_u] = _n
        PAR[_s][_u] = _z


def _tail_input(state: int) -> int:
    return (state & 1) ^ ((state >> 2) & 1)  # drives a -> 0


def rsc_encode(info: Bits) -> Tuple[Bits, Bits]:
    state = 0
    sys: List[int] = []
    par: List[int] = []
    for ui in np.asarray(info, dtype=np.uint8):
        u = int(ui)
        sys.append(u)
        par.append(PAR[state][u])
        state = NXT[state][u]
    for _ in range(3):  # tail termination -> state 0
        u = _tail_input(state)
        sys.append(u)
        par.append(PAR[state][u])
        state = NXT[state][u]
    return np.array(sys, dtype=np.uint8), np.array(par, dtype=np.uint8)


def qpp_perm(K: int, f1: int, f2: int) -> npt.NDArray[np.intp]:
    i = np.arange(K, dtype=np.int64)
    p = (f1 * i + f2 * i * i) % K
    if sorted(p.tolist()) != list(range(K)):
        raise ValueError("QPP parameters do not form a permutation")
    return p.astype(np.intp)


def interleave_qpp(
    x: "npt.NDArray[np.generic]", perm: npt.NDArray[np.intp]
) -> "npt.NDArray[np.generic]":
    return np.asarray(x)[perm]


def deinterleave_qpp(
    x: "npt.NDArray[np.generic]", perm: npt.NDArray[np.intp]
) -> "npt.NDArray[np.generic]":
    a = np.asarray(x)
    out = np.empty_like(a)
    out[perm] = a
    return out
