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


def turbo_encode(info: Bits, perm: npt.NDArray[np.intp]) -> Bits:
    """Turbo encoder (rate 1/3 systematic).

    Layout: [info(K) | tail1_sys(3) | tail2_sys(3) | par1(K+3) | par2(K+3)]
    Total length: 3*K + 12
    """
    info = np.asarray(info, dtype=np.uint8)
    K = info.size
    sys1, par1 = rsc_encode(info)  # len K+3
    sys2, par2 = rsc_encode(info[perm])  # len K+3
    tail1_sys = sys1[K:]  # 3
    tail2_sys = sys2[K:]  # 3
    out = np.concatenate([info, tail1_sys, tail2_sys, par1, par2])
    return out.astype(np.uint8)  # 3K + 12


_NEG = -1e30


def bcjr_maxlogmap(ls: LLRs, lp: LLRs, la: LLRs) -> LLRs:
    """max-log-MAP SISO over the terminated RSC trellis (start/end state 0).

    ls/lp/la: systematic / parity / a-priori LLRs (len N). Returns extrinsic LLR.
    Convention L>0 => bit 0. gamma = 0.5*((1-2u)(ls+la) + (1-2z)lp).
    """
    N = int(ls.size)
    alpha = np.full((N + 1, NSTATES), _NEG)
    beta = np.full((N + 1, NSTATES), _NEG)
    alpha[0, 0] = 0.0
    beta[N, 0] = 0.0

    def gam(k: int, s: int, u: int) -> float:
        z = PAR[s][u]
        return 0.5 * ((1 - 2 * u) * (ls[k] + la[k]) + (1 - 2 * z) * lp[k])

    for k in range(N):
        for s in range(NSTATES):
            a = alpha[k, s]
            if a == _NEG:
                continue
            for u in (0, 1):
                ns = NXT[s][u]
                m = a + gam(k, s, u)
                if m > alpha[k + 1, ns]:
                    alpha[k + 1, ns] = m
    for k in range(N - 1, -1, -1):
        for s in range(NSTATES):
            for u in (0, 1):
                ns = NXT[s][u]
                b = beta[k + 1, ns]
                if b == _NEG:
                    continue
                m = b + gam(k, s, u)
                if m > beta[k, s]:
                    beta[k, s] = m
    le = np.zeros(N, dtype=np.float64)
    for k in range(N):
        m0 = _NEG
        m1 = _NEG
        for s in range(NSTATES):
            a = alpha[k, s]
            if a == _NEG:
                continue
            for u in (0, 1):
                ns = NXT[s][u]
                b = beta[k + 1, ns]
                if b == _NEG:
                    continue
                metric = a + gam(k, s, u) + b
                if u == 0:
                    if metric > m0:
                        m0 = metric
                else:
                    if metric > m1:
                        m1 = metric
        le[k] = (m0 - m1) - ls[k] - la[k]
    return le


def turbo_decode(
    ls_info: LLRs,
    ls_tail1: LLRs,
    ls_tail2: LLRs,
    lp1: LLRs,
    lp2: LLRs,
    perm: npt.NDArray[np.intp],
    max_iters: int = 8,
    scale: float = 0.7,
) -> Bits:
    """Iterative extrinsic-exchange max-log-MAP turbo decoder.

    Returns hard-decided info bits (length K). LLR convention L>0 => bit 0.
    """
    K = int(ls_info.size)
    ls1 = np.concatenate([ls_info, ls_tail1])  # decoder 1 systematic (K+3)
    ls2 = np.concatenate([ls_info[perm], ls_tail2])  # decoder 2 systematic (K+3)
    la1_info = np.zeros(K, dtype=np.float64)
    le1_info = np.zeros(K, dtype=np.float64)
    de = np.zeros(K, dtype=np.float64)
    for _ in range(max_iters):
        la1 = np.concatenate([la1_info, np.zeros(3)])
        le1 = bcjr_maxlogmap(ls1, lp1, la1)
        le1_info = le1[:K]
        la2_info = scale * le1_info[perm]
        la2 = np.concatenate([la2_info, np.zeros(3)])
        le2 = bcjr_maxlogmap(ls2, lp2, la2)
        le2_info = le2[:K]
        # deinterleave le2 extrinsic back to natural order
        de = np.empty(K, dtype=np.float64)
        de[perm] = le2_info
        la1_info = scale * de
    total = ls_info + le1_info + de
    return (total < 0).astype(np.uint8)


PUNCTURE_R12: Tuple[int, ...] = (
    1,
    0,
    0,
    1,
)  # keep p1 even / p2 odd (applied to the two parity streams)


def punctured_parity_masks(K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for de-puncturing in rate-1/2 decoder.

    mask1: drop odd-index par1 over info region (K bits)
    mask2: drop even-index par2 over info region (K bits)
    Both masks keep tails in full.
    """
    mask1 = np.ones(K + 3, dtype=bool)
    mask1[:K][1::2] = False  # drop odd-index par1 over info

    mask2 = np.ones(K + 3, dtype=bool)
    mask2[:K][0::2] = False  # drop even-index par2 over info

    return mask1, mask2


def turbo_encode_punctured(info: Bits, perm: npt.NDArray[np.intp]) -> Bits:
    """Rate 1/2: keep all systematic + tails, keep alternating parity bits."""
    info = np.asarray(info, dtype=np.uint8)
    K = info.size
    sys1, par1 = rsc_encode(info)
    sys2, par2 = rsc_encode(info[perm])
    head = np.concatenate([info, sys1[K:], sys2[K:]])  # systematic + tails
    # keep par1 on even info positions, par2 on odd (tails kept in full)
    mask1, mask2 = punctured_parity_masks(K)
    out = np.concatenate([head, par1[mask1], par2[mask2]])
    return out.astype(np.uint8)
