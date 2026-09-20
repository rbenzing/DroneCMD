"""Raptor-style fountain code PHY for DroneCMD: Robust-Soliton LT inner code
over a systematic sparse precode, per-symbol-CRC erasure detection, and GF(2)
Gaussian-elimination decoding. Reproducible/self-consistent (RFC-inexact),
small-K demonstration scope (see design 0013). No framing import — the outer
CRC-16 lives in core.coding; fountain uses its own per-symbol CRC-8 and an
internal length header.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.typing as npt

Bits = npt.NDArray[np.uint8]
Real = npt.NDArray[np.float64]

_CRC8_POLY = 0x07  # CRC-8 (poly 0x07, init 0x00), MSB-first over a bit array


def crc8(bits: Bits) -> Bits:
    """Compute CRC-8 over a bit array.

    CRC-8 using polynomial 0x07 with zero initialization, MSB-first, over the
    input bit array. Returns the 8-bit CRC as a uint8 array.

    Args:
        bits: Input bit array (uint8 with values 0 or 1).

    Returns:
        8-element uint8 array containing the CRC bits.
    """
    reg = 0
    for bit in np.asarray(bits, dtype=np.uint8):
        reg ^= (int(bit) & 1) << 7
        reg = ((reg << 1) ^ _CRC8_POLY) & 0xFF if (reg & 0x80) else (reg << 1) & 0xFF
    return np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.uint8)


def robust_soliton(n_deg: int, c: float, delta: float) -> Real:
    """Compute Robust Soliton degree distribution.

    Robust Soliton probability mass function over degrees 1..n_deg, with index 0
    reserved (set to 0.0, unused). Parameters control ripple size and coverage
    guarantees per the Robust Soliton construction.

    Args:
        n_deg: Maximum degree (K in the source).
        c: Ripple-size scaling constant.
        delta: Target failure probability.

    Returns:
        Array of shape (n_deg + 1,) of float64, with pmf[0] = 0.0 and
        pmf[1:].sum() = 1.0 (a valid probability mass function).
    """
    mu = np.zeros(n_deg + 1, dtype=np.float64)
    rho = np.zeros(n_deg + 1, dtype=np.float64)
    rho[1] = 1.0 / n_deg
    for d in range(2, n_deg + 1):
        rho[d] = 1.0 / (d * (d - 1))
    s = c * np.log(n_deg / delta) * np.sqrt(n_deg)  # expected ripple size
    kf = max(int(np.floor(n_deg / s)), 1) if s > 0 else n_deg
    tau = np.zeros(n_deg + 1, dtype=np.float64)
    for d in range(1, min(kf, n_deg + 1)):
        tau[d] = s / (n_deg * d)
    if 1 <= kf <= n_deg:
        tau[kf] = s * np.log(s / delta) / n_deg if s > 0 else 0.0
    mu = rho + tau
    total = float(mu.sum())
    if total <= 0:
        mu[1] = 1.0
        total = 1.0
    return mu / total


def sample_degree(pmf: Real, rng: np.random.Generator) -> int:
    """Draw a degree from the Robust Soliton pmf.

    Sample a degree from the given probability mass function using the provided
    Generator (deterministic per seeded RNG state). Samples via CDF inverse
    transform.

    Args:
        pmf: Probability mass function (e.g., from robust_soliton()).
        rng: numpy.random.Generator for deterministic, seeded sampling.

    Returns:
        Sampled degree as an integer.
    """
    cdf = np.cumsum(pmf)
    u = float(rng.random()) * float(cdf[-1])
    return int(np.searchsorted(cdf, u, side="left"))


def symbol_neighbors(
    rng: np.random.Generator, degree: int, span: int
) -> npt.NDArray[np.intp]:
    """Draw distinct neighbor indices for LT encoding.

    Return `degree` distinct sorted indices in [0, span) drawn deterministically
    from the given Generator. Clamps degree to [1, span] to ensure valid output.

    Args:
        rng: numpy.random.Generator for deterministic, seeded sampling.
        degree: Requested number of distinct neighbors.
        span: Maximum index (exclusive); number of symbols to choose from.

    Returns:
        Sorted array of intp indices; size = min(degree, span), all in [0, span).
    """
    d = max(1, min(degree, span))
    return np.sort(rng.choice(span, size=d, replace=False)).astype(np.intp)


def build_precode(
    k: int, seed: int, precode_rate: float, precode_degree: int
) -> List[npt.NDArray[np.intp]]:
    """R = round(k*(1/precode_rate - 1)) systematic parity rows over k sources.

    Parity row j = a seeded sparse subset (size min(precode_degree,k)) of source
    indices. A distinct seed offset keeps precode structure uncorrelated with
    the LT layer. Deterministic given (k, seed).

    Args:
        k: Number of source symbols.
        seed: Base seed for deterministic RNG initialization.
        precode_rate: Code rate (must be < 1.0 for non-empty parity).
        precode_degree: Maximum degree per parity row.

    Returns:
        List of R sorted arrays of intp indices, each representing source
        connections for a parity row. Empty list if rate >= 1.0 or k <= 0.
    """
    if precode_rate >= 1.0 or k <= 0:
        return []
    r = int(round(k * (1.0 / precode_rate - 1.0)))
    rows: List[npt.NDArray[np.intp]] = []
    for j in range(r):
        rng = np.random.default_rng((seed + 0x9E3779B1 + j) & 0xFFFFFFFF)
        deg = max(1, min(precode_degree, k))
        rows.append(np.sort(rng.choice(k, size=deg, replace=False)).astype(np.intp))
    return rows


def _to_symbols(payload_bits: Bits, symbol_bits: int) -> Bits:
    """Convert a flat payload bit array into symbols (rows of S bits each).

    Zero-pads the payload to a multiple of symbol_bits, then reshapes to a
    2D array of shape (K, S) where K = ceil(len(payload_bits) / S).

    Args:
        payload_bits: Flat bit array (uint8 with values 0 or 1).
        symbol_bits: Number of bits per symbol (S).

    Returns:
        2D array of shape (K, S) containing the symbolized payload.
    """
    n = payload_bits.size
    k = int(np.ceil(n / symbol_bits))
    padded = np.zeros(k * symbol_bits, dtype=np.uint8)
    padded[:n] = payload_bits
    return padded.reshape(k, symbol_bits)


def fountain_encode(
    payload_bits: Bits,
    *,
    symbol_bits: int,
    seed: int,
    c: float,
    delta: float,
    precode_rate: float,
    precode_degree: int,
    overhead: float,
) -> Bits:
    """LT-encode payload bits using a Robust-Soliton code with systematic precode.

    Converts payload to K source symbols of S bits each, builds R precode parity
    symbols (XOR of source neighbors), constructs L=K+R intermediate symbols,
    then generates N=ceil((1+overhead)*K) output symbols via Robust-Soliton
    degree sampling and XOR combination, appending per-symbol CRC-8.

    Args:
        payload_bits: Flat bit array (uint8 with values 0 or 1).
        symbol_bits: Number of bits per symbol (S).
        seed: Base seed for deterministic RNG (seeded per output symbol).
        c: Robust Soliton ripple-size constant.
        delta: Robust Soliton target failure probability.
        precode_rate: Code rate for the systematic precode (must be < 1.0).
        precode_degree: Maximum degree per precode parity row.
        overhead: Relative overhead for N vs K (N = ceil((1 + overhead) * K)).

    Returns:
        Flat bit array of size N * (S + 8) containing N encoded symbols,
        each with S data bits and 8 CRC bits.
    """
    src = _to_symbols(np.asarray(payload_bits, dtype=np.uint8), symbol_bits)  # (K,S)
    k = src.shape[0]
    parity_rows = build_precode(k, seed, precode_rate, precode_degree)
    parity = (
        np.array(
            [np.bitwise_xor.reduce(src[row], axis=0) for row in parity_rows],
            dtype=np.uint8,
        ).reshape(-1, symbol_bits)
        if parity_rows
        else np.zeros((0, symbol_bits), dtype=np.uint8)
    )
    inter = np.concatenate([src, parity], axis=0)  # (L,S), L=K+R
    L = inter.shape[0]
    n_out = int(np.ceil((1.0 + overhead) * k))
    pmf = robust_soliton(L, c, delta)
    out = np.empty((n_out, symbol_bits + 8), dtype=np.uint8)
    for i in range(n_out):
        rng = np.random.default_rng((seed + i) & 0xFFFFFFFF)
        deg = sample_degree(pmf, rng)
        nbrs = symbol_neighbors(rng, deg, L)
        enc = np.bitwise_xor.reduce(inter[nbrs], axis=0).astype(np.uint8)
        out[i, :symbol_bits] = enc
        out[i, symbol_bits:] = crc8(enc)
    return out.reshape(-1)


def gf2_solve(
    rows: List[int], rhs: Bits, n_cols: int
) -> Tuple[Bits, npt.NDArray[np.bool_]]:
    """Solve a GF(2) linear system A * x = rhs by Gauss-Jordan elimination.

    Each row of `A` is packed as a Python int bitmask over `n_cols` columns
    (bit c set means column c participates in that row's XOR constraint);
    `rhs` carries the corresponding S-bit right-hand side for each row. The
    system is reduced to row-reduced echelon form (RREF), applying the same
    row XORs to the S-bit RHS vectors so that each pivot row's RHS ends up
    holding the resolved value for its pivot column. A column is only
    reported as resolved when its pivot row is a "clean" single-bit row
    (i.e. no other column's bit remains set in that row) -- this rules out
    columns that never received a full row reduction (e.g. because the
    system ran out of rows before that column could be isolated).

    Args:
        rows: Row incidence bitmasks, one Python int per row; bit c set
            means column c appears in that row's XOR.
        rhs: (n_rows, S) uint8 array; row r holds the S-bit RHS for `rows[r]`.
        n_cols: Total number of columns (L intermediates) in the system.

    Returns:
        Tuple of:
            x: (n_cols, S) uint8 array; x[c] is the resolved value for
                column c if resolved[c] is True, else all-zero.
            resolved: (n_cols,) bool array; True where column c was
                recovered as a clean single-bit pivot.
    """
    a = list(rows)
    b = rhs.copy()
    n_rows = len(a)
    s = b.shape[1]
    pivot_row_of_col: List[int] = [-1] * n_cols
    r = 0
    for c in range(n_cols):
        piv = -1
        for rr in range(r, n_rows):
            if (a[rr] >> c) & 1:
                piv = rr
                break
        if piv == -1:
            continue
        a[r], a[piv] = a[piv], a[r]
        b[[r, piv]] = b[[piv, r]]
        for rr in range(n_rows):
            if rr != r and ((a[rr] >> c) & 1):
                a[rr] ^= a[r]
                b[rr] ^= b[r]
        pivot_row_of_col[c] = r
        r += 1
        if r == n_rows:
            break
    x = np.zeros((n_cols, s), dtype=np.uint8)
    resolved = np.zeros(n_cols, dtype=np.bool_)
    for c in range(n_cols):
        pr = pivot_row_of_col[c]
        if pr != -1 and a[pr] == (1 << c):  # clean single-bit pivot
            x[c] = b[pr]
            resolved[c] = True
    return x, resolved


def _mask(indices: npt.NDArray[np.intp]) -> int:
    """Pack an array of column indices into a single GF(2) row bitmask.

    Args:
        indices: Array of column indices (e.g. LT/precode neighbor set).

    Returns:
        Python int with bit i set for each i in `indices`.
    """
    m = 0
    for i in indices.tolist():
        m |= 1 << int(i)
    return m


def fountain_decode(
    coded_bits: Bits,
    *,
    symbol_bits: int,
    seed: int,
    c: float,
    delta: float,
    precode_rate: float,
    precode_degree: int,
    overhead: float,
) -> Tuple[Bits, bool, int]:
    """Decode fountain-coded bits back to the original payload.

    Replays the encoder's exact seeded structure: recovers K (source symbol
    count) from N (output symbol count) and `overhead`, rebuilds the R
    systematic precode parity rows via `build_precode`, and for each
    surviving output symbol (CRC-8 match) replays `default_rng(seed + i)` ->
    `sample_degree` -> `symbol_neighbors` in the same order the encoder used
    to recover its LT neighbor set. A symbol whose per-symbol CRC-8 fails is
    treated as erased and excluded from the GF(2) system. The L = K + R
    intermediate symbols (K sources + R precode parities) are then solved by
    GF(2) Gaussian elimination (`gf2_solve`) over the surviving LT rows plus
    the R precode constraint rows (parity XOR its source neighbors = 0).
    Decoding succeeds only if all K source columns resolve as clean pivots;
    otherwise it fails loudly rather than returning a partially- or
    incorrectly-recovered payload.

    Args:
        coded_bits: Flat bit array as produced by `fountain_encode` (N
            symbols of `symbol_bits` + 8 CRC bits each), optionally with
            some symbols' data bits corrupted (erasures).
        symbol_bits: Number of data bits per symbol (S); must match the
            encoder's `symbol_bits`.
        seed: Base seed; must match the encoder's `seed`.
        c: Robust Soliton ripple-size constant; must match the encoder's `c`.
        delta: Robust Soliton target failure probability; must match the
            encoder's `delta`.
        precode_rate: Systematic precode rate; must match the encoder's
            `precode_rate`.
        precode_degree: Max degree per precode parity row; must match the
            encoder's `precode_degree`.
        overhead: Relative overhead N vs K; must match the encoder's
            `overhead`.

    Returns:
        Tuple of:
            payload_bits: Flat uint8 bit array of the K recovered source
                symbols (K * symbol_bits bits), valid only if `ok` is True;
                all-zero/empty array on failure.
            ok: True iff all K source symbols were successfully recovered.
            n_erased: Number of output symbols dropped due to CRC-8
                mismatch.
    """
    sym = np.asarray(coded_bits, dtype=np.uint8).reshape(-1, symbol_bits + 8)
    n_out = sym.shape[0]
    # Recover K from N and overhead: the unique cand with
    # ceil((1+overhead)*cand) == n_out (ceil((1+overhead)*.) is
    # non-decreasing in cand, so at most one candidate matches).
    k = -1
    for cand in range(1, n_out + 1):
        if int(np.ceil((1.0 + overhead) * cand)) == n_out:
            k = cand
            break
    if k < 1:
        return np.zeros(0, dtype=np.uint8), False, 0
    parity_rows = build_precode(k, seed, precode_rate, precode_degree)
    L = k + len(parity_rows)
    pmf = robust_soliton(L, c, delta)
    a_rows: List[int] = []
    rhs: List[Bits] = []
    n_erased = 0
    for i in range(n_out):
        data = sym[i, :symbol_bits]
        if not np.array_equal(crc8(data), sym[i, symbol_bits:]):
            n_erased += 1
            continue
        rng = np.random.default_rng((seed + i) & 0xFFFFFFFF)
        deg = sample_degree(pmf, rng)
        nbrs = symbol_neighbors(rng, deg, L)
        a_rows.append(_mask(nbrs))
        rhs.append(data)
    # Precode constraint rows: parity_col (k+j) XOR its source neighbors = 0.
    for j, srow in enumerate(parity_rows):
        a_rows.append(_mask(srow) | (1 << (k + j)))
        rhs.append(np.zeros(symbol_bits, dtype=np.uint8))
    if not a_rows:
        return np.zeros(0, dtype=np.uint8), False, n_erased
    inter, resolved = gf2_solve(a_rows, np.array(rhs, dtype=np.uint8), L)
    if not bool(resolved[:k].all()):
        return np.zeros(0, dtype=np.uint8), False, n_erased
    payload_bits = inter[:k].reshape(-1)
    return payload_bits.astype(np.uint8), True, n_erased
