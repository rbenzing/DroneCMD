"""Channel-coding framework for DroneCMD (registry + codec interface).

Capability descriptors for all seven FEC families are registered here so the
catalog is a complete capability sheet; working codecs are provided for
``uncoded`` and ``repetition`` (P3a), with the heavy decoders arriving in later
sub-phases (P3b convolutional … P3h fountain).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Protocol, Tuple, Union, cast
from typing import Optional  # noqa: F401; used in Task 4 (RS decode)

import numpy as np
import numpy.typing as npt

from core.galois import (  # noqa: F401; used in Task 4 (RS decode)
    GF256,
    GF2m,
    berlekamp_massey,
    chien_search,
)

Bits = npt.NDArray[np.uint8]
LLRs = npt.NDArray[np.float64]
SoftOrHard = Union[Bits, LLRs]


class CodeFamily(Enum):
    UNCODED = "uncoded"
    REPETITION = "repetition"
    CONVOLUTIONAL = "convolutional"
    REED_SOLOMON = "reed_solomon"
    BCH = "bch"
    LDPC = "ldpc"
    TURBO = "turbo"
    POLAR = "polar"
    FOUNTAIN = "fountain"


@dataclass(frozen=True)
class CodingSpec:
    name: str
    family: CodeFamily
    k: int  # info bits/block (0 = rateless)
    n: int  # coded bits/block (0 = rateless)
    params: Mapping[str, object] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return float("nan") if self.n == 0 else self.k / self.n

    @property
    def soft_input(self) -> bool:
        return bool(self.params.get("soft_input", False))


@dataclass
class DecodeResult:
    bits: Bits
    meta: Mapping[str, object] = field(default_factory=dict)


class Codec(Protocol):
    spec: CodingSpec

    def encode(self, info_bits: Bits) -> Bits:
        ...

    def decode(self, received: SoftOrHard) -> DecodeResult:
        ...


class _Uncoded:
    def __init__(self, spec: CodingSpec) -> None:
        self.spec = spec

    def encode(self, info_bits: Bits) -> Bits:
        return np.asarray(info_bits, dtype=np.uint8)

    def decode(self, received: SoftOrHard) -> DecodeResult:
        r_arr = np.asarray(received)
        if r_arr.dtype.kind == "f":
            bits = (cast(npt.NDArray[np.float64], r_arr) < 0).astype(np.uint8)
        else:
            bits = cast(npt.NDArray[np.uint8], r_arr).astype(np.uint8)
        return DecodeResult(bits=bits)


class _Repetition:
    def __init__(self, spec: CodingSpec) -> None:
        self.spec = spec
        self.r = int(spec.params["r"])  # type: ignore[call-overload]

    def encode(self, info_bits: Bits) -> Bits:
        b = np.asarray(info_bits, dtype=np.uint8)
        return np.repeat(b, self.r).astype(np.uint8)

    def decode(self, received: SoftOrHard) -> DecodeResult:
        r_arr = np.asarray(received)
        if r_arr.dtype.kind == "f":
            hard = (cast(npt.NDArray[np.float64], r_arr) < 0).astype(np.uint8)
        else:
            hard = cast(npt.NDArray[np.uint8], r_arr).astype(np.uint8)
        groups = hard[: (hard.size // self.r) * self.r].reshape(-1, self.r)
        bits = (groups.sum(axis=1) * 2 > self.r).astype(np.uint8)  # majority
        return DecodeResult(bits=bits)


def _parity(x: int) -> int:
    """Parity (XOR of all bits) of a non-negative int."""
    return bin(x).count("1") & 1


def _puncture(
    coded: "npt.NDArray[np.generic]", pattern: Tuple[int, ...]
) -> "npt.NDArray[np.generic]":
    """Remove bits according to puncture pattern (keep where pattern==1).

    Args:
        coded: Input array of coded bits or values.
        pattern: Tuple of 0s and 1s indicating which positions to keep (1) or drop (0).
                 Pattern repeats cyclically over the input.

    Returns:
        Array with punctured positions removed.
    """
    if not pattern:
        return coded
    mask = np.resize(np.asarray(pattern, dtype=bool), coded.size)
    return cast("npt.NDArray[np.generic]", coded[mask])


def _depuncture(
    values: "npt.NDArray[np.generic]", pattern: Tuple[int, ...]
) -> "npt.NDArray[np.generic]":
    """Reinsert 0 at punctured positions, restoring the mother-code length.

    Args:
        values: Array of values (bits, LLRs, etc.) from the punctured code.
        pattern: Tuple of 0s and 1s matching the puncture pattern used.
                 Pattern repeats cyclically.

    Returns:
        Array with 0 inserted at punctured positions, restoring original length.
    """
    if not pattern:
        return values
    period = len(pattern)
    ones = sum(pattern)
    full_len = int(values.size) * period // ones
    out = np.zeros(full_len, dtype=values.dtype)
    j = 0
    for i in range(full_len):
        if pattern[i % period]:
            out[i] = values[j]
            j += 1
    return out


def _conv_encode(info_bits: Bits, generators: Tuple[int, int], k: int) -> Bits:
    """Rate-1/2, constraint-length-k convolutional encode with zero-tail.

    ``k-1`` zero bits are appended so the encoder starts and ends in state 0.
    Two output bits per input bit (generator taps XORed over the register).
    """
    g0, g1 = generators
    tail = k - 1
    stream = np.concatenate(
        [np.asarray(info_bits, dtype=np.uint8), np.zeros(tail, dtype=np.uint8)]
    )
    state = 0
    out = np.empty(2 * stream.size, dtype=np.uint8)
    top = 1 << (k - 1)
    mask = top - 1
    for i, u in enumerate(stream):
        reg = (int(u) * top) | state
        out[2 * i] = _parity(reg & g0)
        out[2 * i + 1] = _parity(reg & g1)
        state = (reg >> 1) & mask
    return out


def _viterbi_soft(
    llrs: "npt.NDArray[np.float64]", generators: Tuple[int, int], k: int
) -> Bits:
    """Soft-decision Viterbi over the rate-1/2 mother code (zero-tail).

    Maximizes total correlation ``sum (1-2c)*L``; erasures (L=0) contribute 0.
    Returns the info bits (the last k-1 tail bits are dropped).
    """
    n_states = 1 << (k - 1)
    n_stages = int(llrs.size) // 2
    if n_stages <= (k - 1):
        return np.zeros(0, dtype=np.uint8)
    top = 1 << (k - 1)
    mask = n_states - 1
    g0, g1 = generators
    nxt = np.zeros((n_states, 2), dtype=np.intp)
    e0 = np.zeros((n_states, 2), dtype=np.int8)
    e1 = np.zeros((n_states, 2), dtype=np.int8)
    for s in range(n_states):
        for u in (0, 1):
            reg = (u * top) | s
            e0[s, u] = _parity(reg & g0)
            e1[s, u] = _parity(reg & g1)
            nxt[s, u] = (reg >> 1) & mask
    neg = -1e18
    pm = np.full(n_states, neg, dtype=np.float64)
    pm[0] = 0.0
    prev = np.full((n_stages, n_states), -1, dtype=np.intp)
    inbit = np.zeros((n_stages, n_states), dtype=np.int8)
    for t in range(n_stages):
        l0 = float(llrs[2 * t])
        l1 = float(llrs[2 * t + 1])
        npm = np.full(n_states, neg, dtype=np.float64)
        for s in range(n_states):
            if pm[s] == neg:
                continue
            for u in (0, 1):
                ns = int(nxt[s, u])
                bm = (1 - 2 * int(e0[s, u])) * l0 + (1 - 2 * int(e1[s, u])) * l1
                cand = pm[s] + bm
                if cand > npm[ns]:
                    npm[ns] = cand
                    prev[t, ns] = s
                    inbit[t, ns] = u
        pm = npm
    s = 0  # zero-tail: terminal state is 0
    bits = np.zeros(n_stages, dtype=np.uint8)
    for t in range(n_stages - 1, -1, -1):
        bits[t] = inbit[t, s]
        s = int(prev[t, s])
        if s < 0:
            break
    return bits[: n_stages - (k - 1)]


class _Convolutional:
    def __init__(self, spec: CodingSpec) -> None:
        self.spec = spec
        self.k = int(spec.params["constraint_length"])  # type: ignore[call-overload]
        gens = spec.params["generators_octal"]
        self.generators = (int(gens[0]), int(gens[1]))  # type: ignore[index]
        self.puncture: Tuple[int, ...] = tuple(spec.params.get("puncture", ()))  # type: ignore[arg-type]

    def encode(self, info_bits: Bits) -> Bits:
        coded = _conv_encode(info_bits, self.generators, self.k)
        return cast(Bits, _puncture(coded, self.puncture))

    def decode(self, received: SoftOrHard) -> DecodeResult:
        r = np.asarray(received)
        llr = (
            r.astype(np.float64)
            if r.dtype.kind == "f"
            else (1.0 - 2.0 * r.astype(np.float64))
        )
        full = cast("npt.NDArray[np.float64]", _depuncture(llr, self.puncture))
        bits = _viterbi_soft(full, self.generators, self.k)
        return DecodeResult(bits=bits)


def _bits_to_symbols(bits: Bits) -> List[int]:
    b = np.asarray(bits, dtype=np.uint8)
    if b.size % 8 != 0:
        raise ValueError("RS requires a byte-aligned bit stream")
    packed = np.packbits(b)
    return [int(v) for v in packed]


def _symbols_to_bits(syms: List[int]) -> Bits:
    arr = np.array(syms, dtype=np.uint8)
    return cast(Bits, np.unpackbits(arr).astype(np.uint8))


def _rs_generator_poly(field: GF2m, nsym: int, fcr: int) -> List[int]:
    g = [1]
    for i in range(nsym):
        g = field.poly_mul(g, [1, field.pow(2, i + fcr)])
    return g


def _rs_encode_symbols(field: GF2m, msg: List[int], nsym: int, fcr: int) -> List[int]:
    gen = _rs_generator_poly(field, nsym, fcr)
    _, remainder = field.poly_div(msg + [0] * (len(gen) - 1), gen)
    return msg + remainder


class _ReedSolomon:
    def __init__(self, spec: CodingSpec) -> None:
        self.spec = spec
        self.m = int(spec.params["gf_m"])  # type: ignore[call-overload]
        self.t = int(spec.params["t"])  # type: ignore[call-overload]
        self.fcr = int(spec.params.get("fcr", 1))  # type: ignore[arg-type]
        prim = int(spec.params.get("prim_poly", 0x11D))  # type: ignore[arg-type]
        self.nsym = 2 * self.t
        self.field = GF256 if (self.m == 8 and prim == 0x11D) else GF2m(self.m, prim)
        self.erasure_factor = float(spec.params.get("erasure_factor", 0.5))  # type: ignore[arg-type]

    def encode(self, info_bits: Bits) -> Bits:
        msg = _bits_to_symbols(info_bits)
        coded = _rs_encode_symbols(self.field, msg, self.nsym, self.fcr)
        return _symbols_to_bits(coded)

    def decode(self, received: SoftOrHard) -> DecodeResult:
        raise NotImplementedError("RS decode implemented in Task 4")


def make_codec(spec: CodingSpec) -> Codec:
    if spec.family == CodeFamily.UNCODED:
        return _Uncoded(spec)
    if spec.family == CodeFamily.REPETITION:
        return _Repetition(spec)
    if spec.family == CodeFamily.CONVOLUTIONAL:
        return _Convolutional(spec)
    if spec.family == CodeFamily.REED_SOLOMON:
        return _ReedSolomon(spec)
    raise NotImplementedError(
        f"{spec.family.value}: implemented in a later P3 sub-phase"
    )


CODING_CATALOG: Dict[str, CodingSpec] = {
    "uncoded": CodingSpec("uncoded", CodeFamily.UNCODED, 1, 1),
    "rep3": CodingSpec("rep3", CodeFamily.REPETITION, 1, 3, {"r": 3}),
    # --- capability descriptors (codecs land in later sub-phases) ---
    "conv_k7_r12": CodingSpec(
        "conv_k7_r12",
        CodeFamily.CONVOLUTIONAL,
        1,
        2,
        {
            "constraint_length": 7,
            "generators_octal": (0o133, 0o171),
            "soft_input": True,
        },
    ),
    "conv_k7_r23": CodingSpec(
        "conv_k7_r23",
        CodeFamily.CONVOLUTIONAL,
        2,
        3,
        {
            "constraint_length": 7,
            "generators_octal": (0o133, 0o171),
            "puncture": (1, 1, 1, 0),
            "soft_input": True,
        },
    ),
    "conv_k7_r34": CodingSpec(
        "conv_k7_r34",
        CodeFamily.CONVOLUTIONAL,
        3,
        4,
        {
            "constraint_length": 7,
            "generators_octal": (0o133, 0o171),
            "puncture": (1, 1, 1, 0, 0, 1),
            "soft_input": True,
        },
    ),
    "rs_255_223": CodingSpec(
        "rs_255_223",
        CodeFamily.REED_SOLOMON,
        223 * 8,
        255 * 8,
        {
            "gf_m": 8,
            "t": 16,
            "symbol_bits": 8,
            "prim_poly": 0x11D,
            "fcr": 1,
            "soft_input": True,
            "erasure_factor": 0.5,
        },
    ),
    "rs_255_239": CodingSpec(
        "rs_255_239",
        CodeFamily.REED_SOLOMON,
        239 * 8,
        255 * 8,
        {
            "gf_m": 8,
            "t": 8,
            "symbol_bits": 8,
            "prim_poly": 0x11D,
            "fcr": 1,
            "soft_input": True,
            "erasure_factor": 0.5,
        },
    ),
    "bch_63_51": CodingSpec("bch_63_51", CodeFamily.BCH, 51, 63, {"gf_m": 6, "t": 2}),
    "ldpc_648_r12": CodingSpec(
        "ldpc_648_r12", CodeFamily.LDPC, 324, 648, {"soft_input": True, "max_iters": 50}
    ),
    "turbo_r13": CodingSpec(
        "turbo_r13",
        CodeFamily.TURBO,
        1,
        3,
        {
            "constraint_length": 4,
            "generators_octal": (0o13, 0o15),
            "soft_input": True,
            "max_iters": 8,
        },
    ),
    "polar_256_128": CodingSpec(
        "polar_256_128",
        CodeFamily.POLAR,
        128,
        256,
        {"soft_input": True, "list_size": 8},
    ),
    "fountain_lt": CodingSpec(
        "fountain_lt",
        CodeFamily.FOUNTAIN,
        0,
        0,
        {"kind": "lt", "c": 0.03, "delta": 0.5},
    ),
}


def coding_names() -> List[str]:
    return list(CODING_CATALOG)


_CRC16_POLY = 0x1021


def crc16_ccitt(data_bits: Bits) -> Bits:
    """CRC-16-CCITT (poly 0x1021, init 0xFFFF) over an MSB-first bit array."""
    reg = 0xFFFF
    for bit in np.asarray(data_bits, dtype=np.uint8):
        reg ^= int(bit) << 15
        reg = (
            ((reg << 1) ^ _CRC16_POLY) & 0xFFFF
            if (reg & 0x8000)
            else (reg << 1) & 0xFFFF
        )
    return np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.uint8)


def frame_with_crc(payload_bits: Bits) -> Bits:
    """Append CRC-16-CCITT to payload bits."""
    p = np.asarray(payload_bits, dtype=np.uint8)
    result = np.concatenate([p, crc16_ccitt(p)]).astype(np.uint8)
    return cast(Bits, result)


def check_and_strip_crc(frame_bits: Bits) -> Tuple[Bits, bool]:
    """Verify CRC-16-CCITT and extract payload; return (payload, crc_ok)."""
    f = np.asarray(frame_bits, dtype=np.uint8)
    if f.size < 16:
        return np.zeros(0, dtype=np.uint8), False
    payload, crc = f[:-16], f[-16:]
    ok = bool(np.array_equal(crc, crc16_ccitt(payload)))
    return payload.astype(np.uint8), ok


# Default block-interleaver depth for profile-carried coding.
CODING_INTERLEAVE_DEPTH = 8


def _perm(n: int, depth: int) -> npt.NDArray[np.intp]:
    """Compute permutation indices for rectangular block interleaver."""
    if depth <= 1:
        return np.arange(n, dtype=np.intp)
    rows = int(np.ceil(n / depth))
    idx = np.arange(rows * depth, dtype=np.intp).reshape(rows, depth).T.reshape(-1)
    return idx[idx < n]


def interleave(x: "npt.NDArray[np.generic]", depth: int) -> "npt.NDArray[np.generic]":
    """Interleave array x using rectangular block interleaver at given depth.

    Args:
        x: Input array (bits, LLRs, or other numeric types).
        depth: Interleaver depth; depth <= 1 is identity.

    Returns:
        Interleaved array with same shape and dtype as x.
    """
    a = np.asarray(x)
    return a[_perm(a.size, depth)]


def deinterleave(x: "npt.NDArray[np.generic]", depth: int) -> "npt.NDArray[np.generic]":
    """Deinterleave array x using rectangular block deinterleaver at given depth.

    Args:
        x: Input array (bits, LLRs, or other numeric types).
        depth: Deinterleaver depth; depth <= 1 is identity.

    Returns:
        Deinterleaved array with same shape and dtype as x.
    """
    a = np.asarray(x)
    p = _perm(a.size, depth)
    out = np.empty_like(a)
    out[p] = a
    return out
