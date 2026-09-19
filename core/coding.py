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

import numpy as np
import numpy.typing as npt

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


class _Convolutional:
    def __init__(self, spec: CodingSpec) -> None:
        self.spec = spec
        self.k = int(spec.params["constraint_length"])  # type: ignore[call-overload]
        gens = spec.params["generators_octal"]
        self.generators = (int(gens[0]), int(gens[1]))  # type: ignore[index]
        self.puncture: Tuple[int, ...] = tuple(spec.params.get("puncture", ()))  # type: ignore[arg-type]

    def encode(self, info_bits: Bits) -> Bits:
        coded = _conv_encode(info_bits, self.generators, self.k)
        return coded  # puncturing added in Task 2

    def decode(self, received: SoftOrHard) -> DecodeResult:
        raise NotImplementedError("convolutional decode: P3b Task 3")


def make_codec(spec: CodingSpec) -> Codec:
    if spec.family == CodeFamily.UNCODED:
        return _Uncoded(spec)
    if spec.family == CodeFamily.REPETITION:
        return _Repetition(spec)
    if spec.family == CodeFamily.CONVOLUTIONAL:
        return _Convolutional(spec)
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
    "rs_255_223": CodingSpec(
        "rs_255_223",
        CodeFamily.REED_SOLOMON,
        223 * 8,
        255 * 8,
        {"gf_m": 8, "t": 16, "symbol_bits": 8},
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
