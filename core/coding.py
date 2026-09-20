"""Channel-coding framework for DroneCMD (registry + codec interface).

Capability descriptors for all seven FEC families are registered here so the
catalog is a complete capability sheet; working codecs are provided for
``uncoded`` and ``repetition`` (P3a), with the heavy decoders arriving in later
sub-phases (P3b convolutional … P3h fountain).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional  # noqa: F401; used in Task 4 (RS decode)
from typing import Dict, List, Mapping, Protocol, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from core import ldpc as _ldpc_mod
from core import polar as _polar_mod
from core import turbo as _turbo_mod
from core.galois import GF256, GF2m, berlekamp_massey, chien_search

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


def _rs_syndromes(field: GF2m, r: List[int], nsym: int, fcr: int) -> List[int]:
    # leading 0 pad matches the BM/forney indexing convention
    return [0] + [field.poly_eval(r, field.pow(2, j + fcr)) for j in range(nsym)]


def _rs_errata_locator(field: GF2m, e_pos_rev: List[int]) -> List[int]:
    e_loc = [1]
    for i in e_pos_rev:
        e_loc = field.poly_mul(e_loc, field.poly_add([1], [field.pow(2, i), 0]))
    return e_loc


def _rs_error_evaluator(
    field: GF2m, synd: List[int], err_loc: List[int], nsym: int
) -> List[int]:
    _, rem = field.poly_div(field.poly_mul(synd, err_loc), [1] + [0] * (nsym + 1))
    return rem


def _rs_correct_errata(
    field: GF2m, r: List[int], synd: List[int], err_pos: List[int], fcr: int
) -> List[int]:
    n = len(r)
    coef_pos = [n - 1 - p for p in err_pos]
    err_loc = _rs_errata_locator(field, coef_pos)
    err_eval = _rs_error_evaluator(field, synd[::-1], err_loc, len(err_loc) - 1)[::-1]
    x_list = [field.pow(2, p) for p in coef_pos]
    e = [0] * n
    for i, xi in enumerate(x_list):
        xi_inv = field.inv(xi)
        prime = 1
        for j in range(len(x_list)):
            if j != i:
                prime = field.mul(prime, field.add(1, field.mul(xi_inv, x_list[j])))
        y = field.poly_eval(err_eval[::-1], xi_inv)
        y = field.mul(field.pow(xi, 1 - fcr), y)
        if prime == 0:
            raise ValueError("singular errata magnitude")
        e[err_pos[i]] = field.div(y, prime)
    return field.poly_add(r, e)[-n:]


def _rs_forney_syndromes(
    field: GF2m, synd: List[int], pos: List[int], n: int
) -> List[int]:
    pos_rev = [n - 1 - p for p in pos]
    fsynd = list(synd[1:])
    for i in range(len(pos)):
        x = field.pow(2, pos_rev[i])
        for j in range(len(fsynd) - 1):
            fsynd[j] = field.mul(fsynd[j], x) ^ fsynd[j + 1]
    return fsynd


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

    def _decode_symbols(
        self, r: List[int], erase_pos: List[int]
    ) -> Tuple[List[int], bool]:
        n = len(r)
        k = n - self.nsym
        if k <= 0:
            return [], False
        if len(erase_pos) > self.nsym:
            return r[:k], False  # too many erasures alone
        synd = _rs_syndromes(self.field, r, self.nsym, self.fcr)
        if max(synd) == 0:
            return r[:k], True  # clean
        try:
            fsynd = _rs_forney_syndromes(self.field, synd, erase_pos, n)
            # Convention A: BM finds the ERROR locator over the Forney-reduced
            # syndromes; the known erasure positions are combined below via
            # erase_pos + err_pos in _rs_correct_errata (matches reedsolo).
            err_loc = berlekamp_massey(
                self.field, fsynd, self.nsym, erase_count=len(erase_pos)
            )
            err_pos = chien_search(self.field, err_loc[::-1], n)
            corrected = _rs_correct_errata(
                self.field, r, synd, erase_pos + err_pos, self.fcr
            )
            check = _rs_syndromes(self.field, corrected, self.nsym, self.fcr)
            if max(check) != 0:
                return r[:k], False
            return corrected[:k], True
        except (ValueError, ZeroDivisionError):
            return r[:k], False

    def _erasures_from_llrs(self, llrs: "npt.NDArray[np.float64]") -> List[int]:
        usable = (llrs.size // 8) * 8
        if usable == 0:
            return []
        mag = np.abs(llrs[:usable]).reshape(-1, 8)
        sym_rel = mag.min(axis=1)  # least-reliable bit per symbol
        med = float(np.median(np.abs(llrs[:usable])))
        thresh = self.erasure_factor * med
        flagged = np.nonzero(sym_rel < thresh)[0]
        if flagged.size <= self.nsym:
            return [int(p) for p in flagged]
        # cap at nsym: keep the least reliable
        order = flagged[np.argsort(sym_rel[flagged])]
        return sorted(int(p) for p in order[: self.nsym])

    def decode(self, received: SoftOrHard) -> DecodeResult:
        r_arr = np.asarray(received)
        if r_arr.dtype.kind == "f":
            hard = (cast(npt.NDArray[np.float64], r_arr) < 0).astype(np.uint8)
            erase_pos = self._erasures_from_llrs(cast(npt.NDArray[np.float64], r_arr))
        else:
            hard = cast(npt.NDArray[np.uint8], r_arr).astype(np.uint8)
            erase_pos = []
        usable = (hard.size // 8) * 8
        syms = _bits_to_symbols(hard[:usable])
        msg_syms, ok = self._decode_symbols(syms, erase_pos)
        bits = _symbols_to_bits(msg_syms) if msg_syms else np.zeros(0, dtype=np.uint8)
        return DecodeResult(
            bits=bits, meta={"decode_ok": ok, "n_erasures": len(erase_pos)}
        )


class _BCH:
    def __init__(self, spec: CodingSpec) -> None:
        self.spec = spec
        self.m = int(spec.params["gf_m"])  # type: ignore[call-overload]
        self.t = int(spec.params["t"])  # type: ignore[call-overload]
        prim = int(spec.params.get("prim_poly", 0x11D))  # type: ignore[arg-type]
        self.field = GF256 if (self.m == 8 and prim == 0x11D) else GF2m(self.m, prim)
        self.nsym = 2 * self.t
        self.gen = _bch_generator_poly(self.field, self.t)
        self.parity_len = len(self.gen) - 1

    def encode(self, info_bits: Bits) -> Bits:
        info = [int(b) for b in np.asarray(info_bits, dtype=np.uint8)]
        _, parity = self.field.poly_div(info + [0] * self.parity_len, self.gen)
        out = np.array(info + [int(c) for c in parity], dtype=np.uint8)
        return cast(Bits, out)

    def decode(self, received: SoftOrHard) -> DecodeResult:
        r_arr = np.asarray(received)
        if r_arr.dtype.kind == "f":
            bits = (cast(npt.NDArray[np.float64], r_arr) < 0).astype(np.uint8)
        else:
            bits = cast(npt.NDArray[np.uint8], r_arr).astype(np.uint8)
        n = int(bits.size)
        k = n - self.parity_len
        if k <= 0:
            return DecodeResult(
                bits=np.zeros(0, dtype=np.uint8), meta={"decode_ok": False}
            )
        poly = [int(b) for b in bits]
        synd = [
            self.field.poly_eval(poly, self.field.pow(2, j))
            for j in range(1, self.nsym + 1)
        ]
        if max(synd) == 0:
            return DecodeResult(bits=bits[:k], meta={"decode_ok": True, "n_errors": 0})
        try:
            err_loc = berlekamp_massey(self.field, synd, self.nsym)
            err_pos = chien_search(self.field, err_loc[::-1], n)
            corrected = bits.copy()
            for p in err_pos:
                corrected[p] ^= 1
            check = [
                self.field.poly_eval([int(b) for b in corrected], self.field.pow(2, j))
                for j in range(1, self.nsym + 1)
            ]
            if max(check) != 0:
                return DecodeResult(bits=bits[:k], meta={"decode_ok": False})
            return DecodeResult(
                bits=corrected[:k],
                meta={"decode_ok": True, "n_errors": len(err_pos)},
            )
        except (ValueError, ZeroDivisionError):
            return DecodeResult(bits=bits[:k], meta={"decode_ok": False})


class _LDPC:
    def __init__(self, spec: CodingSpec) -> None:
        self.spec = spec
        self.rate = str(spec.params["rate"])  # type: ignore[index]
        self.seed = int(spec.params.get("seed", 802))  # type: ignore[arg-type]
        self.max_iters = int(spec.params.get("max_iters", 50))  # type: ignore[arg-type]
        self.norm = float(spec.params.get("norm_factor", 0.8))  # type: ignore[arg-type]
        self.code = _ldpc_mod.build_code(self.rate, self.seed)

    def encode(self, info_bits: Bits) -> Bits:
        info = np.asarray(info_bits, dtype=np.uint8)
        k = self.code.k
        if info.size > k:
            raise ValueError("payload exceeds LDPC k")
        padded = np.zeros(k, dtype=np.uint8)
        padded[: info.size] = info
        cw = _ldpc_mod.encode(self.code, padded)
        # shorten: transmit real info + parity (drop the known-zero pad)
        parity = cw[k:]
        out = np.concatenate([info, parity]).astype(np.uint8)
        self._info_len = int(info.size)  # for decode symmetry via meta not needed
        return cast(Bits, out)

    def decode(self, received: SoftOrHard) -> DecodeResult:
        r = np.asarray(received)
        llr_in = (
            r.astype(np.float64)
            if r.dtype.kind == "f"
            else (1.0 - 2.0 * r.astype(np.float64)) * 8.0
        )
        k = self.code.k
        parity_len = self.code.n - k
        info_len = int(llr_in.size) - parity_len
        if info_len <= 0:
            return DecodeResult(
                bits=np.zeros(0, dtype=np.uint8), meta={"decode_ok": False}
            )
        full = np.empty(self.code.n, dtype=np.float64)
        full[:info_len] = llr_in[:info_len]
        full[info_len:k] = 1e6  # known-zero shortened bits: very confident 0
        full[k:] = llr_in[info_len:]
        hard = _ldpc_mod.decode_min_sum(self.code, full, self.max_iters, self.norm)
        bits = hard[:info_len].astype(np.uint8)
        return DecodeResult(bits=cast(Bits, bits), meta={"decode_ok": True})


class _Turbo:
    """LTE-style rate-1/3 (or punctured rate-1/2) turbo codec (P3f).

    Shortening mirrors ``_LDPC``: info is zero-padded to the fixed QPP block
    size ``K``; only the real-info systematic bits are transmitted (the
    known-zero pad is reinserted at decode time as a confident a-priori
    LLR). Layout follows ``core.turbo.turbo_encode``:
    ``[info(K) | tail1_sys(3) | tail2_sys(3) | par1(K+3) | par2(K+3)]``,
    punctured to ``[... | par1[mask1] | par2[mask2]]`` for rate 1/2.
    """

    def __init__(self, spec: CodingSpec) -> None:
        self.spec = spec
        self.K = int(spec.params.get("block_k", 256))  # type: ignore[arg-type]
        f1, f2 = cast(Tuple[int, int], spec.params.get("qpp", (31, 64)))
        self.perm = _turbo_mod.qpp_perm(self.K, int(f1), int(f2))
        self.max_iters = int(spec.params.get("max_iters", 8))  # type: ignore[arg-type]
        self.scale = float(spec.params.get("extrinsic_scale", 0.7))  # type: ignore[arg-type]
        self.punctured = bool(spec.params.get("puncture", False))
        if self.punctured:
            self.mask1, self.mask2 = _turbo_mod.punctured_parity_masks(self.K)
            parity_len = int(self.mask1.sum()) + int(self.mask2.sum())
        else:
            parity_len = 2 * (self.K + 3)
        self.trailer_len = 6 + parity_len  # tail1_sys(3) + tail2_sys(3) + parities

    def encode(self, info_bits: Bits) -> Bits:
        info = np.asarray(info_bits, dtype=np.uint8)
        info_len = int(info.size)
        if info_len > self.K:
            raise ValueError("payload exceeds turbo block_k")
        padded = np.zeros(self.K, dtype=np.uint8)
        padded[:info_len] = info
        coded = (
            _turbo_mod.turbo_encode_punctured(padded, self.perm)
            if self.punctured
            else _turbo_mod.turbo_encode(padded, self.perm)
        )
        # shorten: drop the known-zero padded systematic positions
        real_sys = coded[:info_len]
        trailer = coded[self.K :]
        out = np.concatenate([real_sys, trailer]).astype(np.uint8)
        return cast(Bits, out)

    def decode(self, received: SoftOrHard) -> DecodeResult:
        r = np.asarray(received)
        llr_in = (
            r.astype(np.float64)
            if r.dtype.kind == "f"
            else (1.0 - 2.0 * r.astype(np.float64)) * 8.0
        )
        info_len = int(llr_in.size) - self.trailer_len
        if info_len <= 0:
            return DecodeResult(
                bits=np.zeros(0, dtype=np.uint8), meta={"decode_ok": False}
            )
        ls_info = np.empty(self.K, dtype=np.float64)
        ls_info[:info_len] = llr_in[:info_len]
        ls_info[info_len:] = 1e6  # known-zero shortened bits: very confident 0
        rest = llr_in[info_len:]
        ls_tail1 = rest[:3]
        ls_tail2 = rest[3:6]
        parity = rest[6:]
        if self.punctured:
            n1 = int(self.mask1.sum())
            lp1 = np.zeros(self.K + 3, dtype=np.float64)
            lp1[self.mask1] = parity[:n1]
            lp2 = np.zeros(self.K + 3, dtype=np.float64)
            lp2[self.mask2] = parity[n1:]
        else:
            lp1 = parity[: self.K + 3]
            lp2 = parity[self.K + 3 :]
        hard = _turbo_mod.turbo_decode(
            ls_info, ls_tail1, ls_tail2, lp1, lp2, self.perm, self.max_iters, self.scale
        )
        bits = hard[:info_len].astype(np.uint8)
        return DecodeResult(bits=cast(Bits, bits), meta={"decode_ok": True})


class _Polar:
    """n=256 Arıkan polar codec, CRC-aided SCL (P3g).

    Single-block per frame; shorten-from-the-end conveys the frame length
    (transmit M = (n-K)+L bits, drop the known-0 tail; reinsert +1e6 at
    decode). CA-SCL reuses the framework CRC-16 for list selection.
    """

    def __init__(self, spec: CodingSpec) -> None:
        self.spec = spec
        self.n = int(spec.n)
        self.K = int(spec.k)
        self.list_size = int(spec.params.get("list_size", 8))  # type: ignore[arg-type]
        dsnr = float(spec.params.get("design_snr_db", 2.0))  # type: ignore[arg-type]
        self.code = _polar_mod.build_code(self.n, self.K, dsnr)

    def encode(self, info_bits: Bits) -> Bits:
        info = np.asarray(info_bits, dtype=np.uint8)
        L = int(info.size)
        if L > self.K:
            raise ValueError("payload exceeds polar block k")
        mask = _polar_mod.build_shortened_mask(self.code, L)
        cw = _polar_mod.polar_encode(info, mask)
        s = self.K - L
        out = cw[: self.n - s]  # drop the known-0 tail
        return cast(Bits, out.astype(np.uint8))

    def decode(self, received: SoftOrHard) -> DecodeResult:
        r = np.asarray(received)
        llr_in = (
            r.astype(np.float64)
            if r.dtype.kind == "f"
            else (1.0 - 2.0 * r.astype(np.float64)) * 8.0
        )
        L = int(llr_in.size) - (self.n - self.K)
        if L <= 0 or L > self.K:
            return DecodeResult(
                bits=np.zeros(0, dtype=np.uint8), meta={"decode_ok": False}
            )
        s = self.K - L
        full = np.empty(self.n, dtype=np.float64)
        full[: self.n - s] = llr_in
        full[self.n - s :] = 1e6  # known-0 shortening tail
        mask = _polar_mod.build_shortened_mask(self.code, L)

        def crc_ok(bits: Bits) -> bool:
            if bits.size < 16:
                return False
            return bool(np.array_equal(bits[-16:], crc16_ccitt(bits[:-16])))

        info, passed = _polar_mod.scl_decode(full, mask, self.list_size, crc_ok)
        return DecodeResult(
            bits=cast(Bits, info[:L].astype(np.uint8)), meta={"decode_ok": bool(passed)}
        )


def _bch_min_poly(field: GF2m, i: int) -> List[int]:
    """Minimal polynomial of alpha^i over GF(2): product over the cyclotomic
    coset {i*2^s mod n} of (x - alpha^j). Binary coefficients, highest-first."""
    coset = set()
    j = i % field.n
    while j not in coset:
        coset.add(j)
        j = (j * 2) % field.n
    poly = [1]
    for j in coset:
        poly = field.poly_mul(poly, [1, field.pow(2, j)])
    return poly


def _bch_generator_poly(field: GF2m, t: int) -> List[int]:
    """BCH generator g(x) = lcm of minimal polys of alpha^1 .. alpha^(2t).

    Conjugate roots share a minimal poly, so distinct minimal polys are
    multiplied once each. Binary coefficients, highest-first."""
    seen: List[Tuple[int, ...]] = []
    g = [1]
    for i in range(1, 2 * t + 1):
        mp = _bch_min_poly(field, i)
        key = tuple(mp)
        if key not in seen:
            seen.append(key)
            g = field.poly_mul(g, mp)
    return g


def make_codec(spec: CodingSpec) -> Codec:
    if spec.family == CodeFamily.UNCODED:
        return _Uncoded(spec)
    if spec.family == CodeFamily.REPETITION:
        return _Repetition(spec)
    if spec.family == CodeFamily.CONVOLUTIONAL:
        return _Convolutional(spec)
    if spec.family == CodeFamily.REED_SOLOMON:
        return _ReedSolomon(spec)
    if spec.family == CodeFamily.BCH:
        return _BCH(spec)
    if spec.family == CodeFamily.LDPC:
        return _LDPC(spec)
    if spec.family == CodeFamily.TURBO:
        return _Turbo(spec)
    if spec.family == CodeFamily.POLAR:
        return _Polar(spec)
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
    "bch_63_51": CodingSpec(
        "bch_63_51", CodeFamily.BCH, 51, 63, {"gf_m": 6, "t": 2, "prim_poly": 0x43}
    ),
    "bch_255_239": CodingSpec(
        "bch_255_239", CodeFamily.BCH, 239, 255, {"gf_m": 8, "t": 2, "prim_poly": 0x11D}
    ),
    "bch_255_223": CodingSpec(
        "bch_255_223", CodeFamily.BCH, 223, 255, {"gf_m": 8, "t": 4, "prim_poly": 0x11D}
    ),
    "ldpc_648_r12": CodingSpec(
        "ldpc_648_r12",
        CodeFamily.LDPC,
        324,
        648,
        {
            "soft_input": True,
            "rate": "1/2",
            "max_iters": 50,
            "norm_factor": 0.8,
            "seed": 802,
        },
    ),
    "ldpc_648_r23": CodingSpec(
        "ldpc_648_r23",
        CodeFamily.LDPC,
        432,
        648,
        {
            "soft_input": True,
            "rate": "2/3",
            "max_iters": 50,
            "norm_factor": 0.8,
            "seed": 802,
        },
    ),
    "ldpc_648_r34": CodingSpec(
        "ldpc_648_r34",
        CodeFamily.LDPC,
        486,
        648,
        {
            "soft_input": True,
            "rate": "3/4",
            "max_iters": 50,
            "norm_factor": 0.8,
            "seed": 802,
        },
    ),
    "turbo_r13": CodingSpec(
        "turbo_r13",
        CodeFamily.TURBO,
        256,
        780,
        {
            "constraint_length": 4,
            "generators_octal": (0o13, 0o15),
            "soft_input": True,
            "max_iters": 8,
            "extrinsic_scale": 0.7,
            "block_k": 256,
            "qpp": (31, 64),
        },
    ),
    "turbo_r12": CodingSpec(
        "turbo_r12",
        CodeFamily.TURBO,
        256,
        524,
        {
            "constraint_length": 4,
            "generators_octal": (0o13, 0o15),
            "soft_input": True,
            "max_iters": 8,
            "extrinsic_scale": 0.7,
            "block_k": 256,
            "qpp": (31, 64),
            "puncture": True,
        },
    ),
    "polar_256_128": CodingSpec(
        "polar_256_128",
        CodeFamily.POLAR,
        128,
        256,
        {"soft_input": True, "list_size": 8, "design_snr_db": 2.0, "rate": "1/2"},
    ),
    "polar_256_85": CodingSpec(
        "polar_256_85",
        CodeFamily.POLAR,
        85,
        256,
        {"soft_input": True, "list_size": 8, "design_snr_db": 2.0, "rate": "1/3"},
    ),
    "polar_256_170": CodingSpec(
        "polar_256_170",
        CodeFamily.POLAR,
        170,
        256,
        {"soft_input": True, "list_size": 8, "design_snr_db": 2.0, "rate": "2/3"},
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
