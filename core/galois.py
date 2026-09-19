"""Field-parametric Galois-field GF(2^m) arithmetic and polynomial algebra.

Reusable across FEC codecs (Reed-Solomon P3c, BCH P3d). Polynomials are
``List[int]`` in **highest-degree-first** order (index 0 is the highest power).
Field elements are plain ``int`` in ``[0, 2**m)``; addition is XOR.
"""
from __future__ import annotations

from typing import List, Tuple


class GF2m:
    """GF(2^m) with generator alpha=2 and the given primitive polynomial."""

    def __init__(self, m: int, prim_poly: int) -> None:
        self.m = m
        self.prim = prim_poly
        self.order = 1 << m
        self.n = self.order - 1
        exp = [0] * (2 * self.n)
        log = [0] * self.order
        x = 1
        for i in range(self.n):
            exp[i] = x
            log[x] = i
            x <<= 1
            if x & self.order:
                x ^= prim_poly
        for i in range(self.n, 2 * self.n):
            exp[i] = exp[i - self.n]
        self.exp: List[int] = exp
        self.log: List[int] = log

    def add(self, a: int, b: int) -> int:
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("GF division by zero")
        if a == 0:
            return 0
        return self.exp[(self.log[a] + self.n - self.log[b]) % self.n]

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("GF inverse of zero")
        return self.exp[self.n - self.log[a]]

    def pow(self, a: int, p: int) -> int:
        if a == 0:
            return 0
        return self.exp[(self.log[a] * p) % self.n]

    # --- polynomials (highest-degree-first) ---
    def poly_scale(self, p: List[int], x: int) -> List[int]:
        return [self.mul(c, x) for c in p]

    def poly_add(self, p: List[int], q: List[int]) -> List[int]:
        r = [0] * max(len(p), len(q))
        for i in range(len(p)):
            r[i + len(r) - len(p)] = p[i]
        for i in range(len(q)):
            r[i + len(r) - len(q)] ^= q[i]
        return r

    def poly_mul(self, p: List[int], q: List[int]) -> List[int]:
        r = [0] * (len(p) + len(q) - 1)
        for j in range(len(q)):
            for i in range(len(p)):
                r[i + j] ^= self.mul(p[i], q[j])
        return r

    def poly_eval(self, p: List[int], x: int) -> int:
        y = p[0]
        for c in p[1:]:
            y = self.mul(y, x) ^ c
        return y

    def poly_div(
        self, dividend: List[int], divisor: List[int]
    ) -> Tuple[List[int], List[int]]:
        out = list(dividend)
        for i in range(len(dividend) - (len(divisor) - 1)):
            coef = out[i]
            if coef != 0:
                for j in range(1, len(divisor)):
                    if divisor[j] != 0:
                        out[i + j] ^= self.mul(divisor[j], coef)
        sep = -(len(divisor) - 1)
        return out[:sep], out[sep:]


GF256 = GF2m(8, 0x11D)


def berlekamp_massey(
    field: GF2m,
    synd: List[int],
    nsym: int,
    erase_count: int = 0,
) -> List[int]:
    """Berlekamp-Massey error-locator search (highest-degree-first output).

    ``synd`` is the (Forney-modified) syndrome list; when erasures are present
    they are removed via the Forney syndromes and their known locator is
    combined by the caller afterward (Convention A, matching the canonical
    "Reed-Solomon for coders" reference), so this search finds only the
    error locator over ``nsym - erase_count`` iterations. Raises ``ValueError``
    if the error count exceeds the budget.
    """
    err_loc = [1]
    old_loc = [1]
    synd_shift = len(synd) - nsym if len(synd) > nsym else 0
    for i in range(nsym - erase_count):
        k = i + synd_shift
        delta = synd[k]
        for j in range(1, len(err_loc)):
            delta ^= field.mul(err_loc[-(j + 1)], synd[k - j])
        old_loc = old_loc + [0]
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = field.poly_scale(old_loc, delta)
                old_loc = field.poly_scale(err_loc, field.inv(delta))
                err_loc = new_loc
            err_loc = field.poly_add(err_loc, field.poly_scale(old_loc, delta))
    while len(err_loc) > 1 and err_loc[0] == 0:
        err_loc = err_loc[1:]
    errs = len(err_loc) - 1
    if (errs - erase_count) * 2 + erase_count > nsym:
        raise ValueError("too many errors")
    return err_loc


def chien_search(field: GF2m, err_loc: List[int], nmess: int) -> List[int]:
    """Return error positions (``nmess-1-i`` for each root ``alpha^i``)."""
    errs = len(err_loc) - 1
    err_pos = []
    for i in range(nmess):
        if field.poly_eval(err_loc, field.pow(2, i)) == 0:
            err_pos.append(nmess - 1 - i)
    if len(err_pos) != errs:
        raise ValueError("root count mismatch")
    return err_pos
