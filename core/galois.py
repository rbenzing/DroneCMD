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
