import pytest

from core.galois import GF256, GF2m, berlekamp_massey, chien_search


def test_field_tables_and_inverse() -> None:
    f = GF256
    assert f.n == 255
    # multiplicative inverse: a * a^-1 == 1 for all nonzero a
    for a in range(1, 256):
        assert f.mul(a, f.inv(a)) == 1
    # generator has full order: alpha^255 == 1, alpha^i != 1 for 0<i<255
    assert f.pow(2, 255) == 1
    assert all(f.pow(2, i) != 1 for i in range(1, 255))


def test_mul_div_and_distributive() -> None:
    f = GF256
    assert f.add(0xAB, 0xAB) == 0  # add is XOR (char 2)
    for a, b in [(0, 5), (5, 0), (1, 1), (0x53, 0xCA), (0xFF, 0x02)]:
        if b != 0:
            assert f.mul(f.div(a, b), b) == a
    # distributivity: a*(b+c) == a*b + a*c
    a, b, c = 0x1D, 0x77, 0x9A
    assert f.mul(a, f.add(b, c)) == f.add(f.mul(a, b), f.mul(a, c))


def test_poly_eval_mul_div() -> None:
    f = GF256
    # (x + a1)(x + a2) evaluated at a1 is 0
    p = f.poly_mul([1, 0x02], [1, 0x04])  # highest-first: x^2 + ...
    assert f.poly_eval(p, 0x02) == 0
    assert f.poly_eval(p, 0x04) == 0
    # poly_div: dividend = q*divisor + r
    dividend = [0x40, 0x01, 0x00, 0x05, 0x11]
    divisor = [0x01, 0x0F, 0x36]
    q, r = f.poly_div(dividend, divisor)
    recon = f.poly_add(f.poly_mul(q, divisor), ([0] * (len(divisor) - 1 - len(r)) + r))
    # compare as polynomials (strip nothing; same length as dividend)
    recon = (
        ([0] * (len(dividend) - len(recon)) + recon)
        if len(recon) < len(dividend)
        else recon
    )
    assert recon[-len(dividend) :] == dividend


def test_gf64_smoke_p3d_ready() -> None:
    # Parametric field: GF(2^6), primitive poly x^6 + x + 1 = 0x43.
    f = GF2m(6, 0x43)
    assert f.n == 63
    for a in range(1, 64):
        assert f.mul(a, f.inv(a)) == 1
    assert f.pow(2, 63) == 1


def test_bm_and_chien_locate_known_errors() -> None:
    # Build a codeword-length-20 zero message, inject 2 known symbol errors,
    # recover their positions via syndromes -> BM -> Chien.
    f = GF256
    nsym = 6  # t=3
    fcr = 1
    n = 20
    r = [0] * n
    err_positions = {3, 11}
    r[3] = 0x2A
    r[11] = 0x71
    # syndromes S_j = r(alpha^(j+fcr)), j=0..nsym-1 (no leading pad here)
    synd = [f.poly_eval(r, f.pow(2, j + fcr)) for j in range(nsym)]
    assert any(s != 0 for s in synd)
    err_loc = berlekamp_massey(f, synd, nsym)
    pos = set(chien_search(f, err_loc[::-1], n))
    assert pos == err_positions


def test_bm_raises_on_too_many_errors() -> None:
    f = GF256
    nsym = 2  # t=1, corrects 1 error; inject 2
    fcr = 1
    n = 12
    r = [0] * n
    r[2] = 0x10
    r[7] = 0x40
    synd = [f.poly_eval(r, f.pow(2, j + fcr)) for j in range(nsym)]
    with pytest.raises(ValueError):
        err_loc = berlekamp_massey(f, synd, nsym)
        chien_search(f, err_loc[::-1], n)
