import numpy as np
import pytest

from core.ldpc import MB, NB, Z, build_base, build_code, encode


@pytest.mark.parametrize("rate", ["1/2", "2/3", "3/4"])
def test_base_dims_and_degree_profile(rate: str) -> None:
    B = build_base(rate, seed=802)
    mb = MB[rate]
    kb = NB - mb
    assert B.shape == (mb, NB)
    # info block-columns: weight 3
    for c in range(kb):
        assert int((B[:, c] != -1).sum()) == 3
    # parity block-columns: dual-diagonal (accumulator) — col j has entries at
    # rows j (diag) and j+1 (subdiag) except the last; all shift 0
    for j in range(mb):
        col = kb + j
        rows = np.nonzero(B[:, col] != -1)[0].tolist()
        expected = [j] if j == mb - 1 else [j, j + 1]
        assert rows == expected
        assert all(B[r, col] == 0 for r in rows)


@pytest.mark.parametrize("rate", ["1/2", "2/3", "3/4"])
def test_info_part_is_4cycle_free(rate: str) -> None:
    B = build_base(rate, seed=802)
    mb, nb = B.shape
    kb = NB - mb
    # no two info columns share two rows with equal shift difference mod Z
    for c1 in range(kb):
        for c2 in range(c1 + 1, kb):
            common = [r for r in range(mb) if B[r, c1] != -1 and B[r, c2] != -1]
            diffs = [(int(B[r, c1]) - int(B[r, c2])) % Z for r in common]
            assert len(diffs) == len(set(diffs)), f"4-cycle cols {c1},{c2}"


@pytest.mark.parametrize("rate", ["1/2", "2/3", "3/4"])
def test_build_code_shape_and_determinism(rate: str) -> None:
    code = build_code(rate, seed=802)
    assert code.n == 648 and code.m == MB[rate] * Z
    assert code.k == code.n - code.m
    assert len(code.checks) == code.m and len(code.vars) == code.n
    # determinism: same seed -> identical base matrix
    assert np.array_equal(code.B, build_code(rate, seed=802).B)


@pytest.mark.parametrize("rate", ["1/2", "2/3", "3/4"])
def test_encode_systematic_and_zero_syndrome(rate: str) -> None:
    code = build_code(rate, seed=802)
    rng = np.random.default_rng(1)
    info = rng.integers(0, 2, size=code.k).astype(np.uint8)
    cw = encode(code, info)
    assert cw.size == code.n
    assert np.array_equal(cw[: code.k], info)  # systematic prefix
    # H * cw^T == 0 over GF(2): every check parity is even
    for c in range(code.m):
        assert sum(int(cw[v]) for v in code.checks[c]) % 2 == 0
