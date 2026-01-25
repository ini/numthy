from __future__ import annotations

import importlib
import importlib.util
import itertools
import os
import pathlib
import random
import sys
import unittest
from fractions import Fraction
from typing import Any, Sequence


def _load_under_test():
    """
    Load numthy module under test.

    Priority:
      1) NUMTHY_PATH (explicit path)
      2) Import by name NUMTHY_MODULE (default "numthy")
      3) Fallback: load ./numthy.py next to this test file if present
    """
    module_name = os.getenv("NUMTHY_MODULE", "numthy")
    path_env = os.getenv("NUMTHY_PATH")

    if path_env:
        path = pathlib.Path(path_env).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"NUMTHY_PATH points to missing file: {path}")
        spec = importlib.util.spec_from_file_location("numthy_under_test", str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["numthy_under_test"] = mod
        spec.loader.exec_module(mod)
        return mod

    # Try regular import
    try:
        return importlib.import_module(module_name)
    except Exception:
        pass

    # Fallback: sibling module_name.py (or numthy.py)
    here = pathlib.Path(__file__).resolve().parent
    candidate = here / f"{module_name}.py"
    if not candidate.exists() and module_name != "numthy":
        candidate = here / "numthy.py"
    if candidate.exists():
        spec = importlib.util.spec_from_file_location(
            "numthy_under_test",
            str(candidate),
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {candidate}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["numthy_under_test"] = mod
        spec.loader.exec_module(mod)
        return mod

    raise ImportError(
        "Could not import module under test. "
        "Set NUMTHY_MODULE (importable name) or NUMTHY_PATH (path to .py)."
    )


UT = _load_under_test()

_REQUIRED = (
    "lll_reduce",
    "bkz_reduce",
    "closest_vector",
    "small_roots",
)


def _missing_api() -> list[str]:
    missing: list[str] = []
    for name in _REQUIRED:
        if not hasattr(UT, name):
            missing.append(name)
    return missing


_MISSING = _missing_api()
if _MISSING:
    raise AssertionError(f"numthy missing required APIs: {_MISSING}")


def _is_zero(v: Sequence[int]) -> bool:
    return all(x == 0 for x in v)


def _norm2(v: Sequence[int]) -> int:
    return sum(x * x for x in v)


def _mat_vec(A: Sequence[Sequence[int]], x: Sequence[int]) -> list[int]:
    out: list[int] = []
    for row in A:
        out.append(sum(a * b for a, b in zip(row, x)))
    return out


def _transpose(A: Sequence[Sequence[int]]) -> list[list[int]]:
    if not A:
        return []
    return [list(col) for col in zip(*A)]

def _poly_eval_int(
    coefficients: dict[tuple[int, ...], int],
    x: tuple[int, ...],
    mod: int,
) -> int:
    total = 0
    for monomial, c in coefficients.items():
        term = c
        for x_i, e in zip(x, monomial):
            term *= pow(x_i, e)
        total += term
    return total % mod

def _brute_force_roots(
    coefficients: dict[tuple[int, ...], int],
    bounds: tuple[int, ...],
    mod: int,
) -> list[tuple[int, ...]]:
    ranges = [range(-b + 1, b) for b in bounds]
    roots = []
    for point in itertools.product(*ranges):
        if _poly_eval_int(coefficients, point, mod) == 0:
            roots.append(point)
    return sorted(roots)


def _det_int(A: Sequence[Sequence[int]]) -> int:
    n = len(A)
    if n == 0:
        return 1
    if any(len(row) != n for row in A):
        raise ValueError("det requires a square matrix")

    M = [[Fraction(x) for x in row] for row in A]
    sign = 1
    for i in range(n):
        pivot = None
        for r in range(i, n):
            if M[r][i] != 0:
                pivot = r
                break
        if pivot is None:
            return 0
        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]
            sign *= -1
        piv = M[i][i]
        for r in range(i + 1, n):
            if M[r][i] == 0:
                continue
            factor = M[r][i] / piv
            for c in range(i, n):
                M[r][c] -= factor * M[i][c]

    det = Fraction(sign)
    for i in range(n):
        det *= M[i][i]
    if det.denominator != 1:
        raise AssertionError(f"det should be integer, got {det!r}")
    return int(det)

def _invert_matrix(A: Sequence[Sequence[int]]) -> list[list[Fraction]]:
    n = len(A)
    if n == 0:
        return []
    if any(len(row) != n for row in A):
        raise ValueError("invert requires a square matrix")
    M = [
        [Fraction(x) for x in row] + [Fraction(1 if i == j else 0) for j in range(n)]
        for i, row in enumerate(A)
    ]
    for i in range(n):
        pivot = None
        for r in range(i, n):
            if M[r][i] != 0:
                pivot = r
                break
        if pivot is None:
            raise ValueError("singular matrix")
        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]
        piv = M[i][i]
        for c in range(2 * n):
            M[i][c] /= piv
        for r in range(n):
            if r == i:
                continue
            factor = M[r][i]
            if factor != 0:
                for c in range(2 * n):
                    M[r][c] -= factor * M[i][c]
    return [row[n:] for row in M]

def _mat_mul_frac_int(
    A: Sequence[Sequence[int]],
    B: Sequence[Sequence[Fraction]],
) -> list[list[Fraction]]:
    if not A:
        return []
    rows, cols = len(A), len(B[0])
    inner = len(B)
    out = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            if A[i][k] == 0:
                continue
            for j in range(cols):
                out[i][j] += Fraction(A[i][k]) * B[k][j]
    return out


def _gso_float(B: Sequence[Sequence[int]]) -> tuple[list[list[float]], list[float]]:
    n = len(B)
    if n == 0:
        return [], []
    dim = len(B[0])
    mu = [[0.0 for _ in range(n)] for _ in range(n)]
    bstar = [[0.0 for _ in range(dim)] for _ in range(n)]
    bstar_sq = [0.0 for _ in range(n)]

    def dot(u: Sequence[float], v: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(u, v))

    for i in range(n):
        v = [float(x) for x in B[i]]
        for j in range(i):
            if bstar_sq[j] == 0.0:
                mu[i][j] = 0.0
                continue
            mu[i][j] = dot(v, bstar[j]) / bstar_sq[j]
            v = [vv - mu[i][j] * bj for vv, bj in zip(v, bstar[j])]
        bstar[i] = v
        bstar_sq[i] = dot(v, v)
        mu[i][i] = 1.0

    return mu, bstar_sq


def _is_lll_reduced(
    B: Sequence[Sequence[int]],
    delta: float,
    *,
    eps: float = 1e-7,
    exact: bool = False,
) -> bool:
    if not B:
        return True
    n = len(B)

    if exact:
        # Use exact arithmetic for ill-conditioned matrices
        mu, _, bstar_sq = UT._gso(B, exact=True)
        delta_frac = Fraction(delta).limit_denominator(1000)
        for i in range(1, n):
            for j in range(i - 1, -1, -1):
                if abs(mu[i][j]) > Fraction(1, 2):
                    return False
            if bstar_sq[i] == 0 or bstar_sq[i - 1] == 0:
                continue
            thresh = (delta_frac - mu[i][i - 1] ** 2) * bstar_sq[i - 1]
            if bstar_sq[i] < thresh:
                return False
        return True

    mu, bstar_sq = _gso_float(B)
    for i in range(1, n):
        for j in range(i - 1, -1, -1):
            if abs(mu[i][j]) > 0.5 + eps:
                return False
        if bstar_sq[i] == 0.0 or bstar_sq[i - 1] == 0.0:
            continue
        thresh = (delta - mu[i][i - 1] ** 2) * bstar_sq[i - 1]
        if bstar_sq[i] + eps < thresh:
            return False
    return True


def _closest_vector_bruteforce(
    B: Sequence[Sequence[int]],
    target: Sequence[int],
    bound: int,
) -> tuple[list[int], int]:
    best: list[int] | None = None
    best_d2: int | None = None
    dim = len(target)
    for coeffs in itertools.product(range(-bound, bound + 1), repeat=len(B)):
        cand = [
            sum(c * row[j] for c, row in zip(coeffs, B))
            for j in range(dim)
        ]
        d2 = sum((a - b) ** 2 for a, b in zip(cand, target))
        if best_d2 is None or d2 < best_d2:
            best_d2, best = d2, cand
    if best is None or best_d2 is None:
        raise AssertionError("bruteforce search produced no candidates")
    return best, best_d2


class TestLLLReduce(unittest.TestCase):

    def test_empty_basis(self) -> None:
        self.assertEqual(UT.lll_reduce([]), [])

    def test_does_not_mutate_input(self) -> None:
        B = [[1, 1], [0, 2]]
        B0 = [row[:] for row in B]
        UT.lll_reduce(B)
        self.assertEqual(B, B0)

    def test_single_vector(self) -> None:
        B = [[3, -4, 5]]
        R = UT.lll_reduce(B)
        self.assertEqual(R, B)

    def test_preserves_determinant_square_basis(self) -> None:
        B = [[4, 1], [1, 3]]
        det0 = abs(_det_int(B))
        R = UT.lll_reduce(B)
        self.assertTrue(_is_lll_reduced(R, 0.99))
        self.assertEqual(abs(_det_int(R)), det0)

    def test_dependent_vectors_produce_zero_vector(self) -> None:
        B = [[1, 2], [2, 4]]
        R = UT.lll_reduce(B)
        self.assertEqual(len(R), 2)
        self.assertTrue(any(_is_zero(row) for row in R))
        nonzero = [row for row in R if not _is_zero(row)]
        self.assertEqual(len(nonzero), 1)
        self.assertEqual(_norm2(nonzero[0]), 5)

    def test_zero_vectors_packed_last(self) -> None:
        B = [
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9],
            [0, 0, 0],
        ]
        R = UT.lll_reduce(B)
        zero_flags = [_is_zero(row) for row in R]
        self.assertTrue(any(zero_flags))
        first_zero = zero_flags.index(True)
        self.assertTrue(all(zero_flags[first_zero:]))

    def test_lattice_parity_preserved(self) -> None:
        # Row lattice where coordinates must be divisible by (2, 4, 6)
        B = [
            [2, 0, 0],
            [0, 4, 0],
            [0, 0, 6],
            [2, 4, 6],
        ]
        R = UT.lll_reduce(B)
        for row in R:
            self.assertEqual(row[0] % 2, 0)
            self.assertEqual(row[1] % 4, 0)
            self.assertEqual(row[2] % 6, 0)

    def test_unimodular_transform_for_square_basis(self) -> None:
        B = [
            [4, 1, 3],
            [2, 5, 1],
            [1, 0, 2],
        ]
        self.assertNotEqual(_det_int(B), 0)
        R = UT.lll_reduce(B)
        inv_B = _invert_matrix(B)
        U = _mat_mul_frac_int(R, inv_B)
        U_int = []
        for row in U:
            row_int = []
            for entry in row:
                self.assertEqual(entry.denominator, 1)
                row_int.append(int(entry))
            U_int.append(row_int)
        self.assertIn(abs(_det_int(U_int)), (1,))
        for u_row, r_row in zip(U_int, R):
            reconstructed = [
                sum(u * B[j][k] for j, u in enumerate(u_row))
                for k in range(len(B[0]))
            ]
            self.assertEqual(reconstructed, r_row)

    def test_rectangular_bases_satisfy_lll_conditions(self) -> None:
        rng = random.Random(20240202)
        for rows, cols in ((3, 5), (5, 3), (4, 6)):
            B = [[rng.randint(-15, 15) for _ in range(cols)] for _ in range(rows)]
            R = UT.lll_reduce(B)
            self.assertTrue(_is_lll_reduced(R, 0.99))
            self.assertEqual(len(R), rows)
            self.assertEqual(len(R[0]), cols)
            for row in R:
                for x in row:
                    self.assertIsInstance(x, int)

    def test_random_bases_satisfy_lll_conditions(self) -> None:
        rng = random.Random(20240101)
        for _ in range(30):
            B = [[rng.randint(-20, 20) for _ in range(4)] for _ in range(4)]
            R = UT.lll_reduce(B)
            self.assertTrue(
                _is_lll_reduced(R, 0.99),
                msg=f"Not LLL-reduced for B={B!r}, R={R!r}",
            )
            for row in R:
                for x in row:
                    self.assertIsInstance(x, int)

    def test_large_integers_trigger_precision_escalation(self) -> None:
        # 512-bit integers overflow float, should escalate to exact arithmetic
        # Exact mode uses delta=0.75 for numerical stability
        rng = random.Random(42)
        n, bits = 6, 512
        bound = 2 ** bits
        B = [[rng.randint(-bound, bound) for _ in range(n)] for _ in range(n)]
        R = UT.lll_reduce(B)
        self.assertTrue(_is_lll_reduced(R, 0.75, exact=True))
        self.assertEqual(abs(_det_int(R)), abs(_det_int(B)))
        for row in R:
            for x in row:
                self.assertIsInstance(x, int)

    def test_ill_conditioned_matrix_triggers_precision_escalation(self) -> None:
        # Rows with vastly different scales cause precision issues
        # Exact mode uses delta=0.75 for numerical stability
        rng = random.Random(123)
        n = 8
        B = []
        for i in range(n):
            scale = 2 ** (i * 10)
            row = [rng.randint(-100, 100) * scale for _ in range(n)]
            B.append(row)
        R = UT.lll_reduce(B)
        self.assertTrue(_is_lll_reduced(R, 0.75, exact=True))
        for row in R:
            for x in row:
                self.assertIsInstance(x, int)

    def test_1024_bit_integers(self) -> None:
        # Even larger integers to stress exact mode
        rng = random.Random(999)
        n, bits = 6, 1024
        bound = 2 ** bits
        B = [[rng.randint(-bound, bound) for _ in range(n)] for _ in range(n)]
        R = UT.lll_reduce(B)
        self.assertTrue(_is_lll_reduced(R, 0.75, exact=True))
        for row in R:
            for x in row:
                self.assertIsInstance(x, int)

    def test_overflowing_float_gso_triggers_exact_fallback(self) -> None:
        # Entries that overflow float dot products should still yield an LLL-reduced basis.
        A = 10 ** 250
        B = [[A, A], [A, A + 1]]
        R = UT.lll_reduce(B)
        self.assertTrue(_is_lll_reduced(R, 0.75, exact=True))

    def test_repeated_perturbation_stays_valid(self) -> None:
        # Repeated re-reduction should not accumulate errors
        rng = random.Random(456)
        n = 8
        B = [[rng.randint(-2**64, 2**64) for _ in range(n)] for _ in range(n)]
        B = UT.lll_reduce(B)
        for _ in range(10):
            # Perturb and re-reduce
            for i in range(n):
                B[i] = [x + rng.randint(-2**32, 2**32) for x in B[i]]
            B = UT.lll_reduce(B)
            self.assertTrue(_is_lll_reduced(B, 0.75))


class TestBKZReduce(unittest.TestCase):
    def test_invalid_args_raises(self) -> None:
        with self.assertRaises(ValueError):
            UT.bkz_reduce([[1, 0], [0, 1]], block_size=1)
        with self.assertRaises(ValueError):
            UT.bkz_reduce([[1, 0], [0, 1]], block_size=0)
        with self.assertRaises(ValueError):
            UT.bkz_reduce([[1, 0], [0, 1]], block_size=-5)

    def test_pruning_enumeration_returns_candidate(self) -> None:
        mu = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        bstar_sq = [4.0, 1.0, 2.0]
        coeffs, norm = UT._enumerate_svp_block(
            mu, bstar_sq, 0, 3, pruning=True, max_nodes=10000)
        if coeffs is None:
            self.fail("expected pruning enumeration to return a candidate")
        self.assertTrue(any(coeffs))
        self.assertLess(norm, bstar_sq[0])

    def test_does_not_mutate_input(self) -> None:
        B = [[14, 0, 25, -46], [11, -19, 45, 1], [3, 35, -28, -4]]
        B0 = [row[:] for row in B]
        UT.bkz_reduce(B, block_size=3)
        self.assertEqual(B, B0)

    def test_block_size_two_matches_lll(self) -> None:
        B = [[4, 1], [1, 3]]
        lll = UT.lll_reduce(B)
        bkz = UT.bkz_reduce(B, block_size=2)
        self.assertEqual(bkz, lll)

    def test_deterministic_output(self) -> None:
        B = [
            [14, 0, 25, -46],
            [11, -19, 45, 1],
            [3, 35, -28, -4],
            [20, 39, 49, 36],
        ]
        R1 = UT.bkz_reduce(B, block_size=3)
        R2 = UT.bkz_reduce(B, block_size=3)
        self.assertEqual(R1, R2)

    def test_bkz_can_improve_first_vector_on_known_instance(self) -> None:
        B = [
            [14, 0, 25, -46],
            [11, -19, 45, 1],
            [3, 35, -28, -4],
            [20, 39, 49, 36],
        ]
        det0 = abs(_det_int(B))

        lll = UT.lll_reduce(B)
        bkz = UT.bkz_reduce(B, block_size=3)

        self.assertTrue(_is_lll_reduced(lll, 0.99))
        self.assertTrue(_is_lll_reduced(bkz, 0.99))
        self.assertLessEqual(_norm2(bkz[0]), _norm2(lll[0]))
        self.assertEqual(abs(_det_int(bkz)), det0)

    def test_block_size_exceeds_basis(self) -> None:
        B = [
            [2, 1, 0],
            [1, 1, 1],
            [0, 2, 3],
        ]
        det0 = abs(_det_int(B))
        R = UT.bkz_reduce(B, block_size=10)
        self.assertTrue(_is_lll_reduced(R, 0.99))
        self.assertEqual(abs(_det_int(R)), det0)

    def test_dependent_basis_produces_zero_row(self) -> None:
        B = [
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9],
        ]
        R = UT.bkz_reduce(B, block_size=3)
        self.assertTrue(any(_is_zero(row) for row in R))
        self.assertTrue(_is_lll_reduced(R, 0.99))

    def test_bkz_preserves_lattice_parity(self) -> None:
        B = [
            [2, 0, 0],
            [0, 4, 0],
            [0, 0, 6],
            [2, 4, 6],
        ]
        R = UT.bkz_reduce(B, block_size=3)
        for row in R:
            self.assertEqual(row[0] % 2, 0)
            self.assertEqual(row[1] % 4, 0)
            self.assertEqual(row[2] % 6, 0)

    def test_randomized_pruning_runs_stay_reduced(self) -> None:
        rng = random.Random(20240505)
        for _ in range(3):
            B = [[rng.randint(-30, 30) for _ in range(4)] for _ in range(4)]
            R = UT.bkz_reduce(B, block_size=3)
            self.assertTrue(_is_lll_reduced(R, 0.99))


class TestClosestVector(unittest.TestCase):
    def test_empty_basis_returns_zero(self) -> None:
        self.assertEqual(UT.closest_vector([], [1, 2, 3]), [0, 0, 0])

    def test_dimension_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            UT.closest_vector([[1, 0], [0, 1]], [1, 2, 3])

    def test_orthogonal_basis_matches_rounding(self) -> None:
        B = [[2, 0], [0, 3]]
        target = [5, 7]
        v = UT.closest_vector(B, target)
        self.assertEqual(v, [6, 6])

        # Verify it is actually nearest among a reasonable coefficient search.
        def dist2(u: Sequence[int], w: Sequence[int]) -> int:
            return sum((a - b) ** 2 for a, b in zip(u, w))

        best = None
        best_d2 = None
        for a in range(-5, 6):
            for b in range(-5, 6):
                cand = [2 * a, 3 * b]
                d2 = dist2(cand, target)
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best = cand
        assert best_d2 is not None
        self.assertEqual(dist2(v, target), best_d2)

    def test_tie_breaks_away_from_zero(self) -> None:
        self.assertEqual(UT.closest_vector([[2]], [1]), [2])
        self.assertEqual(UT.closest_vector([[2]], [-1]), [-2])

    def test_result_is_in_lattice_for_square_basis(self) -> None:
        B = [[3, 1], [1, 2]]
        target = [7, -4]
        v = UT.closest_vector(B, target)

        # Verify v is in the lattice by solving B^T * c = v using Cramer's rule
        # For 2x2: det(B^T) = 3*2 - 1*1 = 5
        # c1 = (v0*2 - v1*1) / 5, c2 = (v0*(-1) + v1*3) / 5
        det = 3 * 2 - 1 * 1  # = 5
        c1_num = v[0] * 2 - v[1] * 1
        c2_num = -v[0] * 1 + v[1] * 3
        self.assertEqual(c1_num % det, 0, "v not in lattice")
        self.assertEqual(c2_num % det, 0, "v not in lattice")
        c1, c2 = c1_num // det, c2_num // det
        self.assertEqual([c1 * 3 + c2 * 1, c1 * 1 + c2 * 2], list(v))

    def test_dependent_basis_still_in_lattice(self) -> None:
        B = [
            [2, 0],
            [4, 0],
            [0, 3],
        ]
        target = [5, 4]
        v = UT.closest_vector(B, target)
        found = False
        for a in range(-5, 6):
            for b in range(-5, 6):
                for c in range(-5, 6):
                    cand = [a * 2 + b * 4, c * 3]
                    if cand == v:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        self.assertTrue(found)

    def test_exact_for_nonorthogonal_case(self) -> None:
        B = [[-3, 2], [-2, -4]]
        target = [8, 7]
        v = UT.closest_vector(B, target)
        best, best_d2 = _closest_vector_bruteforce(B, target, bound=6)
        self.assertEqual(v, best)
        self.assertEqual(sum((a - b) ** 2 for a, b in zip(v, target)), best_d2)

    def test_target_already_in_lattice(self) -> None:
        B = [[3, 1], [1, 2]]
        coeffs = (2, -3)
        target = [
            coeffs[0] * B[0][0] + coeffs[1] * B[1][0],
            coeffs[0] * B[0][1] + coeffs[1] * B[1][1],
        ]
        v = UT.closest_vector(B, target)
        self.assertEqual(v, target)


class TestSmallRoots(unittest.TestCase):
    def test_invalid_args_raise(self) -> None:
        with self.assertRaises(ZeroDivisionError):
            UT.small_roots({(1,): 1}, 0)
        with self.assertRaises(ValueError):
            UT.small_roots({(1,): 1}, 101, epsilon=0.0)
        with self.assertRaises(ValueError):
            UT.small_roots({(1, 0): 1, (0, 1): 1}, 101)
        with self.assertRaises(ValueError):
            UT.small_roots({(1,): 1}, 101, bounds=(5, 6))
        with self.assertRaises(ValueError):
            UT.small_roots({(1, 0): 1, (1,): 1}, 101, bounds=(5, 5))
        with self.assertRaises(ValueError):
            UT.small_roots({(-1,): 1}, 101, bounds=(5,))

    def test_trivial_polynomials_return_empty(self) -> None:
        self.assertEqual(UT.small_roots({}, 101), [])
        self.assertEqual(UT.small_roots({(0,): 5}, 101), [])
        self.assertEqual(UT.small_roots({(0,): 101}, 101), [])

    def test_finds_single_small_root(self) -> None:
        # f(x) = (x - 10)^2 = x^2 - 20x + 100  (mod 1009)
        M = 1009
        x0 = 10
        coeffs = {(2,): 1, (1,): -20, (0,): 100}
        roots = UT.small_roots(coeffs, M)
        self.assertEqual(roots, [(x0,)])

    def test_finds_multiple_small_roots(self) -> None:
        # f(x) = (x - 3)(x - 7) = x^2 - 10x + 21  (mod 1009)
        roots = UT.small_roots({(2,): 1, (1,): -10, (0,): 21}, 1009)
        self.assertEqual(roots, [(3,), (7,)])

    def test_negative_root_and_negative_modulus(self) -> None:
        # f(x) = (x + 17)^2 has an integer root at x = -17.
        M = 1009
        roots = UT.small_roots({(2,): 1, (1,): 34, (0,): 289}, -M)
        self.assertEqual(roots, [(-17,)])

    def test_returns_empty_when_no_small_root_exists(self) -> None:
        # For prime p ≡ 3 (mod 4), x^2 + 1 has no roots mod p.
        roots = UT.small_roots({(2,): 1, (0,): 1}, 1019)
        self.assertEqual(roots, [])

    def test_bounds_exclude_root(self) -> None:
        # Root at x = 10 is outside the bounds, so no solutions.
        coeffs = {(2,): 1, (1,): -20, (0,): 100}
        roots = UT.small_roots(coeffs, 1009, bounds=(5,))
        self.assertEqual(roots, [])

    def test_random_univariate_matches_bruteforce(self) -> None:
        rng = random.Random(20240301)
        mod = 101
        bound = 6
        for _ in range(50):
            degree = rng.randint(1, 4)
            coeffs = {(e,): rng.randint(-50, 50) for e in range(degree + 1)}
            if all(coeffs[(e,)] % mod == 0 for e in range(1, degree + 1)):
                coeffs[(degree,)] = 1
            expected = _brute_force_roots(coeffs, (bound,), mod)
            roots = UT.small_roots(coeffs, mod, bounds=(bound,))
            self.assertEqual(roots, expected)

    def test_random_bivariate_matches_bruteforce(self) -> None:
        rng = random.Random(20240302)
        mod = 103
        bounds = (4, 4)
        monomials = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]
        for _ in range(30):
            coeffs = {m: rng.randint(-20, 20) for m in monomials}
            if all(coeffs[m] % mod == 0 for m in monomials if m != (0, 0)):
                coeffs[(1, 0)] = 1
            expected = _brute_force_roots(coeffs, bounds, mod)
            roots = UT.small_roots(coeffs, mod, bounds=bounds)
            self.assertEqual(roots, expected)

    def test_random_trivariate_matches_bruteforce(self) -> None:
        rng = random.Random(20240303)
        mod = 97
        bounds = (3, 3, 3)
        monomials = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
        ]
        for _ in range(10):
            coeffs = {m: rng.randint(-10, 10) for m in monomials}
            if all(coeffs[m] % mod == 0 for m in monomials if m != (0, 0, 0)):
                coeffs[(1, 0, 0)] = 1
            expected = _brute_force_roots(coeffs, bounds, mod)
            roots = UT.small_roots(coeffs, mod, bounds=bounds)
            self.assertEqual(roots, expected)

    def test_multivariate_small_root_bruteforce(self) -> None:
        # f(x, y) = (x - 2)^2 + (y + 3)^2
        coeffs = {
            (2, 0): 1,
            (0, 2): 1,
            (1, 0): -4,
            (0, 1): 6,
            (0, 0): 13,
        }
        roots = UT.small_roots(coeffs, 211, bounds=(5, 5))
        self.assertEqual(roots, [(2, -3)])

    def test_multivariate_multiple_roots_bruteforce(self) -> None:
        # f(x, y) = (x - 1)(y - 2) has solutions whenever x=1 or y=2.
        coeffs = {
            (1, 1): 1,
            (1, 0): -2,
            (0, 1): -1,
            (0, 0): 2,
        }
        roots = UT.small_roots(coeffs, 101, bounds=(4, 4))
        self.assertEqual(len(roots), 13)
        self.assertIn((1, 2), roots)
        self.assertIn((1, -3), roots)
        self.assertIn((3, 2), roots)

    def test_lattice_path_recovers_roots_with_heuristic_relations(self) -> None:
        coeffs = {
            (0, 0): 95,
            (1, 0): -7,
            (0, 1): 6,
            (2, 0): 7,
            (0, 2): 16,
            (1, 1): -20,
        }
        bounds = (6, 6)
        expected = [(-3, 5), (4, 3)]
        called = {"choose": False, "brute_mod": False}
        orig_choose = UT._choose_jochemsz_may_params
        orig_brute = UT._brute_force_polynomial_system

        def choose_wrapper(*args: Any, **kwargs: Any):
            called["choose"] = True
            return orig_choose(*args, **kwargs)

        def brute_wrapper(
            polynomials: list[dict[tuple[int, ...], int]],
            bounds: tuple[int, ...],
            mod: int | None = None,
            brute_force_limit: int = 1_000_000,
        ):
            if mod is not None:
                called["brute_mod"] = True
                return None
            return orig_brute(
                polynomials, bounds, mod=mod, brute_force_limit=brute_force_limit)

        try:
            UT._choose_jochemsz_may_params = choose_wrapper
            UT._brute_force_polynomial_system = brute_wrapper
            roots = UT.small_roots(coeffs, 101, bounds=bounds, epsilon=0.2)
        finally:
            UT._choose_jochemsz_may_params = orig_choose
            UT._brute_force_polynomial_system = orig_brute

        self.assertTrue(called["choose"])
        self.assertTrue(called["brute_mod"])
        self.assertEqual(sorted(roots), expected)

    def test_multivariate_large_bounds_uses_lattice_path(self) -> None:
        # Bounds chosen so (2B-1)^2 > 1e6 to skip brute force.
        B = 501
        x0, y0 = 123, -77
        coeffs = {
            (2, 0): 1,
            (0, 2): 1,
            (1, 0): -2 * x0,
            (0, 1): -2 * y0,
            (0, 0): x0 * x0 + y0 * y0,
        }
        called = {"choose": False, "brute": False}
        orig_choose = UT._choose_jochemsz_may_params
        orig_brute = UT._brute_force_polynomial_system

        def choose_wrapper(*args: Any, **kwargs: Any):
            called["choose"] = True
            return orig_choose(*args, **kwargs)

        def brute_wrapper(
            polynomials: list[dict[tuple[int, ...], int]],
            bounds: tuple[int, ...],
            mod: int | None = None,
            brute_force_limit: int = 1_000_000,
        ):
            called["brute"] = True
            size = 1
            for bound in bounds:
                size *= 2 * bound - 1
            if bounds == (B, B):
                if size <= brute_force_limit:
                    raise AssertionError("expected lattice path, brute force allowed")
                return None
            return orig_brute(
                polynomials, bounds, mod=mod, brute_force_limit=brute_force_limit)

        try:
            UT._choose_jochemsz_may_params = choose_wrapper
            UT._brute_force_polynomial_system = brute_wrapper
            roots = UT.small_roots(coeffs, 1_000_003, bounds=(B, B), epsilon=0.3)
        finally:
            UT._choose_jochemsz_may_params = orig_choose
            UT._brute_force_polynomial_system = orig_brute

        self.assertTrue(called["brute"])
        self.assertTrue(called["choose"])
        self.assertEqual(roots, [(x0, y0)])

    def test_trivariate_large_bounds_uses_lattice_path(self) -> None:
        # Bounds chosen so (2B-1)^3 > 1e6 to skip brute force.
        B = 55
        x0, y0, z0 = 12, -7, 5
        coeffs = {
            (2, 0, 0): 1,
            (0, 2, 0): 1,
            (0, 0, 2): 1,
            (1, 0, 0): -2 * x0,
            (0, 1, 0): -2 * y0,
            (0, 0, 1): -2 * z0,
            (0, 0, 0): x0 * x0 + y0 * y0 + z0 * z0,
        }
        bounds = (B, B, B)
        called = {"choose": False, "brute": False}
        orig_choose = UT._choose_jochemsz_may_params
        orig_brute = UT._brute_force_polynomial_system

        def choose_wrapper(*args: Any, **kwargs: Any):
            called["choose"] = True
            return orig_choose(*args, **kwargs)

        def brute_wrapper(
            polynomials: list[dict[tuple[int, ...], int]],
            bounds: tuple[int, ...],
            mod: int | None = None,
            brute_force_limit: int = 1_000_000,
        ):
            called["brute"] = True
            size = 1
            for bound in bounds:
                size *= 2 * bound - 1
            if bounds == (B, B, B) and mod is not None:
                if size <= brute_force_limit:
                    raise AssertionError("expected lattice path, brute force allowed")
                return None
            return orig_brute(polynomials, bounds, mod=mod, brute_force_limit=brute_force_limit)

        try:
            UT._choose_jochemsz_may_params = choose_wrapper
            UT._brute_force_polynomial_system = brute_wrapper
            roots = UT.small_roots(coeffs, 1_000_003, bounds=bounds, epsilon=0.45)
        finally:
            UT._choose_jochemsz_may_params = orig_choose
            UT._brute_force_polynomial_system = orig_brute

        self.assertTrue(called["brute"])
        self.assertTrue(called["choose"])
        self.assertEqual(roots, [(x0, y0, z0)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
