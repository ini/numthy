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
from typing import Iterable, Sequence


# ----------------------------- Module loading -----------------------------

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

_REQUIRED = ("solve_linear_system", "solve_polynomial_system")
_missing = [name for name in _REQUIRED if not hasattr(UT, name)]
if _missing:
    raise ImportError(f"Module under test is missing expected API: {_missing}")


# ------------------------ Reference / oracle utilities ------------------------

def mat_vec(A: Sequence[Sequence[int]], x: Sequence[int]) -> list[int]:
    return [sum(a * xi for a, xi in zip(row, x)) for row in A]


def is_solution(A: Sequence[Sequence[int]], x: Sequence[int], b: Sequence[int]) -> bool:
    return all(sum(a * xi for a, xi in zip(row, x)) == bi for row, bi in zip(A, b))


def rank_fraction(A: Sequence[Sequence[int]]) -> int:
    if not A:
        return 0
    m, n = len(A), len(A[0])
    M = [[Fraction(v) for v in row] for row in A]
    rank = 0
    for col in range(n):
        pivot = None
        for r in range(rank, m):
            if M[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != rank:
            M[rank], M[pivot] = M[pivot], M[rank]
        pivot_val = M[rank][col]
        for c in range(col, n):
            M[rank][c] /= pivot_val
        for r in range(m):
            if r == rank:
                continue
            if M[r][col] != 0:
                factor = M[r][col]
                for c in range(col, n):
                    M[r][c] -= factor * M[rank][c]
        rank += 1
        if rank == m:
            break
    return rank


def basis_rank(basis: Sequence[Sequence[int]], n: int) -> int:
    if not basis:
        return 0
    # Treat basis vectors as columns in an n x k matrix.
    k = len(basis)
    M = [[Fraction(basis[j][i]) for j in range(k)] for i in range(n)]
    return rank_fraction(M)


def poly_eval(poly: dict[tuple[int, ...], int], x: tuple[int, ...]) -> int:
    total = 0
    for monomial, c in poly.items():
        term = c
        for xi, e in zip(x, monomial):
            term *= xi ** e
        total += term
    return total


def brute_force_poly_system(
    polynomials: Sequence[dict[tuple[int, ...], int]],
    bounds: tuple[int, ...],
) -> list[tuple[int, ...]]:
    ranges = [range(-b + 1, b) for b in bounds]
    solutions = []
    for point in itertools.product(*ranges):
        if all(poly_eval(f, point) == 0 for f in polynomials):
            solutions.append(point)
    return sorted(solutions)


def assert_solutions_valid(
    tc: unittest.TestCase,
    polynomials: Sequence[dict[tuple[int, ...], int]],
    bounds: tuple[int, ...],
    solutions: Iterable[tuple[int, ...]],
) -> None:
    for sol in solutions:
        tc.assertTrue(all(abs(x) < b for x, b in zip(sol, bounds)))
        for f in polynomials:
            tc.assertEqual(poly_eval(f, sol), 0)


# ------------------------------- Test cases --------------------------------

class TestLinearSolveValidation(unittest.TestCase):
    def test_empty_matrix_b_none(self):
        x, basis = UT.solve_linear_system([], None)
        self.assertEqual(x, [])
        self.assertIsNone(basis)

    def test_empty_matrix_b_nonzero(self):
        x, basis = UT.solve_linear_system([], [1])
        self.assertIsNone(x)
        self.assertIsNone(basis)

    def test_ragged_matrix_raises(self):
        with self.assertRaises(ValueError):
            UT.solve_linear_system([[1, 2], [3]], [1, 2])

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            UT.solve_linear_system([[1, 2], [3, 4]], [1])

    def test_row_gcd_infeasible(self):
        A = [[2, 4], [6, 8]]
        b = [3, 2]
        x, basis = UT.solve_linear_system(A, b)
        self.assertIsNone(x)
        self.assertIsNone(basis)

    def test_row_gcd_infeasible_with_nullspace(self):
        A = [[2, 4], [6, 8]]
        b = [3, 2]
        x, basis = UT.solve_linear_system(A, b, nullspace=True)
        self.assertIsNone(x)
        self.assertEqual(basis, [])


class TestLinearSolveSolutions(unittest.TestCase):
    def test_square_unique_solution(self):
        A = [[2, 3], [1, 2]]
        b = [5, 3]
        x, basis = UT.solve_linear_system(A, b)
        self.assertEqual(x, [1, 1])
        self.assertIsNone(basis)

    def test_square_no_integer_solution_due_gcd(self):
        A = [[2, 0], [0, 2]]
        b = [1, 1]
        x, basis = UT.solve_linear_system(A, b)
        self.assertIsNone(x)
        self.assertIsNone(basis)

    def test_overdetermined_consistent(self):
        A = [[1, 0], [0, 1], [1, 1]]
        b = [2, 3, 5]
        x, basis = UT.solve_linear_system(A, b)
        self.assertIsNotNone(x)
        self.assertIsNone(basis)
        self.assertTrue(is_solution(A, x, b))

    def test_overdetermined_inconsistent(self):
        A = [[1, 0], [0, 1], [1, 1]]
        b = [2, 3, 6]  # inconsistent with x=2,y=3
        x, basis = UT.solve_linear_system(A, b)
        self.assertIsNone(x)
        self.assertIsNone(basis)

    def test_inconsistent_system_with_row_gcd_ok(self):
        A = [[1, 1], [2, 2]]
        b = [1, 0]  # second equation contradicts first
        x, basis = UT.solve_linear_system(A, b)
        self.assertIsNone(x)
        self.assertIsNone(basis)

    def test_underdetermined_has_solution(self):
        A = [[1, 1, 1]]
        b = [0]
        x, basis = UT.solve_linear_system(A, b)
        self.assertIsNotNone(x)
        self.assertTrue(is_solution(A, x, b))

    def test_b_none_defaults_to_zero(self):
        A = [[2, 0], [0, 3]]
        x, basis = UT.solve_linear_system(A, None)
        self.assertEqual(x, [0, 0])
        self.assertIsNone(basis)


class TestLinearSolveNullspace(unittest.TestCase):
    def test_nullspace_basis_homogeneous(self):
        A = [[1, 2, 3], [2, 4, 6]]
        b = [0, 0]
        x, basis = UT.solve_linear_system(A, b, nullspace=True)
        self.assertIsNotNone(x)
        self.assertTrue(is_solution(A, x, b))
        self.assertIsInstance(basis, list)
        self.assertEqual(len(basis), 2)
        for v in basis:
            self.assertTrue(is_solution(A, v, b))
        # Basis vectors should be independent
        self.assertEqual(basis_rank(basis, 3), len(basis))

    def test_nullspace_basis_inhomogeneous(self):
        A = [[1, 1, 1]]
        b = [1]
        x, basis = UT.solve_linear_system(A, b, nullspace=True)
        self.assertIsNotNone(x)
        self.assertTrue(is_solution(A, x, b))
        self.assertIsInstance(basis, list)
        self.assertEqual(len(basis), 2)
        for v in basis:
            self.assertTrue(is_solution(A, v, [0]))
        self.assertEqual(basis_rank(basis, 3), len(basis))
        # Any solution plus nullspace vector remains a solution
        for c1 in range(-2, 3):
            for c2 in range(-2, 3):
                combo = [x_i + c1 * basis[0][i] + c2 * basis[1][i] for i, x_i in enumerate(x)]
                self.assertTrue(is_solution(A, combo, b))

    def test_nullspace_dimension_matches_rank(self):
        A = [[3, 6, 9, 12], [1, 2, 3, 4]]
        b = [0, 0]
        x, basis = UT.solve_linear_system(A, b, nullspace=True)
        self.assertIsNotNone(x)
        self.assertTrue(is_solution(A, x, b))
        rank = rank_fraction(A)
        self.assertEqual(len(basis), 4 - rank)

    def test_zero_matrix_nullspace(self):
        A = [[0, 0, 0], [0, 0, 0]]
        b = [0, 0]
        x, basis = UT.solve_linear_system(A, b, nullspace=True)
        self.assertEqual(x, [0, 0, 0])
        self.assertIsInstance(basis, list)
        self.assertEqual(len(basis), 3)
        self.assertEqual(basis_rank(basis, 3), 3)


class TestLinearSolveRandomized(unittest.TestCase):
    def test_random_systems_with_known_solution(self):
        rng = random.Random(0xA51F4E)
        for _ in range(60):
            m = rng.randint(1, 3)
            n = rng.randint(1, 3)
            A = [[rng.randint(-3, 3) for _ in range(n)] for __ in range(m)]
            x0 = [rng.randint(-3, 3) for _ in range(n)]
            b = mat_vec(A, x0)
            x, basis = UT.solve_linear_system(A, b)
            self.assertIsNotNone(x)
            self.assertTrue(is_solution(A, x, b))
            if any(b):
                self.assertIsNone(basis)
            else:
                # Implementation may return a nullspace basis even when nullspace=False.
                if basis is not None:
                    for v in basis:
                        self.assertTrue(is_solution(A, v, [0] * m))


class TestPolynomialSolveBasic(unittest.TestCase):
    def test_empty_polynomials_returns_all_points(self):
        bounds = (2, 2)
        expected = sorted(itertools.product(range(-1, 2), repeat=2))
        got = UT.solve_polynomial_system([], bounds)
        self.assertEqual(got, tuple(expected))

    def test_zero_polynomial_is_ignored(self):
        bounds = (2, 2)
        expected = sorted(itertools.product(range(-1, 2), repeat=2))
        got = UT.solve_polynomial_system([{}], bounds)
        self.assertEqual(got, tuple(expected))

    def test_constant_nonzero_no_solution(self):
        bounds = (3, 3)
        polynomials = [{(0, 0): 1}]
        got = UT.solve_polynomial_system(polynomials, bounds)
        self.assertEqual(got, ())

    def test_linear_unique_solution(self):
        # x + y = 0 and x - y - 2 = 0 => (1, -1)
        polynomials = [
            {(1, 0): 1, (0, 1): 1},
            {(1, 0): 1, (0, 1): -1, (0, 0): -2},
        ]
        bounds = (5, 5)
        got = UT.solve_polynomial_system(polynomials, bounds)
        self.assertEqual(got, ((1, -1),))

    def test_multiple_solutions_circle(self):
        # x^2 + y^2 = 5 within bounds 3
        polynomials = [{(2, 0): 1, (0, 2): 1, (0, 0): -5}]
        bounds = (3, 3)
        expected = sorted(
            {
                (1, 2), (1, -2), (-1, 2), (-1, -2),
                (2, 1), (2, -1), (-2, 1), (-2, -1),
            }
        )
        got = UT.solve_polynomial_system(polynomials, bounds)
        self.assertEqual(got, tuple(expected))

    def test_univariate_no_solution(self):
        polynomials = [{(2,): 1, (0,): 1}]  # x^2 + 1 = 0
        bounds = (5,)
        got = UT.solve_polynomial_system(polynomials, bounds)
        self.assertEqual(got, ())

    def test_conflicting_univariate_system(self):
        polynomials = [
            {(1,): 1},                # x = 0
            {(1,): 1, (0,): -1},      # x - 1 = 0
        ]
        bounds = (5,)
        got = UT.solve_polynomial_system(polynomials, bounds)
        self.assertEqual(got, ())


class TestPolynomialSolveAdvanced(unittest.TestCase):
    def test_non_bruteforce_univariate_path(self):
        # Bounds are large enough to skip brute-force system search.
        # x^2 - 1 = 0 and y - x = 0 => (-1,-1), (1,1)
        polynomials = [
            {(2, 0): 1, (0, 0): -1},
            {(0, 1): 1, (1, 0): -1},
        ]
        bounds = (501, 501)
        got = UT.solve_polynomial_system(polynomials, bounds)
        self.assertEqual(got, ((-1, -1), (1, 1)))

    def test_grobner_path_no_univariate_in_input(self):
        # No univariate polynomial in input, but Grobner basis yields x and y.
        polynomials = [
            {(1, 0): 1, (0, 1): 1},   # x + y
            {(1, 0): 1, (0, 1): -1},  # x - y
        ]
        bounds = (501, 501)
        got = UT.solve_polynomial_system(polynomials, bounds)
        self.assertEqual(got, ((0, 0),))

    def test_backtrack_without_univariate(self):
        # x*y = 0 yields solutions with x=0 or y=0 (no univariate polynomial).
        polynomials = [{(1, 1): 1}]
        bounds = (4, 4)
        got = UT.solve_polynomial_system(polynomials, bounds)
        expected = brute_force_poly_system(polynomials, bounds)
        self.assertEqual(got, tuple(expected))

    def test_variable_permutation_with_bounds(self):
        # Bounds force permutation (second variable has smaller bound).
        polynomials = [
            {(1, 0): 1, (0, 0): -7},  # x - 7
            {(0, 1): 1, (0, 0): -1},  # y - 1
        ]
        bounds = (10, 2)
        got = UT.solve_polynomial_system(polynomials, bounds)
        self.assertEqual(got, ((7, 1),))

    def test_three_variable_system(self):
        # x=y=z=1 solution
        polynomials = [
            {(1, 0, 0): 1, (0, 1, 0): -1},  # x - y
            {(0, 1, 0): 1, (0, 0, 1): -1},  # y - z
            {(1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): -3},
        ]
        bounds = (3, 3, 3)
        got = UT.solve_polynomial_system(polynomials, bounds)
        self.assertEqual(got, ((1, 1, 1),))

    def test_large_bound_hensel_root_path(self):
        # Large bounds trigger non-bruteforce root finding in univariate step.
        polynomials = [
            {(2, 0): 1, (0, 0): -1},   # x^2 - 1 = 0
            {(0, 1): 1, (1, 0): -1},   # y - x = 0
        ]
        bounds = (40001, 40001)
        got = UT.solve_polynomial_system(polynomials, bounds)
        self.assertEqual(got, ((-1, -1), (1, 1)))


class TestPolynomialSolveEdgeCases(unittest.TestCase):
    def test_empty_bounds(self):
        got = UT.solve_polynomial_system([], ())
        self.assertEqual(got, ((),))


class TestPolynomialSolveRandomized(unittest.TestCase):
    def test_random_small_systems_match_bruteforce(self):
        rng = random.Random(0x5EED)
        for _ in range(25):
            num_vars = rng.choice([1, 2, 3])
            bounds = tuple(rng.randint(2, 4) for _ in range(num_vars))
            num_polys = rng.randint(1, 3)
            polynomials = []
            for __ in range(num_polys):
                degree = rng.randint(1, 3)
                terms = rng.randint(1, 4)
                poly: dict[tuple[int, ...], int] = {}
                for _t in range(terms):
                    exponents = tuple(rng.randint(0, degree) for _ in range(num_vars))
                    coeff = rng.randint(-3, 3)
                    if coeff == 0:
                        continue
                    poly[exponents] = poly.get(exponents, 0) + coeff
                polynomials.append(poly)
            expected = brute_force_poly_system(polynomials, bounds)
            got = UT.solve_polynomial_system(polynomials, bounds)
            self.assertEqual(got, tuple(expected))
            assert_solutions_valid(self, polynomials, bounds, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
