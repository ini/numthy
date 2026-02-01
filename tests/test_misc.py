from __future__ import annotations

import importlib
import importlib.util
import itertools
import math
import os
import pathlib
import random
import sys
import unittest
from decimal import Decimal, ROUND_FLOOR, localcontext
from fractions import Fraction
from typing import Iterable


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

_REQUIRED = (
    "integers",
    "integer_pairs",
    "alternating",
    "periodic_continued_fraction",
    "convergents",
    "permutation",
    "polynomial",
    "iroot",
    "ilog",
    "is_square",
    "below",
    "lower_bound",
    "fibonacci",
    "fibonacci_index",
    "polygonal",
    "polygonal_index",
)
_missing = [name for name in _REQUIRED if not hasattr(UT, name)]
if _missing:
    raise ImportError(f"Module under test is missing expected API: {_missing}")


# ------------------------ Reference / oracle utilities ------------------------


def take(it: Iterable[int], k: int) -> list[int]:
    return list(itertools.islice(it, k))


def collect_pairs_until_cover(
    gen: Iterable[tuple[int, int]],
    expected: set[tuple[int, int]],
    cap: int,
) -> None:
    seen: set[tuple[int, int]] = set()
    got: set[tuple[int, int]] = set()
    for i, item in enumerate(gen):
        if i >= cap:
            break
        if item in seen:
            raise AssertionError(f"Duplicate pair {item} at index {i}")
        seen.add(item)
        if item in expected:
            got.add(item)
            if got == expected:
                break
    if got != expected:
        missing = expected - got
        raise AssertionError(f"Missing {len(missing)} pairs in coverage")



def assert_period_repeats(
    test: unittest.TestCase,
    coefficients: Iterable[int],
    initial_len: int,
    period_len: int,
) -> None:
    take_len = initial_len + 2 * period_len
    seq = list(itertools.islice(coefficients, take_len))
    left = seq[initial_len:initial_len + period_len]
    right = seq[initial_len + period_len:initial_len + 2 * period_len]
    test.assertEqual(left, right)


def ref_cf_quadratic(
    D: int,
    P: int,
    Q: int,
    count: int,
    *,
    precision: int = 120,
) -> list[int]:
    if Q == 0:
        raise ZeroDivisionError("Q must be nonzero")
    if D <= 0:
        raise ValueError("D must be positive")
    if int(math.isqrt(D)) ** 2 == D:
        raise ValueError("D must be non-square")

    with localcontext() as ctx:
        ctx.prec = precision
        sqrt_D = Decimal(D).sqrt()
        x = (Decimal(P) + sqrt_D) / Decimal(Q)
        coeffs: list[int] = []
        for _ in range(count):
            a = int(x.to_integral_value(rounding=ROUND_FLOOR))
            coeffs.append(a)
            x = 1 / (x - Decimal(a))
    return coeffs


# ------------------------------- Test cases --------------------------------


class TestIntegers(unittest.TestCase):
    def test_prefix(self):
        expected = [0, 1, -1, 2, -2, 3, -3]
        self.assertEqual(take(UT.integers(), 7), expected)

    def test_covers_small_range(self):
        bound = 6
        seq = take(UT.integers(), 2 * bound + 1)
        self.assertEqual(set(seq), set(range(-bound, bound + 1)))
        self.assertEqual(len(seq), len(set(seq)))


class TestIntegerPairs(unittest.TestCase):
    def test_prefix(self):
        expected = [
            (0, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 0),
            (-2, 0),
            (-1, 1),
            (-1, -1),
            (0, 2),
            (0, -2),
        ]
        self.assertEqual(take(UT.integer_pairs(), 10), expected)

    def test_no_duplicates_prefix(self):
        pairs = take(UT.integer_pairs(), 200)
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_covers_l1_ball(self):
        bound = 2
        expected = {
            (x, y)
            for x in range(-bound, bound + 1)
            for y in range(-bound, bound + 1)
            if abs(x) + abs(y) <= bound
        }
        cap = 1 + 2 * bound * (bound + 1)
        collect_pairs_until_cover(UT.integer_pairs(), expected, cap)

    def test_covers_bounded_square(self):
        bound = 3
        expected = {
            (x, y)
            for x in range(-bound, bound + 1)
            for y in range(-bound, bound + 1)
        }
        max_r = 2 * bound
        cap = 1 + 2 * max_r * (max_r + 1)
        collect_pairs_until_cover(UT.integer_pairs(), expected, cap)


class TestAlternating(unittest.TestCase):
    def test_interleaves(self):
        a = [1, 2, 3]
        b = [10]
        c = [100, 200]
        expected = [1, 10, 100, 2, 200, 3]
        self.assertEqual(list(UT.alternating(a, b, c)), expected)

    def test_single_iterable(self):
        gen = (i for i in range(3))
        self.assertEqual(list(UT.alternating(gen)), [0, 1, 2])

    def test_infinite_then_finite(self):
        gen = itertools.count()
        out = list(itertools.islice(UT.alternating(gen, [100, 101]), 5))
        self.assertEqual(out, [0, 100, 1, 101, 2])

    def test_ignores_empty(self):
        self.assertEqual(list(UT.alternating([], [1, 2], [])), [1, 2])


class TestPeriodicContinuedFraction(unittest.TestCase):
    def test_invalid_D(self):
        for D in (0, 1, 4, 9, -3):
            with self.subTest(D=D):
                with self.assertRaises(ValueError):
                    UT.periodic_continued_fraction(D)

    def test_zero_Q_raises(self):
        cases = [
            (2, 0, 0),
            (2, 1, 0),
            (3, -1, 0),
            (5, 10, 0),
        ]
        for D, P, Q in cases:
            with self.subTest(D=D, P=P, Q=Q):
                with self.assertRaises(ZeroDivisionError):
                    UT.periodic_continued_fraction(D, P, Q)

    def test_known_sqrt2(self):
        coefficients, initial_len, period_len = UT.periodic_continued_fraction(2)
        self.assertEqual(initial_len, 1)
        self.assertEqual(period_len, 1)
        self.assertEqual(take(coefficients, 6), [1, 2, 2, 2, 2, 2])

    def test_known_sqrt3(self):
        coefficients, initial_len, period_len = UT.periodic_continued_fraction(3)
        self.assertEqual(initial_len, 1)
        self.assertEqual(period_len, 2)
        self.assertEqual(take(coefficients, 6), [1, 1, 2, 1, 2, 1])

    def test_known_sqrt5(self):
        coefficients, initial_len, period_len = UT.periodic_continued_fraction(5)
        self.assertEqual(initial_len, 1)
        self.assertEqual(period_len, 1)
        self.assertEqual(take(coefficients, 6), [2, 4, 4, 4, 4, 4])

    def test_known_sqrt7(self):
        coefficients, initial_len, period_len = UT.periodic_continued_fraction(7)
        self.assertEqual(initial_len, 1)
        self.assertEqual(period_len, 4)
        expected = [2, 1, 1, 1, 4, 1, 1, 1, 4]
        self.assertEqual(take(coefficients, 9), expected)

    def test_known_sqrt13(self):
        coefficients, initial_len, period_len = UT.periodic_continued_fraction(13)
        self.assertEqual(initial_len, 1)
        self.assertEqual(period_len, 5)
        expected = [3, 1, 1, 1, 1, 6, 1, 1]
        self.assertEqual(take(coefficients, 8), expected)

    def test_period_repeats(self):
        coefficients, initial_len, period_len = UT.periodic_continued_fraction(6)
        self.assertEqual(initial_len, 1)
        self.assertEqual(period_len, 2)
        assert_period_repeats(self, coefficients, initial_len, period_len)

    def test_last_term_is_2a0(self):
        for D in (2, 3, 5, 6, 7, 8, 10, 11, 13):
            with self.subTest(D=D):
                coefficients, initial_len, period_len = (
                    UT.periodic_continued_fraction(D)
                )
                seq = take(coefficients, initial_len + period_len)
                a0 = int(math.isqrt(D))
                self.assertEqual(seq[initial_len + period_len - 1], 2 * a0)


class TestPeriodicContinuedFractionNonCanonical(unittest.TestCase):
    def test_matches_decimal_reference(self):
        cases = [
            (3, 0, 2),
            (5, 0, 2),
            (6, 1, 4),
            (10, 0, 3),
            (13, 2, 5),
        ]
        for D, P, Q in cases:
            with self.subTest(D=D, P=P, Q=Q):
                self.assertNotEqual((D - P * P) % Q, 0)
                expected = ref_cf_quadratic(D, P, Q, 8)
                coefficients, initial_len, period_len = (
                    UT.periodic_continued_fraction(D, P, Q)
                )
                got = take(coefficients, 8)
                self.assertEqual(got, expected)
                self.assertGreater(period_len, 0)


class TestConvergents(unittest.TestCase):
    def test_sqrt2(self):
        coefficients = itertools.chain([1], itertools.repeat(2))
        got = list(UT.convergents(coefficients, num=5))
        expected = [(1, 1), (3, 2), (7, 5), (17, 12), (41, 29)]
        self.assertEqual(got, expected)

    def test_finite_coeffs(self):
        coefficients = [3, 4, 12, 4]
        got = list(UT.convergents(coefficients))
        expected = [(3, 1), (13, 4), (159, 49), (649, 200)]
        self.assertEqual(got, expected)

    def test_num_zero(self):
        self.assertEqual(list(UT.convergents([1, 2, 3], num=0)), [])

    def test_num_one(self):
        self.assertEqual(list(UT.convergents([3, 4, 5], num=1)), [(3, 1)])

    def test_determinant_identity(self):
        coefficients = [2, 3, 1, 4, 2]
        convergents = list(UT.convergents(coefficients))
        for k in range(1, len(convergents)):
            a_k, b_k = convergents[k]
            a_prev, b_prev = convergents[k - 1]
            det = a_k * b_prev - a_prev * b_k
            expected = 1 if k % 2 == 1 else -1
            self.assertEqual(det, expected)
        for a, b in convergents:
            self.assertEqual(math.gcd(a, b), 1)


class TestPermutation(unittest.TestCase):
    def test_invalid_n(self):
        for n in (0, -5):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    list(UT.permutation(n))

    def test_trivial(self):
        self.assertEqual(list(UT.permutation(1)), [0])

    def test_bijection_and_determinism(self):
        key = b"\x00" * 32
        n = 64
        perm1 = list(UT.permutation(n, master_key=key))
        perm2 = list(UT.permutation(n, master_key=key))
        self.assertEqual(perm1, perm2)
        self.assertEqual(sorted(perm1), list(range(n)))

    def test_bijection_non_power_of_two(self):
        key = b"\x01" * 32
        n = 10
        perm = list(UT.permutation(n, master_key=key))
        self.assertEqual(len(perm), n)
        self.assertEqual(sorted(perm), list(range(n)))


class TestPolynomial(unittest.TestCase):
    def test_integer_coeffs(self):
        f = UT.polynomial([1, -3, 2])
        self.assertEqual(f(3), 10)
        self.assertEqual(f(0), 1)

    def test_empty_coeffs(self):
        f = UT.polynomial([])
        self.assertEqual(f(0), 0)
        self.assertEqual(f(10), 0)

    def test_float_eval(self):
        f = UT.polynomial([1.5, -0.5])
        self.assertAlmostEqual(f(2.0), 0.5)

    def test_fraction_eval(self):
        f = UT.polynomial([0, 1, 1])
        x = Fraction(1, 2)
        self.assertEqual(f(x), Fraction(3, 4))


class TestIRoot(unittest.TestCase):
    def test_invalid_n(self):
        for n in (0, -1):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    UT.iroot(10, n)

    def test_even_root_negative(self):
        with self.assertRaises(ValueError):
            UT.iroot(-4, 2)

    def test_known_values(self):
        self.assertEqual(UT.iroot(15, 2), 3)
        self.assertEqual(UT.iroot(16, 2), 4)
        self.assertEqual(UT.iroot(26, 3), 2)
        self.assertEqual(UT.iroot(27, 3), 3)
        self.assertEqual(UT.iroot(0, 5), 0)
        self.assertEqual(UT.iroot(7, 1), 7)

    def test_large_powers(self):
        self.assertEqual(UT.iroot(2 ** 50, 5), 2 ** 10)
        self.assertEqual(UT.iroot(10 ** 18, 2), 10 ** 9)
        self.assertEqual(UT.iroot(1, 7), 1)

    def test_negative_odd(self):
        self.assertEqual(UT.iroot(-1, 3), -1)
        self.assertEqual(UT.iroot(-8, 3), -2)
        self.assertEqual(UT.iroot(-9, 3), -3)

    def test_floor_property_random(self):
        rng = random.Random(1234)
        for _ in range(50):
            n = rng.randint(2, 6)
            x = rng.randint(0, 10_000)
            r = UT.iroot(x, n)
            self.assertLessEqual(r**n, x)
            self.assertGreater((r + 1) ** n, x)
        for _ in range(25):
            n = rng.choice([3, 5])
            x = -rng.randint(1, 10_000)
            r = UT.iroot(x, n)
            self.assertLessEqual(r**n, x)
            self.assertGreater((r + 1) ** n, x)


class TestILog(unittest.TestCase):
    def test_invalid(self):
        cases = [(0, 2), (1, 1), (-5, 2)]
        for a, b in cases:
            with self.subTest(a=a, b=b):
                with self.assertRaises(ValueError):
                    UT.ilog(a, b)

    def test_base2(self):
        self.assertEqual(UT.ilog(1, 2), 0)
        self.assertEqual(UT.ilog(8, 2), 3)
        self.assertEqual(UT.ilog(9, 2), 3)

    def test_base10(self):
        self.assertEqual(UT.ilog(999, 10), 2)
        self.assertEqual(UT.ilog(1000, 10), 3)

    def test_large_base2(self):
        self.assertEqual(UT.ilog(2 ** 100, 2), 100)
        self.assertEqual(UT.ilog(2 ** 100 - 1, 2), 99)

    def test_base_gt_a(self):
        self.assertEqual(UT.ilog(3, 5), 0)

    def test_property_random(self):
        rng = random.Random(2023)
        for _ in range(50):
            b = rng.randint(2, 10)
            a = rng.randint(1, 50_000)
            k = UT.ilog(a, b)
            self.assertLessEqual(b**k, a)
            self.assertGreater(b ** (k + 1), a)


class TestIsSquare(unittest.TestCase):
    def test_known_values(self):
        squares = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 10_000]
        nonsquares = [2, 3, 5, 6, 7, 8, 10, 12, 15, 18, 20, 24]
        for n in squares:
            with self.subTest(n=n):
                self.assertTrue(UT.is_square(n))
        for n in nonsquares:
            with self.subTest(n=n):
                self.assertFalse(UT.is_square(n))
        for n in (-1, -4, -9):
            with self.subTest(n=n):
                self.assertFalse(UT.is_square(n))

    def test_false_with_square_nibble(self):
        candidates = [17, 65, 129, 257, 513, 1025]
        for n in candidates:
            with self.subTest(n=n):
                self.assertFalse(UT.is_square(n))

    def test_large_square(self):
        n = 123_456_789
        self.assertTrue(UT.is_square(n * n))
        self.assertFalse(UT.is_square(n * n + 1))

    def test_matches_isqrt(self):
        rng = random.Random(99)
        for _ in range(200):
            n = rng.randint(0, 1_000_000)
            r = int(math.isqrt(n))
            self.assertEqual(UT.is_square(n), r * r == n)


class TestBinarySearch(unittest.TestCase):
    def test_quadratic(self):
        f = lambda n: n * n
        self.assertEqual(UT.lower_bound(f, 20), 5)
        self.assertEqual(UT.lower_bound(f, 16), 4)

    def test_auto_high_linear(self):
        f = lambda n: 3 * n + 1
        self.assertEqual(UT.lower_bound(f, 100), 33)

    def test_with_bounds(self):
        f = lambda n: 2 * n + 3
        self.assertEqual(UT.lower_bound(f, 9, low=0, high=10), 3)
        self.assertEqual(UT.lower_bound(f, 1, low=5, high=10), 5)

    def test_raises_when_high_below_threshold(self):
        f = lambda n: n * n
        with self.assertRaises(ValueError):
            UT.lower_bound(f, 10, low=0, high=2)

    def test_threshold_at_low(self):
        f = lambda n: n * n
        self.assertEqual(UT.lower_bound(f, 100, low=10), 10)

    def test_low_not_zero(self):
        f = lambda n: n * n
        self.assertEqual(UT.lower_bound(f, 150, low=10), 13)

    def test_low_not_zero_auto_high(self):
        f = lambda n: n * n
        self.assertEqual(UT.lower_bound(f, 200, low=7), 15)


class TestBelow(unittest.TestCase):
    def test_squares_below_bound(self):
        f = lambda n: n * n
        self.assertEqual(list(UT.below(f, 20)), [0, 1, 2, 3, 4])

    def test_squares_exact_bound(self):
        f = lambda n: n * n
        self.assertEqual(list(UT.below(f, 16)), [0, 1, 2, 3])

    def test_with_start(self):
        f = lambda n: n * n
        self.assertEqual(list(UT.below(f, 50, start=3)), [3, 4, 5, 6, 7])

    def test_empty_when_start_exceeds(self):
        f = lambda n: n * n
        self.assertEqual(list(UT.below(f, 10, start=5)), [])

    def test_linear_function(self):
        f = lambda n: 2 * n + 1
        self.assertEqual(list(UT.below(f, 10)), [0, 1, 2, 3, 4])

    def test_identity(self):
        f = lambda n: n
        self.assertEqual(list(UT.below(f, 5)), [0, 1, 2, 3, 4])

    def test_identity_with_start(self):
        f = lambda n: n
        self.assertEqual(list(UT.below(f, 8, start=3)), [3, 4, 5, 6, 7])

    def test_zero_bound_empty(self):
        f = lambda n: n
        self.assertEqual(list(UT.below(f, 0)), [])

    def test_yields_n_not_fn(self):
        f = lambda n: n * 100
        result = list(UT.below(f, 500))
        self.assertEqual(result, [0, 1, 2, 3, 4])

    def test_negative_start(self):
        f = lambda n: n
        self.assertEqual(list(UT.below(f, 3, start=-2)), [-2, -1, 0, 1, 2])

    def test_negative_start_with_squares(self):
        f = lambda n: n * n
        # (-3)^2=9 >= 10, so starts empty; (-2)^2=4 < 10, etc.
        self.assertEqual(list(UT.below(f, 10, start=-3)), [-3, -2, -1, 0, 1, 2, 3])

    def test_negative_upper_bound(self):
        f = lambda n: n
        self.assertEqual(list(UT.below(f, -5, start=-10)), [-10, -9, -8, -7, -6])

    def test_negative_bound_empty_from_zero(self):
        f = lambda n: n
        self.assertEqual(list(UT.below(f, -1)), [])

    def test_constant_function_below(self):
        f = lambda n: 5
        result = take(UT.below(f, 10), 100)
        self.assertEqual(result, list(range(100)))

    def test_constant_function_at_bound(self):
        f = lambda n: 5
        self.assertEqual(list(UT.below(f, 5)), [])

    def test_constant_function_above_bound(self):
        f = lambda n: 10
        self.assertEqual(list(UT.below(f, 5)), [])

    def test_function_returns_negative(self):
        f = lambda n: -n
        # f(0)=0 < 1, f(1)=-1 < 1, etc. (all negative values < 1)
        result = take(UT.below(f, 1), 10)
        self.assertEqual(result, list(range(10)))

    def test_function_returns_negative_with_positive_bound(self):
        f = lambda n: -n - 10
        result = take(UT.below(f, 100), 5)
        self.assertEqual(result, [0, 1, 2, 3, 4])

    def test_exact_boundary_excluded(self):
        f = lambda n: n
        result = list(UT.below(f, 5, start=5))
        self.assertEqual(result, [])

    def test_one_below_boundary(self):
        f = lambda n: n
        result = list(UT.below(f, 5, start=4))
        self.assertEqual(result, [4])

    def test_large_start(self):
        f = lambda n: n
        start = 10**9
        result = list(UT.below(f, start + 5, start=start))
        self.assertEqual(result, [start, start + 1, start + 2, start + 3, start + 4])

    def test_large_upper_bound(self):
        f = lambda n: n * n
        bound = 10**18
        start = 10**9 - 3
        result = list(UT.below(f, bound, start=start))
        self.assertEqual(result, [start, start + 1, start + 2])

    def test_non_monotonic_stops_at_first_violation(self):
        seq = [0, 1, 2, 100, 3, 4, 5]
        f = lambda n: seq[n] if n < len(seq) else n
        result = list(UT.below(f, 10))
        self.assertEqual(result, [0, 1, 2])

    def test_single_element(self):
        f = lambda n: n * n
        self.assertEqual(list(UT.below(f, 1)), [0])

    def test_alternating_sign_function(self):
        # f(n) = n for even, -n for odd; odd values are always negative
        # stops when f(n) >= bound, i.e., when even n >= 100
        f = lambda n: n if n % 2 == 0 else -n
        result = take(UT.below(f, 100), 50)
        self.assertEqual(result, list(range(50)))

    def test_is_lazy_generator(self):
        call_count = [0]
        def f(n):
            call_count[0] += 1
            return n
        gen = UT.below(f, 5)
        self.assertEqual(call_count[0], 0)
        next(gen)
        self.assertEqual(call_count[0], 1)
        list(gen)
        self.assertEqual(call_count[0], 6)


class TestFibonacci(unittest.TestCase):
    # Known Fibonacci values for reference
    FIB_SEQUENCE = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

    def test_small_values(self):
        for i, expected in enumerate(self.FIB_SEQUENCE):
            with self.subTest(n=i):
                self.assertEqual(UT.fibonacci(i), expected)

    def test_zero(self):
        self.assertEqual(UT.fibonacci(0), 0)

    def test_one(self):
        self.assertEqual(UT.fibonacci(1), 1)

    def test_negative_indices(self):
        # F(-n) = (-1)^(n+1) * F(n)
        # F(-1) = 1, F(-2) = -1, F(-3) = 2, F(-4) = -3, F(-5) = 5, ...
        expected = [1, -1, 2, -3, 5, -8, 13, -21, 34, -55]
        for i, exp in enumerate(expected, start=1):
            with self.subTest(n=-i):
                self.assertEqual(UT.fibonacci(-i), exp)

    def test_large_values(self):
        # F(100) is a known value
        F100 = 354224848179261915075
        self.assertEqual(UT.fibonacci(100), F100)

    def test_very_large_value(self):
        # F(1000) has 209 digits
        F1000 = UT.fibonacci(1000)
        self.assertEqual(len(str(F1000)), 209)
        # Check it ends with known digits
        self.assertTrue(str(F1000).endswith("8875"))

    def test_modular_arithmetic(self):
        mod = 1000000007
        # F(100) mod 10^9+7
        self.assertEqual(UT.fibonacci(100, mod), 354224848179261915075 % mod)

    def test_modular_large(self):
        mod = 10**9 + 7
        # Just verify it returns something in range
        result = UT.fibonacci(10000, mod)
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, mod)

    def test_recurrence_relation(self):
        # F(n) = F(n-1) + F(n-2)
        for n in range(2, 50):
            with self.subTest(n=n):
                self.assertEqual(
                    UT.fibonacci(n),
                    UT.fibonacci(n - 1) + UT.fibonacci(n - 2)
                )

    def test_binet_to_fast_doubling_boundary(self):
        # Test around n=70 where implementation switches
        for n in range(68, 75):
            with self.subTest(n=n):
                F_n = UT.fibonacci(n)
                F_prev = UT.fibonacci(n - 1)
                F_next = UT.fibonacci(n + 1)
                self.assertEqual(F_n + F_prev, F_next)


class TestFibonacciIndex(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(UT.fibonacci_index(0), 0)

    def test_one(self):
        # F(1) = F(2) = 1, so fibonacci_index(1) should return 2
        self.assertEqual(UT.fibonacci_index(1), 2)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            UT.fibonacci_index(-1)

    def test_exact_fibonacci_numbers(self):
        # For exact Fibonacci numbers, should return their index
        fib_values = [(0, 0), (1, 2), (2, 3), (3, 4), (5, 5), (8, 6),
                      (13, 7), (21, 8), (34, 9), (55, 10), (89, 11)]
        for n, expected_idx in fib_values:
            with self.subTest(n=n):
                self.assertEqual(UT.fibonacci_index(n), expected_idx)

    def test_between_fibonacci_numbers(self):
        # fibonacci_index(n) returns largest i where F(i) <= n
        # F(6) = 8, F(7) = 13, so fibonacci_index(10) = 6
        self.assertEqual(UT.fibonacci_index(10), 6)
        # F(9) = 34, F(10) = 55, so fibonacci_index(50) = 9
        self.assertEqual(UT.fibonacci_index(50), 9)

    def test_consistency_with_fibonacci(self):
        # For any n, F(fibonacci_index(n)) <= n < F(fibonacci_index(n) + 1)
        test_values = [2, 4, 6, 7, 10, 15, 20, 50, 100, 500, 1000]
        for n in test_values:
            with self.subTest(n=n):
                idx = UT.fibonacci_index(n)
                self.assertLessEqual(UT.fibonacci(idx), n)
                self.assertGreater(UT.fibonacci(idx + 1), n)

    def test_large_value(self):
        # F(100) = 354224848179261915075
        F100 = 354224848179261915075
        self.assertEqual(UT.fibonacci_index(F100), 100)
        self.assertEqual(UT.fibonacci_index(F100 - 1), 99)
        self.assertEqual(UT.fibonacci_index(F100 + 1), 100)


class TestPolygonal(unittest.TestCase):
    def test_triangular_numbers(self):
        # P(3, i) = i*(i+1)/2 = 1, 3, 6, 10, 15, 21, ...
        triangular = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
        for i, expected in enumerate(triangular, start=1):
            with self.subTest(s=3, i=i):
                self.assertEqual(UT.polygonal(3, i), expected)

    def test_square_numbers(self):
        # P(4, i) = i^2 = 1, 4, 9, 16, 25, ...
        for i in range(1, 15):
            with self.subTest(s=4, i=i):
                self.assertEqual(UT.polygonal(4, i), i * i)

    def test_pentagonal_numbers(self):
        # P(5, i) = i*(3i-1)/2 = 1, 5, 12, 22, 35, 51, ...
        pentagonal = [1, 5, 12, 22, 35, 51, 70, 92, 117, 145]
        for i, expected in enumerate(pentagonal, start=1):
            with self.subTest(s=5, i=i):
                self.assertEqual(UT.polygonal(5, i), expected)

    def test_hexagonal_numbers(self):
        # P(6, i) = i*(2i-1) = 1, 6, 15, 28, 45, 66, ...
        hexagonal = [1, 6, 15, 28, 45, 66, 91, 120, 153, 190]
        for i, expected in enumerate(hexagonal, start=1):
            with self.subTest(s=6, i=i):
                self.assertEqual(UT.polygonal(6, i), expected)

    def test_zeroth_index(self):
        # P(s, 0) should be 0 for all s
        for s in range(3, 10):
            with self.subTest(s=s):
                self.assertEqual(UT.polygonal(s, 0), 0)

    def test_first_index(self):
        # P(s, 1) = 1 for all s
        for s in range(3, 10):
            with self.subTest(s=s):
                self.assertEqual(UT.polygonal(s, 1), 1)

    def test_formula_consistency(self):
        # P(s, i) = (s-2)*i*(i-1)/2 + i
        for s in range(3, 12):
            for i in range(0, 20):
                with self.subTest(s=s, i=i):
                    expected = (s - 2) * i * (i - 1) // 2 + i
                    self.assertEqual(UT.polygonal(s, i), expected)


class TestPolygonalIndex(unittest.TestCase):
    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            UT.polygonal_index(3, -1)

    def test_invalid_s_raises(self):
        with self.assertRaises(ValueError):
            UT.polygonal_index(1, 10)

    def test_zero(self):
        for s in range(3, 10):
            with self.subTest(s=s):
                self.assertEqual(UT.polygonal_index(s, 0), 0)

    def test_s_equals_2(self):
        # P(2, i) = i, so polygonal_index(2, n) = n
        for n in range(0, 20):
            with self.subTest(n=n):
                self.assertEqual(UT.polygonal_index(2, n), n)

    def test_triangular_exact(self):
        # Triangular numbers: 1, 3, 6, 10, 15, 21, ...
        triangular = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
        for i, t in enumerate(triangular, start=1):
            with self.subTest(n=t):
                self.assertEqual(UT.polygonal_index(3, t), i)

    def test_triangular_between(self):
        # Between triangular numbers
        # T(3) = 6, T(4) = 10, so polygonal_index(3, 7) = 3
        self.assertEqual(UT.polygonal_index(3, 7), 3)
        self.assertEqual(UT.polygonal_index(3, 8), 3)
        self.assertEqual(UT.polygonal_index(3, 9), 3)

    def test_square_exact(self):
        for i in range(1, 15):
            with self.subTest(i=i):
                self.assertEqual(UT.polygonal_index(4, i * i), i)

    def test_square_between(self):
        # Between 9 and 16, index is 3
        for n in range(9, 16):
            with self.subTest(n=n):
                self.assertEqual(UT.polygonal_index(4, n), 3)

    def test_pentagonal_exact(self):
        pentagonal = [1, 5, 12, 22, 35, 51, 70, 92]
        for i, p in enumerate(pentagonal, start=1):
            with self.subTest(n=p):
                self.assertEqual(UT.polygonal_index(5, p), i)

    def test_consistency_with_polygonal(self):
        # For any n, P(s, polygonal_index(s, n)) <= n < P(s, polygonal_index(s, n) + 1)
        for s in range(3, 8):
            for n in [1, 5, 10, 20, 50, 100, 500]:
                with self.subTest(s=s, n=n):
                    idx = UT.polygonal_index(s, n)
                    self.assertLessEqual(UT.polygonal(s, idx), n)
                    self.assertGreater(UT.polygonal(s, idx + 1), n)

    def test_large_values(self):
        # Test with larger values
        n = 10**12
        for s in range(3, 8):
            with self.subTest(s=s, n=n):
                idx = UT.polygonal_index(s, n)
                self.assertLessEqual(UT.polygonal(s, idx), n)
                self.assertGreater(UT.polygonal(s, idx + 1), n)


# ------------------------------- Entry point -------------------------------


if __name__ == "__main__":
    unittest.main(verbosity=2)
