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


# ----------------------------- Module loading -----------------------------


def _load_under_test():
    module_name = os.getenv("NUMTHY_MODULE", "numthy")
    path_env = os.getenv("NUMTHY_PATH")

    if path_env:
        path = pathlib.Path(path_env).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"NUMTHY_PATH points to missing file: {path}"
            )
        spec = importlib.util.spec_from_file_location(
            "numthy_under_test", str(path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["numthy_under_test"] = mod
        spec.loader.exec_module(mod)
        return mod

    try:
        return importlib.import_module(module_name)
    except Exception:
        pass

    here = pathlib.Path(__file__).resolve().parent
    candidate = here / f"{module_name}.py"
    if not candidate.exists() and module_name != "numthy":
        candidate = here / "numthy.py"
    if candidate.exists():
        spec = importlib.util.spec_from_file_location(
            "numthy_under_test", str(candidate)
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
    "bezout",
    "cornacchia",
    "conic",
    "pell",
    "pythagorean_triples",
    "pillai",
)
_missing = [name for name in _REQUIRED if not hasattr(UT, name)]
if _missing:
    raise ImportError(f"Module under test is missing expected API: {_missing}")


# ---------------------------- Reference utilities ----------------------------


def take(it: Iterable, k: int):
    return list(itertools.islice(it, k))


def brute_force_conic(a: int, b: int, c: int, d: int, e: int, f: int, bound: int):
    sols = set()
    for x in range(-bound, bound + 1):
        for y in range(-bound, bound + 1):
            v = a * x * x + b * x * y + c * y * y + d * x + e * y + f
            if v == 0:
                sols.add((x, y))
    return sols


def assert_no_duplicates(items, *, msg: str = ""):
    if len(items) != len(set(items)):
        seen = set()
        dups = []
        for it in items:
            if it in seen:
                dups.append(it)
            seen.add(it)
        extra = f" {msg}" if msg else ""
        raise AssertionError(f"Duplicates found: {dups[:10]}{extra}")


def collect_pairs_until_cover(
    gen: Iterable,
    expected: set,
    bound: int,
    cap: int,
    check,
):
    seen = set()
    got = set()
    for i, item in enumerate(gen):
        if i >= cap:
            break
        if item in seen:
            raise AssertionError(f"Duplicate {item} at index {i}")
        seen.add(item)
        check(item)
        x, y = item
        if abs(x) <= bound and abs(y) <= bound:
            got.add(item)
            if got == expected:
                break
    if got != expected:
        missing = list(expected - got)
        extra = list(got - expected)
        raise AssertionError(
            f"Missing {len(missing)} / Extra {len(extra)}"
        )


# ------------------------------ bezout() tests ------------------------------


class TestBezout(unittest.TestCase):
    def test_trivial_a_b_zero(self):
        sols = take(UT.bezout(0, 0, 0), 200)
        assert_no_duplicates(sols)
        for x, y in sols:
            self.assertEqual(0 * x + 0 * y, 0)

        sols2 = list(UT.bezout(0, 0, 5))
        self.assertEqual(sols2, [])

    def test_matches_bruteforce_bounded(self):
        rng = random.Random(1337)
        for _ in range(40):
            a = rng.randint(-10, 10)
            b = rng.randint(-10, 10)
            c = rng.randint(-20, 20)
            if a == 0 and b == 0:
                continue
            bound = 25
            expected = set()
            for x in range(-bound, bound + 1):
                for y in range(-bound, bound + 1):
                    if a * x + b * y == c:
                        expected.add((x, y))

            def check(pt):
                x, y = pt
                self.assertEqual(a * x + b * y, c)

            cap = max(500, 20 * (len(expected) + 1))
            collect_pairs_until_cover(
                UT.bezout(a, b, c),
                expected,
                bound,
                cap,
                check,
            )


# ---------------------------- cornacchia() tests ----------------------------


class TestCornacchia(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            list(UT.cornacchia(0, 5))
        with self.assertRaises(ValueError):
            list(UT.cornacchia(5, 5))
        with self.assertRaises(ValueError):
            list(UT.cornacchia(2, 0))
        with self.assertRaises(ValueError):
            list(UT.cornacchia(2, 6))  # gcd(d, m) != 1

    def test_matches_bruteforce(self):
        cases = [
            (1, 5),
            (1, 25),
            (2, 17),
            (5, 29),
            (10, 109),
        ]
        for d, m in cases:
            with self.subTest(d=d, m=m):
                expected = set()
                x_max = int(math.isqrt(m))
                y_max = int(math.isqrt(m // d))
                for x in range(1, x_max + 1):
                    for y in range(1, y_max + 1):
                        if x * x + d * y * y == m:
                            expected.add((x, y))

                sols = list(UT.cornacchia(d, m))
                assert_no_duplicates(sols, msg=f"d={d},m={m}")
                for x, y in sols:
                    self.assertGreater(x, 0)
                    self.assertGreater(y, 0)
                    self.assertEqual(x * x + d * y * y, m)
                self.assertEqual(set(sols), expected)


# ------------------------------- pell() tests -------------------------------


class TestPell(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            next(UT.pell(0, 1))
        with self.assertRaises(ValueError):
            next(UT.pell(4, 1))

    def test_bounded_completeness_and_no_dupes(self):
        cases = [
            (2, 1, 500),
            (2, -1, 200),
            (3, 1, 500),
            (5, 4, 500),
        ]
        for D, N, x_bound in cases:
            with self.subTest(D=D, N=N):
                expected = set()
                max_y2 = x_bound * x_bound - N
                if max_y2 > 0:
                    y_max = int(math.isqrt(max_y2 // D))
                    for y in range(1, y_max + 1):
                        x2 = N + D * y * y
                        if x2 > 0:
                            x = int(math.isqrt(x2))
                            if x * x == x2 and x <= x_bound:
                                expected.add((x, y))

                seen = set()
                got = set()
                last_x = 0
                for i, (x, y) in enumerate(UT.pell(D, N)):
                    self.assertNotIn((x, y), seen)
                    seen.add((x, y))
                    self.assertEqual(x * x - D * y * y, N)
                    self.assertGreater(x, 0)
                    self.assertGreater(y, 0)
                    self.assertGreaterEqual(x, last_x)
                    last_x = x
                    if x <= x_bound:
                        got.add((x, y))
                    if x > x_bound and got == expected:
                        break
                    if i >= 20000:
                        break

                self.assertEqual(got, expected)


# ------------------------------- conic() tests ------------------------------


class TestConicAgainstBruteforce(unittest.TestCase):
    def _check_eq(self, a, b, c, d, e, f):
        def check(pt):
            x, y = pt
            v = a * x * x + b * x * y + c * y * y + d * x + e * y + f
            self.assertEqual(v, 0)

        return check

    def test_matches_pytest_example_param_cases(self):
        cases = [
            (1, 0, 1, 0, 0, -1),
            (1, 0, 1, 0, 0, -25),
            (1, 0, 1, -2, -4, 4),
            (1, 1, 1, 0, 0, -1),
            (2, 1, 3, -1, 2, -5),
            (1, 0, -2, 0, 0, -1),
            (1, 0, -2, 0, 0, 1),
            (1, 0, -3, 0, 0, -1),
            (1, 2, -1, 0, 0, -1),
            (1, 0, 0, 0, -1, 0),
            (1, 2, 1, 0, -1, 0),
            (1, 0, 0, -2, -1, 1),
            (1, 0, -1, 0, 0, 0),
            (0, 1, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0),
            (1, 0, -1, 1, -1, 0),
            (0, 0, 0, 1, 1, -1),
            (0, 0, 0, 2, 3, -5),
        ]

        for coeffs in cases:
            a, b, c, d, e, f = coeffs
            with self.subTest(coeffs=coeffs):
                bound = 50
                expected = brute_force_conic(a, b, c, d, e, f, bound)
                cap = max(5000, 50 * (len(expected) + 1))
                collect_pairs_until_cover(
                    UT.conic(a, b, c, d, e, f),
                    expected,
                    bound,
                    cap,
                    self._check_eq(a, b, c, d, e, f),
                )

    def test_all_zero_nonzero_coefficient_combinations(self):
        for bits in range(64):
            coeffs = [0 if (bits >> i) & 1 else 1 for i in range(6)]
            a, b, c, d, e, f = coeffs
            with self.subTest(bits=bits, coeffs=coeffs):
                if a == b == c == d == e == f == 0:
                    sols = take(UT.conic(0, 0, 0, 0, 0, 0), 80)
                    assert_no_duplicates(sols)
                    self.assertEqual(len(sols), 80)
                    continue

                bound = 20
                expected = brute_force_conic(a, b, c, d, e, f, bound)
                cap = max(3000, 50 * (len(expected) + 1))
                collect_pairs_until_cover(
                    UT.conic(a, b, c, d, e, f),
                    expected,
                    bound,
                    cap,
                    self._check_eq(a, b, c, d, e, f),
                )

    def test_specific_solutions_from_pytest_example(self):
        sols = take(UT.conic(1, 0, -2, 0, 0, -1), 60)
        assert_no_duplicates(sols)
        self.assertIn((1, 0), sols)
        self.assertIn((-1, 0), sols)
        self.assertTrue((3, 2) in sols or (-3, 2) in sols)

        sols = take(UT.conic(1, 0, -2, 0, 0, 1), 40)
        assert_no_duplicates(sols)
        self.assertTrue((1, 1) in sols or (-1, 1) in sols)

        sols = list(UT.conic(1, 0, 1, 0, 0, -25))
        assert_no_duplicates(sols)
        expected = {
            (0, 5),
            (0, -5),
            (5, 0),
            (-5, 0),
            (3, 4),
            (3, -4),
            (-3, 4),
            (-3, -4),
            (4, 3),
            (4, -3),
            (-4, 3),
            (-4, -3),
        }
        self.assertEqual(set(sols), expected)

        sols = take(UT.conic(1, 0, -1, 0, 2, -1), 120)
        assert_no_duplicates(sols)
        for pt in [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 3),
            (-1, 0),
            (-1, 2),
            (-2, -1),
        ]:
            self.assertIn(pt, sols)

        sols = take(UT.conic(1, 1, -1, 0, 5, -6), 200)
        assert_no_duplicates(sols)
        self.assertIn((0, 2), sols)
        self.assertIn((-2, 2), sols)

        sols = take(UT.conic(1, 0, -4, -2, 0, 1), 200)
        assert_no_duplicates(sols)
        self.assertIn((3, 1), sols)
        self.assertIn((5, 2), sols)
        self.assertIn((-1, 1), sols)
        self.assertIn((-3, 2), sols)

    def test_no_solution_cases_from_pytest_example(self):
        cases = [
            (1, 0, 1, 0, 0, 1),
            (0, 0, 0, 0, 0, 1),
            (1, 0, 1, 0, 0, -3),
            (4, 0, 4, 0, 0, -1),
        ]
        for coeffs in cases:
            a, b, c, d, e, f = coeffs
            with self.subTest(coeffs=coeffs):
                first = take(UT.conic(a, b, c, d, e, f), 1)
                self.assertEqual(first, [])

    def test_random_coefficients_pytest_example(self):
        for seed in range(100):
            rng = random.Random(seed)
            a, b, c, d, e, f = [rng.randint(-5, 5) for _ in range(6)]
            bound = 30
            expected = brute_force_conic(a, b, c, d, e, f, bound)

            cap = 2000
            seen = set()
            got = set()
            for i, (x, y) in enumerate(UT.conic(a, b, c, d, e, f)):
                if i >= cap:
                    break
                if (x, y) in seen:
                    raise AssertionError(f"Duplicate ({x}, {y})")
                seen.add((x, y))
                v = a * x * x + b * x * y + c * y * y + d * x + e * y + f
                self.assertEqual(v, 0)
                if abs(x) <= bound and abs(y) <= bound:
                    got.add((x, y))

            missing = expected - got
            if missing:
                self.fail(f"Missing {len(missing)} solutions")


# ------------------------ pythagorean_triples() tests ------------------------


class TestPythagoreanTriples(unittest.TestCase):
    def test_unbounded_prefix_invariants(self):
        triples = take(UT.pythagorean_triples(), 300)
        assert_no_duplicates(triples)
        last_c = 0
        for a, b, c in triples:
            self.assertLessEqual(a, b)
            self.assertLessEqual(b, c)
            self.assertEqual(a * a + b * b, c * c)
            self.assertGreaterEqual(c, last_c)
            last_c = c

    def test_bounded_modes_no_dupes_and_completeness(self):
        def ref_set(max_c=None, max_sum=None):
            if max_c is None and max_sum is None:
                raise ValueError
            out = set()
            if max_c is None:
                max_c = max_sum
            max_m = int(math.isqrt(max_c)) + 2
            for m in range(2, max_m + 1):
                for n in range(1, m):
                    if (m + n) % 2 == 0:
                        continue
                    if math.gcd(m, n) != 1:
                        continue
                    a = m * m - n * n
                    b = 2 * m * n
                    c = m * m + n * n
                    if a > b:
                        a, b = b, a
                    if c > max_c:
                        continue
                    k_max = max_c // c
                    if max_sum is not None:
                        k_max = min(k_max, max_sum // (a + b + c))
                    for k in range(1, k_max + 1):
                        aa, bb, cc = k * a, k * b, k * c
                        if cc > max_c:
                            break
                        if max_sum is not None and aa + bb + cc > max_sum:
                            break
                        out.add((aa, bb, cc))
            return out

        params = [
            (50, None),
            (120, None),
            (None, 120),
            (200, 300),
        ]
        for max_c, max_sum in params:
            with self.subTest(max_c=max_c, max_sum=max_sum):
                gen = UT.pythagorean_triples(max_c=max_c, max_sum=max_sum)
                triples = list(gen)
                assert_no_duplicates(triples)
                for a, b, c in triples:
                    self.assertEqual(a * a + b * b, c * c)
                    if max_c is not None:
                        self.assertLessEqual(c, max_c)
                    if max_sum is not None:
                        self.assertLessEqual(a + b + c, max_sum)
                self.assertEqual(set(triples), ref_set(max_c, max_sum))


# ------------------------------- pillai() tests ------------------------------


class TestPillai(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            list(UT.pillai(1, 2, 3))
        with self.assertRaises(ValueError):
            list(UT.pillai(2, 1, 3))

    def test_matches_small_bruteforce_windows(self):
        def brute(a, b, c, x_max, y_max):
            sols = []
            for x in range(1, x_max + 1):
                ax = pow(a, x)
                for y in range(1, y_max + 1):
                    if ax - pow(b, y) == c:
                        sols.append((x, y))
            return sorted(sols)

        cases = [
            (2, 3, 5, 12, 12),
            (3, 2, 1, 12, 12),
            (5, 2, 1, 10, 20),
            (2, 5, -1, 18, 10),
        ]
        for a, b, c, x_max, y_max in cases:
            with self.subTest(a=a, b=b, c=c):
                expected = brute(a, b, c, x_max, y_max)
                got = list(UT.pillai(a, b, c))
                assert_no_duplicates(got)
                self.assertEqual(got, sorted(got))
                for x, y in got:
                    self.assertGreater(x, 0)
                    self.assertGreater(y, 0)
                    self.assertEqual(pow(a, x) - pow(b, y), c)
                self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
