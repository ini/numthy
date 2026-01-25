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

from typing import Iterable
from unittest import mock


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

    # Fallback: sibling numthy.py
    here = pathlib.Path(__file__).resolve().parent
    candidate = here / f"{module_name}.py"
    if not candidate.exists() and module_name != "numthy":
        # If a custom name was set but not found, still try numthy.py
        candidate = here / "numthy.py"
    if candidate.exists():
        spec = importlib.util.spec_from_file_location("numthy_under_test", str(candidate))
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

# Sanity: ensure primes section functions exist
_REQUIRED = (
    "is_prime", "next_prime", "random_prime",
    "primes", "count_primes", "sum_primes",
)
_missing = [name for name in _REQUIRED if not hasattr(UT, name)]
if _missing:
    raise ImportError(f"Module under test is missing expected API: {_missing}")


# ------------------------ Reference / oracle utilities ------------------------

# Deterministic Miller–Rabin bases valid for all n < 2^64.
# Widely used set: https://miller-rabin.appspot.com/
_MR64_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)

_SMALL_TRIAL_PRIMES = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61
)


def _mr_decompose(n: int) -> tuple[int, int]:
    """Return (s, d) such that n-1 = 2^s * d with d odd, for odd n >= 3."""
    d = n - 1
    s = 0
    while (d & 1) == 0:
        d >>= 1
        s += 1
    return s, d


def _mr_is_witness(a: int, n: int, s: int, d: int) -> bool:
    """True if 'a' is a Miller–Rabin witness for compositeness of n."""
    x = pow(a % n, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
    return True


def ref_is_prime_64(n: int) -> bool:
    """
    Reference primality oracle for all integers with |n| < 2^64.

    Uses small trial division + deterministic Miller–Rabin with a known base set.
    """
    if n < 0:
        return False
    if n < 2:
        return False
    if n in _SMALL_TRIAL_PRIMES:
        return True
    if (n & 1) == 0:
        return False
    # small trial division
    for p in _SMALL_TRIAL_PRIMES[1:]:
        if n % p == 0:
            return False
    # deterministic MR for < 2^64
    s, d = _mr_decompose(n)
    for a in _MR64_BASES:
        if a % n == 0:
            continue
        if _mr_is_witness(a, n, s, d):
            return False
    return True


def ref_primes_upto(n: int) -> list[int]:
    """Simple sieve of Eratosthenes producing all primes <= n (n >= 0)."""
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    limit = int(math.isqrt(n))
    for p in range(2, limit + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:n + 1:step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


def ref_primes_in_range(low: int, high: int) -> list[int]:
    low = max(low, 2)
    if high < low:
        return []
    primes = ref_primes_upto(high)
    import bisect
    i = bisect.bisect_left(primes, low)
    return primes[i:]


def take(it: Iterable[int], k: int) -> list[int]:
    return list(itertools.islice(it, k))

# ------------------------------- Test cases --------------------------------

class TestIsPrimeKnownValues(unittest.TestCase):
    def test_edge_cases(self):
        self.assertFalse(UT.is_prime(-10))
        self.assertFalse(UT.is_prime(-1))
        self.assertFalse(UT.is_prime(0))
        self.assertFalse(UT.is_prime(1))
        self.assertTrue(UT.is_prime(2))
        self.assertTrue(UT.is_prime(3))
        self.assertFalse(UT.is_prime(4))
        self.assertTrue(UT.is_prime(5))
        self.assertFalse(UT.is_prime(9))
        self.assertFalse(UT.is_prime(21))

    def test_small_primes_and_composites(self):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
        composites = [
            4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 22, 24,
            25, 27, 28, 30, 33, 35, 39, 49, 51, 55, 57,
        ]
        for p in primes:
            with self.subTest(p=p):
                self.assertTrue(UT.is_prime(p))
        for n in composites:
            with self.subTest(n=n):
                self.assertFalse(UT.is_prime(n))

    def test_matches_reference_small_range(self):
        # Exhaustive cross-check for a moderate range.
        for n in range(0, 50000):
            with self.subTest(n=n):
                self.assertEqual(UT.is_prime(n), ref_is_prime_64(n))


class TestIsPrimeSpecialCases(unittest.TestCase):
    def test_mersenne_primes_and_composites(self):
        mersenne_prime_exponents = [2, 3, 5, 7, 13, 17, 19, 31]
        for p in mersenne_prime_exponents:
            M = (1 << p) - 1
            with self.subTest(M=M):
                self.assertTrue(UT.is_prime(M))
                self.assertTrue(ref_is_prime_64(M))

        for p in [11, 23, 29]:
            M = (1 << p) - 1
            with self.subTest(M=M):
                self.assertFalse(UT.is_prime(M))
                self.assertFalse(ref_is_prime_64(M))

    def test_carmichael_numbers_are_composite(self):
        carmichaels = [
            561, 1105, 1729, 2465, 2821, 6601,
            8911, 10585, 15841, 29341, 41041, 46657,
            52633, 62745, 63973, 75361,
        ]
        for n in carmichaels:
            with self.subTest(n=n):
                self.assertFalse(UT.is_prime(n))
                self.assertFalse(ref_is_prime_64(n))

    def test_strong_pseudoprimes_regression(self):
        candidates = [
            2047,        # 23 * 89 (pseudoprime to base 2)
            1373653,     # strong pseudoprime to bases 2,3
            25326001,    # strong pseudoprime to bases 2,3,5
            3215031751,  # strong pseudoprime to bases 2,3,5,7
        ]
        for n in candidates:
            with self.subTest(n=n):
                self.assertFalse(UT.is_prime(n))
                self.assertFalse(ref_is_prime_64(n))


class TestIsPrimeLargeRanges(unittest.TestCase):
    def test_matches_reference_32bit_to_55e12_range(self):
        # Exercises deterministic MR bases for n >= 2^32 and < 55245642489451.
        candidates = [
            4_294_967_297,  # 2^32 + 1 (known composite)
            4_294_967_311,
            10_000_000_019,
        ]
        for n in candidates:
            with self.subTest(n=n):
                self.assertEqual(UT.is_prime(n), ref_is_prime_64(n))

    def test_matches_reference_bpsw_range(self):
        # Exercises Baillie-PSW path for n >= 55245642489451.
        candidates = [
            55_245_642_489_451,  # threshold value
            55_245_642_489_463,
        ]
        for n in candidates:
            with self.subTest(n=n):
                self.assertEqual(UT.is_prime(n), ref_is_prime_64(n))

    def test_large_values_against_reference(self):
        rng = random.Random(0xC0FFEE)
        for _ in range(200):
            n = rng.getrandbits(56) | 1
            n |= 1 << 55
            with self.subTest(n=n):
                self.assertEqual(UT.is_prime(n), ref_is_prime_64(n))


class TestNextPrime(unittest.TestCase):
    def test_small_known(self):
        cases = [
            (-10, 2),
            (0, 2),
            (1, 2),
            (2, 3),
            (3, 5),
            (4, 5),
            (14, 17),
            (17, 19),
            (18, 19),
            (19, 23),
            (20, 23),
        ]
        for n, expected in cases:
            with self.subTest(n=n):
                self.assertEqual(UT.next_prime(n), expected)

    def test_property_prime_and_minimal(self):
        rng = random.Random(12345)
        for _ in range(200):
            n = rng.randrange(-1000, 200000)
            p = UT.next_prime(n)
            with self.subTest(n=n, p=p):
                self.assertGreater(p, n)
                self.assertTrue(ref_is_prime_64(p))
                for k in range(n + 1, p):
                    self.assertFalse(
                        ref_is_prime_64(k),
                        msg=f"Found smaller prime {k} between {n} and {p}",
                    )


class TestPrimesGenerator(unittest.TestCase):
    def test_first_primes_default(self):
        expected_first_25 = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97
        ]
        got = take(UT.primes(), 25)
        self.assertEqual(got, expected_first_25)

    def test_range_basic(self):
        got = list(UT.primes(low=50, high=100))
        self.assertEqual(got, [53, 59, 61, 67, 71, 73, 79, 83, 89, 97])

    def test_range_empty(self):
        self.assertEqual(list(UT.primes(low=100, high=90)), [])
        self.assertEqual(list(UT.primes(low=0, high=1)), [])
        self.assertEqual(list(UT.primes(low=2, high=1)), [])

    def test_count_limit(self):
        got = list(UT.primes(high=1000, count=10))
        self.assertEqual(got, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])

    def test_low_with_count(self):
        got = list(UT.primes(low=100, count=5))
        self.assertEqual(got, [101, 103, 107, 109, 113])

    def test_no_duplicates_strictly_increasing(self):
        got = take(UT.primes(high=20000), 2000)
        self.assertEqual(got, sorted(set(got)))
        self.assertTrue(all(got[i] < got[i + 1] for i in range(len(got) - 1)))
        self.assertTrue(all(ref_is_prime_64(p) for p in got))

    def test_matches_reference_up_to_20000(self):
        expected = ref_primes_upto(20000)
        got = list(UT.primes(high=20000))
        self.assertEqual(got, expected)

    def test_matches_reference_subrange(self):
        expected = ref_primes_in_range(1234, 56789)
        got = list(UT.primes(low=1234, high=56789))
        self.assertEqual(got, expected)


class TestCountPrimes(unittest.TestCase):
    def test_small_known_pi(self):
        known = {
            -10: 0,
            0: 0,
            1: 0,
            2: 1,
            3: 2,
            10: 4,
            100: 25,
            1000: 168,
            10000: 1229,
        }
        for x, pi_x in known.items():
            with self.subTest(x=x):
                self.assertEqual(UT.count_primes(x), pi_x)

    def test_matches_reference_many_small(self):
        primes = ref_primes_upto(20000)
        pi = [0] * 20001
        count = 0
        j = 0
        for x in range(20001):
            while j < len(primes) and primes[j] == x:
                count += 1
                j += 1
            pi[x] = count
        for x in range(0, 20001, 73):
            with self.subTest(x=x):
                self.assertEqual(UT.count_primes(x), pi[x])

    def test_known_large_pi(self):
        self.assertEqual(UT.count_primes(100000), 9592)
        self.assertEqual(UT.count_primes(1000000), 78498)
        self.assertEqual(UT.count_primes(10000000), 664579)


class TestSumPrimes(unittest.TestCase):
    def test_validation_requires_both_or_neither(self):
        with self.assertRaises(ValueError):
            UT.sum_primes(100, f=lambda n: n)  # missing f_prefix_sum
        with self.assertRaises(ValueError):
            UT.sum_primes(100, f_prefix_sum=lambda n: n)  # missing f

    def test_small_known_prime_sums(self):
        known = {
            -10: 0,
            0: 0,
            1: 0,
            2: 2,
            3: 5,
            10: 17,
            100: 1060,
            1000: 76127,
            10000: 5736396,
        }
        for x, s in known.items():
            with self.subTest(x=x):
                self.assertEqual(UT.sum_primes(x), s)

    def test_known_larger_prime_sums(self):
        self.assertEqual(UT.sum_primes(100000), 454396537)
        self.assertEqual(UT.sum_primes(1000000), 37550402023)

    def test_custom_multiplicative_f_constant_one_matches_pi(self):
        f = lambda n: 1
        f_prefix = lambda n: n
        for x in [10, 100, 1000, 10000, 12345]:
            with self.subTest(x=x):
                self.assertEqual(
                    UT.sum_primes(x, f=f, f_prefix_sum=f_prefix), UT.count_primes(x))

    def test_custom_multiplicative_f_square(self):
        f = lambda n: n * n
        f_prefix = lambda n: n * (n + 1) * (2 * n + 1) // 6
        x = 20000
        expected = sum(p * p for p in ref_primes_upto(x))
        got = UT.sum_primes(x, f=f, f_prefix_sum=f_prefix)
        self.assertEqual(got, expected)


class TestCountAndSumPrimesLargeBranches(unittest.TestCase):
    def test_count_primes_high_branch_uses_lmo_params(self):
        with mock.patch.object(UT, "_lmo", return_value=123456) as lmo:
            got = UT.count_primes(1_000_000_001)
            self.assertEqual(got, 123456)
            lmo.assert_called_once()
            args, kwargs = lmo.call_args
            self.assertEqual(args, (1_000_000_001,))
            self.assertEqual(kwargs.get("k"), 15)
            self.assertEqual(kwargs.get("c"), 0.003)

    def test_sum_primes_high_branch_uses_lmo_params(self):
        with mock.patch.object(UT, "_lmo", return_value=987654321) as lmo:
            got = UT.sum_primes(10_000_001)
            self.assertEqual(got, 987654321)
            lmo.assert_called_once()
            args, kwargs = lmo.call_args
            self.assertEqual(args, (10_000_001,))
            self.assertEqual(kwargs.get("k"), 15)
            self.assertEqual(kwargs.get("c"), 0.005)
            self.assertIs(kwargs.get("f"), UT._identity)
            self.assertTrue(callable(kwargs.get("f_prefix_sum")))


class TestRandomPrime(unittest.TestCase):
    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            UT.random_prime(1)
        with self.assertRaises(ValueError):
            UT.random_prime(2, safe=True)
        with self.assertRaises(ValueError):
            UT.random_prime(0, safe=True)

    def test_num_bits_2(self):
        p = UT.random_prime(2)
        self.assertIn(p, (2, 3))
        self.assertTrue(ref_is_prime_64(p))

    def test_bit_length_and_primality(self):
        for bits in [8, 16, 20]:
            for _ in range(10):
                p = UT.random_prime(bits)
                with self.subTest(bits=bits, p=p):
                    self.assertTrue(ref_is_prime_64(p))
                    self.assertEqual(p.bit_length(), bits)
                    self.assertEqual(p & 1, 1)  # odd (except bits=2 case)

    def test_safe_prime_property(self):
        for bits in [3, 8, 12]:
            for _ in range(5):
                q = UT.random_prime(bits, safe=True)
                p = (q - 1) // 2
                with self.subTest(bits=bits, q=q):
                    self.assertTrue(ref_is_prime_64(q))
                    self.assertTrue(ref_is_prime_64(p))
                    self.assertEqual(q, 2 * p + 1)
                    self.assertEqual(q.bit_length(), bits)


class TestAPIStability(unittest.TestCase):
    def test_primes_generator_type(self):
        it = UT.primes()
        self.assertTrue(hasattr(it, "__iter__") and hasattr(it, "__next__"))

    def test_count_and_sum_are_integers(self):
        for x in [0, 10, 1000, 10000]:
            with self.subTest(x=x):
                self.assertIsInstance(UT.count_primes(x), int)
                self.assertIsInstance(UT.sum_primes(x), int)


# ------------------------------- Entry point -------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
