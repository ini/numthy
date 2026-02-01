from __future__ import annotations

import importlib
import importlib.util
import math
import os
import pathlib
import random
import sys
import unittest

from collections import Counter
from typing import Dict, Iterable, List, Tuple
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

_REQUIRED = ("prime_factors", "prime_factorization", "divisors", "perfect_power")
_missing = [name for name in _REQUIRED if not hasattr(UT, name)]
if _missing:
    raise ImportError(f"Module under test is missing expected API: {_missing}")


# ------------------------ Reference / oracle utilities ------------------------

# Deterministic Miller–Rabin bases valid for all n < 2^64.
# Widely used set: https://miller-rabin.appspot.com/
_MR64_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)

_SMALL_TRIAL_PRIMES = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61,
)


def _mr_decompose(n: int) -> Tuple[int, int]:
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
    Reference primality oracle for all integers with 0 <= n < 2^64.

    Uses small trial division + deterministic Miller–Rabin with a known base set.
    """
    if n < 2:
        return False
    if n in _SMALL_TRIAL_PRIMES:
        return True
    if (n & 1) == 0:
        return False
    for p in _SMALL_TRIAL_PRIMES[1:]:
        if n % p == 0:
            return False
    s, d = _mr_decompose(n)
    for a in _MR64_BASES:
        if a % n == 0:
            continue
        if _mr_is_witness(a, n, s, d):
            return False
    return True


def ref_primes_upto(n: int) -> List[int]:
    """Simple sieve of Eratosthenes producing all primes <= n (n >= 0)."""
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    limit = int(math.isqrt(n))
    for p in range(2, limit + 1):
        if sieve[p]:
            start = p * p
            step = p
            sieve[start:n + 1:step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


def ref_factorization_trial(n: int) -> Dict[int, int]:
    """
    Reference prime factorization via trial division (sufficient for small n).

    Intended for n up to ~1e7 in tests; used here for n <= 50_000 range checks.
    """
    if n == 0:
        raise ValueError("Cannot factor 0")
    n = abs(n)
    pf: Dict[int, int] = {}
    if n < 2:
        return pf
    # factor out 2
    e = 0
    while (n & 1) == 0:
        n >>= 1
        e += 1
    if e:
        pf[2] = e
    # odd trial
    p = 3
    limit = int(math.isqrt(n))
    while p <= limit and n > 1:
        if n % p == 0:
            e = 1
            n //= p
            while n % p == 0:
                n //= p
                e += 1
            pf[p] = e
            limit = int(math.isqrt(n))
        p += 2
    if n > 1:
        pf[n] = pf.get(n, 0) + 1
    return pf


def prod_int(values: Iterable[int]) -> int:
    out = 1
    for v in values:
        out *= v
    return out


def int_root(n: int, k: int) -> int:
    if n < 0 or k < 1:
        raise ValueError("int_root expects n >= 0 and k >= 1")
    lo, hi = 0, n
    while lo <= hi:
        mid = (lo + hi) // 2
        p = mid**k
        if p == n:
            return mid
        if p < n:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi


def ref_perfect_power_small(n: int) -> tuple[int, int]:
    if n in (0, 1):
        return (n, 2)
    if n == -1:
        return (-1, 3)
    is_negative = n < 0
    n_abs = -n if is_negative else n
    max_b = int(math.log2(n_abs)) + 2
    start = 3 if is_negative else 2
    step = 2 if is_negative else 1
    for b in range(start, max_b + 1, step):
        a = int_root(n_abs, b)
        if a > 1 and a**b == n_abs:
            return ((-a if is_negative else a), b)
    return (n, 1)


def divisors_from_pf(pf: Dict[int, int]) -> List[int]:
    """Generate all positive divisors from a prime factorization dict."""
    divs = [1]
    for p, e in pf.items():
        current = list(divs)
        p_pow = 1
        for _ in range(e):
            p_pow *= p
            divs.extend(d * p_pow for d in current)
    return sorted(divs)


def assert_pf_valid(testcase: unittest.TestCase, n: int, factors: Tuple[int, ...]):
    """Shared assertions for prime_factors output."""
    if n == 0:
        testcase.fail("assert_pf_valid called with n=0")
    testcase.assertEqual(
        tuple(sorted(factors)), factors, "prime_factors must be sorted")
    testcase.assertEqual(
        prod_int(factors), abs(n), "product of prime_factors must equal |n|")
    for f in factors:
        testcase.assertIsInstance(f, int)
        testcase.assertGreaterEqual(f, 2)
        # Factors should fit in < 2^64 for our test corpus; enforce for oracle safety.
        testcase.assertLess(
            f, 1 << 64, "factor exceeds 2^64; adjust test corpus or oracle")
        testcase.assertTrue(
            ref_is_prime_64(f), f"factor {f} is not prime by reference oracle")


# ------------------------------- Test cases --------------------------------

class TestPrimeFactorsEdgeCases(unittest.TestCase):
    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            UT.prime_factors(0)

    def test_one_and_minus_one(self):
        self.assertEqual(UT.prime_factors(1), ())
        self.assertEqual(UT.prime_factors(-1), ())

    def test_small_examples(self):
        cases = {
            2: (2,),
            3: (3,),
            4: (2, 2),
            6: (2, 3),
            12: (2, 2, 3),
            60: (2, 2, 3, 5),
            -60: (2, 2, 3, 5),
            97: (97,),
            2**20: (2,) * 20,
        }
        for n, expected in cases.items():
            with self.subTest(n=n):
                got = UT.prime_factors(n)
                self.assertEqual(got, expected)
                assert_pf_valid(self, n, got)

    def test_sorted_and_prime_invariants_random_small(self):
        rng = random.Random(0xDEC0DE)
        for _ in range(300):
            n = rng.randrange(1, 1_000_000)
            if rng.random() < 0.2:
                n = -n
            got = UT.prime_factors(n)
            with self.subTest(n=n):
                assert_pf_valid(self, n, got)


class TestPrimeFactorizationAPI(unittest.TestCase):
    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            UT.prime_factorization(0)

    def test_one_empty(self):
        self.assertEqual(dict(UT.prime_factorization(1)), {})
        self.assertEqual(dict(UT.prime_factorization(-1)), {})

    def test_examples(self):
        cases = {
            1: {},
            2: {2: 1},
            12: {2: 2, 3: 1},
            360: {2: 3, 3: 2, 5: 1},
            -360: {2: 3, 3: 2, 5: 1},
            97: {97: 1},
            2**20: {2: 20},
        }
        for n, expected in cases.items():
            with self.subTest(n=n):
                got = dict(UT.prime_factorization(n))
                self.assertEqual(got, expected)
                # sanity: exponents positive
                self.assertTrue(all(e > 0 for e in got.values()))
                # keys are primes (within oracle domain)
                for p in got:
                    self.assertLess(p, 1 << 64)
                    self.assertTrue(ref_is_prime_64(p))

    def test_consistency_with_prime_factors(self):
        rng = random.Random(0xBADC0FFEE)
        for _ in range(400):
            n = rng.randrange(1, 2_000_000)
            if rng.random() < 0.25:
                n = -n
            pf_tuple = UT.prime_factors(n)
            pf_dict = dict(UT.prime_factorization(n))
            with self.subTest(n=n):
                self.assertEqual(Counter(pf_tuple), Counter(pf_dict))
                self.assertEqual(prod_int(p**e for p, e in pf_dict.items()), abs(n))


class TestDivisorsAPI(unittest.TestCase):
    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            UT.divisors(0)

    def test_one(self):
        self.assertEqual(UT.divisors(1), (1,))
        self.assertEqual(UT.divisors(-1), (1,))

    def test_examples(self):
        cases = {
            2: (1, 2),
            3: (1, 3),
            4: (1, 2, 4),
            6: (1, 2, 3, 6),
            28: (1, 2, 4, 7, 14, 28),
            360: tuple(sorted(divisors_from_pf({2: 3, 3: 2, 5: 1}))),
            -360: tuple(sorted(divisors_from_pf({2: 3, 3: 2, 5: 1}))),
        }
        for n, expected in cases.items():
            with self.subTest(n=n):
                got = UT.divisors(n)
                self.assertEqual(got, expected)
                # invariants
                self.assertEqual(
                    got, tuple(sorted(set(got))), "divisors must be sorted and unique")
                self.assertEqual(got[0], 1)
                self.assertEqual(got[-1], abs(n))
                for d in got:
                    self.assertGreaterEqual(d, 1)
                    self.assertEqual(abs(n) % d, 0)

    def test_divisors_match_factorization_random_small(self):
        rng = random.Random(0xFEEDFACE)
        for _ in range(250):
            n = rng.randrange(1, 200_000)
            if rng.random() < 0.2:
                n = -n
            pf = dict(UT.prime_factorization(n))
            expected = tuple(divisors_from_pf(pf))
            got = UT.divisors(n)
            with self.subTest(n=n):
                self.assertEqual(got, expected)


class TestAgainstReferenceTrialDivision(unittest.TestCase):
    def test_exact_factorization_for_range(self):
        # Exhaustive reference cross-check for a moderate range.
        # This is intentionally sized to remain stable/fast in CI.
        for n in range(1, 50_001):
            pf_ref = ref_factorization_trial(n)
            pf_got = dict(UT.prime_factorization(n))
            with self.subTest(n=n):
                self.assertEqual(pf_got, pf_ref)

    def test_negative_matches_positive_reference(self):
        rng = random.Random(0x12345678)
        for _ in range(300):
            n = rng.randrange(1, 1_000_000)
            pf_pos = dict(UT.prime_factorization(n))
            pf_neg = dict(UT.prime_factorization(-n))
            with self.subTest(n=n):
                self.assertEqual(pf_neg, pf_pos)


class TestStructuredLargerInputs(unittest.TestCase):
    def test_large_prime_and_composite_regressions(self):
        # Known prime: M61 = 2^61 - 1
        M61 = (1 << 61) - 1
        got = UT.prime_factors(M61)
        self.assertEqual(got, (M61,))
        assert_pf_valid(self, M61, got)

        # 2^64 - 1 has a well-known factorization:
        # 2^64 - 1 = (2^32 - 1)(2^32 + 1)
        #          = (3*5*17*257*65537) * (641*6700417)
        n = (1 << 64) - 1
        expected = tuple(sorted([3, 5, 17, 257, 641, 65537, 6700417]))
        got = UT.prime_factors(n)
        self.assertEqual(got, expected)
        assert_pf_valid(self, n, got)

    def test_semiprimes_and_prime_powers_over_59(self):
        # Ensure factors > 59 so trial division doesn't immediately solve everything.
        p = 1_000_003
        q = 1_000_033
        self.assertTrue(ref_is_prime_64(p) and ref_is_prime_64(q))
        n = p * q
        got = UT.prime_factors(n)
        self.assertEqual(got, tuple(sorted((p, q))))
        assert_pf_valid(self, n, got)

        # prime square
        r = 999_983
        self.assertTrue(ref_is_prime_64(r))
        n2 = r * r
        got2 = UT.prime_factors(n2)
        self.assertEqual(got2, (r, r))
        assert_pf_valid(self, n2, got2)

        # mixed powers
        n3 = (2**10) * (p**2) * (q**1)
        got3 = UT.prime_factors(n3)
        expected3 = tuple(sorted((2,) * 10 + (p,) * 2 + (q,)))
        self.assertEqual(got3, expected3)
        assert_pf_valid(self, n3, got3)

    def test_random_composites_from_known_primes(self):
        # Randomly construct integers from known primes, then verify exact
        # factorization.
        rng = random.Random(0x0DDBA11)

        prime_pool = ref_primes_upto(5000)  # includes <=59 and >59 primes
        prime_pool = [p for p in prime_pool if p >= 2]
        self.assertIn(2, prime_pool)
        self.assertIn(61, prime_pool)

        for _ in range(250):
            n = 1
            expected_pf: Dict[int, int] = {}
            num_terms = rng.randrange(1, 7)  # up to 6 distinct picks
            for __ in range(num_terms):
                p = rng.choice(prime_pool)
                e = rng.randrange(1, 4)  # exponent 1..3
                # Keep n bounded to ensure factors remain in <2^64 oracle domain
                candidate = n * (p ** e)
                if candidate >= (1 << 63):
                    continue
                n = candidate
                expected_pf[p] = expected_pf.get(p, 0) + e

            if n == 1:
                n = 2 * 3 * 5
                expected_pf = {2: 1, 3: 1, 5: 1}

            if rng.random() < 0.25:
                n = -n

            got_pf = dict(UT.prime_factorization(n))
            got_factors = UT.prime_factors(n)
            with self.subTest(n=n):
                self.assertEqual(got_pf, dict(Counter(got_factors)))
                self.assertEqual(got_pf, expected_pf)
                assert_pf_valid(self, n, got_factors)

                # divisors should match factorization
                self.assertEqual(UT.divisors(n), tuple(divisors_from_pf(got_pf)))


class TestInternalHelpersAndPipeline(unittest.TestCase):
    def test_partial_factorization_helper(self):
        # Guard: internal helper should exist if factorization section was included.
        self.assertTrue(hasattr(UT, "_partial_factorization"))
        pf, co = UT._partial_factorization(2**6 * 3**3 * 5 * 7 * 11, [2, 3, 5, 7])
        self.assertEqual(pf, {2: 6, 3: 3, 5: 1, 7: 1})
        self.assertEqual(co, 11)

        pf2, co2 = UT._partial_factorization(13 * 13 * 17, [2, 3, 5, 7, 11, 13])
        self.assertEqual(pf2, {13: 2})
        self.assertEqual(co2, 17)

    def test_fermat_factorization_success_and_contract(self):
        self.assertTrue(
            hasattr(UT, "_fermat_factorization"),
            "expected internal _fermat_factorization",
        )

        # Close semiprime; Fermat should find the factors immediately.
        n = 101 * 103
        factors = UT._fermat_factorization(n)
        self.assertIsNotNone(factors)
        a, b = factors
        self.assertGreater(a, 1)
        self.assertGreater(b, 1)
        self.assertEqual(a * b, n)
        self.assertEqual(tuple(sorted(factors)), (101, 103))

    def test_fermat_factorization_iteration_boundaries(self):
        self.assertTrue(
            hasattr(UT, "_fermat_factorization"),
            "expected internal _fermat_factorization",
        )

        # This semiprime requires exactly 3 checks of 'a' (default num_iterations)
        # for Fermat to succeed.
        n = 89 * 131
        self.assertIsNone(UT._fermat_factorization(n, num_iterations=2))
        factors = UT._fermat_factorization(n, num_iterations=3)
        self.assertIsNotNone(factors)
        self.assertEqual(prod_int(factors), n)
        self.assertEqual(tuple(sorted(factors)), (89, 131))

    def test_fermat_factorization_returns_none_when_not_found_quickly(self):
        self.assertTrue(
            hasattr(UT, "_fermat_factorization"),
            "expected internal _fermat_factorization",
        )

        # Far-apart semiprime; Fermat should not find within the default iteration cap.
        n = 101 * 1009
        self.assertIsNone(UT._fermat_factorization(n))

        # Also: integers congruent to 2 mod 4 are not differences of squares.
        self.assertIsNone(UT._fermat_factorization(6, num_iterations=10))

    def test_brent_edge_cases_and_capped_failure(self):
        self.assertTrue(hasattr(UT, "_brent"), "expected internal _brent")

        # Early exits
        self.assertEqual(UT._brent(10), 2)
        self.assertEqual(UT._brent(25), 5)

        # With max_iterations=0, the inner loop always exits immediately.
        self.assertEqual(UT._brent(91, max_attempts=1, max_iterations=0), 1)

    def test_brent_finds_nontrivial_factor_when_unlimited(self):
        self.assertTrue(hasattr(UT, "_brent"), "expected internal _brent")

        n = 91  # 7 * 13
        d = UT._brent(n, max_attempts=None, max_iterations=None)
        self.assertIn(d, (7, 13))
        self.assertEqual(n % d, 0)
        self.assertNotIn(d, (1, n))

    def test_brent_returns_1_for_prime(self):
        self.assertTrue(hasattr(UT, "_brent"), "expected internal _brent")

        p = 1019
        self.assertTrue(ref_is_prime_64(p))
        self.assertEqual(UT._brent(p, max_attempts=3, max_iterations=200), 1)

    def test_ecm_finds_factor_with_deterministic_sigma(self):
        self.assertTrue(hasattr(UT, "_ecm"), "expected internal _ecm")

        # Choose n where sigma=101 triggers a factor during curve setup.
        n = 101 * 103

        def det_randbelow(limit: int) -> int:
            # _ecm uses randbelow(n - 7) + 6.
            return 101 - 6

        with mock.patch.object(UT.secrets, "randbelow", side_effect=det_randbelow):
            factor = UT._ecm(n)

        self.assertIn(factor, (101, 103))
        self.assertEqual(n % factor, 0)

    def test_ecm_finds_small_factor_in_gt128bit_composite(self):
        """ECM should find small prime factors in >128-bit composites."""
        self.assertTrue(hasattr(UT, "_ecm"), "expected internal _ecm")

        # Construct >128-bit composite with a small prime factor.
        # Small factor should be found by ECM relatively quickly.
        small_prime = 1009  # small enough for ECM to find
        large_prime = (1 << 127) - 1  # M127, a 127-bit Mersenne prime
        n = small_prime * large_prime
        self.assertGreater(n.bit_length(), 128)

        # ECM should find the small factor
        factor = UT._ecm(n, max_curves=100)
        self.assertEqual(factor, small_prime)
        self.assertEqual(n % factor, 0)

    def test_ecm_returns_1_for_prime_gt128bit(self):
        """ECM should return 1 (failure) when given a prime."""
        self.assertTrue(hasattr(UT, "_ecm"), "expected internal _ecm")

        # M127 is a 127-bit Mersenne prime
        prime = (1 << 127) - 1
        self.assertGreater(prime.bit_length(), 64)

        # ECM cannot factor a prime, should return 1
        factor = UT._ecm(prime, max_curves=10)
        self.assertEqual(factor, 1)

    def test_siqs_finds_nontrivial_factor(self):
        self.assertTrue(hasattr(UT, "_siqs"), "expected internal _siqs")

        n = 101 * 103
        factor = UT._siqs(n)
        self.assertTrue(1 < factor < n)
        self.assertEqual(n % factor, 0)

    def test_pipeline_64bit_brent_retry(self):
        """
        For <=64-bit composites, _gen_prime_factors tries a capped Brent attempt first.
        If it fails, it retries Brent with max_attempts=None (guaranteed eventual
        success).
        We mock _brent to force this branch deterministically.
        """
        self.assertTrue(hasattr(UT, "_brent"), "expected internal _brent")

        # Use a semiprime where Fermat factorization will not succeed within its
        # default iteration cap, so we deterministically exercise the Brent branch.
        n = 257 * 1009  # both >= 256, so primorial GCD won't strip; n fits in 64-bit

        def brent_side_effect(
            arg_n, *, max_attempts=None, max_iterations=None, batch_size=128):
            # Fail the initial capped call; succeed on the unlimited retry.
            if max_attempts is None:
                return 257
            return 1

        with mock.patch.object(UT, "_brent", side_effect=brent_side_effect) as m_brent:
            got = UT.prime_factors(n)

        self.assertEqual(got, (257, 1009))
        assert_pf_valid(self, n, got)

        # Assert we saw both the capped attempt and the unlimited retry.
        calls = m_brent.call_args_list
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(any(kw.get("max_attempts") is None for _, kw in calls),
                        "expected a Brent retry with max_attempts=None")

    def test_pipeline_gt64bit_ecm_used_when_brent_fails(self):
        """
        For >128-bit composites, after a (more aggressively capped) Brent attempt,
        the pipeline tries ECM. We mock _brent to fail and _ecm to return a factor.
        """
        self.assertTrue(hasattr(UT, "_brent"))
        self.assertTrue(hasattr(UT, "_ecm"))
        self.assertTrue(hasattr(UT, "_siqs"))

        # Construct a >128-bit semiprime with known prime factors.
        d = 228_479  # prime
        q = (1 << 127) - 1  # Mersenne prime M127 (127-bit prime)
        self.assertTrue(ref_is_prime_64(d))
        n = d * q
        self.assertGreater(n.bit_length(), 128)

        with mock.patch.object(UT, "_brent", return_value=1) as m_brent, \
             mock.patch.object(UT, "_ecm", return_value=d) as m_ecm, \
             mock.patch.object(
                UT, "_siqs", side_effect=AssertionError("SIQS should not run here")):
            got = UT.prime_factors(n)

        self.assertEqual(got, (d, q))
        # Skip assert_pf_valid since ref_is_prime_64 can't verify 127-bit primes
        self.assertEqual(prod_int(got), n)
        self.assertGreaterEqual(m_brent.call_count, 1)
        self.assertGreaterEqual(m_ecm.call_count, 1)

    def test_pipeline_gt64bit_siqs_fallback_and_B_growth(self):
        """
        If Brent fails for a 64-128 bit composite, the pipeline falls back to SIQS
        and increases B until SIQS yields a non-trivial factor. We mock SIQS to
        fail once and then succeed, and assert B grows according to the implementation.
        Note: ECM is only tried for >128-bit inputs, so it's not called here.
        """
        self.assertTrue(hasattr(UT, "_brent"))
        self.assertTrue(hasattr(UT, "_siqs"))

        p = 2_147_483_647  # prime (M31)
        q = 17_179_869_209  # prime (next after 2^34)
        self.assertTrue(ref_is_prime_64(p))
        self.assertTrue(ref_is_prime_64(q))
        n = p * q
        self.assertGreater(n.bit_length(), 64)
        self.assertLessEqual(n.bit_length(), 128)  # ECM not tried for this range

        # First SIQS call fails; second succeeds with p.
        def siqs_side_effect(
            arg_n,
            B=None,
            M=None,
            large_prime_bound_multiplier=None,
            max_polynomial_count=None,
        ):
            if B is None:
                return 1
            return p

        with mock.patch.object(UT, "_brent", return_value=1) as m_brent, \
             mock.patch.object(UT, "_siqs", side_effect=siqs_side_effect) as m_siqs:
            got = UT.prime_factors(n)

        self.assertEqual(got, tuple(sorted((p, q))))
        assert_pf_valid(self, n, got)
        self.assertGreaterEqual(m_brent.call_count, 1)

        # Validate SIQS was called at least twice and parameters grew
        self.assertGreaterEqual(m_siqs.call_count, 2)
        first_call = m_siqs.call_args_list[0].kwargs
        second_call = m_siqs.call_args_list[1].kwargs
        self.assertIsNone(first_call.get("B", None))
        self.assertIsNone(first_call.get("max_polynomial_count", None))
        self.assertGreater(second_call.get("B"), 60000)
        self.assertGreater(second_call.get("max_polynomial_count"), 60000)


class TestPerfectPower(unittest.TestCase):
    def test_special_cases(self):
        self.assertEqual(UT.perfect_power(0), (0, 2))
        self.assertEqual(UT.perfect_power(1), (1, 2))
        self.assertEqual(UT.perfect_power(-1), (-1, 3))

    def test_known_values(self):
        self.assertEqual(UT.perfect_power(16), (4, 2))
        self.assertEqual(UT.perfect_power(27), (3, 3))
        self.assertEqual(UT.perfect_power(32), (2, 5))
        self.assertEqual(UT.perfect_power(72), (72, 1))
        self.assertEqual(UT.perfect_power(-8), (-2, 3))
        self.assertEqual(UT.perfect_power(-32), (-2, 5))
        self.assertEqual(UT.perfect_power(-12), (-12, 1))

    def test_larger_values(self):
        cases = {
            64: (8, 2),
            81: (9, 2),
            1024: (32, 2),
            3125: (5, 5),
            -64: (-4, 3),
            -243: (-3, 5),
        }
        for n, expected in cases.items():
            with self.subTest(n=n):
                self.assertEqual(UT.perfect_power(n), expected)

    def test_small_range_matches_reference(self):
        for n in range(-200, 201):
            with self.subTest(n=n):
                self.assertEqual(UT.perfect_power(n), ref_perfect_power_small(n))


if __name__ == "__main__":
    unittest.main(verbosity=2)
