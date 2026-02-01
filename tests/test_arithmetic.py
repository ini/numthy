from __future__ import annotations

import importlib
import importlib.util
import math
import os
import pathlib
import random
import sys
import unittest

from typing import Callable, Dict, Iterable, List, Sequence, Tuple


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
    "omega",
    "big_omega",
    "divisor_count",
    "divisor_sum",
    "divisor_function",
    "radical",
    "mobius",
    "totient",
    "carmichael",
    "valuation",
    "partition",
    "multiplicative_range",
)
_missing = [name for name in _REQUIRED if not hasattr(UT, name)]
if _missing:
    raise ImportError(f"Module under test is missing expected API: {_missing}")


# ------------------------ Reference / oracle utilities ------------------------


def prod_int(values: Iterable[int]) -> int:
    """Integer product with empty-product = 1."""
    out = 1
    for v in values:
        out *= int(v)
    return out


def sieve_spf(limit: int) -> Tuple[List[int], List[int]]:
    """Return (spf, primes) where spf[n] is smallest prime factor for n."""
    if limit < 1:
        return [0] * (limit + 1), []

    spf = [0] * (limit + 1)
    primes: List[int] = []
    spf[0] = 0
    spf[1] = 1
    for i in range(2, limit + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            x = i * p
            if x > limit:
                break
            spf[x] = p
            if p == spf[i]:
                break
    return spf, primes


_REF_LIMIT = 200_000
_SPF, _PRIMES = sieve_spf(_REF_LIMIT)


def ref_is_prime(n: int) -> bool:
    """Deterministic trial-division primality for n with sqrt(n) <= _REF_LIMIT."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if (n & 1) == 0:
        return False
    limit = int(math.isqrt(n))
    if limit > _REF_LIMIT:
        raise ValueError("oracle prime check exceeded limit")
    for p in _PRIMES:
        if p > limit:
            break
        if n % p == 0:
            return n == p
    return True


def ref_factorization(n: int) -> Dict[int, int]:
    """Factor |n| via SPF for small n, and trial division for larger n."""
    if n == 0:
        raise ValueError("Cannot factor 0")
    n = abs(int(n))
    pf: Dict[int, int] = {}
    if n < 2:
        return pf

    if n <= _REF_LIMIT:
        while n > 1:
            p = _SPF[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            pf[p] = pf.get(p, 0) + e
        return pf

    limit = int(math.isqrt(n))
    if limit > _REF_LIMIT:
        raise ValueError("oracle factorization exceeded limit")

    for p in _PRIMES:
        if p * p > n:
            break
        if n % p:
            continue
        e = 1
        n //= p
        while n % p == 0:
            n //= p
            e += 1
        pf[p] = pf.get(p, 0) + e
        if n == 1:
            break
    if n > 1:
        pf[n] = pf.get(n, 0) + 1
    return pf


def ref_divisors(n: int) -> List[int]:
    """All positive divisors of n (n must be nonzero)."""
    pf = ref_factorization(n)
    divs = [1]
    for p, e in sorted(pf.items()):
        current = list(divs)
        p_pow = 1
        for _ in range(e):
            p_pow *= p
            divs.extend(d * p_pow for d in current)
    return sorted(divs)


def ref_omega(n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    return len(ref_factorization(n))


def ref_big_omega(n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    return sum(ref_factorization(n).values())


def ref_divisor_count(n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    return prod_int(e + 1 for e in ref_factorization(n).values())


def ref_divisor_sum(n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    pf = ref_factorization(n)
    return prod_int((p ** (e + 1) - 1) // (p - 1) for p, e in pf.items())


def ref_divisor_function(n: int, k: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    if k < 0:
        raise ValueError("k must be non-negative")
    pf = ref_factorization(n)
    if k == 0:
        return prod_int(e + 1 for e in pf.values())
    return prod_int(
        (pow(p, k * (e + 1)) - 1) // (pow(p, k) - 1) for p, e in pf.items()
    )


def brute_partition(n: int, restrict: Callable[[int], bool] | None = None) -> int:
    """Brute-force partition count via DP using allowed parts."""
    if n < 0:
        raise ValueError("n must be non-negative")
    allowed = [k for k in range(1, n + 1) if restrict is None or restrict(k)]
    dp = [0] * (n + 1)
    dp[0] = 1
    for part in allowed:
        for s in range(part, n + 1):
            dp[s] += dp[s - part]
    return dp[n]


def ref_radical(n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    return prod_int(ref_factorization(n).keys())


def ref_mobius(n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return 1
    pf = ref_factorization(n)
    if any(e > 1 for e in pf.values()):
        return 0
    return -1 if (len(pf) & 1) else 1


def ref_totient(n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return 1
    pf = ref_factorization(n)
    phi = n
    for p in pf:
        phi -= phi // p
    return phi


def ref_carmichael(n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    pf = ref_factorization(n)
    terms: List[int] = []
    for p, e in pf.items():
        if p == 2:
            terms.append(e if e < 3 else 2 ** (e - 2))
        else:
            terms.append((p - 1) * (p ** (e - 1)))
    return math.lcm(*terms) if terms else 1


def ref_valuation(n: int, p: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    if p < 2 or not ref_is_prime(p):
        raise ValueError("p must be prime")
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


# ------------------------------- Test cases --------------------------------


class TestArithmeticErrorHandling(unittest.TestCase):
    def test_invalid_n_raises(self):
        fns = [
            UT.omega,
            UT.big_omega,
            UT.divisor_count,
            UT.divisor_sum,
            UT.radical,
            UT.mobius,
            UT.totient,
            UT.carmichael,
        ]
        for fn in fns:
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(ValueError):
                    fn(0)
                with self.assertRaises(ValueError):
                    fn(-1)

    def test_divisor_function_invalid(self):
        with self.assertRaises(ValueError):
            UT.divisor_function(0, 1)
        with self.assertRaises(ValueError):
            UT.divisor_function(-5, 1)
        with self.assertRaises(ValueError):
            UT.divisor_function(10, -1)

    def test_valuation_invalid(self):
        with self.assertRaises(ValueError):
            UT.valuation(0, 2)
        with self.assertRaises(ValueError):
            UT.valuation(-7, 2)
        for p in [0, 1, 4, 9, 15]:
            with self.subTest(p=p):
                with self.assertRaises(ValueError):
                    UT.valuation(10, p)

    def test_multiplicative_range_invalid(self):
        with self.assertRaises(ValueError):
            UT.multiplicative_range(lambda n: 1, -1)


class TestArithmeticKnownValues(unittest.TestCase):
    def test_base_cases_n_1(self):
        self.assertEqual(UT.omega(1), 0)
        self.assertEqual(UT.big_omega(1), 0)
        self.assertEqual(UT.divisor_count(1), 1)
        self.assertEqual(UT.divisor_sum(1), 1)
        self.assertEqual(UT.divisor_function(1, 0), 1)
        self.assertEqual(UT.divisor_function(1, 1), 1)
        self.assertEqual(UT.radical(1), 1)
        self.assertEqual(UT.mobius(1), 1)
        self.assertEqual(UT.totient(1), 1)
        self.assertEqual(UT.carmichael(1), 1)

    def test_examples_small(self):
        cases = {
            2: {
                "omega": 1,
                "big_omega": 1,
                "tau": 2,
                "sigma1": 3,
                "rad": 2,
                "mu": -1,
                "phi": 1,
                "lam": 1,
            },
            12: {
                "omega": 2,
                "big_omega": 3,
                "tau": 6,
                "sigma1": 28,
                "rad": 6,
                "mu": 0,
                "phi": 4,
                "lam": 2,
            },
            30: {
                "omega": 3,
                "big_omega": 3,
                "tau": 8,
                "sigma1": 72,
                "rad": 30,
                "mu": -1,
                "phi": 8,
                "lam": 4,
            },
            72: {
                "omega": 2,
                "big_omega": 5,
                "tau": 12,
                "sigma1": 195,
                "rad": 6,
                "mu": 0,
                "phi": 24,
                "lam": 6,
            },
        }
        for n, exp in cases.items():
            with self.subTest(n=n):
                self.assertEqual(UT.omega(n), exp["omega"])
                self.assertEqual(UT.big_omega(n), exp["big_omega"])
                self.assertEqual(UT.divisor_count(n), exp["tau"])
                self.assertEqual(UT.divisor_sum(n), exp["sigma1"])
                self.assertEqual(UT.radical(n), exp["rad"])
                self.assertEqual(UT.mobius(n), exp["mu"])
                self.assertEqual(UT.totient(n), exp["phi"])
                self.assertEqual(UT.carmichael(n), exp["lam"])

    def test_divisor_function_consistency(self):
        for n in [1, 2, 12, 28, 60, 210]:
            with self.subTest(n=n):
                self.assertEqual(UT.divisor_function(n, 0), UT.divisor_count(n))
                self.assertEqual(UT.divisor_function(n, 1), UT.divisor_sum(n))

        # Known: sigma_2(12) = 1^2+2^2+3^2+4^2+6^2+12^2 = 210
        self.assertEqual(UT.divisor_function(12, 2), 210)

    def test_valuation_examples(self):
        self.assertEqual(UT.valuation(1, 2), 0)
        self.assertEqual(UT.valuation(12, 2), 2)
        self.assertEqual(UT.valuation(12, 3), 1)
        self.assertEqual(UT.valuation(12, 5), 0)
        self.assertEqual(UT.valuation(2**20, 2), 20)


class TestAgainstReferenceSmallRange(unittest.TestCase):
    def test_against_reference_up_to_5000(self):
        for n in range(1, 5001):
            with self.subTest(n=n):
                self.assertEqual(UT.omega(n), ref_omega(n))
                self.assertEqual(UT.big_omega(n), ref_big_omega(n))
                self.assertEqual(UT.divisor_count(n), ref_divisor_count(n))
                self.assertEqual(UT.divisor_sum(n), ref_divisor_sum(n))
                self.assertEqual(UT.radical(n), ref_radical(n))
                self.assertEqual(UT.mobius(n), ref_mobius(n))
                self.assertEqual(UT.totient(n), ref_totient(n))
                self.assertEqual(UT.carmichael(n), ref_carmichael(n))

    def test_divisor_function_against_reference(self):
        ks = [0, 1, 2, 3, 5]
        for n in range(1, 2001):
            for k in ks:
                with self.subTest(n=n, k=k):
                    self.assertEqual(
                        UT.divisor_function(n, k),
                        ref_divisor_function(n, k),
                    )

    def test_valuation_against_reference(self):
        primes = [2, 3, 5, 7, 11, 13, 17, 19]
        for n in range(1, 2001):
            for p in primes:
                with self.subTest(n=n, p=p):
                    self.assertEqual(UT.valuation(n, p), ref_valuation(n, p))


class TestIdentities(unittest.TestCase):
    def test_mobius_sum_over_divisors(self):
        # Sum_{d|n} mu(d) = 1 if n=1 else 0.
        for n in range(1, 2001):
            divs = ref_divisors(n)
            total = sum(UT.mobius(d) for d in divs)
            expected = 1 if n == 1 else 0
            with self.subTest(n=n):
                self.assertEqual(total, expected)

    def test_totient_sum_over_divisors(self):
        # Sum_{d|n} phi(d) = n.
        for n in range(1, 2001):
            divs = ref_divisors(n)
            total = sum(UT.totient(d) for d in divs)
            with self.subTest(n=n):
                self.assertEqual(total, n)

    def test_mobius_inversion_totient(self):
        # phi(n) = sum_{d|n} mu(d) * (n/d).
        for n in range(1, 2001):
            divs = ref_divisors(n)
            total = sum(UT.mobius(d) * (n // d) for d in divs)
            with self.subTest(n=n):
                self.assertEqual(total, UT.totient(n))

    def test_carmichael_divides_totient(self):
        for n in range(1, 5001):
            lam = UT.carmichael(n)
            phi = ref_totient(n)
            with self.subTest(n=n):
                self.assertEqual(phi % lam, 0)

    def test_carmichael_exponent_property(self):
        rng = random.Random(0xC0FFEE)
        for _ in range(200):
            n = rng.randrange(2, 50_000)
            lam = UT.carmichael(n)

            # Try a few random bases; skip those not coprime to n.
            checks = 0
            for _ in range(12):
                a = rng.randrange(2, n)
                if math.gcd(a, n) != 1:
                    continue
                checks += 1
                with self.subTest(n=n, a=a):
                    self.assertEqual(pow(a, lam, n), 1)
            self.assertGreaterEqual(checks, 1)


class TestRandomizedAgainstReference(unittest.TestCase):
    def test_random_numbers_under_oracle_limit(self):
        rng = random.Random(0xB16B00B5)
        max_n = _REF_LIMIT * _REF_LIMIT
        for _ in range(250):
            n = rng.randrange(1, max_n)
            with self.subTest(n=n):
                self.assertEqual(UT.omega(n), ref_omega(n))
                self.assertEqual(UT.big_omega(n), ref_big_omega(n))
                self.assertEqual(UT.divisor_count(n), ref_divisor_count(n))
                self.assertEqual(UT.divisor_sum(n), ref_divisor_sum(n))
                self.assertEqual(UT.divisor_function(n, 2), ref_divisor_function(n, 2))
                self.assertEqual(UT.radical(n), ref_radical(n))
                self.assertEqual(UT.mobius(n), ref_mobius(n))
                self.assertEqual(UT.totient(n), ref_totient(n))
                self.assertEqual(UT.carmichael(n), ref_carmichael(n))


class TestMultiplicativeRange(unittest.TestCase):
    def test_basic_shape_and_f0(self):
        self.assertEqual(UT.multiplicative_range(lambda n: 1, 0), [])
        self.assertEqual(UT.multiplicative_range(lambda n: 1, 1), [1])
        self.assertEqual(UT.multiplicative_range(lambda n: 1, 3), [1, 1, 1])
        self.assertEqual(UT.multiplicative_range(lambda n: 1, 3, f0=7)[0], 7)

    def test_mapping_matches_reference(self):
        N = 50_000
        ref_dc = [1] * N
        ref_ds = [1] * N
        ref_rad = [1] * N
        ref_mu = [1] * N
        ref_phi = [1] * N
        for n in range(1, N):
            pf = ref_factorization(n)
            ref_dc[n] = prod_int(e + 1 for e in pf.values())
            ref_ds[n] = prod_int((p ** (e + 1) - 1) // (p - 1) for p, e in pf.items())
            ref_rad[n] = prod_int(pf.keys())
            if n == 1:
                ref_mu[n] = 1
            elif any(e > 1 for e in pf.values()):
                ref_mu[n] = 0
            else:
                ref_mu[n] = -1 if (len(pf) & 1) else 1
            phi = n
            for p in pf:
                phi -= phi // p
            ref_phi[n] = phi

        got_dc = UT.multiplicative_range(UT.divisor_count, N)
        got_ds = UT.multiplicative_range(UT.divisor_sum, N)
        got_rad = UT.multiplicative_range(UT.radical, N)
        got_mu = UT.multiplicative_range(UT.mobius, N)
        got_phi = UT.multiplicative_range(UT.totient, N)

        self.assertEqual(got_dc, ref_dc)
        self.assertEqual(got_ds, ref_ds)
        self.assertEqual(got_rad, ref_rad)
        self.assertEqual(got_mu, ref_mu)
        self.assertEqual(got_phi, ref_phi)

    def test_custom_prime_power_function(self):
        # f(p^e) = (p^e + 1) is multiplicative (built from prime-powers).
        def f_pp(p: int, e: int) -> int:
            return p ** e + 1

        N = 5000
        got = UT.multiplicative_range(f_pp, N)
        expected = [1] * N
        for n in range(1, N):
            pf = ref_factorization(n)
            expected[n] = prod_int((p ** e + 1) for p, e in pf.items())
        self.assertEqual(got, expected)

    def test_custom_single_arg_function(self):
        # f(n) = n is completely multiplicative.
        f = lambda n: n
        N = 5000
        got = UT.multiplicative_range(f, N, f0=0)
        expected = [0] + list(range(1, N))
        self.assertEqual(got, expected)


class TestPartition(unittest.TestCase):
    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            UT.partition(-1)

    def test_known_values(self):
        known = {
            0: 1,
            1: 1,
            2: 2,
            3: 3,
            4: 5,
            5: 7,
            6: 11,
            7: 15,
            8: 22,
            9: 30,
            10: 42,
            12: 77,
            15: 176,
            20: 627,
        }
        for n, p_n in known.items():
            with self.subTest(n=n):
                self.assertEqual(UT.partition(n), p_n)

    def test_mod_matches_bruteforce(self):
        rng = random.Random(0x51A17)
        for _ in range(120):
            n = rng.randrange(0, 35)
            mod = rng.randrange(2, 97)
            with self.subTest(n=n, mod=mod):
                self.assertEqual(UT.partition(n, mod=mod), brute_partition(n) % mod)

    def test_restrict_odd_parts(self):
        restrict = lambda k: (k & 1) == 1
        for n in range(0, 20):
            with self.subTest(n=n):
                self.assertEqual(UT.partition(n, restrict=restrict), brute_partition(n, restrict))

    def test_restrict_small_parts(self):
        restrict = lambda k: k <= 2
        for n in range(0, 25):
            with self.subTest(n=n):
                # Number of partitions using only 1s and 2s is floor(n/2) + 1.
                self.assertEqual(UT.partition(n, restrict=restrict), n // 2 + 1)

    def test_restrict_none_allowed(self):
        restrict = lambda k: False
        self.assertEqual(UT.partition(0, restrict=restrict), 1)
        for n in range(1, 10):
            with self.subTest(n=n):
                self.assertEqual(UT.partition(n, restrict=restrict), 0)

    def test_restrict_with_mod(self):
        restrict = lambda k: k % 3 == 0
        mod = 101
        for n in range(0, 30):
            with self.subTest(n=n):
                expected = brute_partition(n, restrict) % mod
                self.assertEqual(UT.partition(n, mod=mod, restrict=restrict), expected)


class TestAPIStability(unittest.TestCase):
    def test_return_types_are_int(self):
        funcs: List[Callable[..., int]] = [
            UT.omega,
            UT.big_omega,
            UT.divisor_count,
            UT.divisor_sum,
            UT.radical,
            UT.mobius,
            UT.totient,
            UT.carmichael,
        ]
        for n in [1, 2, 12, 360, 1024, 99991]:
            for fn in funcs:
                with self.subTest(fn=fn.__name__, n=n):
                    self.assertIsInstance(fn(n), int)

        self.assertIsInstance(UT.divisor_function(12, 2), int)
        self.assertIsInstance(UT.valuation(12, 2), int)
        self.assertIsInstance(UT.multiplicative_range(lambda n: 1, 10), list)


# ------------------------------- Entry point -------------------------------


if __name__ == "__main__":
    unittest.main(verbosity=2)
