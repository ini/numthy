from __future__ import annotations

import importlib
import importlib.util
import math
import os
import pathlib
import random
import sys
import unittest
import unittest.mock

from typing import Dict, List, Sequence, Tuple


# ----------------------------- Module loading -----------------------------

def _load_under_test():
    """
    Load the module under test with a bias toward the *local* file next to this test.

    Priority:
      1) NUMTHY_PATH (explicit file path)
      2) ./numthy.py next to this test file (or ./<NUMTHY_MODULE>.py)
      3) Import by name (NUMTHY_MODULE; default "numthy")
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

    here = pathlib.Path(__file__).resolve().parent
    candidate = here / f"{module_name}.py"
    if not candidate.exists():
        candidate = here / "numthy.py"
    if candidate.exists():
        spec = importlib.util.spec_from_file_location(
            "numthy_under_test", str(candidate))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {candidate}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["numthy_under_test"] = mod
        spec.loader.exec_module(mod)
        return mod

    return importlib.import_module(module_name)


UT = _load_under_test()


# ------------------------ Reference / oracle utilities ------------------------

def ref_is_prime_small(n: int) -> bool:
    """
    Simple deterministic primality for n up to ~1e9 (trial division), used for tests.
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if (n & 1) == 0:
        return False
    r = int(math.isqrt(n))
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True

def ref_factorization_trial(n: int) -> Dict[int, int]:
    """
    Trial-division factorization for moderate n (used as oracle in tests).
    """
    if n == 0:
        raise ValueError("Cannot factor 0")
    n = abs(n)
    pf: Dict[int, int] = {}
    if n < 2:
        return pf
    e = 0
    while (n & 1) == 0:
        n >>= 1
        e += 1
    if e:
        pf[2] = e
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

def ref_totient(n: int) -> int:
    """
    Euler totient via reference factorization (trial division).
    """
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return 1
    pf = ref_factorization_trial(n)
    out = n
    for p in pf:
        out -= out // p
    return out

def ref_carmichael_lambda(n: int) -> int:
    """Carmichael lambda via reference factorization (trial division)."""
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return 1
    pf = ref_factorization_trial(n)
    parts: List[int] = []
    for p, e in pf.items():
        if p == 2:
            if e == 1:
                parts.append(1)
            elif e == 2:
                parts.append(2)
            else:
                parts.append(2 ** (e - 2))
        else:
            parts.append((p - 1) * (p ** (e - 1)))
    return math.lcm(*parts) if parts else 1


def ref_multiplicative_order_bruteforce(a: int, mod: int) -> int:
    """Brute-force order in (Z/modZ)^× using lambda(mod) as a safe bound."""
    if mod <= 0:
        raise ValueError("mod must be positive")
    a %= mod
    if math.gcd(a, mod) != 1:
        raise ValueError("Must have gcd(a, mod) = 1")
    lam = ref_carmichael_lambda(mod)
    x = 1
    for k in range(1, lam + 1):
        x = (x * a) % mod
        if x == 1:
            return k
    raise AssertionError("Order not found within Carmichael bound (should be impossible).")


def ref_crt_pair(a1: int, n1: int, a2: int, n2: int) -> Tuple[int, int] | None:
    """
    Reference CRT for two congruences:
        x ≡ a1 (mod n1)
        x ≡ a2 (mod n2)
    Returns (x mod lcm, lcm) if solvable, else None.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Moduli must be positive")
    a1 %= n1
    a2 %= n2
    d = math.gcd(n1, n2)
    diff = a2 - a1
    if diff % d != 0:
        return None
    n1_, n2_ = n1 // d, n2 // d
    # Solve n1_ * t ≡ diff/d (mod n2_)
    t = ((diff // d) * pow(n1_, -1, n2_)) % n2_
    x = a1 + n1 * t
    mod = n1 * n2_  # lcm(n1, n2)
    return (x % mod, mod)


def ref_crt(residues: Sequence[int], moduli: Sequence[int]) -> int | None:
    """Reference CRT solver for multiple congruences via pairwise combination."""
    if len(residues) != len(moduli):
        raise ValueError("residues and moduli must have the same length")
    x, m = 0, 1
    for a, n in zip(residues, moduli):
        res = ref_crt_pair(x, m, a, n)
        if res is None:
            return None
        x, m = res
    return x


def ref_legendre(a: int, p: int) -> int:
    """Reference Legendre symbol for odd prime p."""
    if p <= 2 or not ref_is_prime_small(p):
        raise ValueError("p must be an odd prime")
    a %= p
    if a == 0:
        return 0
    t = pow(a, (p - 1) // 2, p)
    return -1 if t == p - 1 else t  # t in {1, p-1}


def ref_jacobi(a: int, n: int) -> int:
    """
    Reference Jacobi symbol (a|n) for odd positive n using prime factorization:
      n = ∏ p_i^{e_i}
      (a|n) = ∏ (a|p_i)^{e_i}
    """
    if n <= 0 or (n & 1) == 0:
        raise ValueError("n must be an odd positive integer")
    a %= n
    if n == 1:
        return 1
    pf = ref_factorization_trial(n)
    out = 1
    for p, e in pf.items():
        lp = ref_legendre(a, p)  # may be 0
        if lp == 0:
            return 0
        if e & 1:
            out *= lp
    return out


def ref_kronecker(a: int, n: int) -> int:
    """
    Reference Kronecker symbol (a|n) consistent with standard extension:
      - handles n == 0
      - handles sign of n via (a|-1)
      - handles 2-adic part via (a|2)^v2(n)
      - handles odd part via Jacobi
    """
    if n == 0:
        return 1 if a == 1 or a == -1 else 0

    # sign of n
    if n > 0:
        sign = 1
        nn = n
    else:
        # (a|-1) = -1 iff a < 0, else +1
        sign = -1 if a < 0 else 1
        nn = -n

    # factor out 2-adic valuation
    v2 = (nn & -nn).bit_length() - 1
    odd = nn >> v2

    # if v2>0 and a even => (a|2)=0 => whole symbol 0
    if v2 and (a & 1) == 0:
        return 0

    # (a|2)
    if v2 == 0:
        k2 = 1
    else:
        a8 = a & 7
        a2 = 1 if a8 in (1, 7) else -1
        k2 = 1 if (v2 & 1) == 0 else a2

    if odd == 1:
        return sign * k2

    return sign * k2 * ref_jacobi(a, odd)


def complex_close(z: complex, w: complex, tol: float = 1e-12) -> bool:
    return abs(z - w) <= tol


def assert_complex_almost_equal(
    tc: unittest.TestCase, z: complex, w: complex, tol: float = 1e-12, msg: str = "",
):
    tc.assertTrue(complex_close(z, w, tol), msg or f"{z!r} != {w!r} within tol={tol}")


# ------------------------------- Test cases --------------------------------

class TestCoprimes(unittest.TestCase):
    def test_coprimes_exists(self):
        self.assertTrue(hasattr(UT, "coprimes"), "Updated module must define coprimes(n)")

    def test_coprimes_rejects_nonpositive(self):
        for n in (0, -1, -10):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    tuple(UT.coprimes(n))

    def test_coprimes_basic_small(self):
        # n = 1 special-case: only 0 in [0,1); gcd(0,1)=1
        self.assertEqual(tuple(UT.coprimes(1)), (0,))

        for n in range(2, 201):
            got = tuple(UT.coprimes(n))
            with self.subTest(n=n):
                # increasing order, unique
                self.assertEqual(got, tuple(sorted(set(got))))
                self.assertTrue(all(0 <= k < n for k in got))
                self.assertTrue(all(math.gcd(k, n) == 1 for k in got))
                # 0 should not appear for n>1
                self.assertNotIn(0, got)
                # size should match totient
                self.assertEqual(len(got), ref_totient(n))
                # include 1 always
                self.assertIn(1, got)

    def test_coprimes_random_medium(self):
        rng = random.Random(0xC0F1A5E5)
        for _ in range(120):
            # keep sizes moderate so materializing the iterator is safe
            n = rng.randrange(2, 50_000)
            got = tuple(UT.coprimes(n))
            with self.subTest(n=n):
                # exact set equality against gcd-based definition
                expected = tuple(k for k in range(n) if math.gcd(k, n) == 1)
                self.assertEqual(got, expected)

class TestEGCD(unittest.TestCase):
    def test_egcd_known_values(self):
        cases = [
            (0, 0),
            (0, 5),
            (5, 0),
            (240, 46),
            (46, 240),
            (-240, 46),
            (240, -46),
            (-240, -46),
            (1, 1),
            (17, 31),
        ]
        for a, b in cases:
            with self.subTest(a=a, b=b):
                d, x, y = UT.egcd(a, b)
                self.assertIsInstance(d, int)
                self.assertIsInstance(x, int)
                self.assertIsInstance(y, int)
                self.assertGreaterEqual(d, 0)
                self.assertEqual(d, math.gcd(a, b))
                self.assertEqual(a * x + b * y, d)

    def test_egcd_random(self):
        rng = random.Random(0xE6CD)
        for _ in range(500):
            a = rng.randrange(-10**9, 10**9)
            b = rng.randrange(-10**9, 10**9)
            with self.subTest(a=a, b=b):
                d, x, y = UT.egcd(a, b)
                self.assertEqual(d, math.gcd(a, b))
                self.assertEqual(a * x + b * y, d)


class TestCRT(unittest.TestCase):
    def test_crt_empty_system(self):
        # reduce() with initial (0,1) yields x=0 for empty input.
        self.assertEqual(UT.crt([]), 0)

    def test_crt_two_congruences_examples(self):
        # Coprime moduli
        self.assertEqual(UT.crt([(2, 3), (3, 5)]), 8)  # 8 ≡2 (mod3), ≡3 (mod5)
        # Non-coprime but consistent
        self.assertEqual(UT.crt([(1, 2), (3, 4)]), 3)  # x ≡1 (mod2), x≡3 (mod4)
        # Non-coprime and inconsistent
        self.assertIsNone(UT.crt([(1, 2), (0, 4)]))    # no x ≡1 (mod2) and ≡0 (mod4)

    def test_crt_matches_reference_random_small(self):
        rng = random.Random(0xC12A7)
        for _ in range(400):
            k = rng.randrange(1, 6)
            moduli = [rng.randrange(1, 80) for __ in range(k)]
            residues = [rng.randrange(-200, 200) for __ in range(k)]
            # normalize to positive moduli
            moduli = [m if m > 0 else -m for m in moduli]
            # Avoid modulus 0 entirely (undefined)
            moduli = [m if m != 0 else 1 for m in moduli]

            with self.subTest(residues=residues, moduli=moduli):
                got = UT.crt(zip(residues, moduli))
                ref = ref_crt([r % m for r, m in zip(residues, moduli)], moduli)
                self.assertEqual(got, ref)
                if got is not None:
                    for a, m in zip(residues, moduli):
                        self.assertEqual(got % m, a % m)

    def test_crt_solution_is_mod_lcm(self):
        # When solvable, solution is unique modulo lcm of moduli.
        congruences = [(1, 2), (2, 3), (3, 5)]
        x = UT.crt(congruences)
        self.assertIsNotNone(x)
        M = math.lcm(*(m for _, m in congruences))
        for t in range(-3, 4):
            y = x + t * M
            self.assertTrue(all(y % m == a % m for a, m in congruences))


class TestMultiplicativeOrder(unittest.TestCase):
    def test_raises_when_not_unit(self):
        for a, m in [(2, 4), (6, 9), (0, 7), (10, 15)]:
            with self.subTest(a=a, m=m):
                if math.gcd(a, m) != 1:
                    with self.assertRaises(ValueError):
                        UT.multiplicative_order(a, m)

    def test_mod_one(self):
        # In Z/1Z, the unit group is trivial; order is 1.
        self.assertEqual(UT.multiplicative_order(12345, 1), 1)
        self.assertEqual(UT.multiplicative_order(0, 1), 1)

    def test_matches_bruteforce_random(self):
        rng = random.Random(0x0FD3E)
        for _ in range(250):
            mod = rng.randrange(2, 2000)
            # choose a random unit
            a = rng.randrange(1, mod)
            if math.gcd(a, mod) != 1:
                continue
            with self.subTest(a=a, mod=mod):
                got = UT.multiplicative_order(a, mod)
                ref = ref_multiplicative_order_bruteforce(a, mod)
                self.assertEqual(got, ref)
                self.assertEqual(pow(a, got, mod), 1)
                # minimality check: no smaller exponent works
                for k in range(1, got):
                    if pow(a, k, mod) == 1:
                        self.fail(
                            f"order not minimal for a={a}, mod={mod}"
                            f"found smaller k={k}"
                        )

    def test_order_divides_carmichael(self):
        rng = random.Random(0xB0A0D)
        for _ in range(200):
            mod = rng.randrange(2, 5000)
            a = rng.randrange(1, mod)
            if math.gcd(a, mod) != 1:
                continue
            ord_ = UT.multiplicative_order(a, mod)
            lam = ref_carmichael_lambda(mod)
            with self.subTest(a=a, mod=mod):
                self.assertEqual(lam % ord_, 0)


class TestPrimitiveRoot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Make Bach's Las Vegas primitive-root search
        # deterministic for test reproducibility.
        cls._bach_rng = random.Random(0xBACA1234)

        def _det_randbelow(n: int) -> int:
            # secrets.randbelow(n) returns 0 <= r < n
            return cls._bach_rng.randrange(n)

        cls._randbelow_patch = unittest.mock.patch.object(
            UT.secrets, "randbelow", side_effect=_det_randbelow)
        cls._randbelow_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls._randbelow_patch.stop()

    def ref_has_primitive_root(self, n: int) -> bool:
        """n has primitive root iff n in {2,4,p^k,2p^k} for odd prime p."""
        if n < 0:
            n = -n
        if n in (2, 4):
            return True
        if n <= 1:
            return False
        pf = ref_factorization_trial(n)
        # check n = p^k (odd) or n = 2*p^k with odd p
        if len(pf) == 1:
            p = next(iter(pf))
            return (p & 1) == 1
        if len(pf) == 2 and pf.get(2, 0) == 1:
            # exactly one factor 2
            odd_primes = [p for p in pf if p != 2]
            return len(odd_primes) == 1 and (odd_primes[0] & 1) == 1
        return False

    def test_raises_for_small_modulus(self):
        for n in (-1, 0, 1):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    UT.primitive_root(n)

    def test_returns_none_when_no_primitive_root(self):
        # Representative moduli with no primitive root (unit group not cyclic)
        for n in (8, 12, 15, 16, 20, 21, 24, 28, 30, 32, 36, 40):
            with self.subTest(n=n):
                self.assertIsNone(UT.primitive_root(n))

    def test_primitive_root_correctness_for_supported_moduli(self):
        # Test a spread across primes, prime powers, and twice prime powers
        candidates = []
        primes = [p for p in range(3, 200) if ref_is_prime_small(p)]
        for p in primes[::7]:
            candidates.append(p)
            candidates.append(p * p)
            candidates.append(p ** 3)
            candidates.append(2 * p)
            candidates.append(2 * p * p)

        # Deduplicate and remove too-large values
        candidates = sorted({n for n in candidates if 2 <= n <= 20000})

        for n in candidates:
            with self.subTest(n=n):
                should_exist = self.ref_has_primitive_root(n)
                g = UT.primitive_root(n)
                if not should_exist:
                    self.assertIsNone(g)
                    continue
                self.assertIsNotNone(g, f"expected primitive root for n={n}")
                g = int(g)  # type sanity
                self.assertEqual(math.gcd(g, n), 1)
                phi = ref_totient(n)
                self.assertEqual(pow(g, phi, n), 1)
                # generator test: for each prime q | phi, g^(phi/q) != 1
                pf_phi = ref_factorization_trial(phi)
                for q in pf_phi:
                    self.assertNotEqual(pow(g, phi // q, n), 1, f"not primitive: q={q}")

    def test_negative_modulus_returns_valid_root(self):
        # primitive_root(n) reduces negative n to |n|; the *value* may vary
        # due to randomized search, so we only assert correctness properties.
        for n in (3, 5, 7, 9, 10, 14, 18):
            with self.subTest(n=n):
                g_pos = UT.primitive_root(n)
                g_neg = UT.primitive_root(-n)
                self.assertIsNotNone(g_pos)
                self.assertIsNotNone(g_neg)
                phi = ref_totient(n)
                for g in (int(g_pos), int(g_neg)):
                    self.assertEqual(math.gcd(g, n), 1)
                    self.assertEqual(pow(g, phi, n), 1)
                    for q in ref_factorization_trial(phi):
                        self.assertNotEqual(pow(g, phi // q, n), 1)


class TestLegendreJacobiKronecker(unittest.TestCase):
    def test_legendre_raises_on_invalid_p(self):
        for p in (0, 1, 2, 4, 9, -7):
            with self.subTest(p=p):
                with self.assertRaises(ValueError):
                    UT.legendre(3, p)

    def test_legendre_matches_reference_for_primes(self):
        primes = [p for p in range(3, 500) if ref_is_prime_small(p)]
        rng = random.Random(0x1E63AD23)
        for p in primes[::3]:
            for _ in range(25):
                a = rng.randrange(-5000, 5000)
                with self.subTest(a=a, p=p):
                    self.assertEqual(UT.legendre(a, p), ref_legendre(a, p))

        # Exercise the "a >= 60" branch explicitly.
        p = 101
        for a in (60, 61, 1234, -9999):
            with self.subTest(a=a, p=p):
                self.assertEqual(UT.legendre(a, p), ref_legendre(a, p))

    def test_jacobi_raises_on_even_or_nonpositive_n(self):
        for n in (-9, -1, 0, 2, 4, 10):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    UT.jacobi(3, n)

    def test_jacobi_matches_reference_small_range(self):
        rng = random.Random(0xAC0B1)
        for _ in range(800):
            n = rng.randrange(1, 50000)
            n |= 1  # make odd
            a = rng.randrange(-200000, 200000)
            with self.subTest(a=a, n=n):
                self.assertEqual(UT.jacobi(a, n), ref_jacobi(a, n))

    def test_jacobi_multiplicative_in_numerator(self):
        rng = random.Random(0xAA17)
        for _ in range(400):
            n = rng.randrange(3, 50000) | 1
            a = rng.randrange(-5000, 5000)
            b = rng.randrange(-5000, 5000)
            with self.subTest(a=a, b=b, n=n):
                left = UT.jacobi(a * b, n)
                right = UT.jacobi(a, n) * UT.jacobi(b, n)
                self.assertEqual(left, right)

    def test_quadratic_reciprocity_consistency(self):
        # For odd positive a,n with gcd(a,n)=1:
        #   (a|n)(n|a) = (-1)^(((a-1)/2)*((n-1)/2))
        rng = random.Random(0xBEE1F00C)
        for _ in range(350):
            a = (rng.randrange(3, 20000) | 1)
            n = (rng.randrange(3, 20000) | 1)
            if math.gcd(a, n) != 1:
                continue
            with self.subTest(a=a, n=n):
                lhs = UT.jacobi(a, n) * UT.jacobi(n, a)
                exp = ((a - 1) // 2) * ((n - 1) // 2)
                rhs = -1 if (exp & 1) else 1
                self.assertEqual(lhs, rhs)

    def test_kronecker_special_cases_and_matches_reference(self):
        cases = [
            (1, 0), (-1, 0), (2, 0), (0, 0),
            (3, 1), (3, -1), (-3, -1),
            (5, 2), (5, 4), (5, 8),
            (7, -8), (-7, -8),
        ]
        for a, n in cases:
            with self.subTest(a=a, n=n):
                self.assertEqual(UT.kronecker(a, n), ref_kronecker(a, n))

        rng = random.Random(0xABCD1234)
        for _ in range(800):
            a = rng.randrange(-200000, 200000)
            n = rng.randrange(-50000, 50000)
            if n == 0:
                continue
            with self.subTest(a=a, n=n):
                self.assertEqual(UT.kronecker(a, n), ref_kronecker(a, n))

    def test_kronecker_agrees_with_jacobi_for_odd_positive_n(self):
        rng = random.Random(0x0DD)
        for _ in range(600):
            n = rng.randrange(1, 50000) | 1
            a = rng.randrange(-100000, 100000)
            with self.subTest(a=a, n=n):
                self.assertEqual(UT.kronecker(a, n), UT.jacobi(a, n))


class TestDirichletCharacter(unittest.TestCase):
    def test_invalid_parameters(self):
        with self.assertRaises(ZeroDivisionError):
            UT.dirichlet_character(0, 1)
        with self.assertRaises(ValueError):
            UT.dirichlet_character(10, 2)  # gcd(10,2)!=1

    def test_negative_modulus_equivalence(self):
        for m in (3, 5, 7, 12):
            chi_pos = UT.dirichlet_character(m, 1)
            chi_neg = UT.dirichlet_character(-m, 1)
            for n in range(-20, 21):
                with self.subTest(m=m, n=n):
                    self.assertEqual(chi_pos(n), chi_neg(n))

    def test_modulus_one_trivial(self):
        chi = UT.dirichlet_character(1, 1)
        for n in range(-10, 11):
            self.assertEqual(chi(n), 1)

    def test_principal_character_is_indicator_of_coprime(self):
        rng = random.Random(0x9A17C)
        for _ in range(200):
            m = rng.randrange(2, 500)
            chi0 = UT.dirichlet_character(m, 1)
            for __ in range(20):
                n = rng.randrange(-2000, 2000)
                with self.subTest(m=m, n=n):
                    expected = 0 if math.gcd(m, n) != 1 else 1
                    got = chi0(n)
                    # got may be complex(1+0j) or int; accept both
                    if isinstance(got, complex):
                        assert_complex_almost_equal(self, got, complex(expected, 0))
                    else:
                        self.assertEqual(got, expected)

    def test_character_axioms_periodicity_and_multiplicativity(self):
        rng = random.Random(0xD1A1C7)
        for _ in range(120):
            # choose modest m to keep evaluation cheap and avoid pathological huge groups
            m = rng.randrange(2, 400)
            # choose a random unit k != 0 mod m
            ks = [k for k in range(1, m) if math.gcd(k, m) == 1]
            if not ks:
                continue
            k = rng.choice(ks)
            chi = UT.dirichlet_character(m, k)

            for __ in range(40):
                a = rng.randrange(-1000, 1000)
                b = rng.randrange(-1000, 1000)
                # Periodicity
                with self.subTest(m=m, k=k, a=a):
                    v1 = chi(a)
                    v2 = chi(a + m)
                    if isinstance(v1, complex) or isinstance(v2, complex):
                        assert_complex_almost_equal(self, complex(v1), complex(v2))
                    else:
                        self.assertEqual(v1, v2)

                # Multiplicativity (complete multiplicativity with 0-handling)
                with self.subTest(m=m, k=k, a=a, b=b):
                    left = chi(a * b)
                    right = chi(a) * chi(b)
                    assert_complex_almost_equal(
                        self, complex(left), complex(right), tol=1e-10)

                # 0 if not coprime
                with self.subTest(m=m, k=k, a=a):
                    if math.gcd(a, m) != 1:
                        val = chi(a)
                        assert_complex_almost_equal(self, complex(val), 0j)

            # Unit-circle magnitude when nonzero
            for n in range(0, m):
                val = chi(n)
                if complex(val) != 0j:
                    self.assertLessEqual(abs(abs(complex(val)) - 1.0), 1e-10)

    def test_nonprincipal_sum_to_zero_small_moduli(self):
        # Orthogonality: for nonprincipal chi, sum_{n=0}^{m-1} chi(n) = 0
        # We test this for a selection of small moduli.
        for m in range(2, 60):
            units = [k for k in range(1, m) if math.gcd(k, m) == 1]
            if len(units) <= 1:
                continue
            for k in units:
                if k == 1:
                    continue
                chi = UT.dirichlet_character(m, k)
                s = sum(complex(chi(n)) for n in range(m))
                with self.subTest(m=m, k=k):
                    self.assertLessEqual(abs(s), 1e-9)

    def test_large_power_of_two_modulus_crosses_log_cache_threshold(self):
        """
        The implementation uses a cached log-table for mod < 10000 and falls back
        to discrete_log for mod >= 10000. To cover the latter deterministically,
        we evaluate a character modulo 2^14 = 16384.
        """
        m = 1 << 14  # 16384 >= 10000
        # choose a unit k; 1 is principal, pick a different odd unit
        k = 3
        chi = UT.dirichlet_character(m, k)

        # Basic axioms on a small sample
        for n in [1, 3, 5, 7, 9, 11, 13, 15, -1, -3, 0, 2, 4, 6, 8]:
            with self.subTest(n=n):
                val = complex(chi(n))
                if math.gcd(n, m) != 1:
                    self.assertEqual(val, 0j)
                else:
                    self.assertLessEqual(abs(abs(val) - 1.0), 1e-9)
                    # periodicity spot-check
                    assert_complex_almost_equal(self, val, complex(chi(n + m)), tol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
