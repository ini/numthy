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

from typing import Sequence, Tuple


# ----------------------------- Module loading -----------------------------

def _load_under_test():
    """
    Load the module under test with a bias toward a local file next to this test.

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

def poly_eval(coeffs: Sequence[int], x: int) -> int:
    """Evaluate polynomial with integer coeffs a0..ad at integer x (Horner)."""
    acc = 0
    for a in reversed(coeffs):
        acc = a + acc * x
    return acc


def brute_hensel_solutions(coeffs: Sequence[int], p: int, k: int) -> Tuple[int, ...]:
    """Brute-force solutions to f(x) ≡ 0 (mod p^k). Only for small p^k."""
    m = p ** k
    return tuple(x for x in range(m) if poly_eval(coeffs, x) % m == 0)


def brute_discrete_log(a: int, b: int, mod: int) -> int | None:
    """
    Brute-force smallest x >= 0 such that b^x ≡ a (mod mod).
    Works for small mod (<= ~5000) and is used as an oracle in randomized tests.
    """
    if mod == 0:
        raise ZeroDivisionError("mod=0 is invalid")
    m = abs(mod)
    a %= m
    b %= m
    if a == 1 or m == 1:
        return 0

    # Iterate powers until repetition; at most m states.
    seen = set()
    x = 0
    cur = 1 % m
    while cur not in seen and x <= m + 5:
        if cur == a:
            return x
        seen.add(cur)
        x += 1
        cur = (cur * b) % m
    return None


def brute_nth_roots(n: int, k: int, mod: int) -> Tuple[int, ...]:
    """Brute-force all x in [0,|mod|) s.t. x^k ≡ n (mod |mod|)."""
    if k <= 0:
        raise ValueError("k must be positive")
    m = abs(mod)
    if m == 0:
        raise ValueError("mod must be nonzero")
    n %= m
    return tuple(x for x in range(m) if pow(x, k, m) == n)


def brute_polynomial_roots(coeffs: Sequence[int], mod: int) -> Tuple[int, ...]:
    """Brute-force all x in [0,|mod|) s.t. f(x) ≡ 0 (mod |mod|)."""
    m = abs(mod)
    if m == 0:
        raise ValueError("mod must be nonzero")
    coeffs = list(coeffs)
    if not coeffs:
        return tuple(range(m))
    return tuple(x for x in range(m) if poly_eval(coeffs, x) % m == 0)


def is_probable_prime_64(n: int) -> bool:
    """
    Deterministic Miller-Rabin for 64-bit integers.
    Uses the well-known 7-base set for n < 2^64:
      a in {2, 325, 9375, 28178, 450775, 9780504, 1795265022}
    """
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in small_primes:
        return True
    if any(n % p == 0 for p in small_primes):
        return False

    d = n - 1
    s = (d & -d).bit_length() - 1
    d >>= s

    def check(a: int) -> bool:
        a %= n
        if a == 0:
            return True
        x = pow(a, d, n)
        if x in (1, n - 1):
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False

    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if not check(a):
            return False
    return True


def find_safe_prime_for_pollard() -> Tuple[int, int]:
    """
    Find (q, p) with q prime, p = 2q+1 prime, and q > 2^32 (so q.bit_length() > 32).
    Deterministic and fast for 64-bit range.
    """
    q = (1 << 32) + 3  # odd start just above 2^32
    if (q & 1) == 0:
        q += 1
    # search a bounded window; safe primes in this range are plentiful
    for _ in range(20000):
        if is_probable_prime_64(q):
            p = 2 * q + 1
            if is_probable_prime_64(p):
                return q, p
        q += 2
    raise RuntimeError("Failed to find a 64-bit safe prime in search window.")


def find_generator_for_safe_prime(p: int, q: int) -> int:
    """
    For safe prime p=2q+1, the multiplicative group has order 2q.
    g is a generator iff g^2 != 1 (mod p) and g^q != 1 (mod p).
    """
    for g in range(2, 5000):  # should find quickly
        if pow(g, 2, p) != 1 and pow(g, q, p) != 1:
            return g
    for g in range(5000, p):
        if pow(g, 2, p) != 1 and pow(g, q, p) != 1:
            return g
    raise RuntimeError("No generator found (unexpected for safe prime).")


# ------------------------------- Test cases --------------------------------

class TestAPISurface(unittest.TestCase):
    def test_exports_exist(self):
        required = ("hensel", "polynomial_roots", "discrete_log", "nth_roots")
        for name in required:
            with self.subTest(name=name):
                self.assertTrue(hasattr(UT, name), f"Missing expected API: {name}")


class TestHensel(unittest.TestCase):
    def test_rejects_nonprime_p(self):
        for p in (0, 1, 4, 9, 15, -7):
            with self.subTest(p=p):
                with self.assertRaises(ValueError):
                    UT.hensel([0, 1], p, 1)

    def test_rejects_k_lt_1(self):
        with self.assertRaises(ValueError):
            UT.hensel([0, 1], 3, 0)

    def test_k_equals_1_matches_bruteforce(self):
        rng = random.Random(0x51A0E1)  # deterministic
        for _ in range(120):
            p = rng.choice([3, 5, 7, 11, 13, 17, 19])
            deg = rng.randrange(1, 5)
            coeffs = [rng.randrange(-50, 51) for __ in range(deg + 1)]
            with self.subTest(p=p, coeffs=coeffs):
                got = set(UT.hensel(coeffs, p, 1))
                brute = set(brute_hensel_solutions(coeffs, p, 1))
                self.assertEqual(got, brute)
                self.assertTrue(all(0 <= x < p for x in got))

    def test_simple_roots_unique_lift_quadratic(self):
        # f(x)=x^2-2 over p=7 has roots 3,4; derivative 2x not 0 mod 7 at these roots.
        p = 7
        coeffs = [-2, 0, 1]  # x^2 - 2
        roots_p = set(UT.hensel(coeffs, p, 1))
        self.assertEqual(roots_p, {3, 4})

        for k in (2, 3, 4, 5):
            with self.subTest(k=k):
                roots = UT.hensel(coeffs, p, k)
                m = p ** k
                # exactly 2 lifts (simple roots lift uniquely)
                self.assertEqual(len(set(roots)), 2)
                for r in roots:
                    self.assertEqual(poly_eval(coeffs, r) % m, 0)
                    self.assertTrue(0 <= r < m)

        # If we provide only one initial root, only one lift should be returned
        r_only = UT.hensel(coeffs, p, 5, initial=[3])
        self.assertEqual(len(set(r_only)), 1)
        self.assertEqual(poly_eval(coeffs, r_only[0]) % (p ** 5), 0)

    def test_multiple_root_lifting_x_squared(self):
        # f(x)=x^2 has multiple root x=0 mod p.
        p = 5
        coeffs = [0, 0, 1]  # x^2

        for k in (1, 2, 3, 4):
            with self.subTest(k=k):
                roots = set(UT.hensel(coeffs, p, k))
                brute = set(brute_hensel_solutions(coeffs, p, k))
                self.assertEqual(roots, brute)
                m = p ** k
                for r in roots:
                    self.assertEqual((r * r) % m, 0)

        # For k=3, solutions are multiples of p^2: 0, 25, 50, 75, 100
        self.assertEqual(set(UT.hensel(coeffs, p, 3)), {0, 25, 50, 75, 100})

    def test_failure_to_lift_multiple_root(self):
        # f(x)=x^2 - p: x=0 is a root mod p but does not lift to mod p^2
        p = 7
        coeffs = [-p, 0, 1]  # x^2 - 7
        roots_mod_p = set(UT.hensel(coeffs, p, 1))
        self.assertEqual(roots_mod_p, {0})

        roots_mod_p2 = UT.hensel(coeffs, p, 2)
        self.assertEqual(tuple(roots_mod_p2), ())


class TestPolynomialRoots(unittest.TestCase):
    def test_mod_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            UT.polynomial_roots([1, 2, 3], 0)

    def test_mod_one_returns_zero(self):
        self.assertEqual(UT.polynomial_roots([5, 2], 1), (0,))

    def test_empty_coeffs_all_residues(self):
        self.assertEqual(UT.polynomial_roots([], 8), tuple(range(8)))

    def test_zero_polynomial_all_residues(self):
        self.assertEqual(UT.polynomial_roots([0, 0, 0], 9), tuple(range(9)))
        self.assertEqual(UT.polynomial_roots([9, 18, 27], 9), tuple(range(9)))

    def test_constant_nonzero_no_roots(self):
        self.assertEqual(UT.polynomial_roots([1, 0, 0], 7), ())
        self.assertEqual(UT.polynomial_roots([6], 7), ())

    def test_trailing_zero_coeffs_ignored(self):
        coeffs = [0, 0, 1, 0, 0]  # x^2
        roots = UT.polynomial_roots(coeffs, 13)
        ref = brute_polynomial_roots([0, 0, 1], 13)
        self.assertEqual(set(roots), set(ref))

    def test_negative_modulus_equivalence(self):
        rng = random.Random(0xB0A17)
        for _ in range(200):
            mod = rng.randrange(2, 200)
            deg = rng.randrange(0, 5)
            coeffs = [rng.randrange(-200, 200) for __ in range(deg + 1)]
            with self.subTest(mod=mod, coeffs=coeffs):
                self.assertEqual(
                    set(UT.polynomial_roots(coeffs, mod)),
                    set(UT.polynomial_roots(coeffs, -mod)),
                )

    def test_prime_modulus_matches_bruteforce(self):
        rng = random.Random(0xF00D)
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for _ in range(200):
            p = rng.choice(primes)
            deg = rng.randrange(0, 5)
            coeffs = [rng.randrange(-50, 50) for __ in range(deg + 1)]
            with self.subTest(p=p, coeffs=coeffs):
                got = set(UT.polynomial_roots(coeffs, p))
                ref = set(brute_polynomial_roots(coeffs, p))
                self.assertEqual(got, ref)

    def test_prime_power_matches_bruteforce(self):
        cases = [
            (3**2, [1, 0, 1]),          # x^2 + 1
            (3**3, [0, 0, 1]),          # x^2
            (5**2, [-1, 0, 1]),         # x^2 - 1
            (7**2, [-7, 0, 1]),         # x^2 - 7 (no lift)
            (2**5, [0, 0, 1]),          # x^2
            (2**6, [1, 1, 1]),          # x^2 + x + 1
        ]
        for mod, coeffs in cases:
            with self.subTest(mod=mod, coeffs=coeffs):
                got = set(UT.polynomial_roots(coeffs, mod))
                ref = set(brute_polynomial_roots(coeffs, mod))
                self.assertEqual(got, ref)

    def test_composite_modulus_matches_bruteforce(self):
        rng = random.Random(0xC0DE)
        for _ in range(200):
            mod = rng.randrange(2, 160)
            deg = rng.randrange(0, 4)
            coeffs = [rng.randrange(-40, 40) for __ in range(deg + 1)]
            with self.subTest(mod=mod, coeffs=coeffs):
                got = set(UT.polynomial_roots(coeffs, mod))
                ref = set(brute_polynomial_roots(coeffs, mod))
                self.assertEqual(got, ref)

    def test_composite_modulus_crt_combination(self):
        mod = 3**2 * 5
        coeffs = [-1, 0, 1]  # x^2 - 1
        got = set(UT.polynomial_roots(coeffs, mod))
        ref = set(brute_polynomial_roots(coeffs, mod))
        self.assertEqual(got, ref)

    def test_no_roots_when_prime_power_has_none(self):
        mod = 3 * 5
        coeffs = [1, 0, 1]  # x^2 + 1 has no roots mod 3
        self.assertEqual(UT.polynomial_roots(coeffs, mod), ())

    def test_outputs_are_roots_and_unique(self):
        rng = random.Random(0xACE)
        for _ in range(200):
            mod = rng.randrange(2, 200)
            deg = rng.randrange(0, 5)
            coeffs = [rng.randrange(-80, 80) for __ in range(deg + 1)]
            roots = UT.polynomial_roots(coeffs, mod)
            with self.subTest(mod=mod, coeffs=coeffs):
                m = abs(mod)
                self.assertEqual(len(set(x % m for x in roots)), len(roots))
                self.assertTrue(all(0 <= x < m for x in roots))
                for x in roots:
                    self.assertEqual(poly_eval(coeffs, x) % m, 0)

    def test_large_prime_uses_cantor_zassenhaus_path(self):
        p = 101
        coeffs = [-1, 0, 1]  # x^2 - 1
        with unittest.mock.patch.object(
            UT, "_cantor_zassenhaus_ddf", wraps=UT._cantor_zassenhaus_ddf
        ) as wrapped:
            roots = UT.polynomial_roots(coeffs, p)
            self.assertEqual(set(roots), {1, p - 1})
            self.assertGreaterEqual(wrapped.call_count, 1)

    def test_small_prime_skips_cantor_zassenhaus(self):
        p = 11
        coeffs = [-1, 0, 1]  # x^2 - 1
        with unittest.mock.patch.object(
            UT, "_cantor_zassenhaus_ddf", side_effect=AssertionError("CZ should not run")
        ):
            roots = UT.polynomial_roots(coeffs, p)
            self.assertEqual(set(roots), {1, p - 1})


class TestDiscreteLog(unittest.TestCase):
    def test_mod_one_and_a_one(self):
        self.assertEqual(UT.discrete_log(1, 123, 99991), 0)
        self.assertEqual(UT.discrete_log(456, 123, 1), 0)

    def test_negative_modulus_equivalence(self):
        rng = random.Random(0x0E6A71)
        for _ in range(120):
            mod = rng.randrange(2, 2000)
            a = rng.randrange(0, mod)
            b = rng.randrange(0, mod)
            with self.subTest(a=a, b=b, mod=mod):
                pos = UT.discrete_log(a, b, mod)
                neg = UT.discrete_log(a, b, -mod)
                self.assertEqual(pos, neg)

    def test_matches_bruteforce_random_small(self):
        rng = random.Random(0xD106)
        for _ in range(500):
            mod = rng.randrange(2, 2000)
            a = rng.randrange(0, mod)
            b = rng.randrange(0, mod)
            with self.subTest(a=a, b=b, mod=mod):
                got = UT.discrete_log(a, b, mod)
                ref = brute_discrete_log(a, b, mod)
                self.assertEqual(got, ref)

    def test_gcd_reduction_path_examples(self):
        # 2^x ≡ 0 (mod 8) => x=3
        self.assertEqual(UT.discrete_log(0, 2, 8), 3)
        # 2^x ≡ 4 (mod 8) => x=2
        self.assertEqual(UT.discrete_log(4, 2, 8), 2)
        # no solution
        self.assertIsNone(UT.discrete_log(6, 2, 8))

        # Another gcd-reduction case: 6^x ≡ 12 (mod 18)
        self.assertEqual(UT.discrete_log(12, 6, 18), brute_discrete_log(12, 6, 18))

    def test_power_of_two_modulus_special_branch(self):
        # Exercise p==2 and e>=3 in _discrete_log_mod_prime_power.
        rng = random.Random(0x2AD1C)
        for e in range(3, 15):
            mod = 1 << e
            for _ in range(40):
                base = rng.randrange(1, mod, 2)  # odd => unit
                x = rng.randrange(0, 2000)
                target = pow(base, x, mod)
                with self.subTest(e=e, base=base, x=x):
                    got = UT.discrete_log(target, base, mod)
                    self.assertIsNotNone(got)
                    self.assertEqual(pow(base, got, mod), target)
                    ref = brute_discrete_log(target, base, mod)
                    self.assertEqual(got, ref)

    def test_prime_power_modulus(self):
        # Exercise _multiplicative_order_mod_odd_prime_power
        rng = random.Random(0x1B2C3D)
        for p in (3, 5, 7, 11):
            for e in (2, 3, 4):
                mod = p ** e
                for _ in range(30):
                    base = rng.randrange(1, mod)
                    if math.gcd(base, mod) != 1:
                        continue
                    x = rng.randrange(0, 5000)
                    target = pow(base, x, mod)
                    with self.subTest(p=p, e=e, base=base):
                        got = UT.discrete_log(target, base, mod)
                        ref = brute_discrete_log(target, base, mod)
                        self.assertEqual(got, ref)

    def test_pollard_rho_branch_is_exercised(self):
        """
        Ensure the Pollard-rho discrete log path is exercised:
        _pohlig_hellman_prime_power chooses _pollard_rho_log when the prime-order
        factor p has bit_length() > 32.

        We construct a safe prime modulus P = 2Q + 1 where Q is a 33-bit prime.
        Then the subgroup of order Q triggers the Pollard-rho solver.
        """
        if not hasattr(UT, "_pollard_rho_log"):
            self.skipTest("Internal _pollard_rho_log not present")
        if not hasattr(UT, "secrets"):
            self.skipTest("Module does not expose secrets")

        Q, P = find_safe_prime_for_pollard()
        g = find_generator_for_safe_prime(P, Q)

        # deterministic randomness for Pollard-rho internals
        rng = random.Random(0x4F11A6D)

        def det_randbelow(n: int) -> int:
            return rng.randrange(n)

        patch_secrets = unittest.mock.patch.object(
            UT.secrets, "randbelow", side_effect=det_randbelow)
        patch_pollard = unittest.mock.patch.object(
            UT, "_pollard_rho_log", wraps=UT._pollard_rho_log)

        with patch_secrets, patch_pollard as wrapped:
            x = 123456  # small known exponent
            target = pow(g, x, P)
            got = UT.discrete_log(target, g, P)

            self.assertEqual(got, x)
            self.assertGreaterEqual(wrapped.call_count, 1)


class TestModularRoots(unittest.TestCase):
    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            UT.nth_roots(2, 0, 7)
        with self.assertRaises(ValueError):
            UT.nth_roots(2, -1, 7)

    def test_mod_zero_raises(self):
        with self.assertRaises(Exception):
            UT.nth_roots(1, 2, 0)

    def test_mod_one_returns_zero(self):
        self.assertEqual(UT.nth_roots(123, 5, 1), (0,))
        self.assertEqual(UT.nth_roots(0, 3, 1), (0,))

    def test_negative_modulus_equivalence(self):
        rng = random.Random(0x0D0E6)
        for _ in range(200):
            mod = rng.randrange(2, 800)
            k = rng.randrange(1, 7)
            n = rng.randrange(-2000, 2000)
            with self.subTest(n=n, k=k, mod=mod):
                self.assertEqual(
                    set(UT.nth_roots(n, k, mod)),
                    set(UT.nth_roots(n, k, -mod)),
                )

    def test_prime_modulus_square_roots(self):
        self.assertEqual(UT.nth_roots(3, 2, 7), ())
        roots = UT.nth_roots(2, 2, 7)
        self.assertEqual(set(roots), {3, 4})

    def test_prime_modulus_unique_kth_root_when_gcd_is_one(self):
        # p=11, gcd(3, 10)=1 => unique cube root
        p = 11
        k = 3
        inv = pow(k, -1, p - 1)
        for n in range(p):
            with self.subTest(n=n):
                roots = UT.nth_roots(n, k, p)
                self.assertEqual(len(roots), 1)
                self.assertEqual(roots[0] % p, pow(n, inv, p))

    def test_prime_modulus_multiple_roots_when_gcd_gt_one(self):
        # 4th roots of 1 mod 17 should be 4 solutions
        p = 17
        roots = UT.nth_roots(1, 4, p)
        brute = brute_nth_roots(1, 4, p)
        self.assertEqual(set(roots), set(brute))
        self.assertEqual(len(set(roots)), 4)

    def test_composite_modulus_matches_bruteforce(self):
        rng = random.Random(0xC0FFEE)
        for _ in range(350):
            mod = rng.randrange(2, 800)
            k = rng.randrange(1, 7)
            n = rng.randrange(0, mod)
            with self.subTest(mod=mod, k=k, n=n):
                got = set(UT.nth_roots(n, k, mod))
                ref = set(brute_nth_roots(n, k, mod))
                self.assertEqual(got, ref)
                self.assertTrue(all(0 <= x < mod for x in got))

    def test_prime_power_modulus_matches_bruteforce(self):
        # Exercise Hensel lifting in nth_roots()
        cases = [
            (3**2, 2, 2),
            (3**3, 2, 1),
            (5**3, 2, 4),
            (7**2, 3, 1),
            (11**2, 5, 1),
            (2**6, 2, 0),
            (2**7, 3, 1),
        ]
        for mod, k, n in cases:
            with self.subTest(mod=mod, k=k, n=n):
                got = set(UT.nth_roots(n, k, mod))
                ref = set(brute_nth_roots(n, k, mod))
                self.assertEqual(got, ref)

    def test_all_roots_satisfy_congruence_and_are_unique(self):
        rng = random.Random(0x514A73)  # fixed seed (hex digits only)
        for _ in range(200):
            mod = rng.randrange(2, 1000)
            k = rng.randrange(1, 10)
            n = rng.randrange(-2000, 2000)
            roots = UT.nth_roots(n, k, mod)
            with self.subTest(mod=mod, k=k, n=n):
                m = abs(mod)
                self.assertEqual(len(set(x % m for x in roots)), len(roots))
                for x in roots:
                    self.assertEqual(pow(x, k, m), n % m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
