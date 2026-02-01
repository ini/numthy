# Copyright (c) 2026 Ini Oguntola
# Permission is granted to use, copy, modify, and redistribute this work,
# provided acknowledgement of the original author is retained.
# Supports Python 3.10 or later.

from __future__ import annotations

import bisect
import cmath
import hashlib
import hmac
import inspect
import itertools
import secrets
import sys

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from fractions import Fraction
from functools import cache, lru_cache, partial, reduce
from heapq import heappop, heappush
from math import ceil, fsum, gcd, inf, isfinite, isqrt, lcm, log, prod, sqrt
from operator import mul, xor
from typing import Callable, Collection, Iterator, TypeAlias, TypeVar



########################################################################
########################### Table of Contents ##########################
########################################################################

__version__ = '0.1.0b'

__all__ = [
    'Number', 'Vector', 'Matrix', 'Monomial', 'Polynomial', 'clear_cache',
    # Primes
    'is_prime', 'next_prime', 'random_prime', 'primes', 'count_primes', 'sum_primes',
    # Factorization
    'perfect_power', 'prime_factors', 'prime_factorization', 'divisors',
    # Arithmetic Functions
    'omega', 'big_omega', 'divisor_count', 'divisor_sum', 'divisor_function',
    'partition', 'radical', 'mobius', 'totient', 'carmichael', 'valuation',
    'multiplicative_range',
    # Modular Arithmetic
    'egcd', 'crt', 'coprimes', 'multiplicative_order', 'primitive_root',
    'legendre', 'jacobi', 'kronecker', 'dirichlet_character',
    # Nonlinear Congruences
    'hensel', 'polynomial_roots', 'nth_roots', 'discrete_log',
    # Diophantine Equations
    'bezout', 'cornacchia', 'pell', 'conic', 'pythagorean_triples', 'pillai',
    # Algebraic Systems
    'solve_linear_system', 'solve_polynomial_system',
    # Lattices
    'lll_reduce', 'bkz_reduce', 'closest_vector', 'small_roots',
    # Appendix
    'integers', 'integer_pairs', 'alternating', 'below', 'lower_bound', 'permutation',
    'is_square', 'iroot', 'ilog', 'fibonacci', 'fibonacci_index', 'polygonal',
    'polygonal_index', 'periodic_continued_fraction', 'convergents',  'polynomial',
]

_NoSolutionError = type('_NoSolutionError', (Exception,), {})
_PrecisionError = type('_PrecisionError', (Exception,), {})
_T = TypeVar('T', bound='Number')
Number: TypeAlias = int | float | complex | Fraction
Real: TypeAlias = int | float | Fraction
Vector: TypeAlias = list[_T]
Matrix: TypeAlias = list[list[_T]]
Monomial: TypeAlias = tuple[int, ...]
Polynomial: TypeAlias = dict[Monomial, _T]
singleton = lru_cache(maxsize=1)
small_cache = lru_cache(maxsize=1024)
large_cache = lru_cache(maxsize=1048576)

def clear_cache():
    """
    Clear all caches defined in this module.
    """
    module = sys.modules[__name__]
    for obj in vars(module).values():
        if getattr(obj, '__module__', None) == __name__:
            cache_clear = getattr(obj, 'cache_clear', None)
            if callable(cache_clear):
                cache_clear()



########################################################################
################################ Primes ################################
########################################################################

def is_prime(n: int) -> bool:
    """
    Test if a given integer n is prime.

    Uses a combination of trial division, the Miller-Rabin primality test
    with deterministic bases, or the extra-strong variant of the Baillie-PSW
    primality test (this variant has no known pseudoprimes in any range, and
    has been computationally verified to have no counterexamples for all n < 2^64).

    See: https://www.techneon.com/download/is.prime.32.base.data (MR hash for n < 2^32)
    See: https://miller-rabin.appspot.com (other deterministic MR base sets)
    See: https://ntheory.org/pseudoprimes.html (BPSW verification up to 2^64)

    Parameters
    ----------
    n : int
        Integer to test for primality
    """
    if (n & 1) == 0 or n < 3:  # n is even or n < 3
        return n == 2
    if n < 256:
        return n in _ODD_PRIMES_BELOW_256
    if gcd(n, _PRIMORIAL_ODD_PRIMES_BELOW_256) > 1:
        return False
    if n < 65536:  # n < 256^2, and n coprime to all primes < 256 implies n is prime
        return True

    # Check for Mersenne primes
    if n.bit_length() == (k := n.bit_count()):  # n = 2^k - 1
        return _lucas_lehmer(k)

    # Use deterministic set of Miller-Rabin bases for small n
    if n < 132239:
        return _miller_rabin(n, (814494960528 % n,))
    if n < 4294967296:
        # Use hash-based Miller-Rabin witness table for n < 2^32
        h = (0xAD625B89 * n) >> 24 & 255
        return _miller_rabin(n, _MILLER_RABIN_32_BIT_BASES[h:h+1])
    if n < 55245642489451:
        bases = (2, 141889084524735, 1199124725622454117, 11096072698276303650)
        return _miller_rabin(n, (a % n for a in bases))

    return _baillie_psw(n)  # BPSW has zero known pseudoprimes

def next_prime(n: int) -> int:
    """
    Get the smallest prime number greater than n.

    Parameters
    ----------
    n : int
        Strict lower bound for prime number
    """
    if n < 2:
        return 2

    a = (n + 1) | 1  # next odd number
    while not is_prime(a):
        a += 2

    return a

def random_prime(num_bits: int, *, safe: bool = False) -> int:
    """
    Generate a random prime with the given number of bits.

    Parameters
    ----------
    num_bits : int
        Number of bits in the prime to be generated
    safe : bool
        Whether or not to generate a safe prime
        (i.e. prime q of the form q = 2p + 1, where p is also prime)
    """
    # Handle edge cases
    if safe and num_bits < 3:
        raise ValueError("Safe primes require num_bits >= 3")
    if not safe and num_bits < 2:
        raise ValueError("Primes require num_bits >= 2")
    if not safe and num_bits == 2:
        return secrets.randbelow(2) + 2

    # Precompute bitmask
    k = num_bits - 3 if safe else num_bits - 2  # number of random bits per candidate
    batch_size = max(1, int(0.4 * k))
    top_bit, mask = 1 << (k + 1), (1 << k) - 1

    # Generate batches of random bits and test primality
    while True:
        batch = secrets.randbits(batch_size * k)
        for _ in range(batch_size):
            middle = batch & mask  # all random bits except first/last
            p = top_bit | (middle << 1) | 1  # force first/last bit to 1
            if is_prime(p):
                if safe:
                    if is_prime(q := 2*p + 1):
                        return q
                else:
                    return p

            batch >>= k

def primes(
    *,
    low: int = 2,
    high: int | None = None,
    count: int | None = None,
) -> Iterator[int]:
    """
    Generate at most `count` primes in increasing order within the range `[low, high]`.

    Uses the sieve of Eratosthenes, with a segmented approach for large or
    unbounded ranges.

    Parameters
    ----------
    low : int
        Lower bound for prime numbers
    high : int
        Upper bound for prime numbers (default is infinite)
    count : int
        Maximum number of primes to generate (default is infinite)
    """
    DEFAULT_SIEVE_SIZE, MAX_SIEVE_SIZE = 1000, 100_000_000
    low = max(low, 2)
    high = inf if high is None else high
    count = inf if count is None else count

    # Initial list of small primes to use for the segmented sieve
    small_odd_primes = [
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
        43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    ]
    if low == 2 and count <= 25 and high <= 100:
        yield from (p for p in [2, *small_odd_primes][:count] if p <= high)
        return

    # Generate initial prime
    if low <= 2 <= high and count > 0:
        yield 2
        count -= 1
    elif low > high or count <= 0:
        return

    # Set initial sieve size based on the prime number theorem
    # When `high` is given, sieve on range [low, high]
    # When `count` is given, sieve on range [low, n (log n + log log n)],
    # where n is an upper bound on `π(low) + count`
    if high == count == inf:
        sieve_size = DEFAULT_SIEVE_SIZE
    else:
        n = count + 1.25506 * low / max(log(low), 1)  # Rosser & Schoenfeld bound (1962)
        upper_bound = n * (log(n) + log(log(n)))  # upper bound on the nth prime
        sieve_size = int(min(MAX_SIEVE_SIZE, high - low + 1, upper_bound - low))

    # Generate additional primes
    while low <= high and count > 0:
        # If necessary, extend list of small primes via Bertrand intervals
        while (p := small_odd_primes[-1]) < isqrt(low + sieve_size):
            small_odd_primes.extend(_segmented_eratosthenes(p + 1, p, small_odd_primes))

        # Get new primes with segmented sieve
        new_primes = _segmented_eratosthenes(low, sieve_size, small_odd_primes)
        if count < inf:
            new_primes = tuple(itertools.islice(new_primes, count))
            count -= len(new_primes)

        # Yield new primes
        yield from new_primes

        # Update sieve range
        low += sieve_size
        sieve_size = min(2 * sieve_size, MAX_SIEVE_SIZE, high - low + 1)

def count_primes(x: int) -> int:
    """
    Prime counting function π(x). Returns the number of primes p ≤ x.

    Uses the Lagarias-Miller-Odlyzko (LMO) extension of the Meissel-Lehmer algorithm.

    Parameters
    ----------
    x : int
        Upper bound for prime numbers
    """
    if x < 10000:
        return sum(1 for _ in primes(high=x))

    thresholds = [(1000000, (5, 0.015)), (1000000000, (5, 0.008))]
    k, c = _threshold_select(x, thresholds, default=(15, 0.003))
    return _lmo(x, k=k, c=c)

def sum_primes(
    x: int,
    f: Callable[[int], Number] | None = None,
    f_prefix_sum: Callable[[int], Number] | None = None,
) -> Number:
    """
    Compute F(x) as the sum of f(p) over all primes p ≤ x,
    where f is a completely multiplicative function (by default, f(n) = n).

    Uses a generalized version of the LMO prime counting algorithm.
    Ideally `f()` and `f_prefix_sum()` can be calculated efficiently
    in O(1) time via closed-form expression.

    Parameters
    ----------
    x : int
        Upper bound for prime numbers
    f : Callable(int) -> Number
        Completely multiplicative function f(n),
        where f(1) = 1 and f(ab) = f(a) * f(b) for all a, b > 0
    f_prefix_sum : Callable(int) -> Number
        Function to compute the cumulative sum Σ_{1 ≤ k ≤ n} f(k)
    """
    if f is None and f_prefix_sum is None:
        if x < 10000:
            return sum(primes(high=x))
        else:
            f, f_prefix_sum = _identity, (lambda n: n * (n + 1) // 2)
    elif f is None or f_prefix_sum is None:
        raise ValueError("Both f() and f_prefix_sum() must be provided")

    if x < 10000:
        return sum(f(p) for p in primes(high=x))

    thresholds = [(100000, (5, 0.025)), (1000000, (5, 0.015)), (10000000, (5, 0.01))]
    k, c = _threshold_select(x, thresholds, default=(15, 0.005))
    return _lmo(x, k=k, c=c, f=f, f_prefix_sum=f_prefix_sum)

def _miller_rabin(n: int, bases: Iterable[int] | int = (2,)) -> bool:
    """
    Miller-Rabin primality test over the given bases.

    See: https://www.sciencedirect.com/science/article/pii/0022314X80900840

    Complexity
    ----------
    O(k log³n) for k bases, with worst-case error probability 4⁻ᵏ
    """
    # Write n - 1 as 2^s * d with d odd
    d = n - 1
    s = (d & -d).bit_length() - 1
    d >>= s

    # Perform a Miller-Rabin test for each base sequentially
    return _miller_rabin_worker(n, s, d, bases)

def _miller_rabin_worker(n: int, s: int, d: int, bases: Iterable[int] | int) -> bool:
    """
    Miller-Rabin primality test for n over the given bases,
    where n - 1 = 2^s * d with d odd.

    See: https://www.sciencedirect.com/science/article/pii/0022314X80900840
    """
    # Generate random bases, if specific bases have not been given
    if isinstance(bases, int):
        bases = (secrets.randbelow(n - 3) + 2 for _ in range(bases))

    # Run a Miller-Rabin test for each base
    for a in bases:
        x = pow(a, d, n)
        if x == n - 1 or x == 1:
            continue  # probable prime

        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break  # probable prime
        else:
            return False  # composite

    return True  # All bases passed

def _baillie_psw(n: int) -> bool:
    """
    Baillie-PSW primality test for n. Uses an extra strong Lucas step.

    There are no known counterexamples to this primality test,
    and it has been computationally verified for all n < 2^64.

    See: https://math.dartmouth.edu/~carlp/PDF/paper25.pdf
    See: https://ntheory.org/pseudoprimes.html

    Complexity
    ----------
    O(log³n) time
    """
    # Perform a Miller-Rabin test with base a = 2
    if not _miller_rabin(n, bases=(2,)):
        return False

    # Reject perfect squares
    if is_square(n):
        return False

    # Find a suitable D for the extra-strong Lucas test (D = P^2 - 4Q with Q = 1)
    P = 3
    while jacobi(P*P - 4, n) != -1:
        P += 1

    # Write n + 1 = 2^s * d with d odd
    d = n + 1
    s = (d & -d).bit_length() - 1
    d >>= s

    # Generate the Lucas sequence element V_d(P, Q) via binary Lucas chain 
    P %= n
    V, V_next = P, (P*P - 2) % n  # these represent V_k, V_{k+1}
    for bit in format(d, 'b')[1:]:
        if bit != '0':
            V, V_next = (V * V_next - P) % n, (V_next * V_next - 2) % n
        else:
            V, V_next = (V * V - 2) % n, (V * V_next - P) % n

    # 1st extra-strong condition: U_d = 0 (mod n) and V_d = ± 2 (mod n)
    # Since gcd(D, n) = 1, U_d = 0 (mod n) <=> D * U_d = 2V_{d+1} - PV_d = 0 (mod n)
    if V in (2, n - 2) and (2 * V_next - P * V) % n == 0:
        return True

    # 2nd extra-strong condition: V_{2^r * d} = 0 (mod n) for some 0 <= r < s - 1
    for _ in range(s - 2):
        if V == 0: return True
        V = (V*V - 2) % n

    return s > 1 and V == 0

def _lucas_lehmer(p: int) -> bool:
    """
    Run the Lucas-Lehmer test for Mersenne primes of the form M_p = 2^p - 1.

    Complexity
    ----------
    O(p) multiplications
    """
    if p == 2:
        return True

    # Use trial division to determine if p is prime
    for q in primes(high=isqrt(p)):
        if p % q == 0:
            return False

    # Perform Lucas-Lehmer test
    s, M = 4, (1 << p) - 1
    for _ in range(p - 2):
        s = (s*s - 2) % M

    return s == 0

def _segmented_eratosthenes(
    start: int,
    sieve_size: int,
    odd_primes: Sequence[int],
) -> Iterable[int]:
    """
    Segmented sieve of Eratosthenes.
    Returns odd prime numbers in the range [start, start + sieve_size).
    Expects sorted odd primes up to √(start + sieve_size).

    Complexity
    ----------
    O(n log log n) time and O(n) space for segment of size n
    """
    # Initialize sieve segment
    # Only odd numbers are stored in the sieve (sieve[i] corresponds to start + 2i)
    start, end = start | 1, start + sieve_size
    sieve_size = (end - start + 1) >> 1
    sieve = bytearray(b'\x01') * sieve_size
    zeros = bytearray(b'\x00') * sieve_size

    # Handle small primes where p^2 <= start
    cutoff = bisect.bisect_right(odd_primes, isqrt(start))
    for p in odd_primes[:cutoff]:
        # Find next odd multiple of p >= start
        next_odd_multiple = start + (p - start) % (p + p)

        # Mark multiples of p in the odd sieve
        index = (next_odd_multiple - start) >> 1
        count = (sieve_size - index + p - 1) // p
        sieve[index::p] = zeros[:count]

    # Handle large primes where p^2 > start
    for p in odd_primes[cutoff:]:
        if (p_squared := p * p) >= end:
            break

        # Mark multiples of p in the odd sieve
        index = (p_squared - start) >> 1
        count = (sieve_size - index + p - 1) // p
        sieve[index::p] = zeros[:count]

    return itertools.compress(range(start, start + 2 * sieve_size, 2), sieve)

def _lmo(
    x: int,
    k: int = 15,
    c: float = 0.003,
    f: Callable[[int], Number] | None = None,
    f_prefix_sum: Callable[[int], Number] | None = None,
) -> Number:
    """
    Lagarias-Miller-Odlyzko (LMO) extension of the Meissel-Lehmer algorithm.
    Returns the value of the prime counting function π(x), i.e. the number of
    primes less than or equal to x.

    See: https://www-users.cse.umn.edu/~odlyzko/doc/arch/meissel.lehmer.pdf
    See: https://arxiv.org/pdf/2111.15545

    Also includes a generalized version that calculates the sum F(x) = Σ f(p)
    for all primes p ≤ x, where f is any arbitrary completely multiplicative function.

    The generalized LMO sub-expressions become:

        P₂ = Σ f(p) * [F(x/p) − F(p − 1)] for y < p ≤ sqrt(x)
        φ_f(x, a) = φ_f(x, a - 1) - f(pₐ) * φ_f(x/pₐ, a - 1)
        S₁ = Σ μ(n) f(n) φ_f(x/n, k) over ordinary leaves (n, k)
        S₂ = Σ μ(n) f(n) φ_f(x/n, b) over special leaves (n, b)

    and the generalized Meissel-Lehmer formula becomes:

        F(x) = F(y) - 1 - P₂ + φ_f(x, a) = F(y) - 1 - P₂ + S₁ + S₂.

    Ideally `f()` and `f_prefix_sum()` can be calculated efficiently in O(1) time
    via closed-form expression.

    Complexity
    ----------
    O(x²ᐟ³ / log x) time and O(x¹ᐟ³ log²x) space with hyperparameter
    y = c * x¹ᐟ³ log² x, assuming f() and f_prefix_sum() are O(1).
    """
    if x < 2:
        return 0

    # Set hyperparameter y = cx^(1/3) log^2(x) such that x^(1/3) <= y <= x^(2/5)
    # where y is the upper bound on the small primes that are computed directly
    y = int(c * iroot(x, 3) * (log(x) ** 2))
    y = min(max(y, iroot(x, 3)), iroot(x * x, 5))
    y = max(y, 2)  # we need y >= 2 to use an odd-only sieve starting at y + 1

    # Count primes up to y
    small_primes = tuple(primes(high=y))
    a = len(small_primes)
    F_y = a if f is None else sum(map(f, small_primes))

    # Set number of precomputed stages of special leaf sieving
    k = min(max(k, 1), a)

    # Evaluate the 2nd-order partial sieve function P2(x, a)
    # This is the prefix sum Σ f(n) over all n <= x with exactly 2 prime factors,
    # that are both greater than p_a
    P2 = _lmo_p2(x, y, F_y, small_primes, f)

    # Compute the least prime factor (lpf) and Mobius (μ) functions
    # for integers 1 ... y by iterating over the primes in reverse order
    lpf, mu = [0] * (y + 1), [1] * (y + 1)
    for p in reversed(small_primes):
        mu[p*p::p*p] = [0] * (y // (p*p))
        mu[p::p] = [-value for value in mu[p::p]]
        lpf[p::p] = [p] * (y // p)

    # Sum the leaves in the tree created by either
    # the standard recurrence φ(x, a) = φ(x, a - 1) - φ(x/p_a, a - 1)
    # or the weighted recurrence φ_f(x, a) = φ_f(x, a - 1) - f(p_a) * φ_f(x/p_a, a - 1)
    S1 = _lmo_s1(x, k, mu, small_primes, f, f_prefix_sum)  # sum over ordinary leaves
    S2 = _lmo_s2(x, k, lpf, mu, small_primes, f)  # sum over special leaves

    return F_y - 1 - P2 + S1 + S2

def _lmo_p2(
    x: int,
    y: int,
    F_y: Number,
    small_primes: tuple[int, ...],
    f: Callable[[int], Number] | None = None,
) -> Number:
    """
    Compute P2(x, a) from the LMO algorithm.

    This is the prefix sum Σ f(n) over all n ≤ x with exactly 2 prime factors,
    both greater than p_a.
    """
    sqrt_x = isqrt(x)
    sieve_limit = x // y
    sieve_start = (y + 1) | 1  # round up to odd
    sieve_size = y + (y & 1)  # round up to even

    # Compute a generalized P2(x, a) = sum_{y < p <= sqrt(x)} f(p) * [F(x/p) − F(p − 1)]
    # Find the weighted sum f(p) * F(x/p) for all primes in the interval (y, sqrt(x)]
    # Or equivalently, the sum over all x/p in the inverse interval [sqrt(x), x/y)
    # Also accumulate the sum f(p)^2 for all primes in the interval (y, sqrt(x)]
    P2 = 0
    sum_f2 = 0
    F_sqrt_x = F_y
    F_segment = [F_y]
    for low in range(sieve_start, sieve_limit + 1, sieve_size):
        # Sieve the interval [low, high)
        # Only odd numbers are stored in the sieve (sieve[i] corresponds to low + 2i)
        high = min(low + sieve_size, sieve_limit + 1)
        sieve = _lmo_odd_sieve(low, high - low, small_primes[1:], max_prime=isqrt(high))

        # Get f(t) for t ∈ [low, high)
        if f is not None:
            f2_primes = itertools.compress(range(low, min(high, sqrt_x + 1), 2), sieve)
            sum_f2 += sum(f(p)**2 for p in f2_primes)
            f_segment = [f(low + 2*i) if sieve[i] else 0 for i in range(len(sieve))]
        else:
            f_segment = sieve

        # Calculate prime sums F(t) = sum_{p <= t} f(p) for t ∈ [low, high)
        F_segment = list(itertools.accumulate(f_segment, initial=F_segment[-1]))[1:]
        if low <= sqrt_x < high:
            F_sqrt_x = F_segment[(sqrt_x - low) >> 1]

        # Find all primes p ∈ (y, sqrt(x)] such that low <= x/p < high
        # by similarly sieving the inverse interval (x/high, x/low]
        low_ = (max(x // high, y) + 1) | 1
        high_ = min(x // low, sqrt_x)
        sieve_ = _lmo_odd_sieve(
            low_, high_ - low_ + 1, small_primes[1:], max_prime=isqrt(high_))

        # Accumulate over all x/p in our main interval [low, high)
        for p in itertools.compress(range(low_, high_ + 1, 2), sieve_):
            P2 += F_segment[(x // p - low) >> 1] * (f(p) if f else 1)

    if f is None:
        sum_f2 = F_sqrt_x - F_y

    # Now subtract sum_{y < p <= sqrt(x)} f(p) * F(p − 1)
    # We can use the telescoping identity with a_i = f(p_i), A_i = F(p_i)
    # which is A_i^2 - A_{i-1}^2 = 2 a_i A_{i-1} + a_i^2
    # Over y < p_i <= sqrt(x), the sum Σ f(p) * F(p − 1) = Σ a_i A_{i-1}
    # becomes 1/2 [F(sqrt(x))^2 - F(y)^2 - Σ f(p)^2]
    is_int = isinstance(sum_f2, int)
    double_count_sum = F_sqrt_x*F_sqrt_x - F_y*F_y - sum_f2
    double_count_sum = double_count_sum // 2 if is_int else double_count_sum / 2

    return P2 - double_count_sum

def _lmo_s1(
    x: int,
    k: int,
    mu: list[int],
    small_primes: tuple[int, ...],
    f: Callable[[int], Number] | None = None,
    f_prefix_sum: Callable[[int], Number] | None = None,
) -> Number:
    """
    Calculate the S₁ portion of the LMO algorithm.

    This is the sum over "ordinary leaves" (i.e. of the form ± φ(x/n, k) with n <= y)
    in the tree created by the standard recurrence φ(x, a) = φ(x, a-1) - φ(x/pₐ, a-1),
    or the weighted recurrence φ_f(x, a) = φ_f(x, a-1) - f(pₐ) * φ_f(x/pₐ, a-1).
    """
    if f is None:
        phi = partial(_phi_prime_count, small_primes=small_primes[:k])
    elif f == _identity:
        phi = partial(_phi_prime_sum, small_primes=small_primes[:k])
    else:
        phi = lambda x, a: f_prefix_sum(x) if a == 0 else (
            phi(x, a - 1) - f(p := small_primes[a - 1]) * phi(x // p, a - 1))

    S1 = phi(x, k)
    a, y = len(small_primes), len(mu) - 1
    leaves = [(i + 1, small_primes[i]) for i in range(k, a)]
    while leaves:
        b, n = leaves.pop()
        S1 += mu[n] * phi(x // n, k) * (f(n) if f else 1)
        for i in range(b, a):
            m = n * small_primes[i]
            if m > y: break
            leaves.append((i + 1, m))

    return S1

def _lmo_s2(
    x: int,
    k: int,
    lpf: list[int],
    mu: list[int],
    small_primes: tuple[int, ...],
    f: Callable[[int], Number] | None = None,
) -> Number:
    """
    Calculate the S₂ portion of the LMO algorithm.

    This is the sum over "special leaves" (i.e. of the form ± φ(x/n, b) with n > y)
    in the tree created by the standard recurrence φ(x, a) = φ(x, a-1) - φ(x/pₐ, a-1),
    or the weighted recurrence φ_f(x, a) = φ_f(x, a-1) - f(pₐ) * φ_f(x/pₐ, a-1).
    """
    S2 = 0
    a, y = len(small_primes), len(mu) - 1
    if k >= a: return 0
    phi = [0] * a
    sieve_limit = x // y
    sieve_size = isqrt(sieve_limit) - 1
    sieve_size = 2**(sieve_size.bit_length())  # round up to next power of 2
    tree_size = sieve_size // 2

    for low in range(1, sieve_limit, sieve_size):
        # Sieve the segment [low, high) with the first k primes
        # Only odd numbers are stored in the sieve (sieve[i] corresponds to low + 2i)
        # sieve[i] is True if and only if low + 2i is coprime to the first k primes
        high = min(low + sieve_size, sieve_limit)
        odd_sieve = _lmo_odd_sieve(low, sieve_size, small_primes[1:k])

        # Initialize a Binary Indexed Tree
        if f is None:
            tree = _fenwick_tree_init(odd_sieve)
        else:
            values = [f(low + 2*i) if s else 0 for i, s in enumerate(odd_sieve)]
            tree = _fenwick_tree_init(values)

        # Sieve the segment [low, high) with the remaining primes
        # Any part of the sieve or tree outside this range is ignored
        for b in range(k, a):
            p = small_primes[b]
            min_m = max(x // (p * high), y // p)
            max_m = min(x // (p * low), y)
            if p >= max_m: break

            # Find special leaves in the tree (i.e. φ(x/n, b) where n > y)
            for m in range(max_m, min_m, -1):
                if p < lpf[m] and mu[m] != 0:
                    # Compute φ(x/(pm), b) by adding contributions from remaining
                    # elements after sieving the first b primes
                    # μ(pm) * f(pm) * φ_f(x/(pm), b) = -μ(m) * f(p) * f(m) * φ_f(...)
                    index = (x // (p * m) - low) >> 1
                    phi_xn = phi[b] + _fenwick_tree_query(tree, index)
                    S2 -= mu[m] * phi_xn * (f(m) * f(p) if f else 1)

            # Store the accumulated sum over unsieved elements
            phi[b] += _fenwick_tree_query(tree, tree_size - 1)

            # Mark odd prime multiples in the sieve
            # Update the tree for each element being marked for the first time
            next_odd_prime_multiple = (((low + p - 1) // p) | 1) * p
            for index in range((next_odd_prime_multiple - low) >> 1, tree_size, p):
                if odd_sieve[index]:
                    odd_sieve[index] = False
                    value = values[index] if f else 1
                    _fenwick_tree_update(tree, index, -value, tree_size)

    return S2

def _lmo_odd_sieve(
    start: int,
    sieve_size: int,
    odd_primes: Sequence[int],
    max_prime: int | None = None,
) -> bytearray:
    """
    Sieve the interval [start, start + sieve_size) using the given primes.
    Returns a sieve of odd numbers that are coprime to the given primes.
    """
    # Initialize sieve segment
    # Only odd numbers are stored in the sieve (sieve[i] corresponds to start + 2i)
    start, end = start | 1, start + sieve_size
    sieve_size = (end - start + 1) >> 1
    sieve = bytearray(b'\x01') * sieve_size
    zeros = bytearray(b'\x00') * sieve_size

    # Iterate over primes
    for p in odd_primes:
        if max_prime and p > max_prime: break

        # Find next odd multiple of p >= start
        next_odd_multiple = start + (p - start) % (p + p)

        # Mark multiples of p in the odd sieve
        index = (next_odd_multiple - start) >> 1
        count = (sieve_size - index + p - 1) // p
        sieve[index::p] = zeros[:count]

    return sieve

def _fenwick_tree_init(values: Iterable[Number]) -> list[Number]:
    """
    Create a Binary Indexed Tree (Fenwick Tree) from the given values.
    """
    tree = list(values)
    for index, parent_index in _fenwick_tree_edges(len(tree)):
        tree[parent_index] += tree[index]

    return tree

def _fenwick_tree_query(tree: list[Number], index: int) -> Number:
    """
    Query the prefix sum for the tree at the given index.
    """
    total = 0
    for i in _fenwick_tree_query_path(index):
        total += tree[i]

    return total

def _fenwick_tree_update(tree: list[Number], index: int, value: Number, tree_size: int):
    """
    Update the given index in the tree.
    """
    for i in _fenwick_tree_update_path(index, tree_size):
        tree[i] += value

@small_cache
def _fenwick_tree_edges(tree_size: int) -> tuple[tuple[int, int], ...]:
    """
    Get all (index, parent_index) pairs for a Binary Indexed Tree (Fenwick Tree).
    """
    return tuple(
        (index, index | (index + 1))
        for index in range(tree_size - 1)
        if index | (index + 1) < tree_size
    )

@large_cache
def _fenwick_tree_query_path(index: int) -> tuple[int, ...]:
    """
    Get all indices that need to be queried for a prefix sum.
    """
    path, index = [], index + 1
    while index > 0:
        path.append(index - 1)
        index &= index - 1  # clears the lowest set bit

    return tuple(path)

@large_cache
def _fenwick_tree_update_path(index: int, tree_size: int) -> tuple[int, ...]:
    """
    Get all indices that need to be updated for a value change.
    """
    path = []
    while index < tree_size:
        path.append(index)
        index |= index + 1  # sets the lowest unset bit

    return tuple(path)

@large_cache
def _phi_prime_count(x: int, a: int, small_primes: tuple[int, ...]) -> int:
    """
    Evaluate Legendre's partial sieve function φ(x, a),
    which counts the number of positive integers ≤ x coprime to the first a primes.
    """
    if a == 0:
        return x
    elif a < 8:
        # Use the direct formula φ(x, a) = (x/P) * φ(P) + φ(x % P, a)
        q, r = divmod(x, P := _primorial(a))
        totient_P = prod(p - 1 for p in small_primes[:a])
        return q * totient_P + _phi_prime_count_offsets(P)[r]
    else:
        # Use the recursive formula φ(x, a) = φ(x, a - 1) - φ(x/p, a - 1)
        p = small_primes[a - 1]
        return (
            _phi_prime_count(x, a - 1, small_primes)
            - _phi_prime_count(x // p, a - 1, small_primes)
        )

@small_cache
def _phi_prime_count_offsets(d: int) -> tuple[int, ...]:
    """
    Compute values for Legendre's partial sieve function φ(r, a) for r = 0, 1 ... d - 1,
    where d is the product of the first a primes.
    """
    return tuple(itertools.accumulate(_coprime_range(d)))

@large_cache
def _phi_prime_sum(x: int, a: int, small_primes: tuple[int, ...]) -> int:
    """
    Evaluate Legendre's partial sieve function φ_f(x, a) for f(n) = n,
    which gives the sum of positive integers ≤ x coprime to the first a primes.
    """
    if a == 0:
        return x * (x + 1) // 2  # sum of all integers <= x
    elif a == 1:
        return ((x + 1) // 2)**2  # sum of odd integers <= x
    elif a < 8:
        # Use direct formula based on periodicity of coprimes mod P
        q, r = divmod(x, P := _primorial(a))
        count_coprimes, sum_coprimes = _phi_prime_sum_offsets(P)[r]
        return P * q * (q * totient(P) // 2 + count_coprimes) + sum_coprimes
    else:
        # Use the recurrence φ_f(x, a) = φ_f(x, a - 1) - f(p_a) * φ_f(x/p_a, a - 1)
        p = small_primes[a - 1]
        return (
            _phi_prime_sum(x, a - 1, small_primes)
            - p * _phi_prime_sum(x // p, a - 1, small_primes)
        )

@small_cache
def _phi_prime_sum_offsets(d: int) -> tuple[tuple[int, int], ...]:
    """
    Compute cumulative counts/sums for the weighted Legendre partial sieve function
    with f(n) = n. Returns offsets[r] = (φ(r, a), φ_f(r, a)) for r = 0, 1 ... d - 1,
    where d is the product of the first a primes, φ(r, a) counts and φ_f(r, a) sums
    integers ≤ r coprime to the first a primes.
    """
    is_coprime = _coprime_range(d)
    counts = itertools.accumulate(is_coprime)
    sums = itertools.accumulate(map(mul, range(d), is_coprime))
    return tuple(zip(counts, sums))

@small_cache
def _primorial(n: int) -> int:
    """
    Calculate the product of the first n primes.
    """
    return prod(primes(count=n))



########################################################################
############################ Factorization #############################
########################################################################

def perfect_power(n: int) -> tuple[int, int]:
    """
    Find integers a, b such that a^b = n.

    Returns the solution (a, b) with minimal b > 1 if there are any such solutions,
    otherwise returns the trivial solution (n, 1).

    Parameters
    ----------
    n : int
        Integer target
    """
    if n in (0, 1):
        return (n, 2)
    if n == -1:
        return (-1, 3)

    # Handle square roots
    n = -n if (is_negative := n < 0) else n
    if not is_negative and (n & 0xF) in (0, 1, 4, 9) and (r := isqrt(n)) * r == n:
        return (r, 2)

    # Try to find a small prime factor and its multiplicity
    multiplicity = 0
    if n & 1 == 0:
        multiplicity = (n & -n).bit_length() - 1
    elif (g := gcd(n, _PRIMORIAL_ODD_PRIMES_BELOW_256)) > 1:
        multiplicity = next(valuation(n, p) for p in _ODD_PRIMES_BELOW_256 if not g % p)

    # Calculate maximum possible exponent
    max_exponent = n.bit_length() - 1
    if multiplicity == 0:
        max_exponent = min(max_exponent, ilog(n, 257))
    if multiplicity == 1 or max_exponent < 3:
        return (-n if is_negative else n, 1)

    # If we know multiplicity, only check its odd prime divisors
    if multiplicity > 2:
        m = multiplicity
        m >>= (m & -m).bit_length() - 1  # remove factors of 2

        # Trial division to find and check odd prime factors in order
        for p in _ODD_PRIMES_BELOW_256:
            if m % p == 0:
                if p <= max_exponent and pow(r := iroot(n, p), p) == n:
                    return ((-r if is_negative else r), p)
                while m % p == 0:
                    m //= p

        # Find m-th root
        if 1 < m <= max_exponent and pow(r := iroot(n, m), m) == n:
            return ((-r if is_negative else r), m)
    else:
        # Check all odd primes
        for p in primes(low=3, high=max_exponent):
            if pow(r := iroot(n, p), p) == n:
                return ((-r if is_negative else r), p)

    return (-n if is_negative else n, 1)

def prime_factors(n: int) -> tuple[int, ...]:
    """
    Get all prime factors of n in sorted order (with multiplicity).

    Uses a combination of trial division, Fermat's factorization method,
    Brent's variant of Pollard's rho, Lenstra's elliptic curve method (ECM),
    and a self-initializing quadratic sieve (SIQS).

    Parameters
    ----------
    n : int
        Integer to factor
    """
    return tuple(sorted(_gen_prime_factors(n)))

def prime_factorization(n: int) -> dict[int, int]:
    """
    Get the prime factorization of n as a dictionary of {prime: exponent}.

    Parameters
    ----------
    n : int
        Integer to factor
    """
    pf = {}
    for p in _gen_prime_factors(n):
        pf[p] = pf.get(p, 0) + 1
    return pf

def divisors(n: int) -> tuple[int, ...]:
    """
    Get all positive divisors of n in sorted order.

    Parameters
    ----------
    n : int
        Integer to factor
    """
    factors = [1]
    for p, e in prime_factorization(n).items():
        current_factors, prime_power = factors[:], 1
        for _ in range(e):
            prime_power *= p
            factors += [d * prime_power for d in current_factors]

    return tuple(sorted(factors))

def _gen_prime_factors(n: int) -> Iterator[int]:
    """
    Get all prime factors of n (with multiplicity, and in no specific order).

    Uses a combination of trial division, Brent's variant of Pollard's rho
    factorization method, Lenstra's elliptic curve method (ECM),
    and a self-initializing quadratic sieve (SIQS).
    """
    if n == 0:
        raise ValueError("Must have n != 0")

    # Factor out powers of two
    n = -n if n < 0 else n
    num_trailing_zeros = (n & -n).bit_length() - 1
    yield from itertools.repeat(2, num_trailing_zeros)
    n >>= num_trailing_zeros
    if n == 1: return

    # Primorial GCD to find small odd prime factors
    if (g := gcd(n, _PRIMORIAL_ODD_PRIMES_BELOW_256)) > 1:
        for p in _ODD_PRIMES_BELOW_256:
            if g % p == 0:
                while n % p == 0:
                    yield p
                    n //= p

    if n == 1: return

    # Use a pipeline of Brent/ECM/SIQS factorization algorithms
    stack = deque([n])
    while stack:
        n = stack.popleft()
        if is_prime(n):
            yield n
        elif n > 1:
            # Fermat factorization
            if (factors := _fermat_factorization(n)):
                stack.extend(factors)
                continue

            # Brent for small factors (more aggressively capped for 64+ bit inputs)
            num_bits = n.bit_length()
            max_attempts = 2 if num_bits <= 64 else 1
            max_iterations = 2**18 if num_bits <= 64 else 2**16
            d = _brent(n, max_attempts=max_attempts, max_iterations=max_iterations)
            if 1 < d < n:
                stack.extend([d, n // d])
                continue

            # Retry with Brent for any remaining small factors
            # Here Brent has no fixed limit on attempts (i.e. won't return failure)
            if num_bits <= 64:
                d = _brent(n, max_attempts=None, max_iterations=None)
                stack.extend([d, n // d])
                continue

            # ECM to peel off medium-sized factors
            if num_bits > 128:
                d = _ecm(n, max_curves=32)
                if 1 < d < n:
                    stack.extend([d, n // d])
                    continue

            # Fallback to SIQS for remaining large factors
            B, max_polynomial_count = None, None
            while True:
                d = _siqs(n, B=B, max_polynomial_count=max_polynomial_count)
                if 1 < d < n:
                    stack.extend([d, n // d])
                    break
                elif _miller_rabin(n, 64):
                    yield n
                    break
                else:
                    # Increase search parameters
                    B = int((B or 60000) * 1.25)
                    max_polynomial_count = int((max_polynomial_count or 60000) * 1.25)

def _partial_factorization(
    n: int,
    small_primes: Collection[int],
) -> tuple[dict[int, int], int]:
    """
    Factor n with respect to a set of primes (sorted in increasing order).
    Returns a partial prime factorization as a dictionary {prime: exponent},
    and the remaining cofactor after dividing out all given primes.
    """
    partial_pf = {}
    for p in small_primes:
        if n % p: continue  # n not divisible by p
        n, e = n // p, 1  # n divisible by p, exponent is at least 1
        while True:
            quotient, remainder = divmod(n, p)
            if remainder: break  # n no longer divisible by p
            n, e = quotient, e + 1

        partial_pf[p] = e
        if p*p > n: break  # only check termination when we've actually done division

    if n > 1 and n in small_primes:
        partial_pf[n] = n = 1

    return partial_pf, n

def _fermat_factorization(n: int, num_iterations: int = 3) -> tuple[int, int] | None:
    """
    Use Fermat's factorization method to factor n as the difference of two squares.
    """
    b_squared = (a := isqrt(n) + 1) * a - n
    for _ in range(num_iterations):
        b = isqrt(b_squared)
        if b_squared - b*b == 0:
            return (a + b, a - b)
        b_squared += 2*a + 1
        a += 1

def _brent(
    n: int,
    max_attempts: int | None = None,
    max_iterations: int | None = None,
    batch_size: int = 128,
) -> int:
    """
    Algorithm based on Brent's variant of Pollard's rho factorization method.
    Returns an integer factor of n.

    This particular version has a deterministic flavor; when `max_attempts`
    is set to None, we are guaranteed to find a non-trivial factor for any
    composite n in worst-case O(√n) time.

    See: https://maths-people.anu.edu.au/~brent/pd/rpb051i.pdf

    Complexity
    ----------
    Expected Õ(√p) time, where p is smallest prime factor. Õ(n¹ᐟ⁴) time for semiprimes.
    Deterministic O(√n) worst case.
    """
    if (n & 1) == 0:
        return 2
    if n == 25:
        return 5

    # With starting point y = 2 and polynomial of the form f(x) = x^2 + c,
    # for odd composite n != 25, there is an elementary proof that there exists some
    # 0 < c < √n - 1 that finds a nontrivial factor on the very first GCD check
    # Let x0 = 2, c = p - 2 with p | n and p <= √n, and consider gcd(n, f(f(x0)) - x0).
    # With O(log n) attempts of O(n^(1/4)) iterations each, and the remaining
    # O(√n) attempts with 2 iterations each, the overall expected time remains O(n^1/4),
    # with a deterministic worst-case of O(√n).
    random_permutation, iteration_schedule = None, None
    if max_attempts is None:
        max_attempts = isqrt(n)
        random_permutation = permutation(max_attempts)
        iteration_schedule = _brent_iteration_schedule(n)

    for _ in range(max_attempts):
        if random_permutation:
            # Fixed starting point y = 2 and random polynomial f(x) = x^2 + c
            y, c = 2, next(random_permutation) + 1
        else:
            # Random starting point and polynomial f(x) = x^2 + c
            y, c = secrets.randbelow(n - 3) + 2, secrets.randbelow(n - 3) + 1

        # Per-attempt iteration cap
        max_iter = next(iteration_schedule) if iteration_schedule else max_iterations

        # Save checkpoint x, iterate y -> f(y) for r steps, then iterate r more steps
        # while also accumulating products q = prod (x - y) over the range.
        # When gcd(q, n) > 1, we've found a factor.
        G, r, num_iterations = 1, 1, 0  # batch GCD, range, iteration count
        while G == 1:
            x, q = y, 1  # checkpoint, batch product
            num_iterations += r
            for _ in range(r):
                y = (y*y + c) % n
            if max_iter is not None and num_iterations > max_iter: break

            # Batch GCD
            for k in range(0, r, batch_size):
                ys = y
                limit = min(batch_size, r - k)
                num_iterations += limit
                for _ in range(limit):
                    y = (y*y + c) % n
                    q = q * (x - y) % n
                if max_iter is not None and num_iterations > max_iter: break
                if (G := gcd(q, n)) > 1: break

            # Double the range
            r *= 2

        # Move on to next attempt when the maximum iteration has been reached
        if G == 1:
            continue

        # Backtrack if batch GCD failed (i.e. batch product is 0 mod n)
        if G == n:
            G, y = 1, ys
            while G == 1:
                y = (y*y + c) % n
                G = gcd(x - y, n)

        if 1 < G < n:
            return G  # success, found non-trivial factor

    # We never reach here if the caller passed `max_attempts=None`
    return 1  # failure, return trivial factor

def _brent_iteration_schedule(n: int) -> Iterator[int]:
    """
    Per-attempt iteration caps for Brent's rho.
    """
    cap_min, cap_max = 2, max(2, 32 * isqrt(isqrt(n)))

    # O(log n) heavy attempts with iteration limit of O(n^(1/4))
    for _ in range(4 * n.bit_length()):
        yield cap_max

    # Remaining light attempts with constant limit (i.e. once through Brent inner loop)
    while True:
        yield cap_min

def _ecm(
    n: int,
    B1: int | None = None,
    B2: int | None = None,
    max_curves: int | None = None,
) -> int:
    """
    Lenstra's Elliptic Curve Method (ECM) for integer factorization.
    Returns an integer factor of n.

    Uses Montgomery curves with Suyama's parametrization and a two-stage ECM
    (stage 1 + stage 2 baby-step/giant-step).

    See: https://wstein.org/edu/124/misc/montgomery.pdf

    Complexity
    ----------
    O(exp((√2 + o(1)) √(log p log log p))) time, where p is smallest prime factor
    """
    if (n & 1) == 0:
        return 2

    # Select hyperparameters based on input size
    # B1 and B2 are bounds for the primes used in stage 1 and stage 2
    # Defaults for (B1, B2, max_curves) are tuned for 64–128-bit composites
    bits = n.bit_length()
    default_thresholds = [
        (68, (500, 5000, 200)),
        (84, (1000, 20000, 400)),
        (92, (1500, 30000, 800)),
        (108, (4000, 150000, 1500)),
        (128, (5000, 200000, 1000)),
    ]
    defaults = _threshold_select(bits, default_thresholds, (5000, 200000, 1000))
    B1, B2 = B1 or defaults[0], B2 or defaults[1]
    max_curves = max_curves or defaults[2]

    # Precomputation
    prime_powers = _ecm_prime_powers(B1)
    plan = _ecm_stage_2_plan(B1, B2)

    # Loop over elliptic curves
    for _ in range(max_curves):
        # Pick a random curve
        sigma = secrets.randbelow(n - 7) + 6 if n > 7 else 6
        A24, P, factor = _ecm_suyama_curve(n, sigma)
        if factor is not None:
            return factor
        if A24 is None or P is None:
            continue

        # Stage 1
        # Multiply point P by prime powers p^⌊log_p B1⌋ for p <= B1
        # Periodically check the GCD with the Z-coordinate for a non-trivial factor
        Q = P
        for i, prime_power in enumerate(prime_powers, start=1):
            Q = _montgomery_ladder(prime_power, Q, A24, n)
            if i % 32 == 0:
                if 1 < (g := gcd(Q[1], n)) < n:
                    return g

        # Check the Z-coordinate for a non-trivial factor
        if 1 < (g := gcd(Q[1], n)) < n:
            return g
        if g == n:
            # Rare degeneracy, check the X-coordinate instead
            if 1 < (g := gcd(Q[0], n)) < n:
                return g
            continue

        # Stage 2
        # For each prime B1 <= p <= B2, compute point pQ
        # and check GCD with the Z-coordinate
        if B2 > B1:
            if 1 < (g := _ecm_stage_2(n, A24, Q, plan)) < n:
                return g

    return 1  # failure, return trivial factor

def _montgomery_add(
    P: tuple[int, int],
    Q: tuple[int, int],
    diff: tuple[int, int],
    mod: int,
) -> tuple[int, int]:
    """
    Montgomery differential addition in projective x-only coordinates.
    Points P, Q are each represented as (X, Z) with affine x = X/Z.
    Requires diff != O (i.e., P != Q and P != -Q) modulo any prime factor.

    See: https://www.hyperelliptic.org/EFD/g1p/auto-montgom-xz.html
    """
    A, B = P[0] + P[1], P[0] - P[1]
    C, D = Q[0] + Q[1], Q[0] - Q[1]
    DA, CB = D*A % mod, C*B % mod
    plus, minus = DA + CB, DA - CB
    X3 = (diff[1] * ((plus * plus) % mod)) % mod
    Z3 = (diff[0] * ((minus * minus) % mod)) % mod
    return X3, Z3

def _montgomery_double(P: tuple[int, int], A24: int, mod: int) -> tuple[int, int]:
    """
    Montgomery curve point doubling in projective x-only coordinates.
    Point P is represented as (X:Z) with affine x = X/Z.
    Uses the Montgomery parameter A24 = (A + 2) / 4.

    See: https://www.hyperelliptic.org/EFD/g1p/auto-montgom-xz.html
    """
    A, B = P[0] + P[1], P[0] - P[1]
    AA, BB = (A * A) % mod, (B * B) % mod
    C = AA - BB
    X2 = (AA * BB) % mod
    Z2 = (C * (BB + A24*C % mod)) % mod
    return X2, Z2

def _montgomery_ladder(
    k: int,
    P: tuple[int, int],
    A24: int,
    mod: int,
) -> tuple[int, int]:
    """
    Montgomery ladder for scalar multiplication [k]P using x-only arithmetic.
    """
    if k <= 0:
        return (1, 0)  # O (point at infinity)
    if k == 1:
        return (P[0] % mod, P[1] % mod)

    # Initialize with leading bit handled: R0 = P, R1 = 2P
    R0 = diff = (P[0] % mod, P[1] % mod)
    R1 = _montgomery_double(diff, A24, mod)
    for bit in format(k, 'b')[1:]:
        if bit != '0':
            R0 = _montgomery_add(R0, R1, diff, mod)
            R1 = _montgomery_double(R1, A24, mod)
        else:
            R1 = _montgomery_add(R0, R1, diff, mod)
            R0 = _montgomery_double(R0, A24, mod)

    return R0

def _ecm_suyama_curve(
    n: int,
    sigma: int,
) -> tuple[int | None, tuple[int, int] | None, int | None]:
    """
    Construct a Montgomery curve and starting point using Suyama's parametrization.

    Returns (A24, (X, Z), factor). If a non-trivial factor is discovered during setup,
    it is returned in `factor`.

    If setup fails (singular curve), returns (None, None, None).
    """
    sigma %= n
    if sigma == 0:
        return None, None, None  # degenerate, choose another sigma

    u, v = (sigma*sigma - 5) % n, (4*sigma) % n
    if u == 0 or v == 0:
        return None, None, None  # degenerate, choose another sigma

    # Calculate starting point P = (X1:Z1) = (u^3:v^3)
    X1, Z1 = u3, v3 = (u*u*u) % n, (v*v*v) % n

    # Check for non-trivial factor of n
    denominator = (16*u3*v) % n
    if 1 < (g := gcd(denominator, n)) < n:
        return None, None, g  # found non-trivial factor
    if g == n or denominator == 0:
        return None, None, None  # degenerate, choose another sigma

    # Calculate A24 = (A + 2) / 4 = (v-u)^3 * (3u+v) / (16*u^3*v)
    t = (v - u) % n
    t3 = (t*t*t) % n
    numerator = (t3 * ((3*u + v) % n)) % n
    A24 = (numerator * pow(denominator, -1, n)) % n
    if A24 == 0:
        return None, None, None  # degenerate, choose another sigma

    # Reject (likely) singular curves by checking gcd(A^2 - 4, n)
    A = (4*A24 - 2) % n
    discriminant = (A*A - 4) % n
    if 1 < (g := gcd(discriminant, n)) < n:
        return None, None, g  # found non-trivial factor
    if g == n or discriminant == 0:
        return None, None, None  # degenerate, choose another sigma

    return A24, (X1, Z1), None

@small_cache
def _ecm_prime_powers(B1: int) -> tuple[int, ...]:
    """
    Precompute prime powers p^e ≤ B1 for ECM stage 1.
    Cached because ECM is often called repeatedly with the same bounds.
    """
    return tuple(p**ilog(B1, p) for p in primes(high=B1))

@small_cache
def _ecm_stage_2_plan(
    B1: int,
    B2: int,
) -> tuple[int, dict[int, tuple[int, ...]], frozenset[int]]:
    """
    Precompute a stage-2 plan for ECM using baby-step giant-step (BSGS) strategy.

    Represents each prime r in (B1, B2] as r = kD ± offset, where D ≈ √B2
    is the "giant step" size, k indicates which multiple of D is closest to r,
    and offset is the distance from r to that multiple (0 ≤ offset ≤ D/2).

    Returns
    -------
    giant_step_size : int
        The interval D for giant steps
    giant_step_to_offsets : dict[int, tuple[int, ...]]
        Maps each k to tuple of offsets for primes near k*D
    baby_steps : frozenset[int]
        All unique offset values that need to be precomputed
    """
    if B2 <= B1:
        return 0, {}, frozenset()

    # Choose giant step size D ≈ √B2, but ensure D/2 ≤ B1
    # This avoids k = 0 cases and huge baby-step sets
    giant_step_size = max(min(isqrt(B2), 2*B1), 6)
    giant_step_size += giant_step_size & 1  # round up to even

    # For each prime p in (B1, B2], represent as p = kD ± offset
    max_offset = giant_step_size // 2
    giant_step_to_offsets, baby_steps = defaultdict(set), set()
    for p in primes(low=B1+1, high=B2):
        k = (p + max_offset) // giant_step_size
        offset = abs(p - k*giant_step_size)
        giant_step_to_offsets[k].add(offset)
        if offset > 0:
            baby_steps.add(offset)

    # Convert to tuples for faster iteration
    giant_step_to_offsets = {
        k: tuple(sorted(offsets)) for k, offsets in giant_step_to_offsets.items()}

    return giant_step_size, giant_step_to_offsets, frozenset(baby_steps)

def _ecm_stage_2(
    n: int,
    A24: int,
    Q: tuple[int, int],
    plan: tuple[int, dict[int, tuple[int, ...]], frozenset[int]],
) -> int:
    """
    ECM stage 2 using Montgomery baby-step / giant-step.
    Returns a non-trivial factor of n if found, otherwise 1.
    """
    D, giant_step_to_offsets, baby_steps = plan # D is giant-step size
    if not D or not giant_step_to_offsets:
        return 1  # failure, return trivial factor

    # Baby steps - compute [d]Q for small offsets d
    # Primes p in (B1, B2] are written as p = kD ± d. Precompute [d]Q values.
    baby = {1: Q} if baby_steps else {}
    max_baby_step = max(baby_steps, default=0)
    if max_baby_step >= 3:
        Q2 = _montgomery_double(Q, A24, n)
        # Differential ladder for odd multiples [d+2]Q = [d]Q + [2]Q, diff = [d-2]Q
        prev, current, d = Q, _montgomery_add(Q2, Q, Q, n), 3
        while d <= max_baby_step:
            if d in baby_steps: baby[d] = current
            prev, current, d = current, _montgomery_add(current, Q2, prev, n), d + 2

    # Fallback for any missing values
    baby[0] = O = (1, 0)  # point at infinity
    baby.update({d: _montgomery_ladder(d, Q, A24, n) for d in baby_steps - baby.keys()})

    # Giant step base [D]Q
    PD = _montgomery_ladder(D, Q, A24, n)
    for k in range(max(giant_step_to_offsets) + 1):
        if k == 0:
            P_prev, P_current = None, O
        elif k == 1:
            P_prev, P_current = O, PD
        elif k == 2:
            P_prev, P_current = PD, _montgomery_double(PD, A24, n)
        else:
            # Handle k >= 3 with primes p = kD ± d, via differential addition
            P_prev, P_current = P_current, _montgomery_add(P_current, PD, P_prev, n)

        # Cross-ratio trick: in x-only Montgomery, we can't compute [kD±d]Q
        # directly, but (Xk*Zd - Xd*Zk) vanishes iff the points combine to
        # give a Z-coordinate sharing a factor with n
        if k in giant_step_to_offsets:
            Pk = P_current
            values = [
                (Pk[0]*baby[d][1] - baby[d][0]*Pk[1]) % n
                for d in giant_step_to_offsets[k]
            ]
            if (g := _batch_gcd(values, n)) > 1: return g

    return 1

def _batch_gcd(values: list[int], mod: int) -> int:
    """
    Compute gcd(prod(values), mod) with a fallback to per-value gcd when the product
    vanishes with respect to the modulus.
    """
    if values:
        product = 1
        for v in values:
            product = (product * v) % mod

        if 1 < (g := gcd(product, mod)) < mod:
            return g
        if g == mod:
            for v in values:
                if 1 < (gg := gcd(v, mod)) < mod:
                    return gg

    return 1

def _siqs(
    n: int,
    B: int | None = None,
    M: int | None = None,
    large_prime_bound_multiplier: int | None = None,
    max_polynomial_count: int | None = None,
) -> int:
    """
    Self-initializing quadratic sieve (SIQS) with triple large prime variation.
    Returns an integer factor of n.

    See: https://www.ams.org/notices/199612/pomerance.pdf
    See: https://math.dartmouth.edu/~carlp/implementing.pdf
    See: https://ir.cwi.nl/pub/1367/1367D.pdf

    Complexity
    ----------
    O(exp((1 + o(1)) √(log n log log n))) time
    """
    base, exponent = perfect_power(n)
    if exponent > 1:
        return base

    bits = n.bit_length()

    # Use heuristic factor base bound B ≈ e^(1/2 sqrt(log(n) * log(log(n))))
    if B is None:
        log_bits = bits.bit_length()
        B_thresholds = [(128, 30000), (144, 40000), (160, 50000)]
        B = 1 << (isqrt(bits * log_bits) >> 1)
        B = max(300, min(B, _threshold_select(bits, B_thresholds, 60000)))

    # Adaptively set sieve half-width M based on input size
    if M is None:
        M_thresholds = [
            (100, 50000), (112, 60000), (128, 80000),
            (140, 120000), (152, 140000), (160, 170000),
        ]
        M = _threshold_select(bits, M_thresholds, 220000)

    # Adaptively set large prime bound multiplier based on input size
    if large_prime_bound_multiplier is None:
        lpbm_thresholds = [(120, 8), (132, 10), (144, 12), (160, 14)]
        large_prime_bound_multiplier = _threshold_select(bits, lpbm_thresholds, 16)

    # Adaptively set max number of polynomials to use based on input size
    if max_polynomial_count is None:
        poly_thresholds = [(112, 8000), (128, 12000), (144, 25000), (160, 40000)]
        max_polynomial_count = _threshold_select(bits, poly_thresholds, 60000)

    # Collect relations (X, pf) where X^2 ≡ Q (mod n), Q is a B-smooth integer,
    # and pf[p] = e is the prime factorization of Q over the factor base
    factor_base = _build_factor_base(n, B)
    if not factor_base: return 1  # failure, no valid factor base
    min_relation_count = len(factor_base) + 30
    L = B * large_prime_bound_multiplier  # large prime bound
    relations, factor = _collect_relations(
        n, factor_base, B, L, M, min_relation_count, max_polynomial_count)

    # Check early termination conditions
    if factor is not None:
        return factor  # success, one of the large primes was a factor
    if len(relations) < len(factor_base):
        return 1  # failure, null space will be trivial

    # Build a relation matrix over GF(2), where each row is a bit-packed integer
    # and each bit j is set only when prime j has odd exponent in that relation
    fb_primes, _, _ = zip(*factor_base)
    idx = {p: i for i, p in enumerate((-1,) + fb_primes)}  # prime index
    rows = [
        reduce(xor, (1 << idx[p] for p, e in pf.items() if e & 1), 0)
        for _, pf in relations
    ]

    # Find null space of the relation matrix over GF(2)
    # The product of the corresponding relations has exponents that are all even,
    # and thus prod X^2 ≡ prod Q = Y^2 (mod n) is a perfect square mod n
    prod_mod_n = lambda values: reduce(lambda a, b: (a * b) % n, values, 1)
    for mask in _nullspace_gf2(rows):
        X, pf_prod = 1, defaultdict(int)
        for i, (x, pf) in enumerate(relations):
            if (mask >> i) % 2 == 1:
                X = (X * x) % n
                for p, e in pf.items():
                    pf_prod[p] += e

        Y = prod_mod_n(pow(p, e // 2, n) for p, e in pf_prod.items() if p != -1)
        for d in (X - Y, X + Y):
            if 1 < (g := gcd(d, n)) < n:
                return g  # success, found non-trivial factor

    return 1  # failure, return trivial factor

def _build_factor_base(n: int, B: int) -> list[tuple[int, float, int]]:
    """
    Build factor base of primes p ≤ B where for each prime p,
    n is a quadratic residue mod p.
    """
    factor_base = [(2, log(2), 1)] if n % 2 != 0 and B >= 2 else []
    for p in primes(low=3, high=B):
        if pow(n % p, (p - 1) // 2, p) != 1: continue  # skip non-residues
        factor_base.append((p, log(p), _tonelli_shanks(n, p)))

    return factor_base

def _gen_polynomials(
    n: int,
    factor_base: list[tuple[int, float, int]],
    M: int,
) -> Iterator[tuple[int, int, dict[int, int]]]:
    """
    Generate SIQS polynomials Q(x) = (Ax + b)^2 - n,
    where A ≈ √(2n)/M is the product of k primes and b satisfies b^2 ≡ n (mod A).
    """
    sqrt_n = isqrt(n)
    target_A = max(isqrt(2*n) // M, 2)

    # Skip tiny primes (poor A factors, inflate duplicates)
    skip = max(10, len(factor_base) // 10)
    pool = factor_base[skip:] if len(factor_base) >= skip + 3 else factor_base
    pool_primes = [p for p, _, _ in pool]

    # Choose k so target_A^(1/k) falls within prime range
    k = ilog(target_A - 1, pool_primes[-1]) + 1  # ⌈log_{p_max}(target_A)⌉
    k = max(2, min(k, len(pool)))

    # Narrow pool to primes near ideal size
    ideal = iroot(target_A, k)
    center = bisect.bisect_left(pool_primes, ideal)
    half_width = min(max(200, len(pool) // 5), 1200)  # heuristic window half-size
    pool = pool[max(0, center - half_width):center+half_width] or pool
    pool = pool if len(pool) >= k else factor_base

    # Set acceptance bounds [low, high] for A (only if achievable)
    low, high = 1, inf
    if target_A > 10000:
        min_A = prod(p for p, _, _ in sorted(pool, key=lambda x: x[0])[:k])
        band = 25 if target_A >= 10**12 else 15
        if min_A <= target_A * band:
            low, high = max(2, target_A // band), target_A * band

    # Generate polynomials
    rng, seen = secrets.SystemRandom(), set()
    while True:
        # Set A as the product of k randomly sampled primes from the pool
        sample = sorted(rng.sample(pool, k), key=lambda x: x[0])
        A = prod(p for p, _, _ in sample)

        # Reject duplicates and out-of-bounds A values
        if A < low or A > high or A in seen: continue
        seen.add(A)
        if len(seen) > 20000:
            seen.clear()

        # Compute mod inverses for A^(-1) mod p for the factor base
        A_inverses = {p: pow(A, -1, p) for p, _, _ in factor_base if A % p != 0}

        # For each prime p | A, we have two modular roots (±r)^2 = n (mod p)
        # Fix the 1st sign (no ± b duplicates) and set the others to get b^2 ≡ n (mod A)
        # B_i ≡ r_i (mod p_i), B_i ≡ 0 (mod other primes) via CRT
        # so sum(B_i) is the CRT solution for all residues r (mod p_i).
        B = [(A // p) * ((root * pow(A // p, -1, p)) % p) for p, _, root in sample]

        # Shift b-values closer to √n for more efficient sieving (i.e. |Q(x)| small)
        shift = lambda b: b + A * ((sqrt_n - b) // A)

        # Enumerate all 2^(k-1) sign combinations via Gray code (1st sign is fixed)
        b = sum(B) % A  # base solution
        yield (A, shift(b), A_inverses)
        for i in range(1, 1 << (k - 1)):
            j = (i & -i).bit_length()
            sign = 1 if ((i >> j) & 1) else -1
            b = (b + 2 * sign * B[j]) % A  # flip sign of component j
            yield (A, shift(b), A_inverses)

@small_cache
def _byte_subtraction_table(d: int) -> bytes:
    """
    Translation table for byte subtraction. Maps byte v to max(0, v - d).
    """
    return bytes(max(0, v - d) for v in range(256))

@small_cache
def _byte_threshold_table(threshold: int) -> bytes:
    """
    Translation table for threshold filtering.
    Maps byte v to True if v ≤ threshold, otherwise to False.
    """
    return bytes(True if v <= threshold else False for v in range(256))

def _sieve_polynomial(
    n: int,
    factor_base: list[tuple[int, float, int]],
    polynomial: tuple[int, int, dict[int, int]],
    M: int,
) -> Iterator[tuple[int, int]]:
    """
    Sieve the polynomial Q(x) = (Ax + b)^2 - n for x in [-M, M].
    Yields (Q(x), Ax + b) pairs for candidates passing the smoothness threshold.
    """
    A, b, A_inverses = polynomial
    length, offset = 2*M + 1, -M

    # Set smoothness threshold ~ max|Q(x)|
    max_abs_Q = max(abs((-A*M + b)**2 - n), abs((A*M + b)**2 - n))
    base_log = log(max_abs_Q) if max_abs_Q > 1 else 1.0
    threshold = 0.55 * base_log  # higher than usual to account for skipping p < 30

    # Initialize sieve as bytearray
    # Scale logs to fit in a byte (0-255), where base_log -> 255 is the max value
    # Sieve elements start at 255 and decrease by scaled log(p) for each factor p 
    sieve = bytearray([255]) * length
    scale = 255 / base_log
    threshold = round(threshold * scale)

    # Sieve with factor base
    # Skip small primes (which will still be checked later when factoring Q(x))
    for p, log_p, root in factor_base:
        if A % p == 0 or p < 30:
            continue

        # Translation table for subtracting log(p) at byte-scale
        table = _byte_subtraction_table(round(log_p * scale))

        # Mark all x where Q(x) = 0 (mod p)
        inv_A = A_inverses[p]
        x1, x2 = ((root - b) * inv_A) % p, ((-root - b) * inv_A) % p
        start = (x1 - offset) % p
        sieve[start::p] = sieve[start::p].translate(table)
        if x2 != x1:
            start = (x2 - offset) % p
            sieve[start::p] = sieve[start::p].translate(table)

    # Yield only candidates that pass the smoothness threshold
    mask = sieve.translate(_byte_threshold_table(threshold))
    for i in itertools.compress(range(length), mask):
        y = A * (offset + i) + b
        Q = y * y - n
        if Q != 0:
            yield Q, y

def _collect_relations(
    n: int,
    factor_base: list[tuple[int, float, int]],
    B: int,
    large_prime_bound: int,
    M: int,
    min_relation_count: int,
    max_polynomial_count: int,
) -> tuple[list[tuple[int, dict[int, int]]], int | None]:
    """
    Collect smooth relations using SIQS with large prime variants, where
    a "relation" is X such that X^2 ≡ Q (mod n) where Q factors over small primes.

    Returns
    -------
    relations : list[tuple[int, dict[int, int]]]
        List of (X, pf) tuples, where pf is the factorization of Q over the factor base
    factor : int | None
        A non-trivial factor of n if found during collection, otherwise None
    """
    factor_base_primes, _, _ = zip(*factor_base)
    possible_lp = list(primes(low=B+1, high=large_prime_bound))

    # Generate and sieve polynomials for relations
    relations, partials = {}, []  # partials is a list of (X, pf, large_primes) tuples
    polynomial_generator = _gen_polynomials(n, factor_base, M)

    # Streaming GF(2) elimination on large-prime parity vectors.
    # Each partial contributes a sparse column with bits at its large primes (mod 2).
    # When a column reduces to 0, we've found a dependency whose large primes cancel.
    lp_index, lp_pivots, lp_masks = {}, {}, {}
    for polynomial in itertools.islice(polynomial_generator, max_polynomial_count):
        for Q, y in _sieve_polynomial(n, factor_base, polynomial, M):
            # Factor Q over the factor base
            pf, residual = _partial_factorization(abs(Q), factor_base_primes)
            pf[-1] = 1 if Q < 0 else 0  # account for sign with -1 factor

            # Handle large prime variants
            large_primes = _get_large_primes(residual, possible_lp)
            if large_primes is None:
                continue  # unusable
            elif not large_primes:
                relations.setdefault(y % n, pf)  # smooth
            else:
                # Check if any large primes are factors of n
                if 1 < (g := gcd(prod(large_primes), n)) < n:
                    return [], g
                elif g == n:
                    return [], next(p for p in large_primes if n % p == 0)

                # Add this partial and try to eliminate large primes (mod 2)
                partials.append((y % n, pf, large_primes))

                # Build parity vector (bitset) for the large-prime multiset
                parity_vector = 0
                for p in large_primes:
                    parity_vector ^= 1 << lp_index.setdefault(p, len(lp_index))

                # Gaussian elimination over GF(2) on the implicit matrix
                # where each row is a partial (indexed by bit position in combo)
                # and each column is a large prime (indexed by lp_index)
                combo = 1 << (len(partials) - 1)
                while parity_vector:
                    pivot_col = parity_vector.bit_length() - 1  # MSB as a pivot
                    if pivot_col not in lp_pivots:
                        lp_pivots[pivot_col], lp_masks[pivot_col] = parity_vector, combo
                        break
                    parity_vector ^= lp_pivots[pivot_col]
                    combo ^= lp_masks[pivot_col]

                # If v = 0, combo encodes a subset of partials where all large primes
                # occur an even number of times (i.e. their product is a square)
                if parity_vector == 0:
                    pf_combined, lp_counts, X_product = {}, {}, 1
                    while combo:
                        partial_index = combo.bit_length() - 1
                        X, pf, lps = partials[partial_index]
                        X_product = (X_product * X) % n
                        for p, e in pf.items():
                            pf_combined[p] = pf_combined.get(p, 0) + e
                        for p in lps:
                            lp_counts[p] = lp_counts.get(p, 0) + 1
                        combo ^= 1 << partial_index

                    # D = sqrt of large-prime part = prod p^(count/2) (mod n)
                    D = 1
                    for p, count in lp_counts.items():
                        D = D * pow(p, count >> 1, n) % n

                    relations.setdefault((X_product * pow(D, -1, n)) % n, pf_combined)

            if len(relations) >= min_relation_count:
                return list(relations.items()), None

    return list(relations.items()), None

def _get_large_primes(
    v: int,
    possible_large_primes: Sequence[int],
    max_count: int = 3,
) -> tuple[int, ...] | None:
    """
    Factor residue v into large primes.
    Returns tuple of up to `max_count` primes if v factors completely
    over `possible_large_primes`, otherwise returns None.
    """
    if v == 1:
        return ()
    if not possible_large_primes or max_count < 1:
        return None

    # Try to extract a large prime factor and recurse
    for p in possible_large_primes:
        if p * p > v: break
        if v % p == 0:
            rest = _get_large_primes(v // p, possible_large_primes, max_count - 1)
            if rest is not None:
                return (p,) + rest

    return (v,) if v <= possible_large_primes[-1] and is_prime(v) else None

def _nullspace_gf2(rows: list[int]) -> list[int]:
    """
    Find null space of the matrix over GF(2) using Gaussian elimination.
    Rows are bit-packed integers.
    """
    pivots, nullspace = {}, []
    for i, row in enumerate(rows):
        combo = 1 << i
        r = row
        while r:
            pivot_col = r.bit_length() - 1  # most significant bit as a pivot
            if pivot_col in pivots:
                pivot_row, pivot_mask = pivots[pivot_col]
                r ^= pivot_row
                combo ^= pivot_mask
            else:
                pivots[pivot_col] = (r, combo)
                break
        else:
            nullspace.append(combo)

    return nullspace



########################################################################
######################### Arithmetic Functions #########################
########################################################################

def omega(n: int) -> int:
    """
    Compute the value of ω(n), the number of distinct prime factors of n.

    Parameters
    ----------
    n: int
        Positive integer function argument
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    return len(set(_gen_prime_factors(n)))

def big_omega(n: int) -> int:
    """
    Compute the value of Ω(n), the number of prime factors of n (with multiplicity).

    Parameters
    ----------
    n: int
        Positive integer function argument
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    return sum(1 for _ in _gen_prime_factors(n))

def divisor_count(n: int) -> int:
    """
    Compute the value of σ₀(n), the number of divisors of n.

    Parameters
    ----------
    n: int
        Positive integer function argument
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    return prod(e + 1 for e in prime_factorization(n).values())

def divisor_sum(n: int) -> int:
    """
    Compute the value of σ₁(n), the sum of divisors of n.

    Parameters
    ----------
    n: int
        Positive integer function argument
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    return prod((p**(e + 1) - 1) // (p - 1) for p, e in prime_factorization(n).items())

def divisor_function(n: int, k: int = 1) -> int:
    """
    Compute the value of the divisor function σₖ(n), where σₖ(n) = ∑_{d|n} dᵏ.

    Parameters
    ----------
    n: int
        Positive integer function argument
    k: int
        Divisor exponent
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    if k < 0:
        raise ValueError("k must be a non-negative integer")

    pf = prime_factorization(n)
    if k == 0:
        return prod(e + 1 for e in pf.values())
    else:
        return prod((pow(p, k * (e + 1)) - 1) // (pow(p, k) - 1) for p, e in pf.items())

def partition(
    n: int,
    mod: int | None = None,
    restrict: Callable[[int], bool] | None = None,
) -> int:
    """
    Return the value of the partition function p(n).

    Parameters
    ----------
    n : int
        Integer to partition
    mod : int | None
        If provided, return p(n) mod m
    restrict : Callable(int) -> bool
        Function indicating integers that can be used in the partition,
        where restrict(k) = True means integer k can be used (e.g. restrict=nt.is_prime)
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")

    p = _euler_transform(restrict) if restrict else _partition_function(mod)
    return p(n) if mod is None or restrict is None else p(n) % mod

def radical(n: int) -> int:
    """
    Compute rad(n) as the product of the distinct prime factors of n.

    Parameters
    ----------
    n: int
        Positive integer function argument
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    return prod(set(_gen_prime_factors(n)))

def mobius(n: int) -> int:
    """
    Compute the Mobius function μ(n) for a positive integer n.

    Parameters
    ----------
    n: int
        Positive integer function argument
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    if n == 1:
        return 1
    if is_square(n):
        return 0
    if is_prime(n):
        return -1

    mu, seen = 1, set()
    for p in _gen_prime_factors(n):
        if p in seen:
            return 0
        else:
            seen.add(p)
            mu = -mu

    return mu

def totient(n: int) -> int:
    """
    Compute Euler's totient function φ(n) for a positive integer n.

    Parameters
    ----------
    n: int
        Positive integer function argument
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    phi = n
    for p in set(_gen_prime_factors(n)):
        phi -= phi // p

    return phi

def carmichael(n: int) -> int:
    """
    Compute Carmichael's lambda function λ(n) for a positive integer n.

    Parameters
    ----------
    n: int
        Positive integer function argument
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    terms = []
    for p, e in prime_factorization(n).items():
        if p == 2:
            terms.append(e if e < 3 else 2**(e - 2))
        else:
            terms.append((p - 1) * (p**(e - 1)))

    return lcm(*terms)

def valuation(n: int, p: int) -> int:
    """
    Compute the p-adic valuation νₚ(n), the exponent of p
    in the prime factorization of n.

    Parameters
    ----------
    n : int
        Positive integer
    p : int
        Prime number
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    if not is_prime(p):
        raise ValueError("p must be prime")
    if p == 2:
        return (n & -n).bit_length() - 1  # number of trailing 0's

    # For general p, use doubling to achieve O(log v) divisions
    # Build powers p, p^2, p^4, p^8, ... while p^(2^k) <= n
    powers = [(p, 1)]
    power, exponent = p, 1
    while (power := power * power) <= n:
        powers.append((power, exponent := 2*exponent))

    # Greedily divide out largest powers first
    v = 0
    for power, exponent in reversed(powers):
        while n % power == 0:
            n //= power
            v += exponent

    return v

def multiplicative_range(f: Callable[..., int], N: int, f0: int = 1) -> list[int]:
    """
    Find the value of a multiplicative function f(n) for each n = 0, 1, 2, ..., N - 1.
    Uses a sieving approach across the range to calculate function values efficiently.

    Parameters
    ----------
    f : Callable(n) -> int or Callable(p, e) -> int
        Function to compute values f(n) or f(p^e) at prime powers
    N : int
        Upper bound on range (exclusive)
    f0 : int
        Dummy value to include for f(0)
    """
    if N < 0:
        raise ValueError("N must be non-negative")
    if N == 0:
        return []

    # Select prime power function
    mapping = {
        divisor_count: lambda p, e: e + 1,
        divisor_sum: lambda p, e: (p**(e + 1) - 1) // (p - 1),
        radical: lambda p, e: p,
        mobius: lambda p, e: -1 if e == 1 else 0,
        totient: lambda p, e: (p - 1) * (p**(e - 1)),
    }
    if f in mapping:
        f_prime_power = cache(mapping[f])
    else:
        P = inspect.Parameter
        params = [
            p for p in inspect.signature(f).parameters.values()
            if p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)
        ]
        if len(params) >= 2:
            f_prime_power = cache(f)
        else:
            f_prime_power = cache(lambda p, e: f(p**e))

    # Use the multiplicative property
    # prime_divisor[n] = p is the largest prime divisor of n < sqrt(N)
    # prime_power[n] = p^e for the largest prime power p^e | n
    # prime_exponent[n] = p-adic valuation of n (where p = prime_divisor[n])
    prime_power = [1] * N
    prime_exponent = [0] * N
    prime_divisor = _prime_factor_range(N)
    values = [f0] + [1] * (N - 1)
    for n in range(2, N):
        p = prime_divisor[n]
        m = n // p
        if prime_divisor[m] == p:
            prime_exponent[n] = prime_exponent[m] + 1
            prime_power[n] = prime_power[m] * p
        else:
            prime_exponent[n] = 1
            prime_power[n] = p

        values[n] = values[n // prime_power[n]] * f_prime_power(p, prime_exponent[n])

    return values

@small_cache
def _partition_function(mod: int | None) -> Callable[[int], int]:
    """
    Return a callable partition function p(n) for given modulus.
    """
    partitions, pentagonals, k = [1], [], 1

    def p(n: int) -> int:
        nonlocal k
        while (m := len(partitions)) <= n:
            # Extend generalized pentagonal numbers k(3k ± 1)/2
            while not pentagonals or pentagonals[-1][1] <= m:
                sign = 1 if k % 2 == 1 else -1
                pentagonals.append((sign, k * (3 * k - 1) // 2))
                pentagonals.append((sign, k * (3 * k + 1) // 2))
                k += 1

            # Use Euler's recurrence: p(m) = Σ sign * p(m - offset)
            total = 0
            for sign, offset in pentagonals:
                if offset > m: break
                total += sign * partitions[m - offset]

            partitions.append(total if mod is None else total % mod)

        return partitions[n]

    return p

@small_cache
def _euler_transform(a: Callable[[int], int]) -> Callable[[int], int]:
    """
    Return the Euler transform of integer sequence a.

    Parameters
    ----------
    a : Callable(int) -> int
        Integer sequence to transform
    """
    b_values = [1]

    @lru_cache(maxsize=None)
    def c(n: int) -> int:
        return sum(d * a(d) for d in divisors(n))

    def b(n: int) -> int:
        while len(b_values) <= n:
            i = len(b_values)
            total = c(i)
            for k in range(1, i):
                total += c(k) * b_values[i - k]

            b_values.append(total // i)

        return b_values[n]

    return b

def _prime_factor_range(N: int) -> list[int]:
    """
    Find a prime factor for each n = 0, 1, 2, ..., N - 1.
    For composite n, stores the largest prime factor < √N.
    """
    prime_divisor = list(range(N))
    if N >= 1:
        for p in primes(high=isqrt(N-1)):
            prime_divisor[p::p] = [p] * ((N - 1 - p) // p + 1)

    return prime_divisor



########################################################################
########################## Modular Arithmetic ##########################
########################################################################

def egcd(a: int, b: int) -> tuple[int, int, int]:
    """
    Extended Euclidean algorithm.

    Parameters
    ----------
    a : int
        First integer
    b : int
        Second integer

    Returns
    -------
    d : int
        Greatest common divisor of a and b
    x : int
        Coefficient of a in Bézout's identity (ax + by = d)
    y : int
        Coefficient of b in Bézout's identity (ax + by = d)

    Complexity
    ----------
    O(log min(a, b)) time
    """
    d, r = a, b
    x, s = 1, 0

    while r:
        quotient = d // r
        d, r = r, d - quotient * r
        x, s = s, x - quotient * s

    if d < 0:
        d, x = -d, -x

    y = (d - a*x) // b if b != 0 else 0
    return d, x, y

def crt(congruences: Iterable[tuple[int, int]]) -> int | None:
    """
    Solve a system of linear congruences x ≡ aᵢ (mod nᵢ)
    via the Chinese Remainder Theorem.

    Returns a solution to the system of congruences, mod the LCM of the moduli,
    or None if no solution exists.

    Supports non-coprime moduli.

    Parameters
    ----------
    congruences : Iterable[tuple[int, int]]
        Congruences as (residue, moduli) tuples
    """
    try:
        return reduce(_crt_two_congruences, congruences, (0, 1))[0]
    except _NoSolutionError:
        return None

def coprimes(n: int) -> Iterator[int]:
    """
    Generate all integers k in the range [0, n) that are coprime to n.

    Returns the reduced residue system modulo n, i.e., the unit group (Z/nZ)×.
    The size of this set is φ(n) (Euler's totient function).

    For small n, uses an O(n) space sieve for speed. For large n, uses an
    O(1) space generator that checks gcd(k, n) = 1 for each k.

    Parameters
    ----------
    n : int
        Positive integer modulus

    Complexity
    ----------
    O(n * ω(n)) time and O(n) space for n ≤ 10⁷ (sieve approach).
    O(n log n) time and O(1) space for n > 10⁷ (gcd approach).
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    if n < 10_000_000:
        yield from itertools.compress(range(n), _coprime_range(n))
    else:
        yield from (i for i in range(n) if gcd(i, n) == 1)

def multiplicative_order(a: int, mod: int) -> int:
    """
    Compute ordₘ(a), the smallest positive integer such that a^k ≡ 1 (mod m).

    Parameters
    ----------
    a : int
        Integer base
    mod : int
        Integer modulus
    """
    a %= mod
    if gcd(a, mod) != 1:
        raise ValueError("Must have gcd(a, mod) = 1")

    mod = abs(mod)
    order = carmichael(mod)
    for p, e in prime_factorization(order).items():
        for _ in range(e):
            candidate = order // p
            if pow(a, candidate, mod) == 1:
                order = candidate
            else:
                break

    return order

def primitive_root(n: int) -> int | None:
    """
    Find a primitive root modulo n.

    Use Bach's primitive root finding algorithm to search for candidates.

    Parameters
    ----------
    n : int
        Integer modulus
    """
    if -1 <= n <= 1:
        raise ValueError("Must have modulus |n| > 1")
    if n < 0:
        n = -n
    if n in (2, 4):
        return n - 1

    # Check if a primitive root exists
    pf = prime_factorization(n)
    if not ((len(pf) == 1 and (n & 1) == 1) or (len(pf) == 2 and pf.get(2, 0) == 1)):
        return None

    # Find a primitive root mod p
    p = max(pf.keys())
    g = _bach(p)

    # Lift to primitive root mod p^2
    # Any root mod p^2 is a root mod p^e for all e > 1
    if pf[p] > 1:
        g = g + p if pow(g, p - 1, p*p) == 1 else g

    # Force g to be odd
    # Any odd root mod p^e is a root mod 2p^e
    if (n & 1) == 0:
        return g if (g & 1) == 1 else g + n // 2
    else:
        return g

def legendre(a: int, p: int) -> int:
    """
    Compute the Legendre symbol (a | p), where p is an odd prime.

    Parameters
    ----------
    a: int
        Numerator (i.e. quadratic residue class)
    p: int
        Denominator (i.e. prime modulus)
    """
    if p == 2 or not is_prime(p):
        raise ValueError("p must be an odd prime")

    if a < 60:
        L = pow(a % p, (p - 1) // 2, p)
        return -1 if L == p - 1 else L
    else:
        return jacobi(a, p)  # faster for larger inputs

def jacobi(a: int, n: int) -> int:
    """
    Compute the Jacobi symbol (a | n), where n is an odd positive integer.

    Parameters
    ----------
    a: int
        Numerator (i.e. quadratic residue class)
    n: int
        Denominator (i.e. modulus)

    Complexity
    ----------
    O(log a log n) time
    """
    if n <= 0 or not n & 1:
        raise ValueError("n must be an odd positive integer")

    J = 1
    while (a := a % n) != 0:
        # Extract factors of 2 from a
        if (a & 1) == 0:
            s = (a & -a).bit_length() - 1
            a >>= s
            if (s & 1) == 1 and n & 7 in (3, 5):  # s is odd and n = ± 3 (mod 8)
                J = -J

        # Apply quadratic reciprocity
        a, n = n, a
        if a & n & 2:  # a (mod 4) = 3 = n (mod 4)
            J = -J

    return J if n == 1 else 0

def kronecker(a: int, n: int) -> int:
    """
    Compute the Kronecker symbol (a | n).

    Parameters
    ----------
    a: int
        Numerator (i.e. quadratic residue class)
    n: int
        Denominator (i.e. modulus)
    """
    if n == 0:
        return 1 if (a == 1 or a == -1) else 0

    # Calculate sign
    if n > 0:
        sign = 1
    else:
        sign, n = (-1 if a < 0 else 1), -n

    # Factor out powers of 2
    exp = (n & -n).bit_length() - 1
    n >>= exp

    # If both a and n are even, (a | n) = 0
    if (a & 1) == 0 and exp:
        return 0

    # Compute (a | 2)^exp
    K = 1 if (a & 7) in (1, 7) else -1 # check whether a = ± 1 (mod 8)
    if (exp & 1) == 0: K = 1 # check whether exp is even

    return sign * K * jacobi(a % n, n)

def dirichlet_character(m: int, k: int) -> Callable[[int], Number]:
    """
    Return the Dirichlet character χₘ‚ₖ : ℤ → ℂ under Conrey labeling,
    where m is the modulus and k is an index such that gcd(m, k) = 1.

    See: https://www.lmfdb.org/knowledge/show/character.dirichlet.conrey

    Parameters
    ----------
    m : int
        Modulus of the character
    k : int
        Index of the character

    Returns
    -------
    chi : Callable(int) -> Number
        Dirichlet character χₘ‚ₖ(n) as a callable function
        returning the character value at n
    """
    m = abs(m)
    if gcd(m, k) != 1:
        raise ValueError("Must have gcd(m, k) = 1")
    if m == 0:
        raise ZeroDivisionError("Modulus must be nonzero")
    if m == 1:
        return lambda n: 1  # trivial character

    k %= m
    pf = prime_factorization(m)
    characters = [_dirichlet_character_prime_power(p, e, k) for p, e in pf.items()]
    if len(characters) == 1:
        p, chi = next(iter(pf.keys())), characters[0]
        return lambda n: 0 if n % p == 0 else chi(n)
    else:
        return lambda n: 0 if gcd(m, n) > 1 else prod(chi(n) for chi in characters)

def _crt_two_congruences(
    congruence_1: tuple[int, int],
    congruence_2: tuple[int, int],
) -> tuple[int, int]:
    """
    Solve a system of two linear congruences x ≡ a₁ (mod n₁) and x ≡ a₂ (mod n₂)
    via the Chinese remainder theorem.

    Complexity
    ----------
    O(log(max(n₁, n₂))) time
    """
    a1, n1 = congruence_1
    a2, n2 = congruence_2
    diff = a2 - a1
    if diff % (d := gcd(n1, n2)) != 0:
        raise _NoSolutionError("No solution exists for the given pair of congruences.")

    # Reduce to coprime moduli and compute modular inverse
    n1_, n2_ = n1 // d, n2 // d
    k = diff // d
    inv = pow(n1_, -1, n2_)

    # Compute solution
    x = a1 + n1 * ((k * inv) % n2_)
    mod = n1 * n2_  # n1 * n2 // d
    return x % mod, mod

def _coprime_range(N: int) -> bytearray:
    """
    Return whether each integer from 0, 1, 2, ... N - 1 is coprime to N.
    """
    if N < 1:
        return bytearray()
    is_coprime = bytearray(b'\x01') * N
    is_coprime[0] = (N == 1)
    for p in set(_gen_prime_factors(N)):
        is_coprime[p::p] = b'\x00' * ((N - 1) // p)

    return is_coprime

def _bach(p: int) -> int:
    """
    Use Bach's primitive root finding algorithm to search for
    a primitive root modulo p, where p is prime.

    See: https://doi.org/10.1090/S0025-5718-97-00890-9

    Complexity
    ----------
    Las Vegas O((log p)² / (log log p)²) expected multiplications
    """
    if p == 2:
        return 1
    if p == 3:
        return 2

    # Find B such that B log(B) = 30 log(p)
    log_p = ilog(p - 1) + 1  # ⌈log(p)⌉
    log_log_p = ilog(log_p - 1) + 1  # ⌈log⌈log(p)⌉⌉
    B = lower_bound(lambda x: x * ilog(x), 30 * log_p, low=1)

    # Factor φ(p) = p - 1
    pf = prime_factorization(p - 1)

    # Split into a partial factorization with primes q < B
    # and residual Q with primes q >= B
    partial_pf = {q: e for q, e in pf.items() if q < B}
    Q = prod(q**e for q, e in pf.items() if q >= B)

    # Build element of order (p-1)/Q
    # For each q < B, choose b <= 2(log(p))^2 such that b^((p-1)/q) != 1
    a = 1
    b_max = min(2*log_p*log_p, p - 1)
    for q, e in partial_pf.items():
        exponent = (p - 1) // q
        b = secrets.randbelow(b_max - 1) + 2
        while pow(b, exponent, p) == 1:
            b = secrets.randbelow(b_max - 1) + 2

        a = (a * pow(b, (p - 1) // (q**e), p)) % p

    # If Q = 1, a is already a primitive root
    if Q == 1:
        return a

    # Search for b to lift a to a primitive root
    # Assuming the Extended Riemann Hypothesis holds, g = a * b^((p-1)/Q)
    # is a primitive root for some b <= 5(log(p))^4 / (log(log(p)))^2
    exponent = (p - 1) // Q
    b_max = min(5 * -(-(log_p**4) // (log_log_p**2)), p - 1)
    while True:
        # Find b such that b^((p-1)/Q) != 1
        b = secrets.randbelow(b_max - 1) + 2
        while pow(b, exponent, p) == 1:
            b = secrets.randbelow(b_max - 1) + 2

        # Lift by multiplying a * b^((p-1)/Q)
        g = (a * pow(b, exponent, p)) % p

        # Verify solution
        if all(pow(g, (p - 1) // q, p) != 1 for q in pf):
            return g

@small_cache
def _dirichlet_character_prime_power(p: int, e: int, k: int) -> Callable[[int], Number]:
    """
    Return the Dirichlet character χₘ‚ₖ : ℤ → ℂ under Conrey labeling,
    where m = p^e is a prime-power modulus and k is an index such that gcd(m, k) = 1.
    Assumes gcd(p, k) = 1.
    """
    exp, pi = cmath.exp, cmath.pi
    dlog = _dirichlet_log_cache
    k %= (q := p**e)
    if p == 2 and e < 3:
        chi = lambda n: 1 if k == 1 or n % 4 == 1 else -1
    elif p == 2:
        # When q = 2^e > 4, (Z/qZ)× = C_2 × C_{2^{e-2}} = <-1> × <5>
        # Decompose k = ε_a * 5^a (mod q) where ε_a = ±1
        sign_a = -1 if k % 4 == 3 else 1
        a = dlog(sign_a * k, 5, q)

        # Decompose n = ε_b * 5^b (mod q)
        # Character is χ(n) = exp(2πi * ((1-ε_a)(1-ε_b)/8 + ab/2^{e-2}))
        order = 2**(e - 2)
        def chi(n: int) -> Number:
            sign_b = -1 if n % 4 == 3 else 1
            b = dlog(sign_b * n, 5, q)
            t = (a * b) % order
            return exp(2j * pi * ((1 - sign_a)*(1 - sign_b)/8 + t/order))
    else:
        # When q = p^e is an odd prime power, (Z/qZ)× is cyclic
        # Find smallest primitive root g mod p^2 (i.e. the Conrey generator)
        # We have g < 2p, and g will be a primitive root mod p^e for all e > 0
        p2, phi = p * p, p * (p - 1)
        prime_set = set(_gen_prime_factors(phi))
        is_primitive_root = lambda g: all(pow(g, phi // r, p2) != 1 for r in prime_set)
        g = next(i for i in range(2, 2*p) if is_primitive_root(i))

        # Exact values for 1st, 2nd, and 4th roots of unity
        exact = {0: 1}
        order = (p - 1) * p**(e - 1)
        if order % 2 == 0: exact[order // 2] = -1
        if order % 4 == 0: exact[order // 4], exact[3 * order // 4] = 1j, -1j

        # Character is χ(n) = exp(2πi*ab/φ(q)), where g^a = k and g^b = n
        a = dlog(k, g, q)
        def chi(n: int) -> Number:
            t = (a * dlog(n, g, q)) % order
            return exact.get(t) or exp(2j * pi * t / order)

    return chi

@small_cache
def _dirichlet_log_table(b: int, mod: int) -> dict[int, int]:
    """
    Build a log-table of table[x] = a such that b^x = a (mod m).
    """
    a, exp, powers = 1, 0, {1: 0}
    while (a := (a * b) % mod) != 1:
        powers[a] = (exp := exp + 1)

    return powers

@large_cache
def _dirichlet_log_cache(a: int, b: int, mod: int) -> int | None:
    """
    Shared cache for discrete-logs.
    """
    a, b = a % mod, b % mod
    if mod < 10000:
        return _dirichlet_log_table(b, mod).get(a, None)
    else:
        return discrete_log(a, b, mod)



########################################################################
######################### Nonlinear Congruences ########################
########################################################################

def hensel(
    coefficients: Sequence[int],
    p: int,
    k: int,
    initial: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """
    Find all solutions to the polynomial congruence f(x) ≡ 0 (mod pᵏ).

    Assumes f(x) = a₀ + a₁x + a₂x² + a₃x³ ... is a polynomial.
    Uses Hensel lifting to find solutions.

    Parameters
    ----------
    coefficients : Sequence[int]
        Polynomial coefficients, where coefficients[i] = aᵢ is the coefficient for xⁱ
    p : int
        Prime base of modulus
    k : int
        Exponent of modulus
    initial : Iterable[int]
        Initial solutions to f(x) ≡ 0 (mod p)

    Complexity
    ----------
    O(ksd) time, where s is total number of solutions and d = deg(f).
    O(pd) to find initial solutions if not provided.
    """
    if not is_prime(p):
        raise ValueError("p must be prime")
    elif k < 1:
        raise ValueError("Must have k >= 1")

    # Define polynomials f(x) and f'(x)
    f = polynomial(coefficients)
    df = polynomial([i * coefficients[i] for i in range(1, len(coefficients))])

    # Find initial solutions to f(x) = 0 (mod p)
    if initial is None:
        solutions = _polynomial_roots_mod_prime(coefficients, p)
    else:
        solutions = {x % p for x in initial if f(x) % p == 0}

    # Exit early if no solutions or if the exponent is k = 1
    if not solutions or k == 1:
        return tuple(solutions)

    # Hensel lifting to find solutions to f(x) = 0 (mod p^k)
    mod = p
    for _ in range(k - 1):
        new_solutions, new_mod = set(), mod * p
        for root in solutions:
            f_value = f(root) % new_mod
            f_coefficient, df_mod = (f_value // mod) % p, df(root) % p
            if df_mod != 0:
                # Simple root, unique lift
                t = (-f_coefficient * pow(df_mod, -1, p)) % p
                new_solutions.add((root + t*mod) % new_mod)
            elif f_coefficient == 0:
                # Multiple root, p lifts
                new_solutions.update((root + t*mod) % new_mod for t in range(p))

        solutions, mod = new_solutions, new_mod
        if not solutions:
            break

    return tuple(root % mod for root in solutions)

def polynomial_roots(coefficients: Sequence[int], mod: int) -> tuple[int, ...]:
    """
    Find all roots x of a univariate polynomial f(x) ≡ 0 (mod m).

    Factors m into prime powers, finds roots modulo each p^e via
    Cantor-Zassenhaus + Hensel lifting, then combines solutions with CRT.

    Parameters
    ----------
    coefficients : Sequence[int]
        Polynomial coefficients, where coefficients[i] is the coefficient for x^i
    mod : int
        Modulus
    """
    if mod == 0:
        raise ZeroDivisionError("Modulus must be nonzero")

    m = abs(mod)
    coefficients = [c % m for c in coefficients]
    while coefficients and coefficients[-1] == 0:
        coefficients.pop()

    if not coefficients:
        return tuple(range(m))  # zero polynomial, all residues are roots

    residue_sets, moduli = [], []
    for p, e in prime_factorization(m).items():
        roots = _polynomial_roots_mod_prime(coefficients, p)
        roots = roots if e == 1 else hensel(coefficients, p, e, initial=roots)
        if not roots:
            return ()
        residue_sets.append(tuple(roots))
        moduli.append(p**e)

    return tuple(
        crt(zip(residues, moduli))
        for residues in itertools.product(*residue_sets)
    )

def nth_roots(a: int, n: int, mod: int) -> tuple[int, ...]:
    """
    Find all solutions x to x^n ≡ a (mod m).

    Uses the Tonelli-Shanks / Adleman-Manders-Miller to find roots modulo primes,
    Hensel lifting to roots modulo prime powers, and the Chinese Remainder Theorem
    to combine solutions.

    Parameters
    ----------
    a : int
        Target integer
    n : int
        Order of root
    mod : int
        Modulus
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Coefficients to the polynomial f(x) = x^n - a
    coefficients = [-a] + [0]*(n - 1) + [1]

    # Find roots modulo prime powers
    residue_sets, moduli = [], []
    for p, e in prime_factorization(abs(mod)).items():
        roots = _nth_roots_mod_prime(a, n, p)
        roots = roots if e == 1 else hensel(coefficients, p, e, initial=roots)
        if not roots:
            return ()
        residue_sets.append(tuple(roots))
        moduli.append(p**e)

    # Combine solutions via Chinese Remainder Theorem
    return tuple(
        crt(zip(residues, moduli))
        for residues in itertools.product(*residue_sets)
    )

def discrete_log(target: int, base: int, mod: int) -> int | None:
    """
    Find the smallest non-negative integer x such that target ≡ base^x (mod m).
    Returns None if no such integer exists.

    Uses the Pohlig-Hellman algorithm, with either baby-step giant-step or
    Pollard's rho for discrete logarithms on the prime-order sub-problems.

    Parameters
    ----------
    target : int
        Target integer
    base : int
        Base of logarithm
    mod : int
        Modulus
    """
    mod = mod if mod > 0 else -mod
    a, b = target % mod, base % mod

    # Handle edge cases
    if a == 1 or mod == 1:
        return 0

    # Extended reduction to get gcd(b, m) = 1
    # Solving b^x = a * normalization^(-1) (mod m) gives us b^(x + offset) = a (mod m)
    offset, normalization = 0, 1
    while (g := gcd(b, mod)) != 1:
        if a == normalization:
            return offset
        if a % g != 0:
            return None  # no solution exists

        a, mod, offset = a // g, mod // g, offset + 1
        if mod == 1:
            return offset

        b, normalization = b % mod, (normalization * (b // g)) % mod

    # Check early termination conditions
    a = (a * pow(normalization, -1, mod)) % mod  # normalize
    if a == 0:
        return None  # no solution exists for b^x = 0 mod m
    if a == 1:
        return offset

    # Solve a ≡ b^x (mod p^e) for each prime power
    congruences = []
    for p, e in prime_factorization(mod).items():
        try:
            x_i, ord_i = _discrete_log_mod_prime_power(a, b, p, e)
            congruences.append((x_i, ord_i))
        except _NoSolutionError:
            return None  # no solution exists

    # Combine solutions via Chinese Remainder Theorem
    x = crt(congruences)
    return None if x is None else x + offset

def _polynomial_roots_mod_prime(coefficients: Sequence[int], p: int) -> tuple[int, ...]:
    """
    Find all roots of a univariate polynomial f(x) over F_p.

    Uses the Cantor-Zassenhaus algorithm to factor the polynomial, extracting
    roots from linear factors. Computes gcd(f, x^p - x) first to isolate the
    product of linear factors.

    Complexity
    ----------
    O(d² log d log p) expected time for degree-d polynomial
    """
    if not is_prime(p):
        raise ValueError("p must be prime")

    # Handle special cases
    coefficients = [c % p for c in coefficients]
    while coefficients and coefficients[-1] == 0: coefficients.pop()
    if not coefficients: return tuple(range(p))  # zero polynomial
    if len(coefficients) == 1: return ()  # non-zero constant, no roots

    # Brute force for small p where Cantor-Zassenhaus overhead exceeds O(p) iteration
    if p <= 50 + 5 * (len(coefficients) - 1):
        f = polynomial(coefficients, mod=p)
        return tuple(x for x in range(p) if f(x) == 0)

    # Compute gcd(f, x^p - x), the product of all linear factors
    x_term = [0, 1]
    x_to_p = _upoly_fp_powmod(x_term, p, coefficients, p)
    linear_product = _upoly_fp_gcd(coefficients, _upoly_fp_sub(x_to_p, x_term, p), p)
    if len(linear_product) <= 1:
        return ()

    # Factor linear product and extract roots from linear factors
    roots = {
        (-linear[0] * pow(linear[1], -1, p)) % p
        for factor, degree in _cantor_zassenhaus_ddf(linear_product, p) if degree == 1
        for linear in _cantor_zassenhaus_edf(factor, 1, p)
    }

    return tuple(roots)

def _cantor_zassenhaus_ddf(f: list[int], p: int) -> list[tuple[list[int], int]]:
    """
    Distinct-degree factorization of polynomial f over F_p.

    Returns list of (factor, degree) where factor is the product of all
    irreducible factors of that degree.

    See: https://doi.org/10.1090/S0025-5718-1981-0606517-5
    """
    f = _upoly_fp_monic(f, p)
    if len(f) <= 1:
        return []

    factors = []
    x_term = [0, 1]
    frobenius = [0, 1]  # x^(p^i) mod f, the iterated Frobenius map
    degree = 1
    while 2 * degree <= len(f) - 1:
        frobenius = _upoly_fp_powmod(frobenius, p, f, p)
        common = _upoly_fp_gcd(_upoly_fp_sub(frobenius, x_term, p), f, p)
        if common and len(common) > 1:
            factors.append((common, degree))
            f = _upoly_fp_divmod(f, common, p)[0]
            if len(f) <= 1:
                return factors
            frobenius = _upoly_fp_divmod(frobenius, f, p)[1]
        degree += 1

    if f and len(f) > 1:
        factors.append((f, len(f) - 1))

    return factors

def _cantor_zassenhaus_edf(f: list[int], target_degree: int, p: int) -> list[list[int]]:
    """
    Equal-degree factorization of polynomial f over F_p using Cantor-Zassenhaus.

    Given f that is a product of irreducible polynomials all of target_degree,
    returns the list of irreducible factors.

    See: https://doi.org/10.1090/S0025-5718-1981-0606517-5
    """
    deg_f = len(f) - 1
    if deg_f == target_degree:
        return [f]

    if p == 2:
        if target_degree == 1:
            # For p=2: f(0) = f[0] % 2, f(1) = sum(f) % 2
            eval_f2 = lambda g, x: (g[0] if x == 0 else sum(g)) % 2 if g else 0
            roots = [x for x in range(2) if eval_f2(f, x) == 0]
            factors = []
            for root in roots:
                linear = [(-root) % 2, 1]
                while f and eval_f2(f, root) == 0:
                    factors.append(linear)
                    f = _upoly_fp_divmod(f, linear, p)[0]
                if len(f) <= 1:
                    break
            if len(f) > 1:
                factors.append(f)
            return factors

        raise ValueError(
            "Equal-degree factorization over F_2 for d > 1 not supported")

    while True:
        # Try gcd with random polynomial directly
        rand = _upoly_fp_random(len(f) - 2, p)
        common = _upoly_fp_gcd(rand, f, p)

        # If no proper factor, try splitting via rand^((p^d-1)/2) - 1
        if not (common and 1 < len(common) < len(f)):
            exponent = (pow(p, target_degree) - 1) // 2
            splitting = _upoly_fp_sub(_upoly_fp_powmod(rand, exponent, f, p), [1], p)
            common = _upoly_fp_gcd(splitting, f, p)

        # If we found a proper factor, recursively factor both parts
        if common and 1 < len(common) < len(f):
            quotient = _upoly_fp_divmod(f, common, p)[0]
            left = _cantor_zassenhaus_edf(common, target_degree, p)
            right = _cantor_zassenhaus_edf(quotient, target_degree, p)
            return left + right

def _upoly_fp_sub(f: list[int], g: list[int], p: int) -> list[int]:
    """
    Subtract univariate polynomials over F_p. Returns f - g.
    """
    difference = [(a - b) % p for a, b in itertools.zip_longest(f, g, fillvalue=0)]
    while difference and difference[-1] == 0: difference.pop()
    return difference

def _upoly_fp_mul(f: list[int], g: list[int], p: int) -> list[int]:
    """
    Multiply univariate polynomials over F_p. Returns f * g.
    """
    if not f or not g:
        return []

    product = [0] * (len(f) + len(g) - 1)
    for i, coefficient_f in enumerate(f):
        if coefficient_f:
            for j, coefficient_g in enumerate(g):
                product[i + j] = (product[i + j] + coefficient_f * coefficient_g) % p

    while product and product[-1] == 0: product.pop()
    return product

def _upoly_fp_divmod(f: list[int], g: list[int], p: int) -> tuple[list[int], list[int]]:
    """
    Univariate polynomial division with remainder over F_p. Returns (f / g, f % g).
    """
    if not g:
        raise ZeroDivisionError("polynomial division by zero")

    remainder, g = f[:], g[:]
    while g and g[-1] == 0: g.pop()
    if not remainder or len(remainder) < len(g):
        return [], remainder

    inv_lead = pow(g[-1], -1, p)
    quotient = [0] * (len(remainder) - len(g) + 1)
    while remainder and len(remainder) >= len(g):
        c = remainder[-1] * inv_lead % p
        degree = len(remainder) - len(g)
        quotient[degree] = c
        if c:
            for i in range(len(g)):
                remainder[degree + i] = (remainder[degree + i] - c * g[i]) % p
        while remainder and remainder[-1] == 0:
            remainder.pop()

    while quotient and quotient[-1] == 0: quotient.pop()
    return quotient, remainder

def _upoly_fp_monic(f: list[int], p: int) -> list[int]:
    """
    Make univariate polynomial monic over F_p.
    """
    if not f:
        return []
    inv = pow(f[-1], -1, p)
    return f if inv == 1 else [(c * inv) % p for c in f]

def _upoly_fp_gcd(f: list[int], g: list[int], p: int) -> list[int]:
    """
    Univariate polynomial GCD over F_p, returned as monic.
    """
    while g:
        f, g = g, _upoly_fp_divmod(f, g, p)[1]
    return _upoly_fp_monic(f, p)

def _upoly_fp_powmod(base: list[int], exponent: int, g: list[int], p: int) -> list[int]:
    """
    Univariate polynomial exponentiation mod g over F_p via binary exponentiation.
    """
    result = [1]
    base = _upoly_fp_divmod(base, g, p)[1]
    while exponent > 0:
        if exponent & 1:
            result = _upoly_fp_divmod(_upoly_fp_mul(result, base, p), g, p)[1]
        exponent >>= 1
        if exponent:
            base = _upoly_fp_divmod(_upoly_fp_mul(base, base, p), g, p)[1]
    return result

def _upoly_fp_random(max_degree: int, p: int) -> list[int]:
    """
    Generate a random non-zero univariate polynomial over F_p.
    """
    while True:
        coefficients = [secrets.randbelow(p) for _ in range(max_degree + 1)]
        if any(coefficients):
            while coefficients and coefficients[-1] == 0: coefficients.pop()
            return coefficients

def _nth_roots_mod_prime(a: int, n: int, p: int) -> tuple[int, ...]:
    """
    Find all solutions x to x^n ≡ a (mod p), where p is prime.

    Uses the Tonelli-Shanks algorithm when n = 2,
    or the Adleman-Manders-Miller (AMM) algorithm otherwise.
    """
    a %= p
    if n == 1 or a == 0 or p == 2:
        return (a,)
    elif n == 2:
        try:
            r = _tonelli_shanks(a, p)
            return (r, -r % p)
        except _NoSolutionError:
            return ()

    # Use the generalized Euler criterion to test for the existence of an n-th root
    g = gcd(n, p - 1)
    if pow(a, (p - 1) // g, p) != 1:
        return ()

    # If gcd(n, p-1) = 1, unique root via exponent inversion
    if g == 1:
        e = pow(n, -1, p - 1)
        return (pow(a, e, p),)

    # Reduce to a g-th root
    # n = g*n1, p-1 = g*m, gcd(n1, m)=1.
    n1, m = n // g, (p - 1) // g
    inv_n1 = pow(n1, -1, m)

    # We have y^n1 = a (because a^m=1 and inv_n1*n1 = 1 (mod m))
    y = pow(a, inv_n1, p)

    # Solve x^g = y by extracting prime roots along the factorization of g
    pf = prime_factorization(g)
    x = y
    for r, exp in pf.items():
        for _ in range(exp):
            if r == 2:
                x = _tonelli_shanks(x, p)
            else:
                x = _adleman_manders_miller(x, r, p)

    # Find the root of unity ζ^n=1
    e = (p - 1) // g
    omega = next(
        w for a in range(2, p)
        if (w := pow(a, e, p)) != 1
        and all(pow(w, g // q, p) != 1 for q in pf)
    )

    # Now enumerate all n-th roots
    # The set of solutions is {x*ζ where ζ^n = 1}, which is a subgroup of size g
    roots, w = [], 1
    for _ in range(g):
        roots.append((x * w) % p)
        w = (w * omega) % p

    return tuple(roots)

def _tonelli_shanks(a: int, p: int) -> int:
    """
    Tonelli-Shanks algorithm for finding modular square roots.
    Returns a root r such that r² ≡ a (mod p).

    See: https://www.cmat.edu.uy/~tornaria/pub/Tornaria-2002.pdf

    Complexity
    ----------
    O(log p + s²) ⊆ O(log²p) expected multiplications, where p - 1 = 2ˢ * q with q odd
    """
    a %= p
    if a == 0:
        return 0
    elif p == 2:
        return a
    elif p % 4 == 3:
        r = pow(a, (p + 1) // 4, p)
        if r*r % p == a:
            return r
        else:
            raise _NoSolutionError("No solution exists")

    # Write p - 1 as 2^s * q with q odd (by factoring out powers of 2)
    s, q = 0, p - 1
    while q % 2 == 0:
        q //= 2
        s += 1

    # Find a quadratic non-residue
    if p % 8 == 5:
        z = 2
    else:
        # For odd n and p = 1 (mod 4), (p | n) = (n | p) due to quadratic reciprocity
        z = next(n for n in range(3, p, 2) if jacobi(p, n) == -1)

    # Iterative computation to calculate square root
    # Maintain invariant R^2 ≡ a * t (mod p) until t = 1
    M, c, t, R = s, pow(z, q, p), pow(a, q, p), pow(a, (q+1)//2, p)
    while t != 1:
        i, power = 1, (t*t) % p
        while power != 1:
            power = (power*power) % p
            i += 1

        if i >= M:
            raise _NoSolutionError("No solution exists")

        b = pow(c, 2**(M-i-1), p)  # root of unity of order 2^(i+1)
        M = i  # ord(t) = 2^M
        c = (b*b) % p  # root of unity of order 2^i
        t = (t*c) % p  # reduce order of t
        R = (R*b) % p  # update root candidate, maintains R^2 ≡ a * t (mod p)

    return R

def _adleman_manders_miller(delta: int, r: int, p: int) -> int:
    """
    Adleman-Manders-Miller r-th root extraction in finite field Fₚ when r | (p - 1).
    Returns a single root x with x^r = delta (mod p).

    See: https://arxiv.org/pdf/1111.4877
    See: https://www.cs.cmu.edu/~glmiller/Publications/AMM77.pdf

    Complexity
    ----------
    O(t² log r + tr) multiplications, where p - 1 = rᵗ * s
    """
    delta %= p
    if delta == 0:
        return 0
    if r == 1:
        return delta
    if (p - 1) % r != 0:
        raise ValueError("Must have (p - 1) = 0 (mod r)")

    # Use the generalized Euler criterion to test for the existence of an r-th root
    if pow(delta, (p - 1) // r, p) != 1:
        raise _NoSolutionError("No solution exists")

    # Write p - 1 = r^t * s with gcd(r, s) = 1
    t, s = 0, p - 1
    while s % r == 0:
        s //= r
        t += 1

    # Find the smallest α >= 0 such that s | (rα - 1)
    alpha = 0 if s == 1 else pow(r, -1, s)

    # If t = 1 then δ^α is already an r-th root
    if t == 1:
        return pow(delta, alpha, p)

    # Find an r-th non-residue rho
    rho = next(i for i in range(2, p) if pow(i, (p - 1) // r, p) != 1)

    # Initialize algorithm variables
    a = pow(rho, r**(t - 1) * s, p)  # generator of r-th roots of unity (order r)
    b = pow(delta, r*alpha - 1, p)  # satisfies b^(r^(t-1)) = 1
    c = pow(rho, s, p)  # root of unity of order dividing r^t
    h = 1  # accumulates correction factor

    # Iterative computation to calculate an r-th root
    # Maintain invariants b^(r^(t-1)) = 1 (mod p)
    # and (δ^α * h)^r = δ * b^(r^(t-i)) (mod p)
    for i in range(1, t):
        d = pow(b, r**(t - 1 - i), p)
        j = -discrete_log(d, a, p) % r
        h = (h * pow(c, j, p)) % p
        c = pow(c, r, p)
        b = (b * pow(c, j, p)) % p

    return (pow(delta, alpha, p) * h) % p

def _discrete_log_mod_prime_power(a: int, b: int, p: int, e: int) -> tuple[int, int]:
    """
    Solve a ≡ b^x (mod q) in the unit group (Z/qZ)×, where q = p^e.
    Returns both the discrete log x and ord_q(b).
    """
    q = p**e
    a, b = a % q, b % q

    # Solve b^x = a in (Z/qZ)× = C_2 × C_{2^{e-2}} = <-1> × <5>
    if p == 2 and e >= 3:
        # Represent a, b each as (-1)^s * 5^t
        s_a = 0 if a % 4 == 1 else 1
        s_b = 0 if b % 4 == 1 else 1
        t_a = _pohlig_hellman_prime_power(-a % q if s_a == 1 else a, 5, q, 2, e - 2)
        t_b = _pohlig_hellman_prime_power(-b % q if s_b == 1 else b, 5, q, 2, e - 2)

        # Check 5^(t_a) in <5^(t_b)> to determine if a solution exists
        ord_5 = 2**(e - 2)  # size of <5> in (Z/qZ)×
        g = gcd(t_b, ord_5)  # index of <5^{t_b}> inside <5>
        if t_a % g != 0:
            raise _NoSolutionError("No solution exists")  # 5^(t_a) not in <5^(t_b)>

        ord_b = ord_5 // g  # size of the subgroup <5^{t_b}>

        # Handle the degenerate b = ± 1 case (trivial 5-part)
        if ord_b == 1:
            if s_a != 0 and s_b == 0:
                raise _NoSolutionError("No solution exists")
            return (0, 1) if s_b == 0 else (s_a, 2)

        # Solve (t_b/g) * x ≡ (t_a/g) (mod ord_b) in <5>
        inv = pow((t_b // g) % ord_b, -1, ord_b)
        x = ((t_a // g) * inv) % ord_b

        # Enforce sign parity constraint (-1)^{s_a*x} = (-1)^{s_b}
        if (s_b == 0 and s_a != 0) or (s_b == 1 and (x % 2) != s_a):
            raise _NoSolutionError("No solution exists")

        return x, ord_b

    # Solve b^x = a in the cyclic subgroup <b> ≤ (Z/qZ)×
    ord_b = _multiplicative_order_mod_odd_prime_power(b, p, e)
    if pow(a, ord_b, q) != 1:
        raise _NoSolutionError("No solution exists")  # a is not in <b>

    return _pohlig_hellman(a, b, q, ord_b), ord_b

def _pohlig_hellman(h: int, g: int, mod: int, order: int) -> int:
    """
    Pohlig-Hellman algorithm for discrete logarithms.
    Solves g^x = h in the cyclic subgroup <g> of size `order`.

    Complexity
    ----------
    O(∑ eᵢ(log n + √pᵢ)) multiplications, where order n = ∏ pᵢᵉⁱ
    """
    g, h = g % mod, h % mod

    # Validate that g and h lie in the claimed subgroup
    if pow(g, order, mod) != 1 or pow(h, order, mod) != 1:
        raise _NoSolutionError("No solution exists")

    # Handle special case of trivial subgroup
    if order == 1:
        if h == 1 % mod:
            return 0
        else:
            raise _NoSolutionError("No solution exists")

    # Solve g^x = h in each Sylow subgroup of <g> with order p^e
    congruences = []
    for p, e in prime_factorization(order).items():
        q = p**e
        g_i = pow(g, order // q, mod)  # ord(g_i) = q in a cyclic group
        h_i = pow(h, order // q, mod)
        x_i = _pohlig_hellman_prime_power(h_i, g_i, mod, p, e)
        congruences.append((x_i, q))

    # Combine solutions via Chinese Remainder Theorem
    return crt(congruences) % order

def _pohlig_hellman_prime_power(h: int, g: int, mod: int, p: int, e: int) -> int:
    """
    Pohlig-Hellman algorithm for discrete logarithms.
    Solves the discrete logarithm g^x = h in cyclic subgroup <g> of order p^e.

    Complexity
    ----------
    O(e² log p + e√p) multiplications
    """
    # Use BSGS (O(√p) space) for small p, and Pollard-rho (O(1) space) for large p
    discrete_log_solver = _pollard_rho_log if p.bit_length() > 32 else _bsgs
    g, h = g % mod, h % mod
    if e == 1:
        return discrete_log_solver(h, g, mod, p)

    # Find an element with order p
    gamma = pow(g, p**(e - 1), mod)

    # Iteratively compute the p-adic digits of the logarithm
    q = p**e
    x, prime_power, current_target, exponent = 0, 1, h, q
    for i in range(e):
        exponent //= p  # p^(e - 1 - i)
        projected_target = pow(current_target, exponent, mod)
        digit = discrete_log_solver(projected_target, gamma, mod, p)
        x += digit * prime_power
        current_target *= pow(g, (-digit * prime_power) % q, mod)  # h * g^(-x)
        current_target %= mod
        prime_power *= p

    return x % q

def _bsgs(h: int, g: int, mod: int, p: int) -> int:
    """
    Baby-step giant-step algorithm for discrete logarithms.
    Solves the discrete logarithm g^x = h in cyclic group <g> of prime order p.

    Complexity
    ----------
    O(√p) multiplications and space, where p is order of g
    """
    g, h = g % mod, h % mod
    if p == 2:
        if h == 1 % mod: return 0
        if h == g % mod: return 1
        raise _NoSolutionError("No solution in order-2 subgroup")

    table, m, g_m_inv = _bsgs_table(g % mod, mod, p)
    y = h % mod
    for i in range(m):
        j = table.get(y)
        if j is not None:
            return (i*m + j) % p
        else:
            y = (y * g_m_inv) % mod

    raise _NoSolutionError("No solution found (BSGS)")

@small_cache
def _bsgs_table(g: int, mod: int, p: int) -> tuple[dict[int, int], int, int]:
    """
    Computes g⁰, g¹, ..., gᵐ where m = ⌈√p⌉ and stores {gʲ: j}.
    Also returns g⁻ᵐ for giant-step phase.
    """
    m = isqrt(p - 1) + 1
    powers = itertools.accumulate([g] * m, lambda a, b: (a * b) % mod, initial=1)
    table = {power: exponent for exponent, power in enumerate(powers)}
    g_m_inv = pow(pow(g, m, mod), -1, mod)
    return table, m, g_m_inv

def _pollard_rho_log(h: int, g: int, mod: int, p: int, partition_size: int = 32) -> int:
    """
    Pollard Rho algorithm for discrete logarithms.
    Finds x such that g^x ≡ h (mod m), where p is the order of g.
    Uses Brent's algorithm for finding cycles.

    Complexity
    ----------
    Las Vegas with expected O(√p) multiplications and O(1) space
    """
    g, h = g % mod, h % mod

    # Validate that g and h lie in the claimed subgroup
    if pow(g, p, mod) != 1 or pow(h, p, mod) != 1:
        raise _NoSolutionError("No solution exists")

    partition_size = 1 << (partition_size - 1).bit_length()
    mask = partition_size - 1
    max_iterations = 6 * isqrt(p) + 200

    # Adaptive reduction interval based on order size
    bits = p.bit_length()
    reduce_mask = 255 if bits <= 35 else (127 if bits <= 45 else 63)

    while True:
        # Build random multiplier tables
        a_table = [secrets.randbelow(p) for _ in range(partition_size)]
        b_table = [secrets.randbelow(p) for _ in range(partition_size)]
        m_table = [
            pow(g, a, mod) * pow(h, b, mod) % mod for a, b in zip(a_table, b_table)]

        # Random starting point
        a0, b0 = secrets.randbelow(p), secrets.randbelow(p)
        x0 = pow(g, a0, mod) * pow(h, b0, mod) % mod

        # Brent's cycle detection
        x, a, b = x_t, a_t, b_t = x0, a0, b0
        interval, cycle_length = 1, 0
        for j in range(max_iterations):
            i = (x ^ (x >> 32)) & mask # hash
            x = x * m_table[i] % mod
            a += a_table[i]
            b += b_table[i]
            cycle_length += 1

            # Periodically reduce exponents mod p to prevent overflow
            if j & reduce_mask == reduce_mask:
                a, b, a_t, b_t = a % p, b % p, a_t % p, b_t % p

            # Collision detected, solve g^a * h^b ≡ g^a_t * h^b_t for discrete log
            if x == x_t:
                r = (b_t - b) % p
                if r != 0 and gcd(r, p) == 1:
                    result = ((a - a_t) % p) * pow(r, -1, p) % p
                    if pow(g, result, mod) == h:
                        return result
                break

            # Brent checkpoint reached, save position and double checkpoint interval
            if interval == cycle_length:
                x_t, a_t, b_t = x, a, b
                interval, cycle_length = interval * 2, 0

def _multiplicative_order_mod_odd_prime_power(a: int, p: int, e: int) -> int:
    """
    Return the smallest integer k = ord_n(a) such that a^k ≡ 1 (mod n),
    where n = p^e is an odd prime power.
    """
    a %= (n := p**e)
    if gcd(a, n) != 1:
        raise ValueError("Must have gcd(a, p^e) = 1")

    # Set initial order as λ(n)
    order = (p - 1) * (p**(e - 1))

    # Get prime factorization of λ(n)
    pf = prime_factorization(p - 1)
    pf[p] = e - 1

    # Find ord_n(a)
    for q, exp in pf.items():
        for _ in range(exp):
            candidate = order // q
            if pow(a, candidate, n) == 1:
                order = candidate
            else:
                break

    return order



########################################################################
######################## Diophantine Equations #########################
########################################################################

def bezout(a: int, b: int, c: int) -> Iterator[tuple[int, int]]:
    """
    Generate all integer solutions to the linear Diophantine equation ax + by = c.

    Uses the extended Euclidean algorithm to find a pair of Bézout coefficients,
    and then generate an infinite family of solutions.

    Parameters
    ----------
    a : int
        Coefficient of x
    b : int
        Coefficient of y
    c : int
        Constant term

    Yields
    ------
    x : int
        X-coordinate of solution
    y : int
        Y-coordinate of solution

    Complexity
    ----------
    O(log(min(a, b))) time to find initial solution, O(1) per additional solution.
    """
    d, x0, y0 = egcd(a, b)
    if d == 0:
        yield from (integer_pairs() if c == 0 else ())
        return

    # Check if any solutions exist
    if c % d != 0:
        return

    # Scale particular solution
    x0 *= c // d
    y0 *= c // d

    # Generate all solutions (x0 + kb/d, y0 - ka/d) for k ∈ ℤ
    step_x, step_y = b // d, a // d

    # Yield solutions in order (k = 0, 1, -1, 2, -2, 3, -3, ...)
    yield (x0, y0)
    for k in itertools.count(start=1):
        yield (x0 + k * step_x, y0 - k * step_y)
        yield (x0 - k * step_x, y0 + k * step_y)

def cornacchia(d: int, m: int) -> Iterator[tuple[int, int]]:
    """
    Generate all unique positive integer solutions to the equation x² + dy² = m
    where 0 < d < m and gcd(d, m) = 1.

    Parameters
    ----------
    d : int
        Coefficient of y² term
    m : int
        Constant term

    Yields
    ------
    x : int
        X-coordinate of solution
    y : int
        Y-coordinate of solution

    Complexity
    ----------
    O(f(m) + τ(m) g(m) + τ(m) log m) time, where f, τ, g are the cost of
    factorization, divisor count, and cost of modular roots respectively.
    """
    if not 0 < d < m:
        raise ValueError("Must have 0 < d < m")
    if gcd(d, m) != 1:
        raise ValueError("Must have gcd(d, m) = 1")

    # Collect scale factors g where g^2 | m
    factors = [1]
    for p, e in prime_factorization(m).items():
        pk, new = 1, []
        for _ in range(e // 2):
            pk *= p
            new.extend(g * pk for g in factors)
        factors.extend(new)

    # Find solutions
    solutions = set()
    for g in factors:
        n = m // (g * g)
        sqrt_n = isqrt(n)
        for r in nth_roots(-d, 2, mod=n):
            if r > n // 2:
                r = n - r

            # Euclidean reduction until b <= sqrt(n)
            a, b = n, r
            while b > sqrt_n:
                a, b = b, a % b

            # Validate x solution
            x = b
            residual = n - x*x
            if x == 0 or residual <= 0 or residual % d:
                continue

            # Validate y solution
            y_squared = residual // d
            y = isqrt(y_squared)
            if y == 0 or y*y != y_squared:
                continue

            # Yield solution
            solution = (g*x, g*y)
            if solution not in solutions:
                solutions.add(solution)
                yield solution

                # Solution is symmetric when d = 1 (x^2 + dy^2 = y^2 + dx^2 = m)
                if d == 1 and x != y and (solution := (g*y, g*x)) not in solutions:
                    solutions.add(solution)
                    yield solution

def pell(D: int, N: int = 1) -> Iterator[tuple[int, int]]:
    """
    Generate all unique positive integer solutions to the generalized Pell equation
    x² - Dy² = N, where D is not a perfect square.

    Yields infinite positive integer solutions x, y > 0 in order of increasing x.
    Uses the Lagrange-Matthews-Mollin (LMM) algorithm.

    See: https://cjhb.site/Files.php/Books/math/B3.4/pell.pdf
    See: http://www.numbertheory.org/PDFS/patz_improved.pdf

    Parameters
    ----------
    D : int
        Coefficient of y² term
    N : int
        Constant term

    Yields
    ------
    x : int
        X-coordinate of solution
    y : int
        Y-coordinate of solution

    Complexity
    ----------
    O(L(D) + f(|N|) + τ(|N|) * (g(|N|) + L(D))) time, where L, f, τ, g are the
    continued-fraction period length, cost of factoring, divisor count,
    and cost of modular roots respectively.
    """
    if D <= 0:
        raise ValueError("D must be a positive integer")
    if is_square(D):
        raise ValueError("D cannot be a perfect square")

    # Exit early if N = 0 has only the trivial solution
    if N == 0: return

    # Get convergents for continued fraction of sqrt(D)
    coefficients, initial, period = periodic_continued_fraction(D)
    pell_convergents = list(convergents(coefficients, num=initial+2*period))

    # Find minimal solution to x^2 - Dy^2 = -1
    solutions = ((x, y) for x, y in pell_convergents if x*x - D*y*y == -1)
    t, u = next(solutions, (None, None))

    # Find fundamental solutions to x^2 - Dy^2 = N
    sqrt_D = isqrt(D)
    fundamental_solutions = []
    for f in divisors(N):
        m, remainder = divmod(N // f, f)
        if remainder: continue

        # Iterate over modular roots
        m = abs(m)
        for z in nth_roots(D, 2, mod=m):
            z = z if z <= m // 2 else z - m
            a, initial, period = periodic_continued_fraction(D, P=z, Q=m)
            a = [next(a) for _ in range(initial + period)]
            if len(a) < 2: continue
            A, B = zip(*convergents(a[:-1]))
            target = N // (f * f)
            for i in range(1, len(a)):
                if abs(a[i]) <= sqrt_D:
                    continue
                r, s = m*A[i-1] - z*B[i-1], B[i-1]
                value = r*r - D*s*s
                if value == target:
                    fundamental_solutions.append((f*r, f*s))
                elif value == -target and (t, u) != (None, None):
                    x, y = f*r, f*s
                    fundamental_solutions.append(((x*t + y*u*D), (x*u + y*t)))

    # Find minimal solution to x^2 - Dy^2 = 1
    t0, u0 = next((x, y) for x, y in pell_convergents if x*x - D*y*y == 1)

    # Find minimal positive solutions to x^2 - Dy^2 = N
    minimal_positive_solutions = set()
    for x, y in fundamental_solutions:
        if x > 0 and y > 0:
            x, y = x, y
        elif x < 0 and y < 0:
            x, y = -x, -y
        else:
            x, y = (-x*t0 + -y*u0*D, -x*u0 + -y*t0)

        # Reduce within the Pell class to the minimal positive solution.
        while min(solution := (x*t0 - D*y*u0, y*t0 - x*u0)) > 0:
            x, y = solution

        minimal_positive_solutions.add((x, y))

    # Yield minimal positive solutions to x^2 - Dy^2 = N
    minimal_positive_solutions = sorted(minimal_positive_solutions)
    yield from minimal_positive_solutions
    if not minimal_positive_solutions:
        return

    # Yield additional solutions to x^2 - Dy^2 = N
    t, u = t0, u0
    while True:
        for r, s in minimal_positive_solutions:
            yield r*t + s*u*D, r*u + s*t

        t, u = t0*t + D*u0*u, t0*u + u0*t

def conic(a: int, b: int, c: int, d: int, e: int, f: int) -> Iterator[tuple[int, int]]:
    """
    Generate all unique integer solutions (x, y) to the binary quadratic Diophantine
    conic equation ax² + bxy + cy² + dx + ey + f = 0.

    Uses the theory of binary quadratic forms, classifying by discriminant Δ = b² - 4ac:

        Δ < 0 (ellipse): Lagrange reduction, finite solutions
        Δ = 0 (parabola): parametric families via modular square roots
        Δ > 0 (hyperbola): reduction to Pell equation, infinite solutions
        Degenerate cases: factorization into linear forms

    Parameters
    ----------
    a : int
        Coefficient of x² term
    b : int
        Coefficient of xy term
    c : int
        Coefficient of y² term
    d : int
        Coefficient of x term
    e : int
        Coefficient of y term
    f : int
        Constant term

    Yields
    ------
    x : int
        X-coordinate of solution
    y : int
        Y-coordinate of solution
    """
    # Handle trivial form
    if a == b == c == d == e == 0:
        yield from (integer_pairs() if f == 0 else ())
        return
    
    # Remove common factor from coefficients
    if (g := gcd(a, b, c, d, e, f)) > 1:
        a, b, c, d, e, f = a // g, b // g, c // g, d // g, e // g, f // g

    # Handle linear in x (missing x^2 term)
    if a == 0:
        yield from _conic_linear_in_x(b, c, d, e, f)
        return

    discriminant = b*b - 4*a*c
    scaled_determinant = -f*discriminant + b*d*e - c*d*d - a*e*e
    if scaled_determinant == 0:
        yield from _degenerate_conic(a, b, c, d, e, f)
    elif discriminant > 0:
        yield from _hyperbola(a, b, c, d, e, f)
    elif discriminant == 0:
        yield from _parabola(a, b, c, d, e, f)
    elif discriminant < 0:
        yield from _ellipse(a, b, c, d, e, f)

def pythagorean_triples(
    max_c: float | None = None,
    max_sum: float | None = None,
) -> Iterator[tuple[int, int, int]]:
    """
    Generate positive integer solutions to the equation a² + b² = c².

    Uses Euclid's formula to generate unique Pythagorean triples (a, b, c)
    where a ≤ b ≤ c.

    If no bounds are specified, infinitely generates triples in order of increasing c.
    When bounds are specified, no order is guaranteed.

    Parameters
    ----------
    max_c : float
        Upper bound for c in generated triples, where c ≤ max_c
    max_sum : float
        Upper bound for the sum of generated triples, where a + b + c ≤ max_sum
    """
    max_m = None
    if max_c is not None:
        max_c = int(max_c)
        max_m = min(max_m or inf, isqrt(max_c))
    if max_sum is not None:
        max_sum = int(max_sum)
        max_m = min(max_m or inf, isqrt(max_sum // 2))

    # Bounded case
    if max_m is not None:
        for a, b, c in _euclid(max_m=max_m):
            # Generate multiples of primitive triple
            if max_c is not None and max_sum is not None:
                max_k = min(max_c // c, max_sum // (a + b + c))
            elif max_sum is not None:
                max_k = max_sum // (a + b + c)
            else:
                max_k = max_c // c
            for k in range(1, int(max_k) + 1):
                yield (k*a, k*b, k*c)

        return

    # Unbounded case
    queue = []  # (current_c, k, a0, b0, c0)
    primitive_triples = _berggren()
    a0, b0, c0 = next(primitive_triples)
    while True:
        # Queue primitive triples (a0, b0, c0)
        while not queue or c0 <= queue[0][0]:
            heappush(queue, (c0, 1, a0, b0, c0))
            a0, b0, c0 = next(primitive_triples)

        # Yield the next triple (ka, kb, kc)
        _, k, a, b, c = heappop(queue)
        yield (k*a, k*b, k*c)

        # Queue the next multiple of (a, b, c)
        k += 1
        heappush(queue, (k*c, k, a, b, c))

def pillai(a: int, b: int, c: int) -> Iterator[tuple[int, int]]:
    """
    Generate all positive integer solutions (x, y) to the exponential Diophantine
    Pillai equation aˣ - bʸ = c, where a, b >= 2 and x, y > 0.

    Parameters
    ----------
    a : int
        Base of x term
    b : int
        Base of y term
    c : int
        Integer target

    Yields
    ------
    x : int
        X-coordinate of solution
    y : int
        Y-coordinate of solution
    """
    if a < 2 or b < 2:
        raise ValueError("Bases a and b must be >= 2")

    # Use Baker-type bound for provable completeness
    x_max = _pillai_bound(a, b, c)
    a_x_max = pow(a, x_max)
    if a_x_max <= c:
        return
    y_max = ilog(a_x_max - c, b)

    # Select primes p where gcd(p, ab) = 1 and compute multiplicative orders
    max_prime_count = max(3, y_max.bit_length() // 4)
    sieve_primes, orders, modulus = [], [], 1
    for p in primes(low=max(1000, y_max)):
        if a % p == 0 or b % p == 0: continue
        order = multiplicative_order(a, p)
        new_lcm = lcm(modulus, order)
        if new_lcm > modulus:
            modulus = new_lcm
            sieve_primes.append(p)
            orders.append(order)
        if len(sieve_primes) >= max_prime_count or modulus > y_max:
            break

    # Use a discrete log sieve to restrict the search space
    # For each p where gcd(p, ab) = 1, we need a^x = (c + b^y) (mod p)
    # We can take the discrete log for each as x = dlog_a(c + b^y) (mod ord_p(a))
    # and combine these constraints via the Chinese Remainder Theorem
    a_mod = [a % p for p in sieve_primes]
    b_mod = [b % p for p in sieve_primes]
    c_mod = [c % p for p in sieve_primes]
    b_pow_y, b_pow_y_mod = b, [b % p for p in sieve_primes]
    solutions = []
    for y in range(1, y_max + 1):
        residual = c + b_pow_y  # a^x = c + b^y
        if residual > 0:
            congruences = []
            for i, p in enumerate(sieve_primes):
                target_mod_p = (c_mod[i] + b_pow_y_mod[i]) % p
                if target_mod_p == 0: break
                x_mod_order = discrete_log(target_mod_p, a_mod[i], p)
                if x_mod_order is None: break
                congruences.append((x_mod_order % orders[i], orders[i]))
            else:
                x_base = crt(congruences)
                if x_base is not None:
                    x_start = x_base if x_base >= 1 else x_base + modulus
                    for x in range(x_start, x_max + 1, modulus):
                        if pow(a, x) == residual:
                            solutions.append((x, y))

        # Update b^y -> b^(y+1)
        if y < y_max:
            b_pow_y *= b
            for i in range(len(sieve_primes)):
                b_pow_y_mod[i] = (b_pow_y_mod[i] * b_mod[i]) % sieve_primes[i]

    yield from sorted(solutions)

def _parabola(*coefficients: int) -> Iterator[tuple[int, int]]:
    """
    Solve ax² + bxy + cy² + dx + ey + f = 0 when discriminant Δ = b² - 4ac = 0.

    Complexity
    ----------
    O(|A|) preprocessing where A = 2bd - 4ae for modular square roots, O(1) per solution
    """
    a, b, c, d, e, f = coefficients

    # Reduce to a quadratic in x as ax^2 + (by + d)x + (cy^2 + ey + f) = 0
    # The discriminant in x becomes (by + d)^2 - 4a(cy^2 + ey + f) = Ay + B
    # When Δ = 0, this becomes Ay + B where A = 2bd - 4ae and B = d^2 - 4af
    # For integer x = (-(by + d) ± z) / 2a, we need Ay + B = z^2 as a perfect square
    A, B = 2*b*d - 4*a*e, d*d - 4*a*f
    if A == 0 and is_square(B):
        r = isqrt(B)
        family_1 = bezout(2*a, b, r - d)
        family_2 = () if r == 0 else bezout(2*a, b, -r - d)
        yield from alternating(family_1, family_2)
    elif A != 0:
        # Parameterize solutions by z = z0 (mod |A|) with (z0)^2 = B (mod |A|),
        # then recover y and x
        def family(z0: int) -> Iterator:
            for _, z in bezout(A, 1, z0):
                y = (z*z - B) // A
                x, remainder = divmod(z - d - b*y, 2*a)
                if remainder == 0:
                    yield (x, y)

        # We have a family of solutions for each residue z such that z^2 = B (mod |A|)
        residues = sorted(nth_roots(B, 2, abs(A)))
        yield from alternating(*(family(r) for r in residues))

def _ellipse(*coefficients: int) -> Iterator[tuple[int, int]]:
    """
    Solve ax² + bxy + cy² + dx + ey + f = 0 when discriminant Δ = b² - 4ac < 0.

    Complexity
    ----------
    O(log² max(|a|,|b|,|c|)) preprocessing for Lagrange reduction, O(1) per solution
    """
    a, b, c, d, e, f = coefficients
    if a < 0:
        a, b, c, d, e, f = -a, -b, -c, -d, -e, -f
    if a == 0:
        return

    # Use Lagrange reduction to transform to an equivalent form with |b| <= a <= c
    # Track the transformation matrix [[p, q], [r, s]] where (x, y) = (pX + qY, rX + sY)
    p, q, r, s = 1, 0, 0, 1
    while not (abs(b) <= a <= c):
        if a > c:
            a, b, c = c, -b, a
            p, q, r, s = -q, p, -s, r
        elif abs(b) > a:
            k = (b + a) // (2 * a)
            b, c = b - 2*a*k, a*k*k - b*k + c
            q, s = q - k*p, s - k*r

    # Transform linear terms
    d, e = d*p + e*r, d*q + e*s

    # Reduce to a quadratic in X as aX^2 + (bY + d)X + (cY^2 + eY + f) = 0
    # This has real solutions only when its discriminant is >= 0
    # We can rewrite this inequality as (b^2 - 4ac)Y^2 + (2bd - 4ae)Y + (d^2 - 4af) >= 0
    A, B, C = b*b - 4*a*c, 2*b*d - 4*a*e, d*d - 4*a*f
    D = B*B - 4*A*C
    if D < 0:
        return

    # Solve AY^2 + BY + C >= 0 to find Y bounds
    sqrt_D = isqrt(D)
    Y1, Y2 = (-B - sqrt_D) // (2 * A), -((B - sqrt_D) // (2 * A))

    # Enumerate Y in the valid range and solve for X
    # Solve aX^2 + (bY + d)X + (cY^2 + eY + f) = 0 using quadratic formula
    for Y in range(min(Y1, Y2) - 1, max(Y1, Y2) + 2):
        for X in _integer_quadratic_roots(a, b*Y + d, c*Y*Y + e*Y + f):
            yield (p*X + q*Y, r*X + s*Y)  # transform back to (x, y) solutions

def _hyperbola(*coefficients: int) -> Iterator[tuple[int, int]]:
    """
    Solve ax² + bxy + cy² + dx + ey + f = 0 when discriminant Δ = b² - 4ac > 0.

    Complexity
    ----------
    O((√Δ) log Δ) for fundamental Pell solutions, O(1) per additional solution
    """
    a, b, c, d, e, f = coefficients

    # Reduce to Pell-type equation u^2 - Δv^2 = N
    # where v = 2ax + by + d, u = Δy - β, β = 2ae - bd, and N = β^2 + Δ(4af - d^2)
    discriminant, beta = b*b - 4*a*c, 2*a*e - b*d
    N = beta*beta + discriminant * (4*a*f - d*d)

    def get_solutions(uv_values: Iterable) -> Iterator:
        for u, v in uv_values:
            y, remainder = divmod(u + beta, discriminant)
            if remainder: continue
            x, remainder = divmod(v - b*y - d, 2*a)
            if remainder: continue
            yield (x, y)

    # Compute general solutions
    s = isqrt(discriminant)
    if s * s == discriminant:
        # For square discriminant we have u^2 - s^2v^2 = (u + sv)(u - sv) = N
        if N == 0:
            # If N = 0, we have u^2 = s^2v^2, so u = ± sv
            yield from get_solutions((u, v) for v in integers() for u in (s*v, -s*v))
        else:
            # If N != 0, enumerate divisor pairs P, Q with product P * Q = |N|
            for P in divisors(N):
                if P > abs(Q := N // P): break
                U = (P + Q) // 2
                V, remainder = divmod(Q - P, 2*s)
                if remainder: continue
                yield from get_solutions({(U, V), (U, -V), (-U, V), (-U, -V)})
    else:
        # If u = 0 in Pell equation, we have v^2 = -N/Δ
        if beta % discriminant == 0 and is_square(-N // discriminant):
            v = isqrt(-N // discriminant)  # Δ | β implies Δ | N
            yield from get_solutions([(0, v), (0, -v)])

        # If v = 0 in Pell equation, we have u^2 = N
        if N > 0 and is_square(N):
            u = isqrt(N)
            yield from get_solutions([(u, 0), (-u, 0)])

        # Solve the generalized Pell equation
        for U, V in pell(discriminant, N):
            yield from get_solutions({(U, V), (U, -V), (-U, V), (-U, -V)})

def _degenerate_conic(*coefficients: int) -> Iterator[tuple[int, int]]:
    """
    Solve ax² + bxy + cy² + dx + ey + f = 0 when the conic is degenerate.
    """
    a, b, c, d, e, f = coefficients

    # Multiplying by 4a, 4a(ax^2 + bxy + cy^2 + dx + ey + f) = 0
    # becomes (2ax + (b-s)y)(2ax + (b+s)y) + 4adx + 4aey + 4af = 0
    # which becomes (2ax + (b-s)y + g)(2ax + (b+s)y + h) = 0
    # where s^2 = Δ, 4ad = 2a(g+h), 4ae = (b+s)g + (b-s)h, and 4af = gh,
    # which implies g, h are roots of quadratic t^2 - 2dt + 4af
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        # Degenerate ellipse (single point)
        yield from _ellipse(*coefficients)
    elif discriminant == 0:
        # Degenerate parabola (two parallel lines)
        if is_square(d*d - 4*a*f):
            offset = isqrt(d*d - 4*a*f)
            g, h = d - offset, d + offset
            line_1 = bezout(2*a, b, -g)
            line_2 = bezout(2*a, b, -h) if offset != 0 else ()
            yield from alternating(line_1, line_2)
    elif discriminant > 0:
        # Degenerate hyperbola (two intersecting lines)
        if is_square(d*d - 4*a*f):
            s, offset = isqrt(discriminant), isqrt(d*d - 4*a*f)
            if 4*a*e - 2*b*d > 0:
                g, h = d + offset, d - offset
            else:
                g, h = d - offset, d + offset

            # Find intersection
            line_1, line_2 = bezout(2*a, b - s, -g), bezout(2*a, b + s, -h)
            y_int, remainder_y = divmod(g - h, 2*s)
            x_int, remainder_x = divmod((b-s)*h - (b+s)*g, 4*a*s)
            if remainder_x == 0 and remainder_y == 0:
                line_2 = (point for point in line_2 if point != (x_int, y_int))

            yield from alternating(line_1, line_2)

def _conic_linear_in_x(*coefficients) -> Iterator[tuple[int, int]]:
    """
    Solve bxy + cy² + dx + ey + f = 0 (no x² term).
    """
    b, c, d, e, f = coefficients
    quadratic = lambda y: c*y*y + e*y + f
    if b == 0 and c == 0:
        # Reduces to dx + ey + f = 0
        yield from bezout(d, e, -f)
    elif b == 0 and c != 0 and d == 0:
        # Reduces to cy^2 + ey + f = 0
        y_roots = _integer_quadratic_roots(c, e, f)
        yield from alternating(*(bezout(0, 1, y) for y in y_roots))
    elif b == 0 and c != 0 and d != 0:
        # Reduces to x = -(cy^2 + ey + f) / d
        def family(y0: int) -> Iterator:
            for _, y in bezout(d, 1, y0):
                x, remainder = divmod(quadratic(y), -d)
                if remainder == 0:
                    yield (x, y)

        y_residues = [y for y in range(abs(d)) if quadratic(y) % d == 0]
        yield from alternating(*(family(y) for y in y_residues))
    elif b != 0 and d == 0 and f == 0:
        # Reduces to y(bx + cy + e) = 0
        yield from alternating(
            ((x, 0) for x in integers()),
            ((x, y) for x, y in bezout(b, c, -e) if y != 0)
        )
    elif b != 0 and d == 0 and f != 0:
        # Reduces to x = -(f/y + cy + e) / b
        for divisor in divisors(f):
            for y in (divisor, -divisor):
                x, remainder = divmod(f // y + c*y + e, -b)
                if remainder == 0:
                    yield (x, y)
    else:
        # Reduces to x = (cy^2 + ey + f) / (by + d)
        # Polynomial division of numerator by denominator
        # gives cy^2 + ey + f = (by + d) * Q(y) + R
        # where R = cd^2 - ebd + fb^2 is the constant remainder
        # It follows that (by + d) | (cy^2 + ey + f) if and only if (by + d) | R
        R = c*d*d - e*b*d + f*b*b

        # Check for singular line: y = -d/b makes entire equation vanish
        y_singular, remainder = divmod(-d, b)
        singular_line = (remainder == 0 and quadratic(y_singular) == 0)

        # Generate all non-singular solutions
        def other_solutions():
            if R == 0:
                # The numerator factors as (by + d) × (linear), giving line solutions
                # Line solutions from: bx + bcy + (eb - cd) = 0
                for x, y in bezout(b, b*c, c*d - e*b):
                    if not (singular_line and y == y_singular):
                        yield (x, y)
            else:
                # Finitely many points where (by + d) must divide R
                for divisor in divisors(R):
                    for k in (divisor, -divisor):
                        y, remainder = divmod(k - d, b)
                        if remainder: continue
                        x, remainder = divmod(quadratic(y), -k)
                        if remainder == 0:
                            yield (x, y)

        yield from alternating(
            bezout(0, 1, y_singular) if singular_line else (),
            other_solutions(),
        )

def _integer_quadratic_roots(A: int, B: int, C: int) -> list[int]:
    """
    Return integer roots to the quadratic equation Ax² + Bx + C = 0.
    """
    D = B*B - 4*A*C
    if D < 0 or not is_square(D):
        return []

    roots, sqrt_D = [], isqrt(D)
    for sign in (0,) if D == 0 else (1, -1):
        X, remainder = divmod(-B + sign*sqrt_D, 2*A)
        if remainder == 0:
            roots.append(X)

    return roots

def _euclid(max_m: int | None = None) -> Iterator[tuple[int, int, int]]:
    """
    Generate unique primitive Pythagorean triples (a, b, c) with Euclid's formula,
    where a ≤ b ≤ c.
    """
    for m in (itertools.count(start=2) if max_m is None else range(2, max_m + 1)):
        for n in coprimes(m):
            if (m + n) % 2 == 1:
                m_squared, n_squared = m*m, n*n
                a, b, c = m_squared - n_squared, 2*m*n, m_squared + n_squared
                if a > b:
                    a, b = b, a
                yield (a, b, c)

def _berggren() -> Iterator[tuple[int, int, int]]:
    """
    Generate primitive Pythagorean triples (a, b, c) with Berggren's tree method,
    where a ≤ b ≤ c, and triples are generated in order of increasing c.
    """
    triples = [(5, 3, 4)]
    while triples:
        c, a, b = heappop(triples)
        if a > b:
            a, b = b, a
        yield (a, b, c)

        # Apply Berggren's transformations
        heappush(triples, (2*a - 2*b + 3*c, a - 2*b + 2*c, 2*a - b + 2*c))
        heappush(triples, (2*a + 2*b + 3*c, a + 2*b + 2*c, 2*a + b + 2*c))
        heappush(triples, (-2*a + 2*b + 3*c, -a + 2*b + 2*c, -2*a + b + 2*c))

def _pillai_bound(a: int, b: int, c: int) -> int:
    """
    Rigorous bound on B = max(x, y) for solutions to a^x - b^y = c with x, y > 0
    using Laurent–Mignotte–Nesterenko (1995) explicit 2-log lower bound.

    Note that if c = 0 there is no finite bound in general
    (e.g. a = b gives infinitely many).

    See: https://pub.math.leidenuniv.nl/~evertsejh/dio14-5.pdf
    See: https://www.sciencedirect.com/science/article/pii/S0022314X85711419
    """
    if a < 2 or b < 2:
        raise ValueError("Bases a and b must be >= 2")
    if c == 0:
        raise ValueError("No finite bound for c = 0 in general")

    log_a, log_b = log(a), log(b)
    m, M = (log_a, log_b) if log_a <= log_b else (log_b, log_a)

    # If y is small enough that b^y < 2|c|, it's already bounded by log(2|c|)/log b,
    # and similarly for x. Using min(log a, log b) gives a uniform U covering both.
    U = log(2*abs(c)) / m  # abs(c) >= 1 here

    K = 24.34  # LMN constant (as quoted by Evertse)
    # Start from the "21" floor in LMN (so we don't need B inside the log yet)
    B = max(2, ceil(U + K * M * (21.0 ** 2)))

    for _ in range(200):
        # LMN uses max{ log( x/log b + y/log a ) + 0.14, 21 }
        # We upper bound (x/log b + y/log a) <= B*(1/log a + 1/log b)
        t = log(B * (1/log_a + 1/log_b)) + 0.14
        L = max(t, 21.0)
        B_next = ceil(U + K * M * (L * L))
        if B_next <= B:
            return B
        B = B_next

    return B  # extremely conservative fallback



########################################################################
########################## Algebraic Systems  ##########################
########################################################################

def solve_linear_system(
    A: Matrix[int],
    b: Vector[int] | None = None,
    *,
    nullspace: bool = False,
) -> tuple[Vector[int] | None, list[Vector[int]] | None]:
    """
    Find integer solutions to the system of linear equations given by Ax = b.

    Parameters
    ----------
    A : Matrix[int]
        M x N matrix, with M equations of N variables
    b : Vector[int]
        Target vector of length M
    nullspace : bool
        Whether or not compute and return a basis for the null space

    Returns
    -------
    solution : Vector[int] or None
        Particular solution x such that Ax = b, or None if no solution exists
    nullspace_basis : list[Vector[int]] or None
        List of basis vectors for null space, or None if nullspace=False
    """
    num_rows = len(A) if A else 0
    num_cols = len(A[0]) if A and A[0] else 0
    if num_rows == 0 or num_cols == 0:
        return ([], None) if b is None or not any(b) else (None, None)

    # Input validation
    if any(len(row) != num_cols for row in A):
        raise ValueError("Ragged matrix")
    if b is not None and len(b) != num_rows:
        raise ValueError("Dimension mismatch")

    # The GCD of each row in A must divide the corresponding b entry
    b = [0] * num_rows if b is None else b
    for row, b_i in zip(A, b):
        g = gcd(*row)
        if (g == 0 and b_i != 0) or (g != 0 and b_i % g != 0):
            return (None, [] if nullspace else None)

    # Try Bareiss for square matrices
    if not nullspace and num_rows == num_cols:
        x = _bareiss(A, b)
        if x is not None and _verify_linear_system(A, x, b):
            return (x, None)

    # Try modular approach for square / tall matrices
    # Use Hadamard bound: n × n matrix with entries <= K has det <= n^(n/2) * K^n
    if not nullspace and num_rows >= num_cols:
        max_abs = max(abs(v) for row in A for v in row)
        max_abs = max(max_abs, max(map(abs, b)))
        dim = min(num_rows, num_cols)
        hadamard_bits = dim * ((isqrt(dim) + 1).bit_length() + max_abs.bit_length())
        x = _linear_solve_modular(A, b, max(64, hadamard_bits + 10))
        if x is not None:
            return (x, None)

    # Fallback to HNF approach
    transpose = lambda M: [list(col) for col in zip(*M)]
    HT, UT = _hermite_normal_form(transpose(A))

    # Compute nullspace (columns of U past the rank = rows of UT)
    rank = next((r for r in range(len(HT)) if not any(HT[r])), len(HT))
    nullspace_basis = [UT[r] for r in range(rank, len(UT))]
    if not any(b):
        return ([0] * num_cols, nullspace_basis)

    # Find pivots
    H, U = transpose(HT), transpose(UT)
    pivots = []
    for col in range(num_cols):
        pivot_rows = {p[0] for p in pivots}
        non_pivot_rows = (r for r in range(num_rows) if r not in pivot_rows)
        row = next((r for r in non_pivot_rows if H[r][col]), None)
        if row is None: break
        pivots.append((row, col))

    # Solve H @ y = b via back substitution
    y = [0] * num_cols
    for row, col in pivots:
        numerator = b[row] - sum(H[row][c] * y[c] for c in range(col))
        y[col], remainder = divmod(numerator, H[row][col])
        if remainder:
            return (None, nullspace_basis if nullspace else None)

    # Check consistency for non-pivot rows
    pivot_rows = {row for row, _ in pivots}
    for row in range(num_rows):
        if row not in pivot_rows:
            if sum(H[row][c] * y[c] for c in range(num_cols)) != b[row]:
                return (None, nullspace_basis if nullspace else None)

    # Recover x = U @ y
    x = [sum(U[r][c] * y[c] for c in range(num_cols)) for r in range(num_cols)]

    # Size-reduce x using nullspace basis via coordinate descent
    if nullspace_basis and max(map(int.bit_length, x)) > 1000:
        norm = lambda v: sum(vi * vi for vi in v)
        best_x, best_norm, improved = x.copy(), norm(x), True
        while improved:
            improved = False
            for i in range(len(x)):
                for v in nullspace_basis:
                    if v[i] and (scale := x[i] // v[i]):
                        x = [x_i - scale * k_i for x_i, k_i in zip(x, v)]
                        if (x_norm := norm(x)) < best_norm:
                            best_x, best_norm, improved = x.copy(), x_norm, True
                        break
        x = best_x

    return (x, nullspace_basis if nullspace else None)

def solve_polynomial_system(
    polynomials: list[Polynomial[int]],
    bounds: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """
    Find integer solutions to a system of multivariate polynomial equations
    f₁(x₁, x₂, ...) = f₂(x₁, x₂, ...) = ... = fₖ(x₁, x₂, ...) = 0,
    where |xᵢ| < bounds[i] for each variable.

    Polynomials are represented as dictionaries mapping monomial tuples to coefficients,
    e.g. {(2, 0): 3, (0, 1): -5, (0, 0): 7} represents 3x² - 5y + 7.

    Not guaranteed to find *all* solutions for large bounds.

    Parameters
    ----------
    polynomials : list[dict[tuple[int, ...], int]]
        System of multivariate polynomials with integer coefficients
    bounds : tuple[int, ...]
        Bounds on solution size, where |xᵢ| < bounds[i] for each variable

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Integer solutions within the given bounds, sorted lexicographically
    """
    # Permute variables so smallest bounds come first (better for backtracking)
    polynomials, bounds, index = _permute_variables(polynomials, bounds)
    unpermute = lambda x: tuple(x[index[i]] for i in range(len(bounds)))
    solutions = _solve_polynomial_system(polynomials, bounds)
    return tuple(sorted(map(unpermute, solutions)))

def _verify_linear_system(A: Matrix, x: Vector, b: Vector) -> bool:
    """
    Verify the matrix equation Ax = b.
    """
    return all(sum(a * xi for a, xi in zip(row, x)) == bi for row, bi in zip(A, b))

def _bareiss(A: Matrix[int], b: Vector[int]) -> list[int] | None:
    """
    Use the Bareiss algorithm to find an integer solution to Ax = b for square matrix A.

    See: https://doi.org/10.1090/S0025-5718-1968-0226829-0

    Complexity
    ----------
    O(n³) operations
    """
    n = len(A)
    M = [row.copy() + [b_i] for row, b_i in zip(A, b)]  # augmented matrix

    # Gaussian elimination
    prev_pivot = 1
    for i in range(n):
        # Find non-zero pivot in current column
        row = next((r for r in range(i, n) if M[r][i]), None)
        if row is None:
            return None
        if row != i:
            M[i], M[row] = M[row], M[i]

        # Zero out all entries below pivot in current column
        pivot = M[i][i]
        for r in range(i + 1, n):
            if (factor := M[r][i]):
                for c in range(i, n + 1):
                    M[r][c] = (M[r][c] * pivot - M[i][c] * factor) // prev_pivot

        prev_pivot = pivot

    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        numerator = M[i][-1] - sum(M[i][c] * x[c] for c in range(i + 1, n))
        x[i], remainder = divmod(numerator, M[i][i])
        if remainder:
            return None

    return x

def _linear_solve_mod_p(A: Matrix[int], b: Vector[int], p: int) -> list[int] | None:
    """
    Solve Ax ≡ b (mod p) using Gaussian elimination over Z/pZ.
    Returns solution vector x with 0 ≤ xᵢ < p, or None if no solution exists.

    Complexity
    ----------
    O(MN²) operations for an M × N matrix
    """
    num_rows, num_cols = len(A), len(A[0])
    M = [[a % p for a in row] + [b_i % p] for row, b_i in zip(A, b)]  # augmented matrix

    # Gaussian elimination
    pivot_row, pivot_cols = 0, []
    for col in range(num_cols):
        # Find non-zero pivot in current column
        row = next((r for r in range(pivot_row, num_rows) if M[r][col]), None)
        if row is None: continue
        if row != pivot_row:
            M[pivot_row], M[row] = M[row], M[pivot_row]

        # Normalize pivot row
        inv = pow(M[pivot_row][col], -1, p)
        for c in range(col, num_cols + 1):
            M[pivot_row][c] = (M[pivot_row][c] * inv) % p

        # Zero out all entries below pivot in current column
        for r in range(pivot_row + 1, num_rows):
            if (factor := M[r][col]):
                for c in range(col, num_cols + 1):
                    M[r][c] = (M[r][c] - factor * M[pivot_row][c]) % p

        pivot_cols.append(col)
        pivot_row += 1

    # Check for inconsistent zero row with nonzero right-most column
    if any(M[r][-1] and not any(M[r][:-1]) for r in range(num_rows)):
        return None

    # Back substitution
    x = [0] * num_cols
    for r in range(len(pivot_cols) - 1, -1, -1):
        col = pivot_cols[r]
        x[col] = (M[r][-1] - sum(M[r][c] * x[c] for c in range(col + 1, num_cols))) % p

    return x

def _linear_solve_modular(
    A: Matrix[int],
    b: Vector[int],
    max_bits: int,
) -> list[int] | None:
    """
    Solve Ax ≡ b (mod p) modulo multiple primes p, and combine solutions via CRT.
    Returns solution vector x, or None if no solution was found.
    """
    x_mod = [0] * len(A[0])
    p, mod = 1, 1
    while mod.bit_length() < max_bits:
        # Solve Ax = b (mod p) for random 32-bit prime
        while mod % p == 0:
            p = random_prime(32)
        if (solution := _linear_solve_mod_p(A, b, p)) is None:
            return None  # inconsistent mod p implies inconsistent over Z

        # Combine with running solution via Chinese Remainder Theorem
        inv = pow(mod, -1, p)
        x_mod = [x + ((y - x) * inv % p) * mod for x, y in zip(x_mod, solution)]
        mod *= p

        # Check for global solution to Ax = b 
        x = [value - mod if value > mod // 2 else value for value in x_mod]
        if _verify_linear_system(A, x, b):
            return x
    
    return None

def _hermite_normal_form(A: Matrix[int]) -> tuple[list[list[int]], list[list[int]]]:
    """
    Compute the (row) Hermite normal form for the given matrix.
    Returns (H, U) where H is an upper triangular matrix H in row Hermite normal form,
    and U is a unimodular transform such that H = UA.

    Complexity
    ----------
    O(MN² log(K)) operations for an M × N matrix, where K is the size of the max entry
    """
    num_rows, num_cols = len(A), len(A[0])
    H = [row.copy() for row in A]
    U = [[int(r == c) for c in range(num_rows)] for r in range(num_rows)]

    pivots = []
    for col in range(num_cols):
        # Find non-zero pivot in current column
        row = next((r for r in range(len(pivots), num_rows) if H[r][col]), None)
        if row is None: continue
        pivot_row = len(pivots)
        if pivot_row != row:
            H[pivot_row], H[row] = H[row], H[pivot_row]
            U[pivot_row], U[row] = U[row], U[pivot_row]

        # Zero out all entries below pivot in current column
        pivots.append((col, pivot_row))
        for r in range(pivot_row + 1, num_rows):
            while H[r][col]:
                if abs(H[pivot_row][col]) <= abs(H[r][col]):
                    # Reduce row via Euclidean division
                    q = H[r][col] // H[pivot_row][col]
                    H[r] = [H[r][c] - q * H[pivot_row][c] for c in range(num_cols)]
                    U[r] = [U[r][c] - q * U[pivot_row][c] for c in range(num_rows)]
                else:
                    # Swap rows
                    H[pivot_row], H[r] = H[r], H[pivot_row]
                    U[pivot_row], U[r] = U[r], U[pivot_row]

        # Ensure positive pivot
        if H[pivot_row][col] < 0:
            H[pivot_row] = [-value for value in H[pivot_row]]
            U[pivot_row] = [-value for value in U[pivot_row]]

        # Reduce entries above pivot
        pivot = H[pivot_row][col]
        for r in range(pivot_row):
            if H[r][col] and (q := H[r][col] // pivot):
                H[r] = [H[r][c] - q * H[pivot_row][c] for c in range(num_cols)]
                U[r] = [U[r][c] - q * U[pivot_row][c] for c in range(num_rows)]

    return (H, U)

def _poly_num_variables(f: Polynomial) -> int:
    """
    Return the number of variables in multivariate polynomial f.
    """
    return len(next(iter(f))) if f else 0

def _poly_degree(f: Polynomial, weights: tuple[int, ...] | None = None) -> int:
    """
    Return the degree of multivariate polynomial f.
    """
    if weights is None:
        return max((sum(m) for m in f), default=-1)
    return max((sum(w * e for w, e in zip(weights, m)) for m in f), default=-1)

def _poly_eval(f: Polynomial, x: tuple[int, ...], mod: int = None) -> int:
    """
    Evaluate multivariate polynomial f at point x, optionally reducing modulo mod.
    """
    total = sum(c * prod(pow(x_i, e) for x_i, e in zip(x, m)) for m, c in f.items())
    return total if mod is None else total % mod

def _poly_make_canonical(f: Polynomial[int]) -> Polynomial[int]:
    """
    Get canonical polynomial with integer coefficients and positive leading coefficient.
    """
    if not (f := {m: c for m, c in f.items() if c}): return {}
    if (g := gcd(*f.values())) > 1: f = {m: c // g for m, c in f.items()}
    if f[max(f)] < 0: f = {m: -c for m, c in f.items()}
    return f

def _poly_sub(f: Polynomial, g: Polynomial) -> Polynomial:
    """
    Subtract two multivariate polynomials. Returns f - g.
    """
    out = {**f, **{m: f.get(m, 0) - c for m, c in g.items()}}
    return {m: c for m, c in out.items() if c}

def _poly_mul(f: Polynomial, g: Polynomial) -> Polynomial:
    """
    Multiply two multivariate polynomials. Returns f * g.
    """
    out = defaultdict(int)
    for m_f, c_f in f.items():
        for m_g, c_g in g.items():
            out[tuple(ea + eb for ea, eb in zip(m_f, m_g))] += c_f * c_g

    return {m: c for m, c in out.items() if c}

def _poly_make_monic(f: Polynomial) -> Polynomial[Fraction]:
    """
    Divide polynomial f by its leading coefficient to make it monic.
    """
    if not (f := {m: Fraction(c) for m, c in f.items() if c}): return {}
    return f if (lead_c := f[max(f)]) == 1 else {m: c / lead_c for m, c in f.items()}

def _poly_univariate_coefficients(
    f: Polynomial[int],
    variable_index: int,
) -> list[int] | None:
    """
    Extract univariate coefficients for variable at index, or None if not univariate.
    """
    if not f or not (0 <= variable_index < _poly_num_variables(f)): return None
    coefficients = [0] * (max(monomial[variable_index] for monomial in f) + 1)
    for monomial, c in f.items():
        if any(e and i != variable_index for i, e in enumerate(monomial)):
            return None
        coefficients[monomial[variable_index]] += c

    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()

    return coefficients

def _poly_substitute(f: Polynomial, variable_index: int, value: int) -> Polynomial:
    """
    Substitute a value for one variable, reducing the polynomial dimension by one.
    """
    out = defaultdict(int)
    for monomial, c in f.items():
        c *= pow(value, monomial[variable_index])
        out[monomial[:variable_index] + monomial[variable_index+1:]] += c

    return {m: c for m, c in out.items() if c}

def _poly_apply_value(
    polynomials: list[Polynomial[int]],
    variable_index: int,
    value: int,
) -> list[Polynomial[int]] | None:
    """
    Substitute a value at the given variable into all polynomials.
    Returns None if any becomes inconsistent.
    """
    substituted = [_poly_substitute(f, variable_index, value) for f in polynomials]
    non_zero = [g for g in substituted if g]
    return None if any(_poly_num_variables(g) == 0 for g in non_zero) else non_zero

def _poly_reduce(
    f: Polynomial[Fraction],
    polynomials: list[Polynomial[Fraction]],
) -> Polynomial[Fraction]:
    """
    Reduce polynomial f modulo a collection of polynomials via multivariate division.
    """
    f, reduced = dict(f), {}
    while f:
        c_f = f[m_f := max(f)]
        for g in polynomials:
            if not g: continue
            m_g = max(g)
            if all(ea <= eb for ea, eb in zip(m_g, m_f)):
                shift = tuple(eb - ea for ea, eb in zip(m_g, m_f))
                f = _poly_sub(f, _poly_mul(g, {shift: c_f / g[m_g]}))
                break
        else:
            reduced[m_f] = reduced.get(m_f, Fraction(0)) + c_f
            del f[m_f]

    return {m: c for m, c in reduced.items() if c}

def _grobner_basis(
    polynomials: list[Polynomial[int]],
    groebner_max_basis: int = 200,
) -> list[Polynomial[int]] | None:
    """
    Use Buchberger's algorithm to transform a set of polynomials into a Gröbner basis.
    """
    # Find Gröbner basis G
    G = [_poly_make_monic(f) for f in polynomials if f]
    pairs = list(itertools.combinations(range(len(G)), 2))
    while pairs and len(G) < groebner_max_basis:
        # Cancel leading terms of f and g to compute the S-polynomial
        f, g = (G[i] for i in pairs.pop())
        f_monomial, g_monomial = max(f), max(g)
        lcm_monomial = tuple(map(max, f_monomial, g_monomial))
        shift_f = tuple(eb - ea for ea, eb in zip(f_monomial, lcm_monomial))
        shift_g = tuple(eb - ea for ea, eb in zip(g_monomial, lcm_monomial))
        term_f = _poly_mul(f, {shift_f: 1 / f[f_monomial]})
        term_g = _poly_mul(g, {shift_g: 1 / g[g_monomial]})
        S = _poly_sub(term_f, term_g)  # leading terms cancel

        # Reduce S against current basis (multivariate polynomial division)
        h = _poly_make_monic(_poly_reduce(S, G))
        if len(h) == 1 and not any(next(iter(h))):
            return None  # contradiction: 1 = 0
        if h:
            # Add new polynomial to basis and queue pairs with it
            G.append(h)
            pairs.extend((k, len(G) - 1) for k in range(len(G) - 1))

    # Interreduction to simplify the basis
    for i in range(len(G)):
        G[i] = _poly_reduce(G[i], G[:i] + G[i + 1:])

    # Convert back to integer coefficients
    return [
        _poly_make_canonical({m: int(c * denominator) for m, c in g.items()})
        for g in (_poly_make_monic(f) for f in G if f)
        for denominator in [lcm(*(c.denominator for c in g.values()))]
    ]

def _find_integer_roots_bounded_univariate(
    coefficients: list[int],
    bound: int,
) -> set[int]:
    """
    Find all integer roots r with |r| < bound for a univariate polynomial.
    """
    # Handle special cases
    f = polynomial(coefficients)
    if len(coefficients) <= 1:
        return set()  # constant polynomial
    if bound <= 35000:
        return {x for x in range(-bound + 1, bound) if f(x) == 0}  # brute force

    # Find roots modulo primes
    p, gcd_coefficients = 65521, gcd(*coefficients)
    for _ in range(32):
        p = next_prime(p)
        if gcd_coefficients % p == 0:
            continue

        # Find roots mod p^k for some k such that p^k > 2B covers interval (-B, B)
        mod = p**(k := ilog(2*bound, p) + 1)
        mod_roots = (r if r < mod // 2 else r - mod for r in hensel(coefficients, p, k))
        if (roots := set(r for r in mod_roots if abs(r) < bound and f(r) == 0)):
            return roots

    return set()

def _brute_force_polynomial_system(
    polynomials: list[Polynomial[int]],
    bounds: tuple[int, ...],
    mod: int = None,
    brute_force_limit: int = 1000000,
) -> list[tuple[int, ...]] | None:
    """
    Use exhaustive search to find all small roots
    to a system of multivariate polynomials fₖ(x) = 0 where x = (x₁, x₂, ...).
    """
    ranges = [range(-b + 1, b) for b in bounds]
    if prod(2*b - 1 for b in bounds) > brute_force_limit:
        return None
    elif not polynomials:
        return list(itertools.product(*ranges))
    else:
        is_root = lambda f, x: (_poly_eval(f, x, mod) == 0)
        points = itertools.product(*ranges)
        return sorted(x for x in points if all(is_root(f, x) for f in polynomials))

def _solve_polynomial_system(
    polynomials: list[Polynomial[int]],
    bounds: tuple[int, ...],
    max_backtrack_values: int = 200000,
) -> list[tuple[int, ...]]:
    """
    Find integer solutions to a system of multivariate polynomials
    fₖ(x) = 0 where x = (x₁, x₂, ...) and |xᵢ| < bounds[i] for each variable xᵢ.

    Solves by eliminating one variable at a time and backtracking.
    """
    polynomials = [f for f in polynomials if f]
    if len(bounds) == 0:
        return [()]

    # Try brute force if bounds are small enough
    solutions = _brute_force_polynomial_system(polynomials, bounds)
    if solutions is not None:
        return solutions

    # Identify the best univariate polynomial within a group of candidates
    def best_univariate(polynomials: list[dict]) -> tuple[int, list[int]] | None:
        candidates = [
            ((i, coefficients), (len(coefficients), bound, len(f)))
            for f in polynomials
            for i, bound in enumerate(bounds)
            if (coefficients := _poly_univariate_coefficients(f, i))
        ]
        return min(candidates, key=lambda x: x[1])[0] if candidates else None

    # Search for a univariate polynomial among the input polynomials
    # or derive one via Gröbner basis computation
    if (best := best_univariate(polynomials)) is None:
        if (G := _grobner_basis(polynomials)) is None:
            return []  # inconsistent system, no solutions
        else:
            best = best_univariate(G)

    # Determine values to try
    if best is not None:
        i, coefficients = best
        values = _find_integer_roots_bounded_univariate(coefficients, bounds[i])
    else:
        b = min(bounds[0], (max_backtrack_values + 1) // 2)
        i, values = 0, range(-b + 1, b)

    # Substitute x_i = v for our chosen variable over each of the potential values
    solutions = set()
    for value in values:
        if (next_polynomials := _poly_apply_value(polynomials, i, value)) is not None:
            for sol in _solve_polynomial_system(
                next_polynomials,
                bounds[:i] + bounds[i + 1:],  # remove i-th bound
                max_backtrack_values,
            ):
                solutions.add(sol[:i] + (value,) + sol[i:])  # reinsert for x_i

    return sorted(solutions)

def _permute_variables(
    polynomials: list[Polynomial],
    bounds: tuple[int, ...],
) -> tuple[list[Polynomial], tuple[int, ...], dict[int, int]]:
    """
    Permute variables so smallest bounds come first.
    Returns (polynomials, bounds, index_dict).
    """
    permutation = sorted(range(len(bounds)), key=lambda i: (bounds[i], i))
    polynomials = [
        {tuple(monomial[i] for i in permutation): c for monomial, c in f.items() if c}
        for f in polynomials
    ]
    bounds = tuple(bounds[i] for i in permutation)
    return polynomials, bounds, {j: i for i, j in enumerate(permutation)}



########################################################################
############################### Lattices ###############################
########################################################################

def lll_reduce(B: Matrix[int]) -> Matrix[int]:
    """
    Lenstra-Lenstra-Lovász (LLL) lattice basis reduction.

    Returns a reduced basis with shorter, more orthogonal vectors, satisfying:

        Size-reduction: |μ_{i,j}| ≤ 0.5 for all i > j
        Lovász condition: δ‖b*_k‖² ≤ ‖b*_{k+1}‖² + μ_{k+1,k}² ‖b*_k‖²

    Uses floating-point arithmetic for speed, with automatic escalation
    to exact rational arithmetic if precision issues are detected.

    See: https://www.cs.cmu.edu/~avrim/451f11/lectures/lect1129_LLL.pdf

    Parameters
    ----------
    B : Matrix[int]
        Integer matrix whose rows form a lattice basis

    Complexity
    ----------
    O(n⁵d log³B) time for n × d matrix with max entry size B, O(n² + nd) space
    """
    if not B:
        return []

    try:
        return _lll_reduce_block([list(row) for row in B], 0, len(B), 0.99, exact=False)
    except (_PrecisionError, OverflowError, ValueError):
        return _lll_reduce_block([list(row) for row in B], 0, len(B), 0.75, exact=True)

def bkz_reduce(B: Matrix[int], block_size: int = 20) -> Matrix[int]:
    """
    BKZ (Block Korkine-Zolotarev) lattice basis reduction.

    BKZ generalizes LLL by applying an SVP (Shortest Vector Problem) oracle
    to sliding blocks of consecutive basis vectors.

    Uses Schnorr-Euchner enumeration for the SVP oracle.

    See: https://www.sciencedirect.com/science/article/pii/0304397587900648

    Parameters
    ----------
    B : Matrix[int]
        Integer matrix whose rows form a lattice basis
    block_size : int
        Block size β for BKZ reduction.
        Larger values give better reduction but exponentially slower runtime.

    Complexity
    ----------
    O(2^(0.25β²)) per block, O(β^β) worst case
    """
    if not B:
        return []
    if block_size < 2:
        raise ValueError("block_size must be at least 2")

    B = lll_reduce([list(row) for row in B])
    block_size = min(block_size, len(B))
    if block_size <= 2:
        return B  # BKZ-2 is equivalent to LLL

    # BKZ repeatedly slides a window [k, k+block_size) across the basis.
    # For each window, it finds the shortest vector in the "projected lattice"
    # spanned by b*_k, ..., b*_{k+block_size-1}. If this vector is shorter than b*_k,
    # inserting it improves the basis. Repeat until no window improves.
    improved = True
    while improved:
        try:
            B, improved = _bkz_tour(
                B, block_size, pruning=True, delta=0.99, exact=False)
        except (_PrecisionError, OverflowError, ValueError):
            B, improved = _bkz_tour(
                B, block_size, pruning=False, delta=0.75, exact=True)

    return lll_reduce(B)

def closest_vector(B: Matrix[int], target: Vector[int]) -> Vector[int]:
    """
    Find the (approximate) closest vector to the target in the lattice
    with basis given by rows of matrix B.

    Uses Babai nearest-plane algorithm for approximate closest vector.

    Parameters
    ----------
    B : Matrix
        LLL-reduced lattice basis (rows)
    target : Vector
        Target vector in ambient space

    Complexity
    ----------
    O(n²d) time, O(n² + nd) space for n × d matrix (n vectors of dimension d)
    """
    if not B:
        return [0] * len(target)

    # Validate inputs
    n, dim = len(B), len(B[0])
    if len(target) != dim:
        raise ValueError("Dimension mismatch")

    # Compute Gram-Schmidt orthogonalization
    _, bstar, bstar_squared_norm = _gso(B)
    y = [float(x) for x in target]
    coefficients = [0] * n

    # Project target onto each orthogonal component and round to nearest integer
    for i in reversed(range(n)):
        if bstar_squared_norm[i] > 0:
            c = _nearest_int(_dot(y, bstar[i]) / bstar_squared_norm[i])
            y = [y_i - c * b_i for y_i, b_i in zip(y, B[i])]
            coefficients[i] = c

    # Reconstruct lattice vector from integer coefficients
    return [sum(c * b_i[j] for c, b_i in zip(coefficients, B) if c) for j in range(dim)]

def small_roots(
    polynomial: Polynomial[int],
    mod: int,
    bounds: tuple[int, ...] | None = None,
    *,
    epsilon: float = 0.05,
) -> list[tuple[int, ...]]:
    """
    Find small integer roots of a multivariate polynomial f(x₁, x₂, ...) ≡ 0 (mod M).

    Uses the Jochemsz-May multivariate generalization of Coppersmith's method.

    See: https://www.iacr.org/archive/asiacrypt2006/42840270/42840270.pdf
    See: https://cr.yp.to/bib/2001/howgrave-graham.pdf
    See: https://link.springer.com/chapter/10.1007/3-540-68339-9_14

    Parameters
    ----------
    polynomial : dict[tuple[int, ...], int]
        Multivariate polynomial with integer coefficients as {monomial: coefficient}
        where each monomial is a tuple indicating the exponents for each variable
        (e.g. {(1, 0): 5, (0, 1): 3, (0, 0): -7} represents 5x + 3y - 7)
    mod : int
        Modulus
    bounds : tuple[int, ...] or None
        Bound on root size, where |xᵢ| < bᵢ for each variable xᵢ.
        Required for multivariate polynomials. For univariate, defaults to M^(1/deg).
    epsilon : float
        Parameter controlling lattice dimension vs root bound trade-off.
        Smaller epsilon allows for larger bounds but requires larger lattice (slower).

    Complexity
    ----------
    Brute force path is O(Π(2Bᵢ - 1)) time
    Lattice path is dominated by LLL on an H × W matrix,
    about O(H⁵W log³A) time and O(H² + HW) space, where A is the max lattice
    """
    if (M := abs(mod)) == 0:
        raise ZeroDivisionError("Modulus must be nonzero")

    f = {m: r - M if r > M // 2 else r for m, c in polynomial.items() if (r := c % M)}
    num_variables = _poly_num_variables(f)

    # Input validation
    if any(len(monomial) != num_variables for monomial in f):
        raise ValueError("Inconsistent monomial tuple lengths")
    if any(not isinstance(e, int) or e < 0 for monomial in f for e in monomial):
        raise ValueError("Exponents must be nonnegative integers")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if bounds is None and num_variables > 1:
        raise ValueError("bounds required for multivariate instances")
    if bounds and len(bounds) != num_variables:
        raise ValueError("bounds length mismatch")
    if num_variables == 0 or (degree := _poly_degree(f)) <= 0:
        return []
    if bounds is None and num_variables == 1:
        bounds = (max(2, iroot(M, degree)),)

    # If bounds are small enough, brute force the original congruence
    if (roots := _brute_force_polynomial_system([f], bounds, mod=M)) is not None:
        return roots

    # Build lattice, reduce via LLL, and extract relations satisfying Howgrave-Graham.
    weights = _monomial_weights_from_bounds(bounds)
    m, shifts, basis = _choose_jochemsz_may_params(f, bounds, M, epsilon)
    lattice, scales, basis_index = _make_coppersmith_lattice(shifts, basis, bounds)
    relations = _extract_coppersmith_relations(lll_reduce(lattice), basis, scales, M**m)
    hg_relations, other_relations = relations

    # Try solving with increasing numbers of Howgrave-Graham polynomials
    selected = _select_coppersmith_polynomials(hg_relations, weights, basis_index)
    for k in range(min(2, num_variables), len(selected) + 1):
        solutions = solve_polynomial_system(selected[:k], bounds)
        solutions = {x for x in solutions if _poly_eval(f, x) % M == 0}
        if solutions:
            return sorted(solutions)

    # Fallback to check roots of individual non-Howgrave-Graham polynomials
    roots = set()
    for g in other_relations:
        solutions = solve_polynomial_system([g], bounds)
        roots.update(x for x in solutions if _poly_eval(f, x) % M == 0)

    return sorted(roots)

def _nearest_int(q: Real) -> int:
    """
    Round to nearest integer, ties away from zero.
    """
    one_half = 0.5 if isinstance(q, float) else Fraction(1, 2)
    return int(q + one_half) if q >= 0 else -int(-q + one_half)

def _dot(x: Vector, y: Vector, exact: bool = False):
    """
    Compute the dot product of two vectors.
    """
    if exact:
        return sum((x_i * y_i for x_i, y_i in zip(x, y)), Fraction(0))
    return fsum(x_i * y_i for x_i, y_i in zip(x, y))

def _gso(
    B: Matrix[int],
    start: int = 0,
    stop: int | None = None,
    mu: Matrix[Real] | None = None,
    bstar: Matrix[Real] | None = None,
    bstar_squared_norm: Vector[Real] | None = None,
    tolerance: float = 1e-12,
    max_passes: int = 10,
    exact: bool = False,
) -> tuple[Matrix[Real], Matrix[Real], Vector[Real]]:
    """
    Modified Gram-Schmidt orthogonalization with adaptive re-orthogonalization.

    Returns (mu, bstar, bstar_squared_norm) where mu[i][j] are the GSO coefficients,
    bstar[i] is the i-th orthogonalized vector, and bstar_squared_norm[i] is ‖b*ᵢ‖².

    When mu/bstar/bstar_squared_norm are provided, performs partial recomputation
    for rows [start, stop) and updates mu[i][j] for i >= stop, j in [start, stop).

    Complexity
    ----------
    O(n²d) time, O(n² + nd) space for full computation (n × d matrix)
    O((stop - start) · nd) time for partial recomputation
    """
    n, dim = len(B), len(B[0])
    if n == 0:
        return [], [], []

    stop = n if stop is None else stop
    max_passes = max_passes if mu is None else 1
    zero, one = (Fraction(0), Fraction(1)) if exact else (0.0, 1.0)
    number_type = Fraction if exact else float

    # Initialize or reuse coefficients
    if mu is None or bstar is None or bstar_squared_norm is None:
        mu = [[one if i == j else zero for j in range(n)] for i in range(n)]
        bstar = [[zero] * dim for _ in range(n)]
        bstar_squared_norm = [zero] * n

    # Modified Gram-Schmidt with adaptive re-orthogonalization
    for i in range(start, stop):
        mu[i][:i] = [zero] * i
        v = [number_type(x) for x in B[i]]
        for _ in range(max_passes):
            max_projection = zero
            for j in range(i):
                if bstar_squared_norm[j] != 0:
                    projection = _dot(v, bstar[j], exact) / bstar_squared_norm[j]
                    max_projection = max(max_projection, abs(projection))
                    mu[i][j] += projection
                    v = [v[t] - projection * bstar[j][t] for t in range(dim)]

            if max_projection < tolerance:
                break

        bstar[i], bstar_squared_norm[i] = v, _dot(v, v, exact)

    # Update coefficients mu_{i,j} for i >= stop, j in [start, stop)
    for i in range(stop, n):
        for j in range(start, stop):
            if bstar_squared_norm[j] != 0:
                mu[i][j] = _dot(B[i], bstar[j], exact) / bstar_squared_norm[j]

    return mu, bstar, bstar_squared_norm

def _lll_reduce_block(
    B: Matrix[int],
    start: int,
    stop: int,
    delta: float = 0.99,
    exact: bool = False,
) -> Matrix[int]:
    """
    LLL-reduce a contiguous block of the basis B[start:stop] in place.
    """
    n, dim = len(B), len(B[0])
    stop = min(stop, n)
    if stop - start <= 1:
        return B

    number_type = Fraction if exact else float
    delta = Fraction(delta).limit_denominator(1000) if exact else delta
    max_size_reduction_steps = n * n * 100  # Generous bound for size reduction

    # Initial Gram-Schmidt orthogonalization
    mu, bstar, bstar_squared_norm = _gso(B, exact=exact)
    if not exact and not all(isfinite(x) for x in bstar_squared_norm):
        raise _PrecisionError("Non-finite norm detected")

    # LLL reduction
    i = start + 1
    while i < stop:
        size_reduction_steps = 0
        while any(abs(mu[i][j]) > 0.5 for j in range(start, i)):
            # Check for non-converging behavior
            size_reduction_steps += 1
            if not exact and size_reduction_steps > max_size_reduction_steps:
                raise _PrecisionError("Size reduction failed to converge")

            # Size reduction to make |μ_{i,j}| <= 0.5 for all j < i
            for j in range(i - 1, start - 1, -1):
                if (c := _nearest_int(mu[i][j])) != 0:
                    B[i] = [x - c * y for x, y in zip(B[i], B[j])]  # update basis
                    mu[i][:j+1] = [mu[i][k] - c * mu[j][k] for k in range(j + 1)]

        # Recompute bstar[i] and mu[i] after size reduction
        bstar[i] = [number_type(x) for x in B[i]]
        for j in range(i):
            if bstar_squared_norm[j] != 0:
                mu[i][j] = _dot(B[i], bstar[j], exact) / bstar_squared_norm[j]
                bstar[i] = [bstar[i][t] - mu[i][j] * bstar[j][t] for t in range(dim)]

        # Recompute bstar_squared_norm[i] after size reduction
        bstar_squared_norm[i] = _dot(bstar[i], bstar[i], exact)

        # Precision failure detection (only in float mode)
        if not exact:
            if not isfinite(bstar_squared_norm[i]):
                raise _PrecisionError("Non-finite norm detected")
            if bstar_squared_norm[i] < 0:
                raise _PrecisionError("Negative squared norm detected")
            if any(abs(mu[i][j]) > 0.5 + 1e-9 for j in range(start, i)):
                raise _PrecisionError("Size reduction verification failed")

        # Check Lovász condition (any(B[i]) skips zero vectors)
        if bstar_squared_norm[i - 1] != 0 and any(B[i]):
            threshold = (delta - mu[i][i - 1]**2) * bstar_squared_norm[i - 1]
            if bstar_squared_norm[i] < threshold:
                B[i], B[i - 1] = B[i - 1], B[i]
                _gso(B, i - 1, i + 1, mu, bstar, bstar_squared_norm, exact=exact)
                i = max(i - 1, start + 1)
                continue

        i += 1

    # Verify with exact arithmetic if any float μ values are close to 0.5
    if not exact:
        indices = [(i, j) for i in range(1, stop) for j in range(start, i)]
        if any(abs(mu[i][j]) > 0.5 - 1e-9 for i, j in indices):
            mu_exact, _, _ = _gso(B, exact=True)
            if any(abs(mu_exact[i][j]) > Fraction(1, 2) for i, j in indices):
                raise _PrecisionError("Final verification failed: |μ| > 0.5")

    # Pack zero vectors to the back of the reduced block
    B[start:stop] = sorted(B[start:stop], key=any, reverse=True)

    return B

def _bkz_tour(
    B: Matrix[int],
    block_size: int,
    pruning: bool = False,
    delta: float = 0.99,
    exact: bool = False,
) -> tuple[Matrix[int], bool]:
    """
    Perform a single BKZ tour, with size reduction followed by block improvements.
    Returns the updated basis and whether any improvement was made.
    """
    n, dim = len(B), len(B[0]) if B else 0
    mu, _, bstar_squared_norm = _gso(B, exact=exact)

    # Size reduction to make |μ_{i,j}| <= 0.5 for all j < i
    changed = False
    for i in range(1, n):
        for j in range(i - 1, -1, -1):
            if q := _nearest_int(mu[i][j]):
                changed = True
                B[i] = [B[i][t] - q * B[j][t] for t in range(dim)]
                mu[i][:j+1] = [mu[i][t] - q * mu[j][t] for t in range(j + 1)]

    # Update GSO coefficients
    if changed:
        mu, _, bstar_squared_norm = _gso(B, exact=exact)

    # Slide window [k, k+block_size), find SVP in projected block, insert if shorter
    k, improved = 0, False
    while k < n - 1:
        end = min(k + block_size, n)
        if end - k <= 1 or bstar_squared_norm[k] == 0:
            k += 1
            continue

        # Find shortest vector in projected block [k, end)
        coefficients, svp_squared_norm = _enumerate_svp_block(
            mu, bstar_squared_norm, k, end, pruning)
        if not coefficients or svp_squared_norm >= bstar_squared_norm[k] * (1 - 1e-12):
            k += 1
            continue

        # We've found an improvement, so insert v = Σ c_i * b_{k+i} into basis
        improved = True
        v = [
            sum(c * B[k + i][t] for i, c in enumerate(coefficients) if c)
            for t in range(dim)
        ]
        B.insert(k, v)

        # LLL-reduce window to restore Lovasz condition
        # This creates a linear dependency, which we remove
        _lll_reduce_block(B, k, end + 1, delta=delta, exact=exact)

        # Remove the dependent vector (zero GSO norm) to restore original basis size
        mu, _, bstar_squared_norm = _gso(B, exact=exact)
        for i in range(k, min(end + 1, len(B))):
            if bstar_squared_norm[i] == 0:
                B.pop(i)
                mu, _, bstar_squared_norm = _gso(B, exact=exact)
                break
        else:
            raise _PrecisionError("BKZ: failed to find dependent vector")

        k = max(0, k - 1)  # Re-check previous blocks since basis changed

    return B, improved

def _enumerate_svp_block(
    mu: Matrix[Real],
    bstar_squared_norm: Vector[Real],
    start: int,
    end: int,
    pruning: bool = False,
    max_nodes: int | None = None,
) -> tuple[list[int] | None, Real]:
    """
    Schnorr-Euchner enumeration for SVP on a projected lattice block.

    Returns integer coefficients c_i for a linear combination v of basis vectors
    b_i to minimize the projected squared norm ‖v‖^2, where v = Σ c_i * b_i.
    """
    block_size = end - start
    if block_size <= 1 or any(value <= 0 for value in bstar_squared_norm[start:end]):
        return None, float('inf')

    if pruning and block_size > 1:
        # Pruning coefficients based on Gama-Nguyen-Regev extreme pruning heuristic
        # with 50% success probability (log(2) ≈ 0.693)
        c = 1 + log(2) / block_size
        pruning_bound = [(1 - i / block_size) ** c for i in range(block_size)]
    else:
        pruning_bound = [1.0] * block_size
    if max_nodes is None:
        max_nodes = 100000 * block_size if pruning else 200000 * block_size

    # Select the given block
    bstar_squared_norm = bstar_squared_norm[start:end]
    mu = [row[start:end] for row in mu[start:end]]

    # Depth-first search to find integer coefficients that minimize the
    # projected squared norm |v|^2, where v = Σ c_i * b_i
    # At each index i, we pick a value for the corresponding coefficient
    # Start at i = block_size - 1 and count down to i = 0
    step, targets = [0] * block_size, [0.0] * block_size
    coefficients, squared_norms = [0] * block_size, [0.0] * (block_size + 1)
    best_coefficients, best_squared_norm  = None, float(bstar_squared_norm[0])
    i, num_nodes_visited = block_size - 1, 0
    while i < block_size and num_nodes_visited <= max_nodes:
        num_nodes_visited += 1

        # Update squared norm
        delta = coefficients[i] - targets[i]
        squared_norms[i] = squared_norms[i + 1] + delta * delta * bstar_squared_norm[i]

        # If we have not exceeded the pruning bound, explore this partial solution
        if squared_norms[i] < pruning_bound[i] * best_squared_norm:
            if i > 0:
                # Compute target for next index to minimize projected norm
                i -= 1
                targets[i] = -sum(
                    mu[j][i] * coefficients[j] for j in range(i + 1, block_size))
                step[i], coefficients[i] = 0, _nearest_int(targets[i])
                continue  # move on to next index
            else:
                # All coefficients are set, compare with the best we've seen so far
                if squared_norms[0] < best_squared_norm and any(coefficients):
                    best_coefficients = coefficients.copy()
                    best_squared_norm = squared_norms[0]

        # Use Schnorr-Euchner enumeration to try integers near current target
        # Offsets are 0, +1, -1, +2, -2, ...
        while i < block_size:
            step[i] += 1
            offset = (step[i] + 1) // 2
            budget = max(0, pruning_bound[i] * best_squared_norm - squared_norms[i + 1])
            if offset * offset * bstar_squared_norm[i] <= budget:
                target = _nearest_int(targets[i])
                coefficients[i] = target + (offset if step[i] & 1 else -offset)
                break  # found coefficient candidate, evaluate it in outer loop
            else:
                i += 1  # no values left, backtrack to previous index

    return best_coefficients, best_squared_norm

def _monomial_weights_from_bounds(bounds: tuple[int, ...]) -> tuple[int, ...]:
    """
    Compute monomial weights from bounds based on bit lengths.
    """
    m = min(bits := [max(1, b.bit_length()) for b in bounds])
    return tuple(max(1, round(b / m)) for b in bits)

def _enumerate_monomials_weighted(
    n: int,
    weights: tuple[int, ...],
    max_weighted_degree: int,
) -> list[Monomial]:
    """
    Enumerate all n-variable monomials with weighted degree ≤ max_weighted_degree.
    Returns sorted by (weighted degree, total degree, lexicographic).
    """
    if n <= 0: return [()]
    if max_weighted_degree < 0: return []
    wdeg = lambda m: sum(w * e for w, e in zip(weights, m))  # weighted monomial degree
    ranges = itertools.product(*[range(max_weighted_degree // w + 1) for w in weights])
    monomials = [m for m in ranges if wdeg(m) <= max_weighted_degree]
    monomials.sort(key=lambda m: (wdeg(m), sum(m), m))
    return monomials

def _build_coppersmith_shifts(
    f: Polynomial[int],
    bounds: tuple[int, ...],
    M: int,
    m: int,
    t: int,
) -> tuple[list[Polynomial[int]], list[Monomial]]:
    """
    Build shifted polynomials a * f^k * M^(m-k) for the Coppersmith lattice.
    These all vanish at any small root r of f(r) ≡ 0 (mod M)

    Returns (shifts, basis) where basis is the sorted list of monomials.
    """
    n, weights = len(bounds), _monomial_weights_from_bounds(bounds)
    f_weighted_degree = _poly_degree(f, weights)
    k_max = min(m, t // f_weighted_degree)

    # Precompute powers f^0, f^1, ..., f^k_max
    # and scaling factors M^m, M^(m-1), ..., M^(m-k_max)
    f_powers = list(itertools.accumulate([f] * k_max, _poly_mul, initial={(0,) * n: 1}))
    M_powers = [pow(M, m - k) for k in range(k_max + 1)]

    # Generate shifted polynomials a * f^k * M^(m-k) for each valid (k, a) pair
    def generate_polynomials():
        for k, (f_power, scale) in enumerate(zip(f_powers, M_powers)):
            for a in _enumerate_monomials_weighted(n, weights, t - k*f_weighted_degree):
                yield {
                    tuple(map(sum, zip(monomial, a))): coefficient * scale
                    for monomial, coefficient in f_power.items()
                }

    # Estimate the L1 norm of each polynomial's row in the lattice matrix
    norm = lambda f: sum(
        abs(coefficient) * prod(bound**e for bound, e in zip(bounds, monomial))
        for monomial, coefficient in f.items()
    )

    # Keep shifts with smallest estimated row norms, collect monomials into sorted basis
    wdeg = lambda m: sum(w * e for w, e in zip(weights, m))  # weighted monomial degree
    shifted_polynomials = sorted(generate_polynomials(), key=norm)
    key = lambda m: (wdeg(m), sum(m), m)
    monomial_basis = sorted(set().union(*shifted_polynomials), key=key)

    return shifted_polynomials, monomial_basis

def _choose_jochemsz_may_params(
    f: Polynomial[int],
    bounds: tuple[int, ...],
    M: int,
    epsilon: float,
) -> tuple[int, list[Polynomial[int]], list[Monomial]]:
    """
    Choose m and t parameters based on epsilon.
    Returns (m, shifted_polynomials, monomial_basis).
    """
    # Set initial m parameter based on epsilon
    m0 = max(1, ceil(1 / (max(1, _poly_degree(f)) * epsilon)))

    # Scan from m0 down, collect (m, t) candidates where the resulting lattice fits
    candidates = []
    f_weighted_degree = _poly_degree(f, weights=_monomial_weights_from_bounds(bounds))
    for m in range(m0, 0, -1):
        t = m * f_weighted_degree
        polynomials, basis = _build_coppersmith_shifts(f, bounds, M, m, t)
        rows, cols = len(polynomials), len(basis)
        if rows:
            key = (rows - cols, -abs(rows - cols), rows)
            candidates.append((key, (m, polynomials, basis)))

    return max(candidates)[1] if candidates else (0, [], [])

def _make_coppersmith_lattice(
    shifted_polynomials: list[Polynomial[int]],
    monomial_basis: list[Monomial],
    bounds: tuple[int, ...],
) -> tuple[Matrix[int], list[int], Polynomial[int]]:
    """
    Construct scaled lattice matrix from shifted polynomials and monomial basis,
    where each row is a polynomial and columns are monomials.
    """
    basis_index = {monomial: i for i, monomial in enumerate(monomial_basis)}
    scales = [prod(pow(b, e) for b, e in zip(bounds, m)) for m in monomial_basis]
    lattice = [[0] * len(monomial_basis) for _ in range(len(shifted_polynomials))]
    for i, f in enumerate(shifted_polynomials):
        for monomial, coefficient in f.items():
            j = basis_index[monomial]
            lattice[i][j] = coefficient * scales[j]

    return lattice, scales, basis_index

def _extract_coppersmith_relations(
    reduced_lattice: Matrix[int],
    basis: list[Monomial],
    scales: list[int],
    howgrave_graham_bound: int,
) -> tuple[list[Polynomial[int]], list[Polynomial[int]]]:
    """
    Extract polynomials from reduced lattice.
    """
    howgrave_graham_relations, other_relations = [], []
    for row in reduced_lattice:
        f = {m: c // scale for c, m, scale in zip(row, basis, scales) if c}
        f = _poly_make_canonical(f)
        if f and not (len(f) == 1 and not any(next(iter(f)))):  # skip if f is constant
            if sum(abs(v) for v in row) < howgrave_graham_bound:
                howgrave_graham_relations.append(f)
            else:
                other_relations.append(f)

    return howgrave_graham_relations, other_relations

def _select_coppersmith_polynomials(
    polynomials: list[Polynomial[int]],
    weights: tuple[int, ...],
    basis_index: Polynomial[int],
    max_polynomials: int = 8,
) -> list[Polynomial[int]]:
    """
    Deduplicate, rank by complexity, and select linearly independent polynomials.
    """
    # Deduplicate and rank polynomials by (weighted degree, term count, max coefficient)
    degree = partial(_poly_degree, weights=weights)
    unique = list({tuple(sorted(f.items())): f for f in polynomials}.values())
    unique.sort(key=lambda f: (degree(f), len(f), max(map(abs, f.values()))))

    # In-place reduction against the given pivots, where pivots[col] = pivot_row
    def reduce_vector(v: Vector[int], pivots: dict[int, int], mod: int) -> int | None:
        # Gaussian elimination against existing pivots
        for pivot_col in sorted(pivots):
            if v[pivot_col] == 0: continue
            pivot_row = pivots[pivot_col]
            if (factor := (v[pivot_col] * pow(pivot_row[pivot_col], -1, mod)) % mod):
                for col in range(pivot_col, len(v)):
                    v[col] = (v[col] - factor * pivot_row[col]) % mod

        # Find new pivot position and normalize
        for col, a in enumerate(v):
            if a % mod:
                inv = pow(a, -1, mod)
                v[col:] = [(v[k] * inv) % mod for k in range(col, len(v))]
                return col

    # Select linearly independent polynomials via (modular) row reduction
    selected_polynomials, pivots, dim, mod = [], {}, len(basis_index), 2147483647
    for f in unique:
        if len(selected_polynomials) >= max_polynomials: break
        v = [0] * dim
        for monomial, coefficient in f.items():
            v[basis_index[monomial]] = coefficient % mod
        if (pivot_col := reduce_vector(v, pivots, mod)) is not None:
            pivots[pivot_col] = v
            selected_polynomials.append(f)

    return selected_polynomials



########################################################################
############################### Appendix ###############################
########################################################################

def integers() -> Iterator[int]:
    """
    Generate all integers (0, 1, -1, 2, -2, ...) in an infinite generator.
    """
    yield 0
    for i in itertools.count(start=1):
        yield i
        yield -i

def integer_pairs() -> Iterator[tuple[int, int]]:
    """
    Generate all integer pairs (x, y) via diagonal enumeration.
    """
    yield (0, 0)
    for r in itertools.count(start=1):
        for x in range(-r, r + 1):
            y = r - abs(x)
            yield (x, y)
            if y != 0:
                yield (x, -y)

def alternating(*iterables: Iterable) -> Iterator:
    """
    Visit input iterables in a cycle until each is exhausted.
    """
    queue = deque(map(iter, iterables))
    while queue:
        iterable = queue.popleft()
        try: yield next(iterable)
        except StopIteration: continue
        queue.append(iterable)

def below(f: Callable[[int], int], upper_bound: int, start: int = 0) -> Iterable[int]:
    """
    Yield consecutive values of n >= start as long f(n) < upper_bound.
    """
    return itertools.takewhile(lambda n: f(n) < upper_bound, itertools.count(start))

def lower_bound(
    f: Callable[[int], int],
    f_min: int,
    low: int = 0,
    high: int | None = None,
) -> int:
    """
    Given a monotonically increasing function f, find where it first reaches f_min.
    Returns the smallest integer n in [low, high] such that f(n) >= f_min.
    """
    if high is None:
        span = 1
        while f(low + span) < f_min: span *= 2
        high = low + span
    elif f(high) < f_min:
        raise ValueError("f(high) is below the f_min")

    return low + bisect.bisect_left(range(low, high + 1), f_min, key=f)

def permutation(n: int, master_key: bytes | None = None) -> Iterator[int]:
    """
    Generate a pseudorandom permutation of the integers 0, 1, ..., n - 1.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    if n == 1:
        yield 0
        return

    # Derive num_rounds * 32 bytes of round-key material
    master_key = secrets.token_bytes(32) if master_key is None else master_key
    keys = tuple(
        hmac.digest(master_key, b'feistel-round' + i.to_bytes(4, 'big'), hashlib.sha256)
        for i in range(16)
    )

    # Pre-compute mask
    m = (n - 1).bit_length()
    m += (m & 1)  # round up to even
    half = m // 2
    half_bytes = (half + 7) // 8
    mask = (1 << half) - 1

    def expand_hmac_sha256(key: bytes, msg: bytes, output_length: int) -> bytes:
        # HMAC-SHA256 in counter mode
        out, offset, counter = bytearray(output_length), 0, 0
        while offset < output_length:
            block = hmac.digest(key, msg + counter.to_bytes(4, 'big'), hashlib.sha256)
            take = min(len(block), output_length - offset)
            out[offset:offset+take] = block[:take]
            offset += take
            counter += 1

        return bytes(out)

    def feistel(x: int) -> int:
        l, r = (x >> half) & mask, x & mask
        for k in keys:
            msg = r.to_bytes(half_bytes, 'big')
            f = int.from_bytes(expand_hmac_sha256(k, msg, half_bytes), 'big') & mask
            l, r = r, (l ^ f) & mask

        return (l << half) | r

    # Cycle-walking to restrict from [0, 2^m) to [0, n)
    for x in range(n):
        y = x
        while True:
            y = feistel(y)
            if y < n:
                yield y
                break

def is_square(n: int) -> bool:
    """
    Check if an integer n is a square.
    """
    return n >= 0 and (n & 0xF) in (0, 1, 4, 9) and (sqrt_n := isqrt(n)) * sqrt_n == n

def iroot(x: int, n: int) -> int:
    """
    Find the integer n-th root of x.
    Returns the largest integer a such that a^n ≤ x.
    Uses Newton's method.
    """
    # Handle special cases
    if n == 2:
        return isqrt(x)
    if n == 1:
        return x
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if x < 0:
        if n % 2 == 0:
            raise ValueError("Cannot compute even root of negative number")
        return -iroot(-x - 1, n) - 1
    if x == 0:
        return 0

    # Initial guess
    try:
        a = ceil(x**(1/n))
        assert a**n >= x
    except (AssertionError, OverflowError):
        a = 1 << ((x.bit_length() + n - 1) // n)

    # Newton's method
    a, b = a, a + 1
    while a < b:
        b = a
        a = ((n - 1) * a + x // (a ** (n - 1))) // n

    return b

def ilog(a: int, b: int = 2) -> int:
    """
    Find the integer logarithm of a with base b.
    Returns the largest integer n such that b^n ≤ a.
    Uses repeated squaring and binary search.
    """
    if a < 1 or b < 2:
        raise ValueError("Must have a >= 1 and b >= 2")
    elif b == 2:
        return a.bit_length() - 1

    # Find upper bound
    exp, power = 1, b
    while power <= a:
        exp, power = exp * 2, power * power

    # Binary search for exact exponent
    low, high = 0, exp
    while low < high:
        mid = (low + high) // 2
        power = pow(b, mid)
        if power <= a:
            low = mid + 1
        else:
            high = mid

    return low - 1

def fibonacci(n: int, mod: int | None = None) -> int:
    """
    Return the n-th Fibonacci number.

    Parameters
    ----------
    n : int
        Index of the Fibonacci number
    mod : int
        Optional modulus
    """
    # Handle negative n
    n, sign = abs(n), (-1 if n < 0 and n % 2 == 0 else 1)

    # Compute Fibonacci number
    if n <= 70:
        # For small positive n, use Binet's formula for speed
        phi = (1 + sqrt(5)) / 2
        F = round(phi**n / sqrt(5))
    else:
        # Fast doubling for larger Fibonacci numbers
        F, F_next = 1, 1
        for bit in format(n, 'b')[1:]:
            F, F_next = F * (2*F_next - F), F*F + F_next*F_next
            if bit != '0':
                F, F_next = F_next, F + F_next
            if mod is not None:
                F, F_next = F % mod, F_next % mod

    return sign * (F if mod is None else F % mod)

def fibonacci_index(n: int) -> int:
    """
    Find the index of n in the Fibonacci sequence.
    Returns the largest integer i such that F(i) <= n.

    Parameters
    ----------
    n : int
        Upper bound on Fibonacci number

    Complexity
    ----------
    O(log² n) time for logarithmic search with Fibonacci evaluations
    """
    if n < 0:
        raise ValueError("Must have n >= 0")
    if n == 0:
        return 0
    if n == 1:
        return 2

    # Find the maximum exponent representation of n = base^exp
    base, exp = n, 1
    while (power := perfect_power(base)) != (base, 1):
        base = power[0]
        exp *= power[1]

    # Compute parameters in logspace
    phi = (1 + sqrt(5)) / 2  # golden ratio
    log_phi = log(phi)
    log_sqrt5 = 0.5 * log(5.0)
    log_target = exp * log(base)  # log(n) = log(base^exp) = exp * log(base)

    # Find Fibonacci index
    i = max(1, int((log_target + log_sqrt5) / log_phi))
    while i > 1 and fibonacci(i) > n:
        i -= 1
    while fibonacci(i + 1) <= n:
        i += 1

    return i

def polygonal(s: int, i: int) -> int:
    """
    Return the i-th s-gonal number.
    """
    return (s - 2) * i * (i - 1) // 2 + i

def polygonal_index(s: int, n: int) -> int:
    """
    Find the index of n in the s-gonal numbers.
    Returns the largest integer i such that P(s, i) ≤ n.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if n == 0:
        return 0
    if s < 2:
        raise ValueError("s < 2 not supported")
    if s == 2:
        return n

    return (isqrt(8 * n * (s - 2) + (s - 4) * (s - 4)) + s - 4) // (2 * (s - 2))

def periodic_continued_fraction(
    D: int,
    P: int = 0,
    Q: int = 1,
) -> tuple[Iterator[int], int, int]:
    """
    Compute coefficients for the periodic continued fraction
    (P + sqrt(D)) / Q = a₀ + 1 / (a₁ + 1 / (a₂ + ...)).

    Returns
    -------
    coefficients : Iterator[int]
        Coefficients of the continued fraction
    initial_length : int
        Length of the initial non-repeating block
    period_length : int
        Length of the repeating period
    """
    if is_square(D) or D <= 0:
        raise ValueError("D must be a non-square positive integer")
    if Q == 0:
        raise ZeroDivisionError("Q must be nonzero")

    # Convert to canonical form where Q | (D - P^2)
    if (D - P*P) % Q != 0:
        P, D, Q = P * abs(Q), D * Q * Q, Q * abs(Q)

    # Run the PQa algorithm
    coefficients, index, sqrt_D = [], {}, isqrt(D)
    a = (sqrt_D + P) // Q
    while (P, Q, a) not in index:
        index[P, Q, a] = len(coefficients)
        coefficients.append(a)
        P = a*Q - P
        Q = (D - P*P) // Q
        a = (sqrt_D + P) // Q

    period_length = len(coefficients) - index[P, Q, a]
    initial_length = len(coefficients) - period_length
    coefficients = itertools.chain(
        coefficients[:initial_length],
        itertools.cycle(coefficients[initial_length:])
    )

    return coefficients, initial_length, period_length

def convergents(
    coefficients: Iterable[int],
    num: int | None = None,
) -> Iterator[tuple[int, int]]:
    """
    Return convergents of the continued fraction with the given coefficients.

    Parameters
    ----------
    coefficients : Iterable[int]
        Coefficients of the continued fraction
    num : int
        Maximum number of convergents to generate (infinite by default)

    Yields
    ------
    numerator : int
        Numerator of the convergent
    denominator : int
        Denominator of the convergent
    """
    A, A_prev = 1, 0
    B, B_prev = 0, 1
    for a in itertools.islice(coefficients, num):
        A, A_prev = a * A + A_prev, A
        B, B_prev = a * B + B_prev, B
        yield A, B

def polynomial(
    coefficients: Sequence[Number],
    mod: int | None = None,
) -> Callable[[Number], Number]:
    """
    Create a univariate polynomial function with the given coefficients (a₀, ..., aₙ).
    Uses Horner's method for polynomial evaluation.
    """
    coefficients = coefficients if mod is None else [c % mod for c in coefficients]
    reversed_coefficients = coefficients[::-1]

    if mod is None:
        return lambda x: reduce(lambda b, a: a + b*x, reversed_coefficients, 0)
    else:
        return lambda x: reduce(lambda b, a: (a + b*x) % mod, reversed_coefficients, 0)

def _identity(n: int) -> int:
    """
    The identity function f(n) = n.
    """
    return n

def _threshold_select(
    value: int,
    thresholds: list[tuple[int, int]],
    default: int,
) -> int:
    """
    Select result based on threshold ranges.
    Returns result for the smallest (threshold, result) pair where value ≤ threshold.
    If value exceeds all thresholds, returns default.

    Parameters
    ----------
    value : int
        Value to check against thresholds
    thresholds : list[tuple[int, int]]
        List of (threshold, result) pairs
    default : int
        Result to return if value exceeds all thresholds
    """
    for threshold, result in sorted(thresholds, key=lambda x: x[0]):
        if value <= threshold:
            return result
    return default

_ODD_PRIMES_BELOW_256 = frozenset((
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
    61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
    131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251
))

_PRIMORIAL_ODD_PRIMES_BELOW_256 = prod(_ODD_PRIMES_BELOW_256)

# Hash-based Miller-Rabin witness table for n < 2^32
# with hash (0xAD625B89 * n) >> 24 & 255
# See: https://www.techneon.com/download/is.prime.32.base.data
_MILLER_RABIN_32_BIT_BASES = (
    1216, 1836, 8885, 4564, 10978, 5228, 15613, 13941,
    1553, 173, 3615, 3144, 10065, 9259, 233, 2362,
    1598, 551, 2285, 6146, 6804, 6275, 4054, 2057,
    7886, 8334, 5869, 2055, 1578, 2201, 3879, 2614,
    530, 2682, 886, 3118, 8865, 1014, 1676, 7091,
    2856, 4444, 2172, 2143, 2840, 1012, 3330, 696,
    5765, 6844, 4846, 7521, 1094, 7045, 4112, 3576,
    1143, 2320, 6924, 5765, 7373, 4298, 582, 2121,
    1297, 1670, 3350, 3227, 1722, 5765, 9051, 1942,
    2023, 7064, 3641, 306, 7836, 5060, 1278, 6490,
    2128, 3595, 363, 2422, 2039, 3793, 5073, 1565,
    4939, 3693, 152, 5765, 4645, 2403, 8009, 5765,
    2802, 2090, 4881, 2250, 2090, 1441, 7166, 2200,
    1818, 4989, 8609, 3735, 4631, 702, 1585, 6728,
    2809, 7949, 3558, 3552, 3729, 5765, 4302, 6406,
    7041, 4101, 3780, 5765, 9305, 2521, 1286, 5765,
    5765, 2802, 4108, 4285, 2016, 1936, 3937, 2796,
    10510, 5765, 2049, 4936, 6924, 2188, 766, 3752,
    1356, 8882, 7137, 1696, 10630, 4652, 1054, 1109,
    2419, 5765, 1175, 7586, 4404, 6612, 3525, 7668,
    4225, 1986, 1698, 9239, 7, 5765, 6294, 4695,
    2200, 5765, 2142, 3871, 6804, 5765, 4468, 1595,
    578, 4941, 6454, 2258, 5765, 1696, 3859, 5765,
    9033, 3226, 3956, 2268, 4740, 3334, 9225, 3466,
    1056, 6399, 5765, 5765, 5765, 2963, 4618, 4498,
    9238, 3186, 5765, 6398, 1782, 9431, 1829, 1065,
    3614, 9213, 3545, 4387, 1282, 6983, 1008, 1918,
    5765, 5765, 8601, 1112, 2942, 3510, 2553, 5765,
    621, 7921, 7971, 3573, 4502, 2819, 5765, 4802,
    6915, 2718, 8807, 5765, 2737, 5765, 5765, 982,
    3886, 2747, 506, 10042, 4714, 8348, 5765, 1774,
    3662, 1122, 6824, 5765, 4453, 3517, 2278, 7921,
)
