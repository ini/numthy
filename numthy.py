# Copyright (c) 2025 Ini Oguntola
# Permission is granted to use, copy, modify, and redistribute this work,
# provided acknowledgement of the original author is retained.

from __future__ import annotations

import bisect
import itertools
import secrets
import string
import sys

from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache, reduce
from heapq import heappop, heappush
from math import ceil, factorial, gcd, inf, isqrt, lcm, log, prod, sqrt
from operator import sub
from typing import Any, Callable, Hashable, Iterable, Iterator



Number = int | float | complex | Decimal | Fraction
small_cache = lru_cache(maxsize=128)
large_cache = lru_cache(maxsize=1048576)



########################################################################
################################ Primes ################################
########################################################################

def is_prime(n: int) -> bool:
    """
    Test if a given integer n is prime.

    Uses a combination of trial division, then either the Miller-Rabin primality
    test with deterministic bases, the extra-strong variant of the Baillie-PSW
    primality test (which has been computationally verified for all n < 2^64),
    or probabilistic Miller-Rabin with 64 rounds for n >= 2^64.

    See: https://miller-rabin.appspot.com/ (deterministic Miller-Rabin base sets)
    See: https://ntheory.org/pseudoprimes.html (BPSW verification up to 2^64)

    Parameters
    ----------
    n : int
        Integer to test for primality
    """
    if n < 2:
        return False

    # Trial division over first few primes
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59):
        if n % p == 0:
            return n == p

    # Use deterministic set of Miller-Rabin bases for small n
    if n < 341531:
        bases = (9345883071009581737,)
    elif n < 1050535501:
        bases = (336781006125, 9639812373923155)
    elif n < 350269456337:
        bases = (4230279247111683200, 14694767155120705706, 16641139526367750375)
    elif n < 55245642489451:
        bases = (2, 141889084524735, 1199124725622454117, 11096072698276303650)
    elif n < 18446744073709551616: # 2^64
        return _baillie_psw(n)
    else:
        bases = (secrets.randbelow(n - 3) + 2 for _ in range(64))

    return _miller_rabin(n, bases)

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

    a = (n + 1) | 1 # next odd number
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

    # Generate candidates
    while True:
        k = num_bits - 1 if safe else num_bits
        middle = secrets.randbits(k - 2) # all random bits except first/last
        p = (1 << (k - 1)) | (middle << 1) | 1 # force first/last bit to 1
        if is_prime(p):
            if safe:
                if is_prime(q := 2*p + 1):
                    return q
            else:
                return p

def primes(
    low: int = 2,
    high: int | None = None,
    num: int | None = None,
) -> Iterator[int]:
    """
    Generate primes in increasing order within the range `[low, high]`.

    Uses the sieve of Eratosthenes, with a segmented approach for large or
    unbounded ranges.

    Parameters
    ----------
    low : int
        Lower bound for prime numbers
    high : int
        Upper bound for prime numbers (default is infinite)
    num : int
        Maximum number of primes to generate (default is infinite)
    """
    DEFAULT_SIEVE_SIZE, MAX_SIEVE_SIZE = 1000, 10_000_000
    low = max(low, 2)
    high = inf if high is None else high
    num = inf if num is None else num

    # Generate initial prime
    if low <= 2 <= high and num > 0:
        yield 2
        num -= 1
    elif low > high or num <= 0:
        return

    # Set initial sieve size
    # When `high` is given, sieve on range [low, high]
    # When `num` is given, sieve on range [low, n (log n + log log n)],
    # where n is an upper bound on `π(low) + num`
    if high == num == inf:
        sieve_size = DEFAULT_SIEVE_SIZE
    else:
        n = num + 1.25506 * low / max(log(low), 1) # Rosser & Schoenfeld bound (1962)
        upper_bound = n * (log(n) + log(log(n))) # upper bound on the nth prime
        sieve_size = int(min(MAX_SIEVE_SIZE, high - low + 1, upper_bound - low))

    # Initial list of small primes to use for the segmented sieve
    small_primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
        43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    ]

    # Generate additional primes
    while low <= high and num > 0:
        # Extend list of small primes if necessary
        while (p := small_primes[-1]) < isqrt(low + sieve_size):
            small_primes.extend(_segmented_eratosthenes(p + 1, p, small_primes))

        # Get new primes with segmented sieve
        new_primes = _segmented_eratosthenes(low, sieve_size, small_primes)
        if num < inf:
            new_primes = tuple(itertools.islice(new_primes, num))
            num -= len(new_primes)

        # Yield new primes
        yield from new_primes

        # Update sieve range
        low += sieve_size
        sieve_size = min(2 * sieve_size, MAX_SIEVE_SIZE, high + 1 - low)

def count_primes(x: int) -> int:
    """
    Prime counting function π(x). Returns the number of primes p <= x.

    Uses the Lagarias-Miller-Odlyzko (LMO) extension of the Meissel-Lehmer algorithm.

    Parameters
    ----------
    x : int
        Upper bound for prime numbers
    """
    return _lmo(x, k=15, c=0.003)

def sum_primes(
    x: int,
    f: Callable[[int], Number] | None = None,
    f_prefix_sum: Callable[[int], Number] | None = None,
) -> Number:
    """
    Compute F(x) as the sum of f(p) over all primes p <= x,
    where f is a completely multiplicative function (by default, f(n) = n).

    Uses a generalized version of the LMO prime counting algorithm.
    Ideally `f()` and `f_prefix_sum()` can be calculated efficiently
    in O(1) time via closed-form expression.

    Parameters
    ----------
    x : int
        Upper bound for prime numbers
    f : Callable(int) -> int
        Completely multiplicative function f(n),
        where f(1) = 1 and f(ab) = f(a) * f(b) for all a, b > 0
    f_prefix_sum : Callable(int) -> Number
        Function to compute the cumulative sum Σ_{1 <= k <= n} f(k)
    """
    if f is None and f_prefix_sum is None:
        f, f_prefix_sum = (lambda n: n), (lambda n: n * (n + 1) // 2)
    elif f is None or f_prefix_sum is None:
        raise ValueError("Both f() and f_prefix() must be provided.")

    return _lmo(x, k=5, c=0.005, f=f, f_prefix_sum=f_prefix_sum)

def _miller_rabin(n: int, bases: Iterable[int] = (2,)) -> bool:
    """
    Miller-Rabin primality test over the given bases.

    See: https://www.sciencedirect.com/science/article/pii/0022314X80900840
    """
    # Write n - 1 as 2^s * d with d odd (by factoring out powers of 2)
    s, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        s += 1

    # Perform a Miller-Rabin test for each base
    for a in bases:
        if (a := a % n) == 0:
            continue # probable prime

        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue # probable prime

        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break # probable prime
        else:
            return False # composite

    return True

def _baillie_psw(n: int) -> bool:
    """
    Baillie-PSW primality test for n. Uses an extra-strong Lucas step.

    There are no known counterexamples to this primality test,
    and it has been computationally verified for all n < 2^64.

    See: https://math.dartmouth.edu/~carlp/PDF/paper25.pdf
    """
    # Perform a Miller-Rabin test with base a = 2
    if not _miller_rabin(n, bases=[2]):
        return False

    # Reject perfect squares
    if is_square(n):
        return False

    # Find a suitable D for the extra-strong Lucas test (D = P^2 - 4Q with Q = 1)
    P = 3
    while jacobi(D := P*P - 4, n) != -1:
        P += 1

    # Write n + 1 = 2^s * d with d odd (by factoring out powers of 2)
    s, d = 0, n + 1
    while d % 2 == 0:
        d //= 2
        s += 1

    # Division by 2 in Z/nZ
    half = lambda a: (a + n) >> 1 if a & 1 else a >> 1

    # Perform an extra-strong Lucas primality test
    U, V = 1, P
    for bit in format(d, 'b')[1:]:
        # Doubling step
        U, V = (U*V) % n, (V*V - 2) % n

        # Incrementing step
        if bit == '1':
            U, V = half((P*U + V) % n), half((D*U + P*V) % n)

    # 1st extra-strong condition: U_d = 0 (mod n) and V = ± 2 (mod n)
    if U == 0 and (V == 2 or V == n - 2):
        return True
    
    # 2nd extra-strong condition: V_{2^r * d} = 0 (mod n) for some 0 <= r < s - 1
    for _ in range(s - 2):
        if V == 0: return True
        V = (V*V - 2) % n

    return V == 0

def _segmented_eratosthenes(
    start: int,
    sieve_size: int,
    small_primes: Sequence[int],
) -> Iterable[int]:
    """
    Segmented sieve of Eratosthenes.
    Returns odd prime numbers in the range [start, start + sieve_size).
    Expects sorted small primes up to √(start + sieve_size).
    """
    # Initialize sieve segment
    # Only odd numbers are stored in the sieve (sieve[i] corresponds to start + 2i)
    start, end = start | 1, start + sieve_size
    sieve_size = (end - start + 1) >> 1
    sieve = bytearray(b'\x01') * sieve_size

    # Mark odd multiples of odd primes as composite
    for p in small_primes[1:]:
        # Find next odd multiple of p >= start
        next_odd_multiple = start + (p - start) % (2*p)
        if p*p > next_odd_multiple:
            next_odd_multiple = p*p

        # Mark multiples of p in the odd sieve
        index = (next_odd_multiple - start) >> 1
        count = (sieve_size - index + p - 1) // p
        if count > 0:
            sieve[index::p] = b'\x00' * count

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

    See: https://www.ams.org/journals/mcom/1985-44-170/
    S0025-5718-1985-0777285-5/S0025-5718-1985-0777285-5.pdf

    Also includes a generalized version that calculates the sum F(x) = Σ f(p)
    for all primes p <= x, where f is any arbitrary completely multiplicative function.

    The generalized LMO sub-expressions become:

        P2 = Σ f(p) * [F(x/p) − F(p − 1)] for y < p <= sqrt(x)
        φ_f(x, a) = φ_f(x, a - 1) - f(p_a) * φ_f(x/p_a, a - 1)
        S1 = Σ μ(n) f(n) φ_f(x/n, k) over ordinary leaves (n, k)
        S2 = Σ μ(n) f(n) φ_f(x/n, b) over special leaves (n, b)

    and the generalized Meissel-Lehmer formula becomes:
     
        F(x) = F(y) - 1 - P2 + φ_f(x, a) = F(y) - 1 - P2 + S1 + S2.

    Ideally `f()` and `f_prefix_sum()` can be calculated efficiently in O(1) time
    via closed-form expression.
    """
    if x < 2:
        return 0

    # Set hyperparameter y = cx^(1/3) log^2(x) such that x^(1/3) <= y <= x^(2/5)
    # where y is the upper bound on the small primes that are computed directly
    y = int(c * iroot(x, 3) * (log(x) ** 2))
    y = min(max(y, iroot(x, 3)), iroot(x * x, 5))
    y = max(y, 2) # we need y >= 2 to use an odd-only sieve starting at y + 1

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
        p_squared = p * p
        mu[p_squared::p_squared] = [0] * ((y - p_squared) // p_squared + 1)
        mu[p::p] = [-x for x in mu[p::p]]
        lpf[p::p] = [p] * ((y - p) // p + 1)

    # Sum the leaves in the tree created by either
    # the standard recurrence φ(x, a) = φ(x, a-1) - φ(x/p_a, a-1)
    # or the weighted recurrence φ_f(x, a) = φ_f(x, a-1) - f(p_a) * φ_f(x/p_a, a-1)
    S1 = _lmo_s1(x, k, mu, small_primes, f, f_prefix_sum) # sum over ordinary leaves
    S2 = _lmo_s2(x, k, lpf, mu, small_primes, f) # sum over special leaves

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

    This is the prefix sum Σ f(n) over all n <= x with exactly 2 prime factors,
    both greater than p_a.
    """
    sqrt_x = isqrt(x)
    sieve_limit = x // y
    sieve_start = (y + 1) | 1 # round up to odd
    sieve_size = y + (y & 1) # round up to even

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
        primes_ = itertools.compress(range(low_, high_ + 1, 2), sieve_)
        P2 += sum(F_segment[(x // p - low) >> 1] * (f(p) if f else 1) for p in primes_)

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
    Calculate the S1 portion of the LMO algorithm.

    This is the sum over "ordinary leaves" (i.e. of the form ± φ(x/n, k) with n <= y)
    in the tree created by the standard recurrence φ(x, a) = φ(x, a-1) - φ(x/p_a, a-1),
    or the weighted recurrence φ_f(x, a) = φ_f(x, a-1) - f(p_a) * φ_f(x/p_a, a-1).
    """
    if f is None:
        legendre_primes = small_primes[:k]
        phi = lambda x, a: _phi_legendre(x, a, primes=legendre_primes)
    else:
        phi = lambda x, a: sum(
            w * f_prefix_sum(x // t)
            for t, w in _phi_legendre_weighted_coefficients(small_primes[:a], f)
        )

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
    Calculate the S2 portion of the LMO algorithm.

    This is the sum over "special leaves" (i.e. of the form ± φ(x/n, b) with n > y)
    in the tree created by the standard recurrence φ(x, a) = φ(x, a-1) - φ(x/p_a, a-1),
    or the weighted recurrence φ_f(x, a) = φ_f(x, a-1) - f(p_a) * φ_f(x/p_a, a-1).
    """
    S2 = 0
    a, y = len(small_primes), len(mu) - 1
    phi = [0] * a
    sieve_limit = x // y
    sieve_size = isqrt(sieve_limit) - 1
    sieve_size = 2**(sieve_size.bit_length()) # round up to next power of 2
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
            values = [f(low + 2*i) if odd_sieve[i] else 0 for i in range(tree_size)]
            tree = _fenwick_tree_init(values)

        # Sieve the segment [low, high) with the remaining primes
        for b in range(k, a):
            p = small_primes[b]
            min_m = max(x // (p * high), y // p)
            max_m = min(x // (p * low), y)
            if p >= max_m:
                break

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

    # Iterate over primes
    for p in odd_primes:
        if max_prime and p > max_prime:
            break

        # Find next odd multiple of p >= start
        next_odd_multiple = start + (p - start) % (2*p)

        # Mark multiples of p in the odd sieve
        index = (next_odd_multiple - start) >> 1
        count = (sieve_size - index + p - 1) // p
        if count > 0:
            sieve[index::p] = b'\x00' * count

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

def _fenwick_tree_update(
    tree: list[Number],
    index: int,
    value: Number,
    tree_size: int | None = None,
):
    """
    Update the given index in the tree.
    """
    tree_size = tree_size or len(tree)
    for i in _fenwick_tree_update_path(index, tree_size):
        tree[i] += value

@small_cache
def _fenwick_tree_edges(tree_size: int) -> list[tuple[int, int]]:
    """
    Get all (index, parent_index) pairs for a Binary Indexed Tree (Fenwick Tree).
    """
    return [
        (index, index | (index + 1))
        for index in range(tree_size - 1)
        if index | (index + 1) < tree_size
    ]

@large_cache
def _fenwick_tree_query_path(index: int) -> tuple[int, ...]:
    """
    Get all indices that need to be queried for a prefix sum.
    """
    path = []
    index += 1
    while index > 0:
        path.append(index - 1)
        index &= index - 1 # clears the lowest set bit

    return tuple(path)

@large_cache
def _fenwick_tree_update_path(index: int, tree_size: int) -> tuple[int, ...]:
    """
    Get all indices that need to be updated for a value change.
    """
    path = []
    while index < tree_size:
        path.append(index)
        index |= index + 1 # sets the lowest unset bit

    return tuple(path)

@large_cache
def _phi_legendre(x: int, a: int, primes: tuple[int, ...]) -> int:
    """
    Evaluate Legendre's formula φ(x, a)
    which counts the number of positive integers less than or equal to x
    that are not divisible by any of the first a primes.
    """
    if a == 0:
        return x
    elif a < 8:
        # Use the direct formula φ(x, a) = φ(d) * (x // d) + φ(x % d, a)
        d = _primorial(a)
        return totient(d) * (x // d) + _phi_legendre_offsets(d)[x % d]
    else:
        # Use the recursive formula φ(x, a) = φ(x, a - 1) - φ(x // p, a - 1)
        p = primes[a - 1]
        return _phi_legendre(x, a - 1, primes) - _phi_legendre(x // p, a - 1, primes)

@small_cache
def _phi_legendre_offsets(d: int) -> tuple[int, ...]:
    """
    Compute values for Legendre's formula φ(0, a), φ(1, a), ..., φ(d - 1, a)
    where d is the product of the first a primes, and φ(d) is Euler's totient function.
    """
    return tuple(itertools.accumulate(_coprime_range(d)))

@small_cache
def _phi_legendre_weighted_coefficients(
    small_primes: tuple[int, ...],
    f: Callable[[int], Number],
) -> tuple[tuple[int, Number], ...]:
    """
    Precompute coefficients (t, μ(t)f(t)) for all t | d,
    where d is a primorial, equal to the product of the given set of small primes.

    Useful for calculating a weighted version of Legendre's formula
    φ_f(x, a) = φ_f(x, a - 1) - f(p_a) * φ_f(x/p_a, a - 1) = Σ_{t|d} w_t * g(x/t)
    via Mobius weighted inclusion-exclusion, where g is the prefix sum
    g(x) = Σ_{n <= x} f(n).
    """
    coefficients = [(1, 1)]
    for p, weight in zip(small_primes, map(f, small_primes)):
        coefficients.extend((t * p, -c * weight) for (t, c) in coefficients[:])

    return tuple(coefficients)

@small_cache
def _primorial(n: int) -> int:
    """
    Calculate the product of the first n primes.
    """
    return prod(primes(num=n))



########################################################################
############################ Factorization #############################
########################################################################

def prime_factors(n: int) -> tuple[int, ...]:
    """
    Get all prime factors of n in sorted order (with multiplicity).

    Uses a combination of trial division, Brent's variant of Pollard's rho
    factorization method, Lenstra's elliptic curve method (ECM),
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
    return Counter(_gen_prime_factors(n))

def divisors(n: int) -> tuple[int, ...]:
    """
    Get all divisors of n in sorted order (including both 1 and n).

    Parameters
    ----------
    n : int
        Integer to factor
    """
    factors = [1]
    for p, e in Counter(_gen_prime_factors(n)).items():
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
    and self-initializing quadratic sieve (SIQS).
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")

    # Trial division over first few primes
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
        while n % p == 0:
            yield p
            n //= p

    if n == 1:
        return

    # Use probabilistic factorization algorithms (Las Vegas strategy)
    # On failure, simply push n back onto the stack for further attempts
    stack = deque([n])
    while stack:
        n = stack.pop()
        if is_prime(n):
            yield n
        elif n > 1:
            # Use ECM or SIQS for large inputs
            num_bits = n.bit_length()
            if num_bits >= 64:
                # Try ECM initially (good at finding small/medium size factors)
                max_curves = 12 if num_bits >= 128 else None # None -> use ECM default
                d = _ecm(n, max_curves=max_curves)
                if 1 < d < n:
                    stack.append(d)
                    stack.append(n // d)
                    continue

                # Fallback to SIQS
                d = _siqs(n)
                if 1 < d < n:
                    stack.append(d)
                    stack.append(n // d)
                    continue

            # Fallback to Brent for small inputs (or if ECM/SIQS fails)
            d = _brent(n)
            stack.append(d)
            stack.append(n // d)

def _partial_factorization(
    n: int,
    small_primes: Iterable[int]
) -> tuple[dict[int, int], int]:
    """
    Factor n with respect to a set of primes.
    Returns a partial prime factorization as a dictionary {prime: exponent},
    and the remaining cofactor after dividing out all given primes.
    """
    partial_pf = {}
    for p in small_primes:
        while n % p == 0:
            n //= p
            partial_pf[p] = partial_pf.get(p, 0) + 1

    return partial_pf, n

def _brent(n: int, max_attempts: int = 8, batch_size: int = 256) -> int:
    """
    Brent's variant of Pollard's rho factorization method.
    Returns an integer factor of n.

    See: https://maths-people.anu.edu.au/~brent/pd/rpb051i.pdf
    """
    for _ in range(max_attempts):
        # Random starting point and polynomial f(x) = x^2 + c
        y = secrets.randbelow(n - 3) + 2
        c = secrets.randbelow(n - 3) + 2
        G, r = 1, 1 # batch GCD, range

        # Save checkpoint x, iterate y -> f(y) for r steps, then iterate r more steps
        # while also accumulating products q = prod (x - y) over the range.
        # When gcd(q, n) > 1, we've found a factor.
        while G == 1:
            x, q = y, 1 # checkpoint, batch product
            for _ in range(r):
                y = (y * y + c) % n

            # Batch GCD
            for k in range(0, r, batch_size):
                ys = y
                for _ in range(min(batch_size, r - k)):
                    y = (y * y + c) % n
                    q = q * (x - y) % n

                if (G := gcd(q, n)) > 1:
                    break

            # Double the range
            r *= 2

        # Backtrack if batch GCD failed (i.e. batch product is 0 mod n)
        if G == n:
            G, y = 1, ys
            while G == 1:
                y = (y * y + c) % n
                G = gcd(abs(x - y), n)

        if G < n:
            return G # success, found non-trivial factor

    return n # failure, return trivial factor

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
    """
    if n < 2:
        return 1
    if n % 2 == 0:
        return 2

    # Heuristics tuned for 64–128-bit composites.
    # In this range, each curve should be cheap and we expect SIQS to take over
    # for hard semiprimes (e.g. ~64-bit factors). So we cap stage-2 more
    # aggressively than in "big integer" ECM parameter tables.
    bits = n.bit_length()
    if B1 is None:
        B1_thresholds = [(90, 2000), (105, 4000), (128, 11000), (160, 50000)]
        B1 = _threshold_select(bits, B1_thresholds, 250000)
    if B2 is None:
        if bits <= 128: B2 = min(B1 * 50, 2_000_000)
        else: B2 = min(B1 * 100, 10_000_000)
    if max_curves is None:
        curve_thresholds = [(90, 10), (105, 12), (128, 14), (160, 40)]
        max_curves = _threshold_select(bits, curve_thresholds, 80)

    # Precompute prime powers p^e <= B1
    prime_powers = _ecm_prime_powers(B1)

    # Precompute stage 2 plan (shared across curves)
    plan = _ecm_stage_2_plan(B1, B2)

    for _ in range(max_curves):
        # Suyama's parametrization typically uses sigma >= 6.
        sigma = secrets.randbelow(n - 7) + 6 if n > 7 else 6

        A24, P, factor = _ecm_suyama_curve(n, sigma)
        if factor is not None:
            return factor
        if A24 is None or P is None:
            continue

        # Stage 1
        Q = P
        for i, prime_power in enumerate(prime_powers, start=1):
            Q = _montgomery_ladder(prime_power, Q, A24, n)
            if i % 32 == 0:
                if 1 < (g := gcd(Q[1], n)) < n:
                    return g

        if 1 < (g := gcd(Q[1], n)) < n:
            return g
        if g == n:
            # Rare degeneracy; try to salvage.
            g2 = gcd(Q[0], n)
            if 1 < g2 < n:
                return g2
            continue

        # Stage 2
        if B2 > B1:
            g = _ecm_stage_2(n, A24, Q, plan)
            if 1 < g < n:
                return g

    return 1 # failure, return trivial factor

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
    (X1, Z1), (X2, Z2) = P, Q
    Xd, Zd = diff
    A, B = X1 + Z1, X1 - Z1
    C, D = X2 + Z2, X2 - Z2
    DA, CB = D*A % mod, C*B % mod
    plus, minus = DA + CB, DA - CB
    X3 = (Zd * ((plus  * plus)  % mod)) % mod
    Z3 = (Xd * ((minus * minus) % mod)) % mod
    return X3, Z3

def _montgomery_double(
    P: tuple[int, int],
    A24: int,
    mod: int,
) -> tuple[int, int]:
    """
    Montgomery curve point doubling in projective x-only coordinates.
    Point P is represented as (X:Z) with affine x = X/Z.
    Uses the Montgomery parameter A24 = (A + 2) / 4.

    See: https://www.hyperelliptic.org/EFD/g1p/auto-montgom-xz.html
    """
    X, Z = P
    A, B = X + Z, X - Z
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
        return (1, 0) # O (point at infinity)
    if k == 1:
        return (P[0] % mod, P[1] % mod)

    R0 = (1, 0) # O (point at infinity)
    R1 = diff = (P[0] % mod, P[1] % mod)
    for bit in bin(k)[2:]: # include leading 1
        if bit == '0':
            R1 = _montgomery_add(R0, R1, diff, mod)
            R0 = _montgomery_double(R0, A24, mod)
        else:
            R0 = _montgomery_add(R0, R1, diff, mod)
            R1 = _montgomery_double(R1, A24, mod)

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
        return None, None, None # degenerate, choose another sigma

    u, v = (sigma*sigma - 5) % n, (4*sigma) % n
    if u == 0 or v == 0:
        return None, None, None # degenerate, choose another sigma

    # Calculate starting point P = (X1:Z1) = (u^3:v^3)
    X1, Z1 = u3, v3 = (u*u*u) % n, (v*v*v) % n

    # Check for non-trivial factor of n
    denominator = (16*u3*v) % n
    g = gcd(denominator, n)
    if 1 < g < n:
        return None, None, g # found non-trivial factor
    if g == n or denominator == 0:
        return None, None, None # degenerate, choose another sigma

    # Calculate A24 = (A + 2) / 4 = (v-u)^3 * (3u+v) / (16*u^3*v)
    t = (v - u) % n
    t3 = (t*t*t) % n
    numerator = (t3 * ((3*u + v) % n)) % n
    A24 = (numerator * pow(denominator, -1, n)) % n
    if A24 == 0:
        return None, None, None # degenerate, choose another sigma

    # Reject (likely) singular curves by checking gcd(A^2 - 4, n)
    A = (4*A24 - 2) % n
    discriminant = (A*A - 4) % n
    g = gcd(discriminant, n)
    if 1 < g < n:
        return None, None, g # found non-trivial factor
    if g == n or discriminant == 0:
        return None, None, None # degenerate, choose another sigma

    return A24, (X1, Z1), None

@small_cache
def _ecm_prime_powers(B1: int) -> tuple[int, ...]:
    """
    Precompute prime powers p^e <= B1 for ECM stage 1.
    Cached because ECM is often called repeatedly with the same bounds.
    """
    return tuple(p**ilog(B1, p) for p in primes(high=B1))

@small_cache
def _ecm_stage_2_plan(
    B1: int,
    B2: int,
) -> tuple[int, dict[int, tuple[int, ...]], tuple[int, ...]]:
    """
    Precompute a stage-2 plan for ECM using baby-step giant-step (BSGS) strategy.

    Represents each prime r in (B1, B2] as r = k*D ± offset, where D ≈ √B2
    is the "giant step" size, k indicates which multiple of D is closest to r,
    and offset is the distance from r to that multiple (0 <= offset <= D/2).

    Returns
    -------
    giant_step_size : int
        The interval D for giant steps
    giant_step_to_offsets : dict[int, tuple[int, ...]]
        Maps each k to tuple of offsets for primes near k*D
    baby_steps : tuple[int, ...]
        All unique offset values that need to be precomputed
    """
    if B2 <= B1:
        return 0, {}, ()

    # Choose giant step size D ≈ √B2, but ensure D/2 ≤ B1
    # This avoids k = 0 cases and huge baby-step sets
    giant_step_size = max(min(isqrt(B2), 2*B1), 6)
    giant_step_size += giant_step_size % 2 # round up to even

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
    plan: tuple[int, dict[int, tuple[int, ...]], tuple[int, ...]],
) -> int:
    """
    ECM stage 2 using Montgomery baby-step / giant-step.
    Returns a non-trivial factor of n if found, otherwise 1.
    """
    D, giant_step_to_offsets, baby_steps = plan
    if not D or not giant_step_to_offsets: return 1 # failure, return trivial factor

    # Baby steps - compute [d]Q for small offsets d
    # Primes p in (B1, B2] are written as p = k*D ± d. Precompute [d]Q values.
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
    baby.update({d: _montgomery_ladder(d, Q, A24, n) for d in baby_steps - baby.keys()})

    # Giant step base [D]Q
    PD = _montgomery_ladder(D, Q, A24, n)
    k_max = max(giant_step_to_offsets)

    # Batch GCD: accumulate Z-coordinates, and compute gcd(product, n) once instead of
    # many individual GCDs. Chunk size 512 balances batching vs early termination.
    chunk: list[int] = []

    def flush() -> int:
        g = _batch_gcd(chunk, n) if chunk else 1
        chunk.clear()
        return g if 1 < g < n else 1

    def collect(k: int, Pk: tuple[int, int]) -> int:
        """
        Accumulate checks for primes p = kD ± d. Returns factor if found.
        """
        if k not in giant_step_to_offsets: return 1
        Xk, Zk = Pk
        for d in giant_step_to_offsets[k]:
            if d == 0:
                chunk.append(Zk)
            else:
                # Cross-ratio trick: in x-only Montgomery, we can't compute [kD±d]Q
                # directly, but (Xk*Zd - Xd*Zk) vanishes iff the points combine to
                # give a Z-coordinate sharing a factor with n
                Xd, Zd = baby[d]
                chunk.append((Xk*Zd - Xd*Zk) % n)

        return flush() if len(chunk) >= 512 else 1

    # Check k = 0 with primes p = d
    if 0 in giant_step_to_offsets:
        chunk.extend([baby.get(d, (1, 0))[1] for d in giant_step_to_offsets[0] if d])
        if (g := flush()) > 1:
            return g

    # Check k = 1 with primes p = D ± d
    if (g := collect(1, PD)) > 1: return g
    if k_max == 1: return flush()

    # Check k = 2 with primes p = 2D ± d via doubling, not differential addition, as
    # diff = [D]Q - [D]Q = O (point at infinity) means Montgomery's formula is undefined
    P_prev, P_current = PD, _montgomery_double(PD, A24, n)
    if (g := collect(2, P_current)) > 1: return g

    # Handle k >= 3 with primes p = kD ± d, via differential addition
    # [kD]Q = [(k-1)D]Q + [D]Q with diff = [(k-2)D]Q (always a valid point)
    for k in range(3, k_max + 1):
        P_prev, P_current = P_current, _montgomery_add(P_current, PD, P_prev, n)
        if (g := collect(k, P_current)) > 1: return g

    return flush()

def _batch_gcd(values: list[int], mod: int) -> int:
    """
    Compute gcd(prod(values), mod) with a fallback to per-value gcd when the product
    vanishes with respect to the modulus.
    """
    if not values:
        return 1
    prod_mod = 1
    for v in values:
        prod_mod = (prod_mod * (v % mod)) % mod

    g = gcd(prod_mod, mod)
    if 1 < g < mod:
        return g
    if g == mod:
        # Degenerate: isolate a factor by checking individually.
        for v in values:
            gg = gcd(v, mod)
            if 1 < gg < mod:
                return gg
    return 1

def _siqs(
    n: int,
    B: int | None = None,
    M: int | None = None,
    large_prime_multiplier: int | None = None,
    max_polynomial_count: int | None = None,
) -> int:
    """
    Self-initializing quadratic sieve (SIQS) with large prime variants.
    Returns an integer factor of n.

    See: https://www.ams.org/notices/199612/pomerance.pdf
    See: https://math.dartmouth.edu/~carlp/implementing.pdf
    See: https://ir.cwi.nl/pub/1367/1367D.pdf
    """
    base, _ = perfect_power(n)
    if base is not None:
        return base

    bits = n.bit_length()

    # Use heuristic factor base bound B ≈ e^(1/2 sqrt(log(n) * log(log(n))))
    if B is None:
        log_bits = bits.bit_length()
        B = 1 << (isqrt(bits * log_bits) >> 1) 
        B = max(300, min(B, 50000))

    # Adaptively set sieve half-width M based on input size and factor base bound
    if M is None:
        if bits <= 100: M = max(50000, B * 30)
        elif bits <= 120: M = max(80000, B * 16)
        elif bits <= 140: M = max(100000, B * 10)
        else: M = max(120000, B * 8)
        M = min(M, 400000)

    # Adaptively set large prime multiplier based on input size
    if large_prime_multiplier is None:
        large_prime_multiplier = _threshold_select(bits, [(130, 8), (155, 16)], 20)

    # Adaptively set max number of polynomials to use based on input size
    if max_polynomial_count is None:
        poly_thresholds = [(140, 2000), (155, 6000)]
        max_polynomial_count = _threshold_select(bits, poly_thresholds, 9000) 

    # Collect relations (X, pf) where X^2 ≡ Q (mod n), Q is a B-smooth integer,
    # and pf[p] = e is the prime factorization of Q over the factor base
    factor_base = _build_factor_base(n, B)
    min_relation_count = len(factor_base) + 30
    L = B * large_prime_multiplier # large prime bound
    relations, lucky_factor = _collect_relations(
        n, factor_base, B, L, M, min_relation_count, max_polynomial_count)

    # Check early termination conditions
    if lucky_factor is not None:
        return lucky_factor # success, gcd(lp, n) was non-trivial
    if len(relations) <= len(factor_base):
        return 1 # failure, null space will be trivial

    # Build a relation matrix over GF(2), where each row is a bit-packed integer
    # and each bit j is set only when prime j has odd exponent in that relation
    fb_primes, _, _ = zip(*factor_base)
    idx = {p: i for i, p in enumerate((-1,) + fb_primes)} # prime index
    rows = [sum(2**idx[p] for p, e in pf.items() if e % 2 == 1) for _, pf in relations]

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
                return g # success, found non-trivial factor

    return 1 # failure, return trivial factor

def _build_factor_base(n: int, B: int) -> list[tuple[int, float, int]]:
    """
    Build factor base of primes p <= B where for each prime p,
    n is a quadratic residue mod p.
    """
    factor_base = [(2, log(2), 1)] if n % 2 != 0 and B >= 2 else []
    for p in primes(low=3, high=B):
        if pow(n % p, (p - 1) // 2, p) != 1: continue # skip non-residues
        if (root := _tonelli_shanks(n, p)) is not None:
            factor_base.append((p, log(p), root))

    return factor_base

def _gen_polynomials(
    n: int,
    factor_base: list[tuple[int, float, int]],
    M: int,
) -> Iterator[tuple[int, int]]:
    """
    Generate SIQS polynomials Q(x) = (Ax + B)^2 - n,
    where A ≈ √(2n)/M is the product of k primes and B satisfies B^2 ≡ n (mod A).
    """
    sqrt_n = isqrt(n)
    target_A = isqrt(2*n) // M

    # Skip tiny primes (poor A factors, inflate duplicates)
    skip = max(10, len(factor_base) // 10)
    pool = factor_base[skip:] if len(factor_base) >= skip + 3 else factor_base
    pool_primes = [p for p, _, _ in pool]

    # Choose k so target_A^(1/k) falls within prime range
    k = ilog(target_A - 1, pool_primes[-1]) + 1 # ⌈log_{p_max}(target_A)⌉
    k = max(3, min(k, 8, len(pool)))

    # Narrow pool to primes near ideal size
    ideal = iroot(target_A, k)
    center = bisect.bisect_left(pool_primes, ideal)
    half_width = min(max(200, len(pool) // 5), 1200) # heuristic window half-size
    pool = pool[max(0, center - half_width) : center + half_width] or pool

    # Set acceptance bounds [low, high] for A
    if target_A > 10000:
        band = 25 if target_A >= 10**12 else 15
        low, high = max(2, target_A // band), target_A * band
    else:
        low, high = 1, inf

    # Generate polynomials
    rng, seen = secrets.SystemRandom(), set()
    while True:
        # Set A as the product of k randomly sampled primes from the pool
        source = pool if len(pool) >= k else factor_base
        sample = sorted(rng.sample(source, k), key=lambda x: x[0])
        sample_primes, _, sample_roots = zip(*sample)
        A = prod(sample_primes)

        # Reject duplicates and out-of-bounds A values
        if A < low or A > high or A in seen: continue
        seen.add(A)
        if len(seen) > 20000:
            seen.clear()

        # For each prime p in A, we have two modular roots ±r (mod p)
        # Try all sign combinations and use CRT to get B^2 ≡ n (mod A)
        for signs in itertools.product((1, -1), repeat=k-1):
            signs = (1,) + signs
            residues = [s*r % p for s, r, p in zip(signs, sample_roots, sample_primes)]
            B = crt(residues, sample_primes)
            if B is not None:
                # Shift B closer to √n for more efficient sieving (i.e. |Q(x)| small)
                yield A, B + A * ((sqrt_n - B) // A)

def _sieve_polynomial(
    n: int,
    factor_base: list[tuple[int, float, int]],
    A: int,
    B: int,
    M: int,
) -> Iterator[tuple[int, int]]:
    """
    Sieve the polynomial Q(x) = (Ax + B)^2 - n for x in [-M, M].
    Yields (Q(x), Ax + B) pairs for candidates passing the smoothness threshold.
    """
    length, offset = 2*M + 1, -M

    # Set smoothness threshold ~ max|Q(x)|
    max_abs_Q = max(abs((-A*M + B)**2 - n), abs((A*M + B)**2 - n))
    base_log = log(max_abs_Q) if max_abs_Q > 1 else 1.0
    threshold = base_log * 0.55

    # Initialize sieve
    sieve = [base_log] * length

    # Sieve with factor base
    # Skip small primes (which will still be checked later when factoring Q(x))
    for p, log_p, root in factor_base:
        if A % p == 0 or p < 30:
            continue

        # Mark all x where Q(x) = 0 (mod p)
        inv_A = pow(A, -1, p)
        logp = itertools.repeat(log_p)
        for x in {((root - B) * inv_A) % p, ((-root - B) * inv_A) % p}:
            start = (x - offset) % p
            sieve[start::p] = map(sub, sieve[start::p], logp) # subtract log(p)

    # Yield only candidates that pass the smoothness threshold
    # (i.e. sieve value is small)
    for i, value in enumerate(sieve):
        if value <= threshold:
            y = A * (offset + i) + B
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
    lucky_factor : int | None
        A non-trivial factor of n if found during collection, otherwise None
    """
    factor_base_primes, _, _ = zip(*factor_base)
    possible_large_primes = list(primes(low=B+1, high=large_prime_bound))

    # Generate and sieve polynomials for relations
    relations, partial_relations = {}, {}
    polynomial_generator = _gen_polynomials(n, factor_base, M)
    for A, Bp in itertools.islice(polynomial_generator, max_polynomial_count):
        for Q, y in _sieve_polynomial(n, factor_base, A, Bp, M):
            # Factor Q over the factor base
            pf, residual = _partial_factorization(abs(Q), factor_base_primes)
            pf[-1] = 1 if Q < 0 else 0 # account for sign with -1 factor

            # Handle large prime variants
            large_primes = _get_large_primes(residual, possible_large_primes)
            if large_primes is None:
                continue # unusable
            elif large_primes == ():
                relations.setdefault(y % n, pf) # smooth
            else:
                # Check if gcd of large primes with n yields a non-trivial factor
                large_prime_product = prod(large_primes)
                if 1 < (g := gcd(large_prime_product, n)) < n:
                    return [], g

                # If we find two partials (X1)^2 = Q1 (mod n) and (X2)^2 = Q2 (mod n)
                # with the same large primes with product d, then we can combine them
                # and (X1 * X2 * d^(-1))^2 = (Q1/d) * (Q2/d) (mod n) is smooth
                X1 = y % n
                if large_primes in partial_relations:
                    X2, partial_pf = partial_relations.pop(large_primes)
                    for p, e in partial_pf.items():
                        pf[p] = pf.get(p, 0) + e

                    X = (X1 * X2 * pow(large_prime_product, -1, n)) % n
                    relations.setdefault(X, pf)
                else:
                    partial_relations[large_primes] = (X1, pf)

            if len(relations) >= min_relation_count:
                return list(relations.items()), None

    return list(relations.items()), None

def _get_large_primes(
    v: int,
    possible_large_primes: Sequence[int],
    max_large_prime_count: int = 2,
) -> tuple | None:
    """
    Factor residue v into large primes.
    Returns tuple of up to `max_large_prime_count` primes if v factors completely
    over `possible_large_primes`, otherwise returns None.
    """
    L = possible_large_primes[-1]
    if v == 1:
        return ()
    if v <= L:
        return (v,) if is_prime(v) else None
    if max_large_prime_count <= 1:
        return None

    # Try to extract a large prime factor and recurse
    for p in possible_large_primes:
        if p * p > v: break
        if v % p == 0:
            rest = _get_large_primes(
                v // p, possible_large_primes, max_large_prime_count - 1)
            if rest is not None:
                return tuple(sorted((p,) + rest))

    return None

def _nullspace_gf2(rows: list[int]) -> list[int]:
    """
    Find null space of the matrix over GF(2) using Gaussian elimination.
    Rows are bit-packed integers.
    """
    pivots = {}
    nullspace = []

    for i, row in enumerate(rows):
        combo = 1 << i
        r = row
        while r:
            # Pick the least significant bit as a pivot
            pivot_col = (r & -r).bit_length() - 1
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
####################### Multiplicative Functions #######################
########################################################################

def totient(n: int) -> int:
    """
    Compute Euler's totient function φ(n) for a positive integer n.

    Parameters
    ----------
    n: int
        Positive integer function argument
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")

    return _totient_from_pf(prime_factorization(n))

def totient_range(N: int) -> list[int]:
    """
    Find the value of Euler's totient function φ(n) for each n = 0, 1, 2, ..., N - 1.
    Includes dummy value φ(0) = 1.

    Parameters
    ----------
    N : int
        Upper bound on range (exclusive)
    """
    phi = [1] * N
    prime_factor_array = _prime_factor_range(N)
    for n in range(2, N):
        if (p := prime_factor_array[n]) == n:
            phi[n] = n - 1 # n is prime
        else:
            m = n // p
            if m % p == 0:
                phi[n] = phi[m] * p # φ(p^k) = p * φ(p^(k-1))
            else:
                phi[n] = phi[m] * (p - 1) # φ(p * m) = (p - 1) * φ(m) if p ∤ m

    return phi

def mobius(n: int) -> int:
    """
    Compute the Mobius function μ(n) for a positive integer n.

    Parameters
    ----------
    n: int
        Positive integer function argument
    """
    pf = prime_factorization(n)
    if any(v > 1 for v in pf.values()):
        return 0

    return 1 if len(pf) % 2 == 0 else -1

def mobius_range(N: int) -> list[int]:
    """
    Find the value of the Mobius function μ(n) for each n = 0, 1, 2, ..., N - 1.
    Includes dummy value μ(0) = 1.

    Parameters
    ----------
    N : int
        Upper bound on range (exclusive)
    """
    mu = [1] * N
    prime_factor_array = _prime_factor_range(N)
    for n in range(2, N):
        p = prime_factor_array[n]
        m = n // p
        if prime_factor_array[m] == p:
            mu[n] = 0
        else:
            mu[n] = -mu[m]

    return mu

def radical(n: int) -> int:
    """
    Compute rad(n) as the product of the distinct prime factors of n.

    Parameters
    ----------
    n: int
        Positive integer function argument
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")
    return prod(set(_gen_prime_factors(n)))

def radical_range(N: int) -> list[int]:
    """
    Find the value of the radical function rad(n) for each n = 0, 1, 2, ..., N - 1,
    where rad(n) is the product of the distinct prime factors of n.
    Includes dummy value rad(0) = 1.

    Parameters
    ----------
    N : int
        Upper bound on range (exclusive)
    """
    rad = [1] * N
    prime_factor_array = _prime_factor_range(N)
    for n in range(2, N):
        p = prime_factor_array[n]
        m = n // p
        if prime_factor_array[m] == p:
            rad[n] = rad[m]
        else:
            rad[n] = rad[m] * p

    return rad

def divisor_function(n: int, k: int = 1) -> int:
    """
    Compute the value of the divisor function σ_k(n),
    where σ_k(n) = ∑_{d|n} d^k.

    Parameters
    ----------
    n: int
        Positive integer function argument
    k: int
        Divisor exponent
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")
    if k < 0:
        raise ValueError("k must be a non-negative integer.")

    pf = prime_factorization(n)
    if k == 0:
        return prod(e + 1 for e in pf.values())
    else:
        return prod((pow(p, k * (e + 1)) - 1) // (pow(p, k) - 1) for p, e in pf.items())

def divisor_count_range(N: int) -> list[int]:
    """
    Find the number of divisors d(n) for each n = 0, 1, 2, ..., N - 1.
    Includes dummy value d(0) = 1.

    Parameters
    ----------
    N : int
        Upper bound on range (exclusive)
    """
    # Use the multiplicative property of the divisor count function
    # exp[n] = exponent of prime_factor_array[n] in n
    divisor_counts = [1] * N
    exp = [0] * N
    prime_factor_array = list(_prime_factor_range(N))
    for n in range(2, N):
        p = prime_factor_array[n]
        m = n // p
        if prime_factor_array[m] == p:
            e = exp[m] + 1
            exp[n], divisor_counts[n] = e, (divisor_counts[m] // e) * (e + 1)
        else:
            exp[n], divisor_counts[n] = 1, divisor_counts[m] * 2

    return divisor_counts

def divisor_function_range(N: int, k: int = 1) -> list[int]:
    """
    Find the values of the divisor function σ_k(n) for each n = 0, 1, 2, ..., N - 1,
    where σ_k(n) = ∑_{d|n} d^k. Includes dummy value σ_k(0) = 0.

    Parameters
    ----------
    N : int
        Upper bound on range (exclusive)
    """
    if k == 0:
        return divisor_count_range(N)
    elif k < 0:
        raise ValueError("k must be a non-negative integer.")

    # Use the multiplicative property of the divisor sum function
    # power[n] = p^(ke) for the largest prime power p^e | n,
    # where p = prime_factor_array[n]
    # sum_of_powers[n] = 1 + p^k + p^(2k) + ... + p^(ke)
    divisor_sums = [1] * N
    power, sum_of_powers = [0] * N, [0] * N
    prime_factor_array = _prime_factor_range(N)
    for n in range(2, N):
        p = prime_factor_array[n]
        m = n // p
        if prime_factor_array[m] == p:
            power[n] = power[m] * p**k
            sum_of_powers[n] = sum_of_powers[m] + power[n]
            divisor_sums[n] = divisor_sums[m] // sum_of_powers[m] * sum_of_powers[n]
        else:
            power[n] = p**k
            sum_of_powers[n] = 1 + power[n]
            divisor_sums[n] = divisor_sums[m] * sum_of_powers[n]

    divisor_sums[0] = 0
    return divisor_sums

def aliquot_sum_range(N: int) -> list[int]:
    """
    Find the value of the aliquot sum s(n) for each n = 0, 1, 2, ..., N - 1,
    where s(n) = σ(n) - n is the sum of proper divisors of n.
    Includes dummy value s(0) = 0.

    Parameters
    ----------
    N : int
        Upper bound on range (exclusive)
    """
    divisor_sums = divisor_function_range(N)
    return [d - i for i, d in enumerate(divisor_sums)]

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

    L = pow(a % p, (p - 1) // 2, p)
    return -1 if L == p - 1 else L

def jacobi(a: int, n: int) -> int:
    """
    Compute the Jacobi symbol (a | n), where n is an odd positive integer.

    Parameters
    ----------
    a: int
        Numerator (i.e. quadratic residue class)
    n: int
        Denominator (i.e. modulus)
    """
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be an odd positive integer.")

    J = 1
    while (a := a % n) != 0:
        # Extract factors of 2 from a
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                J = -J

        # Apply quadratic reciprocity
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
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
    if a % 2 == 0 and exp > 0:
        return 0

    # Compute (a | 2)^t
    K = 1 if a % 8 in (1, 7) else -1 # (a | 2)
    K = K if exp % 2 == 1 else 1 # (a | 2)^exp

    return sign * K * jacobi(a % n, n)

def _totient_from_pf(pf: dict[int, int]):
    """
    Compute Euler's totient function φ(n) given the prime factorization of n.
    """
    return prod(p**(e - 1) * (p - 1) for p, e in pf.items())

def _prime_factor_range(N: int) -> list[int]:
    """
    Find a prime factor for each n = 0, 1, 2, ..., N - 1.
    """
    prime_factor_array = list(range(N))
    if N >= 1:
        for p in primes(high=isqrt(N-1)):
            prime_factor_array[p::p] = [p] * ((N - 1 - p) // p + 1)

    return prime_factor_array

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

def crt(residues: Iterable[int], moduli: Iterable[int]) -> int | None:
    """
    Solve a system of linear congruences x ≡ a_i (mod n_i)
    via the Chinese Remainder Theorem.

    Parameters
    ----------
    residues : Iterable[int]
        Sequence of residues
    moduli : Iterable[int]
        Sequence of moduli

    Returns
    -------
    x : int | None
        Solution to the system of congruences, mod the LCM of the moduli,
        or None if no solution exists
    """
    try:
        return reduce(_crt_two_congruences, zip(residues, moduli), (0, 1))[0]
    except ValueError:
        return None

def hensel(
    coefficients: Sequence[int],
    p: int,
    k: int,
    initial: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """
    Find all solutions to the polynomial congruence f(x) ≡ 0 (mod p^k).

    Assumes f(x) = a_0 + a_1x + a_2x^2 + a_3x^3 ... is a polynomial.
    Uses Hensel lifting to find solutions.

    Parameters
    ----------
    coefficients : Sequence[int]
        Polynomial coefficients, where coefficients[i] is the coefficient for x^i
    p : int
        Prime base of modulus
    k : int
        Exponent of modulus
    initial : list[int]
        Initial solutions to f(x) ≡ 0 (mod p)
    """
    if not is_prime(p):
        raise ValueError("p must be prime")

    # Define polynomials f(x) and f'(x)
    f = polynomial(coefficients)
    df = polynomial([i * coefficients[i] for i in range(1, len(coefficients))])

    # Find initial solutions to f(x) = 0 (mod p)
    if initial is None:
        solutions = {x for x in range(p) if f(x) % p == 0}
    else:
        solutions = {x % p for x in initial if f(x) % p == 0}

    # Exit early if no solutions or if the exponent is k = 1
    if k <= 0 or not solutions:
        return ()
    if k == 1:
        return tuple(solutions)

    # Hensel lifting to find solutions to f(x) = 0 (mod p^k)
    mod = p
    for _ in range(k - 1):
        new_solutions, new_mod = set(), mod * p
        for root in solutions:
            base = root % mod
            f_val = f(base) % new_mod
            if f_val % mod != 0:
                continue

            f_coeff, df_mod = (f_val // mod) % p, df(base) % p
            if df_mod != 0:
                # Simple root, unique lift
                t = (-f_coeff * pow(df_mod, -1, p)) % p
                new_solutions.add((base + t * mod) % new_mod)
            elif f_coeff == 0:
                # Multiple root, p (potential) lifts
                new_solutions.update((base + t * mod) % new_mod for t in range(p))

        solutions, mod = new_solutions, new_mod
        if not solutions:
            break

    return tuple(root % mod for root in solutions)

def carmichael(n: int) -> int:
    """
    Compute Carmichael's lambda function λ(n) for a positive integer n.
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")

    terms = []
    for p, e in prime_factorization(n).items():
        if p == 2:
            terms.append(e if e < 3 else 2**(e - 2))
        else:
            terms.append((p - 1) * (p**(e - 1)))

    return lcm(*terms)

def multiplicative_order(a: int, mod: int) -> int:
    """
    Return the smallest integer k = ord_n(a) such that a^k ≡ 1 (mod n).

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

    Use the Itoh-Bach algorithm to search for candidates.

    Parameters
    ----------
    n : int
        Integer modulus
    """
    if n == 0:
        raise ZeroDivisionError("Modulus n must be nonzero")
    if n < 0:
        n = -n
    if n in (1, 2, 4):
        return n - 1

    # Check if a primitive root exists
    pf = prime_factorization(n)
    if not ((len(pf) == 1 and n % 2 == 1) or (len(pf) == 2 and pf.get(2, 0) == 1)):
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
    if n % 2 == 0:
        return g if g % 2 == 1 else g + n // 2
    else:
        return g

def _crt_two_congruences(
    congruence_1: tuple[int, int],
    congruence_2: tuple[int, int],
) -> tuple[int, int]:
    """
    Solve a system of two linear congruences x ≡ a1 (mod n1) and x ≡ a2 (mod n2)
    via the Chinese remainder theorem.
    """
    a1, n1 = congruence_1
    a2, n2 = congruence_2
    d = gcd(n1, n2)
    diff = a2 - a1
    if diff % d != 0:
        raise ValueError("No solution exists for the given system of congruences.")

    # Reduce to coprime moduli and compute modular inverse
    n1_, n2_ = n1 // d, n2 // d
    k = diff // d
    inv = pow(n1_, -1, n2_)

    # Compute solution
    x = a1 + n1 * ((k * inv) % n2_)
    mod = n1 * n2_ # n1 * n2 // d
    return x % mod, mod

def _bach(
    p: int,
    eps: float = 0.01,
    B: int | None = None,
    B_max: int = 5_000_000,
    max_tries: int = 64,
) -> int:
    """
    Use the Itoh/Bach algorithm to search for a primitive root modulo p,
    where p is prime.

    See: https://www.jstor.org/stable/2153696
    """
    if p == 2:
        return 1
    if p == 3:
        return 2
    if not is_prime(p):
        raise ValueError("p must be prime")
    if not 0.0 < eps < 1.0:
        raise ValueError("must have 0 < eps < 1")

    # Use heuristic B ~ log((p-1)/2) / eps
    # This can be huge for tiny eps, so we cap with B_max.
    B = int(log((p - 1) // 2) / eps) if B is None else B
    B = max(3, min(B, B_max, p - 1))

    # Factor φ(p) = p - 1
    pf = prime_factorization(p - 1)

    # Split into a partial factorization with primes q < B
    # and residual Q with primes q >= B
    partial_pf = {q: e for q, e in pf.items() if q < B}
    Q = prod(q**e for q, e in pf.items() if q >= B)

    # Search for primitive root
    k = (p - 1) // Q
    for _ in range(max_tries):
        # Build element of order (p-1)/Q
        a = 1
        for q, e in partial_pf.items():
            # Choose b such that b^((p-1)/q) != 1
            b = secrets.randbelow(p - 3) + 2
            while pow(b, (p - 1) // q, p) == 1:
                b = secrets.randbelow(p - 3) + 2

            a = (a * pow(b, (p - 1) // pow(q, e), p)) % p

        # Find candidate solution
        if Q == 1:
            g = a
        else:
            # Choose b with b^k != 1
            b = secrets.randbelow(p - 3) + 2 # random in [2, p-2]
            while pow(b, k, p) == 1:
                b = secrets.randbelow(p - 3) + 2 # random in [2, p-2]

            # Lift by multiplying a * b^k
            g = (a * pow(b, k, p)) % p

        # Verify solution
        if all(pow(g, (p - 1) // q, p) != 1 for q in pf):
            return g

    raise RuntimeError(
        "Failed to find a verified primitive root; try increasing B or max_tries.")



########################################################################
######################## Exponential Congruences #######################
########################################################################

def discrete_log(a: int, b: int, mod: int) -> int | None:
    """
    Find the smallest non-negative integer x such that a^x ≡ b (mod m).

    Uses the Pohlig-Hellman algorithm, with either baby-step giant-step or
    Pollard's rho for discrete logarithms on the prime-order sub-problems.

    Parameters
    ----------
    a : int
        Base of logarithm
    b : int
        Target integer
    mod : int
        Modulus
    """
    mod = mod if mod > 0 else -mod
    a, b = a % mod, b % mod

    # Handle edge case
    if mod == 1:
        return 0

    # Extended reduction to get gcd(a, m) = 1
    # Solving a^x = b * normalization^(-1) (mod m) gives us a^(x + offset) = b (mod m)
    offset, normalization = 0, 1
    while (g := gcd(a, mod)) != 1:
        if b == normalization:
            return offset
        if b % g != 0:
            return None # no solution exists

        b, mod, offset = b // g, mod // g, offset + 1
        if mod == 1:
            return offset

        a, normalization = a % mod, (normalization * (a // g)) % mod

    # Check early termination conditions
    b = (b * pow(normalization, -1, mod)) % mod # normalize
    if b == 0:
        return None # no solution exists for a^x = 0 mod m
    if b == 1:
        return offset

    # Solve a^x ≡ b (mod p^e) for each prime power
    congruences = []
    for p, e in prime_factorization(mod).items():
        try:
            x_i, ord_i = _discrete_log_mod_prime_power(a, b, p, e)
            congruences.append((x_i, ord_i))
        except ValueError:
            return None # no solution exists

    # Combine solutions via Chinese Remainder Theorem
    residues, moduli = zip(*congruences)
    x = crt(residues, moduli)
    return None if x is None else x + offset 

def modular_roots(n: int, k: int, mod: int) -> tuple[int, ...]:
    """
    Find all solutions x to x^k ≡ n (mod m).

    Uses the Tonelli-Shanks / Adleman-Manders-Miller to find roots modulo primes,
    Hensel lifting to roots modulo prime powers, and the Chinese Remainder Theorem
    to combine solutions.

    Parameters
    ----------
    n : int
        Target integer
    k : int
        Order of root
    mod : int
        Modulus
    """
    m = abs(mod)

    # Coefficients to the polynomial f(x) = x^k - n
    coefficients = [-n] + [0]*(k - 1) + [1]

    # Find roots modulo prime powers
    residue_sets = []
    moduli = []
    for p, e in prime_factorization(m).items():
        roots_mod_prime = _modular_roots_mod_prime(n, k, p)
        roots_mod_prime_power = hensel(coefficients, p, e, initial=roots_mod_prime)
        if not roots_mod_prime_power:
            return ()
        residue_sets.append(tuple(roots_mod_prime_power))
        moduli.append(p**e)

    # Combine solutions via Chinese Remainder Theorem
    return tuple(
        crt(residues, moduli)
        for residues in itertools.product(*residue_sets)
    )

def _discrete_log_mod_prime_power(a: int, b: int, p: int, e: int) -> tuple[int, int]:
    """
    Solve a^x ≡ b (mod q) in the unit group (Z/qZ)×, where q = p^e.
    Returns both the discrete log x and ord_q(a).
    """
    q = p**e
    a, b = a % q, b % q

    # Solve a^x = b in (Z/qZ)× = C_2 × C_{2^{e-2}} = <-1> × <5>
    if p == 2 and e >= 3:
        # Represent a, b each as (-1)^s * 5^t
        s_a = 0 if a % 4 == 1 else 1
        s_b = 0 if b % 4 == 1 else 1
        t_a = _pohlig_hellman_prime_power(5, -a % q if s_a == 1 else a, q, 2, e - 2)
        t_b = _pohlig_hellman_prime_power(5, -b % q if s_b == 1 else b, q, 2, e - 2)

        # Check 5^(t_b) in <5^(t_a)> to determine if a solution exists
        ord_5 = 2**(e - 2) # size of <5> in (Z/qZ)×
        g = gcd(t_a, ord_5) # index of <5^{t_a}> inside <5>
        if t_b % g != 0:
            raise ValueError("No solution exists") # 5^(t_b) not in <5^(t_a)>

        ord_a = ord_5 // g # size of the subgroup <5^{t_a}>

        # Handle the degenerate a = ± 1 case (trivial 5-part)
        if ord_a == 1:
            if s_a == 0 and s_b != 0:
                raise ValueError("No solution exists")
            return (0, 1) if s_a == 0 else (s_b, 2)

        # Solve (t_a/g) * x ≡ (t_b/g) (mod ord_a) in <5>
        inv = pow((t_a // g) % ord_a, -1, ord_a)
        x = ((t_b // g) * inv) % ord_a

        # Enforce sign parity constraint (-1)^{s_a*x} = (-1)^{s_b}
        if (s_a == 0 and s_b != 0) or (s_a == 1 and (x % 2) != s_b):
            raise ValueError("No solution exists")

        return x, ord_a

    # Solve a^x = b in the cyclic subgroup <a> ≤ (Z/qZ)×
    ord_a = _multiplicative_order_mod_odd_prime_power(a, p, e)
    if pow(b, ord_a, q) != 1:
        raise ValueError("No solution exists") # b is not in <a>

    return _pohlig_hellman(a, b, q, ord_a), ord_a

def _pohlig_hellman(g: int, h: int, mod: int, order: int) -> int:
    """
    Pohlig-Hellman algorithm for discrete logarithms.
    Solves g^x = h in the cyclic subgroup <g> of size `order`.
    """
    g, h = g % mod, h % mod

    # Validate that g and h lie in the claimed subgroup
    if pow(g, order, mod) != 1 or pow(h, order, mod) != 1:
        raise ValueError("No solution exists")

    # Handle special case of trivial subgroup
    if order == 1:
        if h == 1 % mod:
            return 0
        else:
            raise ValueError("No solution exists")

    # Solve g^x = h in each Sylow subgroup of <g> with order p^e
    congruences = []
    for p, e in prime_factorization(order).items():
        q = p**e
        g_i = pow(g, order // q, mod) # ord(g_i) = q in a cyclic group
        h_i = pow(h, order // q, mod)
        x_i = _pohlig_hellman_prime_power(g_i, h_i, mod, p, e)
        congruences.append((x_i, q))

    # Combine solutions via Chinese Remainder Theorem
    residues, moduli = zip(*congruences)
    return crt(residues, moduli) % order

def _pohlig_hellman_prime_power(g: int, h: int, mod: int, p: int, e: int) -> int:
    """
    Pohlig-Hellman algorithm for discrete logarithms.
    Solves the discrete logarithm g^x = h in cyclic subgroup <g> of order p^e.
    """
    q = p**e
    g, h = g % mod, h % mod
    discrete_log_solver = _pollard_rho_log if p.bit_length() > 48 else _bsgs
    if e == 1:
        return discrete_log_solver(g, h, mod, p)

    # Find an element with order p
    gamma = pow(g, p**(e - 1), mod)

    # Iteratively compute the p-adic digits of the logarithm
    x, prime_power, current_target, exponent = 0, 1, h, q
    for i in range(e):
        exponent //= p # p^(e - 1 - i)
        projected_target = pow(current_target, exponent, mod)
        digit = discrete_log_solver(gamma, projected_target, mod, p)
        x += digit * prime_power
        current_target *= pow(g, (-digit * prime_power) % q, mod) # h * g^(-x)
        current_target %= mod
        prime_power *= p

    return x % q

def _bsgs(g: int, h: int, mod: int, p: int) -> int:
    """
    Baby-step giant-step algorithm for discrete logarithms.
    Solves the discrete logarithm g^x = h in cyclic group <g> of prime order p.
    """
    g, h = g % mod, h % mod
    if p == 2:
        if h == 1 % mod: return 0
        if h == g % mod: return 1
        raise ValueError("No solution in order-2 subgroup")

    table, m, g_m_inv = _bsgs_table(g % mod, mod, p)
    y = h % mod
    for i in range(m):
        j = table.get(y)
        if j is not None:
            return (i*m + j) % p
        else:
            y = (y * g_m_inv) % mod

    raise ValueError("No solution found (BSGS)")

@small_cache
def _bsgs_table(g: int, mod: int, p: int) -> tuple[dict[int, int], int, int]:
    """
    Computes g^0, g^1, ..., g^m where m = ⌈√p⌉ and stores {g^j: j}.
    Also returns g^(-m) for giant-step phase.
    """
    m = isqrt(p - 1) + 1
    powers = itertools.accumulate([g] * m, lambda a, b: (a * b) % mod, initial=1)
    table = {power: exponent for exponent, power in enumerate(powers)}
    g_m_inv = pow(pow(g, m, mod), -1, mod)
    return table, m, g_m_inv

def _pollard_rho_log(g: int, h: int, mod: int, p: int, partition_size: int = 32) -> int:
    """
    Pollard Rho algorithm for discrete logarithms.
    Finds x such that g^x ≡ h (mod m), where p is the order of g.
    Uses Brent's algorithm for finding cycles.
    """
    g, h = g % mod, h % mod

    # Validate that g and h lie in the claimed subgroup
    if pow(g, p, mod) != 1 or pow(h, p, mod) != 1:
        raise ValueError("No solution exists")

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
            i = (x ^ (x >> 32)) & mask
            x = x * m_table[i] % mod
            a += a_table[i]
            b += b_table[i]
            cycle_length += 1

            # Periodically reduce exponents mod p to prevent overflow
            if j & reduce_mask == reduce_mask:
                a, b, a_t, b_t = a % p, b % p, a_t % p, b_t % p

            # Collision detected, solve g^a·h^b ≡ g^a_t·h^b_t for discrete log
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

def _modular_roots_mod_prime(n: int, k: int, p: int) -> tuple[int, ...]:
    """
    Find all solutions x to x^k ≡ n (mod p).

    Uses the Tonelli-Shanks algorithm when k = 2,
    or the Adleman-Manders-Miller (AMM) algorithm otherwise.
    """
    n %= p
    if not is_prime(p):
        raise ValueError("p must be prime")
    elif k <= 0:
        raise ValueError("k must be a positive integer.")
    elif k == 1 or n == 0 or p == 2:
        return (n,)
    elif k == 2:
        r = _tonelli_shanks(n, p)
        return () if r is None else (r, -r % p)

    # Use the generalized Euler criterion to test for the existence of a k-th root
    g = gcd(k, p - 1)
    if pow(n, (p - 1) // g, p) != 1:
        return ()

    # If gcd(k, p-1) = 1, unique root via exponent inversion
    if g == 1:
        e = pow(k, -1, p - 1)
        return (pow(n, e, p),)

    # Reduce to a g-th root
    # k = g*k1, p-1 = g*m, gcd(k1, m)=1.
    k1, m = k // g, (p - 1) // g
    inv_k1 = pow(k1, -1, m)

    # y^k1 = n (because n^(N/g)=1 and inv_k1*k1 = 1 (mod m))
    y = pow(n, inv_k1, p)

    # Solve x^g = y by extracting prime roots along the factorization of g
    pf = prime_factorization(g)
    x = y
    for r, exp in pf.items():
        for _ in range(exp):
            if r == 2:
                x = _tonelli_shanks(x, p)
            else:
                x = _adleman_manders_miller(x, r, p)

    # Find the root of unity ζ^k=1
    e = (p - 1) // g
    omega = next(
        w for a in range(2, p)
        if (w := pow(a, e, p)) != 1
        and all(pow(w, g // q, p) != 1 for q in pf)
    )

    # Now enumerate all k-th roots:
    # solutions = x * ζ where ζ^k=1, and that subgroup has size g
    roots = []
    w = 1
    for _ in range(g):
        roots.append((x * w) % p)
        w = (w * omega) % p

    return tuple(roots)

def _tonelli_shanks(n: int, p: int) -> int | None:
    """
    Tonelli-Shanks algorithm for finding modular square roots.
    Returns a root r such that r^2 ≡ n (mod p), or None if no root exists.

    See: https://www.cmat.edu.uy/~tornaria/pub/Tornaria-2002.pdf
    """
    n %= p
    if n == 0:
        return 0
    elif p == 2:
        return n
    elif p % 4 == 3:
        r = pow(n, (p + 1) // 4, p)
        return r if (r*r) % p == n else None

    # Write p - 1 as 2^s * q with q odd (by factoring out powers of 2)
    s, q = 0, p - 1
    while q % 2 == 0:
        q //= 2
        s += 1

    # Find a quadratic non-residue
    z = next(a for a in range(2, p) if pow(a, (p - 1) // 2, p) == p - 1)

    # Iterative computation to calculate square root
    # Maintain invariant R^2 ≡ n * t (mod p) until t = 1
    M, c, t, R = s, pow(z, q, p), pow(n, q, p), pow(n, (q+1)//2, p)
    while t != 1:
        i, power = 1, (t*t) % p
        while power != 1:
            power = (power*power) % p
            i += 1

        if i >= M:
            return None

        b = pow(c, 2**(M-i-1), p) # root of unity of order 2^(i+1)
        M = i # ord(t) = 2^M
        c = (b*b) % p # root of unity of order 2^i
        t = (t*c) % p # reduce order of t
        R = (R*b) % p # update root candidate, maintains R^2 ≡ n * t (mod p)

    return R

def _adleman_manders_miller(delta: int, r: int, p: int) -> int | None:
    """
    Adleman-Manders-Miller r-th root extraction in finite field F_p when r | (p - 1).
    Returns a single root x with x^r = delta (mod p), or None if no root exists.

    See: https://arxiv.org/pdf/1111.4877
    See: https://www.cs.cmu.edu/~glmiller/Publications/AMM77.pdf
    """
    delta %= p
    if delta == 0:
        return 0
    if r == 1:
        return delta
    if (p - 1) % r != 0:
        raise ValueError("Precondition: r must divide p - 1")

    # Use the generalized Euler criterion to test for the existence of an r-th root
    if pow(delta, (p - 1) // r, p) != 1:
        return None

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
    a = pow(rho, r**(t - 1) * s, p) # generator of r-th roots of unity (order r)
    b = pow(delta, r*alpha - 1, p) # satisfies b^(r^(t-1)) = 1
    c = pow(rho, s, p) # root of unity of order dividing r^t
    h = 1 # accumulates correction factor

    # Iterative computation to calculate an r-th root
    # Maintain invariants b^(r^(t-1)) = 1 (mod p)
    # and (δ^α * h)^r = δ * b^(r^(t-i)) (mod p)
    for i in range(1, t):
        d = pow(b, r**(t - 1 - i), p)
        j = 0 if d == 1 else (-discrete_log(a, d, p) % r)
        h = (h * pow(c, j, p)) % p
        c = pow(c, r, p)
        b = (b * pow(c, j, p)) % p

    return (pow(delta, alpha, p) * h) % p

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
    """
    d, x0, y0 = egcd(a, b)

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
    Generate positive integer solutions to the equation x^2 + dy^2 = m
    where 0 < d < m and gcd(d, m) = 1.

    Parameters
    ----------
    d : int
        Coefficient of y^2 term
    m : int
        Constant term

    Yields
    ------
    x : int
        X-coordinate of solution
    y : int
        Y-coordinate of solution
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
        for r in modular_roots(-d, 2, n):
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
    Generate positive integer solutions to the equation x^2 - Dy^2 = N.

    Yields positive integer solutions x, y > 0 in order of increasing x.
    When D is not a perfect square, this is the generalized Pell equation.
    Uses the Lagrange-Matthews-Mollin (LMM) algorithm.

    See: https://cjhb.site/Files.php/Books/math/B3.4/pell.pdf

    Parameters
    ----------
    D : int
        Coefficient of y^2 term
    N : int
        Constant term

    Yields
    ------
    x : int
        X-coordinate of solution
    y : int
        Y-coordinate of solution
    """
    if D <= 0:
        raise ValueError("D must be a positive integer.")

    # Handle special case where D is a perfect square
    sqrt_D = isqrt(D)
    if sqrt_D * sqrt_D == D:
        # Infinitely many solutions if N = 0
        if N == 0:
            yield from map(lambda y: (sqrt_D * y, y), itertools.count(start=1))

        # There are only finitely many solutions if N != 0
        else:
            d = 2 * sqrt_D
            factors = sorted(divisors(abs(N)))
            for i in range(len(factors) // 2):
                a, b = factors[i], factors[-i - 1]
                if N > 0 and (b - a) % d == 0:
                    yield (a + b) // 2, (b - a) // d
                elif N < 0 and (a + b) % d == 0:
                    yield (b - a) // 2, (a + b) // d

        return

    # Exit early if N = 0 has only the trivial solution
    if N == 0: return

    # Get convergents for continued fraction of sqrt(D)
    coefficients, initial, period = periodic_continued_fraction(D)
    pell_convergents = list(convergents(coefficients, num=initial+2*period))

    # Find minimal solution to x^2 - Dy^2 = -1
    solutions = ((x, y) for x, y in pell_convergents if x*x - D*y*y == -1)
    t, u = next(solutions, (None, None))

    # Find fundamental solutions to x^2 - Dy^2 = N
    fundamental_solutions = []
    for f in divisors(abs(N)):
        if (N // f) % f != 0:
            continue

        m = N // (f * f)
        for z in range(int((-abs(m) + 1) / 2), abs(m) // 2 + 1):
            if (z*z - D) % abs(m) == 0:
                a, initial, period = periodic_continued_fraction(D, P=z, Q=abs(m))
                a = [next(a) for _ in range(initial + period)]
                i = next((i for i in range(1, len(a)) if abs(a[i]) > sqrt_D), None)
                if i is not None:
                    A, B = zip(*list(convergents(a[:-1])))
                    x, y = f*(abs(m)*A[i-1] - z*B[i-1]), f*B[i-1]
                    if x*x - D*y*y == N:
                        fundamental_solutions.append((x, y))
                    elif (t, u) != (None, None):
                        fundamental_solutions.append(((x*t + y*u*D), (x*u + y*t)))

    # Find minimal solution to x^2 - Dy^2 = 1
    t0, u0 = next((x, y) for x, y in pell_convergents if x*x - D*y*y == 1)

    # Find minimal positive solutions to x^2 - Dy^2 = N
    minimal_positive_solutions = []
    for x, y in fundamental_solutions:
        if x > 0 and y > 0:
            minimal_positive_solutions.append((x, y))
        elif x < 0 and y < 0:
            minimal_positive_solutions.append((-x, -y))
        else:
            minimal_positive_solutions.append((-x*t0 + -y*u0*D, -x*u0 + -y*t0))

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

def pythagorean_triples(
    max_c: float | None = None,
    max_sum: float | None = None,
) -> Iterator[tuple[int, int, int]]:
    """
    Generate positive integer solutions to the equation a^2 + b^2 = c^2.

    Uses Euclid's formula to generate unique Pythagorean triples (a, b, c)
    where a <= b <= c.

    If no bounds are specified, infinitely generates triples in order of increasing c.
    When bounds are specified, no order is guaranteed.

    Parameters
    ----------
    max_c : float
        Upper bound for c in generated triples, where c <= max_c
    max_sum : float
        Upper bound for the sum of generated triples, where a + b + c <= max_sum
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
    queue = [] # (current_c, k, a0, b0, c0)
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

def periodic_continued_fraction(
    D: int,
    P: int = 0,
    Q: int = 1,
) -> tuple[Iterator[int], int, int]:
    """
    Compute coefficients for the periodic continued fraction
    (P + sqrt(D)) / Q = a0 + 1 / (a1 + 1 / (a2 + ...)).

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
        raise ValueError("D must be a non-square positive integer.")

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

def _euclid(max_m: int | None = None) -> Iterator[tuple[int, int, int]]:
    """
    Generate unique primitive Pythagorean triples (a, b, c) with Euclid's formula,
    where a <= b <= c.
    """
    for m in (itertools.count(start=2) if max_m is None else range(2, max_m + 1)):
        for n in itertools.compress(range(m), _coprime_range(m)):
            if (m + n) % 2 == 1:
                m_squared, n_squared = m*m, n*n
                a, b, c = m_squared - n_squared, 2*m*n, m_squared + n_squared
                if a > b:
                    a, b = b, a
                yield (a, b, c)

def _berggren() -> Iterator[tuple[int, int, int]]:
    """
    Generate primitive Pythagorean triples (a, b, c) with Berggren's tree method,
    where a <= b <= c, and triples are generated in order of increasing c.    
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



########################################################################
############################ Combinatorics #############################
########################################################################

def pascal(num_rows: int | None = None) -> Iterator[tuple[tuple[int, int], int]]:
    """
    Generate values in Pascal's triangle, row by row, left to right.

    Parameters
    ----------
    num_rows : int
        Number of rows to generate (infinite by default)

    Yields
    ------
    (n, k) : tuple[int, int]
        Index of binomial coefficient
    a : int
        Binomial coefficient a = (n choose k)
    """
    row = [1]
    for n in itertools.count() if num_rows is None else range(num_rows):
        yield from (((n, k), v) for k, v in enumerate(row)) # current row
        row = [1, *map(int.__add__, row, row[1:]), 1] # next row

def factorial_valuation(n: int, p: int) -> int:
    """
    Compute the p-adic valuation of n!, i.e., the exponent of p in n!.

    Uses Legendre's formula: v_p(n!) = (n - S_p(n)) / (p - 1),
    where S_p(n) is the digit sum of n in base p.

    Parameters
    ----------
    n : int
        Non-negative integer
    p : int
        Prime number
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if p < 2:
        raise ValueError("p must be at least 2")
    if n == 0:
        return 0

    digit_sum = sum(digits_in_base(n, p))
    return (n - digit_sum) // (p - 1)

def binomial_valuation(m: int, n: int, p: int) -> int:
    """
    Compute the p-adic valuation of binomial coefficient C(m, n).

    Uses Kummer's theorem: v_p(C(m, n)) equals the number of carries
    when adding n and (m - n) in base p.

    Parameters
    ----------
    m : int
        Non-negative integer
    n : int
        Non-negative integer with n <= m
    p : int
        Prime number
    """
    if not (0 <= n <= m):
        raise ValueError("Must have 0 <= n <= m")
    if not is_prime(p):
        raise ValueError("p must be prime")
    if n == 0 or n == m:
        return 0

    a_digits, b_digits = digits_in_base(n, p), digits_in_base(m - n, p)
    total_carries, current_carry = 0, 0
    for a, b in itertools.zip_longest(a_digits, b_digits, fillvalue=0):
        total = a + b + current_carry
        current_carry = total // p
        total_carries += current_carry

    return total_carries

def partition_numbers(mod: int | None = None) -> Iterator[int]:
    """
    Generate the values of the partition function.
    """
    yield 1
    n, k = 0, 1
    partitions, euler_pentagonal = [1], deque()

    while True:
        n += 1

        # Extend generalized pentagonal numbers to cover offsets up to n
        while not euler_pentagonal or euler_pentagonal[-1][1] <= n:
            sign = 1 if k % 2 == 1 else -1
            euler_pentagonal.append((sign, k * (3 * k - 1) // 2))
            euler_pentagonal.append((sign, k * (3 * k + 1) // 2))
            k += 1

        # Euler's recurrence: p(n) = Σ sign * p(n - offset)
        p = 0
        for sign, off in euler_pentagonal:
            if off > n: break
            p += sign * partitions[n - off]

        p = p % mod if mod else p
        yield p
        partitions.append(p)

def count_partitions(n: int, restrict: Callable[[int], bool] | None = None) -> int:
    """
    Return the number of partitions of integer n.

    Parameters
    ----------
    n : int
        Integer to partition
    restrict : Callable(int) -> bool
        Function indicating integers that can be used in the partition
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if restrict:
        return euler_transform(restrict)(n)    
    else:
        return next(p for i, p in enumerate(partition_numbers()) if i == n)

@small_cache
def euler_transform(a: Callable[[int], int]) -> Callable[[int], int]:
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



########################################################################
############################ Linear Algebra ############################
########################################################################

Matrix = list[list[Number]]
Vector = list[Number]

def linear_solve(A: Matrix, b: Vector) -> Vector:
    """
    Solve the system of linear equations given by Ax = b.

    Parameters
    ----------
    A : Matrix
        M × N matrix of coefficients
    b : Vector
        List of M values
    """
    if len(A) != len(b): raise ValueError("Matrix dimensions do not match")

    # Get reduced row-echelon form of augmented matrix [A | b]
    augmented_matrix = [[*coefs, value] for coefs, value in zip(A, b)]
    rref = _gauss_jordan(augmented_matrix)

    # Validate solution
    pivot_value_by_col, pivot_cols = {}, set()
    is_zero = lambda x: abs(x) < 1e-12 if isinstance(x, float) else x == 0
    for row in rref:
        lead = None
        for j in range(len(row)):
            if not is_zero(row[j]):
                lead = j
                break

        if lead is None:
            if not is_zero(row[-1]):
                raise ValueError("No solution")
        else:
            pivot_cols.add(lead)
            pivot_value_by_col[lead] = row[-1]

    if len(pivot_cols) < len(A[0]):
        raise ValueError("Infinite solutions")

    return [row[-1] for row in rref]

def identity_matrix(n: int) -> Matrix:
    """
    Return the n × n identity matrix.
    """
    return [[int(i == j) for j in range(n)] for i in range(n)]

def matrix_apply(function: Callable[[Number], Number], A: Matrix) -> Matrix:
    """
    Apply a function elementwise to a matrix A.
    """
    return [[function(x) for x in row] for row in A]

def matrix_transpose(A: Matrix) -> Matrix:
    """
    Return the transpose of matrix A.
    """
    return [list(col) for col in zip(*A)]

def matrix_sum(A: Matrix, B: Matrix) -> Matrix:
    """
    Return A + B.
    """
    return _matrix_binary_op(A, B, op=lambda a, b: a + b)

def matrix_difference(A: Matrix, B: Matrix) -> Matrix:
    """
    Return A - B.
    """
    return _matrix_binary_op(A, B, op=lambda a, b: a - b)

def matrix_product(A: Matrix, B: Matrix) -> Matrix:
    """
    Return the product of two matrices A and B.
    """
    if len(A[0]) != len(B): raise ValueError("Matrix dimensions do not match")
    return [[sum(a*b for a, b in zip(row, col)) for col in zip(*B)] for row in A]

def _matrix_binary_op(
    A: Matrix,
    B: Matrix,
    op: Callable[[Number, Number], Number],
) -> Matrix:
    """
    Apply a binary operation elementwise to two matrices A and B.
    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrix dimensions do not match")

    return [[op(a, b) for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]

def _gauss_jordan(A: Matrix) -> Matrix:
    """
    Gauss-Jordan elimination. Returns the given matrix in reduced row-echelon form.
    """
    num_rows, num_cols = len(A), len(A[0])
    row = col = 0
    while row < num_rows and col < num_cols - 1:
        # Find pivot in current column
        pivot_row = max(range(row, num_rows), key=lambda r: abs(A[r][col]))
        if A[pivot_row][col] == 0:
            col += 1
            continue

        # Move pivot row into position and normalize it
        A[row], A[pivot_row] = A[pivot_row], A[row]
        pivot = A[row][col]
        A[row] = [value / pivot for value in A[row]]

        # Eliminate the current column from all other rows
        for r in range(num_rows):
            if r == row: continue
            if (k := A[r][col]) == 0: continue
            A[r] = [value - k * pivot_value for value, pivot_value in zip(A[r], A[row])]

        row += 1
        col += 1

    return A



########################################################################
################################ Graphs ################################
########################################################################

Node = Hashable

def search(
    start: Iterable[Node],
    find_next: Callable[[Node], Iterable[Node]],
    found: Callable[[Node], bool] | None = None,
    stop: Callable[[Node], bool] | None = None,
) -> Iterator[Node]:
    """
    Depth first search.

    Parameters
    ----------
    start : Iterable[Node]
        Initial nodes to start searching from
    find_next : Callable(Node) -> Iterable[Node]
        Generate all candidates for the next node in the search
    found : Callable(Node) -> bool
        Whether or not a given node is a solution
    stop : Callable(Node) -> bool
        Whether or not to stop searching from a given node
    """
    stack = deque(start)
    get_node, visit_nodes = stack.pop, stack.extend
    while stack:
        node = get_node()
        if found is None or found(node):
            yield node
        if stop is None or not stop(node):
            visit_nodes(find_next(node))

def dijkstra(
    source: Node,
    neighbors_fn: Callable[[Node], Iterable[Node]],
    edge_weight_fn: Callable[[Node, Node], float],
) -> tuple[dict[Node, float], dict[Node, Node]]:
    """
    Use Dijkstra's algorithm to find the shortest path from a source node
    to all other nodes in a graph.

    Parameters
    ----------
    source : Node
        Source node
    neighbors_fn : Callable(Node) -> Iterable[Node]
        Function that returns all neighbors of a given node.
    edge_weight_fn : Callable(Node, Node) -> float
        Function that returns the weight of the edge between two nodes.

    Returns
    -------
    dist : dict[Node, float]
        Shortest distance from the source node to each node
    prev : dict[Node, Node]
        Predecessor of each node in the shortest path
    """
    dist = defaultdict(lambda: inf, {source: 0})
    prev = defaultdict(lambda: None)
    queue = [(0, source)]
    while queue:
        dist_u, u = heappop(queue)
        if dist_u > dist[u]:
            continue
        for v in neighbors_fn(u):
            alt = dist[u] + edge_weight_fn(u, v)
            if alt < dist[v]:
                dist[v], prev[v] = alt, u
                heappush(queue, (alt, v))

    return dist, prev

def find_cycles(
    find_next: Callable[[tuple[Node, ...]], Iterable[Node]] = lambda path: [],
    current_path: list[Node] | None = None,
) -> Iterator[list[Node]]:
    """
    Find cycles in a directed graph.

    Parameters
    ----------
    find_next : Callable(tuple[Node, ...]) -> Iterable[Node]
        Function that returns all candidates for the next node in the path
    current_path : list[Node]
        Current path in the graph
    """
    start_path = tuple(current_path or ())
    find_next_path = lambda path: (path + (v,) for v in find_next(path))
    found = lambda path: path and path[-1] in path[:-1]
    for cycle in search([start_path], find_next_path, found=found, stop=found):
        yield cycle[cycle.index(cycle[-1]):-1]

def find_functional_cycles(
    f: Callable[[int], int],
    search: Iterable[int],
    domain: range,
    on_cycle: Callable[[int, int], None],
) -> None:
    """
    Find cycles in the functional graph defined by f(n).

    Parameters
    ----------
    f : Callable(int) -> int
        Function defining the graph
    search : Iterable[int]
        Starting points to search for cycles
    domain : range
        Range of valid nodes in the graph
    on_cycle : Callable(cycle_start: int, cycle_node: int)
        Callback function for when a cycle is found,
        called for each node in the cycle
    """
    if domain.step != 1:
        raise ValueError("domain must have step size 1")

    low = domain.start
    cycle_id = [None] * len(domain)
    for start in search:
        # Advance until we find a cycle
        x, i = start, start - low
        while x in domain and cycle_id[i] is None:
            cycle_id[i], x = start, f(x)
            i = x - low

        # If this is a new cycle, walk through it
        if x in domain and cycle_id[i] == start:
            y = x
            on_cycle(x, y)
            while (y := f(y)) != x:
                on_cycle(x, y)

def topological_sort(graph: dict[Node, Iterable[Node]]) -> list[Node]:
    """
    Perform a topological sort on a directed acyclic graph (DAG).
    Uses depth-first search.

    Parameters
    ----------
    graph : dict[Node, Iterable[Node]]
        Graph represented as an adjacency list
    """
    visited, current_path, order = set(), set(), []
    nodes = set(graph.keys()).union(*(set(neighbors) for neighbors in graph.values()))
    for start in nodes:
        if start in visited:
            continue

        # Maintain a stack of (node, state) tuples
        # where state takes on values: 0 = enter, 1 = exit
        stack = [(start, 0)]
        while stack:
            v, state = stack.pop()
            if state == 0:
                # Skip visited nodes and detect cycles
                if v in visited:
                    continue
                if v in current_path:
                    raise ValueError("Detected cycle in graph.")

                # Schedule exit and push neighbors
                current_path.add(v)
                stack.append((v, 1))
                for u in graph.get(v, ()):
                    if u not in visited:
                        stack.append((u, 0))
            else:
                # Add node to topological ordering
                current_path.remove(v)
                visited.add(v)
                order.append(v)

    order.reverse()
    return order

def bron_kerbosch(
    graph: dict[Node, set[Node]],
    R: set[Node] | None = None,
    P: set[Node] | None = None,
    X: set[Node] | None = None,
) -> list[set[Node]]:
    """
    Recursive implementation of the Bron-Kerbosch algorithm
    for finding maximal cliques.

    Parameters
    ----------
    graph : dict[Node, set[Node]]
        Graph represented as an adjacency list
    R : set[Node]
        Current clique
    P : set[Node]
        Nodes that can be added to clique
    X : set[Node]
        Nodes to be excluded from clique

    Returns
    -------
    maximal_cliques : list[set[Node]]
        List of maximal cliques in the graph
    """
    R = set() if R is None else R
    P = set(graph.keys()) if P is None else P
    X = set() if X is None else X

    maximal_cliques = []
    if not P and not X:
        maximal_cliques.append(R)

    # Choose pivot node u to maximize |P ∩ N(u)|
    u = max(P | X, key=lambda v: len(graph[v]), default=None)
    candidates = P - (graph[u] if u is not None else set())

    # Explore candidates
    for v in candidates:
        maximal_cliques += bron_kerbosch(graph, R | {v}, P & graph[v], X & graph[v])
        P = P - {v}
        X = X | {v}

    return maximal_cliques

def kruskal(
    nodes: Iterable[Node],
    edges: Iterable[tuple[Node, Node]],
    get_edge_weight: Callable[[Node, Node], float],
) -> list[tuple[Node, Node]]:
    """
    Use Kruskal's algorithm to find a minimum spanning tree.

    Parameters
    ----------
    nodes : Iterable[Node]
        Nodes in the graph
    edges : Iterable[tuple[Node, Node]]
        Edges in the graph
    get_edge_weight : Callable(Node, Node) -> float
        Function that returns the weight of the edge between two nodes

    Returns
    -------
    minimum_spanning_tree : list[tuple[Node, Node]]
        Edges in the minimum spanning tree
    """
    parent, rank = {v: v for v in nodes}, defaultdict(int)

    # Path compression
    def find(x: Node) -> Node:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    # Union by rank
    def union(x: Node, y: Node) -> bool:
        root_x, root_y = find(x), find(y)
        if root_x == root_y:
            return False
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1
        return True

    minimum_spanning_tree = []
    for u, v in sorted(edges, key=lambda edge: get_edge_weight(*edge)):
        if union(u, v):
            minimum_spanning_tree.append((u, v))

    return minimum_spanning_tree



########################################################################
############################## Utilities ###############################
########################################################################

def nth(iterable: Iterable, n: int, default: Any = None) -> Any:
    """
    Return the n-th item from an iterable (1-based index).
    If the iterable has fewer than n items, return default.
    """
    return next(itertools.islice(iterable, n - 1, None), default)

def group_by_key(
    iterable: Iterable,
    key: Callable[[Any], Hashable],
) -> dict[Hashable, list]:
    """
    Group items in an iterable by a given key function.
    Returns a dictionary mapping each key to a list of items with that same key.

    Parameters
    ----------
    iterable : Iterable
        Items to group
    key : Callable(item) -> key
        Function that returns the key for each item
    """
    groups = defaultdict(list)
    for item in iterable:
        groups[key(item)].append(item)

    return groups

def group_permutations(iterable: Iterable[Sequence]) -> Iterable[list[Sequence]]:
    """
    Group permutations together.

    Returns a collection of lists of permutations, where for any given list,
    all its items are permutations of each other.

    Parameters
    ----------
    iterable : Iterable[Sequence]
        Sequences to group
    """
    key = lambda sequence: tuple(sorted(sequence))
    return iter(group_by_key(iterable, key=key).values())

def powerset(iterable: Iterable) -> Iterable[tuple]:
    """
    Generate all subsets of the given iterable as tuples, in order of increasing size.
    """
    iterable = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(iterable, n)
        for n in range(len(iterable) + 1)
    )

def disjoint_subset_pairs(
    iterable: Iterable,
    include_empty: bool = False,
    equal_size_only: bool = False,
) -> Iterable[tuple[tuple, tuple]]:
    """
    Generate all (unordered) pairs of disjoint subsets from the given iterable.

    Parameters
    ----------
    iterable : Iterable
        Items to form subsets from
    include_empty : bool
        Whether to include the empty set as a valid subset
    equal_size_only : bool
        Whether to only include pairs of subsets with the same size
    """
    items = list(iterable)
    idx = range(len(items))
    n = len(items)
    yield from (
        (tuple(items[i] for i in A_idx), tuple(items[i] for i in B_idx))
        for i in range(0 if include_empty else 1, n // 2 + 1)
        for j in range(i, i + 1 if equal_size_only else n + 1)
        for A_idx in itertools.combinations(idx, i)
        for A_idx_set in (set(A_idx),)
        for B_idx in itertools.combinations((x for x in idx if x not in A_idx_set), j)
        if i != j or i == 0 or A_idx[0] <= B_idx[0]
    )

def polynomial(coefficients: Sequence[Number]) -> Callable[[Number], Number]:
    """
    Create a polynomial function with the given coefficients (a_0, ..., a_n).
    Uses Horner's method for polynomial evaluation.
    """
    def horner(x: Number) -> Number:
        b = 0
        for a in reversed(coefficients): b = a + b*x
        return b

    return horner

def iroot(x: int, n: int = 2) -> int:
    """
    Find the integer n-th root of x.
    Returns the largest integer a such that a^n <= x.
    Uses Newton's method.
    """
    # Handle special cases
    if x < 0:
        if n % 2 == 0:
            raise ValueError("Cannot compute even root of negative number")
        return -iroot(-x - 1, n) - 1
    if x == 0:
        return 0
    if n <= 0:
        raise ValueError("n must be a positive integer") 
    if n == 1:
        return x
    if n == 2:
        return isqrt(x)

    # Set initial guess to 2^ceil(log2(x)/n)
    a = 1 << ((x.bit_length() + n - 1) // n)

    # Run Newton's method on f(a) = a^n - x = 0
    a, b = a, a + 1
    while a < b:
        b = a
        a = ((n - 1) * a + x // pow(a, n - 1)) // n

    return b

def ilog(a: int, b: int = 2) -> int:
    """
    Find the integer logarithm of a with base b.
    Returns the largest integer n such that b^n <= a.
    Uses repeated squaring and binary search.
    """
    if a < 1 or b < 2:
        raise ValueError("Invalid input for integer logarithm")

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

def is_square(n: int) -> bool:
    """
    Check if an integer n is a square.
    """
    return n >= 0 and (n & 0xF) in (0, 1, 4, 9) and (sqrt_n := isqrt(n)) * sqrt_n == n

def non_squares(N: int) -> Iterator[int]:
    """
    Return all non-square positive integers <= N.
    """
    return (n for n in range(2, N + 1) if not is_square(n))

def squares(low: int = 0, high: int | None = None) -> Iterator[int]:
    """
    Generate square numbers in the range [low, high].
    """
    low, high = max(low, 0), inf if high is None else high
    i = isqrt(int(low))
    while (n := i*i) < low:
        i += 1
    while n <= high:
        yield n
        n += 2*i + 1
        i += 1

def cubes(low: int = 0, high: int | None = None) -> Iterator[int]:
    """
    Generate cube numbers in the range [low, high].
    """
    high = inf if high is None else high
    i = iroot(low, 3)
    while (n := i*i*i) < low:
        i += 1
    while n <= high:
        yield n
        n += 3*i*i + 3*i + 1
        i += 1

def perfect_power(n: int) -> tuple[int | None, int | None]:
    """
    Find integers a, b with b > 1 such that a^b = n (if any such integers exist).

    Parameters
    ----------
    n : int
        Integer target
    """
    if n in (0, 1):
        return (n, 2)
    if n == -1:
        return (-1, 3)

    is_negative = n < 0
    m = -n if is_negative else n

    # Squares only make sense for non-negative n (and we special-case them for speed)
    if not is_negative:
        r = isqrt(m)
        if r * r == m:
            return (r, 2)

    for k in primes(low=3, high=m.bit_length()):
        r = iroot(m, k)
        if pow(r, k) == m:
            return (-r if is_negative else r), k

    return None, None

def binary_search(
    f: Callable[[int], int],
    threshold: int,
    low: int = 0,
    high: int | None = None,
) -> int:
    """
    Given a monotonically increasing function f, find where it crosses a threshold.
    Returns the smallest integer n in [low, high] such that f(n) >= threshold.
    """
    if high is None:
        span = 1
        while f(low + span) < threshold: span *= 2
        high = low + span

    return low + bisect.bisect_left(range(low, high + 1), threshold, key=f)

def _threshold_select(
    value: int,
    thresholds: list[tuple[int, int]],
    default: int) -> int:
    """
    Select result based on threshold ranges.
    Returns the result for the smallest (max_val, result) pair where value <= max_val.
    If value exceeds all thresholds, returns default.

    Parameters
    ----------
    value : int
        Value to check against thresholds
    thresholds : list[tuple[int, int]]
        List of (max_value, result) pairs
    default : int
        Value to return if value exceeds all thresholds
    """
    for max_val, result in sorted(thresholds, key=lambda x: x[0]):
        if value <= max_val:
            return result
    return default



########################################################################
############################### Extras ################################
########################################################################

@small_cache
def fibonacci(i: int, mod: int | None = None) -> int:
    """
    Return the i-th Fibonacci number.

    Parameters
    ----------
    i : int
        Index of the Fibonacci number
    mod : int
        Optional modulus
    """
    if i < 0:
        f = (-1)**((i % 2) + 1) * fibonacci(-i, mod)
    elif i <= 70:
        phi = (1 + sqrt(5)) / 2
        f = round(phi**i / sqrt(5))
    elif i % 2 == 0:
        i = i // 2
        f = fibonacci(i + 1, mod)**2 - fibonacci(i - 1, mod)**2
    else:
        i = (i + 1) // 2
        f = fibonacci(i, mod)**2 + fibonacci(i - 1, mod)**2

    return f % mod if mod else f

def fibonacci_index(base: int, exp: int = 1) -> int:
    """
    Find the index of base^exp in the Fibonacci sequence.
    Returns the largest integer i such that F(i) <= base^exp.

    Parameters
    ----------
    base : int
        Base of the Fibonacci number
    exp : int
        Exponent of the Fibonacci number
    """
    if base <= 1 or exp <= 0:
        raise ValueError("The value of `base^exp` must be greater than one.")

    # Parameters in logspace
    phi = (1 + sqrt(5)) / 2  # golden ratio
    log_phi = log(phi)
    log_sqrt5 = 0.5 * log(5.0)
    log_target = exp * log(base) # log(base^exp) = exp * log(base)

    # Find Fibonacci index
    i = max(1, int((log_target + log_sqrt5) / log_phi))
    while i > 1 and log(fibonacci(i)) > log_target:
        i -= 1
    while log(fibonacci(i + 1)) <= log_target:
        i += 1

    return i

def fibonacci_numbers(a: int = 0, b: int = 1, mod: int | None = None) -> Iterator[int]:
    """
    Generate Fibonacci numbers.

    Parameters
    ----------
    a : int
        First element of the Fibonacci sequence
    b : int
        Second element of the Fibonacci sequence
    mod : int
        Optional modulus
    """
    a = a % mod if mod else a
    b = b % mod if mod else b
    while True:
        yield a
        a, b = b, a + b
        b = b % mod if mod else b

def polygonal(s: int, i: int) -> int:
    """
    Return the i-th s-gonal number.
    """
    return (s - 2) * i * (i - 1) // 2 + i

def polygonal_index(s: int, n: int) -> int:
    """
    Find the index of n in the s-gonal numbers.
    Returns the largest integer i such that P(s, i) <= n.
    """
    if s == 2:
        return n
    else:
        return (isqrt(8 * n * (s - 2) + (s - 4) * (s - 4)) + s - 4) // (2 * (s - 2))

def polygonal_numbers(s: int, low: int = 1, high: int | None = None) -> Iterator[int]:
    """
    Generate all s-gonal numbers in the range [low, high].
    """
    i = polygonal_index(s, low - 1) + 1
    n = polygonal(s, i)
    while high is None or n <= high:
        yield n
        n += (s - 2) * i + 1
        i += 1

def is_polygonal(s: int, n: int) -> bool:
    """
    Check if n is an s-gonal number.
    """
    if s == 2:
        return True

    D = 8 * n * (s - 2) + (s - 4) * (s - 4)
    sqrt_D = isqrt(D)
    return sqrt_D*sqrt_D == D and (sqrt_D + s - 4) % (2*s - 4) == 0

def digit_sum(n: int) -> int:
    """
    Return the sum of the digits in the decimal integer n.
    """
    n = abs(n)
    mod = _int_str_mod()
    total = 0
    while n > 0:
        n, r = divmod(n, mod)
        total += sum(map(int, str(r)))

    return total

def digit_count(n: int) -> int:
    """
    Return the number of digits in the decimal integer n.
    """
    n = abs(n)
    if n < _int_str_mod(): return len(str(n))
    return ilog(n, 10) + 1 if n != 0 else 1

def digit_combinations(max_digits: int) -> Iterator[tuple[int, int]]:
    """
    Generate unique digit combinations as integers.

    Parameters
    ----------
    max_digits : int
        Maximum number of digits in the combinations

    Yields
    ------
    a : int
        Digit combination as an integer
    count : int
        Number of unique permutations of the digit combination
    """
    factorials = [factorial(i) for i in range(max_digits + 1)]
    for d in range(1, max_digits + 1):
        for digits in itertools.combinations_with_replacement(string.digits, d):
            i = next((i for i, ch in enumerate(digits) if ch != '0'), None)
            if i is None: continue # skip all-zero case
            digit_counts = Counter(digits)
            count = factorials[d - 1] * (d - digit_counts['0'])
            count //= prod(map(factorials.__getitem__, digit_counts.values()))
            yield int(''.join(digits[i:] + digits[:i])), count

def digit_permutations(n: int) -> Iterator[int]:
    """
    Generate all unique permutations of the digits in integer n.

    Parameters
    ----------
    n : int
        Integer whose digit to permute
    """
    digits = f'{abs(n)}'
    sign = -1 if n < 0 else 1

    if n == 0:
        yield 0
        return

    # Fast path: all digits distinct -> just use permutations
    if len(set(digits)) == len(digits):
        for p in itertools.permutations(digits):
            if p[0] != '0':
                yield sign * int(''.join(p))
        return

    # General path: multiset permutations via digit counts
    counts, length = Counter(map(int, digits)), len(digits)

    def backtrack(pos: int, cur: int) -> Iterator[int]:
        if pos == length:
            yield sign * cur
            return
        for d in counts:
            if counts[d] and (pos or d): # no leading zero
                counts[d] -= 1
                yield from backtrack(pos + 1, cur * 10 + d)
                counts[d] += 1

    yield from backtrack(0, 0)

def digits_in_base(n: int, b: int) -> tuple[int, ...]:
    """
    Return the digits of integer n in base b,
    from least significant digit to most significant digit.
    """
    if abs(b) < 2:
        raise ValueError("|b| must be greater than or equal to 2")
    if n < 0 and b > 0:
        raise ValueError("Positive base b requires n >= 0")
    elif n == 0:
        return (0,)

    # Long division to extract digits
    digits = []
    while n != 0:
        n, r = divmod(n, b)
        if r < 0:
            r += abs(b)
            n += 1
        digits.append(r)

    return tuple(digits)



########################################################################
############################## Constants ###############################
########################################################################

@small_cache
def _int_str_mod():
    """
    Return safe modulus (10^n) for chunking integers during string conversion.
    Respects sys.get_int_max_str_digits() limit, capped at 10^10000.
    """
    num_digits = getattr(sys, 'get_int_max_str_digits', lambda: 0)()
    return 10**min(num_digits or 10000, 10000)
