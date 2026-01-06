import numthy as nt
import itertools
import math
import string

from collections import Counter, defaultdict, deque
from functools import cache
from heapq import heappop, heappush
from itertools import combinations, combinations_with_replacement, islice, permutations
from math import ceil, comb, factorial, floor, gcd, isqrt, lcm, log, prod, sqrt



def problem_1(N=1000, factors=(3, 5)):
    """
    Find the sum of all natural numbers below N that are divisible
    by any of the given factors.

    Notes
    -----
    The sum of the first n multiples of k is equal to k * T_n,
    where T_n is the n-th triangular number.

    From there we can use the principle of inclusion-exclusion to find the sum
    of all multiples of the given factors below N.
    """
    factors = set(factors)
    T = lambda n: n * (n + 1) // 2
    sum_of_multiples = lambda k: k * T((N - 1) // k)
    return sum(
        (-1)**(i + 1) * sum_of_multiples(lcm(*subset))
        for i in range(1, len(factors) + 1)
        for subset in combinations(factors, i)
    )

def problem_2(N=4_000_000):
    """
    Find the sum of all even Fibonacci numbers less than or equal to N.

    Notes
    -----
    F(n) is even if and only if n is a multiple of 3.
    The sum of F(3i) from i = 1 ... n = (F(3n+2) - 1) / 2.
    """
    n = nt.fibonacci_index(N) // 3
    return (nt.fibonacci(3*n + 2) - 1) // 2

def problem_3(n=600851475143):
    """
    Find the largest prime factor of n.
    """
    return max(nt.prime_factors(n))

def problem_4(n=3):
    """
    Find the largest palindrome that is the product of two n-digit numbers.

    Notes
    -----
    Assume that the solution x * y is a palindrome with 2n digits.

    Consider the specific case of x = 10^n - 1, y = 10^n - 10^(n/2) + 1.
    If n is even, then xy is the palindrome 999...000...999 with exactly n/2
    leading and trailing 9s.

    Let k <= n be a lower bound on the # of consecutive digits of 9 that start and end
    the palindromnt. This implies the following:

        * both x, y > (10^k - 1) * 10^(n-k) (since x, y must each also start with k 9s)
        * xy = -1 mod 10^k (since xy ends with k 9s)
        * one of x, y is divisible by 11 (xy is a palindrome with even # of digits)

    We use the heuristic lower bound k = ⌊n/2⌋ (provably holds when n is even).
    """
    is_palindrome = lambda n: (s := f'{n}') == s[::-1]
    best_palindrome = None if n != 1 else 9
    k = n // 2 # lower bound on # of leading / trailing 9s
    while best_palindrome is None and k > 0:
        mod = 10**k # modulus for finding modular inverse
        min_palindrome = 10**(2*n) - 10**(2*n-k) + 10**k - 1 # starts & ends with k 9s
        max_factor = 10**n - 1 # largest n-digit number

        # Iterate x over odd multiples of 11
        x_min = min_palindrome // max_factor
        x_max = ((max_factor - 11) // 22) * 22 + 11 # round down to odd multiple of 11
        for x in range(x_max, x_min - 1, -22):
            # Find modular inverse of x mod 10^k
            if x % 5 == 0: continue # no inverse exists if gcd(x, 10^k) != 1
            else: x_inv = pow(x, -1, mod=mod)

            # Iterate over y such that xy = -1 mod 10^k (i.nt. xy ends with k 9s)
            y_min = (best_palindrome or min_palindrome) // x
            y_max = max_factor + 1 - x_inv
            for y in range(y_max, y_min - 1, -mod):
                if is_palindrome(product := x * y):
                    best_palindrome = product
                    break

            if x * max_factor < (best_palindrome or min_palindrome):
                break

        k -= 1

    return best_palindrome

def problem_5(N=20):
    """
    Find the smallest number that is evenly divisible by all numbers from 1 to N.

    Notes
    -----
    LCM(1, ... N) = product of p^⌊log_p(N)⌋ for p <= N.
    """
    return prod(p**nt.ilog(N, p) for p in nt.primes(high=N))

def problem_6(N=100):
    """
    Find the difference (1 + 2 + ... + N)^2 - (1^2 + 2^2 + ... + N^2).

    Notes
    -----
    We can apply Faulhaber's formula for k = 1 and k = 2.
    The sum of the first N natural numbers is N * (N + 1) / 2.
    The sum of the first N squares is N * (N + 1) * (2N + 1) / 6.
    """
    square_of_sums = (N * (N + 1) // 2) ** 2
    sum_of_squares = N * (N + 1) * (2*N + 1) // 6
    return square_of_sums - sum_of_squares

def problem_7(n=10001):
    """
    Find the n-th prime number.

    Notes
    -----
    For all n >= 6, the n-th prime p_n satisfies the following bounds:

        n(log(n) + log(log(n)) - 1) < p_n < n(log(n) + log(log(n)))

    We can find p_n via binary search with the prime counting function π(x),
    which is asymptotically faster than sieving all primes up to p_n for large n.
    """
    if n < 6:
        return (2, 3, 5, 7, 11)[n - 1]

    low, high = int(n * (log(n) + log(log(n)) - 1)), int(n * (log(n) + log(log(n))))
    return nt.binary_search(nt.count_primes, threshold=n, low=low, high=high)

def problem_8(n=13, path='data/problem_8.txt'):
    """
    Given a string of digits, find the largest product of n consecutive digits.

    Notes
    -----
    We can ignore any subsequence containing a zero.
    """
    with open(path) as file:
        digits = ''.join(line.strip() for line in file.readlines())

    # Split into subsequences of nonzero digits of length at least n
    sub_seqs = [list(map(int, s)) for s in digits.split('0') if len(s) >= n]
    if len(sub_seqs) == 0:
        return 0

    # Find the maximum of the max products in each subsequence
    return max(max([prod(s[i:i+n]) for i in range(len(s) - n + 1)]) for s in sub_seqs)

def problem_9(n=1000):
    """
    Find the product abc of a Pythagorean triple (a, b, c) where a + b + c = n.

    Notes
    -----
    Euclid's formula generates all primitive Pythagorean triples:
    a = x^2 - y^2, b = 2xy, c = x^2 + y^2, for all x > y > 0 where gcd(x, y) = 1
    and exactly one of x, y is even.

    Thus for any Pythagorean triple (ka, kb, kc) with perimeter n,
    we have n = ka + kb + kc = 2kx(x + y), with x, y satisfying the above conditions.

    This implies that n must be even, and x, (x + y) must be factors of n/2.
    We can also infer additional bounds x > 1 and (x + y) < n/2.
    """
    if n % 2 != 0:
        return None

    # Iterate over divisor pairs, where factors[i] = x and factors[j] = x + y
    factors = nt.divisors(n // 2)
    for i in range(1, len(factors) // 2): # x > 1
        for j in range(i + 1, len(factors) - 1): # (x + y) < n/2
            x, y = factors[i], factors[j] - factors[i]
            if x <= y:
                break

            # Check for primitive conditions
            if (x + y) % 2 == 1 and gcd(x, y) == 1:
                a, b, c = x*x - y*y, 2*x*y, x*x + y*y
                k = n // (a + b + c)
                return k*a * k*b * k*c

def problem_10(N=2_000_000):
    """
    Find the sum of all primes below N.
    """
    return nt.sum_primes(N - 1)

def problem_11(n=4, path='data/problem_11.txt'):
    """
    Find the largest product of k consecutive numbers in a grid.
    """
    # Load grid
    with open(path) as file:
        grid = [[int(i) for i in line.strip().split()] for line in file.readlines()]
        H, W = len(grid), len(grid[0])

    # Generate rows, columns, diagonals, and antidiagonals
    rows, cols = [[] for _ in range(H)], [[] for _ in range(W)]
    diagonals = [[] for _ in range(H + W - 1)]
    anti_diagonals = [[] for _ in range(H + W - 1)]
    for r in range(H):
        for c in range(W):
            rows[r].append(grid[r][c])
            cols[c].append(grid[r][c])
            diagonals[r + c].append(grid[r][c])
            anti_diagonals[H - r + c - 1].append(grid[r][c])

    # Find max product of sub-segment of length k
    segments = rows + cols + diagonals + anti_diagonals
    return max(
        prod(segment[i:i+n])
        for segment in segments
        for i in range(len(segment) - n + 1)
    )

def problem_12(D=500):
    """
    Find the smallest triangle number with more than D divisors.

    Notes
    -----
    If gcd(a, b) = 1, then d(ab) = d(a)d(b), where d(n) is the # of divisors of n.
    For all positive integers n we have gcd(n, n + 1) = 1.
    """
    T = lambda n: n * (n + 1) // 2
    i, divisor_counts = 0, nt.divisor_count_range(2)
    while True:
        # Add more divisor counts if necessary
        if i >= len(divisor_counts) - 1:
            divisor_counts = nt.divisor_count_range(2 * i)

        # Calculate the number of divisors for the i-th triangle number
        if i % 2 == 0:
            num_divisors = divisor_counts[i // 2] * divisor_counts[i + 1]
        else:
            num_divisors = divisor_counts[(i + 1) // 2] * divisor_counts[i]

        # Check if the number of divisors exceeds n
        if num_divisors > D:
            return T(i)
        else:
            i += 1

def problem_13(n=10, path='data/problem_13.txt'):
    """
    Find the first n digits of the sum of the given numbers.
    """
    with open(path) as file:
        return str(sum(map(int, file.readlines())))[:n]

def problem_14(N=1_000_000):
    """
    Find the number below N that produces the longest Collatz sequence.

    Notes
    -----
    We can collapse steps in the Collatz sequence by considering n mod 4:

        If n = 4k, we have 4k → 2k → k
        If n = 4k + 1, we have 4k + 1 → 12k + 4 → 6k + 2 → 3k + 1
        If n = 4k + 2, we have 4k + 2 → 2k + 1 → 6k + 4 → 3k + 2
        If n = 4k + 3, we have 4k + 3 → 12k + 10 → 6k + 5 → 18k + 16 → 9k + 8

    Also note that if n < N/2, then 2n has a longer sequence than n,
    so we need only look for n >= N/2.
    """
    @cache
    def collatz(n):
        if n <= 2:
            return n
        elif n % 4 == 0:
            return collatz(n // 4) + 2
        elif n % 4 == 1 or n % 4 == 2:
            return collatz(n - n // 4) + 3
        else:
            return collatz(9 * (n - 3) // 4 + 8) + 4

    return max(range(N // 2, N), key=collatz)

def problem_15(H=20, W=20):
    """
    Find the number of paths from the top-left to bottom-right corner of a grid.

    Notes
    -----
    We need to make H + W moves in total, H of which are down and W of which are right.
    The number of different ways we can do this is (H + W) choose H.
    """
    return comb(H + W, H)

def problem_16(n=1000):
    """
    Find the sum of the digits of 2^n.
    """
    return nt.digit_sum(2**n)

def problem_17(N=1000):
    """
    Find the number of letters used to write out the numbers 1 to N in words.
    """
    digits = {
        0: '', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
        5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
    }
    teens = {
        10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
        15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen',
    }
    tens = {
        2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty',
        6: 'sixty', 7: 'seventy', 8: 'eighty', 9: 'ninety',
    }

    def int_to_words(n: int) -> str:
        if n in digits:
            return digits[n]
        elif n in teens:
            return teens[n]
        elif n < 100:
            return tens[n // 10] + digits[n % 10]
        elif n < 1000:
            return digits[n // 100] + 'hundred' + (
                'and' + int_to_words(n % 100) if 0 < n % 100 else '')
        elif n < 1000000:
            return int_to_words(n // 1000) + 'thousand' + (
                'and' if 0 < n % 1000 < 100 else '') + int_to_words(n % 1000)

    return sum(len(int_to_words(n)) for n in range(1, N + 1))

def problem_18(path='data/problem_18.txt'):
    """
    Find the maximum sum of a path from the top to the bottom of a triangle.
    """
    with open(path) as file:
        pyramid = [list(map(int, line.strip().split(' '))) for line in file.readlines()]

    # Calculate path sums via dynamic programming
    # where pyramid[i][j] is max sum from (i, j) to bottom
    for i in range(len(pyramid) - 2, -1, -1):
        for j in range(len(pyramid[i])):
            pyramid[i][j] += max(pyramid[i + 1][j], pyramid[i + 1][j + 1])

    return pyramid[0][0]

def problem_19(start_year=1901, end_year=2000):
    """
    How many Sundays fell on the first of the month during the 20th century?
    """
    import datetime
    return sum(
        datetime.date(year, month=month, day=1).weekday() == 6 # Sunday
        for year in range(start_year, end_year + 1)
        for month in range(1, 12 + 1)
    )

def problem_20(n=100):
    """
    Find the sum of the digits in the number n!.
    """
    return nt.digit_sum(factorial(n))

def problem_21(N=10000):
    """
    Find the sum of all amicable numbers below N.

    Notes
    -----
    The aliquot sum s(n) = σ(n) - n is the sum of proper divisors of n.
    A number n is amicable if and only if s(s(n)) = n and s(n) != n.
    """
    s = nt.aliquot_sum_range(N)
    return sum(n for n in range(N) if s[n] < N and n != s[n] and s[s[n]] == n)

def problem_22(path='data/problem_22.txt'):
    """
    Find the sum of the name scores in the given file,
    where the score of the n-th name is n times the sum of its character values
    ('a' = 1, 'b' = 2, etc).
    """
    with open(path) as file:
        names = [name.strip('"') for name in ''.join(file.readlines()).split(',')]
        names = sorted(names)

    char_values = {c: ord(c) - ord('A') + 1 for c in string.ascii_uppercase}
    name_values = [sum(map(char_values.__getitem__, name)) for name in names]
    return sum((i + 1) * value for i, value in enumerate(name_values))

def problem_23(N=28123):
    """
    Find the sum of all the positive integers below N which cannot be written
    as the sum of two abundant numbers.

    Notes
    -----
    It can be verified that every integer >= 20162 can be written as the sum of
    two abundant numbers.

    Every multiple of 6 above 6 is abundant (since 6 is a perfect number).

    Then if a_k is the smallest abundant number congruent to k mod 6, it follows that
    a_k + 6i is an abundant sum for all integers i > 1.

    In each residue class k mod 6, we must also check numbers below a_k + 12
    for abundant sums. It can be verified that the only such sum is 40 = 20 + 20
    for k = 4 mod 6.

    We can also verify that all abundant numbers below 20162 are 0, 2, 3, or 4 mod 6.

    Then if n = 1 mod 6 is an abundant sum, the two summands must 3 and 4 mod 6,
    and if n = 5 mod 6 is an abundant sum, the two summands must be 3 and 2 mod 6.
    """
    N = min(N, 20162) # all numbers >= 20162 are abundant sums

    # Calculate the sum of proper divisors for n = 1 ... N - 1
    proper_divisor_sums = nt.aliquot_sum_range(N)

    # Get abundant numbers (grouped mod 6)
    is_abundant = lambda n: n > 0 and proper_divisor_sums[n] > n
    abundant_numbers = [n for n in range(12, N) if is_abundant(n)]
    abundant_numbers = nt.group_by_key(abundant_numbers, key=lambda n: n % 6)

    # Find abundant sums (0, 2, 3, or 4 mod 6)
    abundant_sums = {
        *range(12 + min(abundant_numbers[0], default=N), N, 6), # 0 mod 6
        *range(12 + min(abundant_numbers[2], default=N), N, 6), # 2 mod 6
        *range(12 + min(abundant_numbers[3], default=N), N, 6), # 3 mod 6
        *range(12 + min(abundant_numbers[4], default=N), N, 6), # 4 mod 6
        *([40] if N > 40 else []) # edge case (40 = 20 + 20)
    }

    # Find abundant sums (1 or 5 mod 6)
    for n in range(N):
        if n % 6 in (1, 5):
            if any(is_abundant(n - a) for a in abundant_numbers[3]):
                abundant_sums.add(n)

    return sum(n for n in range(N) if n not in abundant_sums)

def problem_24(n=1_000_000):
    """
    Find the n-th lexicographic permutation of the digits 0-9.

    Notes
    -----
    There are 9! permutations starting with 0, another 9! starting with 1, etc.
    Within each group of 9! permutations, there are 8! permutations where the next
    digit is 0, another 8! permutations where the next digit is 1, etc.
    """
    a, remaining_digits, permutation = n - 1, list(range(10)), []
    while remaining_digits:
        i, a = divmod(a, factorial(len(remaining_digits) - 1))
        permutation.append(remaining_digits.pop(i))

    return ''.join(map(str, permutation))

def problem_25(n=1000):
    """
    Find the index of the smallest n-digit Fibonacci number.

    Notes
    -----
    By Carmichael's theorem, if i > 12, then the i-th Fibonacci number F(i)
    has at least one prime divisor that does not divide any F(j) for j < i.

    Since F(3) = 2 and F(5) = 5, it follows that F(i) cannot be a power of 10
    for any i > 12. Combined with manual verification of F(1) ... F(12),
    it follows that given n > 1, there exists no i such that F(i) = 10^(n - 1).

    Therefore if F(i) is the largest Fibonacci number F(i) <= 10^(n - 1),
    then F(i + 1) is the smallest Fibonacci number with at least n digits.

    We also have F(0) = 0 as the smallest 1-digit Fibonacci number.
    """
    return 0 if n == 1 else nt.fibonacci_index(10**(n - 1)) + 1

def problem_26(N=1000):
    """
    Find the integer reciprocal 1/n with the longest decimal repetend.

    Notes
    -----
    The length L(n) of the decimal repetend of 1/n always divides φ(n),
    where φ is the totient function, with equality if and only if
    10 is a primitive root mod n.

    The totient function is maximized for primes, where φ(p) = p - 1.
    """
    # Check primes in descending order
    is_primitive_root = lambda g, p: nt.multiplicative_order(g, p) == p - 1
    for n in range(N, 1, -1):
        if nt.is_prime(n) and is_primitive_root(10, n):
            return n

def problem_27(N=1000):
    """
    Find the product of coefficients a, b where |a|, |b| <= N such that the
    quadratic expression f(x) = x^2 + ax + b produces the maximum number of primes
    for consecutive integer values of x, starting with x = 0.

    Notes
    -----
    We know that f(0) = b must be prime, and f(1) = 1 + a + b must be prime.
    We can also infer the bounds f(0) <= N and f(1) <= 2N + 1.
    """
    f = lambda a, b, x: x*x + a*x + b
    prime_list = list(nt.primes(high=2*N+1))

    # Iterate over a, b such that f(0) and f(1) are prime by construction
    max_x, best_a, best_b = 0, 0, 0
    for f0 in prime_list:
        if f0 > N: break
        for f1 in prime_list:
            a, b = f1 - f0 - 1, f0
            if a > N: break
            x = 2 # start with f(2), since we already know f(0) and f(1) must be prime
            while nt.is_prime(f(a, b, x)):
                x += 1
            if x > max_x:
                max_x, best_a, best_b = x, a, b

    return best_a * best_b

def problem_28(n=1001):
    """
    Find the sum of the numbers on the diagonals of the n x n Ulam spiral.

    Notes
    -----
    We can represent the sum for an n x n spiral via recurrence S(n),
    where S(1) = 1 and S(2n + 1) = S(2n - 1) + 4n^2 + 10n + 10.

    Solving the recurrence: S(2n + 1) = (16n^3 + 30n^2 + 26n + 3) / 3
    Or alternatively: S(n) = (4n^3 + 3n^2 + 8n - 9) / 6
    """
    assert n % 2 == 1
    return (4*n*n*n + 3*n*n + 8*n - 9) // 6

def problem_29(N=100):
    """
    Given integers 2 <= a,b <= N, find the number of distinct values of a^b.

    Notes
    -----
    We will say that a ∈ ℕ has "exponential order k" if k is the largest integer
    for which there exists some c ∈ ℕ such that a = c^k.

    Thus for a given k, the order-k integers are 2^k, 3^k, 5^k, 6^k, 7^k, 10^k, etc.
    Instances like 4^k or 8^k are of order 2k and 3k respectively.

    Notice that for each a of order k, the number of distinct values of a^b
    is the same (nt.g. 2^k, 3^k, 5^k, 6^k, ... all have the same count).

    It suffices to calculate the number of distinct values of a^b for each order k.

    Let a be an order-k integer. Then the duplicates are of the form a^b,
    where b = m * lcm(j, k) / k for j = 1 ... k - 1 and m = 1 ... N * j / lcm(j, k).
    """
    # Store the number of duplicates for each exponential order k
    num_duplicates = {1: 0}
    for k in range(2, nt.ilog(N, 2) + 1):
        is_duplicate = bytearray([False]) * N
        for j in range(1, k):
            stop, step = (N * j) // k + 1, lcm(j, k) // k
            count = (stop - 1) // step + 1
            is_duplicate[0:stop:step] = bytearray([True]) * count

        num_duplicates[k] = sum(is_duplicate[2:])

    # Calculate exponential order of each base integer a
    exp_order = {}
    for a in range(2, N + 1):
        x = 1
        for k in range(1, nt.ilog(N, a) + 1):
            x *= a
            if x not in exp_order:
                exp_order[x] = k

    # Get total number of distinct powers
    total_num_duplicates = sum([num_duplicates[exp_order[a]] for a in range(2, N + 1)])
    return (N - 1) * (N - 1) - total_num_duplicates

def problem_30(n=5):
    """
    Find the sum of all numbers (>= 2 digits) that can be written as the sum of the
    n-th powers of their digits.

    Notes
    -----
    A k-digit number is at least 10^(k-1), and the sum of n-th powers of
    a k-digit number is at most k * 9^n. This means we can upper bound the
    number of digits by the largest k satisfying 10^(k-1) <= k * 9^n.

    Notice that there are at most ((10 + k - 1) choose k) unique digit power sums,
    as the order of the digits doesn't matter.
    """
    digit_powers = {str(d): d**n for d in range(10)}
    digit_power_sum = lambda digits: sum(map(digit_powers.__getitem__, digits))

    # Find upper bound on number of digits
    k = 2
    while 10**k <= (k + 1) * 9**n:
        k += 1

    # Find numbers that satisfy the condition
    values = set()
    for digits in combinations_with_replacement(string.digits, k):
        a = digit_power_sum(digits)
        b = digit_power_sum(str(a))
        if a == b and a >= 10:
            values.add(a)

    return sum(values)

def problem_31(total=200, coin_values=(1, 2, 5, 10, 20, 50, 100, 200)):
    """
    How many different ways can the specified total be made using
    any number of coins from the given denominations?
    """
    return nt.count_partitions(total, restrict=set(coin_values).__contains__)

def problem_32():
    """
    Find the sum of all products c = a * b such that abc is a 1-9 pandigital number.

    Notes
    -----
    Given a * b = c, if the total # of digits across a, b, c is 9,
    then c must have exactly 4 digits.

    Without loss of generality, assume a < b. Then a must have 1 or 2 digits,
    and b must have 3 or 4 digits.
    """
    digits = set('123456789')
    is_pandigital = lambda s: len(s) == len(digits) and set(s) == digits
    return sum({
        a * b
        for a in range(1, 98 + 1)
        for b in range(123, 9876 // a + 1)
        if is_pandigital(f'{a}{b}{a*b}')
    })

def problem_33():
    """
    Find the denominator of the product of all fractions that can be reduced
    by cancelling a nonzero digit from a 2-digit numerator & denominator.

    Notes
    -----
    We are looking for fractions of the form (10a + x) / (10x + b) = a / b,
    where 0 < a < b < 10 and 0 < x < 10.

    Let k = b - a > 0. Then this reduces to finding integer solutions to
    9a^2 + 9ak - 9ax + kx = 0.
    """
    fractions = [
        (10 * a + x, 10 * x + b)
        for a in range(1, 10)
        for b in range(a + 1, 10)
        for x in range(1, 10)
        if (10*a + x) * b == (10*x + b) * a
    ]

    numerator, denominator = (prod(group) for group in zip(*fractions))
    return denominator // gcd(numerator, denominator)

def problem_34():
    """
    Find the sum of all numbers that are equal to the
    sum of the factorial of their digits.

    Notes
    -----
    The sum of the factorial of the digits of a k-digit number is at most k * 9!.
    We have 10^(k-1) > k * 9! for k > 8, so we only need to check up to 7-digit numbers.

    Notice that there are at most ((10 + k - 1) choose k) unique digit factorial sums,
    as the order of the digits doesn't matter.
    """
    digit_factorials = {str(d): factorial(d) for d in range(10)}
    digit_factorial_sum = lambda digits: sum(map(digit_factorials.__getitem__, digits))

    values = set()
    for k in range(2, 8):
        for digits in combinations_with_replacement(string.digits, k):
            a = digit_factorial_sum(digits)
            b = digit_factorial_sum(str(a))
            if a == b and a >= 10:
                values.add(a)
    
    return sum(values)

def problem_35(N=1_000_000):
    """
    Find the number of circular primes below N.

    Notes
    -----
    For k >= 2, all k-digit circular primes contain only the digits 1, 3, 7, 9.
    """
    to_integers = lambda iterable: map(int, map(''.join, iterable))

    # Get candidate primes with digits 1, 3, 7, 9
    prime_list = list(nt.primes(high=min(9, N - 1))) # single-digit primes
    for k in range(2, nt.digit_count(N - 1) + 1):
        for n in to_integers(itertools.product('1379', repeat=k)):
            if n < N and nt.is_prime(n):
                prime_list.append(n)

    # Find circular primes
    circular_primes = []
    for p in prime_list:
        rotations = (int(str(p)[i:] + str(p)[:i]) for i in range(len(str(p))))
        if all(map(nt.is_prime, rotations)):
            circular_primes.append(p)

    return len(circular_primes)

def problem_36(N=1_000_000):
    """
    Find the sum of all numbers less than N which are palindromic in
    both base 2 and base 10.
    """
    max_num_digits = nt.digit_count(N - 1)
    is_binary_palindrome = lambda n: bin(n)[2:] == bin(n)[2:][::-1]

    # Generate even-length decimal palindromes
    even_limit = 10**(max_num_digits // 2)
    palindromes = [int(f'{n}{str(n)[::-1]}') for n in range(1, even_limit)]

    # Generate odd-length decimal palindromes
    odd_limit = 10**((max_num_digits - 1) // 2)
    palindromes += list(range(1, 10))
    palindromes += [
        int(f'{n}{d}{str(n)[::-1]}')
        for n in range(1, odd_limit) for d in range(10)
    ]

    # Filter for binary palindromes
    return sum(n for n in palindromes if n < N and is_binary_palindrome(n))

def problem_37():
    """
    Find the sum of all primes that are both left-truncatable and right-truncatable.

    Notes
    -----
    If k-digit prime q is right-truncatable, then q = 10*p + d,
    where d is 1, 3, 7, or 9 and p is a (k-1)-digit right-truncatable prime.
    """
    is_left_truncatable = lambda p: all(
        nt.is_prime(int(str(p)[i:]))
        for i in range(1, len(str(p)))
    )

    # Generate all right-truncatable primes
    right_truncatable_primes = list(nt.primes(high=9)) # single-digit primes
    for k in itertools.count(start=1):
        k_digit_primes = [p for p in right_truncatable_primes if len(str(p)) == k]
        if len(k_digit_primes) == 0:
            break

        for p in k_digit_primes:
            for d in (1, 3, 7, 9): # cannot end in even digit or 5
                if nt.is_prime(q := 10 * p + d):
                    right_truncatable_primes.append(q)

    # Filter for primes that are also left-truncatable
    truncatable_primes = [p for p in right_truncatable_primes if is_left_truncatable(p)]
    return sum(truncatable_primes) - sum(nt.primes(high=9))

def problem_38():
    """
    What is the largest 1-9 pandigital number that can be formed as the
    concatenated product concat(a, 2a, ..., na) for some integer n > 1?
    """
    digits = set('123456789')
    concat_prod = lambda a, n: ''.join(str(a * i) for i in range(1, n + 1))
    is_pandigital = lambda s: len(s) == len(digits) and set(s) == digits
    has_repeated_digits = lambda n: len(set(s := f'{n}')) < len(s)

    # Find the largest pandigital concatenated product
    max_concat_prod = int(concat_prod(9, 5))
    for a in range(10000):
        # Skip numbers that cannot form a larger pandigital number
        num_digits = nt.digit_count(a)
        if a < max_concat_prod // 10**(9 - num_digits):
            continue
        if has_repeated_digits(a) or '0' in str(a):
            continue

        # Generate concatenated products
        for n in range(2, 9 // num_digits + 1):
            if is_pandigital(concat_prod(a, n)):
                max_concat_prod = max(max_concat_prod, int(concat_prod(a, n)))

    return max_concat_prod

def problem_39(N=1000):
    """
    Find p <= N that produces the most Pythagorean triples (a, b, c)
    with a + b + c = p.
    """
    counts = Counter(map(sum, nt.pythagorean_triples(max_sum=N)))
    return max(range(N + 1), key=counts.__getitem__)

def problem_40(idx=tuple(10**i for i in range(7))):
    """
    Find the product of the digits at the specified indices
    in Champernowne's constae.
    """
    idx, digits = sorted(idx), []
    n, current_index = 1, 0
    while idx:
        # Find the n-digit number containing the next desired index
        num, i = divmod(idx[0] - current_index - 1, n)
        num += 10**(n - 1)
        if num < 10**n:
            # Store the i-th digit of this number
            digits.append(int(str(num)[i]))
            idx.pop(0)
        else:
            # Update the current Champernowne index (i.nt. append all n-digit numbers)
            current_index += 9 * 10**(n - 1) * n
            n += 1

    return prod(digits)

def problem_41():
    """
    Find the largest pandigital prime.

    Notes
    -----
    All n-digit pandigital numbers are divisible by 3 for n = 2, 3, 5, 6, 8, 9,
    as the sum of their digits is divisible by 3, and the only 1-digit pandigital
    number is 1, which is not prime.

    Thus, we only need to check 4-digit and 7-digit pandigital numbers.
    """
    # Iterate over candidates in descending order
    for n in (7, 4):
        for digits in permutations(range(n, 0, -1)):
            a = int(''.join(map(str, digits)))
            if nt.is_prime(a):
                return a

def problem_42(path='data/problem_42.txt'):
    """
    Find the number of words in a file for which the sum of their character values
    is a triangle number ('a' = 1, 'b' = 2, etc).
    """
    with open(path) as file:
        words = (word.strip('"') for word in ','.join(file.readlines()).split(','))

    char_values = {c: ord(c) - ord('A') + 1 for c in string.ascii_uppercase}
    word_values = [sum(map(char_values.__getitem__, word)) for word in words]
    triangle_numbers = set(nt.polygonal_numbers(3, high=max(word_values)))
    return sum(x in triangle_numbers for x in word_values)

def problem_43():
    """
    Find the sum of all 10-digit pandigital numbers where the number from 
    digits 2-4 is divisible by 2, digits 3-5 divisible by 3,
    digits 4-6 are divisible by 5, etc.

    Notes
    -----
    We can construct pandigital numbers from right to left,
    checking each divisibility condition as we go.
    """
    has_repeated_digits = lambda n: len(set(s := f'{n}')) < len(s)
    candidates = {f'{n:03d}' for n in range(17, 1000, 17)}
    candidates = {n for n in candidates if not has_repeated_digits(n)}

    # Construct numbers from right to left
    digits = set(string.digits)
    for p in (13, 11, 7, 5, 3, 2, 1):
        candidates = {
            f'{d}{n}' # prepend digit d
            for n in candidates # to each of our existing candidates
            for d in digits - set(n) # for each remaining unused digit
            if int(f'{d}{n[:2]}') % p == 0 # if first 3 digits are divisible by p
        }

    return sum(map(int, candidates))

def problem_44(n=1):
    """
    Find the difference of the n-th pair of pentagonal numbers for which
    both their sum and difference are pentagonal, in increasing order of
    their difference.

    Notes
    -----
    Let P(i) = (3i^2 - i)/2 be the i-th pentagonal number.

    We are looking for a pair of pentagonal numbers P(i) and P(j) with i > j
    such that there exist s, d > 0 with P(i) + P(j) = P(s) and P(i) - P(j) = P(d).

    Since we are looking to minimize P(d), we can equivalently rewrite this as
    P(d) + P(j) = P(i) and P(d) + 2*P(j) = P(s), which allows us to successively
    check each d = 1, 2, ... until we find a solution.

    For any given d, we are looking for integers j that satisfy the two conditions.
    Examining the first condition, we can see that P(d) + P(j) = P(i) is pentagonal
    when (3d^2 - d)/2 + (3j^2 - j)/2 = (3i^2 - i)/2. Using the quadratic formula to
    solve for j, we find that this has positive integer solutions j = (1 + y) / 6
    only when 12d - 36d^2 + (6i - 1)^2 = y^2 is a perfect square.

    Let x = 6i - 1 and N = 36d^2 - 12d. Then this reduces to the quadratic Diophantine
    equation x^2 - y^2 = N, which has integer solutions when (x - y)(x + y) = N.
    Thus, for each factorization of N = A * B with A < B and A = B mod 2,
    we have x = (A + B) / 2 and y = (B - A) / 2, and therefore i = (x + 1) / 6
    and j = (y + 1) / 6 as integer solutions.

    Now given that d, i, j are integer indices such that P(d) = P(i) - P(j),
    we only need to check whether P(i) + P(j) is pentagonal.
    """
    count = 0
    P = lambda k: nt.polygonal(5, k)
    for d in itertools.count(start=1):
        factors = nt.divisors(36*d*d - 12*d)
        for k in range(len(factors) // 2):
            # Get factor pair a*b = N
            a, b = factors[k], factors[-k-1]
            if (b - a) % 2 != 0:
                continue

            # Get corresponding x, y values
            x, y = (a + b) // 2, (b - a) // 2
            if x % 6 != 5 or y % 6 != 5:
                continue

            # Get pentagonal indices i, j
            i, j = (x + 1) // 6, (y + 1) // 6
            if nt.is_polygonal(5, P(i) + P(j)):
                count += 1

            # Return the pentagonal difference
            if count == n:
                return P(d)

def problem_45(N=40755):
    """
    Find the next triangle number above N that is also pentagonal and hexagonal.

    Notes
    -----
    All hexagonal numbers are also triangular.

    From the closed-form expressions for pentagonal and hexagonal numbers,
    we have that n is pentagonal if x^2 = 24n + 1 = 5 mod 6 is a perfect square,
    and that n is hexagonal if y^2 = 8n + 1 = 3 mod 4 is a perfect square.

    This reduces to the Pell equation x^2 - 3y^2 = -2.
    """
    for x, y in nt.pell(D=3, N=-2):
        if x % 6 == 5 and y % 4 == 3:
            n = (y*y - 1) // 8
            if n > N:
                return n

def problem_46():
    """
    Find the smallest odd composite number that cannot be written as
    n = p + 2a^2, where p is prime and a is a positive integer.
    """
    for n in itertools.count(start=9, step=2):
        if nt.is_prime(n):
            continue
        if not any(nt.is_prime(n - 2*a*a) for a in range(1, isqrt(n // 2) + 1)):
            return n

def problem_47(n=4, k=4):
    """
    Find the first n consecutive integers to have k distinct prime factors each.

    Notes
    -----
    The number of distinct prime factors of n is given by the omega function ω(n).
    """
    def omega(block_size=1000):
        yield from ((0, 0), (1, 0))
        low, high = 2, 2 + block_size
        small_primes = list(nt.primes(high=isqrt(high)))
        while True:
            # Add more primes if needed
            while (p := small_primes[-1]) <= isqrt(high - 1):
                small_primes += nt.primes(low=p+1, high=2*p)

            # Initialize block
            num_prime_factors = [0] * block_size
            residual = list(range(low, high)) # unfactored portion of each number

            # Factor out primes
            for p in small_primes:
                for idx in range((-low) % p, block_size, p):
                    num_prime_factors[idx] += 1
                    x = residual[idx] // p
                    while x % p == 0:
                        x //= p
                    residual[idx] = x

            # Any residual > 1 now contributes one more (large) prime factor
            for i, x in enumerate(residual):
                num_prime_factors[i] += (x > 1)

            # Yield from this segment
            yield from enumerate(num_prime_factors, start=low)
            low, high = high, high + block_size

    num_consecutive = 0
    for i, num_factors in omega():
        if num_factors == k:
            num_consecutive += 1
        else:
            num_consecutive = 0
        if num_consecutive == n:
            return i - n + 1

def problem_48(N=1000, k=10):
    """
    Find the k trailing digits of the series 1^1 + 2^2 + ... + N^N.
    """
    mod = 10**k
    return sum([pow(i, i, mod=mod) for i in range(1, N + 1)]) % mod

def problem_49(n=4, k=3, exclude=(1487, 4817, 8147)):
    """
    Find an arithmetic sequence of k n-digit primes, each of which are permutations.
    """
    seen = set()
    prime_set = set(nt.primes(low=10**(n-1), high=10**n)) # n-digit primes
    for p in prime_set:
        if p in seen:
            continue

        # Generate all prime permutations of p
        prime_permutations = {int(''.join(digits)) for digits in permutations(str(p))}
        prime_permutations = prime_set.intersection(prime_permutations)
        seen.update(prime_permutations)

        # Find arithmetic sequence
        for sequence in combinations(sorted(prime_permutations), k):
            diffs = [a - b for a, b in itertools.pairwise(sequence)]
            if diffs == [diffs[0]] * (k - 1) and sequence != exclude:
                return ''.join(map(str, sequence))

def problem_50(N=1_000_000):
    """
    Which prime below N can be written as the sum of the most consecutive primes?

    Notes
    -----
    Let s(k) be the cumulative sum of the first k primes.
    Then the sum of primes p_i + ... + p_j is s(j) - s(i - 1).
    """
    # Calculate cumulative sums of primes
    prime_cumsum = itertools.accumulate(nt.primes())
    prime_cumsum = list(itertools.takewhile(lambda s: s < N, prime_cumsum))
    max_chain_length = len(prime_cumsum)

    # Search for the longest valid chain of consecutive primes
    for chain_length in range(max_chain_length, 0, -1):
        best_prime = None
        for i in range(max_chain_length - chain_length + 1):
            prime_sum = prime_cumsum[i + chain_length - 1]
            prime_sum -= prime_cumsum[i - 1] if i > 0 else 0
            if prime_sum >= N:
                break
            if nt.is_prime(prime_sum):
                best_prime = prime_sum

        if best_prime is not None:
            return best_prime

def problem_51(n=8):
    """
    Find the smallest prime which, by replacing part of the number
    with the same digit, is part of a family of exactly n primes.

    Notes
    -----
    For a family of k-digit primes, they differ by multiples of step size Δ
    which contains 1 at the digit positions to be replaced, and 0 everywhere else,
    forming an arithmetic progression.

    For example, the family (56003, 56113, 56333, 56443, 56663, 56773, 56993)
    differs by multiples of 00110.

    Any family of size n must have at least one number with digit d < 11 - n
    at the replaced positions, as otherwise there would be fewer than n members
    in the family.

    If n > 4, then the last digit cannot be replaced, since the last digit must be
    1, 3, 7, or 9 to be prime.

    If n > 7, then the number of digits to be replaced must be a multiple of 3, as
    otherwise there are at least 3 multiples of 3 among the 10 potential family members.
    """
    find_indices = lambda digits, d: [i for i, x in enumerate(digits) if x == d]
    for k in itertools.count(start=1):
        # Generate all k-digit primes
        prime_list = list(nt.primes(low=10**(k-1), high=10**k)) # k-digit primes
        prime_set = set(prime_list)

        # Check each k-digit prime for valid families of size n
        for p in prime_list:
            for d in string.digits[:11-n]: # only check digits d <= 10 - n
                # Replace a subset of the positions where digit d occurs
                for idx in nt.powerset(find_indices(str(p), d)):
                    if not idx: continue # skip the empty set
                    if n > 4 and k - 1 in idx: continue # our n > 4 condition
                    if n > 7 and len(idx) % 3 != 0: continue # our n > 7 condition
                    step = sum(10**(k - i - 1) for i in idx)
                    start, stop = p - int(d) * step, p + (10 - int(d)) * step
                    family = [q for q in range(start, stop, step) if q in prime_set]
                    if len(family) == n:
                        return family[0]

def problem_52(k=6):
    """
    Find the smallest positive integer n such that 2n, 3n, ..., kn
    all have the same digits in some order.

    Notes
    -----
    If 2n, 3n, ... kn all have the same digits, then they have the same digital
    sum S, and thus are all congruent to S mod 9. If k > 2, this implies n = 0 mod 9.

    Given that 2n, 3n, ... kn all have the same length L, then this also implies
    the bounds 10^(L-1) / 2 <= n <= (10^L - 1) / k.
    """
    if k <= 2:
        return 1

    is_permutation = lambda a, b: sorted(f'{a}') == sorted(f'{b}')
    for L in itertools.count(start=2):
        low, high = 10**(L-1) // 2, (10**L - 1) // k
        low = ((low + 9 - 1) // 9) * 9 # round up to nearest multiple of 9
        for x in range(low, high + 1, 9):
            if all(is_permutation(2*x, i*x) for i in range(3, k + 1)):
                return x

def problem_53(N=100, T=1_000_000):
    """
    Find the number of pairs (n, k) such that (n choose k) > T, for n = 1, 2, ..., N.

    Notes
    -----
    For each n, let k_n <= ⌊n/2⌋ be the smallest k such that (n choose k) > T.
    Then for each n there are (n + 1 - 2k_n) total pairs that exceed the threshold T.

    We can also infer that k_n <= k_{n-1}, since (n choose k) must be greater than
    ((n - 1) choose k).

    In addition, we can iteratively calculate the binomial coefficient 
    via the following identities:

        (n choose (k - 1)) = (n choose k) * k / (n - k + 1)

        ((n + 1) choose k) = (n choose k) * (n + 1) / (n - k + 1)

    Finally, given threshold T > 1, it can be shown that (n choose k) <= T
    for all n <= log_2(T) + 1 (proof omitted).
    """
    # Find lower bound on n
    n_min = nt.ilog(T, 2) + 2
    n_min = nt.binary_search(lambda n: comb(n, n // 2), threshold=T+1, low=n_min)

    # Count the total number of values that exceed the threshold
    k, count = n_min // 2, 0
    binomial_coefficient = comb(n_min - 1, k - 1)
    for n in range(n_min, N + 1):
        # Update binomial coefficient
        binomial_coefficient *= n
        binomial_coefficient //= (n - k + 1)

        # Decrement k until (n choose (k - 1)) falls below threshold
        while binomial_coefficient > T:
            k -= 1
            binomial_coefficient *= k
            binomial_coefficient //= (n - k + 1)

        # Update count
        count += n + 1 - 2*k

    return count

def problem_54(path='data/problem_54.txt'):
    """
    Given a list of poker hands, find the number of hands won by player 1.
    """
    CARD_VALUES = {
        **{str(i): i for i in range(2, 10)},
        'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    }

    def get_rank(hand):
        values = sorted([CARD_VALUES[card[0]] for card in hand], reverse=True)
        values = [5, 4, 3, 2, 1] if values == [14, 5, 4, 3, 2] else values
        suits = {card[1] for card in hand}
        is_straight = (values == list(range(values[0], values[0] - 5, -1)))
        is_flush = (len(suits) == 1)
        pairs = sorted({v for v in values if values.count(v) == 2}, reverse=True)
        triples = sorted({v for v in values if values.count(v) == 3}, reverse=True)
        quads = sorted({v for v in values if values.count(v) == 4}, reverse=True)

        if is_straight and is_flush:
            return (8, *values) # straight flush
        elif quads:
            return (7, *quads, *values) # four of a kind
        elif triples and pairs:
            return (6, *triples, *pairs, *values) # full house
        elif is_flush:
            return (5, *values) # flush
        elif is_straight:
            return (4, *values) # straight
        elif triples:
            return (3, *triples, *values) # three of a kind
        elif pairs:
            return (len(pairs), *pairs, *values) # one or two pair
        else:
            return (0, *values) # high card

    with open(path) as file:
        hands = [line.strip().split(' ') for line in file.readlines()]
        hands = [(hand[:5], hand[5:]) for hand in hands]

    return sum(get_rank(player_1) > get_rank(player_2) for player_1, player_2 in hands)

def problem_55(N=10000, max_iterations=50):
    """
    Find the number of Lychrel numbers below N.

    Notes
    -----
    If a non-Lychrel number has sequence a_1, a_2, ..., a_n, then
    all of a_1, ..., a_{n-1} are also non-Lychrel numbers.
    """
    is_palindrome = lambda n: (s := f'{n}') == s[::-1]
    non_lychrel = set()
    for n in range(N):
        sequence = [n]
        for _ in range(max_iterations):
            # Extend the sequence
            sequence.append(sequence[-1] + int(str(sequence[-1])[::-1]))

            # Check if new element is a palindrome (or leads to one)
            if sequence[-1] in non_lychrel or is_palindrome(sequence[-1]):
                non_lychrel.update(sequence[:-1])
                break

    return sum(n not in non_lychrel for n in range(N))

def problem_56(N=100):
    """
    Considering natural numbers of the form a^b with a, b < N,
    find the maximum digital sum.

    Notes
    -----
    The exponential a^b has ⌊b * log_10(a) + 1⌋ digits with a digital sum
    no greater than 9 * ⌊b * log_10(a) + 1⌋ <= 9 * b * ⌊log_10(a) + 1⌋.
    """
    max_digit_sum = 0
    for a in range(N - 1, 0, -1):
        for b in range(N - 1, 0, -1):
            if 9 * b * (nt.ilog(a, 10) + 1) < max_digit_sum:
                break
            if (s := nt.digit_sum(a**b)) > max_digit_sum:
                max_digit_sum = s

    return max_digit_sum

def problem_57(N=1000):
    """
    For each of the first N convergents a/b of the continued fraction for √2,
    how many fractions have a numerator with more digits than the denominator?
    """
    coefficients, _, _ = nt.periodic_continued_fraction(2)
    return sum(
        nt.digit_count(a) > nt.digit_count(b)
        for a, b in nt.convergents(coefficients, num=N+1) # skip 1/1
    )

def problem_58(ratio=0.1):
    """
    Find the side length of the Ulam spiral for which the ratio of primes
    along both diagonals first falls below the given ratio.

    Notes
    -----
    We can see that the total number of diagonal elements for an n x n
    Ulam spiral is 2n - 1.

    The next 4 numbers on the diagonals of the Ulam spiral reaching
    odd side length n are given by n^2 - i*(n - 1) for i = 0, 1, 2, 3,
    where i = 0 can be ignored as n^2 cannot be prime.
    """
    num_primes = 0
    for n in itertools.count(start=3, step=2):
        diagonals = (n*n - i*(n - 1) for i in range(1, 4))
        num_primes += sum(map(nt.is_prime, diagonals))
        if num_primes < ratio * (2*n - 1):
            return n

def problem_59(path='data/problem_59.txt'):
    """
    Decrypt the message in the given file using a simple XOR cipher.
    Find the sum of the ASCII values of the decrypted text.
    """
    common_characters = (' ', 'e', 'a', 'r', 'i', 'o', 't', 'n', 's', 'l', 'c')
    printable_codes = set(map(ord, string.printable))

    def decrypt(data, key):
        m = len(key)
        for i, encrypted_code in enumerate(data):
            decrypted_code = encrypted_code ^ key[i % m]
            if decrypted_code in printable_codes:
                yield decrypted_code
            else:
                raise ValueError

    # Load encrypted data
    with open(path) as file:
        data = list(map(int, file.read().split(',')))

    # Get most frequent encrypted code at each key offset
    most_frequent_encrypted_codes = [
        Counter(data[offset::3]).most_common(1)[0][0]
        for offset in range(3)
    ]

    # Generate possible key codes at each offset
    # Match most frequent encrypted code to most frequent English characters
    key_codes = [
        [code ^ ord(char) for char in common_characters]
        for code in most_frequent_encrypted_codes
    ]

    # Attempt to decrypt data with each key
    for key in itertools.product(*key_codes):
        try:
            return sum(decrypt(data, key))
        except ValueError:
            pass

def problem_60(n=5):
    """
    Find the lowest sum for a set of n primes for which
    any pair of primes concatenate to produce another prime.

    Notes
    -----
    This is equivalent to finding the minimum-weight clique of size n
    in a graph where each vertex is a prime number and an edge exists
    between two primes p and q if both concatenations p||q and q||p are prime.

    Note that every positive integer is congruent to the sum of its digits mod 3.
    Then 3 divides the concatenation of p and q if and only if 3 divides p + q.

    It is also clear that 2 and 5 cannot be part of any clique for n > 1,
    as no other primes can end in 2 or 5.
    """
    if n == 1: return 2
    is_prime = nt.is_prime
    is_prime_concat = lambda p, q: is_prime(int(f'{q}{p}')) and is_prime(int(f'{p}{q}'))

    # Initialize the graph of primes
    graph = {3: set()}
    prev_high, high = 6, 10000

    # Search for cliques of size n
    while True:
        # Add new primes to the graph with edges indicating concatenatable pairs
        new_primes = tuple(nt.primes(low=prev_high+1, high=high))
        new_prime_pairs = list(combinations(new_primes, 2))
        new_prime_pairs += list(itertools.product(new_primes, graph.keys()))
        graph.update({p: set() for p in new_primes})
        for p, q in new_prime_pairs:
            if (p + q) % 3 == 0: continue
            if is_prime_concat(p, q):
                graph[p].add(q)
                graph[q].add(p)

        # Find cliques of size n and (n - 1)
        maximal_cliques = nt.bron_kerbosch(graph)
        big_cliques = [c for mc in maximal_cliques for c in combinations(mc, n)]
        small_cliques = [c for mc in maximal_cliques for c in combinations(mc, n - 1)]

        # Increase our range of primes if no n-cliques are found
        # or if our (n-1)-cliques are not guaranteed to contain a minimal sum
        if not big_cliques or min(map(sum, small_cliques)) > high:
            high, prev_high = high * 2, high
            continue

        # Try to construct additional n-cliques from (n-1)-cliques by
        # adding a new prime p, where high < p < clique_sum_bound
        clique_sum_bound = min(map(sum, big_cliques))
        prime_list = list(nt.primes(low=high+1, high=clique_sum_bound))
        for clique in small_cliques:
            for p in prime_list:
                if sum(clique) + p >= clique_sum_bound: break
                if (p + clique[0]) % 3 == 0: continue
                if all(is_prime_concat(p, q) for q in clique):
                    big_cliques.append((*clique, p))

        return min(map(sum, big_cliques))

def problem_61(n=8, k=4):
    """
    Find the sum of an ordered set of cyclic k-digit numbers where in
    each consecutive pair (x, y) the last k/2 digits of x match the first k/2
    digits of y, and each s-gonal type from s = 3 ... n, is represented by a
    different number in the set.

    Notes
    -----
    We can create a graph with edges (i, x) -> (j, y) if the last k/2 digits of
    i-gonal number x match the first k/2 digits of j-gonal number y.

    Then the problem reduces to finding a cycle in this graph.
    """
    # Generate k-digit figurate numbers
    figurate_numbers = {
        s: list(nt.polygonal_numbers(s, low=10**(k-1), high=10**k - 1))
        for s in range(3, n + 1)
    }

    # Create a graph of figurate numbers
    graph = defaultdict(set)
    for i, j in permutations(figurate_numbers, 2):
        for x, y in itertools.product(figurate_numbers[i], figurate_numbers[j]):
            if f'{x}'[-k//2:] == f'{y}'[:k//2]:
                graph[(i, x)].add((j, y))

    def find_next(path):
        if not path:
            return list(graph.keys())

        # No repeated figure types (unless we have already used them all)
        figures, values = zip(*path)
        if len(path) == len(figurate_numbers):
            # We've used all types, only allow closing the cycle
            return [node for node in graph[path[-1]] if node[0] not in figures[1:]]
        else:
            # Don't repeat any polygon types
            return [node for node in graph[path[-1]] if node[0] not in figures]

    # Find cycles in the graph
    cycle = next(nt.find_cycles(find_next=find_next), None)
    return sum(value for figure, value in cycle) if cycle else None

def problem_62(n=5):
    """
    Find the smallest cube for which exactly n permutations of its digits are cubes.
    """
    for k in itertools.count(start=1):
        # Generate all k-digit cubes
        low = nt.iroot(10**(k - 1) - 1, 3) + 1
        high = nt.iroot(10**k - 1, 3)
        cubes = (str(i*i*i) for i in range(low, high + 1))

        # Check for a permutation group of size n
        cube_permutations = nt.group_permutations(cubes)
        cube_permutations = [group for group in cube_permutations if len(group) == n]
        if cube_permutations:
            return min(min(map(int, group)) for group in cube_permutations)

def problem_63():
    """
    Find the number of n-digit positive integers that are n-th powers.

    Notes
    -----
    Any such number a^n must satisfy 10^(n-1) <= a^n < 10^n.
    This implies that a < 10 and n < 22.
    """
    return sum(nt.digit_count(a**n) == n for a in range(1, 10) for n in range(1, 22))

def problem_64(N=10000):
    """
    Find the number of continued fractions for √n for which the period is odd.
    """
    get_period_length = lambda n: nt.periodic_continued_fraction(n)[-1]
    return sum(get_period_length(n) % 2 == 1 for n in nt.non_squares(N))

def problem_65(n=100):
    """
    Find the sum of digits in the numerator of the n-th convergent for e.
    """
    coefficients = [2, 1] + [a for k in range(1, n // 3 + 1) for a in (2*k, 1, 1)]
    return nt.digit_sum(nt.nth(nt.convergents(coefficients), n)[0])

def problem_66(N=1000):
    """
    Find the value of D <= N that maximizes x in the fundamental solution
    to Pell's equation x^2 - Dy^2 = 1.
    """
    return max(nt.non_squares(N), key=lambda D: next(nt.pell(D))[0])

def problem_67(path='data/problem_67.txt'):
    """
    Find the maximum sum of a path from the top to the bottom of a triangle.
    """
    return problem_18(path=path)

def problem_68(n=5):
    """
    Find the lexicographically maximum solution for an n-gon ring as a
    concatenated string, where:

        * The ring consists of numbers from 1 ... 2n
        * The ring consists of n overlapping groups of three numbers
        * The last number in each group is the middle number in the next group
        * The last number of the last group is the middle number of the first
        * No other numbers are shared between groups
        * The sum of the numbers in each group is the same
        * The first group has the numerically lowest external node

    Notes
    -----
    Given group (a, b, c), in order to maintain the same group sum, the next 
    group must be of the form (a + i, c, b - i), with the constraint that
    max(1 - a, b - 2n) < i < min(b - 1, 2n - a).

    The middle numbers from each group form an inner ring of size n,
    while the first numbers from each group form an outer ring of size n.

    Finding a valid n-gon ring is equivalent to finding a cycle in the directed
    graph where each node is a triple (a, b, c) representing a group, and there
    is an edge from (a, b, c) to (a + i, c, b - i) for each group that does not
    violate the n-gon constraints.
    """
    contains_duplicates = cache(lambda a: len(set(a)) < len(a))
    all_values = set(range(1, 2*n + 1))

    def find_next(path):
        # Find unused values from 1 ... 2n that are not in the path
        # Go back to start (if possible) when there are no remaining values
        remaining_values = all_values - {value for group in path for value in group}
        if not remaining_values and path[0][1] == path[-1][-1]:
            return [path[0]]

        # Otherwise extend the path
        a, b, c = path[-1]
        min_offset, max_offset = max(path[0][0] + 1 - a, b - 2*n), min(b - 1, 2*n - a)
        external, internal = remaining_values, remaining_values | {path[0][1]}
        return [
            (a + i, c, b - i)
            for i in range(min_offset, max_offset + 1)
            if a + i in external and b - i in internal
            if not contains_duplicates((a + i, c, b - i))
        ]

    # Search for cycles starting from each possible group
    return max(
        ''.join(map(str, sum(cycle, ())))
        for group in permutations(range(1, 2*n + 1), 3)
        for cycle in nt.find_cycles(find_next=find_next, current_path=[group])
    )

def problem_69(N=1_000_000):
    """
    Find the value of n with 1 <= n <= N for which n / φ(n) is a maximum,
    where φ is Euler's totient function.

    Notes
    -----
    From Euler's product formula, we see that n / φ(n) is maximized
    when n is a product of as many different small primes as possible.

    The solution is the largest primorial less than or equal to N.
    """
    primorials = itertools.accumulate(nt.primes(), lambda x, y: x * y)
    return max(itertools.takewhile(lambda p: p <= N, primorials), default=1)

def problem_70(N=10_000_000):
    """
    Find the value of n with 1 < n < N for which φ(n) is a permutation of n
    and the ratio n / φ(n) produces a minimum, where φ is Euler's totient function.

    Notes
    -----
    From Euler's product formula, we see that n / φ(n) is minimized
    when n is a product of as few different large primes as possible.

    It can be shown that φ(p^k) cannot be a permutation of p^k, so
    we need to consider the product of at least two primes.

    If 2817 < N <= 2991, then the optimal solution is n = 2817 = 3 * 3 * 313.

    Conjecture: All other optimal solutions are semiprimes n = pq
    (verified empirically for N <= 10^9).

    Also note that if n = pq and φ(n) = (p - 1)(q - 1) are permutations,
    then they have the same digital sum, and thus must be congruent mod 9.
    Therefore we must have n - φ(n) = p + q - 1 = 0 mod 9, or p + q = 1 mod 9.
    """
    is_permutation = lambda a, b: sorted(f'{a}') == sorted(f'{b}')
    pi = nt.count_primes
    prime_list = list(nt.primes(high=N//2))

    # Special case that needs product of 3 primes
    if 2817 < N <= 2991:
        return 2817 # 3 * 3 * 313

    # Search over semiprimes
    best_n, best_ratio = 0, float('inf')
    for p in reversed(prime_list[:pi(isqrt(N))]):
        for q in reversed(prime_list[:pi(N // p)]):
            n, totient_n = p*q, (p - 1)*(q - 1)
            if (ratio := n / totient_n) >= best_ratio:
                break
            if (p + q) % 9 == 1 and is_permutation(n, totient_n):
                best_n, best_ratio = n, ratio
                break

    return best_n

def problem_71(N=1_000_000, a=3, b=7):
    """
    Given proper fraction a/b such that a < b <= N, find the numerator of
    the largest reduced proper fraction c/d less than a/b, where c < d <= N
    and gcd(c, d) = 1.

    Notes
    -----
    For positive integers a, b, c, d, given that gcd(a, b) = 1,
    the difference a/b - c/d = (ad - bc) / bd will be minimized
    when ad - bc = 1. Bézout's lemma guarantees us that a solution
    exists with c <= a <= N and d <= b <= N.

    Then we have 1 = ad - bc = ad mod b, so our denominator d
    must be congruent to modular inverse d = a^(-1) mod b.

    We need only to select the largest such d <= N, and set the
    numerator to c = (ad - 1) / b.

    Furthermore, ad - bc = 1 also implies gcd(c, d) = 1, so c/d
    must already be a reduced proper fraction.
    """
    g = gcd(a, b)
    a, b = a // g, b // g # reduce fraction
    offset = N - b + 1
    denominator = (pow(a, -1, b) - offset) % b + offset
    numerator = (a * denominator - 1) // b
    return numerator

def problem_72(N=1_000_000):
    """
    Find the number of reduced proper fractions a/b for b <= N.

    Notes
    -----
    We know a/b is a proper reduced fraction if and only if a < b and gcd(a, b) = 1.
    Given b, the number of coprime integers below b is given by the totient function.
    """
    return sum(nt.totient_range(N + 1)[2:])

def problem_73(N=12000, low=1/3, high=1/2):
    """
    Find the number of reduced proper fractions such that low < n/d < high with d <= N.

    Notes
    -----
    For each denominator d, let f(d) = ⌈d*high⌉ - ⌊d*low⌋ - 1 be the total number
    of fractions n/d with low < n/d < high.

    We can see that any fraction a/b can be reduced to (a/k) / (b/k),
    where k = gcd(a, b). It follows that we can also express f(d) as follows:

        f(d) = ∑_{k | d} g(k)

    where g(k) is the number of reduced fractions n/d with low < n/d < high.

    The Möbius inversion formula allows us to express g in terms of f:

        g(d) = ∑_{k | d} μ(k) * f(d/k).

    where μ(k) is the Möbius function.
    """
    count = lambda d: ceil(high * d) - floor(low * d) - 1
    cumsum = list(itertools.accumulate(map(count, range(1, N + 1)), initial=0))
    mu = nt.mobius_range(N + 1)
    return sum(mu[k] * cumsum[N // k] for k in range(1, N + 1))

def problem_74(n=6, k=60):
    """
    For any integer a > 0, consider the sequence a, s(a), s(s(a)), ...
    where s(a) is the sum of the factorial of the digits of a.
    Find the number of integers with at most n digits that result in a
    chain containing exactly k non-repeating terms.

    Notes
    -----
    All chains eventually enter one of the following cycles:

        (1 → 1), (2 → 2), (145 → 145), (40585 → 40585)
        (871 → 45361 → 871), (872 → 45362 → 872)
        (169 → 363601 → 1454 → 169)

    Ignoring leading zeros, integers with the same set of digits result in
    the same value of s(n).

    If the digit counts are a0, ..., a9 with sum a0 + ... + a9 = d,
    then there are P = (d - a0) (d - 1)! / (a0! * ... * a9!) such integers.

    When a is not a permutation of any element in a cycle, then all P
    permutations lead to chains of the same length L.

    When a is a permutation of an element in a cycle of length L,
    then (P - 1) of these permutations will lead to a chain of length L + 1,
    while only the cycle element itself has a chain of length L.

    Therefore we only need to consider one integer for each set of digits
    and adjust by its multiplicity.
    """
    digit_values = {str(d): factorial(d) for d in range(10)}
    s = lambda n: sum(map(digit_values.__getitem__, f'{n}'))
    base_cases = {
        1: 1, 2: 1, 145: 1, 40585: 1,
        871: 2, 45361: 2, 872: 2, 45362: 2,
        169: 3, 363601: 3, 1454: 3,
    }

    @cache
    def chain_length(n: int) -> int:
        return base_cases[n] if n in base_cases else 1 + chain_length(s(n))

    # Map digit permutations of cycle elements to cycle lengths
    permutation_cycle_lengths = {
        b: L for a, L in base_cases.items() for b in nt.digit_permutations(a)}

    # Iterate over digit combinations
    count = 0
    for a, multiplicity in nt.digit_combinations(n):
        if a in permutation_cycle_lengths:
            if k == permutation_cycle_lengths[a]:
                count += 1
            elif k == permutation_cycle_lengths[a] + 1:
                count += multiplicity - 1
        elif chain_length(a) == k:
            count += multiplicity

    return count

def problem_75(N=1_500_000):
    """
    Find the number of n <= N such that a + b + c = n for exactly one
    Pythagorean triple (a, b, c).
    """
    perimeter_counts = Counter(map(sum, nt.pythagorean_triples(max_sum=N)))
    return list(perimeter_counts.values()).count(1)

def problem_76(n=100):
    """
    Return the number of non-trivial partitions of integer n.
    """
    return nt.count_partitions(n) - 1

def problem_77(N=5000):
    """
    Find the smallest integer n that can be written as the sum of primes in
    more than N different ways.
    """
    for n in itertools.count():
        if nt.count_partitions(n, restrict=nt.is_prime) > N:
            return n

def problem_78(N=1_000_000):
    """
    Find the smallest positive integer n for which the value of partition function
    p(n) is divisible by N.
    """
    return next(n for n, p in enumerate(nt.partition_numbers(mod=N)) if p == 0)

def problem_79(path='data/problem_79.txt'):
    """
    Given a set of subsequences, find the shortest possible supersequence.

    Notes
    -----
    If we assume the supersequence has no repeated elements, then we can
    simply create a directed graph of character precedence and perform a
    topological sort.

    Otherwise, this becomes the Shortest Common Supersequence (SCS) problem,
    which is known to be NP-complete.

    Given n subsequences, we keep a state (i_1, i_2, ..., i_n) indicating
    how many characters have been matched from each subsequence so far.

    We can then perform a breadth-first search (BFS) over the state space,
    where from each state we can transition to a new state by adding
    a character that matches the next unmatched character in at least
    one of the subsequences.
    """
    # Create a directed (hopefully acyclic) graph
    # where an edge (u, v) indicates u precedes v
    with open(path) as file:
        sequences = [line.strip() for line in file.readlines()]
        graph = defaultdict(set)
        for sequence in sequences:
            for u, v in itertools.pairwise(sequence):
                graph[u].add(v)

    # Try to perform topological sort on the graph
    try:
        return ''.join(nt.topological_sort(graph))
    except ValueError:
        pass # cycle detected, fall back to SCS solution

    sequence_lengths = tuple(len(s) for s in sequences)
    start, target = (0,) * len(sequences), sequence_lengths
    parent = {start: (None, None)} # (prev_state, item_added)
    queue = deque([start])

    # Run BFS over state space
    while queue:
        state = queue.popleft()
        if state == target:
            break

        # Find candidate sequence items
        # Store which sequences can advance when adding each potential item
        candidates = defaultdict(list)
        for j, sequence in enumerate(sequences):
            i = state[j]
            if i < sequence_lengths[j]:
                candidates[sequence[i]].append(j)

        # Explore candidate states
        for item, idx in candidates.items():
            new_state = list(state)
            for j in idx: new_state[j] = new_state[j] + 1
            new_state = tuple(new_state)
            if new_state not in parent:
                parent[new_state] = (state, item)
                queue.append(new_state)

    # Reconstruct item order
    order, state = [], target
    while parent[state][0] is not None:
        state, item = parent[state]
        order.insert(0, item)

    return ''.join(order)

def problem_80(N=100, k=100):
    """
    Find the sum of the first k digits in the decimal expansion for the
    irrational square roots of integers 1, 2, ..., N (including the integer part).

    Notes
    -----
    To find the decimal expansion of √n to k digits, we can compute
    ⌊√(n * 10^(2k))⌋ and extract the first k digits.
    """
    return sum(
        nt.digit_sum(isqrt(n * 10**(2 * max(0, k - d))) // 10**max(0, d - k))
        for n in nt.non_squares(N)
        for d in (nt.digit_count(isqrt(n)),)
    )

def problem_81(
    path='data/problem_81.txt',
    allowed_moves=('right', 'down'),
    start_nodes=((0, 0),),
    end_nodes=((79, 79),),
):
    """
    Find the minimum path sum from the top left to the bottom right,
    moving only right and down.

    Notes
    -----
    We can model the problem as finding the shortest path in a directed
    weighted graph, where each matrix cell is a node, and edges connect
    nodes according to allowed moves. The weight of each edge is given
    by the value of the destination node.

    We can also generalize to find the shortest path from any set of start nodes
    to any set of end nodes, by adding a 'source' node connected to all start nodes,
    and a 'sink' node connected to all end nodes.
    """
    start_nodes, end_nodes = set(start_nodes), set(end_nodes)
    moves = {'left': (0, -1), 'right': (0, 1), 'up': (-1, 0), 'down': (1, 0)}
    deltas = [moves[move] for move in allowed_moves]

    # Load matrix
    with open(path) as file:
        matrix = [[*map(int, line.strip().split(','))] for line in file.readlines()]
        height, width = len(matrix), len(matrix[0])

    # Modify graph to include 'source' node connected to start nodes
    # and 'sink' node connected to end nodes
    def get_neighbors(node):
        if node == 'source':
            return start_nodes
        elif node == 'sink':
            return []

        r, c = node
        neighbors = [
            (r + dr, c + dc) for dr, dc in deltas
            if 0 <= r + dr < height and 0 <= c + dc < width
        ]

        return neighbors + ['sink'] if node in end_nodes else neighbors

    # Find shortest path from 'source' to 'sink'
    get_edge_weight = lambda u, v: matrix[v[0]][v[1]] if v != 'sink' else 0
    dist, _ = nt.dijkstra('source', get_neighbors, get_edge_weight)
    return dist['sink']

def problem_82(
    path='data/problem_82.txt',
    allowed_moves=('right', 'up', 'down'),
    start_nodes=tuple((r, 0) for r in range(80)),
    end_nodes=tuple((r, 79) for r in range(80)),
):
    """
    Find the minimum path sum from the far right column to the far left column,
    moving only right, up, and down.
    """
    return problem_81(path, allowed_moves, start_nodes, end_nodes)

def problem_83(
    path='data/problem_83.txt',
    allowed_moves=('right', 'left', 'up', 'down'),
    start_nodes=((0, 0),),
    end_nodes=((79, 79),),
):
    """
    Find the minimum path sum from the top left to the bottom right,
    moving right, left, up, and down.
    """
    return problem_81(path, allowed_moves, start_nodes, end_nodes)

def problem_84(n=4, k=3):
    """
    Return the k most likely squares landed on in a game of Monopoly
    with two n-sided dice.

    Notes
    -----
    We can model the game as a Markov chain.

    A game state is a pair (square, d), where square ∈ [0, 39] is the current
    square on the board, and d ∈ [0, 2] is the number of consecutive doubles rolled.
    Thus there are 120 states in total, and and the transitions between states
    are determined by the rules of the game.

    We also collapse all jail states (JAIL, d) into a single state (JAIL, 0),
    as the doubles count is reset when arriving in jail.

    We ignore the fact that each CH or CC card is returned to the bottom of
    the pile after it is used, and instead assume each draw is independent,
    otherwise handling the (16!/14!) * (16!/(6!2!)) ≈ 3.5 trillion unique
    deck states alone becomes infeasible.

    This should have a negligible effect on the overall probabilities,
    and our relative probability ordering should be the same.
    """
    num_squares, max_doubles = 40, 3
    GO, JAIL, GO_TO_JAIL = 0, 10, 30
    CH, CC = (7, 22, 36), (2, 17, 33)
    next_R, next_U = dict(zip(CH, (15, 25, 5))), dict(zip(CH, (12, 28, 12)))
    chance_targets = (GO, JAIL, 11, 24, 39, 5)

    # Create transition matrix T, where T[i][j] is the probability of
    # moving from state i to state j after a single roll
    num_states = num_squares * max_doubles
    get_state = lambda square, d: square + num_squares * d * (square != JAIL)
    T = [[0] * num_states for _ in range(num_states)]

    # Distribute probability mass from a roll in a given state
    def distribute_probability(state, landing_square, d, p):
        landing_square = JAIL if landing_square == GO_TO_JAIL else landing_square
        if landing_square in CH:
            T[state][get_state(landing_square, d)] += p * (6/16)
            T[state][get_state(next_R[landing_square], d)] += p * (2/16)
            back_3 = (landing_square - 3) % num_squares
            for target in (*chance_targets, next_U[landing_square], back_3):
                distribute_probability(state, target, d, p * (1/16))
        elif landing_square in CC:
            for target, count in ((GO, 1), (JAIL, 1), (landing_square, 14)):
                T[state][get_state(target, d)] += p * (count / 16)
        else:
            T[state][get_state(landing_square, d)] += p

    # Enumerate all possible rolls from each state
    for state in range(num_states):
        d, square = divmod(state, num_squares)
        for a, b in itertools.product(range(1, n + 1), repeat=2):
            next_d = d + 1 if a == b else 0
            jail = (a == b) and (next_d == max_doubles)
            target = JAIL if jail else (square + a + b) % num_squares
            distribute_probability(state, target, 0 if jail else next_d, 1/(n*n))

    # Find stationary distribution
    I = nt.identity_matrix(num_states)
    A = nt.matrix_difference(nt.matrix_transpose(T), I)
    A[-1], b = [1] * num_states, [0] * (num_states - 1) + [1]
    probs = nt.linear_solve(A, b)

    # Find most likely squares
    square_probs = [sum(probs[i::num_squares]) for i in range(num_squares)]
    argsort = sorted(range(num_squares), key=square_probs.__getitem__)
    return ''.join(f'{i:02}' for i in argsort[-k:][::-1])

def problem_85(n=2_000_000):
    """
    Find the area of the grid containing the number of rectangles nearest to n.

    Notes
    -----
    Without loss of generality, let h <= w be the height and width of the grid.

    The number of rectangles in an h x w grid is the number of ways
    to choose the two x-coordinates and the two y-coordinates.
    This can be expressed as (h choose 2) * (w choose 2) = T(h) * T(w),
    where T(i) is the i-th triangular number.

    For any given height h, we can target w such that T(w) ≈ n / T(h) = x
    by finding the roots for the quadratic expression for T(w). Since
    w must be an integer, after taking the floor we can look at midpoint
    (T(w) + T(w + 1)) / 2 = (w + 1)^2 / 2 to determine whether to round up or down.
    """
    best_area, best_error = 0, float('inf')
    for h, T_h in enumerate(nt.polygonal_numbers(3, low=1, high=isqrt(n)), start=1):
        w = nt.polygonal_index(3, x := n // T_h)
        w += (2*x > (w + 1) * (w + 1))
        T_w = nt.polygonal(3, w)
        if (error := abs(T_h * T_w - n)) < best_error:
            best_area, best_error = h * w, error

    return best_area

def problem_86(N=1_000_000):
    """
    Find the least value of M such there exist at least N distinct cuboids,
    ignoring rotations, with integer dimensions (x, y, z), where x, y, z <= M,
    for which the shortest path between opposite corners has integral length.

    Notes
    -----
    For any cuboid with dimensions (x, y, z) where x <= y <= z,
    the shortest path from one corner to the opposite corner has integral length
    if and only if (x + y, z) form the legs of a Pythagorean triple.

    Say we are given a Pythagorean triple (a, b, c) with a <= b.
    If z = a, we have satisfying dimensions (b - i, i, a) for i = ⌈b/2⌉ ... a.
    If z = b, we have satisfying dimensions (a - i, i, b) for i = ⌈a/2⌉ ... a - 1.

    For every Pythagorean triple of the form (3k, 4k, 5k), setting z = 4k
    results in ⌊3k/2⌋ satisfying cuboids. Given the constraint z <= M,
    summing Σ 3k/2 over all k = 1 ... M/4 results in a lower bound of
    N >= (3/64) * M * (M + 4) > (3/64) * M^2.

    Thus for any given N, we have the upper bound M < √(64N/3).
    Given that b <= 2a, this also implies the upper bound c <= √5 * M.
    """
    # Compute upper bounds
    max_M, max_c = isqrt(64 * N // 3), isqrt(5 * 64 * N // 3)

    # Calculate the number of cuboids for each M
    num_cuboids = [0] * (max_M + 1)
    for a, b, _ in nt.pythagorean_triples(max_c=max_c):
        if b <= max_M:
            num_cuboids[b] += a // 2
        if a <= max_M and b <= 2 * a:
            num_cuboids[a] += a - (b - 1) // 2

    # Find the smallest M to reach at least N total cuboids
    cumulative_sum = itertools.accumulate(num_cuboids)
    return next(M for M, total in enumerate(cumulative_sum) if total >= N)

def problem_87(N=50_000_000):
    """
    Find the number of positive integers n = p^2 + q^3 + r^4 < N for prime p, q, r.

    Notes
    -----
    Given p, q, r >= 2, this implies the following bounds:

        p <= (N - 1 - 2^3 - 2^4)^(1/2) = (N - 25)^(1/2)
        q <= (N - 1 - 2^2 - 2^4)^(1/3) = (N - 21)^(1/3)
        r <= (N - 1 - 2^2 - 2^3)^(1/4) = (N - 13)^(1/4)

    We can also infer an upper bound on the intermediate sum q^3 + r^4 < N - 4.
    """
    # Precompute prime powers
    squares = [p*p for p in nt.primes(high=isqrt(N - 25))]
    cubes = [q*q*q for q in nt.primes(high=nt.iroot(N - 21, 3))]
    fourths = [r*r*r*r for r in nt.primes(high=nt.iroot(N - 13, 4))]

    # Get all unique q^3 + r^4 sums
    cube_fourth_sums = sorted({x + y for x in fourths for y in cubes if x + y < N - 4})

    # Combine with p^2 values
    seen = bytearray(N)
    for square in squares:
        for m in cube_fourth_sums:
            if (n := m + square) >= N: break
            seen[n] = True

    return seen.count(1)

def problem_88(K=12000):
    """
    Consider a minimal product-sum number as the smallest positive integer that
    is equal to both the sum and product of a sequence of k positive integers.

    Find the sum of all distinct minimal product-sum numbers for sequences of
    length 2 <= k <= K.

    Notes
    -----
    We can represent product-sum numbers as a non-decreasing sequence of integers
    A = (a_1, a_2, ..., a_k).

    For a given sequence A = (a_1, ..., a_j), if n = prod(A) >= sum(A), then by
    padding the sequence with 1's until the sum matches the product, we find that
    n is a product-sum number for sequence length k = len(A) + prod(A) - sum(A).

    Now consider the k-length sequence A = (k, 2, 1, 1, ... 1) with (k - 2) 1's.
    We can see that sum(A) = prod(A) = 2k, so it follows that 2k is an upper bound
    for the minimal product-sum number for sequence length k.
    """
    def find_next(state):
        sequence_length, sequence_sum, sequence_product, sequence_max = state
        if sequence_product == 1:
            limit = 2*K
        else:
            limit = (K - sequence_length + sequence_sum - 1) // (sequence_product - 1)
        for a in range(sequence_max or 2, limit + 1):
            yield (sequence_length + 1, sequence_sum + a, sequence_product * a, a)

    minimal_product_sum_numbers = [2*K] * (K + 1)
    search = nt.search([(0, 0, 1, None)], find_next)
    for sequence_length, sequence_sum, sequence_product, _ in search:
        k = sequence_length + sequence_product - sequence_sum
        if sequence_product < minimal_product_sum_numbers[k]:
            minimal_product_sum_numbers[k] = sequence_product

    return sum(set(minimal_product_sum_numbers[2:]))

def problem_89(path='data/problem_89.txt'):
    """
    Find the total number of characters saved by writing the given
    Roman numerals in their minimal form.

    Notes
    -----
    Assuming the numeral contains no more than four consecutive identical units,
    considering subtractive combinations we find there are only two valid general
    forms of non-minimality:

        VIIII → IX
        IIII  → IV

    where I, X can be swapped with any two consecutive powers of 10,
    and V with the corresponding multiple of 5.
    """
    def to_minimal(roman: str):
        roman = roman.replace('DCCCC', 'CM')
        roman = roman.replace('CCCC', 'CD')
        roman = roman.replace('LXXXX', 'XC')
        roman = roman.replace('XXXX', 'XL')
        roman = roman.replace('VIIII', 'IX')
        roman = roman.replace('IIII', 'IV')
        return roman

    with open(path) as file:
        roman_numerals = file.read()
        return len(roman_numerals) - len(to_minimal(roman_numerals))

def problem_90(targets=(1, 4, 9, 16, 25, 36, 49, 64, 81), rule=str.maketrans('6', '9')):
    """
    Consider a cube with a different digit on each facnt. Given a set of n-digit
    numbers, find the number of distinct arrangements of n such cubes allow for
    all the numbers to be displayed, where the cubes can be used in any order,
    and where one is allowed to flip a cube's 6 to represent a target digit 9
    (or flip a 9 to represent a target digit 6) if needed.

    Notes
    -----
    In this problem, digits 6 and 9 are interchangeable in terms of
    representation power, but still count as distinct faces in terms
    of counting cube arrangements.

    We can ignore the 6/9 ambiguity by replacing all 6's with 9's on
    both the target numbers and the cube faces, essentially collapsing
    them into a single 6/9 symbol, and any cube may use at most two such
    symbols.

    We treat 6/9 as a single symbolic digit in the matching problem,
    but still count them as distinct faces combinatorially.
    """
    n = len(str(max(targets))) if targets else 0
    targets = {tuple(sorted(str(t).zfill(n).translate(rule))) for t in targets}

    # Check if all targets can be covered by some ordering of the groups
    def is_valid(groups):
        for sequence in targets:
            for ordering in permutations(groups):
                for item, group in zip(sequence, ordering):
                    if item not in group: break # missing item
                else: break # every item was found in its group
            else: return False # every ordering led to at least one missing item
        return True # every sequence found at least one good ordering

    # Count how many cube arrangements can display all numbers
    faces = string.digits.translate(rule) # don't deuplicate here
    cubes = combinations(faces, 6) # some cubes will be repeated due to 6/9
    arrangements = combinations_with_replacement(map(set, cubes), n)
    return sum(is_valid(arrangement) for arrangement in arrangements)

def problem_91(n=50):
    """
    How many right triangles with integer coordinates (0, 0), (x1, y1), (x2, y2)
    exist such that 0 <= x1, y1, x2, y2 <= n?

    Notes
    -----
    There are n^2 right triangles with the right angle at the origin.

    Let P = (x1, y1) and Q = (x2, y2) be the other two points, where x1 >= x2.

    If the right angle is at P, then the slope from P to Q (-x1/y1) is the negative
    reciprocal of the slope from the origin to P (y1/x1).

    Then the number of right triangles is the number of integer coordinates
    (x1 - y1 * k/d, y1 + x1 * k/d) within the bounds, where k is a positive integer
    and d = gcd(x1, y1). Of these, exactly n^2 of these triangles have y1 = y2 = 0.

    A similar analysis gives us the same number of right triangles with
    right angle at Q, where exactly n^2 of these triangles have x1 = x2 = 0.
    """
    return 3*n*n + 2*sum(
        min((d := gcd(x1, y1)) * x1 // y1, d * (n - y1) // x1)
        for x1 in range(1, n + 1)
        for y1 in range(1, n + 1)
    )

def problem_92(n=7):
    """
    Given chains of positive integers where each term is followed by the
    sum of the squares of its digits, find the number of starting integers
    below 10^n that arrive at 89.

    Notes
    -----
    Every integer with the same digits is followed by the same chain.
    If the digit counts are a_1, ..., a_k, then there are
    (a_1 + ... + a_k)! / (a_1! * ... * a_k!) such integers.

    Therefore we only need to consider one integer for each unique set of digits.
    """
    digit_values = {str(d): d*d for d in range(10)}
    end = lambda x: x if x in (0, 1, 89) else end(sum(map(digit_values.get, str(x))))
    end = cache(end)
    return sum(count for a, count in nt.digit_combinations(n) if end(a) == 89)

def problem_93(n=4):
    """
    Find the set of n digits for which the longest sequence of consecutive
    target values 1 ... n can be generated using each digit exactly once
    and the operations (+, -, *, /).

    Notes
    -----
    An expression of binary operators can be represented as a binary tree,
    where the leaves are the input digits and the internal nodes are the operations.

    Given n leaves, there are C_{n-1} unique structures for binary trees,
    where C_i is the i-th Catalan number.

    We can reduce the number of trees substantially by treating all operations
    as commutativnt. To do this, we must also include two new binary operators:
    rsub(x, y) := y - x and rdiv(x, y) := y / x.
    """
    nan = float('nan')

    @cache
    def get_targets(variables: frozenset):
        if (k := len(variables)) <= 1: return variables
        tiebreaker = next(iter(variables)) # avoids double-counting commutative splits
        return frozenset(
            z
            for i in range(1, k // 2 + 1)
            for subset in map(frozenset, combinations(variables, i))
            if i != k - i or tiebreaker in subset
            for left, right in ((subset, variables - subset),)
            for x, y in itertools.product(get_targets(left), get_targets(right))
            for z in (x + y, x - y, y - x, x * y, x / (y or nan), y / (x or nan))
        )

    # Account for floating point precision issues
    def smoothing(group):
        return frozenset(round(x) for x in group if x > 0 and abs(x - round(x)) < 1e-9)

    # Find longest sequence of consecutive target values starting from 1
    get_sequence = lambda x: range(1, next(i for i in itertools.count(1) if i not in x))
    best_group = max(
        map(frozenset, combinations(range(1, 10), n)),
        key=lambda group: len(get_sequence(smoothing(get_targets(group)))),
    )

    return ''.join(map(str, sorted(best_group)))

def problem_94(N=10**9):
    """
    Find the sum of perimeters of all triangles with side lengths of the form
    (a, a, a ± 1) with integer area and perimeter not exceeding N.

    Notes
    -----
    From Heron's formula, a triangle with sides (a, a, a + 1) has integer area
    if and only if (3a^2 - 2a - 1) / 4 = y^2 is a perfect square.

    Let a = 2k - 1 and x = 3k - 2 (perimeter = 3a + 1 = 2x + 2).
    Then this reduces to the Pell equation x^2 - 3y^2 = 1.

    Similarly, a triangle with sides (a, a, a - 1) has integer area
    if and only if (3a^2 + 2a - 1) / 4 = y^2 is a perfect square.

    Let a = 2k + 1 and x = 3k + 2 (perimeter = 3a - 1 = 2x - 2).
    This also reduces to the same Pell equation x^2 - 3y^2 = 1.
    """
    total = 0
    for x, y in nt.pell(3):
        if 2*x - 2 > N:
            break
        if x % 3 == 1 and 4 < (perimeter := 2*x + 2) <= N:
            total += perimeter
        elif x % 3 == 2 and 4 < (perimeter := 2*x - 2) <= N:
            total += perimeter

    return total

def problem_95(N=1_000_000):
    """
    Find the smallest member of the longest aliquot cycle with no element exceeding N.

    Notes
    -----
    This is equivalent to finding the longest cycle in the functional graph
    defined by the aliquot sum function s(n) = σ(n) - n.
    """
    if N < 6:
        return None # no cycles exist for N < 6

    aliquot_sums = nt.aliquot_sum_range(N + 1)
    cycle_lengths, cycle_minimums = defaultdict(int), defaultdict(lambda: float('inf'))

    def on_cycle(cycle_start, cycle_current):
        cycle_lengths[cycle_start] += 1
        cycle_minimums[cycle_start] = min(cycle_minimums[cycle_start], cycle_current)

    nt.find_functional_cycles(
        f=aliquot_sums.__getitem__,
        search=range(1, N + 1),
        domain=range(1, N + 1),
        on_cycle=on_cycle,
    )

    max_cycle_length = max(cycle_lengths.values())
    return min(
        cycle_minimums[cycle]
        for cycle, length in cycle_lengths.items()
        if length == max_cycle_length
    )

def problem_96(path='data/problem_96.txt'):
    """
    Find the sum of the first three digits of the solutions
    to the given sudoku puzzles.
    """
    # Get indices of all cells in the same row, column, or box as the cell at (r, c)
    eliminate_idx = {}
    for r, c in itertools.product(range(9), range(9)):
        idx = {(3*(r//3) + i, 3*(c//3) + j) for i in range(3) for j in range(3)} # box
        idx |= {(i, c) for i in range(9)} # row
        idx |= {(r, j) for j in range(9)} # column
        eliminate_idx[r, c] = list(idx)

    def solve_sudoku(grid: list[list[int]]) -> list[list[int]]:
        # Find the empty cell with the fewest possible options
        all_options = set(range(1, 10))
        best_cell, best_options = None, all_options
        for (r, c) in itertools.product(range(9), range(9)):
            if grid[r][c] == 0:
                options = all_options - {grid[i][j] for i, j in eliminate_idx[r, c]}
                if len(options) < len(best_options):
                    best_cell, best_options = (r, c), options
                if len(options) <= 1:
                    break

        # Check if puzzle is already solved
        if not best_cell: return grid

        # Try each option
        r, c = best_cell
        for option in best_options:
            grid[r][c] = option
            if solve_sudoku(grid):
                return grid

        grid[r][c] = 0
        return None

    # Read sudoku puzzles from file
    with open(path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
        puzzles = [lines[i:i+9] for i in range(1, len(lines), 10)]
        puzzles = [[list(map(int, line)) for line in puzzle] for puzzle in puzzles]

    # Solve puzzles
    total = 0
    for puzzle in puzzles:
        solution = solve_sudoku(puzzle)
        total += int(''.join(map(str, solution[0][:3])))

    return total

def problem_97(a=28433, n=7830457, k=10):
    """
    Find the last k digits of a * 2^n + 1.
    """
    mod = 10**k
    return (a * pow(2, n, mod=mod) + 1) % mod

def problem_98(path='data/problem_98.txt'):
    """
    Given a list of words, consider a pair of perfect square anagrams that
    correspond to a pair of word anagrams by replacing letters with digits.
    Find the largest square in such a pair.

    Notes
    -----
    Each sequence can be identified by a "signature" based on the first
    occurrence of each unique element. For instance, the word "BEER"
    can be represented by the signature (0, 1, 1, 3).

    This can also be applied to anagrams, which can be represented
    generically as ordered pairs of signatures that are permutations of each other.
    For instance, the anagram "CARE" → "RACE" can be represented by the pair
    ((0, 1, 2, 3), (2, 0, 1, 3)).

    Then we need only to examine square number anagram pairs with matching signatures.
    """
    def signature(s: str, index = None) -> tuple:
        return tuple(map(index or s.index, s))

    # Find word anagram pairs
    with open(path) as file:
        words = [s.strip('"') for s in file.read().split(',')]
        anagram_pairs = {
            (signature(w1), signature(w2, index=w1.index))
            for group in nt.group_permutations(words)
            for w1, w2 in permutations(group, 2)
        }

    # Generate square numbers grouped by signature
    max_anagram_length = max(len(pair[0]) for pair in anagram_pairs)
    squares = nt.squares(low=0, high=10**max_anagram_length - 1)
    squares = nt.group_by_key(map(str, squares), key=signature)

    # Find square anagram pairs corresponding to word anagram pairs
    return max(
        n
        for signature_1, signature_2 in anagram_pairs
        for square in squares[signature_1]
        if not (s := ''.join(square[i] for i in signature_2)).startswith('0')
        if nt.is_square(n := int(s))
    )

def problem_99(path='data/problem_99.txt'):
    """
    Given a sequence of (a, b) pairs, find the index that maximizes a^b.

    Notes
    -----
    For any positive a, b we have a > b if and only if log(a) > log(b).
    Therefore, the index that maximizes a^b also maximizes log(a^b) = b * log(a).
    """
    with open(path) as file:
        numbers = (map(int, line.split(',')) for line in file.readlines())
        logs = (b * log(a) for a, b in numbers)
        argmax, _ = max(enumerate(logs, start=1), key=lambda x: x[1]) # 1-indexed
        return argmax

def problem_100(N=10**12):
    """
    Consider a set of n discs, b of which are blue.
    Find the smallest b such that there exists some n > N where the
    probability of drawing two blue discs without replacement is exactly 1/2.

    Notes
    -----
    If n is total number of discs and b is number of blue discs,
    the probability of drawing two blue discs is (b/n) * ((b-1)/(n-1)) = 1/2.

    Thus we need to find integer solutions to n^2 - n - 2b^2 + 2b = 0,
    or equivalently to 4n^2 - 4n - 8b^2 + 8b = 0.

    Let x = 2n - 1 and y = 2b - 1. Then this reduces to the negative
    Pell equation x^2 - 2y^2 = -1.
    """
    for x, y in nt.pell(D=2, N=-1):
        if (x + 1) // 2 > N:
            return (y + 1) // 2

def problem_101(coefficients=(1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1)):
    """
    Find the sum of OP(k, k + 1) such that OP(k, k + 1) != f(k),
    where f(n) is the polynomial defined by the given coefficients,
    and OP(k, n) is the (k-1)-th order polynomial approximation.
    """
    f = nt.polynomial(coefficients)
    values = [f(n) for n in range(1, len(coefficients) + 2)]

    def OP(k, n):
        A = [[i**j for j in range(k)] for i in range(1, k + 1)]
        coefs = list(map(int, nt.linear_solve(A, values[:k])))
        return nt.polynomial(coefs)(n)

    terms = [OP(k, k + 1) for k in range(1, len(coefficients) + 1)]
    return sum(term for term, value in zip(terms, values[1:]) if term != value)

def problem_102(path='data/problem_102.txt'):
    """
    Given a list of triangles defined by their vertices, find the number of
    triangles that contain the origin.

    Notes
    -----
    A triangle contains the point (x, y) if and only if the area of the triangle
    is equal to the total area of the triangles formed by each pair of vertices
    and the point (x, y).
    """
    with open(path) as file:
        triangles = [list(map(int, line.split(','))) for line in file.readlines()]

    def area(x1, y1, x2, y2, x3, y3):
        return abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2)

    count = 0
    for x1, y1, x2, y2, x3, y3 in triangles:
        p0, p1, p2, p3 = (0, 0), (x1, y1), (x2, y2), (x3, y3)
        A = area(*p0, *p1, *p2) + area(*p0, *p2, *p3) + area(*p0, *p1, *p3)
        count += (A == area(*p1, *p2, *p3))

    return count

def problem_103(n=7):
    """
    Find the minimum-sum special sum set of size n, where any two non-empty
    disjoint subsets have different sums, and where any larger subset has
    larger sum than a smaller subset.

    Notes
    -----
    We have the following two constraints:

        1) For any two disjoint non-empty subsets B, C of A: sum(B) != sum(C)
        2) For any such B, C with |B| > |C|: sum(B) > sum(C)

    We can check the 2nd constraint efficiently by computing prefix sums
    and suffix sums of the sorted set, and ensuring that each prefix sum
    of k elements is greater than the corresponding suffix sum of (k - 1) elements.
    """
    def condition_1(A):
        m = len(A)
        # if m < 4:
        #     # For m <= 3 and strictly increasing A, there cannot be
        #     # two disjoint equal-size subsets with equal sum.
        #     return True

        idx = range(m)
        max_r = m // 2
        for r in range(1, max_r + 1):
            sums = {}
            for comb in itertools.combinations(idx, r):
                s = sum(A[i] for i in comb)
                if s in sums:
                    Scomb = set(comb)
                    for prev in sums[s]:
                        if prev.isdisjoint(Scomb):
                            return False
                    sums[s].append(Scomb)
                else:
                    sums[s] = [set(comb)]
        return True
 
    def condition_2(A):
        prefix_sums = list(itertools.accumulate(A))
        suffix_sums = list(itertools.accumulate(reversed(A)))
        for x, y in zip(prefix_sums[1:], suffix_sums):
            if x <= y:
                return False

        return True

    def partial_ok(A):
        """Combined cheap check for partial sets."""
        return condition_2(A) and condition_1(A)

    # --- Build a special seed of size n using the proven "middle element" rule ---

    if n == 1:
        return "1"

    A = [1]
    for _ in range(2, n + 1):
        b = A[len(A) // 2]          # a_{⌊m/2⌋+1}, guaranteed sufficient
        A = [b] + [b + x for x in A]

    # A is now a valid special sum set of size n (by the proof we discussed)
    best_set = tuple(A)
    best_sum = sum(best_set)

    # --- One global DFS with branch-and-bound and cheap partial pruning ---

    def dfs(i, prev_val, partial, partial_sum):
        nonlocal best_set, best_sum

        if i == n:
            if partial_sum < best_sum and condition_2(partial) and condition_1(partial):
                best_set = tuple(partial)
                best_sum = partial_sum
            return

        rem = n - i

        # Global lower bound:
        # fill remaining rem elements with prev_val+1, prev_val+2, ...
        tail_min = rem * (prev_val + 1) + rem * (rem - 1) // 2
        if partial_sum + tail_min >= best_sum:
            return

        rem_after = rem - 1
        # Solve for x range from sum bound:
        # total = partial_sum + x + rem_after*(x+1) + rem_after*(rem_after-1)//2
        # < best_sum
        # => (rem_after+1)*x < best_sum - partial_sum - rem_after*(rem_after+1)//2
        numerator = best_sum - 1 - partial_sum - rem_after * (rem_after + 1) // 2
        if numerator <= 0:
            return

        x_lo = prev_val + 1
        x_hi = numerator // (rem_after + 1)
        if x_lo > x_hi:
            return

        for x in range(x_lo, x_hi + 1):
            partial.append(x)
            if condition_2(partial) and condition_1(partial):
                dfs(i + 1, x, partial, partial_sum + x)
            partial.pop()

    dfs(0, 0, [], 0)
    return ''.join(map(str, best_set))

def problem_104(digits=123456789):
    """
    Find the index of the first Fibonacci number that both begins and ends
    with a permutation of the given digits.

    Notes
    -----
    The k-th Fibonacci number is approximately F(k) ≈ ϕ^k / √5,
    where ϕ is golden ratio.

    Its first n digits are given by int(10**(n - 1 + {log10(F(k))})),
    where log10(F(k)) ≈ k * log10(ϕ) - log10(√5) and {x} denotes the
    fractional part of x.
    """
    is_permutation = lambda a, b: sorted(f'{a}') == sorted(f'{b}')
    num_digits = nt.digit_count(digits)
    mod = 10**num_digits
    log_phi = log((1 + sqrt(5)) / 2, 10)
    log_sqrt_5 = log(sqrt(5), 10)

    for k in itertools.count():
        # Check leading digits
        log_F = k * log_phi - log_sqrt_5
        leading_digits = int(10**(num_digits - 1 + (log_F % 1)))
        if not is_permutation(digits, leading_digits):
            continue

        # Check trailing digits
        trailing_digits = nt.fibonacci(k, mod=mod)
        if is_permutation(digits, trailing_digits):
            return k

def problem_105(path='data/problem_105.txt'):
    """
    TODO: writeup
    """
    def constraint(A):
        # Check constraint 2
        prefix_sums = list(itertools.accumulate(A))
        suffix_sums = list(itertools.accumulate(reversed(A)))
        for x, y in zip(prefix_sums[1:], suffix_sums):
            if x <= y:
                return False

        # Check constraint 1
        for B, C in nt.disjoint_subset_pairs(A, equal_size_only=True):
            if sum(B) == sum(C):
                return False

        return True

    with open(path) as file:
        sets = [tuple(sorted(map(int, line.split(',')))) for line in file.readlines()]
        return sum(sum(A) for A in sets if constraint(A))

def problem_106(n=12):
    """
    TODO: writeup
    """
    return sum(
        1
        for B, C in nt.disjoint_subset_pairs(range(n), equal_size_only=True)
        if any(b < c for b, c in zip(B, C))
        if any(b > c for b, c in zip(B, C))
    )

def problem_107(path='data/problem_107.txt'):
    """
    Find the weight difference between a graph and its and minimum spanning tree.
    """
    # Load adjacency matrix
    with open(path) as file:
        graph = [line.replace('-', '0').split(',') for line in file.readlines()]
        graph = [list(map(int, row)) for row in graph]

    # Find minimum spanning tree
    nodes = range(len(graph))
    edges = [(u, v) for u, v in itertools.combinations(nodes, 2) if graph[u][v] != 0]
    minimum_spanning_tree = nt.kruskal(nodes, edges, lambda u, v: graph[u][v])

    # Calculate weight difference
    initial_weight = sum(graph[u][v] for u, v in edges)
    final_weight = sum(graph[u][v] for u, v in minimum_spanning_tree)
    return initial_weight - final_weight

def problem_108(K=1000):
    """
    Find the smallest value of n such that the number of distinct integer
    solutions to 1/x + 1/y = 1/n exceeds K.

    Notes
    -----
    The number of distinct solutions is given by k = (d(n^2) + 1) / 2,
    where d(n) is the divisor function.

    It also follows that ⌈log(2k - 1) / log(3)⌉ is an upper bound on the
    number of unique prime factors.
    """
    prime_list = list(nt.primes(num=ceil(log(2*K - 1, 3))))
    prev_k = 0
    heap = [(1, 1, [0] * len(prime_list))]
    while True:
        n, k, exponents = heappop(heap)
        if k > K: return n
        if k <= prev_k: continue
        prev_k = k
        for i in range(len(prime_list)):
            if i == 0 or exponents[i] < exponents[i - 1]:
                new_n = n * prime_list[i]
                new_k = k + (2*k - 1) // (2 * exponents[i] + 1)
                new_exponents = exponents.copy()
                new_exponents[i] += 1
                heappush(heap, (new_n, new_k, new_exponents))

def problem_110(K=4_000_000):
    """
    Find the smallest value of n such that the number of distinct solutions
    to 1/x + 1/y = 1/n exceeds K.
    """
    return problem_108(K)

def problem_112(ratio=0.99):
    is_bouncy = lambda n: list(n) != sorted(n) and list(n) != sorted(n, reverse=True)
    count = 0
    for n in itertools.count(start=1):
        if is_bouncy(str(n)):
            count += 1
        if count == n * ratio:
            return n

def problem_118(digits=range(1, 10)):
    """
    How many distinct sets of decimal integers containing each of the
    given digits exactly once, contain only prime elements?

    Notes
    -----
    If (A_1, A_2, ..., A_n) is a partition of the digits set, and P_i is the set of
    primes formed by permuting the elements of A_i, then the prime sets contributed
    by this partition are given by the Cartesian product P_1 * P_2 * ... * P_n.
    """
    def get_partitions(iterable):
        """
        Generate all possible partitions of an iterable.
        """
        iterable = iter(iterable)

        try:
            item = next(iterable)
            for partition in get_partitions(iterable):
                yield ((item,), *partition)
                for i, subset in enumerate(partition):
                    yield ((item, *subset), *partition[:i], *partition[i+1:])

        except StopIteration:
            yield () # empty partition

    prime_sets = []
    for partition in get_partitions(map(str, digits)):
        prime_permutations = []
        for subset in partition:
            perms = permutations(subset) # get digit permutations
            perms = map(int, map(''.join, perms)) # convert to integers
            perms = filter(nt.is_prime, perms) # filter for primes
            prime_permutations.append(perms)

        prime_sets += itertools.product(*prime_permutations)

    return len(prime_sets)

def problem_119(n=30):
    """
    Find the n-th integer that is a power of its digital sum.

    Notes
    -----
    A k-digit integer has a digital sum of at most 9k.
    """
    digit_power_sums = []
    for k in itertools.count(start=2):
        # Iterate over k-digit powers base^exp
        for base in range(2, 9*k + 1):
            min_exp = ceil((k - 1) / math.log10(base))
            max_exp = floor(k / math.log10(base))
            for exp in range(min_exp, max_exp + 1):
                power = base**exp
                if base == nt.digit_sum(power):
                    digit_power_sums.append(power)

        # Check if we have generated enough digit power sums
        if len(digit_power_sums) >= n:
            return sorted(digit_power_sums)[n - 1]

def problem_120(N=1000):
    """
    Let r_a be the maximum residue (a - 1)^n + (a + 1)^n mod a^2 over all n.
    Find the sum r_3 + r_4 + ... + r_N.

    Notes
    -----
    We can ignore all terms quadratic or higher in the binomial expansion of
    (a - 1)^n and (a + 1)^n, as they will all be divisible by a^2.

    If n is even we have r = (a - 1)^n + (a + 1)^n = 2 (mod a^2),
    and if n is odd we have r = (a - 1)^n + (a + 1)^n = 2an (mod a^2).

    Then given a > 2, it follows that the maximal residue is
    r = a(a - 1) for odd a, and r = a(a - 2) for even a.
    """
    return sum(a * (a - gcd(a, 2)) for a in range(3, N + 1))

def problem_121(n=15):
    """
    Find the inverse probability of winning more than half of n total turns,
    when the probability of winning the i-th turn is given by 1 / (i + 1).
    """
    # Use dynamic programming to store probabilities after each turn
    # P[i, j] / i! is the probability of j wins after i turns
    P = defaultdict(int, {(0, 0): 1})
    for i in range(1, n + 1):
        for j in range(i + 1):
            P[i, j] = P[i - 1, j - 1] + i * P[i - 1, j]

    numerator = sum(P[n, j] for j in range(n//2 + 1, n + 1))
    denominator = factorial(n + 1)
    return denominator // numerator

def problem_122(N=200):
    """
    Let m(k) be the length of the shortest addition chain for k.
    Find the sum m(1) + m(2) + ... m(N).

    Notes
    -----
    TODO: star chains, pruning, DFS, what not
    """
    # num_multiplications = {1: 0}
    # seen = set()

    # MAX_DEPTH = 2 * ceil(log(N, 2))
    # def DFS(path=(1,)):
    #     if path in seen or len(path) > MAX_DEPTH:
    #         return
    #     else:
    #         seen.add(path)

    #     for a in path:
    #         for b in reversed(path):
    #             if a + b <= path[-1]: break
    #             if a + b <= N and a + b not in path:
    #                 if (a + b not in num_multiplications
    #                     or len(path) < num_multiplications[a + b]):
    #                     num_multiplications[a + b] = len(path)
    #                 DFS((*path, a + b))

    # DFS()
    # for i in range(1, N + 1):
    #     print(i, num_multiplications[i])

    num_multiplications = {1: 0}
    addition_chains = {(1,)}
    for i in itertools.count(start=1):
        #print(i, len(addition_chains))
        bound = next(i for i in range(1, N + 1) if i not in num_multiplications)

        # Generate new addition chains of length i + 1
        new_addition_chains = {
            (*chain, a + b)
            for chain in addition_chains
            for a, b in itertools.product(chain, chain)
            if chain[-1] < a + b <= N
            and a + b >= bound
        }

        # Update number of multiplications needed for new chains
        for chain in new_addition_chains:
            if chain[-1] not in num_multiplications:
                num_multiplications[chain[-1]] = i

        # Check if we have generated all k <= N
        addition_chains = new_addition_chains
        if all(k in num_multiplications for k in range(1, N + 1)):
            break

    # print(num_multiplications[12509])
    return sum(num_multiplications[n] for n in range(1, N + 1))

def problem_123(N=10**14):
    """
    Find the smallest index n for which the residue
    (p_n - 1)^n + (p_n + 1)^n mod (p_n)^2 exceeds N.

    Notes
    -----
    For any integer a, the residue (a - 1)^n + (a + 1)^n) mod a^2 is
    r = 2 (mod a^2) when n is even, and r = 2an (mod a^2) when n is odd.

    Also note that for all x >= 17, we have π(x) > x / log(x), where π is the
    prime counting function.
    """
    # Find prime upper bound (such that 2 * p_n * n > N)
    p_max = 17
    while 2 * p_max * p_max / log(p_max) <= N:
        p_max *= 2

    # Find residue exceeding N
    for n, p in enumerate(nt.primes(high=p_max), start=1):
        r = 2 if n % 2 == 0 else 2 * n * p
        if r > N:
            return n

def problem_124(N=100000, k=10000):
    """
    Find the k-th positive integer not exceeding N when sorted by rad(n),
    where rad(n) is the product of distinct prime factors of n.
    """
    rad = nt.radical_range(N + 1)
    return sorted(range(1, N + 1), key=rad.__getitem__)[k - 1]

def problem_125(N=10**8):
    """
    Find the sum of all the numbers less than N that are both palindromic
    and can be written as the sum of consecutive squares.
    """
    # max_num_digits = nt.digit_count(N - 1)

    # # Generate even-length decimal palindromes
    # even_limit = 10**(max_num_digits // 2)
    # palindromes = [int(f'{n}{str(n)[::-1]}') for n in range(1, even_limit)]

    # # Generate odd-length decimal palindromes
    # odd_limit = 10**((max_num_digits - 1) // 2)
    # palindromes += list(range(1, 10))
    # palindromes += [
    #     int(f'{n}{d}{str(n)[::-1]}')
    #     for n in range(1, odd_limit) for d in range(10)
    # ]

    # for x in palindromes:


    # print(sorted(palindromes))

    is_palindrome = lambda n: (s := f'{n}') == s[::-1]
    palindromes = set()
    for i in range(1, isqrt(N)):
        sum_of_squares = i*i
        for j in range(i + 1, isqrt(N)):
            sum_of_squares += j*j
            if sum_of_squares >= N: break
            if is_palindrome(sum_of_squares):
                palindromes.add(sum_of_squares)

    return sum(palindromes)

def problem_126():
    """
    1st layer: 2ab + 2ac + 2bc
    6+16+24+32+40
    """

def problem_127(N=120000):
    """
    Find the sum of all c < 120000 such that
    a + b = c, gcd(a, b) = gcd(a, c) = gcd(b, c) = 1,
    and rad(abc) < c.

    Notes
    -----
    If a, b, c are pairwise coprime, then rad(abc) = rad(a) * rad(b) * rad(c).
    Also note that gcd(a, b) = 1 and a + b = c implies gcd(a, c) = gcd(b, c) = 1.
    """
    def coprime_pairs(limit):
        queue = deque([(1, 2)])
        while queue:
            a, b = queue.popleft()
            yield a, b
            if a + b < limit:
                queue.append((a, a + b))
                queue.append((b, a + b))
 
    rad = nt.radical_range(N)
    idx = sorted(range(1, N), key=rad.__getitem__, reverse=True)
    count, total = 0, 0

    # for b, c in coprime_pairs(N):
    #     a = c - b
    #     if a >= b:
    #         continue
    #     if rad[a] * rad[b] * rad[c] < c:
    #         total += c
    #         count += 1
    #         print(a, b, c)

    for c in [17]:
        for a in idx:
            if a > c or gcd(a, c) != 1:
                continue
            b = c - a
            print(a, b, c, rad[a] * rad[b] * rad[c])
            if rad[a] * rad[b] * rad[c] < c:
                total += c
                count += 1
                print(a, b, c)

    # for c in range(1, N):
    #     for a in range(1, c // 2):
    #         if gcd(a, c) != 1:
    #             continue
    #         b = c - a
    #         if rad[a] * rad[b] * rad[c] < c:
    #             total += c
    #             count += 1
    #             print(a, b, c)

    print("Count:", count)
    return total

def problem_128(n=2000):
    """
    Find the 2000th tile in a hexagonal grid for which the difference
    in prime factors between it and its six neighbors is exactly three.

    Notes
    -----
    Each layer n contains 6n tiles (except for layer 0, which has 1 tile),
    and the total number of tiles up to and including layer n is given by
    T(n) = 3n^2 + 3n + 2.

    Each regular tile n has the neighbors n - 1 and n + 1, with a difference of 1.
    Of the remaining four neighbors on different layers, it can be shown that two
    of them will be odd and two will be even (and thus cannot be prime).

    Therefore no tile n with both neighbors n + 1 and n - 1 prime can satisfy
    PD(n) = 3, and thus we only need to consider the first and last tile in
    each layer.
    """
    def get_neighbor_diffs(layer, position):
        layer_count = 6 * layer
        next_layer_count = layer_count + 6
        prev_layer_count = layer_count - 6
        up = position + position // layer # position in next layer
        down = position - position // layer # position in previous layer

        if position % layer == 0:
            # Corner tile
            return [
                layer_count - 1 if position == 0 else 1, # right
                position if position == layer_count - 1 else 1, # left
                prev_layer_count + position - down % prev_layer_count, # down
                layer_count + (up - 1) % (layer_count + 6) - position, # up left
                layer_count + up % (layer_count + 6) - position, # up
                layer_count + (up + 1) % (layer_count + 6) - position, # up right
            ]
        else:
            # Edge tile
            return [
                layer_count - 1 if position == 0 else 1, # right
                position if position == layer_count - 1 else 1, # left
                prev_layer_count + position - (down - 1) % prev_layer_count, # down left
                prev_layer_count + position - down % prev_layer_count, # down right
                layer_count + up % next_layer_count - position, # up left
                layer_count + (up + 1) % next_layer_count - position, # up right
            ]

    def PD(layer, position):
        if layer == 0: return 3
        if layer == 1: return (3, 2, 2, 0, 2, 2)[position]
        return sum(map(nt.is_prime, get_neighbor_diffs(layer, position)))

    sequence = (
        3*layer*layer - 3*layer + 1 + (layer > 0) + position
        for layer in itertools.count()
        for position in ((0, 6*layer - 1) if layer > 0 else (0,))
        if PD(layer, position) == 3
    )
    return nt.nth(sequence, n)

def problem_129(N=1_000_000):
    """
    Find the smallest integer n such that A(n) > N, where A(n)
    is the smallest k such that R(k) = (10^k - 1) / 9 = 0 mod n.

    Notes
    -----
    A(n) is the smallest k such that R(k) = 10^k - 1 / 9 = 0 mod n,
    or equivalently 10^k = 1 mod 9n. This is exactly ord(10, 9n).

    Given the multiplicative group of integers mod 9n, consider the
    subgroup G of its residues such that x = 1 mod 9. Since 10 = 1 mod 9,
    the cyclic group H generated by 10 is a subgroup of this subgroup,
    and A(n) is the order of 10 in this subgroup.

    Then A(n) = |H| <= |G| <= n.
    """
    for n in itertools.count(start=N):
        if gcd(n, 10) == 1 and nt.multiplicative_order(10, 9*n) > N:
            return n

def problem_130(m=25):
    """
    Find the sum of the first m composite values of n such that
    A(n) divides n - 1, where A(n) is defined the least k where
    R(k) = (10^k - 1) / 9 = 0 mod n.

    Notes
    -----
    A(n) is the smallest k such that R(k) = 10^k - 1 / 9 = 0 mod n,
    or equivalently 10^k = 1 mod 9n. This is exactly ord(10, 9n).
    If A(n) divides n - 1, then we must have 10^(n - 1) = 1 mod 9n.
    """
    composites = (
        n for n in itertools.count(start=7)
        if not nt.is_prime(n)
        if pow(10, n - 1, 9 * n) == 1
    )

    return sum(itertools.islice(composites, m))

def problem_131(N=1_000_000):
    """
    Find the number of primes p below N such that there exists
    an integer n > 0 where n^3 + n^2 * p is a perfect cube.

    Notes
    -----
    We can write n^3 + n^2 * p = n^2 (n + p) = m^3 for some prime p.
    The product of coprime integers is a perfect cube if and only if
    each integer is a perfect cube, so we consider gcd(n^2, n + p).

    Any prime dividing n^2 must also divide n, and any prime dividing
    both n and n + p must also divide their difference p. Thus, either
    p | n or gcd(n^2, n + p) = 1.

    If p | n, then let n = kp for some integer k. Then our expression
    becomes k^2 * p^3 * (k + 1) = m^3. Since p is prime, we have
    k^2 * (k + 1) = a^3 for integer a = m/p. As gcd(k^2, k + 1) = 1,
    this implies both k and k + 1 must be perfect cubes, which is impossible.

    Thus we have gcd(n^2, n + p) = 1, and therefor that both n^2 and n + p
    are perfect cubes.

    Let n = a^3 and n + p = b^3 for some integers a, b > 0. Then we have:
    p = b^3 - a^3 = (b - a)(b^2 + ab + a^2), and since p is prime, this
    implies that b - a = 1.

    Thus p is given by the difference of consecutive cubes.
    """
    primes = (p for i in itertools.count() if nt.is_prime(p := 3*i*i + 3*i + 1))
    return sum(1 for _ in itertools.takewhile(lambda p: p < N, primes))

def problem_132(n=9, m=40):
    """
    Find the sum of the first m primes p such that R(10^n) is divisible by p,
    where R(k) = (10^k - 1) / 9.

    Notes
    -----
    It is clear that primes p = 2, 5 can never divide any repunit R(k),
    and that p = 3 cannot divide repunits R(10^n) whose digit counts
    are powers of 10 and thus not multiples of 3.

    Let p >= 7 be a prime with gcd(p, 10) = 1. Then we can infer that
    R(k) = (10^k - 1) / 9 = 0 mod p if and only if 10^k = 1 mod p, which
    is true exactly when ord(10, p) divides k.
    """
    k = 10**n
    primes = (p for p in nt.primes(low=7) if k % nt.multiplicative_order(10, p) == 0)
    return sum(islice(primes, m))

def problem_133(N=100_000):
    """
    Find the sum of all primes p < N that never divide any repunit R(10^n),
    where R(k) = (10^k - 1) / 9.

    Notes
    -----
    It is clear that primes p = 2, 5 can never divide any repunit R(k),
    and that p = 3 cannot divide repunits R(10^n) whose digit counts
    are powers of 10 and thus not multiples of 3.

    Let p >= 7 be a prime with gcd(p, 10) = 1. Then we can infer that
    R(k) = (10^k - 1) / 9 = 0 mod p if and only if 10^k = 1 mod p.

    Then R(10^n) = 0 mod p if 10^(10^n) = 1 mod p, which is never true
    if ord(10, p) has any prime factor other than 2 or 5.
    """
    total = 2 + 3 + 5
    for p in nt.primes(low=7, high=N):
        k = nt.multiplicative_order(10, p)
        while k % 2 == 0: k //= 2
        while k % 5 == 0: k //= 5
        if k > 1: total += p

    return total

def problem_134(N=1_000_000):
    """
    Find the sum of the smallest positive integers n in which the last
    digits are formed by those of p and the number is divisible by q,
    for each consecutive prime pair (p, q) where 5 <= p <= N.

    Notes
    -----
    We are looking for n such that n = p (mod 10^k) and n = 0 (mod q),
    where k is the number of digits in p.

    We can write n = p + a * 10^k = 0 (mod q) for some integer a,
    or equivalently a * 10^k = -p = q - p (mod q).
    If we find the modular inverse b = (10^k)^(-1) (mod q),
    we can solve directly for a = b * (q - p) (mod q).
    """
    primes = nt.primes(low=5, high=nt.next_prime(N))
    return sum(
        p + a*t
        for p, q in itertools.pairwise(primes)
        for t in (10**(nt.digit_count(p)),)
        for a in (pow(t, -1, q) * (q - p) % q,)
    )

def problem_135(N=1_000_000, k=10):
    """
    Find how many values of 0 < n < N have exactly k solutions to the equation
    x^2 - y^2 - z^2 = n, where x, y, z > 0 form an arithmetic progression.

    Notes
    -----
    We can represent this as (y + c)^2 - y^2 - (y - c)^2 = y(4c - y) = n.
    For any pair of factors d * y = n, we need c = (d + y) / 4 and y > c > 0.

    We get positive integer solutions for each (y, c) pair such that
    y/4 < c < y and y(4c - y) < N.
    """
    num_solutions = [0] * N
    for y in range(1, N):
        for c in range((y + 3) // 4, y):
            n = y * (4*c - y)
            if n >= N: break
            num_solutions[n] += 1

    return num_solutions.count(k)

def problem_136(N=50_000_000):
    """
    Find how many values of 0 < n < N have exactly 1 solution to the equation
    x^2 - y^2 - z^2 = n, where x, y, z > 0 form an arithmetic progression.

    Notes
    -----
    We can represent this as (y + c)^2 - y^2 - (y - c)^2 = y(4c - y) = n.
    For any pair of factors d * y = n, we need c = (d + y) / 4 and y > c > 0.
    We can show that there are three cases that yield exactly one solution:

        1) n = p where p ≡ 3 (mod 4)
        2) n = 4p where p is an odd prime (or p = 1)
        3) n = 16p where p is an odd prime (or p = 1)

    We can also count them efficiently without iterating over all primes < N.
    Let π(x) be the prime counting function, and let χ_4(x) be the Dirichlet character
    modulo 4, which we know is a completely multiplicative function.
    Then for any N > 2 the number of primes p <= N with p ≡ 3 (mod 4) is given by:

        (π(N) - 1 - Σ_{p <= N} χ_4(p)) / 2

    And our total count n <= N is given by: π(N/4) + π(N/16) + (π(N) - π_4(N) - 1) / 2.
    """
    # Define prime counting functions
    chi4 = lambda x: 0 if x % 2 == 0 else 1 if x % 4 == 1 else -1
    Chi4 = lambda x: 1 if x % 4 in (1, 2) else 0
    count_primes_3_mod_4 = lambda x: (
        (nt.count_primes(x) - nt.sum_primes(x, chi4, Chi4) - (x > 2)) // 2
    )

    # Count primes
    count = (N > 4) + (N > 16) # account for n = 4 and n = 16 cases
    count += nt.count_primes((N - 1) // 4) - 1 # n = 4p, exclude p = 2
    count += nt.count_primes((N - 1) // 16) - 1 # n = 16p, exclude p = 2
    count += count_primes_3_mod_4(N - 1) # n = p, p ≡ 3 (mod 4)

    return count

def problem_137(i=15, j=15, a=0, b=1):
    """
    Find the sum of the i-th through j-th values of A(x) such that x is rational
    and A(x) = G(x) - G_0 is a positive integer, where G_i = G_{i-1} + G_{i-2}
    is the generalized Fibonacci sequence with G_0 = a and G_1 = b, and with
    G(x) = G_0 + G_1 x + G_2 x^2 + ... as its generating function.

    Notes
    -----
    We are looking for rational x such that A(x) = G(z) - G_0 is a positive integer.
    It can be shown that the generating function for the generalized Fibonacci
    sequence where G_0 = a and G_1 = b is given by:

        G(z) = (a + (b - a) z) / (1 - z - z^2)

    The quadratic formula tells us that G(z) - k = 0 has rational roots only when
    Δ = 5k^2 + 2k(b - 3a) + (b - a)^2 = Y^2 is a perfect square.

    Thus we need to find integer solutions to:

        5k^2 + 2k(b - 3a) + (b - a)^2 - Y^2 = 0

    or equivalently to 25k^2 + 2k(b - 3a) + 5(b - a)^2 - 5Y^2 = 0.

    Let X = 5k - 3a + b. Then this reduces to the Pell equation
    X^2 - 5Y^2 = 4a^2 + 4ab - 4b^2.
    """
    solutions = nt.pell(D=5, N=(4*a*a + 4*a*b - 4*b*b))
    nuggets = ((x + 3*a - b) // 5 - a for x, _ in solutions if (x + 3*a - b) % 5 == 0)
    return sum(islice(nuggets, i, j + 1)) # skip trivial solution

def problem_138(n=12):
    """
    Find the sum of L for the n smallest (L, L, b) isosceles triangles
    with height h = b ± 1, where b, L are positive integers.

    Notes
    -----
    We are looking for Pythagorean triples of the form (b/2, b ± 1, L).
    Expanding the Pythagorean formula, we have L^2 - (5/4)b^2 ± 2b - 1 = 0.

    Let x = (5/2)b ± 2. Then this reduces to the negative Pell equation
    x^2 - 5L^2 = -1.
    """
    generator = nt.pell(D=5, N=-1)
    solutions = [next(generator) for _ in range(n + 1)]
    return sum(L for _, L in solutions[1:]) # skip trivial solution

def problem_139(N=100_000_000):
    """
    Find the number of Pythagorean triples (a, b, c) such that
    a + b + c < N and (a - b) divides c.

    Notes
    -----
    Let (a, b, c) be a primitive Pythagorean triple.

    If (a - b) divides c, then (a - b)^2 = (c^2 - 2ab) divides c^2,
    and so (c^2 - 2ab) must must also divide 2ab. As a common divisor
    of 2ab and c^2, we must also have (c^2 - 2ab) divides gcd(2ab, c^2).

    Since (a, b, c) is a primitive triple, gcd(ab, c^2) = 1, and since c
    must be odd, gcd(2ab, c^2) = 1. It follows that we must have (a - b)^2 = 1,
    and thus (a - b) = ±1.

    By Euclid's formula we have a = m^2 - n^2, b = 2mn, c = m^2 + n^2 for m > n > 0.
    Then m, n must also satisfy (a - b) = (m^2 - n^2) - 2mn = (m - n)^2 - 2n^2 = ±1.

    Let x = m - n. Then this reduces to the Pell equations given by x^2 - 2n^2 = ±1.
    """
    count = 0
    for C in (-1, 1):
        solutions = nt.pell(D=2, N=C)
        while True:
            x, y = next(solutions)
            m, n = x + y, y
            a, b, c = m*m - n*n, 2*m*n, m*m + n*n
            num_triples = (N - 1) // (a + b + c)
            count += num_triples
            if num_triples == 0: break

    return count

def problem_140(i=1, j=30, a=3, b=1):
    """
    Find the sum of the i-th through j-th values of A(x) such that x is rational
    and A(x) = G(x) - G_0 is a positive integer, where G_i = G_{i-1} + G_{i-2}
    is the generalized Fibonacci sequence with G_0 = a and G_1 = b, and with
    G(x) = G_0 + G_1 x + G_2 x^2 + ... as its generating function.
    """
    return problem_137(i, j, a, b)

def problem_142(n=3):
    """
    Find the smallest set of n integers such that for any two elements x and y,
    x + y and x - y are both perfect squares.

    TODO: clean up
    """
    def make_graph(k):
        graph = defaultdict(set)
        squares = [i*i for i in range(1, k + 1)]
        for i, b in enumerate(squares):
            for a in squares[i+2::2]:
                x, y = (a + b) // 2, (a - b) // 2
                graph[x].add(y)
                graph[y].add(x)
        # for b, a in itertools.combinations(squares, 2):
        #     if (a - b) % 2 == 0:
        #         x, y = (a + b) // 2, (a - b) // 2
        #         graph[x].add(y)
        #         graph[y].add(x)

        #print("Graph", k, len(graph), sum(len(v) for v in graph.values()))
        return graph

    k = 1000
    best_clique = []
    while len(best_clique) < n:
        graph = make_graph(k)
        cliques = [clique for clique in nt.bron_kerbosch(graph) if len(clique) >= n]
        #print(k, cliques)
        if cliques:
            best_clique = min(cliques, key=lambda c: sum(sorted(c)[:n]))
            #print(best_clique)
            return sum(best_clique)
        else:
            k *= 2

def problem_145(N=10**9):
    """
    Find the number of integers n < N such that the sum [n + reverse(n)]
    consists entirely of odd (decimal) digits.

    TODO: optimize
    """
    ODD_DIGITS = set('13579')
    count = 0
    for i in range(N):
        #if i % 1000000 == 0: print(i)
        if i % 10 == 0:
            continue
        a = str(i + int(str(i)[::-1]))
        if set(a).issubset(ODD_DIGITS):
            count += 1

    return count

def problem_146(N=150_000_000):
    """
    Find the sum of all integers n < N such that n^2 + k forms a 
    run of consecutive primes for k in {1, 3, 7, 9, 13, 27}.

    Notes
    -----
    We can infer that n^2 = 0 mod 10, which implies that n = 0 mod 10.

    Given this and also that all primes p > 3 are of the form 6a ± 1,
    for the prime run to be consecutive we also need to check that
    n^2 + 19 and n^2 + 21 are composite.
    """
    offsets = (1, 3, 7, 9, 13, 27)
    residues = {}
    for p in nt.primes(high=100):
        if p in (2, 5): continue
        residues[p] = {
            i for i in range(p)
            if not any((i*i + k) % p == 0 for k in offsets)
        }

    def is_valid(n):
        for p in residues:
            if n % p not in residues[p]:
                return False

        for k in offsets:
            if not nt.is_prime(n*n + k):
                return False

        if nt.is_prime(n*n + 19) or nt.is_prime(n*n + 21):
            return False

        return True

    return sum(n for n in range(10, N, 10) if is_valid(n))

def problem_148(N=10**9, p=7):
    """
    Find the number of entries in the first N rows of Pascal's triangle
    that are not divisible by prime p.

    Notes
    -----
    By Lucas's Theorem, a binomial coefficient C(n, k) is divisible by prime p
    if and only if at least one of the base p digits of n is greater than the
    corresponding digit of k.

    Thus the number of binomial coefficients C(n, k) not divisible by p
    is given by prod_i (a_i + 1), where a_i are the digits of n in base p.
    """
    T = lambda n: n * (n + 1) // 2
    assert nt.is_prime(p)
    power = 1 # power of T(p) = (1 + 2 + ... + p)^i
    count = 0
    while N > 0:
        N, digit = divmod(N, p)
        count *= (digit + 1)
        count += T(digit) * power
        power *= T(p)

    return count

def problem_173(N=1_000_000):
    """
    TODO: writeup the problem and explain the trick

    Notes
    -----

    x^2 - y^2 = (x - y)(x + y) = ab = k has valid solutions when a and b
    have the same parity.
    """
    return sum(N // (4*i) - i for i in range(1, isqrt(N // 4) + 1))

def problem_174(T=1_000_000):
    """
    TODO: writeup

    Notes
    -----
    x^2 - y^2 = (x - y)(x + y) = ab = k has valid solutions when a and b
    have the same parity.
    """
    N = defaultdict(int)
    for t in range(0, T + 1, 2):
        n = 0
        factors = nt.divisors(t)
        n = sum((factors[i] - factors[-i-1]) % 2 == 0 for i in range(len(factors) // 2))
        N[n] += 1

    return sum(N[i] for i in range(1, 11))

def problem_179(N=10**7):
    """
    Find the number of integers 1 < n < N such that d(n) = d(n + 1),
    where d(n) is the number of divisors of n.
    """
    d = nt.divisor_count_range(N + 1)
    return sum(d[n] == d[n + 1] for n in range(2, N))

def problem_187(N=10**8):
    """
    Count the number of semiprimes below N.

    Notes
    -----
    We can write any semiprime as n = pq, where p <= √n <= q.
    Then the sum of semiprimes less than equal to n is given by:

        sum(π(n/p_i) - i + 1) from i = 1 ... √n

    where π is the prime counting function.
    """
    pi = nt.count_primes
    return sum(pi((N - 1) // p) - i for i, p in enumerate(nt.primes(high=isqrt(N - 1))))

def problem_188(a=1777, b=1855, n=8):
    """
    Find the last 8 digits of the tetration a ↑↑ b.

    TODO: generalize
    """
    mod = 10**n
    tetration = a
    for _ in range(b - 1):
        tetration = pow(a, tetration, mod=mod)

    return tetration

def problem_196(a=5678027, b=7208785):
    """
    TODO: write up problem description, optimize
    """
    T = cache(lambda n: n * (n + 1) // 2)

    primes_set = set()
    primes_set |= set(nt.primes(low=T(a - 3), high=T(a + 2)))
    primes_set |= set(nt.primes(low=T(b - 3), high=T(b + 2)))

    def get_neighbors(n):
        i = nt.polygonal_index(3, n - 1)
        neighbors = (
            *range(max(T(i - 1) + 1, n - i - 1), min(T(i) + 1, n - i + 2)),
            *(x for x in (n - 1, n + 1) if T(i) < x <= T(i + 1)),
            *range(max(T(i + 1) + 1, n + i), min(T(i + 2) + 1, n + i + 3)),
        )
        return tuple(filter(primes_set.__contains__, neighbors))

    def is_part_of_prime_triplet(n):
        neighbors = get_neighbors(n)
        if len(neighbors) >= 2:
            return True
        else:
            for m in neighbors:
                if len(get_neighbors(m)) >= 2:
                    return True

            return False

    def S(n):
        total = 0
        for p in nt.primes(low=T(n - 1) + 1, high=T(n)):
            if is_part_of_prime_triplet(p):
                total += p
    
        return total

    return S(a) + S(b)

def problem_203(n=51):
    """
    Return the sum of all distinct square-free integers
    in the first n rows of Pascal's triangle.

    Notes
    -----
    Consider the binomial coefficient (n choose k).

    For any p such that p^2 > n, it is clear that the base-p representation of n
    has at most 2 digits, and thus by Kummer's theorem, the p-adic valuation of
    (n choose k) must be 0 or 1, since a second carry in the sum n + (n - k)
    would produce a p^2 term.

    It follows that p^2 cannot divide (n choose k) for any p > √n, which implies
    we need only test divisibility by squares of primes p <= √n.
    """
    moduli = [p*p for p in nt.primes(high=isqrt(n))]
    return sum({a for _, a in nt.pascal(n) if all(a % m != 0 for m in moduli)})

def problem_206():
    """
    Find unique positive integer whose square has the form 1_2_3_4_5_6_7_8_9_0,
    where each '_' is a single digit.

    TODO: clean up
    """
    low, high = 1020304050607080900, 1929394959697989900
    low, high = ((isqrt(low) + 10 - 1) // 10) * 10, isqrt(high)
    for n in range(low, high + 1, 10):
        digits = str(n*n)
        if digits[::2] == '1234567890':
            return n

def problem_211(N=64_000_000):
    """
    Find the sum of all integers 1 < n < N such that
    the sum of squares of divisors of n is itself a perfect square.
    """
    divisor_square_sums = nt.divisor_function_range(N, k=2)
    return sum(n for n in range(1, N) if nt.is_square(divisor_square_sums[n]))

def problem_214(N=40_000_000, k=25):
    """
    For any positive integer n, consider the sequence n, φ(n), φ(φ(n)), ..., 1
    where φ is Euler's totient function. Find the sum of all primes below N
    that produce a chain of length k.

    Notes
    -----
    Let f(n) be the function that is n if n is prime and has a chain length of k,
    and 0 otherwise.
    """
    phi = nt.totient_range(N)
    chain_length = cache(lambda n: 1 if n == 1 else 1 + chain_length(phi[n]))
    return sum(p for p in nt.primes(high=N-1) if chain_length(p) == k)

def problem_216(N=50_000_000):
    """
    Count the number of integers 1 < n < N such that 2n^2 - 1 is prime.

    Notes
    -----
    2n^2 - 1 = p
    p + 1 = 2n^2

    TODO: reimplement with a sieve + Shanks-Tonelli
    """
    return sum(nt.is_prime(2*n*n - 1) for n in range(2, N + 1))

def problem_218(N=10**16):
    """
    Find the number of primitive Pythagorean triples (a, b, c) such that
    c <= N is a perfect square and the area ab/2 is not divisible by 6 or 28.

    Notes
    -----
    Let (a, b, c) be a primitive Pythagorean triple:
    
        a = x^2 - y^2, b = 2xy, c = x^2 + y^2

    If c is square, then (x, y, sqrt(c)) is also a primitive Pythagorean triple:

        x = m^2 - n^2, y = 2mn, c = m^2 + n^2

    Then the area of the triangle is:

        A = ab / 2 = 2(m^7)(n) - 14(m^5)(n^3) + 14(m^3)(n^5) - 2(m)(n^7).

    By Fermat's little theorem:

        A mod 7 = (2mn - 2mn) mod 7 = 0
        A mod 3 = (2mn - 14mn + 14mn - 2mn) mod 3 = 0

    And since one of m, n is even, A mod 4 = 0.

    It follows that A mod 6 = 0 and A mod 28 = 0 for every such triple.
    """
    return 0

def problem_231(n=20_000_000, k=15_000_000):
    """
    Find the sum of all prime factors (not distinct) of the
    binomial coefficient (n choose k).

    Notes
    -----
    By Kummer's theorem, the p-adic valuation of the binomial coefficient
    (n choose k) is (S_p(k) + S_p(n - k) - S_p(n)) / (p - 1), where S_p(a)
    is the sum of the base-p digits of integer a.
    """
    return sum(p * nt.binomial_valuation(n, k, p) for p in nt.primes(high=n))
    S = lambda a, p: sum(nt.digits_in_base(a, p))
    return sum(
        p * (S(k, p) + S(n - k, p) - S(n, p)) // (p - 1)
        for p in nt.primes(high=n)
    )

def problem_243(threshold=15499/94744):
    """
    Find the smallest integer d such that R(d) = φ(d) / (d - 1) is less than
    the given threshold, where φ is Euler's totient function.

    Notes
    -----
    The function R(d) reaches a new minimimum exactly when d is a multiple of the
    largest primorial less than or equal to d.
    """
    # Find the largest primorial n such that R(n) >= threshold
    primorial, p = 1, 2
    while nt.totient(primorial * p) / (primorial * p - 1) >= threshold:
        primorial *= p
        p = nt.next_prime(p)

    # Increment multiples of the primorial until R(d) < threshold
    for n in itertools.count(start=primorial, step=primorial):
        if nt.totient(n) / (n - 1) < threshold:
            return n

def problem_268(N=10**16, M=100, k=4):
    """
    Find how many positive integers less than N are divisible by at least
    k distinct primes less than M.

    Notes
    -----
    By the inclusion-exclusion principle, this is the number divisible by k primes,
    minus the number divisible by k + 1 primes, plus the number divisible by
    k + 2 primes, etc.

    Every number that is divisible by k + a primes is divisible by 
    ((k + a) choose k) products of k primes.
    """
    prime_list = list(nt.primes(high=M-1))
    count = 0
    for i in range(k, len(prime_list) + 1):
        s = sum(N // prod(factors) for factors in combinations(prime_list, i))
        count += (-1)**(i - k) * comb(i - 1, k - 1) * s

    return count

def problem_271(n=13082761331670030):
    """
    Find the sum of all integers x such that 1 < x < n and x^3 = 1 (mod n).
    """
    a = (-1, 0, 0, 1) # coefficients to f(x) = -1 + x^3
    pf = nt.prime_factorization(n).items()
    prime_powers = [p**e for p, e in pf]
    return sum(
        nt.crt(residues, prime_powers)[0]
        for residues in itertools.product(*(nt.hensel(a, p, k) for p, k in pf))
    ) - 1 # ignore trivial solution x = 1

def problem_277(N=10**15, sequence='UDDDUdddDDUDDddDdDddDDUDDdUUDd'):
    """
    Find the smallest positive integer n > N such that the sequence of operations
    "U" (multiply by 3), "D" (divide by 2), and "d" (decrement by 1) on n
    produces the given sequence.

    Notes
    -----
    Any two integers whose sequences begin with the same k elements
    must be equivalent mod 3^k.

    We can reverse-engineer the sequence by starting with the final value and
    working backwards, obtaining a rational number a/b.

    Then the smallest integer that produces our sequence is n = a * b^(-1) mod 3^k,
    where b^(-1) is the modular inverse of b mod 3^k.
    """
    a, b = 1, 1
    for step in reversed(sequence):
        if step == 'D':
            a *= 3
        elif step == 'U':
            a, b = 3*a - 2*b, 4*b
        elif step == 'd':
            a, b = 3*a + b, 2*b

    mod = 3**len(sequence)
    n = a * pow(b, -1, mod=mod)
    return N + (n - N) % mod

def problem_293(N=10**9):
    """
    TODO: writeup problem description

    Notes
    -----
    Every admissible number can be uniquely represented as a sequence of
    non-decreasing exponents on consecutive primes.

    We can generate them in increasing order with a min-heap starting with 2,
    and repeatedly multiplying either by the next prime or by the largest prime
    factor of the current number.
    """
    pseudo_fortunate_numbers = set()
    primes = list(nt.primes(high=nt.ilog(N)))

    # Create a min-heap of admissible numbers
    # Each element of the form (n, (e1, e2, ...)) where n = p1^e1 * p2^e2 * ...
    queue = [(2, (1,))] 
    while queue[0][0] < N:
        # Get admissible and find corresponding pseudo-fortunate number
        n, exponents = heappop(queue)
        m = nt.next_prime(n + 1) - n
        pseudo_fortunate_numbers.add(m)

        # Add new admissible numbers to the queue
        i = len(exponents)
        heappush(queue, (n * primes[i], (*exponents, 1)))
        heappush(queue, (n * primes[i - 1], (*exponents[:-1], exponents[-1] + 1)))

    return sum(pseudo_fortunate_numbers)

def problem_304(N=100000, k=10**14, m=1234567891011):
    """
    Find the sum of F(p) for the first N primes greater than k,
    where F(i) is the i-th Fibonacci number.
    """
    # Get the first N primes greater than k
    prime_list = list(nt.primes(low=k, num=N))

    # Calculate Fibonacci numbers mod m
    p_min, p_max = prime_list[0], prime_list[-1]
    prime_list = set(prime_list)
    fib = nt.fibonacci_numbers(
        nt.fibonacci(p_min, m), nt.fibonacci(p_min + 1, m), mod=m)
    fib = zip(range(p_min, p_max + 1), fib)

    return sum(F for i, F in fib if i in prime_list) % m

def problem_357(N=10**8):
    """
    TODO: write problem description

    Notes
    -----
    Excluding the trivial n = 1, we notice ...

        * n must be even
        * n must be square-free
        * we must have n = p - 1 for some prime p such that p % 4 = 3
        * we must have n = 2q - 4 for some prime q
    """
    total = 1 # account for n = 1
    for p in nt.primes(low=3, high=((N + 4) // 2)):
        n = 2*p - 4

        # Initial filters
        if not nt.is_prime(n + 1): continue # n + 1 must be prime
        factors = nt.divisors(n)
        if len(factors) % 2 == 1: continue # must be square-free

        # Check that d + n/d is prime for all divisors d of n
        for d in factors:
            if not nt.is_prime(d + n // d):
                break
        else:
            total += n

    return total

def problem_407(N=10**7):
    """
    TODO: write problem description and optimize
    """
    def M(n):
        prime_powers = [p**e for p, e in nt.prime_factorization(n)]
        roots = itertools.product((0, 1), repeat=len(prime_powers))
        return max(nt.crt(a, prime_powers)[0] for a in roots)

    total = 0
    for n in range(2, N + 1):
        #print(n)
        total += M(n)

    return total

def problem_451(N=20_000_000):
    """
    Let I(n) be the largest integer x < n - 1 such that x^2 = 1 mod n.
    Find the sum of I(3) + I(4) + ... + I(N).

    TODO: clean up

    Notes
    -----
    We are looking for non-trivial solutions to x^2 - 1 = (x - 1)(x + 1) = 0 mod n.
    One way to do this is to iterate over divisors.

    If n has a primitive root, then the only solutions to x^2 = 1 mod n
    are x = ±1, and thus I(n) = 1.

    Otherwise, we can decompose n as a product of prime powers,
    and use the Chinese remainder theorem to find solutions to x = ±1 mod p^e.
    """
    def I2(n):
        return next((x for x in range(n - 2, 0, -1) if x*x % n == 1))

    def has_primitive_root(prime_factorization):
        if len(prime_factorization) == 1:
            return 2 not in prime_factorization or prime_factorization[2] <= 2
        elif len(prime_factorization) == 2:
            return 2 in prime_factorization and prime_factorization[2] == 1

        return False

    def get_prime_factorizations(N: int) -> list[dict[int, int]]:
        """
        Get the prime factorization for each n = 0, 1, 2, ..., N.

        Parameters
        ----------
        N : int
            Upper bound for n
        """
        factorizations = [defaultdict(int) for _ in range(N + 1)]
        for p in nt.primes(high=N):
            for e in range(1, nt.ilog(N, p) + 1):
                prime_power = p**e
                for factorization in factorizations[prime_power::prime_power]:
                    factorization[p] += 1

        return factorizations

    prime_factorizations = get_prime_factorizations(N)

    roots = [[()] for _ in range(N + 1)]
    for p in nt.primes(high=N):
        for e in range(1, int(log(N, p)) + 1):
            power = p**e
            p_roots = (1, -1)
            for i in range(power, N + 1, power):
                roots[i] = [
                    (*x, root)
                    for x in roots[i]
                    for root in p_roots
                ]

    print(list(roots[24]))

    def I(n):
        prime_factorization = dict(prime_factorizations[n])
        if has_primitive_root(prime_factorization):
            return 1

        if n % 8 == 0:
            prime_factorization[2] -= 1

        prime_powers = [p**e for p, e in prime_factorization.items()]
        solutions = [
            nt.crt(list(zip(tuple(a), prime_powers)))[0]
            for a in itertools.product((-1, 1), repeat=len(prime_powers))
        ]

        x = sorted(set(solutions))[:-1][-1]
        return x + n // 2 if n % 8 == 0 else x

    total = 0
    for n in range(3, N + 1):
        # print(n, I(n))
        # assert I(n) == I2(n), (n, I(n), I2(n))
        if n % 10000 == 0: print(n)
        total += I(n)

    return total
    return sum(I(n) for n in range(3, N + 1))

def problem_500(n=500500, m=500500507):
    """
    Find the smallest number (mod m) with exactly 2^500500 divisors.

    Notes
    -----
    The smallest number with at least 2^n divisors is the product of the
    first n Fermi-Dirac primes.
    """
    prime_generator = nt.primes()
    heap = [next(prime_generator)]
    next_prime = next(prime_generator)
    count, product = 0, 1

    while count < n:
        p = heappop(heap)
        product = (product * p) % m
        count += 1

        heappush(heap, p*p)
        while next_prime < heap[0]:
            heappush(heap, next_prime)
            next_prime = next(prime_generator)

    return product

def problem_501(N=10**12):
    """
    Find the number of integers n <= N such that n has exactly 8 divisors.

    Notes
    -----
    Either n = pqr, n = p^3 * q, or n = p^7, where p, q, r are prime.
    """
    import bisect

    # Define prime counting function
    primes = list(nt.primes(high=10**9))
    pi = lambda x: bisect.bisect(primes, x) if x < primes[-1] else nt.count_primes(x)
    pi = cache(pi)

    # Count primes p such that p^7 <= N
    count = pi(nt.iroot(N, 7))

    # Count primes p, q such that p != q and p^3 * q <= N
    for p in nt.primes(high=nt.iroot(N//2, 3)):
        n = N // (p*p*p)
        count += pi(n) - (p <= n)

    # Count primes p, q, r such that p < q < r and pqr <= N
    for p in nt.primes(high=nt.iroot(N, 3)):
        for q in nt.primes(low=p+1, high=isqrt(N//p)-1):
            n = N // (p*q)
            print(p, q, n)
            count += pi(n) - pi(q)

    return count

def problem_601(T=31):
    """
    Let streak(n) = k be the smallest integer k such that n + k is not
    divisible by k + 1, and let P(s, N) be the number of integers 1 < n < N
    such that streak(n) = s. Find the sum of P(i, 4^i) for 1 <= i <= T.

    Notes
    -----
    If streak(n) = k, then we have n = 1 mod 2, n = 1 mod 3, ..., n = 1 mod k,
    AND n != 1 mod (k + 1).

    We can reduce the first (k - 1) congruences via the Chinese remainder theorem
    to n = a1 mod m1. Clearly, 1 is a solution, so a1 = 1 and m1 = lcm(2, 3, ..., k).

    If we also include the equality of the final congruence n = 1 mod (k + 1),
    we can reduce the full set of congruences to n = a2 mod m2, where a2 = 1 and
    m2 = lcm(2, 3, ..., k + 1).

    Now let C(m, N) = ⌊(N - 2) / m⌋ be the count of integers in the range 1 < n < N
    such that n = 1 mod m.

    By the inclusion-exclusion principle, we have:
    
        P(s, N) = ⌊(N - 2) / lcm(2, 3, ... s)⌋ - ⌊(N - 2) / lcm(2, 3, ..., s + 1)⌋
    """
    return sum(
        (N - 2) // lcm(*range(2, i + 1)) - (N - 2) // lcm(*range(2, i + 2))
        for i in range(1, T + 1)
        for N in (4**i,)
    )

def problem_668(N=10**10):
    """
    Find the number of "smooth" integers n <= N, numbers where
    all of its prime factors are strictly less than its square root.

    Notes
    -----
    We can instead count the number of "non-smooth" integers n <= N,
    each identified by its largest prime factor.

    For each prime p <= sqrt(N), we have p non-smooth numbers: p, 2p ... p^2.

    For each prime sqrt(N) < p <= N, we have ⌊N/p⌋ non-smooth numbers: p, 2p ... p⌊N/p⌋,
    where 1 <= ⌊N/p⌋ < sqrt(N).

    We know that n = ⌊N/p⌋ if and only if N/(n + 1) < p <= N/n.
    It follows that for each n, we can count the number of primes in this range
    as π(N/n) - π(N/(n + 1)), each of which contributes n non-smooth numbers.
    """
    import bisect

    # Define prime counting function
    primes = list(nt.primes(high=nt.iroot(N**4, 5)))
    pi = lambda x: bisect.bisect(primes, x) if x < primes[-1] else nt.count_primes(x)
    pi = cache(pi)

    # Direct sum for primes less than sqrt(N)
    count = sum(nt.primes(high=isqrt(N)))

    # Use prime counting for primes greater than sqrt(N)
    for n in range(1, isqrt(N)):
        count += n * (pi(N // n) - pi(N // (n + 1)))

    return N - count

def problem_719(N=10**12):
    def canmake(total, digs):
        if digs < total:
            return False
        elif digs == total:
            return True
        else:
            t = 10
            while t < digs:
                cutoff = digs // t
                rest = digs % t
                if rest < total:
                    if canmake(total - rest, cutoff):
                        return True
                t *= 10
        return False

    total = 0
    for n in (range(0, isqrt(N) + 1, 9)):
        for root in (n, n + 1):
            square = root*root
            if root%1000 == 0:
                print(root)
            if canmake(root, square):
                total += square
    print(total-1)

def problem_800(base=800800, exp=800800):
    """
    Find the number of integers p^q * q^p < N, where p and q are prime
    and N = base^exp.

    Notes
    -----
    We note that the above inequality implies qlog(p) + plog(q) < log(N).
    Given a fixed prime p, let f(x) = xlog(p) + plog(x). Since f(x) is
    a monotonically increasing function for x > 1, we can use binary search
    to find the largest integer x such that f(x) < log(N).

    Then the number of primes q > p such that f(x) < log(N) is given by
    π(x) - π(p), where π is the prime counting function.

    Without loss of generality, assume p < q. Then 2^q < p^q * q^p < N,
    which implies upper bound p < q < log_2(N).
    """
    import bisect

    # Compute bounds
    log_N, p_max = exp * log(base), int(exp * log(base, 2))

    # Generate primes
    primes = list(nt.primes(high=p_max))
    pi = lambda x: bisect.bisect(primes, x)

    # Iterate over primes p
    count = 0
    for i, p in enumerate(primes):
        # Find the largest integer x such that xlog(p) + plog(x) < log(N)
        f = lambda x: x*log(p) + p*log(x)
        x = nt.binary_search(f, threshold=log_N, low=p+1, high=p_max) - 1

        # Find the number of primes q > p such that qlog(p) + plog(q) < log(N)
        num_primes = pi(x) - (i + 1)
        if num_primes == 0: break
        count += num_primes

    return count

def problem_836():
    """
    April Fools!
    """
    return 'aprilfoolsjoke'



if __name__ == '__main__':
    import argparse
    import cProfile

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('config', nargs='*', default=[], type=int)
    args = parser.parse_args()

    problem = globals()[f'problem_{args.problem}']

    def compute_answer():
        for _ in range(args.num):
            ans = problem(*args.config)
        print(ans)

    if args.profile:
        cProfile.run('compute_answer()', sort='cumtime')
    else:
        compute_answer()



# import timeit
# for i in range(1, 900):
#     try:
#         problem = eval(f'problem_{i}')
#         t = timeit.timeit('problem()', number=1, globals=globals())
#         print(i, f'{t:.6f}', 'seconds')
#     except NameError:
#         pass



"""

#################
### TODO List ###
#################

* LLL lattice
* Refactor divisibility sieves to a segmented approach to save memory
* Write a bitmap "set with limited domain" interface
* Solve problem 185 (CSP)
* Write binary quadratic form solver
* Write general quadratic diophantine solver
* Complexity analysis for each of the algorithms
* Dirichlet characters
* Bareiss algorithm for linear_solve
* Generalize fibonacci to Lucas sequences
    The mathematically natural object is lucas(n, P, Q) returning (U_n, V_n).
    Fibonacci is lucas(n, 1, -1)[0].
"""
