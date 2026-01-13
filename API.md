<div align="center">

# numthy API Reference

**A number theory library for Python**

`v0.0.0`

</div>

---

## Contents

**[General](#general)**
<br>
[`Number`](#number) · [`Vector`](#vector) · [`Matrix`](#matrix) · [`clear_cache`](#clear_cache)

**[Primes](#primes)**
<br>
[`is_prime`](#is_prime) · [`next_prime`](#next_prime) · [`random_prime`](#random_prime) · [`primes`](#primes) · [`count_primes`](#count_primes) · [`sum_primes`](#sum_primes)

**[Factorization](#factorization)**
<br>
[`prime_factors`](#prime_factors) · [`prime_factorization`](#prime_factorization) · [`divisors`](#divisors)

**[Arithmetic Functions](#arithmetic-functions)**
<br>
[`mobius`](#mobius) · [`mobius_range`](#mobius_range) · [`radical`](#radical) · [`radical_range`](#radical_range) · [`divisor_function`](#divisor_function) · [`divisor_count_range`](#divisor_count_range) · [`divisor_function_range`](#divisor_function_range) · [`aliquot_sum_range`](#aliquot_sum_range) · [`totient`](#totient) · [`totient_range`](#totient_range) · [`carmichael`](#carmichael)

**[Modular Arithmetic](#modular-arithmetic)**
<br>
[`egcd`](#egcd) · [`crt`](#crt) · [`multiplicative_order`](#multiplicative_order) · [`primitive_root`](#primitive_root) · [`legendre`](#legendre) · [`jacobi`](#jacobi) · [`kronecker`](#kronecker) · [`dirichlet_character`](#dirichlet_character)

**[Nonlinear Congruences](#nonlinear-congruences)**
<br>
[`hensel`](#hensel) · [`discrete_log`](#discrete_log) · [`modular_roots`](#modular_roots)

**[Diophantine Equations](#diophantine-equations)**
<br>
[`bezout`](#bezout) · [`cornacchia`](#cornacchia) · [`pell`](#pell) · [`binary_quadratic_solve`](#binary_quadratic_solve) · [`pythagorean_triples`](#pythagorean_triples) · [`pillai`](#pillai)

**[Lattice Methods](#lattice-methods)**
<br>
[`integer_solve`](#integer_solve) · [`integer_nullspace`](#integer_nullspace) · [`lll_reduce`](#lll_reduce) · [`babai_closest_vector`](#babai_closest_vector)

**[Combinatorics](#combinatorics)**
<br>
[`pascal`](#pascal) · [`factorial_valuation`](#factorial_valuation) · [`binomial_valuation`](#binomial_valuation) · [`partition_numbers`](#partition_numbers) · [`count_partitions`](#count_partitions) · [`euler_transform`](#euler_transform)

**[Integer Sequences](#integer-sequences)**
<br>
[`integers`](#integers) · [`integer_pairs`](#integer_pairs) · [`lucas`](#lucas) · [`fibonacci`](#fibonacci) · [`fibonacci_index`](#fibonacci_index) · [`fibonacci_numbers`](#fibonacci_numbers) · [`polygonal`](#polygonal) · [`polygonal_index`](#polygonal_index) · [`polygonal_numbers`](#polygonal_numbers) · [`is_polygonal`](#is_polygonal)

**[Appendix](#appendix)**
<br>
[`nth`](#nth) · [`alternating`](#alternating) · [`periodic_continued_fraction`](#periodic_continued_fraction) · [`convergents`](#convergents) · [`permutation`](#permutation) · [`polynomial`](#polynomial) · [`iroot`](#iroot) · [`ilog`](#ilog) · [`is_square`](#is_square) · [`non_squares`](#non_squares) · [`squares`](#squares) · [`perfect_power`](#perfect_power) · [`binary_search`](#binary_search)


---

<br>

# General

### Number

```python
Number = int | float | complex | Fraction
```

Type alias for numeric types supported by this library.

---

### Vector

```python
Vector = list[Number]
```

Type alias for a list of numbers.

---

### Matrix

```python
Matrix = list[list[Number]]
```

Type alias for a 2D list of numbers.

---

### clear_cache

```python
clear_cache()
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L75)</sup>

> Clear all caches in this module.

---

<br>

# Primes

### is_prime

```python
is_prime(n: int) -> bool
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L89)</sup>

> Test if a given integer n is prime.

Uses a combination of trial division, the Miller-Rabin primality test
with deterministic bases, or the extra-strong variant of the Baillie-PSW
primality test (this variant has no known pseudoprimes in any range, and
has been computationally verified to have no counterexamples for all n < 2^64).

See: https://miller-rabin.appspot.com/ (deterministic Miller-Rabin base sets)
See: https://ntheory.org/pseudoprimes.html (BPSW verification up to 2^64)

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer to test for primality |

---

### next_prime

```python
next_prime(n: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L133)</sup>

> Get the smallest prime number greater than n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Strict lower bound for prime number |

---

### random_prime

```python
random_prime(num_bits: int, *, safe: bool=False) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L151)</sup>

> Generate a random prime with the given number of bits.

Uses 64 rounds of probabilistic Miller-Rabin to test primality.

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_bits` | `int` | Number of bits in the prime to be generated |
| `safe` | `bool` | Whether or not to generate a safe prime (i.e. prime q of the form q = 2p + 1, where p is also prime) |

---

### primes

```python
primes(*, low: int=2, high: int | None=None, count: int | None=None) -> Iterator[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L186)</sup>

> Generate at most `count` primes in increasing order within the range `[low, high]`.

Uses the sieve of Eratosthenes, with a segmented approach for large or
unbounded ranges.

| Parameter | Type | Description |
|-----------|------|-------------|
| `low` | `int` | Lower bound for prime numbers |
| `high` | `int` | Upper bound for prime numbers (default is infinite) |
| `count` | `int` | Maximum number of primes to generate (default is infinite) |

---

### count_primes

```python
count_primes(x: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L258)</sup>

> Prime counting function π(x). Returns the number of primes p <= x.

Uses the Lagarias-Miller-Odlyzko (LMO) extension of the Meissel-Lehmer algorithm.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `int` | Upper bound for prime numbers |

---

### sum_primes

```python
sum_primes(x: int, f: Callable[[int], Number] | None=None, f_prefix_sum: Callable[[int], Number] | None=None) -> Number
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L278)</sup>

> Compute F(x) as the sum of f(p) over all primes p <= x,
where f is a completely multiplicative function (by default, f(n) = n).

Uses a generalized version of the LMO prime counting algorithm.
Ideally `f()` and `f_prefix_sum()` can be calculated efficiently
in O(1) time via closed-form expression.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `int` | Upper bound for prime numbers |
| `f` | `Callable(int) -> int` | Completely multiplicative function f(n), where f(1) = 1 and f(ab) = f(a) * f(b) for all a, b > 0 |
| `f_prefix_sum` | `Callable(int) -> Number` | Function to compute the cumulative sum Σ_{1 <= k <= n} f(k) |

---

<br>

# Factorization

### prime_factors

```python
prime_factors(n: int) -> tuple[int, ...]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L899)</sup>

> Get all prime factors of n in sorted order (with multiplicity).

Uses a combination of trial division, Brent's variant of Pollard's rho
factorization method, Lenstra's elliptic curve method (ECM),
and a self-initializing quadratic sieve (SIQS).

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer to factor |

---

### prime_factorization

```python
prime_factorization(n: int) -> dict[int, int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L914)</sup>

> Get the prime factorization of n as a dictionary of {prime: exponent}.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer to factor |

---

### divisors

```python
divisors(n: int) -> tuple[int, ...]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L925)</sup>

> Get all positive divisors of n in sorted order (including both 1 and n).

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer to factor |

---

<br>

# Arithmetic Functions

### mobius

```python
mobius(n: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L1876)</sup>

> Compute the Mobius function μ(n) for a positive integer n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### mobius_range

```python
mobius_range(N: int) -> list[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L1917)</sup>

> Find the value of the Mobius function μ(n) for each n = 0, 1, 2, ..., N - 1.
Includes dummy value μ(0) = 1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `N` | `int` | Upper bound on range (exclusive) |

---

### radical

```python
radical(n: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L1939)</sup>

> Compute rad(n) as the product of the distinct prime factors of n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### radical_range

```python
radical_range(N: int) -> list[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L1952)</sup>

> Find the value of the radical function rad(n) for each n = 0, 1, 2, ..., N - 1,
where rad(n) is the product of the distinct prime factors of n.
Includes dummy value rad(0) = 1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `N` | `int` | Upper bound on range (exclusive) |

---

### divisor_function

```python
divisor_function(n: int, k: int=1) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L1975)</sup>

> Compute the value of the divisor function σ_k(n),
where σ_k(n) = ∑_{d|n} d^k.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |
| `k` | `int` | Divisor exponent |

---

### divisor_count_range

```python
divisor_count_range(N: int) -> list[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L1998)</sup>

> Find the number of divisors d(n) for each n = 0, 1, 2, ..., N - 1.
Includes dummy value d(0) = 1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `N` | `int` | Upper bound on range (exclusive) |

---

### divisor_function_range

```python
divisor_function_range(N: int, k: int=1) -> list[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2024)</sup>

> Find the values of the divisor function σ_k(n) for each n = 0, 1, 2, ..., N - 1,
where σ_k(n) = ∑_{d|n} d^k. Includes dummy value σ_k(0) = 0.

| Parameter | Type | Description |
|-----------|------|-------------|
| `N` | `int` | Upper bound on range (exclusive) |

---

### aliquot_sum_range

```python
aliquot_sum_range(N: int) -> list[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2061)</sup>

> Find the value of the aliquot sum s(n) for each n = 0, 1, 2, ..., N - 1,
where s(n) = σ(n) - n is the sum of proper divisors of n.
Includes dummy value s(0) = 0.

| Parameter | Type | Description |
|-----------|------|-------------|
| `N` | `int` | Upper bound on range (exclusive) |

---

### totient

```python
totient(n: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2075)</sup>

> Compute Euler's totient function φ(n) for a positive integer n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### totient_range

```python
totient_range(N: int) -> list[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2093)</sup>

> Find the value of Euler's totient function φ(n) for each n = 0, 1, 2, ..., N - 1.
Includes dummy value φ(0) = 1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `N` | `int` | Upper bound on range (exclusive) |

---

### carmichael

```python
carmichael(n: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2117)</sup>

> Compute Carmichael's lambda function λ(n) for a positive integer n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

<br>

# Modular Arithmetic

### egcd

```python
egcd(a: int, b: int) -> tuple[int, int, int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2169)</sup>

> Extended Euclidean algorithm.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | First integer |
| `b` | `int` | Second integer |

**Returns**

| Name | Type | Description |
|------|------|-------------|
| `d` | `int` | Greatest common divisor of a and b |
| `x` | `int` | Coefficient of a in Bézout's identity (ax + by = d) |
| `y` | `int` | Coefficient of b in Bézout's identity (ax + by = d) |

**Complexity:** `O(log min(a, b)) time`

---

### crt

```python
crt(residues: Iterable[int], moduli: Iterable[int]) -> int | None
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2207)</sup>

> Solve a system of linear congruences x ≡ a_i (mod n_i)
via the Chinese Remainder Theorem.

| Parameter | Type | Description |
|-----------|------|-------------|
| `residues` | `Iterable[int]` | Sequence of residues |
| `moduli` | `Iterable[int]` | Sequence of moduli |

**Returns**

| Name | Type | Description |
|------|------|-------------|
| `x` | `int | None` | Solution to the system of congruences, mod the LCM of the moduli, or None if no solution exists |

---

### multiplicative_order

```python
multiplicative_order(a: int, mod: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2230)</sup>

> Return the smallest integer k = ord_n(a) such that a^k ≡ 1 (mod n).

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | Integer base |
| `mod` | `int` | Integer modulus |

---

### primitive_root

```python
primitive_root(n: int) -> int | None
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2256)</sup>

> Find a primitive root modulo n.

Use Bach's primitive root finding algorithm to search for candidates.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer modulus |

---

### legendre

```python
legendre(a: int, p: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2295)</sup>

> Compute the Legendre symbol (a | p), where p is an odd prime.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | Numerator (i.e. quadratic residue class) |
| `p` | `int` | Denominator (i.e. prime modulus) |

---

### jacobi

```python
jacobi(a: int, n: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2315)</sup>

> Compute the Jacobi symbol (a | n), where n is an odd positive integer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | Numerator (i.e. quadratic residue class) |
| `n` | `int` | Denominator (i.e. modulus) |

**Complexity:** `O(log a log n) time`

---

### kronecker

```python
kronecker(a: int, n: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2349)</sup>

> Compute the Kronecker symbol (a | n).

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | Numerator (i.e. quadratic residue class) |
| `n` | `int` | Denominator (i.e. modulus) |

---

### dirichlet_character

```python
dirichlet_character(m: int, k: int) -> Callable[[int], Number]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2383)</sup>

> Return the Dirichlet character χₘ‚ₖ : ℤ → ℂ under Conrey labeling,
where m is the modulus and k is an index such that gcd(m, k) = 1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `m` | `int` | Modulus of the character |
| `k` | `int` | Index of the character |

**Returns**

| Name | Type | Description |
|------|------|-------------|
| `chi` | `Callable(int) -> Number` | Dirichlet character χₘ‚ₖ(n) as a callable function returning the character value at n |
| `See` | `https://www.lmfdb.org/knowledge/show/character.dirichlet.conrey` |  |

---

<br>

# Nonlinear Congruences

### hensel

```python
hensel(coefficients: Sequence[int], p: int, k: int, initial: Iterable[int] | None=None) -> tuple[int, ...]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2588)</sup>

> Find all solutions to the polynomial congruence f(x) ≡ 0 (mod p^k).

Assumes f(x) = a_0 + a_1x + a_2x^2 + a_3x^3 ... is a polynomial.
Uses Hensel lifting to find solutions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `coefficients` | `Sequence[int]` | Polynomial coefficients, where coefficients[i] is the coefficient for x^i |
| `p` | `int` | Prime base of modulus |
| `k` | `int` | Exponent of modulus |
| `initial` | `list[int]` | Initial solutions to f(x) ≡ 0 (mod p) |

**Complexity:** `O(ksd) time, where s is total number of solutions and d = deg(f).
O(pd) to find initial solutions if not provided.`

---

### discrete_log

```python
discrete_log(a: int, b: int, mod: int) -> int | None
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2656)</sup>

> Find the smallest non-negative integer x such that a^x ≡ b (mod m).

Uses the Pohlig-Hellman algorithm, with either baby-step giant-step or
Pollard's rho for discrete logarithms on the prime-order sub-problems.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | Base of logarithm |
| `b` | `int` | Target integer |
| `mod` | `int` | Modulus |

---

### modular_roots

```python
modular_roots(n: int, k: int, mod: int) -> tuple[int, ...]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L2715)</sup>

> Find all solutions x to x^k ≡ n (mod m).

Uses the Tonelli-Shanks / Adleman-Manders-Miller to find roots modulo primes,
Hensel lifting to roots modulo prime powers, and the Chinese Remainder Theorem
to combine solutions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Target integer |
| `k` | `int` | Order of root |
| `mod` | `int` | Modulus |

---

<br>

# Diophantine Equations

### bezout

```python
bezout(a: int, b: int, c: int) -> Iterator[tuple[int, int]]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L3183)</sup>

> Generate all integer solutions to the linear Diophantine equation ax + by = c.

Uses the extended Euclidean algorithm to find a pair of Bézout coefficients,
and then generate an infinite family of solutions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | Coefficient of x |
| `b` | `int` | Coefficient of y |
| `c` | `int` | Constant term |

**Yields**

| Name | Type | Description |
|------|------|-------------|
| `x` | `int` | X-coordinate of solution |
| `y` | `int` | Y-coordinate of solution |

**Complexity:** `O(log(min(a, b))) time to find initial solution, O(1) per additional solution.`

---

### cornacchia

```python
cornacchia(d: int, m: int) -> Iterator[tuple[int, int]]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L3232)</sup>

> Generate positive integer solutions to the equation x^2 + dy^2 = m
where 0 < d < m and gcd(d, m) = 1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `d` | `int` | Coefficient of y^2 term |
| `m` | `int` | Constant term |

**Yields**

| Name | Type | Description |
|------|------|-------------|
| `x` | `int` | X-coordinate of solution |
| `y` | `int` | Y-coordinate of solution |

**Complexity:** `O(f(m) + τ(m) g(m) + τ(m) log m) time, where f, τ, g are the cost of
factorization, divisor count, and cost of modular roots respectively.`

---

### pell

```python
pell(D: int, N: int=1) -> Iterator[tuple[int, int]]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L3307)</sup>

> Generate positive integer solutions to the generalized Pell equation x^2 - Dy^2 = N
where D is not a perfect square.

Yields infinite positive integer solutions x, y > 0 in order of increasing x.
Uses the Lagrange-Matthews-Mollin (LMM) algorithm.

See: https://cjhb.site/Files.php/Books/math/B3.4/pell.pdf
See: http://www.numbertheory.org/PDFS/patz_improved.pdf

| Parameter | Type | Description |
|-----------|------|-------------|
| `D` | `int` | Coefficient of y^2 term |
| `N` | `int` | Constant term |

**Yields**

| Name | Type | Description |
|------|------|-------------|
| `x` | `int` | X-coordinate of solution |
| `y` | `int` | Y-coordinate of solution |

**Complexity:** `O(L(D) + f(|N|) + τ(|N|) * (g(|N|) + L(D))) time, where L, f, τ, g are the
continued-fraction period length, cost of factoring, divisor count,
and cost of modular roots respectively.`

---

### binary_quadratic_solve

```python
binary_quadratic_solve(a: int, b: int, c: int, n: int) -> Iterator[tuple[int, int]]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L3403)</sup>

> Generate all integer solutions (x, y) to the binary quadratic equation
ax^2 + bxy + cy^2 = n.

Uses the theory of binary quadratic forms to find solutions based on the
discriminant Δ = b^2 - 4ac:

    Δ < 0: definite form, finite solutions (via Cornacchia)
    Δ = 0: parabolic form, reduces to linear (via Bezout's identity)
    Δ > 0, square: degenerate form, factors into linear forms
    Δ > 0, non-square: indefinite form, infinite solutions (via Pell equation)

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | Coefficient of x^2 term |
| `b` | `int` | Coefficient of xy term |
| `c` | `int` | Coefficient of y^2 term |
| `n` | `int` | Target value |

**Yields**

| Name | Type | Description |
|------|------|-------------|
| `x` | `int` | X-coordinate of solution |
| `y` | `int` | Y-coordinate of solution |

**Complexity:** `For definite forms: O(τ(n)*f(n)) where τ, f are divisor count and factorization cost
For indefinite forms: O(1) per solution after initial setup`

---

### pythagorean_triples

```python
pythagorean_triples(max_c: float | None=None, max_sum: float | None=None) -> Iterator[tuple[int, int, int]]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L3461)</sup>

> Generate positive integer solutions to the equation a^2 + b^2 = c^2.

Uses Euclid's formula to generate unique Pythagorean triples (a, b, c)
where a <= b <= c.

If no bounds are specified, infinitely generates triples in order of increasing c.
When bounds are specified, no order is guaranteed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_c` | `float` | Upper bound for c in generated triples, where c <= max_c |
| `max_sum` | `float` | Upper bound for the sum of generated triples, where a + b + c <= max_sum |

---

### pillai

```python
pillai(a: int, b: int, c: int) -> Iterator[tuple[int, int]]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L3522)</sup>

> Generate all positive integer solutions (x, y) to the Pillai equation a^x - b^y = c,
where a, b >= 2 and x, y > 0.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | Base of x term |
| `b` | `int` | Base of y term |
| `c` | `int` | Integer target |

**Yields**

| Name | Type | Description |
|------|------|-------------|
| `x` | `int` | X-coordinate of solution |
| `y` | `int` | Y-coordinate of solution |

---

<br>

# Lattice Methods

### integer_solve

```python
integer_solve(A: Matrix[int], b: Vector[int]) -> Vector[int] | None
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L3853)</sup>

> Find an integer solution x to Ax = b, if one exists.

Uses Smith normal form to reduce the system to diagonal form,
then checks divisibility conditions for solvability.

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `Matrix` | Integer coefficient matrix |
| `b` | `Vector` | Integer target vector |

---

### integer_nullspace

```python
integer_nullspace(A: Matrix[int]) -> list[Vector[int]]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L3908)</sup>

> Return a Z-basis for the integer nullspace of A.

The nullspace consists of all integer vectors x such that Ax = 0.
Uses Smith normal form to identify free variables.

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `Matrix` | Integer matrix |

---

### lll_reduce

```python
lll_reduce(B: Matrix[int], delta: float=0.75) -> Matrix[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L3933)</sup>

> LLL lattice basis reduction.

Given a basis for a lattice, returns a "reduced" basis whose vectors
are more orthogonal and shorter. This is useful for solving approximate
closest vector problems and breaking certain cryptographic schemes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `B` | `Matrix` | Integer matrix whose rows form a lattice basis |
| `delta` | `float` | Lovasz parameter (default 0.75). Must satisfy 0.25 < delta < 1. Higher values give better reduction but slower runtime. |

**Complexity:** `O(n⁵ log³B) time for n × n matrix with max entry size B.
O(n²) space for intermediate rational arithmetic.`

---

### babai_closest_vector

```python
babai_closest_vector(B: Matrix[int], target: Vector[int]) -> Vector[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L3996)</sup>

> Babai nearest-plane algorithm for approximate closest vector.

Given a lattice basis B and a target vector, finds a lattice vector
that is close to the target. For best results, use an LLL-reduced basis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `B` | `Matrix` | LLL-reduced lattice basis (rows) |
| `target` | `Vector` | Target vector in ambient space |

**Complexity:** `O(n³) time for n × n basis (dominated by Gram-Schmidt).
O(n²) space for orthogonalized basis.`

---

<br>

# Combinatorics

### pascal

```python
pascal(num_rows: int | None=None) -> Iterator[tuple[tuple[int, int], int]]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4268)</sup>

> Generate values in Pascal's triangle, row by row, left to right.

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_rows` | `int` | Number of rows to generate (infinite by default) |

**Yields**

| Name | Type | Description |
|------|------|-------------|
| `a` | `int` | Binomial coefficient a = (n choose k) |

---

### factorial_valuation

```python
factorial_valuation(n: int, p: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4289)</sup>

> Compute the p-adic valuation of n!, i.e., the exponent of p in n!.

Uses Legendre's formula: v_p(n!) = (n - S_p(n)) / (p - 1),
where S_p(n) is the digit sum of n in base p.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Non-negative integer |
| `p` | `int` | Prime number |

**Complexity:** `O(logₚ n) time for base conversion and digit sum.
O(logₚ n) space.`

---

### binomial_valuation

```python
binomial_valuation(n: int, k: int, p: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4317)</sup>

> Compute the p-adic valuation of binomial coefficient C(n, k).

Uses Kummer's theorem: v_p(C(n, k)) equals the number of carries
when adding k and (n - k) in base p.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Non-negative integer |
| `k` | `int` | Non-negative integer with k <= n |
| `p` | `int` | Prime number |

**Complexity:** `O(logₚ n) time and space for base conversion and carry computation.`

---

### partition_numbers

```python
partition_numbers(mod: int | None=None) -> Iterator[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4353)</sup>

> Generate the values of the partition function using Euler's pentagonal recurrence.

**Complexity:** `O(n³ᐟ²) amortized time per term (n-th partition uses O(√n) pentagonal offsets).
O(n) space to store previous partition values.`

---

### count_partitions

```python
count_partitions(n: int, restrict: Callable[[int], bool] | None=None) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4386)</sup>

> Return the number of partitions of integer n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer to partition |
| `restrict` | `Callable(int) -> bool` | Function indicating integers that can be used in the partition, where restrict(k) = True means integer k can be used |

---

### euler_transform

```python
euler_transform(a: Callable[[int], int]) -> Callable[[int], int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4406)</sup>

> Return the Euler transform of integer sequence a.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `Callable(int) -> int` | Integer sequence to transform |

---

<br>

# Integer Sequences

### integers

```python
integers() -> Iterator[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4463)</sup>

> Generate all integers (0, 1, -1, 2, -2, ...) in an infinite generator.

---

### integer_pairs

```python
integer_pairs() -> Iterator[tuple[int, int]]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4472)</sup>

> Generate all integer pairs (x, y) via diagonal enumeration.

---

### lucas

```python
lucas(n: int, P: int=1, Q: int=-1, mod: int | None=None) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4485)</sup>

> Return the n-th Lucas sequence number U_n(P, Q).

The Lucas sequence is defined by the recurrence
    U_0 = 0
    U_1 = 1
    U_n = P * U_{n-1} - Q * U_{n-2}

The Fibonacci sequence is the special case where P=1 and Q=-1.

Uses fast doubling with formulas
    U_{2k} = U_k * (2*U_{k+1} - P*U_k)
    U_{2k-1} = U_k^2 - Q*U_{k-1}^2
    U_{2k+1} = P*U_{2k} - Q*U_{2k-1}

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Index of the Lucas sequence number |
| `P` | `int` | First parameter of the Lucas sequence (default 1) |
| `Q` | `int` | Second parameter of the Lucas sequence (default -1) |
| `mod` | `int` | Optional modulus |

**Complexity:** `O(log n) time using binary fast doubling.
O(log n) space for recursion depth.`

---

### fibonacci

```python
fibonacci(i: int, mod: int | None=None) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4559)</sup>

> Return the i-th Fibonacci number.

The Fibonacci sequence is a special case of the Lucas sequence U_n(1, -1).

| Parameter | Type | Description |
|-----------|------|-------------|
| `i` | `int` | Index of the Fibonacci number |
| `mod` | `int` | Optional modulus |

---

### fibonacci_index

```python
fibonacci_index(n: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4579)</sup>

> Find the index of n in the Fibonacci sequence.
Returns the largest integer i such that F(i) <= n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Upper bound on Fibonacci number |

**Complexity:** `O(log² n) time for logarithmic search with Fibonacci evaluations.
O(log n) space.`

---

### fibonacci_numbers

```python
fibonacci_numbers(a: int=0, b: int=1, mod: int | None=None) -> Iterator[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4622)</sup>

> Generate Fibonacci numbers.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | First element of the Fibonacci sequence |
| `b` | `int` | Second element of the Fibonacci sequence |
| `mod` | `int` | Optional modulus |

---

### polygonal

```python
polygonal(s: int, i: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4642)</sup>

> Return the i-th s-gonal number.

---

### polygonal_index

```python
polygonal_index(s: int, n: int) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4648)</sup>

> Find the index of n in the s-gonal numbers.
Returns the largest integer i such that P(s, i) <= n.

---

### polygonal_numbers

```python
polygonal_numbers(s: int, low: int=1, high: int | None=None) -> Iterator[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4662)</sup>

> Generate all s-gonal numbers in the range [low, high].

---

### is_polygonal

```python
is_polygonal(s: int, n: int) -> bool
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4673)</sup>

> Check if n is an s-gonal number.

---

<br>

# Appendix

### nth

```python
nth(iterable: Iterable, n: int, default: Any=None) -> Any
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4694)</sup>

> Return the n-th item from an iterable (1-based index).
If the iterable has fewer than n items, return default.

---

### alternating

```python
alternating(*iterables: Iterable) -> Iterator
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4703)</sup>

> Visit input iterables in a cycle until each is exhausted.

---

### periodic_continued_fraction

```python
periodic_continued_fraction(D: int, P: int=0, Q: int=1) -> tuple[Iterator[int], int, int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4714)</sup>

> Compute coefficients for the periodic continued fraction
(P + sqrt(D)) / Q = a0 + 1 / (a1 + 1 / (a2 + ...)).

**Returns**

| Name | Type | Description |
|------|------|-------------|
| `coefficients` | `Iterator[int]` | Coefficients of the continued fraction |
| `initial_length` | `int` | Length of the initial non-repeating block |
| `period_length` | `int` | Length of the repeating period |

---

### convergents

```python
convergents(coefficients: Iterable[int], num: int | None=None) -> Iterator[tuple[int, int]]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4753)</sup>

> Return convergents of the continued fraction with the given coefficients.

| Parameter | Type | Description |
|-----------|------|-------------|
| `coefficients` | `Iterable[int]` | Coefficients of the continued fraction |
| `num` | `int` | Maximum number of convergents to generate (infinite by default) |

**Yields**

| Name | Type | Description |
|------|------|-------------|
| `numerator` | `int` | Numerator of the convergent |
| `denominator` | `int` | Denominator of the convergent |

---

### permutation

```python
permutation(n: int, master_key: bytes | None=None) -> Iterator[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4781)</sup>

> Generate a pseudorandom permutation of the integers 0, 1, ..., n - 1.

---

### polynomial

```python
polynomial(coefficients: Sequence[Number]) -> Callable[[Number], Number]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4835)</sup>

> Create a polynomial function with the given coefficients (a_0, ..., a_n).
Uses Horner's method for polynomial evaluation.

---

### iroot

```python
iroot(x: int, n: int=2) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4847)</sup>

> Find the integer n-th root of x.
Returns the largest integer a such that a^n <= x.
Uses Newton's method.

---

### ilog

```python
ilog(a: int, b: int=2) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4878)</sup>

> Find the integer logarithm of a with base b.
Returns the largest integer n such that b^n <= a.
Uses repeated squaring and binary search.

---

### is_square

```python
is_square(n: int) -> bool
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4904)</sup>

> Check if an integer n is a square.

---

### non_squares

```python
non_squares(N: int) -> Iterator[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4923)</sup>

> Return all non-square positive integers <= N.

---

### squares

```python
squares(low: int=0, high: int | None=None) -> Iterator[int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4910)</sup>

> Generate square numbers in the range [low, high].

---

### perfect_power

```python
perfect_power(n: int) -> tuple[int, int]
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4929)</sup>

> Find integers a, b such that a^b = n.

Returns the solution (a, b) with minimal b > 1 if there are any such solutions,
otherwise returns the trivial solution (n, 1).

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer target |

---

### binary_search

```python
binary_search(f: Callable[[int], int], threshold: int, low: int=0, high: int | None=None) -> int
```
<sup>[source](https://github.com/ini/numthy/blob/main/numthy.py#L4954)</sup>

> Given a monotonically increasing function f, find where it crosses a threshold.
Returns the smallest integer n in [low, high] such that f(n) >= threshold.

---
