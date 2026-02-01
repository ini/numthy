<div align="center">

# NumThy API Reference

*Computational number theory. Pure Python. Zero dependencies. Unreasonably fast.*

`v0.1.0`

</div>

---

## Contents

- [Primes](#primes)
- [Factorization](#factorization)
- [Arithmetic Functions](#arithmetic-functions)
- [Modular Arithmetic](#modular-arithmetic)
- [Nonlinear Congruences](#nonlinear-congruences)
- [Diophantine Equations](#diophantine-equations)
- [Algebraic Systems](#algebraic-systems)
- [Lattices](#lattices)
- [Appendix](#appendix)

---

<br>

# Primes

### is_prime

```python
is_prime(n: int) -> bool
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L91)

> Test if a given integer n is prime.

Uses a combination of trial division, the Miller-Rabin primality test
with deterministic bases, or the extra-strong variant of the Baillie-PSW
primality test (this variant has no known pseudoprimes in any range, and
has been computationally verified to have no counterexamples for all n < 2^64).

See: https://www.techneon.com/download/is.prime.32.base.data (MR hash for n < 2^32)
See: https://miller-rabin.appspot.com (other deterministic MR base sets)
See: https://ntheory.org/pseudoprimes.html (BPSW verification up to 2^64)

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer to test for primality |

---

### next_prime

```python
next_prime(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L135)

> Get the smallest prime number greater than n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Strict lower bound for prime number |

---

### random_prime

```python
random_prime(num_bits: int, *, safe: bool=False) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L153)

> Generate a random prime with the given number of bits.

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_bits` | `int` | Number of bits in the prime to be generated |
| `safe` | `bool` | Whether or not to generate a safe prime (i.e. prime q of the form q = 2p + 1, where p is also prime) |

---

### primes

```python
primes(*, low: int=2, high: int | None=None, count: int | None=None) -> Iterator[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L193)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L265)

> Prime counting function π(x). Returns the number of primes p ≤ x.

Uses the Lagarias-Miller-Odlyzko (LMO) extension of the Meissel-Lehmer algorithm.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `int` | Upper bound for prime numbers |

---

### sum_primes

```python
sum_primes(x: int, f: Callable[[int], Number] | None=None, f_prefix_sum: Callable[[int], Number] | None=None) -> Number
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L283)

> Compute F(x) as the sum of f(p) over all primes p ≤ x,
where f is a completely multiplicative function (by default, f(n) = n).

Uses a generalized version of the LMO prime counting algorithm.
Ideally `f()` and `f_prefix_sum()` can be calculated efficiently
in O(1) time via closed-form expression.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `int` | Upper bound for prime numbers |
| `f` | `Callable(int) -> Number` | Completely multiplicative function f(n), where f(1) = 1 and f(ab) = f(a) * f(b) for all a, b > 0 |
| `f_prefix_sum` | `Callable(int) -> Number` | Function to compute the cumulative sum Σ_{1 ≤ k ≤ n} f(k) |

---

<br>

# Factorization

### perfect_power

```python
perfect_power(n: int) -> tuple[int, int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L907)

> Find integers a, b such that a^b = n.

Returns the solution (a, b) with minimal b > 1 if there are any such solutions,
otherwise returns the trivial solution (n, 1).

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer target |

---

### prime_factors

```python
prime_factors(n: int) -> tuple[int, ...]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L967)

> Get all prime factors of n in sorted order (with multiplicity).

Uses a combination of trial division, Fermat's factorization method,
Brent's variant of Pollard's rho, Lenstra's elliptic curve method (ECM),
and a self-initializing quadratic sieve (SIQS).

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer to factor |

---

### prime_factorization

```python
prime_factorization(n: int) -> dict[int, int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L982)

> Get the prime factorization of n as a dictionary of {prime: exponent}.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer to factor |

---

### divisors

```python
divisors(n: int) -> tuple[int, ...]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L996)

> Get all positive divisors of n in sorted order.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer to factor |

---

<br>

# Arithmetic Functions

### omega

```python
omega(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1945)

> Compute the value of ω(n), the number of distinct prime factors of n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### big_omega

```python
big_omega(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1958)

> Compute the value of Ω(n), the number of prime factors of n (with multiplicity).

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### divisor_count

```python
divisor_count(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1971)

> Compute the value of σ₀(n), the number of divisors of n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### divisor_sum

```python
divisor_sum(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1984)

> Compute the value of σ₁(n), the sum of divisors of n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### divisor_function

```python
divisor_function(n: int, k: int=1) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1997)

> Compute the value of the divisor function σₖ(n), where σₖ(n) = ∑_{d|n} dᵏ.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |
| `k` | `int` | Divisor exponent |

---

### partition

```python
partition(n: int, mod: int | None=None, restrict: Callable[[int], bool] | None=None) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2019)

> Return the value of the partition function p(n).

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer to partition |
| `mod` | `int | None` | If provided, return p(n) mod m |
| `restrict` | `Callable(int) -> bool` | Function indicating integers that can be used in the partition, where restrict(k) = True means integer k can be used (e.g. restrict=nt.is_prime) |

---

### radical

```python
radical(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2043)

> Compute rad(n) as the product of the distinct prime factors of n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### mobius

```python
mobius(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2056)

> Compute the Mobius function μ(n) for a positive integer n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### totient

```python
totient(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2084)

> Compute Euler's totient function φ(n) for a positive integer n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### carmichael

```python
carmichael(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2102)

> Compute Carmichael's lambda function λ(n) for a positive integer n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### valuation

```python
valuation(n: int, p: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2123)

> Compute the p-adic valuation νₚ(n), the exponent of p
in the prime factorization of n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer |
| `p` | `int` | Prime number |

---

### multiplicative_range

```python
multiplicative_range(f: Callable[..., int], N: int, f0: int=1) -> list[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2158)

> Find the value of a multiplicative function f(n) for each n = 0, 1, 2, ..., N - 1.
Uses a sieving approach across the range to calculate function values efficiently.

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | `Callable(n) -> int or Callable(p, e) -> int` | Function to compute values f(n) or f(p^e) at prime powers |
| `N` | `int` | Upper bound on range (exclusive) |
| `f0` | `int` | Dummy value to include for f(0) |

---

<br>

# Modular Arithmetic

### egcd

```python
egcd(a: int, b: int) -> tuple[int, int, int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2296)

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
crt(congruences: Iterable[tuple[int, int]]) -> int | None
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2334)

> Solve a system of linear congruences x ≡ aᵢ (mod nᵢ)
via the Chinese Remainder Theorem.

Returns a solution to the system of congruences, mod the LCM of the moduli,
or None if no solution exists.

Supports non-coprime moduli.

| Parameter | Type | Description |
|-----------|------|-------------|
| `congruences` | `Iterable[tuple[int, int]]` | Congruences as (residue, moduli) tuples |

---

### coprimes

```python
coprimes(n: int) -> Iterator[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2354)

> Generate all integers k in the range [0, n) that are coprime to n.

Returns the reduced residue system modulo n, i.e., the unit group (Z/nZ)×.
The size of this set is φ(n) (Euler's totient function).

For small n, uses an O(n) space sieve for speed. For large n, uses an
O(1) space generator that checks gcd(k, n) = 1 for each k.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer modulus |

**Complexity:** `O(n * ω(n)) time and O(n) space for n ≤ 10⁷ (sieve approach).
O(n log n) time and O(1) space for n > 10⁷ (gcd approach).`

---

### multiplicative_order

```python
multiplicative_order(a: int, mod: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2381)

> Compute ordₘ(a), the smallest positive integer such that a^k ≡ 1 (mod m).

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | Integer base |
| `mod` | `int` | Integer modulus |

---

### primitive_root

```python
primitive_root(n: int) -> int | None
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2408)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2447)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2467)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2501)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2535)

> Return the Dirichlet character χₘ‚ₖ : ℤ → ℂ under Conrey labeling,
where m is the modulus and k is an index such that gcd(m, k) = 1.

See: https://www.lmfdb.org/knowledge/show/character.dirichlet.conrey

| Parameter | Type | Description |
|-----------|------|-------------|
| `m` | `int` | Modulus of the character |
| `k` | `int` | Index of the character |

**Returns**

| Name | Type | Description |
|------|------|-------------|
| `chi` | `Callable(int) -> Number` | Dirichlet character χₘ‚ₖ(n) as a callable function returning the character value at n |

---

<br>

# Nonlinear Congruences

### hensel

```python
hensel(coefficients: Sequence[int], p: int, k: int, initial: Iterable[int] | None=None) -> tuple[int, ...]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2753)

> Find all solutions to the polynomial congruence f(x) ≡ 0 (mod pᵏ).

Assumes f(x) = a₀ + a₁x + a₂x² + a₃x³ ... is a polynomial.
Uses Hensel lifting to find solutions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `coefficients` | `Sequence[int]` | Polynomial coefficients, where coefficients[i] = aᵢ is the coefficient for xⁱ |
| `p` | `int` | Prime base of modulus |
| `k` | `int` | Exponent of modulus |
| `initial` | `Iterable[int]` | Initial solutions to f(x) ≡ 0 (mod p) |

**Complexity:** `O(ksd) time, where s is total number of solutions and d = deg(f).
O(pd) to find initial solutions if not provided.`

---

### polynomial_roots

```python
polynomial_roots(coefficients: Sequence[int], mod: int) -> tuple[int, ...]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2821)

> Find all roots x of a univariate polynomial f(x) ≡ 0 (mod m).

Factors m into prime powers, finds roots modulo each p^e via
Cantor-Zassenhaus + Hensel lifting, then combines solutions with CRT.

| Parameter | Type | Description |
|-----------|------|-------------|
| `coefficients` | `Sequence[int]` | Polynomial coefficients, where coefficients[i] is the coefficient for x^i |
| `mod` | `int` | Modulus |

---

### nth_roots

```python
nth_roots(a: int, n: int, mod: int) -> tuple[int, ...]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2860)

> Find all solutions x to x^n ≡ a (mod m).

Uses the Tonelli-Shanks / Adleman-Manders-Miller to find roots modulo primes,
Hensel lifting to roots modulo prime powers, and the Chinese Remainder Theorem
to combine solutions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | Target integer |
| `n` | `int` | Order of root |
| `mod` | `int` | Modulus |

---

### discrete_log

```python
discrete_log(target: int, base: int, mod: int) -> int | None
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2899)

> Find the smallest non-negative integer x such that target ≡ base^x (mod m).
Returns None if no such integer exists.

Uses the Pohlig-Hellman algorithm, with either baby-step giant-step or
Pollard's rho for discrete logarithms on the prime-order sub-problems.

| Parameter | Type | Description |
|-----------|------|-------------|
| `target` | `int` | Target integer |
| `base` | `int` | Base of logarithm |
| `mod` | `int` | Modulus |

---

<br>

# Diophantine Equations

### bezout

```python
bezout(a: int, b: int, c: int) -> Iterator[tuple[int, int]]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L3599)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L3648)

> Generate all unique positive integer solutions to the equation x² + dy² = m
where 0 < d < m and gcd(d, m) = 1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `d` | `int` | Coefficient of y² term |
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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L3723)

> Generate all unique positive integer solutions to the generalized Pell equation
x² - Dy² = N, where D is not a perfect square.

Yields infinite positive integer solutions x, y > 0 in order of increasing x.
Uses the Lagrange-Matthews-Mollin (LMM) algorithm.

See: https://cjhb.site/Files.php/Books/math/B3.4/pell.pdf
See: http://www.numbertheory.org/PDFS/patz_improved.pdf

| Parameter | Type | Description |
|-----------|------|-------------|
| `D` | `int` | Coefficient of y² term |
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

### conic

```python
conic(a: int, b: int, c: int, d: int, e: int, f: int) -> Iterator[tuple[int, int]]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L3830)

> Generate all unique integer solutions (x, y) to the binary quadratic Diophantine
conic equation ax² + bxy + cy² + dx + ey + f = 0.

Uses the theory of binary quadratic forms, classifying by discriminant Δ = b² - 4ac:

    Δ < 0 (ellipse): Lagrange reduction, finite solutions
    Δ = 0 (parabola): parametric families via modular square roots
    Δ > 0 (hyperbola): reduction to Pell equation, infinite solutions
    Degenerate cases: factorization into linear forms

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | Coefficient of x² term |
| `b` | `int` | Coefficient of xy term |
| `c` | `int` | Coefficient of y² term |
| `d` | `int` | Coefficient of x term |
| `e` | `int` | Coefficient of y term |
| `f` | `int` | Constant term |

**Yields**

| Name | Type | Description |
|------|------|-------------|
| `x` | `int` | X-coordinate of solution |
| `y` | `int` | Y-coordinate of solution |

---

### pythagorean_triples

```python
pythagorean_triples(max_c: float | None=None, max_sum: float | None=None) -> Iterator[tuple[int, int, int]]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L3889)

> Generate positive integer solutions to the equation a² + b² = c².

Uses Euclid's formula to generate unique Pythagorean triples (a, b, c)
where a ≤ b ≤ c.

If no bounds are specified, infinitely generates triples in order of increasing c.
When bounds are specified, no order is guaranteed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_c` | `float` | Upper bound for c in generated triples, where c ≤ max_c |
| `max_sum` | `float` | Upper bound for the sum of generated triples, where a + b + c ≤ max_sum |

---

### pillai

```python
pillai(a: int, b: int, c: int) -> Iterator[tuple[int, int]]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L3950)

> Generate all positive integer solutions (x, y) to the exponential Diophantine
Pillai equation aˣ - bʸ = c, where a, b >= 2 and x, y > 0.

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

# Algebraic Systems

### solve_linear_system

```python
solve_linear_system(A: Matrix[int], b: Vector[int] | None=None, *, nullspace: bool=False) -> tuple[Vector[int] | None, list[Vector[int]] | None]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4368)

> Find integer solutions to the system of linear equations given by Ax = b.

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `Matrix[int]` | M x N matrix, with M equations of N variables |
| `b` | `Vector[int]` | Target vector of length M |
| `nullspace` | `bool` | Whether or not compute and return a basis for the null space |

**Returns**

| Name | Type | Description |
|------|------|-------------|
| `solution` | `Vector[int] or None` | Particular solution x such that Ax = b, or None if no solution exists |
| `nullspace_basis` | `list[Vector[int]] or None` | List of basis vectors for null space, or None if nullspace=False |

---

### solve_polynomial_system

```python
solve_polynomial_system(polynomials: list[Polynomial[int]], bounds: tuple[int, ...]) -> tuple[tuple[int, ...], ...]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4483)

> Find integer solutions to a system of multivariate polynomial equations
f₁(x₁, x₂, ...) = f₂(x₁, x₂, ...) = ... = fₖ(x₁, x₂, ...) = 0,
where |xᵢ| < bounds[i] for each variable.

Polynomials are represented as dictionaries mapping monomial tuples to coefficients,
e.g. {(2, 0): 3, (0, 1): -5, (0, 0): 7} represents 3x² - 5y + 7.

Not guaranteed to find *all* solutions for large bounds.

| Parameter | Type | Description |
|-----------|------|-------------|
| `polynomials` | `list[dict[tuple[int, ...], int]]` | System of multivariate polynomials with integer coefficients |
| `bounds` | `tuple[int, ...]` | Bounds on solution size, where \|xᵢ\| < bounds[i] for each variable |

---

<br>

# Lattices

### lll_reduce

```python
lll_reduce(B: Matrix[int]) -> Matrix[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4984)

> Lenstra-Lenstra-Lovász (LLL) lattice basis reduction.

Returns a reduced basis with shorter, more orthogonal vectors, satisfying:

    Size-reduction: |μ_{i,j}| ≤ 0.5 for all i > j
    Lovász condition: δ‖b*_k‖² ≤ ‖b*_{k+1}‖² + μ_{k+1,k}² ‖b*_k‖²

Uses floating-point arithmetic for speed, with automatic escalation
to exact rational arithmetic if precision issues are detected.

See: https://www.cs.cmu.edu/~avrim/451f11/lectures/lect1129_LLL.pdf

| Parameter | Type | Description |
|-----------|------|-------------|
| `B` | `Matrix[int]` | Integer matrix whose rows form a lattice basis |

**Complexity:** `O(n⁵d log³B) time for n × d matrix with max entry size B, O(n² + nd) space`

---

### bkz_reduce

```python
bkz_reduce(B: Matrix[int], block_size: int=20) -> Matrix[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5015)

> BKZ (Block Korkine-Zolotarev) lattice basis reduction.

BKZ generalizes LLL by applying an SVP (Shortest Vector Problem) oracle
to sliding blocks of consecutive basis vectors.

Uses Schnorr-Euchner enumeration for the SVP oracle.

See: https://www.sciencedirect.com/science/article/pii/0304397587900648

| Parameter | Type | Description |
|-----------|------|-------------|
| `B` | `Matrix[int]` | Integer matrix whose rows form a lattice basis |
| `block_size` | `int` | Block size β for BKZ reduction. Larger values give better reduction but exponentially slower runtime. |

**Complexity:** `O(2^(0.25β²)) per block, O(β^β) worst case`

---

### closest_vector

```python
closest_vector(B: Matrix[int], target: Vector[int]) -> Vector[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5063)

> Find the (approximate) closest vector to the target in the lattice
with basis given by rows of matrix B.

Uses Babai nearest-plane algorithm for approximate closest vector.

| Parameter | Type | Description |
|-----------|------|-------------|
| `B` | `Matrix` | LLL-reduced lattice basis (rows) |
| `target` | `Vector` | Target vector in ambient space |

**Complexity:** `O(n²d) time, O(n² + nd) space for n × d matrix (n vectors of dimension d)`

---

### small_roots

```python
small_roots(polynomial: Polynomial[int], mod: int, bounds: tuple[int, ...] | None=None, *, epsilon: float=0.05) -> list[tuple[int, ...]]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5104)

> Find small integer roots of a multivariate polynomial f(x₁, x₂, ...) ≡ 0 (mod M).

Uses the Jochemsz-May multivariate generalization of Coppersmith's method.

See: https://www.iacr.org/archive/asiacrypt2006/42840270/42840270.pdf
See: https://cr.yp.to/bib/2001/howgrave-graham.pdf
See: https://link.springer.com/chapter/10.1007/3-540-68339-9_14

| Parameter | Type | Description |
|-----------|------|-------------|
| `polynomial` | `dict[tuple[int, ...], int]` | Multivariate polynomial with integer coefficients as {monomial: coefficient} where each monomial is a tuple indicating the exponents for each variable (e.g. {(1, 0): 5, (0, 1): 3, (0, 0): -7} represents 5x + 3y - 7) |
| `mod` | `int` | Modulus |
| `bounds` | `tuple[int, ...] or None` | Bound on root size, where \|xᵢ\| < bᵢ for each variable xᵢ. Required for multivariate polynomials. For univariate, defaults to M^(1/deg). |
| `epsilon` | `float` | Parameter controlling lattice dimension vs root bound trade-off. Smaller epsilon allows for larger bounds but requires larger lattice (slower). |

**Complexity:** `Brute force path is O(Π(2Bᵢ - 1)) time
Lattice path is dominated by LLL on an H × W matrix,
about O(H⁵W log³A) time and O(H² + HW) space, where A is the max lattice`

---

<br>

# Appendix

### integers

```python
integers() -> Iterator[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5684)

> Generate all integers (0, 1, -1, 2, -2, ...) in an infinite generator.

---

### integer_pairs

```python
integer_pairs() -> Iterator[tuple[int, int]]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5693)

> Generate all integer pairs (x, y) via diagonal enumeration.

---

### alternating

```python
alternating(*iterables: Iterable) -> Iterator
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5705)

> Visit input iterables in a cycle until each is exhausted.

---

### below

```python
below(f: Callable[[int], int], upper_bound: int, start: int=0) -> Iterable[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5716)

> Yield consecutive values of n >= start as long f(n) < upper_bound.

---

### lower_bound

```python
lower_bound(f: Callable[[int], int], f_min: int, low: int=0, high: int | None=None) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5722)

> Given a monotonically increasing function f, find where it first reaches f_min.
Returns the smallest integer n in [low, high] such that f(n) >= f_min.

---

### permutation

```python
permutation(n: int, master_key: bytes | None=None) -> Iterator[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5741)

> Generate a pseudorandom permutation of the integers 0, 1, ..., n - 1.

---

### is_square

```python
is_square(n: int) -> bool
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5795)

> Check if an integer n is a square.

---

### iroot

```python
iroot(x: int, n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5801)

> Find the integer n-th root of x.
Returns the largest integer a such that a^n ≤ x.
Uses Newton's method.

---

### ilog

```python
ilog(a: int, b: int=2) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5836)

> Find the integer logarithm of a with base b.
Returns the largest integer n such that b^n ≤ a.
Uses repeated squaring and binary search.

---

### fibonacci

```python
fibonacci(n: int, mod: int | None=None) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5864)

> Return the n-th Fibonacci number.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Index of the Fibonacci number |
| `mod` | `int` | Optional modulus |

---

### fibonacci_index

```python
fibonacci_index(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5895)

> Find the index of n in the Fibonacci sequence.
Returns the largest integer i such that F(i) <= n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Upper bound on Fibonacci number |

**Complexity:** `O(log² n) time for logarithmic search with Fibonacci evaluations`

---

### polygonal

```python
polygonal(s: int, i: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5937)

> Return the i-th s-gonal number.

---

### polygonal_index

```python
polygonal_index(s: int, n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5943)

> Find the index of n in the s-gonal numbers.
Returns the largest integer i such that P(s, i) ≤ n.

---

### periodic_continued_fraction

```python
periodic_continued_fraction(D: int, P: int=0, Q: int=1) -> tuple[Iterator[int], int, int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5959)

> Compute coefficients for the periodic continued fraction
(P + sqrt(D)) / Q = a₀ + 1 / (a₁ + 1 / (a₂ + ...)).

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L6005)

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

### polynomial

```python
polynomial(coefficients: Sequence[Number], mod: int | None=None) -> Callable[[Number], Number]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L6033)

> Create a univariate polynomial function with the given coefficients (a₀, ..., aₙ).
Uses Horner's method for polynomial evaluation.

---
