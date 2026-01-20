<div align="center">

# NumThy API Reference

*Computational number theory. Pure Python. Zero dependencies. Unreasonably fast.*

`v0.0.0`

</div>

---

## Contents

- [Primes](#primes)
- [Factorization](#factorization)
- [Arithmetic Functions](#arithmetic-functions)
- [Modular Arithmetic](#modular-arithmetic)
- [Nonlinear Congruences](#nonlinear-congruences)
- [Diophantine Equations](#diophantine-equations)
- [Lattices](#lattices)
- [Sequences](#sequences)
- [Appendix](#appendix)

---

<br>

# Primes

### is_prime

```python
is_prime(n: int) -> bool
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L90)

> Test if a given integer n is prime.

Uses a combination of trial division, the Miller-Rabin primality test
with deterministic bases, or the extra-strong variant of the Baillie-PSW
primality test (this variant has no known pseudoprimes in any range, and
has been computationally verified to have no counterexamples for all n < 2^64).

See: https://miller-rabin.appspot.com (deterministic Miller-Rabin base sets)
See: https://ntheory.org/pseudoprimes.html (BPSW verification up to 2^64)

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer to test for primality |

---

### next_prime

```python
next_prime(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L134)

> Get the smallest prime number greater than n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Strict lower bound for prime number |

---

### random_prime

```python
random_prime(num_bits: int, *, safe: bool=False) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L152)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L187)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L259)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L279)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L901)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L916)

> Get the prime factorization of n as a dictionary of {prime: exponent}.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer to factor |

---

### divisors

```python
divisors(n: int) -> tuple[int, ...]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L927)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1878)

> Compute the value of ω(n), the number of distinct prime factors of n.

---

### big_omega

```python
big_omega(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1886)

> Compute the value of Ω(n), the number of prime factors of n (with multiplicity).

---

### divisor_count

```python
divisor_count(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1894)

> Compute the value of σ₀(n), the number of divisors of n.

---

### divisor_sum

```python
divisor_sum(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1902)

> Compute the value of σ₁(n), the sum of divisors of n.

---

### divisor_function

```python
divisor_function(n: int, k: int=1) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1910)

> Compute the value of the divisor function σₖ(n), where σₖ(n) = ∑_{d|n} dᵏ.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |
| `k` | `int` | Divisor exponent |

---

### radical

```python
radical(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1932)

> Compute rad(n) as the product of the distinct prime factors of n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### mobius

```python
mobius(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1945)

> Compute the Mobius function μ(n) for a positive integer n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### totient

```python
totient(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1973)

> Compute Euler's totient function φ(n) for a positive integer n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### carmichael

```python
carmichael(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L1991)

> Compute Carmichael's lambda function λ(n) for a positive integer n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer function argument |

---

### valuation

```python
valuation(n: int, p: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2012)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2036)

> Find the value of a multiplicative function f(n) for each n = 0, 1, 2, ..., N - 1.

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2115)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2153)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2173)

> Generate all integers k in the range [0, n) that are coprime to n.

Returns the reduced residue system modulo n, i.e., the unit group (Z/nZ)×.
The size of this set is φ(n) (Euler's totient function).

For small n, uses an O(n) space sieve for speed. For large n, uses an
O(1) space generator that checks gcd(k, n) = 1 for each k.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Positive integer modulus |

**Complexity:** `O(n * ω(n)) time and O(n) space for n <= 10⁷ (sieve approach).
O(n log n) time and O(1) space for n > 10⁷ (gcd approach).`

---

### multiplicative_order

```python
multiplicative_order(a: int, mod: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2200)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2227)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2266)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2286)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2320)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2354)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2572)

> Find all solutions to the polynomial congruence f(x) ≡ 0 (mod pᵏ).

Assumes f(x) = a₀ + a₁x + a₂x² + a₃x³ ... is a polynomial.
Uses Hensel lifting to find solutions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `coefficients` | `Sequence[int]` | Polynomial coefficients, where coefficients[i] = aᵢ is the coefficient for xⁱ |
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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2640)

> Find the smallest non-negative integer x such that b^x ≡ a (mod m).

Uses the Pohlig-Hellman algorithm, with either baby-step giant-step or
Pollard's rho for discrete logarithms on the prime-order sub-problems.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `int` | Target integer |
| `b` | `int` | Base of logarithm |
| `mod` | `int` | Modulus |

---

### modular_roots

```python
modular_roots(n: int, k: int, mod: int) -> tuple[int, ...]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L2698)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L3163)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L3212)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L3287)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L3396)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L3455)

> Generate positive integer solutions to the equation a² + b² = c².

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L3516)

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

# Lattices

### coppersmith

```python
coppersmith(coefficients: Sequence[int], mod: int, X: int | None=None, epsilon: float=0.05, m: int | None=None) -> tuple[int, ...]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L3934)

> Find small integer roots of a monic polynomial modulo M using Coppersmith's method.

Uses LLL lattice reduction to find roots x where |x| < X for f(x) ≡ 0 (mod M).
Based on Howgrave-Graham's formulation of Coppersmith's theorem.

See: https://cr.yp.to/bib/2001/howgrave-graham.pdf
See: https://link.springer.com/chapter/10.1007/3-540-68339-9_14

| Parameter | Type | Description |
|-----------|------|-------------|
| `coefficients` | `Sequence[int]` | Polynomial coefficients (a₀, a₁, a₂, ...) representing f(x) = a₀ + a₁x + a₂x² + ... |
| `mod` | `int` | Modulus |
| `X` | `int` | Bound on root size (\|x\| < X). If None, uses theoretical bound X ≈ M^(1/d - epsilon), where M is the modulus |
| `epsilon` | `float` | Parameter controlling lattice dimension vs root bound trade-off. Smaller epsilon allows larger X but requires larger lattice. Default: 0.05. |
| `m` | `int` | Lattice parameter controlling the number of polynomial shifts. If None, computed from epsilon as m = ceil(1 / (d * epsilon)). Larger m gives better success probability but slower runtime. |

**Complexity:** `O(d⁶ log³N) time and O(d⁴) space, where d is the polynomial degree.`

---

### lll_reduce

```python
lll_reduce(B: Matrix[int]) -> Matrix[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4054)

> Lenstra-Lenstra-Lovász (LLL) lattice basis reduction.

Returns a reduced basis with shorter, more orthogonal vectors,
where |b₁| ≤ 2^((n-1)/2) * λ₁(L) and the Lovász condition holds:
δ‖b*_k‖² ≤ ‖b*_{k+1}‖² + μ_{k+1,k}² ‖b*_k‖².

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
bkz_reduce(B: Matrix[int], block_size: int=20, pruning: float=1.0) -> Matrix[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4084)

> BKZ (Block Korkine-Zolotarev) lattice basis reduction.

BKZ generalizes LLL by applying SVP (Shortest Vector Problem) reduction
to blocks of consecutive basis vectors. Larger block sizes yield better
reduction quality at the cost of exponentially increasing runtime.

Uses Schnorr-Euchner enumeration for the SVP oracle with optional pruning.

See: https://www.sciencedirect.com/science/article/pii/0304397587900648

| Parameter | Type | Description |
|-----------|------|-------------|
| `B` | `Matrix[int]` | Integer matrix whose rows form a lattice basis |
| `block_size` | `int` | Block size β for BKZ reduction. Larger values give better reduction but exponentially slower runtime. |
| `pruning` | `float` | Pruning parameter in (0, 1]. Lower values prune more aggressively, speeding up enumeration but potentially missing the shortest vector. |

**Complexity:** `O(β^β) per block, or O(2^(0.25β²)) with pruning. 2-5 tours typical.`

---

### babai_closest_vector

```python
babai_closest_vector(B: Matrix[int], target: Vector[int]) -> Vector[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4204)

> Babai nearest-plane algorithm for approximate closest vector.

Given a lattice basis B and a target vector, finds a lattice vector
that is close to the target. For best results, use an LLL-reduced basis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `B` | `Matrix` | LLL-reduced lattice basis (rows) |
| `target` | `Vector` | Target vector in ambient space |

**Complexity:** `O(n²d) time, O(n² + nd) space for n × d matrix (n vectors of dimension d)`

---

<br>

# Sequences

### lucas

```python
lucas(n: int, P: int=1, Q: int=-1, mod: int | None=None) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4535)

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

**Complexity:** `O(log n) time using binary fast doubling`

---

### fibonacci

```python
fibonacci(i: int, mod: int | None=None) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4608)

> Return the i-th Fibonacci number.

The Fibonacci sequence is a special case of the Lucas sequence Uₙ(1, -1).

| Parameter | Type | Description |
|-----------|------|-------------|
| `i` | `int` | Index of the Fibonacci number |
| `mod` | `int` | Optional modulus |

---

### fibonacci_index

```python
fibonacci_index(n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4628)

> Find the index of n in the Fibonacci sequence.
Returns the largest integer i such that F(i) <= n.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Upper bound on Fibonacci number |

**Complexity:** `O(log² n) time for logarithmic search with Fibonacci evaluations`

---

### fibonacci_numbers

```python
fibonacci_numbers(a: int=0, b: int=1, mod: int | None=None) -> Iterator[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4670)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4690)

> Return the i-th s-gonal number.

---

### polygonal_index

```python
polygonal_index(s: int, n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4696)

> Find the index of n in the s-gonal numbers.
Returns the largest integer i such that P(s, i) <= n.

---

### polygonal_numbers

```python
polygonal_numbers(s: int, low: int=1, high: int | None=None) -> Iterator[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4712)

> Generate all s-gonal numbers in the range [low, high].

---

### is_polygonal

```python
is_polygonal(s: int, n: int) -> bool
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4723)

> Check if n is an s-gonal number.

---

### partition

```python
partition(n: int, restrict: Callable[[int], bool] | None=None) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4738)

> Return the n-th partition number p(n).

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Integer to partition |
| `restrict` | `Callable(int) -> bool` | Function indicating integers that can be used in the partition, where restrict(k) = True means integer k can be used |

---

### partition_numbers

```python
partition_numbers(mod: int | None=None) -> Iterator[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4757)

> Generate values of the partition function p(n) via Euler's pentagonal recurrence.

**Complexity:** `O(n³ᐟ²) amortized time per term (n-th partition uses O(√n) pentagonal offsets).
O(n) space to store previous partition values.`

---

### euler_transform

```python
euler_transform(a: Callable[[int], int]) -> Callable[[int], int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4791)

> Return the Euler transform of integer sequence a.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `Callable(int) -> int` | Integer sequence to transform |

---

<br>

# Appendix

### integers

```python
integers() -> Iterator[int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4825)

> Generate all integers (0, 1, -1, 2, -2, ...) in an infinite generator.

---

### integer_pairs

```python
integer_pairs() -> Iterator[tuple[int, int]]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4834)

> Generate all integer pairs (x, y) via diagonal enumeration.

---

### alternating

```python
alternating(*iterables: Iterable) -> Iterator
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4846)

> Visit input iterables in a cycle until each is exhausted.

---

### periodic_continued_fraction

```python
periodic_continued_fraction(D: int, P: int=0, Q: int=1) -> tuple[Iterator[int], int, int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4857)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4903)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4931)

> Generate a pseudorandom permutation of the integers 0, 1, ..., n - 1.

---

### polynomial

```python
polynomial(coefficients: Sequence[Number]) -> Callable[[Number], Number]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4985)

> Create a polynomial function with the given coefficients (a₀, ..., aₙ).
Uses Horner's method for polynomial evaluation.

---

### iroot

```python
iroot(x: int, n: int) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L4997)

> Find the integer n-th root of x.
Returns the largest integer a such that a^n <= x.
Uses Newton's method.

---

### ilog

```python
ilog(a: int, b: int=2) -> int
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5028)

> Find the integer logarithm of a with base b.
Returns the largest integer n such that b^n <= a.
Uses repeated squaring and binary search.

---

### is_square

```python
is_square(n: int) -> bool
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5056)

> Check if an integer n is a square.

---

### perfect_power

```python
perfect_power(n: int) -> tuple[int, int]
```
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5062)

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
[[source]](https://github.com/ini/numthy/blob/main/numthy.py#L5089)

> Given a monotonically increasing function f, find where it crosses a threshold.
Returns the smallest integer n in [low, high] such that f(n) >= threshold.

---
