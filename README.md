<h1 align="center" style="margin-top: 24px;">
  <img src="https://i.imgur.com/IjaDdYO.png" alt="NumThy" width="256">
</h1>

Computational number theory. Pure Python. Zero dependencies. Unreasonably fast.

## Installation

```bash
pip install numthy
```

Or just drop [`numthy.py`](https://raw.githubusercontent.com/ini/numthy/main/numthy.py) into your project.

## Quick Start

```python
import numthy as nt

# Primality
nt.is_prime(2**89 - 1)  # True

# Factorization (SIQS handles 50+ digits)
nt.prime_factors(2**128 + 1)  # (59649589127497217, 5704689200685129054721)

# Prime counting
nt.count_primes(10**9)  # 50847534

# Discrete log
nt.discrete_log(1000, 3, 65537)  # 50921 (i.e., 1000 ≡ 3^50921 mod 65537)

# Diophantine equations
# Solve 2x² + 3xy - 3y² + 7x - 10y - 24 = 0
solutions = nt.conic(2, 3, -3, 7, -10, -24)
next(solutions)  # (8, 10)
next(solutions)  # (376, -174)
next(solutions)  # (17304, 25218)
```

## Demo

Try NumThy in the browser: [ini.github.io/numthy/demo](https://ini.github.io/numthy/demo)

## Documentation

See [API.md](API.md) for the full reference.

## Under The Hood

One file, with everything implemented from scratch. Simple API, with heavy-duty algorithms under the hood:

* Extra-strong variant of the Baillie-PSW primality test
* Lagarias-Miller-Odlyzko (LMO) algorithm for prime counting, generalized to sums over primes of any arbitrary completely multiplicative function
* Two-stage Lenstra's ECM factorization with Montgomery curves and Suyama parametrization
* Self-initializing quadratic sieve (SIQS) with triple-large-prime variation
* Cantor-Zassenhaus → Hensel lifting → Chinese Remainder Theorem pipeline for finding modular roots of polynomials
* Adleman-Manders-Miller algorithm for general n-th roots over finite fields
* General solver for all binary quadratic Diophantine equations (ax² + bxy + cy² + dx + ey + f = 0)
* Lenstra–Lenstra–Lovász lattice basis reduction algorithm with automatic precision escalation
* Jochemsz-May generalization of Coppersmith's method for multivariate polynomials with any number of variables

## Requirements

Python 3.10+

That's it.

## License

[MIT](LICENSE.md)
