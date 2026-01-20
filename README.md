<h1 align="center" style="margin-top: 24px;">
  <img src="https://i.imgur.com/IjaDdYO.png" alt="NumThy" width="256">
</h1>

Computational number theory. Pure Python. Zero dependencies. Unreasonably fast.

## Installation

```bash
pip install numthy
```

Or save [`numthy.py`](https://raw.githubusercontent.com/ini/numthy/main/numthy.py) directly into your project.

## Quick Start

```python
import numthy as nt

# Primality testing
nt.is_prime(2**89 - 1)  # True (Mersenne prime)

# Factorization
nt.prime_factors(2**64 + 1)  # (274177, 67280421310721)

# Modular arithmetic
nt.discrete_log(3, 1000, 2**16 + 1)  # 50921

# Diophantine equations
solutions = nt.pell(2)
next(solutions)  # (3, 2) since 3² - 2·2² = 1
```

## Highlights

The same algorithms used by Mathematica, SageMath, and research-grade CAS systems — implemented from scratch in pure Python.

- **Primality** — Miller-Rabin, Baillie-PSW, Lucas-Lehmer
- **Factorization** — Brent's variant of Pollard's rho, Lenstra's ECM (Elliptic Curve Method), Self-Initializing Quadratic Sieve with up to 3 large primes
- **Prime counting** — Lagarias-Miller-Odlyzko combinatorial method
- **Discrete log** — Pohlig-Hellman, Baby-step giant-step, Pollard rho
- **Modular roots** — Tonelli-Shanks, Adleman-Manders-Miller for arbitary k-th roots, Hensel lifting with both simple and multiple roots
- **Diophantine** — Cornacchia, Pell solver, binary quadratic forms, Pillai's equation
- **Lattices** — LLL reduction, Babai's nearest plane, Smith normal form

## Documentation

See [API.md](API.md) for the full API reference.

## Requirements

Python 3.10+

That's it.

## License

[Free to use, copy, modify, and redistribute with attribution.](LICENSE.md)
