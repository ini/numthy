# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-28

### Changed

- **`small_roots()`**: The `epsilon` keyword argument has been replaced by `m: int | None`.
  `m` directly controls the number of Coppersmith shift polynomials. When omitted, the smallest
  sufficient `m` is chosen automatically. Code using the old `epsilon` parameter must be updated.

- **`solve_polynomial_system()`**: Now uses the GVW (Gao-Volny-Wang) algorithm over a finite
  field Fₚ to compute a Gröbner basis, followed by FGLM (Faugère-Gianni-Lazard-Mora) conversion
  to lexicographic order. This replaces the previous Buchberger computation over the rationals,
  which was significantly slower and less reliable for larger instances. The solver now also
  reduces the problem by extracting and solving any linear subsystem before invoking Gröbner
  basis machinery.

- **`prime_count()` and `prime_sum()`** (and related generalized prime counting functions):
  The P2 accumulation step is faster. Prime sums over sieve segments now use block-partitioned
  prefix sums, avoiding a full cumulative accumulation pass per segment.

### Fixed

- **`conic()`**: Degenerate conics with a non-square discriminant (b² − 4ac is not a perfect
  square) represent a pair of conjugate lines with no rational intersection. Previously, this
  case incorrectly delegated to the ellipse solver and could produce wrong output. Now the
  singular point is computed directly and returned only if it satisfies the equation.

- **`fibonacci(n, mod)`**: For negative `n`, the sign was applied after the modulo reduction
  (`sign * (F % mod)`) rather than before (`(sign * F) % mod`), yielding wrong results.

## [0.1.0] - 2026-01-31
