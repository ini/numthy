#!/usr/bin/env python3
"""
Natural language CLI for numthy.

Ask number theory questions in plain English and get answers.

Usage:
    numthy "Is 2^89 - 1 prime?"
    numthy "Factor 2^64 + 1"
    numthy "Find the first 20 primes"
    numthy "Solve x^2 - 2y^2 = 1"
"""

import argparse
import sys
import os

try:
    import anthropic
except ImportError:
    anthropic = None

import numthy as nt

SYSTEM_PROMPT = """You are a number theory assistant. You translate natural language questions into numthy Python code.

Available functions (import numthy as nt):

PRIMES:
- nt.is_prime(n) -> bool: Test if n is prime
- nt.next_prime(n) -> int: Smallest prime greater than n
- nt.primes(low=2, high=None, count=None) -> Iterator[int]: Generate primes
- nt.count_primes(x) -> int: Count primes <= x
- nt.prime_factors(n) -> tuple[int, ...]: Prime factors with multiplicity
- nt.prime_factorization(n) -> dict[int, int]: {prime: exponent}
- nt.divisors(n) -> tuple[int, ...]: All divisors of n

ARITHMETIC FUNCTIONS:
- nt.totient(n) -> int: Euler's totient φ(n)
- nt.mobius(n) -> int: Möbius function μ(n)
- nt.divisor_function(n, k=1) -> int: σ_k(n), sum of k-th powers of divisors

MODULAR ARITHMETIC:
- nt.egcd(a, b) -> tuple[int, int, int]: Extended GCD (g, x, y) where ax + by = g
- nt.crt(remainders, moduli) -> int: Chinese Remainder Theorem
- nt.multiplicative_order(a, n) -> int: Smallest k where a^k ≡ 1 (mod n)
- nt.primitive_root(n) -> int: Smallest primitive root mod n
- nt.discrete_log(a, b, mod) -> int: Find x where a^x ≡ b (mod mod)
- nt.modular_roots(n, k, mod) -> tuple[int, ...]: Solutions to x^k ≡ n (mod mod)
- nt.legendre(a, p) -> int: Legendre symbol (a/p)
- nt.jacobi(a, n) -> int: Jacobi symbol (a/n)

DIOPHANTINE EQUATIONS:
- nt.pell(d) -> Iterator[tuple[int, int]]: Solutions to x² - d·y² = 1
- nt.cornacchia(d, p) -> tuple[int, int] | None: Solutions to x² + d·y² = p
- nt.bezout(a, b, c) -> Iterator[tuple[int, int]]: Solutions to ax + by = c
- nt.pythagorean_triples(max_c) -> Iterator: Primitive Pythagorean triples

SEQUENCES:
- nt.fibonacci(n) -> int: n-th Fibonacci number
- nt.lucas(n) -> int: n-th Lucas number
- nt.partition(n) -> int: n-th partition number p(n)
- nt.partition_numbers(mod=None) -> Iterator[int]: Partition numbers p(0), p(1), ...

UTILITIES:
- nt.iroot(x, n=2) -> int: Integer n-th root (floor)
- nt.ilog(a, b=2) -> int: Integer log base b
- nt.is_square(n) -> bool: Check if n is a perfect square
- nt.perfect_power(n) -> tuple[int, int]: Find (a, b) where a^b = n

Respond with ONLY Python code that answers the question. The code should:
1. Import nothing (numthy is already imported as nt)
2. Store the result in a variable called `result`
3. Be concise - usually 1-3 lines

Examples:
User: "Is 2^89 - 1 prime?"
result = nt.is_prime(2**89 - 1)

User: "Factor 2^64 + 1"
result = nt.prime_factors(2**64 + 1)

User: "Find the first 20 primes"
result = list(nt.primes(count=20))

User: "Solve x^2 - 2y^2 = 1"
solutions = nt.pell(2)
result = [next(solutions) for _ in range(5)]

User: "What are the divisors of 60?"
result = nt.divisors(60)
"""


def run_with_claude(query: str) -> str:
    """Use Claude to translate query to numthy code and execute it."""
    if anthropic is None:
        return "Error: anthropic package not installed. Run: pip install anthropic"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY environment variable not set"

    client = anthropic.Anthropic(api_key=api_key)

    # Get code from Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}]
    )

    code = response.content[0].text.strip()

    # Clean up code (remove markdown fences if present)
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    # Execute the code
    try:
        local_vars = {"nt": nt}
        exec(code, {"nt": nt, "__builtins__": __builtins__}, local_vars)
        result = local_vars.get("result", "No result variable set")
        return f"{result}"
    except Exception as e:
        return f"Error executing code: {e}\n\nGenerated code:\n{code}"


def run_direct(expr: str) -> str:
    """Try to directly evaluate a simple expression."""
    try:
        # Build namespace with all numthy functions available directly
        namespace = {name: getattr(nt, name) for name in nt.__all__}
        namespace["nt"] = nt
        namespace["__builtins__"] = {}

        result = eval(expr, namespace)
        return str(result)
    except:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Natural language interface to numthy",
        epilog="Examples:\n"
               "  numthy 'Is 2^89-1 prime?'\n"
               "  numthy 'Factor 100'\n"
               "  numthy 'nt.primes(count=10)'",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query", nargs="*", help="Question or expression")
    parser.add_argument("--code", action="store_true", help="Show generated code")

    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        return

    query = " ".join(args.query)

    # Try direct evaluation first (for expressions like "is_prime(17)" or "nt.is_prime(17)")
    result = run_direct(query)
    if result:
        print(result)
        return

    # Use Claude for natural language
    result = run_with_claude(query)
    print(result)


if __name__ == "__main__":
    main()
