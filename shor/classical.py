from typing import Tuple

import numpy
import shor.util as util


def continued_fraction(numerator: int, denominator: int) -> Tuple[int, ...]:
    """Find the integer continued fraction representation of the rational measured."""
    if denominator == 1:
        return (numerator,)
    if denominator == 0:
        return ()
    return (numerator // denominator,) + continued_fraction(
        denominator, numerator % denominator
    )


def convergent(continued_fractions: Tuple[int, ...],) -> Tuple[int, int]:
    """Get the convergent for a given continued fractions sequence of integers"""
    num, den = 1, 0
    for u in reversed(continued_fractions):
        num, den = den + num * u, num
    return num, den


def convergents(continued_fractions: Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
    """Get all the convergents possible given the complete continued fraction.
    Theorem 5.1 means that one of these must be the true fraction, excluding the error
    and the denominator can be used for r in our order finding.
    """
    return tuple(
        convergent(continued_fractions[:m]) for m in range(len(continued_fractions) + 1)
    )


def possible_orders(convergents_vals: Tuple[Tuple[int, int], ...]) -> Tuple[int, ...]:
    """Given the (numerator, denominator) pairs, extract the denominators which represent the possible
    order r and do not include trivial cases of 0 or 1."""
    return tuple(c[1] for c in convergents_vals if (c[1] != 0 and c[1] != 1))


def is_order(r: int, x: int, N: int) -> bool:
    """Return True if r is the order of x % N"""
    return ((x ** r) % N) == 1


def first_order(possible_orders: Tuple[int, ...], x: int, N: int) -> int:
    """Get the first true order in the list of possible orders."""
    for r in possible_orders:
        if is_order(r, x, N):
            return r
    raise util.ShorError(
        f"In list of possible orders no order of N was "
        f"found (x,N){x,N}, possible_orders={possible_orders}",
        fail_reason=util.ExitStatus.NO_ORDER_FOUND,
    )


def find_factor_from_order(r: int, x: int, N: int) -> Tuple[int, ...]:
    """Given an order r of x % N, try to find non trivial factors of N."""
    if r % 2 == 0 and (x ** (r // 2) % N != N - 1):
        gcd_1 = numpy.gcd(x ** (r // 2) - 1, N)
        gcd_2 = numpy.gcd(x ** (r // 2) + 1, N)
        factors = tuple(
            gcd for gcd in (gcd_1, gcd_2) if (N % gcd == 0 and gcd != 1 and gcd != N)
        )
        if len(factors) > 0:
            return factors
        raise util.ShorError(
            f"Neither gcd_1={gcd_1} nor gcd_2={gcd_2} were factors of N.",
            failed_factor=(gcd_1, gcd_2),
            fail_reason=util.ExitStatus.FAILED_FACTOR,
        )
    raise util.ShorError(
        f"For (r, x,N)={r,x,N}: r%2==0 and (x**(r//2)%N==N-1) was False",
        fail_reason=util.ExitStatus.ORDER_CONDITION,
    )


def initial_checks(N, x):
    """Perform simple initial checks on N and x to see if factors of N can be found classically and quickly."""
    if N % 2 == 0:
        print(f"N={N} was even, try factorising N/2={N//2} instead")
        return 2
    gcd = numpy.gcd(x, N)
    if gcd > 1:
        print(
            f"Instantly found a factor classically gcd={gcd}. Factorize N/{gcd}={N//gcd}"
        )
        return gcd
