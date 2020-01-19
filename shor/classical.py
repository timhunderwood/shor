from typing import Tuple, Optional

import numpy
import shor.util as util


def continued_fraction(numerator: int, denominator: int) -> Tuple[int, ...]:
    """Find the integer continued fraction representation of the rational measured.

    :param numerator: numerator of continued fraction
    :param denominator: denominator of continued fraction
    :return: Tuple representation of continue fraction
    """
    if denominator == 1:
        return (numerator,)
    if denominator == 0:
        return ()
    return (numerator // denominator,) + continued_fraction(
        denominator, numerator % denominator
    )


def _convergent(continued_fractions: Tuple[int, ...],) -> Tuple[int, int]:
    """Get the convergent for a given continued fractions sequence of integers

    :param continued_fractions: Tuple representation of continued fraction
    :return: reduced continued fraction representation of convergent
    """
    num, den = 1, 0
    for u in reversed(continued_fractions):
        num, den = den + num * u, num
    return num, den


def _convergents(continued_fractions: Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
    """Get all the convergents possible given the complete continued fraction.

    One of these must be the true fraction, excluding the error.

    :param continued_fractions:
    :return: Tuple of all convergents found (each convergent is a Tuple[int,int])
    """
    return tuple(
        _convergent(continued_fractions[:m])
        for m in range(len(continued_fractions) + 1)
    )


def possible_orders(convergents_vals: Tuple[Tuple[int, int], ...]) -> Tuple[int, ...]:
    """Given the (numerator, denominator) pairs, extract the denominators which represent the possible
    order r and do not include trivial cases of 0 or 1.

    :param convergents_vals:
    :return: Tuple of possible orders
    """
    return tuple(c[1] for c in convergents_vals if (c[1] != 0 and c[1] != 1))


def _is_order(N: int, x: int, r: int) -> bool:
    """Return True if r is the order of x % N

    :param N: number to factorise
    :param x: random integer 0 < x < N
    :param r: integer we are testing is the order of x mod N
    :return:
    """
    return ((x ** r) % N) == 1


def first_order(N: int, x: int, possible_orders: Tuple[int, ...]) -> int:
    """Get the first true order in the list of possible orders.

    :param N: number to factorise
    :param x: random integer 0 < x < N
    :param possible_orders: Tuple of integer possible orders
    :return: first r in possible orders that is a true order of x mod N
    """
    for r in possible_orders:
        if _is_order(N, x, r):
            return r
    raise util.ShorError(
        f"In list of possible orders no order of N was "
        f"found (x,N){x,N}, possible_orders={possible_orders}",
        fail_reason=util.ExitStatus.NO_ORDER_FOUND,
    )


def find_factor_from_order(N: int, x: int, r: int) -> Tuple[int, ...]:
    """Given an order r of x % N, try to find non trivial factors of N.

    :param N: number to factorise
    :param x: random integer 0 < x < N
    :param r: order of x mod N
    :return:
    """
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


def initial_checks(N: int, x: int) -> Optional[int]:
    """Perform simple initial checks on N and x to see if factors of N can be found classically and quickly.

    :param N: number to factorise
    :param x: random integer 0 < x < N
    :return: trivial factors
    """
    if N % 2 == 0:
        print(f"N={N} was even, try factorising N/2={N//2} instead")
        return 2
    gcd = numpy.gcd(x, N)
    if gcd > 1:
        print(
            f"Instantly found a factor classically gcd={gcd}. Factorize N/{gcd}={N//gcd}"
        )
        return gcd
