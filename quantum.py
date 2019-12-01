#
# first_register = qutip.qip.qubit_states(N=t)
#
# for i in range(t):
#     first_register = qutip.qip.snot(N=t, target=i) * first_register
import enum
import numpy
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class ExitStatus(enum.IntEnum):
    """Enum to track the Exit Status of an iteration of Shor's algorithm"""

    NO_ORDER_FOUND = enum.auto()
    ORDER_CONDITION = enum.auto()
    FAILED_FACTOR = enum.auto()
    NO_FAIL = enum.auto()


class ShorError(Exception):
    """Exception that can be raised when Shor's algorithm fails during an iteration."""

    def __init__(self, message, fail_reason, failed_factor=None):
        self.message = message
        self.failed_factors = failed_factor
        self.fail_reason: enum.Enum = fail_reason


# Classical
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
    raise ShorError(
        f"In list of possible orders no order of N was "
        f"found (x,N){x,N}, possible_orders={possible_orders}",
        fail_reason=ExitStatus.NO_ORDER_FOUND,
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
        raise ShorError(
            f"Neither gcd_1={gcd_1} nor gcd_2={gcd_2} were factors of N.",
            failed_factor=(gcd_1, gcd_2),
            fail_reason=ExitStatus.FAILED_FACTOR,
        )
    raise ShorError(
        f"For (r, x,N)={r,x,N}: r%2==0 and (x**(r//2)%N==N-1) was False",
        fail_reason=ExitStatus.ORDER_CONDITION,
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


# Quantum
def y_ks_ifft(x_js: numpy.ndarray) -> numpy.ndarray:
    """Quantum fourier transform is the discrete FFT on the quantum state.

    Matching sign convention and notation of Nielsen and Chuang."""
    return numpy.fft.ifft(x_js, norm="ortho")


def measure(probability_array: numpy.ndarray) -> int:
    """Given the cumulative probability values for states [0, 1, 2... , n], get state according to CDF.

    PDF of [0.1, 0.2,0.0,0.3,0.4]
    gives cum_probability_array e.g. [0.1, 0.3, 0.3, 0.6, 1. ]
    will return 0 , 1, 3 or 4 with 4 being twice as likely as 1.

    :param cum_probability_array:
    :return:
    """
    index = numpy.random.random()
    cum_probability_array = probability_array.cumsum()
    return numpy.searchsorted(cum_probability_array, index, side="left")


def _initial_state(x, N, t) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generate a representation of the first and second register for Shor's algorithm."""
    first_register = numpy.ones(2 ** t)
    second_register = numpy.array([(x ** k) % N for k in range(0, 2 ** t)])
    return first_register, second_register


def _measure_system(
    first_register: numpy.ndarray,
    second_register: numpy.ndarray,
    normalisation: float,
    plot: bool,
) -> int:
    """Perform a measurement, given the initial state of the first a second register for a given x,N,t combination.

    :param first_register:
    :param second_register:
    :param normalisation:
    :param plot:
    :return:
    """
    first_register = first_register.copy()
    second_register = second_register.copy()

    measurement_0 = numpy.random.choice(second_register)
    # print(f"measurement of 2nd register = {measurement_0}")
    mask: numpy.ndarray = second_register != measurement_0
    first_register[mask] = 0
    normalisation *= numpy.sqrt(
        first_register.size / (~mask).sum()
    )  # normalisation changes as now many states are zero

    state_1 = y_ks_ifft(normalisation * first_register)
    state_1_pdf = numpy.absolute(state_1) ** 2

    measured_value = measure(state_1_pdf)
    # print(f"Measured value (for phase approximation) = {measured_value}")

    if plot:
        general_color = "#1f77b4"
        measured_color = "#ff7f0e"
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
        bin_count = numpy.bincount(second_register)
        colors = numpy.full(bin_count.shape, general_color)
        colors[measurement_0] = measured_color
        ax0.bar(x=numpy.arange(bin_count.size), height=bin_count, color=colors)
        ax1.plot(numpy.absolute(first_register) ** 2)
        ax2.plot(numpy.absolute(state_1) ** 2)
        ax2.axvline(measured_value, color=measured_color)
        plt.show()

    return measured_value


def measure_system(x, N, t, reps=1, plot=False) -> Tuple[int, ...]:
    """Create an initial state and simulate reps measurements on it.

    :param x:
    :param N:
    :param t:
    :param reps:
    :param plot:
    :return:
    """
    first_register, second_register = _initial_state(x, N, t)
    return tuple(
        _measure_system(
            first_register, second_register, normalisation=2 ** (-t / 2), plot=plot
        )
        for _ in range(reps)
    )


def _classical_routine_on_result(
    x, N, t, measurement
) -> Tuple[enum.Enum, Optional[Tuple[int, ...]]]:
    """Try to find factors, given x,N,t and the result of a single quantum measurement."""
    try:
        continued_fraction_ints = continued_fraction(measurement, 2 ** t)
        convergents_vals = convergents(continued_fraction_ints)
        rs = possible_orders(convergents_vals)
        r = first_order(rs, x, N)
        factor = find_factor_from_order(r, x, N)
    except ShorError as e:
        if e.fail_reason == ExitStatus.FAILED_FACTOR:
            return (ExitStatus.FAILED_FACTOR, e.failed_factors)
        return e.fail_reason, None
    return (ExitStatus.NO_FAIL, factor)


def find_factors(
    x, N, t, measurements: Tuple[int, ...]
) -> Tuple[List[int], List[int], List[enum.Enum]]:
    """Try to find factors, given x,N,t and a list of measurements for a system x,N,t

    :param x:
    :param N:
    :param t:
    :param measurements:
    :return:
    """
    factors: List[int] = []
    failed_factors: List[int] = []
    fail_reasons: List[enum.Enum] = []
    for measurement in measurements:
        (status, factor) = _classical_routine_on_result(x, N, t, measurement)
        # print(f"status = {status} and factor = {factor}")
        if status == ExitStatus.NO_FAIL:
            factors.extend(factor)
        elif status == ExitStatus.FAILED_FACTOR:
            failed_factors.extend(factor)
        fail_reasons.append(status)
    return factors, failed_factors, fail_reasons


def plot_results(factors, failed_factors, fail_reasons):
    """Produce summary plots from the output of find_factors

    :param factors:
    :param failed_factors:
    :param fail_reasons:
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    correct_color = "#1f77b4"
    failed_color = "#ff7f0e"

    bin_count_factors = numpy.bincount(factors)
    bin_count_failed_factors = numpy.bincount(failed_factors)
    ax1.bar(
        x=numpy.arange(bin_count_factors.size),
        height=bin_count_factors,
        color=correct_color,
    )
    ax1.bar(
        x=numpy.arange(bin_count_failed_factors.size),
        height=bin_count_failed_factors,
        color=failed_color,
    )
    reasons, counts = numpy.unique(fail_reasons, return_counts=True)

    ax2.bar(x=numpy.arange(reasons.size), height=counts)
    reason_names = [str(ExitStatus(reason)) for reason in reasons]
    ax2.set_xticks(numpy.arange(reasons.size))
    ax2.set_xticklabels(reason_names)

    plt.show()


def example_single_run(N=21, t=12):
    """Factorises 21 showing the quantum state of the first and second register.

    Highlights the measurements made in orange (probability of measuring a value determined by
    quantum state).

    Some iterations find multiples 7 and 3, others fail

    """
    numpy.random.seed(14)
    x = 13
    measurements = measure_system(x, N, t, reps=2, plot=True)
    factors, failed_factors, fail_reasons = find_factors(x, N, t, measurements)
    print(factors)
    return factors


def example_statistics(N=7 * 13, t=12):
    """Shows the statistics from iterating Shors algorithm to factorise N with t qubits.

    Summarise the factors found and how many failures were observed as a plot.
    """
    numpy.random.seed(2)
    main(N, t, plot_state=False, plot_summary=True)


def main(N, t, reps=100, plot_state=False, plot_summary=True):
    """Factorise N with t qubits repeating for each random trialled x value reps times.

    Plots a summary of the results and success rate.

    :param N:
    :param t:
    :param reps:
    :param plot_state:
    :param plot_summary:
    :return:
    """
    possible_xs = range(1, N)  # 1 <= x <=N
    for x in numpy.random.permutation(possible_xs):  # try a random x
        print(f"trying to factorize N={N} using random x={x}")
        factor = initial_checks(N, x)
        if factor is not None:
            continue
        measurements = measure_system(x, N, t, reps, plot=plot_state)
        factors, failed_factors, fail_reasons = find_factors(x, N, t, measurements)
        if len(factors) > 0:
            if plot_summary:
                print(factors)
                plot_results(factors, failed_factors, fail_reasons)
            return factors


if __name__ == "__main__":
    # example_single_run()
    # example_statistics()
    t = 19
    N = 53 * 59
    N = 7 * 3
    main(N, t, plot_summary=True)
