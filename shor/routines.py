import enum
import numpy
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

import shor.classical as classical
import shor.quantum as quantum
import shor.util as util


def _classical_routine_on_result(
    x, N, t, measurement
) -> Tuple[enum.Enum, Optional[Tuple[int, ...]]]:
    """Try to find factors, given x,N,t and the result of a single quantum measurement."""
    try:
        continued_fraction_ints = classical.continued_fraction(measurement, 2 ** t)
        convergents_vals = classical.convergents(continued_fraction_ints)
        rs = classical.possible_orders(convergents_vals)
        r = classical.first_order(rs, x, N)
        factor = classical.find_factor_from_order(r, x, N)
    except util.ShorError as e:
        if e.fail_reason == util.ExitStatus.FAILED_FACTOR:
            return (util.ExitStatus.FAILED_FACTOR, e.failed_factors)
        return e.fail_reason, None
    return (util.ExitStatus.NO_FAIL, factor)


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
        if status == util.ExitStatus.NO_FAIL:
            factors.extend(factor)
        elif status == util.ExitStatus.FAILED_FACTOR:
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
    reason_names = [str(util.ExitStatus(reason)) for reason in reasons]
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
    measurements = quantum.measure_system(x, N, t, reps=2, plot=True)
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
        factor = classical.initial_checks(N, x)
        if factor is not None:
            continue
        measurements = quantum.measure_system(x, N, t, reps, plot=plot_state)
        factors, failed_factors, fail_reasons = find_factors(x, N, t, measurements)
        if len(factors) > 0:
            if plot_summary:
                print(factors)
                plot_results(factors, failed_factors, fail_reasons)
            return factors


if __name__ == "__main__":
    example_single_run()
    example_statistics()
    # t = 19
    # N = 53 * 59
    # N = 7 * 3
    # main(N, t, plot_summary=True)
