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
    return (util.ExitStatus.SUCCESS, factor)


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
        if status == util.ExitStatus.SUCCESS:
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
    ax1.set_xlabel("integer factors found")
    ax1.set_ylabel("count")
    ax2.set_ylabel("count")

    plt.show()


def example_single_run(N=21, t=12, x=13, random_seed=14, plot=True):
    """Factorises 21 showing the quantum state of the first and second register.

    Highlights the measurements made in orange (probability of measuring a value determined by
    quantum state).

    Some iterations find multiples 7 and 3, others fail

    """
    numpy.random.seed(random_seed)
    measurements = quantum.measure_system(x, N, t, reps=1, plot=plot)
    factors, failed_factors, fail_reasons = find_factors(x, N, t, measurements)
    return factors


def example_statistics(N=7 * 5, t=12, reps=50):
    """Shows the statistics from iterating Shors algorithm to factorise N with t qubits.

    Summarise the factors found and how many failures were observed as a plot.
    """
    numpy.random.seed(200)
    main(N, t, reps=reps, plot_state=False, plot_summary=True)


def find_good_examples(N=21, t=12):
    """Simple loop over random seeds and random xs to find example inputs.

    These examples will find non trivial factors on the first try."""
    for x in range(1, N):
        for random_seed in range(0, 20):
            factors = example_single_run(N, t, x, random_seed, plot=False)
            if factors:
                print(N, t, x, random_seed, factors)
                break


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
    all_factors = []
    all_failed_factors = []
    all_fail_reasons = []
    possible_xs = range(1, N)  # 1 <= x <=N
    for x in numpy.random.permutation(possible_xs):  # try a random x
        print(f"trying to factorize N={N} using random x={x}")
        factor = classical.initial_checks(N, x)
        if factor is not None:
            continue
        measurements = quantum.measure_system(x, N, t, reps, plot=plot_state)
        factors, failed_factors, fail_reasons = find_factors(x, N, t, measurements)
        all_factors.extend(factors)
        all_failed_factors.extend(all_failed_factors)
        all_fail_reasons.extend(fail_reasons)
    if plot_summary:
        print(all_factors)
        plot_results(all_factors, all_failed_factors, all_fail_reasons)
    return all_factors


if __name__ == "__main__":
    # Example from blog post: 35 = 7 * 5
    print(example_single_run(N=35, x=8, random_seed=9, t=12))
    # statistics of algorithm on 35
    example_statistics(N=35, t=12)
