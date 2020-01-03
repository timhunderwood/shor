from typing import Tuple

import numpy
from matplotlib import pyplot as plt


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
        ax0.set_ylabel("probability")
        ax1.set_ylabel("state amplitude")
        ax2.set_ylabel("probability")
        ax2.set_xlabel("basis state")
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
