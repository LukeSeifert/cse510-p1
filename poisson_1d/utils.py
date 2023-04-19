import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from typing import Callable


def gauss_quadrature(order: int, hx: float):
    # Abscissas on (-1, 1)
    abscissas, weights = roots_legendre(order)

    # Linear transform from (-1, 1) to (0, hx)
    abscissas = (abscissas + 1) * (0.5 * hx)
    weights *= 0.5 * hx

    def compute_integral(func: Callable, offset=0.0) -> float:
        return float(
            np.sum(weights * func(offset + abscissas))
        )

    return compute_integral


def lap_superimposed_sine(x, modes: list[float]):
    func_vals = np.zeros_like(x)
    for k in modes:
        func_vals += (2.0 * np.pi * k) ** 2 * np.sin(2.0 * np.pi * k * x)

    return func_vals


def superimposed_sine(x, modes: list[float]):
    func_vals = np.zeros_like(x)
    for k in modes:
        func_vals += np.sin(2.0 * np.pi * k * x)

    return func_vals


def logistic_map(x):
    return 4.0 * x * (1.0 - x)


def neg_lap_logistic_map(x):
    return np.ones_like(x) * 8.0


def format_plot():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = "True"
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["lines.markersize"] = 5
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.grid.which"] = "both"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 1
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 6.0
    plt.rcParams["ytick.major.size"] = 6.0
    plt.rcParams["xtick.minor.size"] = 3.0
    plt.rcParams["ytick.minor.size"] = 3.0
    plt.rcParams["figure.autolayout"] = True