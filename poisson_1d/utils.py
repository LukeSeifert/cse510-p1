import numpy as np
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
