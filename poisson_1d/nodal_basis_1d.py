import numpy as np
from typing import Callable
from utils import gauss_quadrature


class NodalBasis1D:
    def __init__(self, basis_type: str, hx: float, rhs_func: Callable):
        self.basis_type = basis_type
        self.hx = hx
        self.rhs_func = rhs_func
        self.basis_funcs: list[Callable] = []

        if basis_type == "linear":
            self.basis_funcs.append(lambda x: 1.0 - 1.0 / hx * x)
            self.basis_funcs.append(lambda x: 1.0 / hx * x)
            self.quadrature = gauss_quadrature(order=2, hx=hx)

        elif basis_type == "quadratic":
            self.basis_funcs.append(
                lambda x: 2.0 * (x - 0.5 * hx) * (x - hx) / hx ** 2
            )
            self.basis_funcs.append(
                lambda x: 4.0 * x * (hx - x) / hx ** 2
            )
            self.basis_funcs.append(
                lambda x: 2.0 * x * (x - 0.5 * hx) / hx ** 2
            )
            self.quadrature = gauss_quadrature(order=3, hx=hx)

        else:
            raise ValueError("Unrecognized basis type")

    def compute_inner_product(self, offset: float) -> np.ndarray:
        to_return = []
        for basis_func in self.basis_funcs:
            # Compute <phi, f> on (0, hx)
            to_return.append(
                self.quadrature(
                    func=lambda x: basis_func(x) * self.rhs_func(x + offset),
                )
            )

        return np.array(to_return)
