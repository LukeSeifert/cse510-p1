import numpy as np
from typing import Literal, Callable
from stencil_weighted_jacobi_1d import StencilWeightedJacobiIteration1D
import matplotlib.pyplot as plt


class FiniteElementPoisson1D:
    def __init__(
        self,
        num_element: int,
        domain_size: float,
        rhs_func: Callable,
        left_boundary_type: Literal["Dirichlet", "Neumann"] = "Dirichlet",
        left_boundary_value: float = 0.0,
        right_boundary_type: Literal["Dirichlet", "Neumann"] = "Dirichlet",
        right_boundary_value: float = 0.0,
        basis_func_type: Literal["linear", "quadratic"] = "linear",
        **kwargs,
    ):
        self.num_element = num_element
        self.domain_size = domain_size
        self.rhs_func = rhs_func
        self.basis_func_type = basis_func_type
        self.hx = domain_size / num_element
        self.left_boundary_type = left_boundary_type
        self.right_boundary_type = right_boundary_type
        self.left_boundary_value = left_boundary_value
        self.right_boundary_value = right_boundary_value
        self.kwargs = kwargs
        self._initialize_fields()
        self._initialize_iterative_solver()

    def _construct_rhs_vector_linear(self):
        nodal_values = self.rhs_func(self.domain_x)
        self.rhs = np.zeros(self.num_element + 1)
        self.rhs[1:-1] = self.hx ** 2 * nodal_values[1:-1]
        self.rhs[0] = self.hx ** 2 / 2 * self.rhs_func(self.domain_x[0])
        self.rhs[-1] = self.hx ** 2 / 2 * self.rhs_func(self.domain_x[-1])

    def _initialize_fields(self):
        self.domain_x = np.linspace(
            0.0, self.domain_size, self.num_element + 1
        )
        self.soln_field = np.zeros_like(self.domain_x)
        if self.left_boundary_type == "Dirichlet":
            self.soln_field[0] = self.left_boundary_value
        if self.right_boundary_type == "Dirichlet":
            self.soln_field[-1] = self.right_boundary_value

        if self.basis_func_type == "linear":
            self._construct_rhs_vector_linear()

    def _initialize_iterative_solver(self):
        self.iter_solver = StencilWeightedJacobiIteration1D(
            rhs_field=self.rhs,
            stencil=np.array([1.0, -2.0, 1.0]),
            boundary_conditions=(self.left_boundary_type, self.right_boundary_type),
            boundary_values=(self.left_boundary_value, self.right_boundary_value),
            weight=self.kwargs.get("weight", 1.0),
            tol=self.kwargs.get("tol", 1e-5),
        )

    def solve(self):
        result = self.iter_solver.solve()
        print(f"Total number of iterations: {self.iter_solver.num_iter}")
        return result


def superimposed_sine_amp(x, modes: list[float]):
    func_vals = np.zeros_like(x)
    for k in modes:
        func_vals += (2.0 * np.pi * k) ** 2 * np.sin(2.0 * np.pi * k * x)

    return func_vals


def superimposed_sine(x, modes: list[float]):
    func_vals = np.zeros_like(x)
    for k in modes:
        func_vals += np.sin(2.0 * np.pi * k * x)

    return func_vals


if __name__ == "__main__":
    sine_modes = [1.0, 2.0, 4.0]
    rhs_func = lambda x: superimposed_sine_amp(x, modes=sine_modes)
    exact_soln = lambda x: superimposed_sine(x, modes=sine_modes) + 1

    cls = FiniteElementPoisson1D(
        num_element=128,
        domain_size=1.0,
        rhs_func=rhs_func,
        left_boundary_type="Dirichlet",
        left_boundary_value=1.0,
        right_boundary_type="Dirichlet",
        right_boundary_value=1.0
    )
    num_soln = cls.solve()
    plt.plot(cls.domain_x, num_soln, label="fem")
    plt.plot(cls.domain_x, exact_soln(cls.domain_x), label="exact")
    plt.legend()
    plt.show()