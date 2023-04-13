import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from typing import Literal, Callable
from nodal_basis_1d import NodalBasis1D
from iterative_lin_solve import IterativeLinSolve


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
        self._assemble_rhs_vector()
        self._assemble_stiffness_matrix()
        self._initialize_iterative_solver()

    def _assemble_rhs_vector(self):
        """
        Assemble RHS of the finite element weak form
        a(u, phi) = <f, phi> + phi * (du/dx(1) - du/dx(0))
        """
        self.basis = NodalBasis1D(
            basis_type=self.basis_func_type,
            hx=self.hx,
            rhs_func=self.rhs_func
        )

        if self.basis_func_type == "linear":
            self.rhs = np.zeros(self.num_element + 1)
            for i in range(self.num_element):
                self.rhs[i: i + 2] += self.basis.compute_inner_product(offset=i * self.hx)

            if self.left_boundary_type == "Dirichlet":
                self.rhs[1] += self.left_boundary_value

            if self.right_boundary_type == "Dirichlet":
                self.rhs[-2] += self.right_boundary_value

        if self.basis_func_type == "quadratic":
            self.rhs = np.zeros(2 * self.num_element + 1)
            for i in range(self.num_element):
                self.rhs[2 * i: 2 * i + 3] += self.basis.compute_inner_product(offset=i * self.hx)

            if self.left_boundary_type == "Dirichlet":
                self.rhs[1] += self.left_boundary_value * 8.0 / 3.0 / self.hx

            if self.right_boundary_type == "Dirichlet":
                self.rhs[-2] += self.right_boundary_value * 8.0 / 3.0 / self.hx

        if self.left_boundary_type == "Neumann":
            self.rhs[0] -= self.left_boundary_value

        if self.right_boundary_type == "Neumann":
            self.rhs[-1] += self.right_boundary_value

    def _initialize_fields(self):
        self.domain_x = np.linspace(
            0.0, self.domain_size, self.num_element + 1
        )
        self.end_index = {"left": 0, "right": None}
        self.soln_field = np.zeros_like(self.domain_x)
        self.dirichlet_boundary_count = 0

        if self.left_boundary_type == "Dirichlet":
            self.soln_field[0] = self.left_boundary_value
            self.dirichlet_boundary_count += 1
            self.end_index["left"] = 1

        if self.right_boundary_type == "Dirichlet":
            self.soln_field[-1] = self.right_boundary_value
            self.dirichlet_boundary_count += 1
            self.end_index["right"] = -1

    def _assemble_stiffness_matrix(self):
        if self.basis_func_type == "linear":    # Linear basis
            self.stiffness_matrix = sp.sparse.diags(
                diagonals=np.array([-1.0, 2.0, -1.0]) / self.hx,
                offsets=[-1, 0, 1],
                shape=(
                    self.num_element + 1 - self.dirichlet_boundary_count,
                    self.num_element + 1 - self.dirichlet_boundary_count,
                ),
            )

            if self.left_boundary_type == "Neumann":
                self.stiffness_matrix[0, 0] = 1.0
            if self.right_boundary_type == "Neumann":
                self.stiffness_matrix[-1, -1] = 1.0

        else:   # Quadratic basis
            local_stiffness = np.array([
                [7.0, -8.0, 1.0],
                [-8.0, 16.0, -8.0],
                [1.0, -8.0, 7.0],
            ]) / 3.0 / self.hx

            temp_stiffness_matrix = np.zeros(
                (
                    2 * self.num_element + 1,
                    2 * self.num_element + 1,
                )
            )

            # Assemble the stiffness matrix
            for i in range(self.num_element):
                temp_stiffness_matrix[2 * i:2 * i + 3, 2 * i:2 * i + 3] += local_stiffness

            self.stiffness_matrix = sp.sparse.csr_matrix(
                temp_stiffness_matrix[
                    self.end_index["left"]:self.end_index["right"],
                    self.end_index["left"]:self.end_index["right"],
                ]
            )

    def _initialize_iterative_solver(self):
        self.iter_solver = IterativeLinSolve(
            matrix=self.stiffness_matrix,
            solve_type=self.kwargs.get("solve_type", "w-jacobi"),
            weight=self.kwargs.get("weight", 1.0),
            rhs=self.rhs[self.end_index["left"]:self.end_index["right"]]
        )

    def solve(self):
        result = self.iter_solver.solve(
            tol=self.kwargs.get("tol", 1e-3),
            max_iter=self.kwargs.get("max_iter", 1e6)
        )
        if self.basis_func_type == "linear":
            self.soln_field[self.end_index["left"]:self.end_index["right"]] = result
        else:
            temp_soln_field = np.zeros(2 * self.num_element + 1)
            temp_soln_field[self.end_index["left"]:self.end_index["right"]] = result
            self.soln_field[...] = temp_soln_field[::2]

        return self.soln_field


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
    sine_modes = [1.0, 2.0, 4.5]
    rhs_func = lambda x: superimposed_sine_amp(x, modes=sine_modes)
    exact_soln = lambda x: superimposed_sine(x, modes=sine_modes)
    # rhs_func = lambda x: 8.0
    # exact_soln = lambda x: 4.0 * x * (1 - x)

    cls = FiniteElementPoisson1D(
        num_element=128,
        domain_size=1.0,
        rhs_func=rhs_func,
        left_boundary_type="Dirichlet",
        left_boundary_value=0.0,
        right_boundary_type="Dirichlet",
        right_boundary_value=0.0,
        basis_func_type="quadratic",
        solve_type="gauss-seidel",
        weight=0.8,
        max_iter=1e5,
        tol=1e-4,
    )
    num_soln = cls.solve()
    plt.plot(cls.domain_x, num_soln, label="fem")
    plt.plot(cls.domain_x, exact_soln(cls.domain_x), label="exact")
    plt.legend()
    plt.show()
