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
            for i in range(self.num_element):
                self.rhs_field[i: i + 2] += self.basis.compute_inner_product(offset=i * self.hx)

            if self.left_boundary_type == "Dirichlet":
                self.rhs_field[1] += self.left_boundary_value

            if self.right_boundary_type == "Dirichlet":
                self.rhs_field[-2] += self.right_boundary_value

        if self.basis_func_type == "quadratic":
            for i in range(self.num_element):
                self.rhs_field[2 * i: 2 * i + 3] += self.basis.compute_inner_product(offset=i * self.hx)

            if self.left_boundary_type == "Dirichlet":
                self.rhs_field[1] += self.left_boundary_value * 8.0 / 3.0 / self.hx

            if self.right_boundary_type == "Dirichlet":
                self.rhs_field[-2] += self.right_boundary_value * 8.0 / 3.0 / self.hx

        if self.left_boundary_type == "Neumann":
            self.rhs_field[0] -= self.left_boundary_value

        if self.right_boundary_type == "Neumann":
            self.rhs_field[-1] += self.right_boundary_value

    def _initialize_fields(self):
        self.num_iter = 0
        self.dirichlet_boundary_count = 0
        self.end_index = {"left": 0, "right": None}

        if self.basis_func_type == "linear":
            self.domain_x = np.linspace(
                0.0, self.domain_size, self.num_element + 1
            )
        else:
            self.domain_x = np.linspace(
                0.0, self.domain_size, 2 * self.num_element + 1
            )

        self.soln_field = np.zeros_like(self.domain_x)
        self.rhs_field = np.zeros_like(self.domain_x)
        self.res_field = np.zeros_like(self.domain_x)

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
        self._iter_solver = IterativeLinSolve(
            matrix=self.stiffness_matrix,
            solve_type=self.kwargs.get("solve_type", "w-jacobi"),
            weight=self.kwargs.get("weight", 1.0),
            rhs=self.rhs_field[self.end_index["left"]:self.end_index["right"]]
        )

    def solve(self, tol: float = 1e-3, max_iter: int = 1e6):
        result, total_iter = self._iter_solver.solve(tol=tol, max_iter=max_iter)
        self.num_iter += total_iter

        self.soln_field[self.end_index["left"]:self.end_index["right"]] = result
        self.res_field[self.end_index["left"]:self.end_index["right"]] = self._iter_solver.res.copy()

        return self.soln_field
