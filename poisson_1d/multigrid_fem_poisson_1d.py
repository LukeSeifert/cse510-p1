import numpy as np
import scipy.sparse as spp
import scipy.sparse.linalg as sla
from typing import Literal, Callable

from fem_poisson_1d import FiniteElementPoisson1D
from iterative_lin_solve import IterativeLinSolve


class MultigridFEMPoisson1D:
    def __init__(
        self,
        multigrid_levels: int,
        num_element: int,
        domain_size: float,
        rhs_func: Callable,
        basis_func_type: Literal["linear", "quadratic"] = "linear",
        boundary_conditions: tuple[
            Literal["Dirichlet", "Neumann"], Literal["Dirichlet", "Neumann"]
        ] = ("Dirichlet", "Dirichlet"),
        boundary_values: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ):
        assert num_element % 2 ** (multigrid_levels - 1) == 0, ValueError(
            "Number of elements must be a multiple of (multigrid_levels - 1)"
        )
        self.multigrid_levels = multigrid_levels
        self.num_elements = num_element
        self.domain_size = domain_size
        self.rhs_func = rhs_func
        self.basis_func_type = basis_func_type
        self.left_bc, self.right_bc = boundary_conditions
        self.left_bv, self.right_bv = boundary_values
        self.kwargs = kwargs
        self.num_iter = 1
        self.flops = 0

        if self.basis_func_type == "linear":
            self.grid_size = self.num_elements + 1
        else:
            self.grid_size = 2 * self.num_elements + 1
        self._initialize_fem_solver()
        self._initialize_restriction_prolongation_ops()
        self._initialize_iterative_solvers()

    def _initialize_fem_solver(self):
        self.root_solver = FiniteElementPoisson1D(
            num_element=self.num_elements,
            domain_size=self.domain_size,
            rhs_func=self.rhs_func,
            left_boundary_type=self.left_bc,
            left_boundary_value=self.left_bv,
            right_boundary_type=self.right_bc,
            right_boundary_value=self.right_bv,
            basis_func_type=self.basis_func_type,
            **self.kwargs,
        )

    def _initialize_iterative_solvers(self):
        if self.multigrid_levels == 1:
            return

        left_index = 0 if self.left_bc == "Dirichlet" else 1
        right_index = None if self.right_bc == "Dirichlet" else -1
        original_mat = self.root_solver.stiffness_matrix[
            left_index:right_index,
            left_index:right_index,
        ].copy()

        self.iterative_solvers = []
        self.subgrid_stiffness = []
        curr_grid_size = self.grid_size - 2

        for i in range(self.multigrid_levels):
            if i == 0:
                self.subgrid_stiffness.append(original_mat)
                self.iterative_solvers.append(self.root_solver)
            else:
                self.subgrid_stiffness.append(
                    self.r_matrix[i - 1] @ self.subgrid_stiffness[i - 1] @ self.p_matrix[i - 1]
                )
                self.iterative_solvers.append(
                    IterativeLinSolve(
                        matrix=self.subgrid_stiffness[i],
                        **self.kwargs,
                    )
                )
                self.iterative_solvers[i].save_res = False

            curr_grid_size = (curr_grid_size - 1) // 2

    def solve(self, max_iter=100, tol=1e-5):
        if self.multigrid_levels == 1:
            result = self.root_solver.solve(max_iter=int(1e6), tol=tol)
            self.num_iter += result["num_iter"]
            self.flops += result["flops"]

        else:
            save_res = self.kwargs.get("save_res", False)
            res_file_name = self.kwargs.get("res_file_name", "data/residual.txt")
            iterations = []
            residual_history = []

            tol *= np.amax(np.abs(self.root_solver.rhs_field))
            res = np.sqrt(np.sum(self.root_solver.rhs_field ** 2) / self.grid_size)

            while res > tol:
                if save_res:
                    iterations.append(self.root_solver.num_iter)
                    residual_history.append(res)
                res = self._solve(level=0, max_iter=max_iter, tol=tol)

            if save_res:
                iterations.append(self.root_solver.num_iter)
                residual_history.append(res)
                to_save = np.vstack((np.array(iterations), np.array(residual_history))).T
                np.savetxt(res_file_name, to_save, delimiter=",")

        return self.root_solver.domain_x.copy(), self.root_solver.soln_field.copy()

    def _solve(self, level, max_iter, tol):
        if level == self.multigrid_levels - 1:
            # result = self.iterative_solvers[level].solve(tol=tol, max_iter=int(1e6))
            # self.num_iter += result["num_iter"]
            # self.flops += result["flops"]
            # return result["residual"]
            self.iterative_solvers[level].soln[...] = sla.spsolve(
                self.iterative_solvers[level].matrix,
                self.iterative_solvers[level].rhs
            )

            return 0.0

        # Pre-smoothing
        result = self.iterative_solvers[level].solve(max_iter=max_iter)
        self.num_iter += result["num_iter"]
        self.flops += result["flops"]

        # Restriction
        if level == 0:
            self.iterative_solvers[level + 1].set_rhs(
                (self.r_matrix[level] @ self.root_solver.res_field[1:-1])
            )

        else:
            self.iterative_solvers[level + 1].set_rhs(
                (self.r_matrix[level] @ self.iterative_solvers[level].res)
            )

        # Recursive call
        self._solve(level=level + 1, max_iter=max_iter, tol=tol)

        # Prolongation
        if level == 0:
            self.root_solver.soln_field[1:-1] += (
                self.p_matrix[level] @ self.iterative_solvers[level + 1].soln
            )

            self.root_solver.update_soln_field(
                new_soln_field=self.root_solver.soln_field[
                    self.root_solver.end_index["left"]:self.root_solver.end_index["right"]
                ]
            )

        else:
            self.iterative_solvers[level].modify_soln_field(
                self.p_matrix[level] @ self.iterative_solvers[level + 1].soln
            )

        # Post-smoothing
        result = self.iterative_solvers[level].solve(max_iter=max_iter // 2)
        self.num_iter += result["num_iter"]
        self.flops += result["flops"]

        return result["residual"]

    def _initialize_restriction_prolongation_ops(self):
        self.r_matrix = []  # Restriction operators
        self.p_matrix = []  # Prolongation operators

        fine_grid_size = self.grid_size - 2
        coarse_grid_size = (fine_grid_size - 1) // 2

        interpolation_scheme = np.array([0.5, 1.0, 0.5])

        for _ in range(self.multigrid_levels - 1):
            curr_p_matrix = np.zeros((fine_grid_size, coarse_grid_size), dtype=float)
            for j in range(coarse_grid_size):
                curr_p_matrix[2 * j:2 * j + 3, j] = interpolation_scheme

            curr_p_matrix = spp.csr_matrix(curr_p_matrix)

            curr_r_matrix = curr_p_matrix.transpose() * 0.5

            self.p_matrix.append(curr_p_matrix)
            self.r_matrix.append(curr_r_matrix)

            fine_grid_size = coarse_grid_size
            coarse_grid_size = (fine_grid_size - 1) // 2
