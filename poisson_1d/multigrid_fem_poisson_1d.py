from typing import Literal, Callable

from fem_poisson_1d import FiniteElementPoisson1D


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
        self.num_iter = 0
        self._initialize_fem_solvers()

    def _initialize_fem_solvers(self):
        self.fem_solvers = []
        for i in range(self.multigrid_levels):
            if i == 0:
                self.fem_solvers.append(
                    FiniteElementPoisson1D(
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
                )

            else:
                self.fem_solvers.append(
                    FiniteElementPoisson1D(
                        num_element=self.num_elements // 2 ** i,
                        domain_size=self.domain_size,
                        rhs_func=self.rhs_func,
                        basis_func_type=self.basis_func_type,
                        **self.kwargs,
                    )
                )

        # Create an alias to the root solver
        self.root_solver = self.fem_solvers[0]

    def solve(self):
        self._solve(level=0)

        # Post smoothing
        self.root_solver.solve(tol=1e-4, max_iter=int(1e5))

        # Sum the numbers of iterations from each solver
        for i in range(self.multigrid_levels):
            self.num_iter += self.fem_solvers[i].num_iter

        return self.root_solver.domain_x.copy(), self.root_solver.soln_field.copy()

    def _solve(self, level):
        if level == self.multigrid_levels - 1:
            self.fem_solvers[level].solve(tol=1e-4, max_iter=int(1e5))
            return

        # Pre-smoothing
        self.fem_solvers[level].solve(max_iter=1000)

        # Restriction
        self._restriction(
            fine_residual=self.fem_solvers[level].res_field,
            coarse_residual=self.fem_solvers[level + 1].rhs_field,
        )

        # Recursive call
        self._solve(level=level + 1)

        # Prolongation
        self._prolongation(
            coarse_correction=self.fem_solvers[level + 1].soln_field,
            fine_correction=self.fem_solvers[level].soln_field
        )

    @staticmethod
    def _restriction(fine_residual, coarse_residual) -> None:
        coarse_residual[...] = fine_residual[::2]

    @staticmethod
    def _prolongation(coarse_correction, fine_correction) -> None:
        fine_correction[::2] += coarse_correction[...]
        fine_correction[1::2] += 0.5 * (
            coarse_correction[:-1] + coarse_correction[1:]
        )
