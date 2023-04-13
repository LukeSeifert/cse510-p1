import numpy as np
import scipy.sparse as spp
from scipy.sparse import linalg as sla


class IterativeLinSolve:
    def __init__(
        self,
        matrix: spp.csr_matrix,
        solve_type: str,
        **kwargs,
    ):
        acceptable_solver_type = ("w-jacobi", "gauss-seidel", "sor")
        assert solve_type.lower() in acceptable_solver_type, ValueError(
            f"solver_type must be one of the followings: {acceptable_solver_type}"
        )
        assert matrix.shape[0] == matrix.shape[1], ValueError(
            "Input matrix must be a square matrix"
        )

        self.matrix = matrix
        self.dim = matrix.shape[0]
        self.solve_type = solve_type.lower()
        self.kwargs = kwargs
        self.eps = np.finfo(float).eps

        self.rhs = np.zeros(self.dim)
        self.soln = np.zeros(self.dim)
        self.res = np.zeros(self.dim)

        self._initialize_solver()

        temp_rhs = kwargs.get("rhs", None)
        temp_initial_x = kwargs.get("phi_init", None)

        if temp_rhs is not None:
            self.set_rhs(set_rhs=temp_rhs)
        if temp_initial_x is not None:
            self.set_initial_x(initial_x=temp_initial_x)

    def _initialize_solver(self):
        weight = self.kwargs.get("weight", 1.0)

        if self.solve_type == "w-jacobi":
            assert 0.0 < weight <= 1.0, ValueError(
                "Jacobi weight must be between 0 and 1"
            )

            self.scaling_mat = spp.spdiags(
                data=weight / (self.matrix.diagonal() + self.eps),
                diags=0,
                m=self.matrix.shape,
                format="csr",
            )

        elif self.solve_type == "gauss-seidel":
            self.scaling_mat = sla.inv(spp.tril(self.matrix, k=0))

        else:
            assert 1.0 <= weight < 2.0, ValueError(
                "SOR weight must be between 1 and 2"
            )
            diagonal = spp.spdiags(
                data=weight / (self.matrix.diagonal() + self.eps),
                diags=0,
                m=self.matrix.shape,
                format="csr",
            )
            strict_lower_trig = spp.tril(self.matrix, k=-1)
            self.scaling_mat = sla.inv(diagonal / weight + strict_lower_trig)

    def solve(self, tol=1e-3, max_iter=1e6):
        """ Solve linear system Ax = f """
        print("===== Start iterative solve =====")
        tol *= np.amax(np.abs(self.rhs))
        scalar_res = tol + 1.0
        num_iter = 0

        while num_iter < max_iter and scalar_res > tol:
            scalar_res = self.do_step()
            num_iter += 1

        print(
            f"Total number of iteration: {num_iter}\n"
            f"Final residual: {scalar_res}\n"
            "===== Iterative solve ends =====\n"
        )
        return self.soln.copy()

    def do_step(self):
        """
        Do one step of fixed-point iteration
        x = x + M (f - Ax)
        """
        self.compute_vector_residual()
        self.soln[...] += self.scaling_mat @ self.res

        return self.compute_scalar_residual()

    def compute_vector_residual(self):
        """ Compute r = f - Ax """
        self.res[...] = self.rhs - self.matrix @ self.soln

    def compute_scalar_residual(self):
        """
        Compute norm of residual in 1D
        |r| = sqrt(sum(r ** 2) * dx / l) = sqrt(sum(r ** 2) / n)
        """
        return np.sqrt(np.sum(self.res ** 2) / self.dim)

    def set_initial_x(self, initial_x):
        if type(initial_x) == np.ndarray:
            assert self.soln.shape == initial_x.shape, ValueError(
                "Incompatible array size"
            )

        elif type(initial_x) != float:
            raise ValueError("initial_x must be a scalar (float) or numpy array")

        self.soln[...] = initial_x

    def set_rhs(self, set_rhs):
        if type(set_rhs) == np.ndarray:
            assert self.soln.shape == set_rhs.shape, ValueError(
                "Incompatible array size"
            )

        elif type(set_rhs) != float:
            raise ValueError("set_rhs must be a scalar (float) or numpy array")

        self.rhs[...] = set_rhs
