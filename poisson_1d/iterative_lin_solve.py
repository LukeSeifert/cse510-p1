import numpy as np
import scipy.sparse as spp
from scipy.sparse import linalg as sla


class IterativeLinSolve:
    def __init__(
        self,
        matrix: spp.csr_matrix,
        **kwargs,
    ):
        self.solve_type = kwargs.get("solve_type", "w-jacobi")
        acceptable_solver_type = ("w-jacobi", "gauss-seidel", "sor")
        assert self.solve_type.lower() in acceptable_solver_type, ValueError(
            f"solver_type must be one of the followings: {acceptable_solver_type}"
        )
        assert matrix.shape[0] == matrix.shape[1], ValueError(
            "Input matrix must be a square matrix"
        )

        self.matrix = matrix
        self.dim = matrix.shape[0]
        self.kwargs = kwargs
        self.eps = np.finfo(float).eps

        self.rhs = np.zeros(self.dim)
        self.soln = np.zeros(self.dim)
        self.res = np.zeros(self.dim)

        self.verbose = kwargs.get("verbose", True)
        self.save_res = kwargs.get("save_res", False)

        self._initialize_solver()

        temp_rhs = kwargs.get("rhs", None)
        temp_initial_x = kwargs.get("phi_init", None)

        if temp_rhs is not None:
            self.set_rhs(set_rhs=temp_rhs)
        if temp_initial_x is not None:
            self.modify_soln_field(set_soln=temp_initial_x)

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
        if self.verbose:
            print("===== Start iterative solve =====")

        tol *= np.amax(np.abs(self.rhs))

        res_history = []
        iterations = []
        scalar_res = tol + 1.0
        num_iter = 0

        while num_iter < max_iter and scalar_res > tol:
            scalar_res = self.do_step()
            num_iter += 1

            if num_iter % 50 == 0 and self.save_res:
                iterations.append(num_iter)
                res_history.append(scalar_res)

        if self.verbose:
            print(
                f"Total number of iteration: {num_iter}\n"
                f"Final residual: {scalar_res}\n"
                "===== Iterative solve ends =====\n"
            )

        if self.save_res:
            filename = self.kwargs.get("res_file_name", "data/residuals.txt")
            to_save = np.vstack((np.array(iterations), np.array(res_history))).T
            np.savetxt(filename, to_save, delimiter=",")

        return {
            "soln": self.soln.copy(),
            "num_iter": num_iter,
            "residual": scalar_res,
            "flops": num_iter * 4 * self.dim ** 2,
        }

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

    def modify_soln_field(self, set_soln):
        if type(set_soln) == np.ndarray:
            assert self.soln.shape == set_soln.shape, ValueError(
                "Incompatible array size"
            )

        elif type(set_soln) != float:
            raise ValueError("initial_x must be a scalar (float) or numpy array")

        self.soln[...] += set_soln

    def reset_soln_field(self):
        self.soln[...] *= 0.0

    def set_rhs(self, set_rhs):
        if type(set_rhs) == np.ndarray:
            assert self.soln.shape == set_rhs.shape, ValueError(
                "Incompatible array size"
            )

        elif type(set_rhs) != float:
            raise ValueError("set_rhs must be a scalar (float) or numpy array")

        self.rhs[...] = set_rhs
