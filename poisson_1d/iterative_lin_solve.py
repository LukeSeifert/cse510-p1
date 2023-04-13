import numpy as np


class IterativeLinSolve:
    def __init__(
        self,
        matrix: np.ndarray,
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
        if temp_rhs is not None:
            self.set_rhs(set_rhs=temp_rhs)

    def _initialize_solver(self):
        weight = self.kwargs.get("weight", 1.0)
        self.scaling_mat = np.zeros_like(self.matrix)

        if self.solve_type == "w-jacobi":
            assert 0.0 < weight <= 1.0, ValueError(
                "Jacobi weight must be between 0 and 1"
            )

            self.scaling_mat[...] = np.diag(
                weight / (np.diag(self.matrix) + self.eps)
            )

        if self.solve_type == "gauss-seidel":
            self.scaling_mat[...] = np.linalg.inv(np.tril(self.matrix))

        if self.solve_type == "sor":
            assert 1.0 <= weight < 2.0, ValueError(
                "SOR weight must be between 1 and 2"
            )
            diagonal = np.diag(np.diag(self.matrix))
            strict_lower_trig = np.tril(self.matrix) - diagonal
            self.scaling_mat[...] = np.linalg.inv(diagonal / weight + strict_lower_trig)

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


if __name__ == "__main__":
    n = 65
    x, dx = np.linspace(0.0, 1.0, n, retstep=True)
    modes = [1, 2]
    b = np.zeros_like(x)
    for k in modes:
        b += (2 * np.pi * k) ** 2 * np.sin(2 * np.pi * k * x)

    rhs = np.zeros_like(b)
    rhs[1:-1] = dx ** 2 * b[1:-1]
    rhs[0] = dx ** 2 / 2 * b[0]
    rhs[-1] = dx ** 2 / 2 * b[-1]

    exact_soln = np.zeros_like(x)
    for k in modes:
        exact_soln += np.sin(2 * np.pi * k * x)

    import scipy.sparse as spp
    a = spp.diags(
        diagonals=[-1.0, 2.0, -1.0],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="csr",
    ).toarray()
    a[0, 0] = 1.0
    a[1, 0] = 0.0
    a[-1, -1] = 1.0
    a[-1, -2] = 0.0
    cls = IterativeLinSolve(matrix=a, solve_type="sor", rhs=rhs, weight=1.4)
    result = cls.solve()

    import matplotlib.pyplot as plt

    plt.plot(x, exact_soln, label="exact")
    plt.plot(x, result, label="jacobi")
    plt.legend()
    plt.show()