import numpy as np
from typing import Literal, Optional


class StencilWeightedJacobiIteration1D:
    def __init__(
        self,
        rhs_field: np.ndarray,
        stencil: np.ndarray = np.array([1.0, -2.0, 1.0]),
        boundary_conditions: tuple[
            Literal["Dirichlet", "Neumann"], Literal["Dirichlet", "Neumann"]
        ] = ("Dirichlet", "Dirichlet"),
        boundary_values: tuple[float, float] = (0.0, 0.0),
        weight: float = 1.0,
        tol: float = 1e-5,
        phi_init: Optional[np.ndarray] = None,
        **kwargs,
    ):
        assert 0.0 < weight <= 1.0, ValueError("Weighted Jacobi must have weight 0 < w <= 1")
        self.rhs_field = rhs_field
        self.stencil = stencil
        self.boundary_conditions = boundary_conditions
        self.boundary_values = boundary_values
        self.tol = tol
        self.weight = weight
        self.max_iter = kwargs.get("max_iter", None)
        self._initialize_fields(phi_init)
        self._initialize_iteration_params()

    def solve(self):
        while not self.converged:
            self._do_step()

        return self.phi[:, self.curr_idx].copy()

    def _check_convergence(self):
        self._compute_residual()
        if self.residual <= self.tol:
            self.converged = True

        if self.max_iter is not None and self.max_iter < self.num_iter:
            self.converged = True

    def _compute_residual(self):
        self.residual = np.sqrt(
            np.sum((self.phi[:, 0] - self.phi[:, 1]) ** 2) / self.grid_size
        )

    def _do_step(self):
        self.phi[1:-1, 1 - self.curr_idx] = (
            1 - self.weight
        ) * self.phi[1:-1, self.curr_idx] - self.weight / self.stencil[1] * (
            self.rhs_field[1:-1]
            + self.stencil[0] * self.phi[:-2, self.curr_idx]
            + self.stencil[2] * self.phi[2:, self.curr_idx]
        )
        if self.boundary_conditions[0] == "Neumann":
            self.phi[0, 1 - self.curr_idx] = self.rhs_field[0] + self.phi[1, self.curr_idx]
        if self.boundary_conditions[1] == "Neumann":
            self.phi[-1, 1 - self.curr_idx] = self.rhs_field[-1] + self.phi[-2, self.curr_idx]

        self.curr_idx = 1 - self.curr_idx
        self.num_iter += 1
        self._check_convergence()

    def _initialize_fields(self, phi_init):
        self.grid_size = self.rhs_field.shape[0]
        self.phi = np.zeros((self.grid_size, 2))
        if phi_init is not None:
            self.phi[:, 0] = phi_init.copy()

        if self.boundary_conditions[0] == "Dirichlet":
            self.phi[0, :] = self.boundary_values[0]
        if self.boundary_conditions[1] == "Dirichlet":
            self.phi[-1, :] = self.boundary_values[1]

    def _initialize_iteration_params(self):
        self.curr_idx: int = 0
        self.num_iter: int = 0
        self.converged: bool = False
        self.residual: float = self.tol + 1.0


if __name__ == "__main__":
    x, dx = np.linspace(0.0, 1.0, 128, retstep=True)
    modes = [1, 2, 4, 8]
    b = np.zeros_like(x)
    for k in modes:
        b += (2 * np.pi * k) ** 2 * np.cos(2 * np.pi * k * x)

    rhs = np.zeros_like(b)
    rhs[1:-1] = dx ** 2 * b[1:-1]
    rhs[0] = dx ** 2 / 2 * b[0]
    rhs[-1] = dx ** 2 / 2 * b[-1]

    exact_soln = np.zeros_like(x)
    for k in modes:
        exact_soln += np.cos(2 * np.pi * k * x)

    jacobi_iterate = StencilWeightedJacobiIteration1D(
        rhs_field=rhs,
        weight=1.0,
        boundary_conditions=("Neumann", "Neumann"),
        # max_iter=10000,
    )
    result = jacobi_iterate.solve()
    print(jacobi_iterate.num_iter)

    import matplotlib.pyplot as plt
    plt.plot(x, exact_soln, label="exact")
    plt.plot(x, result, label="jacobi")
    plt.legend()
    plt.show()
