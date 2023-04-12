import numpy as np
from typing import Literal, Optional


class Multigrid1DPoisson:
    def __init__(
        self,
        levels: int,
        rhs_field: np.ndarray,
        stencil: np.ndarray = np.array([1.0, -2.0, 1.0]),
        boundary_conditions: tuple[
            Literal["Dirichlet", "Neumann"], Literal["Dirichlet", "Neumann"]
        ] = ("Dirichlet", "Dirichlet"),
        boundary_values: tuple[float, float] = (0.0, 0.0),
        weight: float = 1.0,
        relative_tol: float = 1e-3,
        phi_init: Optional[np.ndarray] = None,
    ):
        assert (len(rhs_field) - 1) % (2 ** (levels - 1)) == 0, ValueError(
            "Grid size should be 2^(levels - 1) * n + 1"
        )
        self.levels = levels
        self.rhs_field = rhs_field
        self.stencil = stencil
        self.boundary_conditions = boundary_conditions
        self.boundary_values = boundary_values
        self.weight = weight
        self.relative_tol = relative_tol
        self.num_iter = 0
        self._initialize_fields(phi_init)

    def solve(self, threshold=0.25):
        # Recursive multi-grid
        if self.levels > 1:
            self._solve(level=0, threshold=threshold)

        # Post-smoothing
        self.num_iter = 0
        self._smoothing_at_level(level=0, res_factor=1e-14, absolute=self.relative_tol)

        return self.phi_container[0].copy()

    def _solve(self, level, threshold):
        print(f"===== Now at level {level} =====")
        if level == self.levels - 1:
            self._smoothing_at_level(level, res_factor=1e-14, absolute=self.relative_tol)
            return

        self._smoothing_at_level(level, res_factor=threshold)
        self._restriction(
            fine_residual=self._compute_residual_array_at_level(level),
            coarse_residual=self.rhs_container[level + 1],
        )
        self._solve(level=level + 1, threshold=threshold)
        self._prolongation(self.phi_container[level + 1], self.phi_container[level])

    def _compute_residual_array_at_level(self, level):
        lap_phi = np.zeros_like(self.phi_container[level])
        lap_phi[1:-1] = -(
                self.stencil[0] * self.phi_container[level][:-2]
                + self.stencil[1] * self.phi_container[level][1:-1]
                + self.stencil[2] * self.phi_container[level][2:]
        )
        return lap_phi

    def _compute_residual_at_level(self, level):
        return self._compute_residual(
            phi=self._compute_residual_array_at_level(level),
            rhs=self.rhs_container[level],
        )

    def _do_step_jacobi(self, phi, rhs):
        self.num_iter += 1
        phi[1:-1] = (1 - self.weight) * phi[1:-1] - self.weight / self.stencil[1] * (
            rhs[1:-1] + self.stencil[0] * phi[:-2] + self.stencil[2] * phi[2:]
        )

    def _initialize_fields(self, phi_init):
        phi = np.zeros_like(self.rhs_field)
        if phi_init is not None:
            phi[...] = phi_init.copy()

        if self.boundary_conditions[0] == "Dirichlet":
            phi[0] = self.boundary_values[0]
        if self.boundary_conditions[1] == "Dirichlet":
            phi[-1] = self.boundary_values[1]

        self.phi_container = [phi]
        self.rhs_container = [self.rhs_field]
        for lvl in range(self.levels - 1):
            self.phi_container.append(
                self._generate_coarse_grid(self.phi_container[lvl])
            )
            self.rhs_container.append(
                self._generate_coarse_grid(self.rhs_container[lvl])
            )

    def _smoothing_at_level(self, level, res_factor, absolute=-1.0):
        initial_res = self._compute_residual_at_level(level)
        res = initial_res
        scale = np.amax(np.abs(self.rhs_container[level]))
        while res > absolute * scale and res > res_factor * initial_res:
            self._do_step_jacobi(
                phi=self.phi_container[level],
                rhs=self.rhs_container[level],
            )
            res = self._compute_residual_at_level(level)

    @staticmethod
    def _generate_coarse_grid(fine_grid: np.ndarray) -> np.ndarray:
        return fine_grid[::2].copy()

    @staticmethod
    def _restriction(fine_residual, coarse_residual) -> None:
        coarse_residual[...] = fine_residual[::2]

    @staticmethod
    def _prolongation(coarse_correction, fine_correction) -> None:
        fine_correction[::2] += coarse_correction[...]
        fine_correction[1::2] += 0.5 * (
            coarse_correction[:-1] + coarse_correction[1:]
        )

    @staticmethod
    def _compute_residual(phi: np.ndarray, rhs: np.ndarray) -> float:
        return np.sqrt(
            np.sum((rhs - phi) ** 2) / len(phi)
        )


if __name__ == "__main__":
    num_levels = 1
    x, dx = np.linspace(0.0, 1.0, 257, retstep=True)
    # modes = [1, 2, 4, 8, 16]
    # b = np.zeros_like(x)
    # for k in modes:
    #     b += (2 * np.pi * k) ** 2 * np.sin(2 * np.pi * k * x)
    #
    b = 4.0 * x * (1.0 - x)

    exact_soln = x / 3 * (x ** 3 - 2 * x ** 2 + 1)

    # for k in modes:
    #     exact_soln += np.sin(2 * np.pi * k * x)

    b_field = np.zeros_like(b)
    b_field[1:-1] = dx ** 2 * b[1:-1]
    b_field[0] = dx ** 2 / 2 * b[0]
    b_field[-1] = dx ** 2 / 2 * b[-1]

    solver = Multigrid1DPoisson(
        levels=num_levels,
        rhs_field=b_field,
    )
    result = solver.solve(threshold=0.3)
    print(solver.num_iter)

    import matplotlib.pyplot as plt
    plt.plot(x, exact_soln, label="exact")
    plt.plot(x, result, label="jacobi")
    plt.legend()
    plt.show()
