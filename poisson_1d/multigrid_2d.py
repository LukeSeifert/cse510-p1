import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


class Multigrid2DPoisson:
    def __init__(
        self,
        levels: int,
        rhs_field: np.ndarray,
        boundary_values: dict[str, float | np.ndarray] = None,
        weight: float = 1.0,
        relative_tol: float = 1e-3,
        iter_per_level: int = 10,
        phi_init: Optional[np.ndarray] = None,
    ):
        self.grid_size_x, self.grid_size_y = rhs_field.shape
        assert (
            (self.grid_size_x - 1) % (2 ** (levels - 1)) == 0
        ) and (
            (self.grid_size_y - 1) % (2 ** (levels - 1)) == 0
        ), ValueError(
            "Grid size should be 2^(levels - 1) * n + 1"
        )
        self.levels = levels
        self.rhs_field = rhs_field
        self.boundary_values = boundary_values
        self.weight = weight
        self.relative_tol = relative_tol
        self.iter_per_level = iter_per_level
        self.num_iter = 0
        self._initialize_fields(phi_init)

    def solve(self):
        # Recursive multi-grid
        if self.levels > 1:
            self._solve(level=0)

        # Post-smoothing
        self.num_iter = 0
        # plt.contourf(xx, yy, self.phi_container[0])
        # plt.colorbar()
        # plt.show()
        # 1/0
        self._smoothing_at_level(level=0, max_iter=int(1e6), absolute=self.relative_tol)

        return self.phi_container[0].copy()

    def _solve(self, level):
        print(f"===== Now at level {level} =====")
        self.num_iter = 0
        if level == self.levels - 1:
            self._smoothing_at_level(level, max_iter=int(1e6), absolute=1e-8)
            return

        self._smoothing_at_level(level, max_iter=self.iter_per_level)
        # if level == 1:
        #     plt.contourf(xx, yy, self.phi_container[0])
        #     plt.colorbar()
        #     plt.show()
        #     1/0
        self._restriction(
            fine_residual=self._compute_residual_array_at_level(level),
            coarse_residual=self.rhs_container[level + 1],
        )
        self._solve(level=level + 1)
        self._prolongation(self.phi_container[level + 1], self.phi_container[level])

    def _compute_residual_array_at_level(self, level):
        lap_phi = np.zeros_like(self.phi_container[level])
        lap_phi[1:-1, 1:-1] = (
            4.0 * self.phi_container[level][1:-1, 1:-1]
            - self.phi_container[level][:-2, 1:-1]
            - self.phi_container[level][2:, 1:-1]
            - self.phi_container[level][1:-1, 2:]
            - self.phi_container[level][1:-1, :-2]
        )
        return lap_phi

    def _compute_residual_at_level(self, level):
        return self._compute_residual(
            phi=self._compute_residual_array_at_level(level)[1:-1, 1:-1],
            rhs=self.rhs_container[level][1:-1, 1:-1],
        )

    def _do_step_jacobi(self, phi, rhs):
        self.num_iter += 1
        phi[1:-1, 1:-1] = (
            1.0 - self.weight
        ) * phi[1:-1, 1:-1] + self.weight / 4.0 * (
            rhs[1:-1, 1:-1]
            + phi[:-2, 1:-1]
            + phi[2:, 1:-1]
            + phi[1:-1, 2:]
            + phi[1:-1, :-2]
        )

        # (Slow) Gauss-Seidel
        # for i in np.arange(1, self.grid_size_x - 1):
        #     for j in np.arange(1, self.grid_size_y - 1):
        #         phi[i, j] = 0.25 * (
        #             rhs[i, j]
        #             + phi[i - 1, j]
        #             + phi[i, j - 1]
        #             + phi[i + 1, j]
        #             + phi[i, j + 1]
        #         )

    def _initialize_fields(self, phi_init):
        phi = np.zeros_like(self.rhs_field)
        if phi_init is not None:
            phi[...] = phi_init.copy()

        self.phi_container = [phi]
        self.rhs_container = [self.rhs_field]
        for lvl in range(self.levels - 1):
            self.phi_container.append(
                self._generate_coarse_grid(self.phi_container[lvl])
            )
            self.rhs_container.append(
                self._generate_coarse_grid(self.rhs_container[lvl] * 0.0)
            )

        if self.boundary_values is not None:
            self.phi_container[0][:, 0] = self.boundary_values["west"]
            self.phi_container[0][:, -1] = self.boundary_values["east"]
            self.phi_container[0][0, :] = self.boundary_values["south"]
            self.phi_container[0][-1, :] = self.boundary_values["north"]

    def _smoothing_at_level(self, level, max_iter, absolute=-1.0):
        initial_res = self._compute_residual_at_level(level)
        res = initial_res
        scale = absolute * np.amax(np.abs(self.rhs_container[level]))
        num_iter = 0
        while num_iter < max_iter and res > scale:
            self._do_step_jacobi(
                phi=self.phi_container[level],
                rhs=self.rhs_container[level],
            )
            res = self._compute_residual_at_level(level)
            num_iter += 1

    @staticmethod
    def _generate_coarse_grid(fine_grid: np.ndarray) -> np.ndarray:
        return fine_grid[::2, ::2].copy()

    @staticmethod
    def _restriction(fine_residual: np.ndarray, coarse_residual: np.ndarray) -> None:
        coarse_residual[1:-1, 1:-1] = 0.25 * fine_residual[2:-2:2, 2:-2:2]
        coarse_residual[1:-1, 1:-1] += 0.125 * (
            fine_residual[2:-2:2, 1:-3:2]
            + fine_residual[2:-2:2, 3:-1:2]
            + fine_residual[1:-3:2, 2:-2:2]
            + fine_residual[3:-1:2, 2:-2:2]
        )
        coarse_residual[1:-1, 1:-1] += 0.0625 * (
            fine_residual[1:-3:2, 1:-3:2]
            + fine_residual[1:-3:2, 3:-1:2]
            + fine_residual[3:-1:2, 1:-3:2]
            + fine_residual[3:-1:2, 3:-1:2]
        )

    @staticmethod
    def _prolongation(coarse_correction: np.ndarray, fine_correction: np.ndarray) -> None:
        fine_correction[::2, ::2] += coarse_correction[...]
        fine_correction[1::2, ::2] += 0.5 * (
            coarse_correction[:-1, :] + coarse_correction[1:, :]
        )
        fine_correction[::2, 1::2] += 0.5 * (
            coarse_correction[:, 1:] + coarse_correction[:, :-1]
        )
        fine_correction[1::2, 1::2] += 0.25 * (
            coarse_correction[:-1, :-1]
            + coarse_correction[:-1, 1:]
            + coarse_correction[1:, :-1]
            + coarse_correction[1:, 1:]
        )

    @staticmethod
    def _compute_residual(phi: np.ndarray, rhs: np.ndarray) -> float:
        return np.sqrt(
            np.sum((rhs - phi) ** 2) / (phi.shape[0] * phi.shape[1])
        )


if __name__ == "__main__":
    num_levels = 1
    x, dx = np.linspace(0.0, 1.0, 129, retstep=True)
    y = x.copy()

    xx, yy = np.meshgrid(x, y)
    b = -5.0 * np.exp(xx) * np.exp(-2.0 * yy) / dx ** 2
    exact_soln = np.exp(xx) * np.exp(-2.0 * yy) / dx ** 2

    b_field = np.zeros_like(b)
    b_field[1:-1, 1:-1] = b[1:-1, 1:-1]
    b_field[0, 1:-1] = 0.5 * b[0, 1:-1]
    b_field[-1, 1:-1] = 0.5 * b[-1, 1:-1]
    b_field[1:-1, 0] = 0.5 * b[1:-1, 0]
    b_field[1:-1, -1] = 0.5 * b[1:-1, -1]
    b_field *= dx ** 2

    bcs = {
        "west": exact_soln[:, 0],
        "east": exact_soln[:, -1],
        "south": exact_soln[0, :],
        "north": exact_soln[-1, :],
    }

    solver = Multigrid2DPoisson(
        levels=num_levels,
        rhs_field=b * dx ** 2,
        boundary_values=bcs,
        relative_tol=1e-2,
        iter_per_level=10000
    )
    result = solver.solve()
    print(solver.num_iter)

    fig, ax = plt.subplots(nrows=2)
    m1 = ax[0].contourf(xx, yy, exact_soln)
    ax[0].set_title("Exact")
    m2 = ax[1].contourf(xx, yy, result)
    ax[1].set_title("Jacobi")
    plt.colorbar(mappable=m1)
    plt.colorbar(mappable=m2)
    plt.show()
