import numpy as np
import matplotlib.pyplot as plt

from fem_poisson_1d import FiniteElementPoisson1D
from multigrid_fem_poisson_1d import MultigridFEMPoisson1D


def lap_superimposed_sine(x, modes: list[float]):
    func_vals = np.zeros_like(x)
    for k in modes:
        func_vals += (2.0 * np.pi * k) ** 2 * np.sin(2.0 * np.pi * k * x)

    return func_vals


def superimposed_sine(x, modes: list[float]):
    func_vals = np.zeros_like(x)
    for k in modes:
        func_vals += np.sin(2.0 * np.pi * k * x)

    return func_vals


def run_finite_element_solver_with_sine_forcing(
    num_element,
    modes,
    domain_size=1.0,
    boundary_values=(0.0, 0.0),
    basis_func_type="linear",
    solve_type="gauss_seidel",
    weight=0.8,
):
    rhs_func = lambda x: lap_superimposed_sine(x, modes=modes)
    exact_soln = lambda x: superimposed_sine(x, modes=modes)

    fem_solver = FiniteElementPoisson1D(
        num_element=num_element,
        domain_size=domain_size,
        rhs_func=rhs_func,
        left_boundary_type="Dirichlet",
        left_boundary_value=boundary_values[0],
        right_boundary_type="Dirichlet",
        right_boundary_value=boundary_values[1],
        basis_func_type=basis_func_type,
        solve_type=solve_type,
        weight=weight,
    )

    num_soln = fem_solver.solve(tol=1e-4, max_iter=int(1e5))
    plt.plot(fem_solver.domain_x, num_soln, label="fem")
    plt.plot(fem_solver.domain_x, exact_soln(fem_solver.domain_x), label="exact")
    plt.legend()
    plt.show()


def run_multigrid_fem_with_sine_forcing(
    multigrid_levels,
    num_element,
    modes,
    domain_size=1.0,
    boundary_values=(0.0, 0.0),
    basis_func_type="linear",
    solve_type="gauss-seidel",
    weight=0.8,
    plot_figure=True,
):
    # rhs_func = lambda x: lap_superimposed_sine(x, modes=modes)
    # exact_soln = lambda x: superimposed_sine(x, modes=modes)
    rhs_func = lambda x: 8.0
    exact_soln = lambda x: 4 * x * (1 - x)

    multigrid_solver = MultigridFEMPoisson1D(
        multigrid_levels=multigrid_levels,
        num_element=num_element,
        domain_size=domain_size,
        rhs_func=rhs_func,
        basis_func_type=basis_func_type,
        boundary_values=boundary_values,
        solve_type=solve_type,
        weight=weight,
    )

    num_domain_x, num_soln = multigrid_solver.solve()
    if plot_figure:
        plt.plot(num_domain_x, num_soln, label="fem")
        plt.plot(num_domain_x, exact_soln(num_domain_x), label="exact")
        plt.legend()
        plt.show()

    return multigrid_solver.num_iter


if __name__ == "__main__":
    sine_modes = [1.0, 2.0, 4.5]

    # run_finite_element_solver_with_sine_forcing(
    #     num_element=128,
    #     modes=sine_modes,
    #     domain_size=1.0,
    #     boundary_values=(0.0, 0.0),
    #     basis_func_type="linear",
    #     solve_type="gauss-seidel",
    #     weight=0.8,
    # )

    num_iter = run_multigrid_fem_with_sine_forcing(
        multigrid_levels=4,
        num_element=128,
        modes=sine_modes,
        domain_size=1.0,
        boundary_values=(0.0, 0.0),
        basis_func_type="quadratic",
        solve_type="gauss-seidel",
        weight=0.8,
        plot_figure=False,
    )
    print(f"Total number of iterations {num_iter}")
