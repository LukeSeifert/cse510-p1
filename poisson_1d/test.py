import numpy as np
import matplotlib.pyplot as plt

from fem_poisson_1d import FiniteElementPoisson1D
from multigrid_fem_poisson_1d import MultigridFEMPoisson1D
from utils import superimposed_sine, lap_superimposed_sine


def plot_convergence(axis, file_name):
    data = np.loadtxt(file_name, delimiter=",")
    curr_plot, = axis.loglog(data[:, 0], data[:, 1], "-x")
    return curr_plot


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

    num_soln = fem_solver.solve(tol=1e-4, max_iter=int(1e5))["soln"]
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
    plot_figure=True,
    **kwargs,
):
    rhs_func = lambda x: lap_superimposed_sine(x, modes=modes)
    exact_soln = lambda x: superimposed_sine(x, modes=modes)
    # rhs_func = lambda x: 8.0
    # exact_soln = lambda x: 4 * x * (1 - x)

    multigrid_solver = MultigridFEMPoisson1D(
        multigrid_levels=multigrid_levels,
        num_element=num_element,
        domain_size=domain_size,
        rhs_func=rhs_func,
        basis_func_type=basis_func_type,
        boundary_values=boundary_values,
        **kwargs,
    )

    num_domain_x, num_soln = multigrid_solver.solve(max_iter=10, tol=1e-5)
    if plot_figure:
        plt.plot(num_domain_x, num_soln, label="fem")
        plt.plot(num_domain_x, exact_soln(num_domain_x), label="exact")
        plt.legend()
        plt.show()

    return multigrid_solver.root_solver.num_iter, multigrid_solver.flops


if __name__ == "__main__":
    sine_modes = [1.0, 2.0]

    # run_finite_element_solver_with_sine_forcing(
    #     num_element=128,
    #     modes=sine_modes,
    #     domain_size=1.0,
    #     boundary_values=(0.0, 0.0),
    #     basis_func_type="quadratic",
    #     solve_type="gauss-seidel",
    #     weight=0.8,
    # )

    grid_depth = 3
    num_element = 256
    basis_func_type = "linear"
    res_file_name = f"data/multigrid_{grid_depth}_n_{num_element}_gs_" + basis_func_type + ".txt"
    num_iter, flops = run_multigrid_fem_with_sine_forcing(
        multigrid_levels=grid_depth,
        num_element=num_element,
        modes=sine_modes,
        domain_size=1.0,
        boundary_values=(0.0, 0.0),
        basis_func_type=basis_func_type,
        solve_type="gauss-seidel",
        weight=0.5,
        plot_figure=False,
        save_res=True,
        res_file_name=res_file_name,
        verbose=False,
    )
    print(f"Total number of iterations is {num_iter}")
    print(f"Total numer of floating point ops is {flops:.2E}")
    # fig, ax = plt.subplots()
    # for i in range(4):
    #     g = plot_convergence(ax, f"data/multigrid_{i + 1}_n_256_gs_linear.txt")
    #     g.set_label(f"Multigrid levels = {i + 1}")
    # ax.legend()
    # plt.show()
