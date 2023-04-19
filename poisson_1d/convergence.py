import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Callable

from utils import lap_superimposed_sine, neg_lap_logistic_map, format_plot
from multigrid_fem_poisson_1d import MultigridFEMPoisson1D


def compute_error_norm(num_soln, exact_soln):
    assert(len(num_soln) == len(exact_soln))
    return np.linalg.norm(num_soln - exact_soln) / np.sqrt(len(num_soln))


def get_file_name(
    num_element: int,
    multigrid_levels: int,
    solve_type: Literal["w-jacobi", "gauss-seidel", "sor"],
    basis_func_type: Literal["linear", "quadratic"],
    rhs_func_type: Literal["sine", "logistic"],
):
    return (
        f"data/multigrid_ne_{num_element}"
        f"_grid_{multigrid_levels}_solver_"
        + solve_type
        + "_basis_type_"
        + basis_func_type
        + "_rhs_type_"
        + rhs_func_type
        + ".txt"
    )


def run_multigrid_convergence(
    multigrid_levels: list[int],
    rhs_func: Callable,
    rhs_func_type: Literal["sine", "logistic"],
    num_element: int = 256,
    basis_func_type: Literal["linear", "quadratic"] = "linear",
    solve_type: Literal["w-jacobi", "gauss-seidel", "sor"] = "gauss-seidel",
    **kwargs,
):
    for ml in multigrid_levels:
        res_file_name = get_file_name(
            num_element=num_element,
            multigrid_levels=ml,
            solve_type=solve_type,
            basis_func_type=basis_func_type,
            rhs_func_type=rhs_func_type,
        )
        multigrid_solver = MultigridFEMPoisson1D(
            multigrid_levels=ml,
            num_element=num_element,
            domain_size=1.0,
            rhs_func=rhs_func,
            basis_func_type=basis_func_type,
            solve_type=solve_type,
            save_res=True,
            res_file_name=res_file_name,
            **kwargs,
        )

        multigrid_solver.solve(max_iter=10, tol=1e-5)


def plot_convergence(
    multigrid_levels: list[int],
    num_element: int,
    solve_type: Literal["w-jacobi", "gauss-seidel", "sor"],
    basis_func_type: Literal["linear", "quadratic"],
    rhs_func_type: Literal["sine", "logistic"],
    save_fig: bool = False,
):
    format_plot()
    fig, ax = plt.subplots()
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Residual")
    ax.set_xlim(left=1, right=1e6)
    colors = ["k", "r", "b", "g"]
    for idx, ml in enumerate(multigrid_levels):
        file_name = get_file_name(
            num_element=num_element,
            multigrid_levels=ml,
            solve_type=solve_type,
            basis_func_type=basis_func_type,
            rhs_func_type=rhs_func_type,
        )
        data = np.loadtxt(file_name, delimiter=",")
        ax.loglog(data[:, 0], data[:, 1], "-8", color=colors[idx], label=f"Grid depth = {ml}")
        ax.legend()

    if save_fig:
        fig_name = (
            f"figs/multigrid_ne_{num_element}"
            f"_solver_"
            + solve_type
            + "_basis_type_"
            + basis_func_type
            + "_rhs_type_"
            + rhs_func_type
            + ".png"
        )
        fig.savefig(fig_name, dpi=200)


if __name__ == "__main__":
    gen_num_element = 256
    gen_multigrid_levels = [1, 2]
    gen_rhs_func_type = "logistic"
    gen_basis_func_type = "linear"
    gen_solve_type = "gauss-seidel"

    # run_multigrid_convergence(
    #     multigrid_levels=gen_multigrid_levels,
    #     # rhs_func=lambda x: lap_superimposed_sine(x, modes=[1, 2]),
    #     rhs_func=neg_lap_logistic_map,
    #     rhs_func_type=gen_rhs_func_type,
    #     basis_func_type=gen_basis_func_type,
    #     solve_type=gen_solve_type,
    #     verbose=False,
    # )

    plot_convergence(
        multigrid_levels=gen_multigrid_levels,
        num_element=gen_num_element,
        rhs_func_type=gen_rhs_func_type,
        basis_func_type=gen_basis_func_type,
        solve_type=gen_solve_type,
        save_fig=True,
    )
    plt.show()