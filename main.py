# Main run file
from firedrake import *
from firedrake.solving_utils import KSPReasons
import time
import matplotlib.pyplot as plt
import numpy as np
import os

# PETSc
# Solve defaults {'ksp_rtol': 1e-07, 'ksp_type': 'preonly', 'mat_mumps_icntl_14': 200, 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps', 'pc_type': 'lu'}
# ksp_type: https://w3.pppl.gov/m3d/petsc-dev/docs/manualpages/KSP/KSPType.html
    # pre_only: stub method that ONLY applies the preconditioner. May be used in inner iterations.
    # cg: preconditioned conjugate gradient iterative method.
# pc_type: https://w3.pppl.gov/m3d/petsc-dev/docs/manualpages/PC/PCType.html
    # pcmg: Multigrid preconditioning (optional settings https://w3.pppl.gov/m3d/petsc-dev/docs/manualpages/PC/PCMG.html)
        # Full/default types https://w3.pppl.gov/m3d/petsc-dev/docs/manualpages/PC/PCMGType.html
        # There should be an option pc_mg_cycles which takes either v or w, meaning v, w, and f can be implemented currently.



#def run_solve(V, a, L, bcs, parameters):
#    u = Function(V)
#    # https://www.firedrakeproject.org/firedrake.html?highlight=solve#firedrake.solving.solve
#    solve(a == L, u, bcs=bcs, solver_parameters=parameters)
#    return u

def linear_var_solve(V, a, L, bcs, parameters):
    u = Function(V)
    # https://www.firedrakeproject.org/firedrake.html?highlight=solve#firedrake.solving.solve
    # https://www.firedrakeproject.org/firedrake.html?highlight=solve#firedrake.variational_solver.LinearVariationalProblem
    vpb = LinearVariationalProblem(a, L, u, bcs=bcs)
    solver = LinearVariationalSolver(vpb, solver_parameters=parameters)
    solver.solve()
    cells = u.function_space().mesh().num_cells() 
    iterations = solver.snes.ksp.getIterationNumber()
    return u, cells, iterations

def error(u, V, exact):
    expect = Function(V).interpolate(exact)
    return norm(assemble(u - expect))

def sub_solver(name, parameters, linear_var_solve, V, a, L, bcs, error, times, iterations, errors, cell_count, exact):
    start = time.time()

    #ksp_rtol = 0
    #ksp_atol = 0
    #ksp_max_it = 30
    #parameters['ksp_rtol'] = ksp_rtol
    #parameters['ksp_atol'] = ksp_atol
    #parameters['ksp_max_it'] = ksp_max_it
    

    u, cells, iters = linear_var_solve(V, a, L, bcs, parameters)
    #u = run_solve(V, a, L, bcs, parameters)
    err = error(u, V, exact)

    end = time.time()
    net_time = end-start

    #vpb = LinearVariationalProblem(a, L, u, bcs=bcs)
    #solver = LinearVariationalSolver(vpb, solver_parameters=parameters)
    #solver.solve()



    print(f'{name} error ', err)
    print(f'Cells: {cells}\nIterations: {iters}')
    # CONVERGED_ITS means 1 iter of preconditioner applied
    # https://w3.pppl.gov/m3d/petsc-dev/docs/manualpages/KSP/KSP_CONVERGED_ITS.html#KSP_CONVERGED_ITS
    #print(f'Reason: {KSPReasons[solver.snes.ksp.getConvergedReason()]}')
    print(f'Took {net_time}s')
    print('-'*10)
    times[name] = net_time
    iterations[name] = iters
    errors[name] = err
    cell_count[name] = cells
    return times, iterations, errors, cell_count





def compare_solvers(u, error, sub_solver, V, a, L, bcs, exact):
    times = dict()
    iterations = dict()
    errors = dict()
    cell_count = dict()
    global CG_solvers
    global MG_solvers
    
    print('-'*20)

    # ITS
    name = 'LU Direct Solve'
    parameters = {"ksp_type": "preonly", "pc_type": "lu"}
    times, iterations, errors, cell_count = sub_solver(name, parameters, linear_var_solve, V, a, L, bcs, error, times, iterations, errors, cell_count, exact)

    ## RTOL
    #name = 'CG Solve'
    #parameters = {"ksp_type": "cg", "pc_type": "none", 'mat_type': 'mat_free', 'ksp_monitor': None}
    #times, iterations, errors, cell_count = sub_solver(name, parameters, linear_var_solve, V, a, L, bcs, error, times, iterations, errors, cell_count, exact)

    if MG_solvers:
        # ITS
        name = 'MG V-cycle Solve'
        parameters = {"ksp_type": "preonly", "pc_type": "mg", 'pc_mg_cycles': 'v'}
        times, iterations, errors, cell_count = sub_solver(name, parameters, linear_var_solve, V, a, L, bcs, error, times, iterations, errors, cell_count, exact)
    
    if CG_solvers:
        # RTOL
        name = 'MG V-cycle PC + CG Solve'
        parameters = {"ksp_type": "cg", "pc_type": "mg"}
        times, iterations, errors, cell_count = sub_solver(name, parameters, linear_var_solve, V, a, L, bcs, error, times, iterations, errors, cell_count, exact)
    
    if MG_solvers:
        # ITS
        name = 'MG W-cycle Solve'
        parameters = {"ksp_type": "preonly", "pc_type": "mg", 'pc_mg_cycles': 'w'}
        times, iterations, errors, cell_count = sub_solver(name, parameters, linear_var_solve, V, a, L, bcs, error, times, iterations, errors, cell_count, exact)
    
    if CG_solvers:
        # RTOl
        name = 'MG W-cycle PC + CG Solve'
        parameters = {"ksp_type": "cg", "pc_type": "mg", 'pc_mg_cycles': 'w'}
        times, iterations, errors, cell_count = sub_solver(name, parameters, linear_var_solve, V, a, L, bcs, error, times, iterations, errors, cell_count, exact)

    if MG_solvers:
        # ITS
        name = 'MG F-cycle Solve'
        # The mg_levels_ksp_max_it is half of original depth?
    #    parameters = {
    #   "ksp_type": "preonly",
    #   "pc_type": "mg",
    #   "pc_mg_type": "full",
    #   "mg_levels_ksp_type": "chebyshev",
    #   "mg_levels_ksp_max_it": 2,
    #   "mg_levels_pc_type": "jacobi"
    #    }
        parameters = {
       "ksp_type": "preonly",
       "pc_type": "mg",
       "pc_mg_type": "full",
        }
        times, iterations, errors, cell_count = sub_solver(name, parameters, linear_var_solve, V, a, L, bcs, error, times, iterations, errors, cell_count, exact)

    if CG_solvers:
        # RTOL
        name = 'MG F-cycle PC + CG Solve'
        # The mg_levels_ksp_max_it is half of original depth?
    #    parameters = {
    #   "ksp_type": "cg",
    #   "pc_type": "mg",
    #   "pc_mg_type": "full",
    #   "mg_levels_ksp_type": "chebyshev",
    #   "mg_levels_ksp_max_it": 2,
    #   "mg_levels_pc_type": "jacobi"
    #    }
        parameters = {
       "ksp_type": "cg",
       "pc_type": "mg",
       "pc_mg_type": "full",
       "mg_levels_ksp_max_it": 1,
        }
        times, iterations, errors, cell_count = sub_solver(name, parameters, linear_var_solve, V, a, L, bcs, error, times, iterations, errors, cell_count, exact)

    return times, iterations, errors, cell_count

def convergence(compare_solvers, error, mesh_list, depth, family, degree_FEM, sub_solver):
    full_time_dict = dict()
    full_cell_dict = dict()
    full_err_dict = dict()
    full_iter_dict = dict()
    for mindex, mesh in enumerate(mesh_list):
        xmesh = mesh
        ymesh = mesh
        mesh = UnitSquareMesh(xmesh, ymesh)
        hierarchy = MeshHierarchy(mesh, depth)


        mesh = hierarchy[-1] # Grab the finest mesh
        V = FunctionSpace(mesh, family, degree=degree_FEM)
        u = TrialFunction(V)
        v = TestFunction(V)
        a = dot(grad(u), grad(v)) * dx
        
        bcs = DirichletBC(V, zero(), (1, 2, 3, 4))

        # Forcing Function
        x, y = SpatialCoordinate(mesh)
        f = -0.5*pi*pi*(4*cos(pi*x) - 5*cos(pi*x*0.5) + 2)*sin(pi*y)
        exact = sin(pi*x)*tan(pi*x*0.25)*sin(pi*y)
        L = f * v * dx

        times, iterations, errors, cell_count = compare_solvers(u, error, sub_solver, V=V, a=a, L=L, bcs=bcs, exact=exact)
        for name in times.keys():
            if mindex == 0:
                full_time_dict[name] = [times[name]]
                full_cell_dict[name] = [cell_count[name]]
                full_err_dict[name] = [errors[name]]
                full_iter_dict[name] = [iterations[name]]
            else:
                full_time_dict[name].append(times[name])
                full_cell_dict[name].append(cell_count[name])
                full_err_dict[name].append(errors[name])
                full_iter_dict[name].append(iterations[name])
    return full_time_dict, full_cell_dict, full_err_dict, full_iter_dict


def subplotter(x, y, xlabel, ylabel, image_dir):
    savename = xlabel+ylabel
    markers = ['.', '*', '^', 's', 'v']
    marker_index = 0
    for name in x.keys():
        plt.plot(x[name], y[name], label=name, marker=markers[marker_index%len(markers)])
        marker_index += 1
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlabel == 'Mesh Elements' or xlabel == 'Error':
        plt.xscale('log')
    if ylabel == 'Mesh Elements' or ylabel == 'Error':
        plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{image_dir}/{savename}.png')
    plt.close()
    return 


def plot_gens(times, cells, errs, iters, subplotter, image_dir):
    print('Plotting')
    # Go through each name and plot all together on same figure
    # y time, x cells
    x = cells
    y = times
    xlabel = 'Mesh Elements'
    ylabel = 'Time [s]'
    subplotter(x, y, xlabel, ylabel, image_dir)
    # y time, x iters
    x = iters
    y = times
    xlabel = 'Iterations'
    ylabel = 'Time [s]'
    subplotter(x, y, xlabel, ylabel, image_dir)
    # y iters, x times
    x = times
    y = iters
    xlabel = 'Time [s]'
    ylabel = 'Iterations'
    subplotter(x, y, xlabel, ylabel, image_dir)
    # y iters, x cells
    x = cells
    y = iters
    xlabel = 'Mesh Elements'
    ylabel = 'Iterations'
    subplotter(x, y, xlabel, ylabel, image_dir)
    # y err, x times
    x = times
    y = errs
    xlabel = 'Time [s]'
    ylabel = 'Error'
    subplotter(x, y, xlabel, ylabel, image_dir)
    # y err, x iters
    x = iters
    y = errs
    xlabel = 'Iterations'
    ylabel = 'Error'
    subplotter(x, y, xlabel, ylabel, image_dir)
    # y err, x cells
    x = cells
    y = errs
    xlabel = 'Mesh Elements'
    ylabel = 'Error'
    subplotter(x, y, xlabel, ylabel, image_dir)
    return


if __name__ == '__main__':
    initial_start = time.time()
    run_type = 'mg-nmg'
    CG_solvers = True
    MG_solvers = False
    depth = 4
    family = 'Lagrange' #CG
    degree_FEM = 1
    min_mesh = 1
    max_mesh = 50
    image_dir = f'./images-d{depth}-f{family}-r{degree_FEM}-m{max_mesh}'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    mesh_list = np.arange(min_mesh, max_mesh)
    #mesh_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # 20 is a good value for speed and good results

    # Current setup uses 1e-7 constant rtol

    if run_type == 'mg-nmg':
        # Compare non-MG with MG
        times, cells, errs, iters = convergence(compare_solvers, error, mesh_list, depth, family, degree_FEM, sub_solver)
        plot_gens(times, cells, errs, iters, subplotter, image_dir)
    elif run_type == 'mg-mg':
        # Compare MG against MG 
        times, cells, errs, iters = convergence(compare_MG_solvers, error, mesh_list, depth, family, degree_FEM, sub_solver_MG)
        plot_gens(times, cells, errs, iters, subplotter, image_dir)
    else:
        raise Exception

##    fig, axes = plt.subplots()
##    triplot(mesh, axes=axes)
##    axes.legend();
##    plt.savefig('check_mesh_coarse.png')
##    plt.close()
##    fig, axes = plt.subplots()
##    triplot(hierarchy[-1], axes=axes)
##    axes.legend();
##    plt.savefig('check_mesh_fine.png')
##    plt.close()

    net_time = time.time() - initial_start
    print(f'Total Time: {net_time}s')
    print('MODIFIED SUB-SOLVER')
