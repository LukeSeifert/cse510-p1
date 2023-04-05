# Main run file
from firedrake import *
import time
import matplotlib.pyplot as plt

# PETSc
# Solve defaults {'ksp_rtol': 1e-07, 'ksp_type': 'preonly', 'mat_mumps_icntl_14': 200, 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps', 'pc_type': 'lu'}
# ksp_type: https://w3.pppl.gov/m3d/petsc-dev/docs/manualpages/KSP/KSPType.html
    # pre_only: stub method that ONLY applies the preconditioner. May be used in inner iterations.
    # cg: preconditioned conjugate gradient iterative method.
# pc_type: https://w3.pppl.gov/m3d/petsc-dev/docs/manualpages/PC/PCType.html
    # pcmg: Multigrid preconditioning (optional settings https://w3.pppl.gov/m3d/petsc-dev/docs/manualpages/PC/PCMG.html)
        # Full/default types https://w3.pppl.gov/m3d/petsc-dev/docs/manualpages/PC/PCMGType.html



def run_solve(V, a, L, bcs, parameters):
    u = Function(V)
    # https://www.firedrakeproject.org/firedrake.html?highlight=solve#firedrake.solving.solve
    solve(a == L, u, bcs=bcs, solver_parameters=parameters)
    return u

def error(u):
    expect = Function(V).interpolate(exact)
    return norm(assemble(u - expect))

def compare_solvers(u, error, run_solve, **kwargs):
    start = time.time()
    u = run_solve(V, a, L, bcs, {"ksp_type": "preonly", "pc_type": "lu"})
    end = time.time()
    print('Direct LU solve error ', error(u))
    print(f'Took {end-start}s')
   
    start = time.time()
    u = run_solve(V, a, L, bcs, {"ksp_type": "cg", "pc_type": "mg"})
    end = time.time()
    print('MG V-cycle PC + CG solver error', error(u))
    print(f'Took {end-start}s')

    # The mg_levels_ksp_max_it is half of original depth?
    parameters = {
   "ksp_type": "preonly",
   "pc_type": "mg",
   "pc_mg_type": "full",
   "mg_levels_ksp_type": "chebyshev",
   "mg_levels_ksp_max_it": 2,
   "mg_levels_pc_type": "jacobi"
    }

    start = time.time()
    u = run_solve(V, a, L, bcs, parameters)
    end = time.time()
    print('MG F-cycle error', error(u))
    print(f'Took {end-start}s')
    return

def convergence():
    return


if __name__ == '__main__':



    xmesh = 4
    ymesh = 4
    depth = 1
    family = 'Lagrange' #CG
    degree_FEM = 1

    print(f'n = {xmesh*ymesh*2}')

    mesh = UnitSquareMesh(xmesh, ymesh)
    hierarchy = MeshHierarchy(mesh, depth)


#    fig, axes = plt.subplots()
#    triplot(mesh, axes=axes)
#    axes.legend();
#    plt.savefig('check_mesh_coarse.png')
#    plt.close()
#    fig, axes = plt.subplots()
#    triplot(hierarchy[-1], axes=axes)
#    axes.legend();
#    plt.savefig('check_mesh_fine.png')
#    plt.close()

    # Defining the Poisson equation problem
    # d/dx^2 + d/dy^2 = f
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

    # Solving
    # Conjugate Gradient preconditioned by geometric multigrid v-cycle
    #solve_params = {'ksp_type': 'cg',
    #                'pc_type' : 'mg'}
    #u = run_solve(V, a, L, bcs, solve_params)
    compare_solvers(u, error, run_solve, V=V, a=a, L=L, bcs=bcs)
