import time

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
import scipy as sci
import sympy as sp

from RichardsEqFEM.source.basisfunctions.lagrange_element import finite_element
from RichardsEqFEM.source.LocalGlobalMapping.map_P1 import \
    Local_to_Global_table
#from RichardsEqFEM.source.MatrixAssembly.Model_class_parallell import LN_alg_heter
from RichardsEqFEM.source.MatrixAssembly.Model_class_heterogeneous_pyFF import LN_alg_heter
from RichardsEqFEM.source.utils.boundary_conditions import dirichlet_BC, dirichlet_BC_func

# ! ---- Hydraulic properties

# Van Genuchten parameteres
a_g = 0.423
n_g = 2.06
k_abs = 4.96*10**(-2)
the_r = 0.131
the_s = 0.396
exp_1 = n_g/(n_g-1)
exp_2 = (n_g-1)/n_g


# Sympy versions of saturation and mobility
def theta_sp(u):

    val = sp.Piecewise(
        (the_r + (the_s - the_r) * (1 + (np.abs(-a_g * u)) ** n_g) ** (-exp_2), u < 0),
        (the_s, u >= 0),
    )
    return val


def K_sp(thetaa):
    val = thetaa**3

    return val


# Numpy versions (faster) of saturation and mobility
def theta_np(u):

    return np.piecewise(
        u,
        [u < 1, u >= 1],
        [
            lambda u: 1/(2-u)**(1/3),
            1,
        ],
    )


def theta_prime_np(u):
    # NOTE: Created using sympy.diff applied to theta.
    # val = sp.Piecewise((-0.132919732761843*sp.Abs(u)**1.9*sp.sign(u)/(0.177557751485229*sp.Abs(u)**2.9 + 1)**1.6551724137931, u < 0), (0, True))

    val = np.piecewise(
        u,
        [u < 1, u >= 1],
        [lambda u:
            1/(3*(2-u)**(4/3)),
            0,
         ],
    )

    return val


def K_np(thetaa):

    val = thetaa**3

    return val


def K_prime_np(x):
    return (3*x**2)


# Source term
def f(t,x,y):
    return 0


# ! ---- Main script

if __name__ == "__main__":

    tic0 = time.time()

    # ! ---- Preliminaries

    # # Compute derivatives and print for determining the numpy versions above.
    # x = sp.symbols("x", real=True)
    # theta_prime = sp.diff(theta_sp(x), x)
    # K_prime = sp.diff(K_sp(x), x)
    # print(theta_prime)
    # print(K_prime)
    # assert False

    # ! ---- Mesh

    # Define mesh partitions
    x_part = 80
    y_part = 80

    # Physical dimensions
    phys_dim = [1, 1]

    # Create grid
    g = pp.StructuredTriangleGrid(np.array([x_part, y_part]), phys_dim)
    g.compute_geometry()

    coordinates = g.nodes[0:2]

    # ! ---- FEM

    order = 1  # Order of polynomial basis


    # Lagrange element and local to global mapping
    element = finite_element(order)
    d = Local_to_Global_table(g, element, x_part, y_part)
    print(d.total_geometric_pts)
    # Initialize the FE solution
    u_h = np.zeros((d.total_geometric_pts, 1))

    for k in range(d.total_geometric_pts):
        if coordinates[1][k] >= 0.5:
            u_h[k] = 0.9
        else:
            u_h[k] = -3

    # Initalize auxiliary functions (iterates, etc.)
    psi_k = u_h.copy()
    psi_t = u_h.copy()
    psi_L_old = u_h.copy()

    # ! ---- Boundary conditions
    # Extract boundary nodes
    b_nodes = d.boundary
    sec_b_nodes = np.zeros((int(y_part/2)),dtype=int)
    first_b_nodes = np.zeros((int(x_part/2)),dtype=int)
    
    # Find dirichlet boundary nodes
    j=0
    n=0
    for i in range(len(b_nodes)):
        
        if coordinates[0][b_nodes[i]]== 1 and 0.5>coordinates[1][b_nodes[i]]>=0:
            
            sec_b_nodes[j] = b_nodes[i]
            j=j+1
            
        if coordinates[1][b_nodes[i]]==1 and 0.5>coordinates[0][b_nodes[i]]>=0:
            first_b_nodes[n] = b_nodes[i]
            n=n+1


    first_b_nodes = np.flip(first_b_nodes)
    
    #print(sec_b_nodes,first_b_nodes)
    # bcval1 = -4  # Constant boundary value
    # bcval2 = -0.2

    # Define boundary values
    def boundary_one(x,y):
        return 0.9

    def boundary_one_alt(t,x,y):
        val=1.1
        return val
    def boundary_two(x,y):
        return -3
    # ! ---- Time discretization
    timesteps = 20
    t = 0
    dt = 0.1

    # ! ---- Solver parameters

    # Stopping criterion
    TOL = 10 ** (-7)

    # L-scheme parameter
    #L = 0.15
    L = 0.33

    # ! ---- Solver
    qr = np.pi/3
    Q = np.array([[np.cos(qr),-np.sin(qr)],
                  [np.sin(qr),np.cos(qr)]])
    K=np.array([[1,0],
                [0,0.5]])
    K_phi = 0.1
    K2=K_phi*Q.T@K@Q
    scheme = LN_alg_heter(
        L, dt, d, g, order, psi_t, theta_np, K_np, theta_prime_np, K_prime_np,f,0.5, K,K2
    )


    # ! ---- Statistics
    
    
    # Iteration counters
    count_tot = 0
    L_count_tot = 0
    N_count_tot = 0

    # Time
    time_assemble_and_solve = 0
    time_N_to_L = 0
    time_L_to_N = 0
    time_CN = 0
    time_linearization_error = 0

    # ! ---- Time loop

    for j in range(timesteps):

        # ! ---- Preliminaries

        # Single timestep counter
        count = 0
        L_count = 0
        N_count = 0
        ind = 0

        # Set Switch to false, i.e., start with L-scheme, to just run Newton's 
        # method set to True
        Switch = False
        Switch = True
        # Update time
        t = t + dt
        scheme.update_at_newtime(psi_t, t)

        # ! ---- Nonlinear iterations
        while True:

            # ! ---- Linearization step
            Switch = True
            tic = time.time()
            scheme.update_at_iteration(psi_k, ind, Switch)
            scheme.assemble(psi_k, Switch)

            lhs = scheme.lhs
            rhs = scheme.rhs
            
            lhs,rhs = dirichlet_BC_func(boundary_one,first_b_nodes,lhs,rhs,coordinates,t,time=False)
            # if t<0.501:
            #     lhs,rhs = dirichlet_BC_func(boundary_one,first_b_nodes,lhs,rhs,coordinates,t,time=False)
            # else:
            #     lhs,rhs = dirichlet_BC_func(boundary_one_alt,first_b_nodes,lhs,rhs,coordinates,t,time=True)
            lhs,rhs = dirichlet_BC_func(boundary_two,sec_b_nodes,lhs,rhs,coordinates,t,time=False)

            psi = sci.sparse.linalg.spsolve(lhs, rhs)

            psi = np.resize(psi, (psi.shape[0], 1))
            time_assemble_and_solve += time.time() - tic
            print(f"CPU time for assemble and solve: {time.time() - tic}.")
            scheme.linearization_error(psi_k, psi - psi_k)
            valstop = scheme.linear_norm
            print(f"CPU time for assemble and solve: {time.time() - tic}.")
            # # ! ---- Adaptive scheme

            # if Switch:
            #     N_count += 1  # Update Newton count

            #     # Compute Newton to L-scheme switching indicators
            #     tic = time.time()
            #     scheme.N_to_L_eta(psi, psi_k)
            #     time_N_to_L += time.time() - tic
            #     print(f"CPU time for N to L: {time.time() - tic}.")

            #     # Stopping criterion
            #     valstop = scheme.linear_norm
            #     print(scheme.eta_NtoL,'switch back to L-scheme')
            #     if scheme.eta_NtoL > 1:
            #         Switch = False

            #         if ind == 1:

            #             # Failure at first Newton iteration - apply L-scheme step
            #             tic = time.time()
            #             scheme.assemble(psi_k, Switch)

            #             lhs = scheme.lhs
            #             rhs = scheme.rhs

            #             lhs,rhs = dirichlet_BC_func(boundary_one,first_b_nodes,lhs,rhs,coordinates,t,time=False)
           
            #             lhs,rhs = dirichlet_BC_func(boundary_two,sec_b_nodes,lhs,rhs,coordinates,t,time=False)


            #             psi = sci.sparse.linalg.spsolve(lhs, rhs)
            #             time_assemble_and_solve += time.time() - tic
            #             print("CPU time for assemble and solve: {time.time() - tic}.")

            #             psi = np.resize(psi, (psi.shape[0], 1))
            #             L_count += 1
            #             count += 1
            #             ind = 0

            #             # Compute L-scheme to Newton switching indicators
            #             tic = time.time()
            #             scheme.L_to_N_eta(psi, psi_k)
            #             time_L_to_N += time.time() - tic
            #             print(f"CPU time of L to N: {time.time() - tic}.")

            #             # Cache the latest L-scheme solution
            #             psi_L_old = psi.copy()

            #             # Stopping criterion
            #             valstop = scheme.linear_norm

            #             if scheme.eta_LtoN < 1.5:
            #                 # Switch to Newton in the next iteration
            #                 Switch = True
            #                 ind = 1
            #             else:
            #                 # Switch to L-scheme in the next iteration
            #                 Switch = False
            #         else:

            #             # Reset to last L-scheme iteration
            #             ind = 1
            #             psi = psi_L_old.copy()

            #     else:
            #         # Switch to Newton's method in the next iteration
            #         Switch = True
            #         tic = time.time()

            #         # Determine the linearization error
            #         #scheme.linearization_error(psi_k, psi - psi_k)
            #         time_linearization_error += time.time() - tic
            #         print(f"CPU time for linearization error {time.time() - tic}.")

            #         # Stopping criterion
            #         #valstop = scheme.linear_norm

            #         # Prepare for the next iteration
            #         ind = 0
            # else:

            #     L_count += 1  # Update L-scheme counter
            #     # Cache the latest L-scheme solution
            #     psi_L_old = psi.copy()
            #     # Estimate C_N^j
            #     tic = time.time()
            #     scheme.estimate_CN(psi)
            #     time_CN += time.time() - tic
            #     print(f"CPU time for CN: {time.time() - tic}.")

            #     # Compute L-scheme to Newton switching indicator
            #     tic = time.time()
            #     scheme.L_to_N_eta(psi, psi_k)
            #     time_L_to_N += time.time() - tic
            #     print(f"CPU fime for L to N: {time.time() - tic}.")

            #     # Stopping criterion
            #     valstop = scheme.linear_norm
            #     print(scheme.eta_LtoN, 'switch to newton')
            #     if scheme.eta_LtoL >= 1:  # Check failulre of L-scheme

            #         scheme.update_L(2 * scheme.L)  # Increase L
            #         psi = psi_t

            #     elif scheme.CN >= 2:
            #         # Switch to L-scheme
            #         Switch = False
            #         ind = 1

            #     elif scheme.eta_LtoN < 1.5: #1.5
            #         # Switch to Newton
            #         Switch = True
            #         ind = 1
            #     else:
            #         # Switch to L-scheme
            #         Switch = False

            # Update counter
            count = count + 1
            
            # # Set to Flase for L-scheme or True for Newton's method
            # Switch = True
            print("Iteration dependent norm:", valstop)
            if valstop <= TOL:
                break
            else:
                psi_k = psi.copy()

        # Management of iteration counts
        print("Total number of iterations: ", count)
        print("Newton iterations", N_count, "L-scheme iterations", L_count)
        print("Current time",t, "Number of time steps;", j)
        print("----------------------------------")
        count_tot = count_tot + count
        L_count_tot = L_count_tot + L_count
        N_count_tot = N_count_tot + N_count

        print("Current total number of iterations for all time steps: ", count_tot)
        print("Total Newton iterations", N_count_tot, "Total L-scheme iterations", L_count_tot)
        print("----------------------------------")
        # Propagate the solution in time
        psi_t = psi.copy()
        psi_k = psi.copy()
        sap=False
        if 0.91<t<1.09 or 1.91<t:
            sap=True
        #sap=True
        # Plotting
        if False:
            psi = psi.squeeze()
            #psitt = theta_np(psi)
            # Extract geometric information
            coordinates = g.nodes
            xcoords, ycoords = coordinates[0:2]
            elements = g.cell_nodes()
            cn = d.mapping.T
            flat_list = d.local_dofs_corners

            psi_plot = psi.copy()
            plt.tricontourf(xcoords, ycoords, cn[:, flat_list], psi_plot, 40)
            plt.colorbar()
            plt.show()
            theta_plot = theta_np(psi)
            plt.tricontourf(xcoords, ycoords, cn[:, flat_list], theta_plot, 40)
            plt.colorbar()
            plt.show()

    print("Total number of iterations", count_tot)
    print("Total", "Newton iterations", N_count_tot, "L-scheme iterations", L_count_tot)
    print(f"CPU time assemble and solve {time_assemble_and_solve}.")
    print(f"CPU time L to N {time_L_to_N}.")
    print(f"CPU time N to L {time_N_to_L}.")
    print(f"CPU time CN {time_CN}.")
    print(f"CPU time linearzation error {time_linearization_error}.")
    print(f"Total time {time.time() - tic0}.")