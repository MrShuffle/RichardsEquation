# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 20:48:14 2022

@author: jakob
"""

# This file contains the script to run Example 3 for a fixed setup (mesh, timestep etc.)

import time

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
import scipy as sci
import sympy as sp

from RichardsEqFEM.source.basisfunctions.lagrange_element import finite_element
from RichardsEqFEM.source.LocalGlobalMapping.map_P1 import \
    Local_to_Global_table
from RichardsEqFEM.source.MatrixAssembly.Model_class_parallell import LN_alg
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
    val = ((k_abs) * ((thetaa - the_r) / (the_s - the_r)) ** (1 / 2)) * (
        1
        - (
            sp.functions.elementary.complexes.Abs(
                1
                - (
                    sp.functions.elementary.complexes.Abs(
                        (thetaa - the_r) / (the_s - the_r)
                    )
                )
                ** exp_1
            )
        )
        ** exp_2
    ) ** 2

    return val


# Numpy versions (faster) of saturation and mobility
def theta_np(u):

    return np.piecewise(
        u,
        [u < 0, u >= 0],
        [
            lambda u: the_r
            + (the_s - the_r) * (1 + (np.abs(-a_g * u)) ** n_g) ** (-exp_2),
            the_s,
        ],
    )


def theta_prime_np(u):
    # NOTE: Created using sympy.diff applied to theta.
    # val = sp.Piecewise((-0.132919732761843*sp.Abs(u)**1.9*sp.sign(u)/(0.177557751485229*sp.Abs(u)**2.9 + 1)**1.6551724137931, u < 0), (0, True))

    val = np.piecewise(
        u,
        [u < 0, u >= 0],
        [lambda u:
            -0.0477323587424072*np.abs(u)**1.06
            * np.sign(u)/(0.1699265174169*np.abs(u)
                          ** 2.06 + 1)**1.51456310679612,
            0,
         ],
    )

    return val


def K_np(thetaa):

    val = ((k_abs) * ((thetaa - the_r) / (the_s - the_r)) ** (1 / 2)) * (
        1 - (1 - ((thetaa - the_r) / (the_s - the_r)) ** exp_1) ** exp_2
    ) ** 2

    return val


def K_prime_np(x):
    return (0.0481757787729203
            * (1 - np.abs(np.abs(3.77358490566038*x - 0.494339622641509)**1.94339622641509 - 1)**0.514563106796116)**2
            / (x - 0.131)**0.5 - 0.727181566383703
            * (1 - np.abs(np.abs(3.77358490566038*x - 0.494339622641509)**1.94339622641509 - 1)**0.514563106796116)*(x - 0.131)**0.5
            * np.abs(3.77358490566038*x - 0.494339622641509)**0.943396226415094
            * np.sign(3.77358490566038*x - 0.494339622641509)
            * np.sign(np.abs(3.77358490566038*x - 0.494339622641509)**1.94339622641509 - 1)
            / np.abs(np.abs(3.77358490566038*x - 0.494339622641509)**1.94339622641509 - 1)**0.485436893203884)


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
    # print(theta_prime,'theta')
    # print('k',K_prime)
    # # assert False

    # ! ---- Mesh

    # Define mesh partitions
    x_part = 40
    y_part = 60

    # Physical dimensions
    phys_dim = [2, 3]

    # Create grid
    g = pp.StructuredTriangleGrid(np.array([x_part, y_part]), phys_dim)
    g.compute_geometry()

    coordinates = g.nodes[0:2]

    # ! ---- FEM

    order = 1  # Order of polynomial basis

    # Lagrange element and local to global map
    element = finite_element(order)
    d = Local_to_Global_table(g, element, x_part, y_part)

    # Initialize the FE solution
    u_h = np.zeros((d.total_geometric_pts, 1))

    for k in range(d.total_geometric_pts):
        u_h[k]= 1-coordinates[1][k]

    # Initalize auxiliary functions (iterates, etc.)
    psi_k = u_h.copy()
    psi_t = u_h.copy()
    psi_L_old = u_h.copy()

    # ! ---- Boundary conditions

    # Extract boundary nodes
    b_nodes = d.boundary
    sec_b_nodes = np.zeros((int(y_part/3)+1),dtype=int)
    first_b_nodes = np.zeros((int(x_part/2)+1),dtype=int)
    
    # Find dirichlet boundary nodes
    j=0
    n=0
    for i in range(len(b_nodes)):
        
        if coordinates[0][b_nodes[i]]==2 and 1>=coordinates[1][b_nodes[i]]>=0:
            
            sec_b_nodes[j] = b_nodes[i]
            j=j+1
            
        if coordinates[1][b_nodes[i]]==3 and 1>=coordinates[0][b_nodes[i]]>=0:
            first_b_nodes[n] = b_nodes[i]
            n=n+1


    first_b_nodes = np.flip(first_b_nodes)

    # Define boundary values
    def boundary_one(x,y,t):
        return -2+2.2*t/(1/16)

    def boundary_one_alt(x,y):
        return 0.2

    def boundary_two(x,y):
        return 1-y

    # ! ---- Time discretization
    timesteps = 9
    t = 0
    dt = 1/48

    # ! ---- Solver parameters
    L = 3.501*10**(-2)
    TOL = 10 ** (-7)

    # ! ---- Solver
    scheme = LN_alg(
        L, dt, d, g, order, psi_t, theta_np, K_np, theta_prime_np, K_prime_np, f
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

        # Update time
        t = t + dt
        scheme.update_at_newtime(psi_t, t)

        # ! ---- Nonlinear iterations
        while True:

            # ! ---- Linearization step

            tic = time.time()
            scheme.update_at_iteration(psi_k, ind, Switch)
            scheme.assemble(psi_k, Switch)

            lhs = scheme.lhs
            rhs = scheme.rhs
            if t <= 1/16:
                lhs,rhs = dirichlet_BC_func(boundary_one,first_b_nodes,lhs,rhs,coordinates,t,time=True)
            else:
                lhs,rhs = dirichlet_BC_func(boundary_one_alt,first_b_nodes,lhs,rhs,coordinates,t,time=False)
            lhs,rhs = dirichlet_BC_func(boundary_two,sec_b_nodes,lhs,rhs,coordinates,t,time=False)

            psi = sci.sparse.linalg.spsolve(lhs, rhs)

            psi = np.resize(psi, (psi.shape[0], 1))
            time_assemble_and_solve += time.time() - tic
            print(f"CPU time for assemble and solve: {time.time() - tic}.")
            
            
            # ! ---- Adaptive scheme

            if Switch:
                N_count += 1  # Update Newton count

                # Compute Newton to L-scheme switching indicators
                tic = time.time()
                scheme.N_to_L_eta(psi, psi_k)
                time_N_to_L += time.time() - tic
                print(f"CPU time for N to L: {time.time() - tic}.")

                # Stopping criterion
                valstop = scheme.linear_norm
                #print(scheme.eta_NtoL)
                if scheme.eta_NtoL > 1:
                    Switch = False

                    if ind == 1:

                        # Failure at first Newton iteration - apply L-scheme step
                        tic = time.time()
                        scheme.assemble(psi_k, Switch)

                        lhs = scheme.lhs
                        rhs = scheme.rhs

                        if t <= 1/16:
                            lhs,rhs = dirichlet_BC_func(boundary_one,first_b_nodes,lhs,rhs,coordinates,t,time=True)
                        else:
                            lhs,rhs = dirichlet_BC_func(boundary_one_alt,first_b_nodes,lhs,rhs,coordinates,t,time=False)
                        lhs,rhs = dirichlet_BC_func(boundary_two,sec_b_nodes,lhs,rhs,coordinates,t,time=False)

                        psi = sci.sparse.linalg.spsolve(lhs, rhs)
                        time_assemble_and_solve += time.time() - tic
                        print("CPU time for assemble and solve: {time.time() - tic}.")

                        psi = np.resize(psi, (psi.shape[0], 1))
                        L_count += 1
                        count += 1
                        ind = 0

                        # Compute L-scheme to Newton switching indicators
                        tic = time.time()
                        scheme.L_to_N_eta(psi, psi_k)
                        time_L_to_N += time.time() - tic
                        print(f"CPU time of L to N: {time.time() - tic}.")

                        # Cache the latest L-scheme solution
                        psi_L_old = psi.copy()

                        # Stopping criterion
                        valstop = scheme.linear_norm

                        if scheme.eta_LtoN < 1.5:
                            # Switch to Newton in the next iteration
                            Switch = True
                            ind = 1
                        else:
                            # Switch to L-scheme in the next iteration
                            Switch = False
                    else:

                        # Reset to last L-scheme iteration
                        ind = 1
                        psi = psi_L_old.copy()

                else:
                    # Switch to Newton's method in the next iteration
                    Switch = True
                    tic = time.time()

                    # Determine the linearization error
                    #scheme.linearization_error(psi_k, psi - psi_k)
                    time_linearization_error += time.time() - tic
                    print(f"CPU time for linearization error {time.time() - tic}.")

                    # Stopping criterion
                    #valstop = scheme.linear_norm

                    # Prepare for the next iteration
                    ind = 0
            else:

                L_count += 1  # Update L-scheme counter

                # Estimate C_N^j
                tic = time.time()
                scheme.estimate_CN(psi)
                time_CN += time.time() - tic
                print(f"CPU time for CN: {time.time() - tic}.")

                # Compute L-scheme to Newton switching indicator
                tic = time.time()
                scheme.L_to_N_eta(psi, psi_k)
                time_L_to_N += time.time() - tic
                print(f"CPU fime for L to N: {time.time() - tic}.")

                # Stopping criterion
                valstop = scheme.linear_norm

                if scheme.eta_LtoL >= 1:  # Check failulre of L-scheme

                    scheme.update_L(2 * scheme.L)  # Increase L
                    psi = psi_t

                elif scheme.CN >= 2:
                    # Switch to L-scheme
                    Switch = False
                    ind = 1

                elif scheme.eta_LtoN < 1.5:
                    # Switch to Newton
                    Switch = True
                    ind = 1
                else:
                    # Switch to L-scheme
                    Switch = False

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
        count_tot = count_tot + count
        L_count_tot = L_count_tot + L_count
        N_count_tot = N_count_tot + N_count

        # Propagate the solution in time
        psi_t = psi.copy()
        psi_k = psi.copy()

        # Plotting
        if False:
            psi = psi.squeeze()

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

    print("Total number of iterations", count_tot)
    print("Total", "Newton iterations", N_count_tot, "L-scheme iterations", L_count_tot)
    print(f"CPU time assemble and solve {time_assemble_and_solve}.")
    print(f"CPU time L to N {time_L_to_N}.")
    print(f"CPU time N to L {time_N_to_L}.")
    print(f"CPU time CN {time_CN}.")
    print(f"CPU time linearzation error {time_linearization_error}.")
    print(f"Total time {time.time() - tic0}.")
