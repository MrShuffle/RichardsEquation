# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:53:17 2022

@author: jakob
"""


import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
import scipy as sci
import sympy as sp

from sympy.utilities.autowrap import ufuncify

from RichardsEqFEM.source.basisfunctions.lagrange_element import finite_element
from RichardsEqFEM.source.LocalGlobalMapping.map_P1 import \
    Local_to_Global_table
from RichardsEqFEM.source.MatrixAssembly.Model_class_parallell import LN_alg
from RichardsEqFEM.source.utils.boundary_conditions import dirichlet_BC
import time

# Van Genuchten parameteres

a_g = 0.551
n_g = 2.9
k_abs = 0.12
the_r = 0.026
the_s = 0.42
exp_1 = n_g / (n_g - 1)
exp_2 = (n_g - 1) / n_g



#def K_sp(thetaa):
#    val = ((k_abs) * ((thetaa - the_r) / (the_s - the_r)) ** (1 / 2)) * (
#        1
#        - (
#            sp.functions.elementary.complexes.Abs(
#                1
#                - (
#                    sp.functions.elementary.complexes.Abs(
#                        (thetaa - the_r) / (the_s - the_r)
#                    )
#                )
#                ** exp_1
#            )
#        )
#        ** exp_2
#    ) ** 2
#
#    return val
#
#
#def K(thetaa):
#    val = ((k_abs) * ((thetaa - the_r) / (the_s - the_r)) ** (1 / 2)) * (
#        1 - (1 - ((thetaa - the_r) / (the_s - the_r)) ** exp_1) ** exp_2
#    ) ** 2
#
#    return val
#
#
#def theta_sp(u):
#
#    val = sp.Piecewise(
#        (the_r + (the_s - the_r) * (1 + (np.abs(-a_g * u)) ** n_g) ** (-exp_2), u < 0),
#        (the_s, u >= 0),
#    )
#    # if u<0:
#    #     val= the_r+(the_s-the_r)*(1+(np.abs(-a_g*u))**n_g)**(-exp_2)
#    # else:
#    #     val=the_s
#    return val

def theta_np(u):

    return np.piecewise(
        u,
        [u<0, u>=0],
        [lambda u: the_r + (the_s - the_r) * (1 + (np.abs(-a_g * u)) ** n_g) ** (-exp_2), the_s]
    )

def theta_prime_np(u):
    #NOTE: Created using sympy.diff applied to theta.
    #val = sp.Piecewise((-0.132919732761843*sp.Abs(u)**1.9*sp.sign(u)/(0.177557751485229*sp.Abs(u)**2.9 + 1)**1.6551724137931, u < 0), (0, True))

    val = np.piecewise(
        u, 
        [u < 0, u>=0],
        [lambda u: -0.132919732761843*np.abs(u)**1.9*np.sign(u)/(0.177557751485229*np.abs(u)**2.9 + 1)**1.6551724137931, 0]
    )

    return val

def K_np(thetaa):

    val = ((k_abs) * ((thetaa - the_r) / (the_s - the_r)) ** (1 / 2)) * (
        1 - (1 - ((thetaa - the_r) / (the_s - the_r)) ** exp_1) ** exp_2
    ) ** 2

    return val

def K_prime_np(x):
    return 0.0955879481815749*(1 - np.abs(np.abs(2.53807106598985*x - 0.065989847715736)**1.52631578947368 - 1)**0.655172413793103)**2/(x - 0.026)**0.5 - 0.970436022147969*(1 - np.abs(np.abs(2.53807106598985*x - 0.065989847715736)**1.52631578947368 - 1)**0.655172413793103)*(x - 0.026)**0.5*np.abs(2.53807106598985*x - 0.065989847715736)**0.526315789473684*np.sign(2.53807106598985*x - 0.065989847715736)*np.sign(np.abs(2.53807106598985*x - 0.065989847715736)**1.52631578947368 - 1)/np.abs(np.abs(2.53807106598985*x - 0.065989847715736)**1.52631578947368 - 1)**0.344827586206897


# Source term
def f(t, x, y):
    if y > 1 / 4:
        val = 0.06 * np.cos(4 / 3 * np.pi * (y)) * np.sin(((x)))
    else:
        val = 0
    return val


if __name__ == "__main__":

    # Compute derivatives
    #x = sp.symbols("x", real=True)
    #K_prime = sp.diff(K_sp(x), x)
    #theta_prime = sp.diff(theta_sp(x), x)
    #print(K_prime)
    #print(theta_prime)

    # Define mesh partitions
    x_part = 40
    y_part = 40
    # Physical dimensions
    phys_dim = [1, 1]
    # Create grid
    g = pp.StructuredTriangleGrid(np.array([x_part, y_part]), phys_dim)
    g.compute_geometry()

    coordinates = g.nodes[0:2]

    order = 1  # Order of polynomial basis

    # Lagrange element and local to global map
    element = finite_element(order)
    d = Local_to_Global_table(g, element, x_part, y_part)

    # Vectorize u_h
    u_h = np.zeros((d.total_geometric_pts, 1))

    for k in range(d.total_geometric_pts):
        if coordinates[1][k] > 1 / 4:
            u_h[k] = -4
        else:
            u_h[k] = -coordinates[1][k] - 1 / 4

    # Initalize
    psi_k = u_h.copy()
    psi_t = u_h.copy()
    psi_L_old = u_h.copy()

    # Extract boundary nodes
    b_nodes = d.boundary
    Dirichlet_boundary = np.zeros((int(len(b_nodes) / 4 + 1)), dtype=int)

    # Find dirichlet boundary nodes
    n = 0
    for i in range(len(b_nodes)):

        if coordinates[1][b_nodes[i]] == 1 and 1 >= coordinates[0][b_nodes[i]] >= 0:
            Dirichlet_boundary[n] = b_nodes[i]
            n = n + 1

    Dirichlet_boundary = np.flip(Dirichlet_boundary)

    timesteps = 1
    t = 0
    dt = 0.01

    L = 0.1

    TOL = 10 ** (-7)

    testL = np.zeros((1000, 1))
    store = np.zeros((1000, 1))
    count_tot = 0
    L_count_tot = 0
    N_count_tot = 0

    scheme = LN_alg(L, dt, d, g, order, psi_t, theta_np, K_np, theta_prime_np, K_prime_np, f)

    bcval = -4
    for j in range(timesteps):
        # Single timestep counter
        count = 0
        L_count = 0
        N_count = 0
        ind = 0

        # Set Switch to false
        Switch = False

        t = t + dt  # Update time
        tic = time.time()
        scheme.update_at_newtime(psi_t, t)
        print(f"Update time: {time.time() - tic}")

        while True:

            tic = time.time()
            scheme.update_at_iteration(psi_k, ind, Switch)
            print(f"update at iteraion {time.time() - tic}")

            tic = time.time()
            scheme.assemble(psi_k, Switch)
            print(f"assemble {time.time() - tic}")

            lhs = scheme.lhs
            rhs = scheme.rhs
            tic = time.time()
            lhs, rhs = dirichlet_BC(bcval, Dirichlet_boundary, lhs, rhs, g)
            print(f"bc {time.time() - tic}")

            tic = time.time()
            psi = sci.sparse.linalg.spsolve(lhs, rhs)
            print(f"solve {time.time() - tic}")

            psi = np.resize(psi, (psi.shape[0], 1))

            if Switch == True:
                N_count += 1  # Update Newton count

                # Compute Newton to L-scheme switching indicators
                tic = time.time()
                scheme.N_to_L_eta(psi, psi_k)
                print(f"N to L {time.time() - tic}")

                # Stopping criterion
                valstop = scheme.linear_norm

                if scheme.eta_NtoL > 1:
                    Switch = False

                    if ind == 1:  # Failure at first Newton iteration

                        tic = time.time()
                        scheme.assemble(psi_k, Switch)
                        print(f"Assemble: {time.time() - tic}.")

                        lhs = scheme.lhs
                        rhs = scheme.rhs

                        tic = time.time()
                        lhs, rhs = dirichlet_BC(bcval, Dirichlet_boundary, lhs, rhs, g)
                        print(f"BC: {time.time() - tic}.")

                        tic = time.time()
                        psi = sci.sparse.linalg.spsolve(lhs, rhs)
                        print(np.linalg.norm(psi))
                        print(f"Solve: {time.time() - tic}.")

                        psi = np.resize(psi, (psi.shape[0], 1))
                        L_count += 1
                        count += 1
                        ind = 0

                        # Compute L-scheme to Newton switching indicators
                        tic = time.time()
                        scheme.L_to_N_eta(psi, psi_k, K_prime, theta_prime)
                        print(f"Switching crit: {time.time() - tic}.")

                        psi_L_old = psi
                        valstop = scheme.linear_norm

                        if scheme.eta_LtoN < 1.5:
                            Switch = True
                            ind = 1
                        else:
                            Switch = False
                    else:  # Reset to last L-scheme iteration
                        ind = 1
                        psi = psi_L_old

                else:
                    Switch = True
                    tic = time.time()
                    scheme.linearization_error(psi_k, psi - psi_k)
                    print(f"linearization error {time.time() - tic}")
                    valstop = scheme.linear_norm
                    ind = 0
            else:

                print("not switching")

                L_count += 1  # Update L-scheme counter

                # Estimate C_N^j
                tic = time.time()
                scheme.estimate_CN(psi)
                print(f"Estimate CN new {time.time() - tic}.")
                # Compute L-scheme to Newton switching indicator
                tic = time.time()
                scheme.L_to_N_eta(psi, psi_k)
                print(f"Estimate L to N {time.time() - tic}.")

                # Stopping criterion
                tic = time.time()
                valstop = scheme.linear_norm

                if scheme.eta_LtoL >= 1:  # Check failulre of L-scheme

                    scheme.update_L(2 * scheme.L)  # Increase L
                    psi = psi_t

                elif scheme.CN >= 2:
                    Switch = False
                    ind = 1

                elif scheme.eta_LtoN < 1.5:
                    print("switch to Newton")
                    Switch = True
                    ind = 1
                else:
                    Switch = False

                print(f"stopping crit: {time.time() - tic}")

            # Update counter
            count = count + 1

            print("Iteration dependent norm:", valstop)
            if valstop <= TOL:  # +TOL*np.linalg.norm(psi):
                break
            else:
                psi_k = psi.copy()

            print()

        print("Total number of iterations: ", count)
        print("Newton iterations", N_count, "L-scheme iterations", L_count)
        count_tot = count_tot + count
        L_count_tot = L_count_tot + L_count
        N_count_tot = N_count_tot + N_count
        psi_t = psi.copy()
        psi_k = psi.copy()

        # Plotting #
        psi = psi.squeeze()
        # Extract geometric information
        coordinates = g.nodes
        xcoords, ycoords = coordinates[0:2]
        elements = g.cell_nodes()

        # local to global map
        cn = d.mapping.T
        flat_list = d.local_dofs_corners

        psi_plot = psi.copy()
        plt.tricontourf(xcoords, ycoords, cn[:, flat_list], psi_plot, 40)
        plt.colorbar()
        plt.show()

    print("Total number of iterations", count_tot)
    print("Total", "Newton iterations", N_count_tot, "L-scheme iterations", L_count_tot)