# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:03:21 2022

@author: jakob
"""

def dirichlet_BC(BC, boundary_segment, A, f, g):
    """
    Imposes dirichlet boundary conditions
    Parameters
    ----------
    BC : Dirichlet boundary value.
    boundary_segment : A matrix containing the nodes at the boundary.
    A : A matrix which originates from the problem.
    f : A vector which originates from the problem.
    Returns
    -------
    A : A matrix modified to impose dirichlet BCs.
    f : A vector modified to impose dirichlet BCs.
    """

    for e in range(len(boundary_segment)):


        A[boundary_segment[e], :] = 0
        A[boundary_segment[e],boundary_segment[e]] = 1
        f[boundary_segment[e]] = BC

    return A, f

def dirichlet_BC_func(BC, boundary_segment, A, f, coords,t,time=False):
    """
    Imposes dirichlet boundary conditions
    Parameters
    ----------
    BC : Dirichlet boundary value being a function.
    boundary_segment : A matrix containing the nodes at the boundary.
    A : A matrix which originates from the problem.
    f : A vector which originates from the problem.
    Returns
    -------
    A : A matrix modified to impose dirichlet BCs.
    f : A vector modified to impose dirichlet BCs.
    """

    for e in range(len(boundary_segment)):


        A[boundary_segment[e], :] = 0
        A[boundary_segment[e],boundary_segment[e]] = 1
        #print(coords[0][boundary_segment[e]],coords[1][boundary_segment[e]])
        if time == True:
            f[boundary_segment[e]] = BC(coords[0][boundary_segment[e]],coords[1][boundary_segment[e]],t)
        else:
            f[boundary_segment[e]] = BC(coords[0][boundary_segment[e]],coords[1][boundary_segment[e]])

    return A, f