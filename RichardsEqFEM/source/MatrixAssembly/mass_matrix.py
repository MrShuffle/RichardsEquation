# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:40:16 2022

@author: jakob
"""
import numpy as np
from RichardsEqFEM.source.localevaluation.local_evaluation import localelement_function_evaluation

from scipy.sparse.coo import coo_matrix
import math
from RichardsEqFEM.source.basisfunctions.lagrange_element import global_element_geometry, finite_element
from RichardsEqFEM.source.basisfunctions.Gauss_quadrature_points import *
import porepy as pp
from RichardsEqFEM.source.LocalGlobalMapping.map_local_to_global import Local_to_Global_table
import sympy as sp
from RichardsEqFEM.source.utils.operators import reference_to_local




def mass_matrix(element,d,quadrature_deg, g):
    
    
    order = element.degree
    
    # Number of data points
    numDataPts = d.global_vals[2]*element.num_dofs**2
   
    
    _i = np.empty(numDataPts, dtype=int)
    _j = np.empty(numDataPts, dtype=int)
    
    _data_m = np.empty(numDataPts, dtype=np.double)
    
    
    # Extract geometric information
    elements = g.cell_nodes()
    points = g.nodes
    
    loc_node_idx = d.local_dofs_corners
    
    n=0

    for e in range(g.num_cells):
        
        
        cn = d.mapping[:,e]
        
        corners = points[0:2,cn[loc_node_idx]]  # corners of element
        
        
        # PK element
        P_El = global_element_geometry(element, corners,g,order)
        # Map reference element to local element
        [J, c, J_inv] = reference_to_local(e, P_El.element_coord,cn)
        jac = np.linalg.det(J)
        
        # Fetch quadrature points
        quadrature = gauss_quadrature_points(quadrature_deg)
        quadrature_points = quadrature[:,0:2]
        
        # Evaluate basis functions in quadrature points
        Phi = P_El.phi_eval(quadrature_points)
        weights = 1/2*quadrature[:,2]
        
        # Local mass matrix
        localMass = Phi.T @ (np.multiply(weights, Phi.T).T) * np.abs(jac)
        for k,l in np.ndindex(element.num_dofs,element.num_dofs):
            _i[n] = cn[k]
            _j[n] = cn[l]   
            
            _data_m[n] = localMass[k,l]
            n+=1
            
    M  = coo_matrix((_data_m, (_i, _j))).todense() 
        
    return M

'''following function should produce the same result, verfied that they indeed give the same result.'''

def mass_matrix_alternative(d,g,order):
    """
    Assembles the matrix which originates from integrating the product of two
    shape functions. Typically connected to the time derivative.
    Parameters
    ----------
    g : Geometry generated from porepy
    order : Order of polynomial basis
    Returns
    -------
    B : A matrix.
    """
    element = finite_element(order)
    #d = Local_to_Global_table(g, element)
    B = np.zeros((d.total_geometric_pts, d.total_geometric_pts))  # initalize matrix
    elements = g.cell_nodes()
    points = g.nodes
    loc_node_idx = d.local_dofs_corners
    for e in range(g.num_cells):
        
        #print(e)
        cn = d.mapping[:,e]
        corners = points[0:2,cn[loc_node_idx]]
        
        
        # PK element
        P_El = global_element_geometry(element, corners,g,order)
        
        # quadrature points
        
        quadrature = gauss_quadrature_points(order+1)
        quadrature_points = quadrature[:,0:2]
        Phi = P_El.phi_eval(quadrature_points)
        
      
        [J2, c2, Jinv] = reference_to_local(e,P_El.corners,cn)
        jac = np.linalg.det(J2)

        for j in range(P_El.num_dofs):
            for i in range(P_El.num_dofs):
                val=0
                for k in range(len(Phi)):
                    val = val + quadrature[k][2]*Phi[k][i]*Phi[k][j]
                #print(val)
                B[cn[j]][cn[i]] = B[cn[j]][cn[i]] + 0.5*np.abs(jac)*val
    return B

    
