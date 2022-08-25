# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:33:39 2022

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



def quad_2d(quadrature_deg,element_num,cn, f, loc_node, P_El, t, time):

    # Mapping
    [J, c, J_inv] = reference_to_local(element_num, P_El.corners,cn)
    
    # Fetch quadrature points
    quadrature = gauss_quadrature_points(quadrature_deg)
    quadrature_points = quadrature[:,0:2]
    Phi = P_El.phi_eval(quadrature_points)
   
    ### this fixes problem is there a way to do it efficiently????
    quadrature_pt = np.array([quadrature_points]).T
    
    del quadrature_points
    
    val=0
    if time == True:
        for k in range(len(quadrature_pt[0])):
            vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
            q_i = J@vec +c # transformed quadrature points
            w_i = quadrature[k][2] # weights
            val += w_i*f(t,q_i[0][0],q_i[1][0])*Phi[k][loc_node]
 
    else:
        for k in range(len(quadrature_pt[0])):
            vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
            q_i = J@vec +c # transformed quadrature points
            w_i = quadrature[k][2] # weights
            val += w_i*f(q_i[0][0],q_i[1][0])*Phi[k][loc_node]
                
    val = 0.5*val 
    return val
def Source_term_assembly(f,element,d,quadrature_deg, g, t, time):

    order = element.degree 
    # Initialize source term vector
    f_vect = np.zeros((d.total_geometric_pts, 1))
    # Extract geometric information
    elements = g.cell_nodes()
    points = g.nodes
    loc_node_idx = d.local_dofs_corners
    

    for e in range(g.num_cells):
        cn = d.mapping[:,e]
        corners = points[0:2,cn[loc_node_idx]] 
        
        # PK element
        P_El = global_element_geometry(element, corners,g,order)
        [J, c, J_inv] = reference_to_local(e, P_El.corners,cn)
        jac = np.linalg.det(J)

        # Local assembler
        for j in range(P_El.num_dofs):
            f_vect[cn[j]] = f_vect[cn[j]] + \
                quad_2d(quadrature_deg,e,cn, f, j, P_El, t, time)*np.abs(jac)
    return f_vect