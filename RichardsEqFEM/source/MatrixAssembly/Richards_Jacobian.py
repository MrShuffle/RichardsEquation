# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:40:41 2022

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





def quad_rule_jacobian(quadrature_deg,e,cn, j, P_El,local_vals,i):
    # Mapping
    [J, c, J_inv] = reference_to_local(e, P_El.corners,cn)
    
    # Fetch quadrature points
    quadrature = gauss_quadrature_points(quadrature_deg)
    quadrature_points = quadrature[:,0:2]
    Phi = P_El.phi_eval(quadrature_points)
    dPhi = P_El.grad_phi_eval(quadrature_points) 
    ### this fixes problem is there a way to do it efficiently????
    quadrature_pt = np.array([quadrature_points]).T
    
     
    local_vals_in_Q = np.dot(local_vals.val.T.reshape(len(local_vals.val)),Phi)
    
    del quadrature_points
    
    val=0
    transform = J_inv@J_inv.T
    
    for k in range(len(dPhi)):
        
        vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
        q_i = J@vec +c # transformed quadrature points
        w_i = quadrature[k][2] # weights
        
      
        val += local_vals.K_prime_Q[k]*local_vals_in_Q[k]*w_i*Phi[k][i]*dPhi[k][i]@transform@dPhi[k][j].T
            
    val = 0.5*val 
    #print(val, local_vals.K_prime_Q)
    return val
    

def Jacobian_matrices(g,element,d,K,theta,K_prime,theta_prime,psi):
    order = element.degree 
    # Initialize source term vector
    A = np.zeros((d.total_geometric_pts, d.total_geometric_pts))
    
    # Extract geometric information
    elements = g.cell_nodes()
    points = g.nodes
    loc_node_idx = d.local_dofs_corners
    
    # Manually set quadrature degree
    quadrature_deg=2

    for e in range(g.num_cells):
        cn = d.mapping[:,e]
        corners = points[0:2,cn[loc_node_idx]] 
        
        # PK element
        P_El = global_element_geometry(element, corners,g,order)
        [J, c, J_inv] = reference_to_local(e, P_El.corners,cn)
        jac = np.linalg.det(J)
        if P_El.degree ==1:
            psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]]])
        if P_El.degree ==2:
            if (e % 2) == 0:
                psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]],psi[cn[3]],psi[cn[4]],psi[cn[5]]])
            else:
                psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]],psi[cn[3]],psi[cn[4]],psi[cn[5]]])
                
        local_vals = localelement_function_evaluation(K,theta,K_prime,theta_prime,psi_local,P_El)
        # Local assembler
        for j in range(P_El.num_dofs):
            for i in range(P_El.num_dofs):
                A[cn[j],cn[i]] = A[cn[j],cn[i]] + \
                    quad_rule_jacobian(quadrature_deg,e,cn, i, P_El,local_vals,j)*np.abs(jac)
                
    return A
def quad_rule_jacobiansat(quadrature_deg,e,cn, j, P_El,local_vals,i):
    # Mapping
    [J, c, J_inv] = reference_to_local(e, P_El.corners,cn)
    
    # Fetch quadrature points
    quadrature = gauss_quadrature_points(quadrature_deg)
    quadrature_points = quadrature[:,0:2]
    Phi = P_El.phi_eval(quadrature_points)
    #dPhi = P_El.grad_phi_eval(quadrature_points) 
    
    
    val=0
    
    for k in range(len(Phi)):
       
        w_i = quadrature[k][2] # weights
        
        val += local_vals.theta_prime_Q[k]*w_i*Phi[k][j]*Phi[k][i]
            
    val = 0.5*val 
    #print(val, local_vals.K_prime_Q)
    return val


def Jacobian_saturation(g,element,d,K,theta,K_prime,theta_prime,psi):
    order = element.degree 
    # Initialize source term vector
    A = np.zeros((d.total_geometric_pts, d.total_geometric_pts))
    
    # Extract geometric information
    elements = g.cell_nodes()
    points = g.nodes
    loc_node_idx = d.local_dofs_corners
    
    # Manually set quadrature degree
    quadrature_deg=2

    for e in range(g.num_cells):
        cn = d.mapping[:,e]
        corners = points[0:2,cn[loc_node_idx]] 
        
        # PK element
        P_El = global_element_geometry(element, corners,g,order)
        [J, c, J_inv] = reference_to_local(e, P_El.corners,cn)
        jac = np.linalg.det(J)
        if P_El.degree ==1:
            psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]]])
        if P_El.degree ==2:
            if (e % 2) == 0:
                psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]],psi[cn[3]],psi[cn[4]],psi[cn[5]]])
            else:
                psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]],psi[cn[3]],psi[cn[4]],psi[cn[5]]])
                
        local_vals = localelement_function_evaluation(K,theta,K_prime,theta_prime,psi_local,P_El)
        # Local assembler
        for j in range(P_El.num_dofs):
            for i in range(P_El.num_dofs):
                A[cn[j],cn[i]] = A[cn[j],cn[i]] + \
                    quad_rule_jacobiansat(quadrature_deg,e,cn, j, P_El,local_vals,i)*np.abs(jac)
                
    return A
                