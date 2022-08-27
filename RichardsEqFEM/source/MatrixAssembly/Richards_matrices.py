# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:19:33 2022

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



def permability_matrix_assembly(d,g,order,K,theta,K_prime,theta_prime,psi):
 
    
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
        dPhi = P_El.grad_phi_eval(quadrature_points) 
        Phi = P_El.phi_eval(quadrature_points)
        #Phi = Phi[::-1]
        if P_El.degree ==1:
            psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]]])
        if P_El.degree ==2:
            if (e % 2) == 0:
                psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]],psi[cn[3]],psi[cn[4]],psi[cn[5]]])
            else:
                psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]],psi[cn[3]],psi[cn[4]],psi[cn[5]]])
        local_vals = localelement_function_evaluation(K,theta,K_prime,theta_prime,psi_local,P_El)

        [J2, c2, Jinv] = reference_to_local(e,P_El.corners,cn)
        jac = np.linalg.det(J2)
        transform = Jinv@Jinv.T
        for j in range(P_El.num_dofs):
            for i in range(P_El.num_dofs):
                val=0
                for k in range(len(Phi)):
                    val = val +local_vals.K_in_Q[k]*quadrature[k][2]*dPhi[k][i]@transform@dPhi[k][j].T
                #print(val)
                B[cn[j]][cn[i]] = B[cn[j]][cn[i]] + 0.5*np.abs(jac)*val
    return B

def saturation_matrix_assembly(d,g,order,K,theta,K_prime,theta_prime,psi):
 
    
    element = finite_element(order)
    #d = Local_to_Global_table(g, element)
    B = np.zeros((d.total_geometric_pts, 1))  # initalize matrix
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
        #Phi = Phi[::-1]
        if P_El.degree ==1:
            psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]]])
        if P_El.degree ==2:
            if (e % 2) == 0:
                psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]],psi[cn[3]],psi[cn[4]],psi[cn[5]]])
            else:
                psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]],psi[cn[3]],psi[cn[4]],psi[cn[5]]])
        local_vals = localelement_function_evaluation(K,theta,K_prime,theta_prime,psi_local,P_El)

        [J2, c2, Jinv] = reference_to_local(e,P_El.corners,cn)
        jac = np.linalg.det(J2)

        for j in range(P_El.num_dofs):
            #for i in range(P_El.num_dofs):
            val=0
            for k in range(len(Phi)):
                
                val = val +local_vals.theta_in_Q[k]*quadrature[k][2]*Phi[k][j]
                #print(val)
            B[cn[j]] = B[cn[j]] + 0.5*np.abs(jac)*val
    return B