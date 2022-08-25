# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 17:16:35 2022

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



def grav_quad_2d(quadrature_deg,element_num,cn, loc_node, P_El,local_vals):
    
    gravity = np.array([0,1])
    # Mapping
    [J, c, J_inv] = reference_to_local(element_num, P_El.corners,cn)
    
    # Fetch quadrature points
    quadrature = gauss_quadrature_points(quadrature_deg)
    quadrature_points = quadrature[:,0:2]
    #Phi = P_El.phi_eval(quadrature_points)
    dPhi = P_El.grad_phi_eval(quadrature_points) 
    ### this fixes problem is there a way to do it efficiently????
    quadrature_pt = np.array([quadrature_points]).T
    
    del quadrature_points
    
    val=0
    #print(local_vals.K_in_Q)
    for k in range(len(dPhi)):
        vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
        q_i = J@vec +c # transformed quadrature points
        w_i = quadrature[k][2] # weights
        val += local_vals.K_in_Q[k]*w_i*np.inner(gravity,dPhi[k][loc_node]@J_inv.T)
    
    val = 0.5*val 
    return val

def gravity_vector(element,d,quadrature_deg, g,K,theta,K_prime,theta_prime,psi):
    
    order = element.degree 
    # Initialize source term vector
    grav_vect = np.zeros((d.total_geometric_pts, 1))
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
            grav_vect[cn[j]] = grav_vect[cn[j]] + \
                grav_quad_2d(quadrature_deg,e,cn, j, P_El,local_vals)*np.abs(jac)
                
    return grav_vect

def grav_quad_2d_prime(quadrature_deg,element_num,cn, loc_node, P_El,local_vals,i):
    
    gravity = np.array([0,1])
    # Mapping
    [J, c, J_inv] = reference_to_local(element_num, P_El.corners,cn)
    
    # Fetch quadrature points
    quadrature = gauss_quadrature_points(quadrature_deg)
    quadrature_points = quadrature[:,0:2]
    Phi = P_El.phi_eval(quadrature_points)
    dPhi = P_El.grad_phi_eval(quadrature_points) 
    ### this fixes problem is there a way to do it efficiently????
    quadrature_pt = np.array([quadrature_points]).T
    
    del quadrature_points
    
    val=0
    #print(local_vals.K_in_Q)
    for k in range(len(dPhi)):
        vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
        q_i = J@vec +c # transformed quadrature points
        w_i = quadrature[k][2] # weights
     
        val += local_vals.K_prime_Q[k]*w_i*Phi[k][i]*np.inner(gravity,dPhi[k][loc_node]@J_inv.T)
            
    val = 0.5*val 
    return val

def gravity_vector_prime(element,d,quadrature_deg, g,K,theta,K_prime,theta_prime,psi):
    
    order = element.degree 
    # Initialize source term vector
    grav_vect = np.zeros((d.total_geometric_pts, d.total_geometric_pts))
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
                grav_vect[cn[j],cn[i]] = grav_vect[cn[j],cn[i]] + \
                    grav_quad_2d_prime(quadrature_deg,e,cn, j, P_El,local_vals,i)*np.abs(jac)
                
    return grav_vect
