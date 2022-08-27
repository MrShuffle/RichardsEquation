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





def quad_rule_jacobian(quadrature_deg,e,cn, i,j, P_El,local_vals):
    # Mapping
    [J, c, J_inv] = reference_to_local(e, P_El.corners,cn)
    
    # Fetch quadrature points
    quadrature = gauss_quadrature_points(quadrature_deg)
    quadrature_points = quadrature[:,0:2]
    
    Phi = P_El.phi_eval(quadrature_points)
    dPhi = P_El.grad_phi_eval(quadrature_points) 
    
    
     
    local_vals_in_Q = np.dot(local_vals.val.T.reshape(len(local_vals.val)),Phi)
    
    del quadrature_points
    
    val=0
    transform = J_inv@J_inv.T
    #print(transform, J@J.T)
    for k in range(len(dPhi)):
     
        w_i = quadrature[k][2] # weights
        
      
        #val += local_vals.K_prime_Q[k]*local_vals.val[k]*w_i*Phi[k][i]*dPhi[k][i]@transform@dPhi[k][j].T
        val += local_vals.K_prime_Q[k]*w_i*Phi[k][i]*local_vals.valgrad_Q[k]@transform@dPhi[k][j].T   
    val = 0.5*val 
   
    #print(val[0])
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
        #print(local_vals.K_prime_Q)
        #print(local_vals.K_d_theta)
        # Local assembler
        for j in range(P_El.num_dofs):
            for i in range(P_El.num_dofs):
                A[cn[j],cn[i]] = A[cn[j],cn[i]] + \
                    quad_rule_jacobian(quadrature_deg,e,cn, i,j, P_El,local_vals)*np.abs(jac)
                
    return A
def quad_rule_jacobiansat(quadrature_deg,e,cn,i, j, P_El,local_vals):
    # Mapping
    [J, c, J_inv] = reference_to_local(e, P_El.corners,cn)
    
    # Fetch quadrature points
    quadrature = gauss_quadrature_points(quadrature_deg)
    quadrature_points = quadrature[:,0:2]
    Phi = P_El.phi_eval(quadrature_points)
    #dPhi = P_El.grad_phi_eval(quadrature_points) 
    
    
    val=0
    
    # sum over quadrature points
    for k in range(len(Phi)):
       
        w_i = quadrature[k][2] # weights
        
        val += local_vals.theta_prime_Q[k]*w_i*Phi[k][i]*Phi[k][j]
            
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
                    quad_rule_jacobiansat(quadrature_deg,e,cn, i,j, P_El,local_vals)*np.abs(jac)
                
    return A


def Newton_Assembly_alt(g,element,d,K,theta,K_prime,theta_prime,psi):
    
    order = element.degree
    
    numDataPts = d.global_vals[2]*element.num_dofs**2
    #print(numDataPts)
    
    theta_derivative = np.zeros((d.total_geometric_pts,1))
    
    _i = np.empty(numDataPts, dtype=int)
    _j = np.empty(numDataPts, dtype=int)
    
    _data_s = np.empty(numDataPts, dtype=np.double)
    _data_f = np.empty(numDataPts, dtype=np.double)
    _data_g = np.empty(numDataPts, dtype=np.double)
    _data_m = np.empty(numDataPts, dtype=np.double)

    
    elements = g.cell_nodes()
    points   = g.nodes
    A        = np.zeros((d.total_geometric_pts,d.total_geometric_pts)) # Initialize
    
    loc_node_idx = d.local_dofs_corners
    n=0
    for e in range(g.num_cells):
        
        
        cn = d.mapping[:,e]
 
        
        corners = points[0:2,cn[loc_node_idx]]    
        
        
        
        # PK element
        P_El = global_element_geometry(element,corners,g,order)
        
        [J, c,J_inv] = reference_to_local(e,P_El.corners,cn)
        transform = J_inv.dot(J_inv.transpose())  # J^(-1)*J^(-t); derivative transformation
        # Determinant of tranformation matrix = inverse of area of local elements
        jac = np.linalg.det(J)
        if P_El.degree ==1:
            psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]]])
        if P_El.degree ==2:
            if (e % 2) == 0:
                psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]],psi[cn[3]],psi[cn[4]],psi[cn[5]]])
            else:
                psi_local = np.array([psi[cn[0]],psi[cn[1]],psi[cn[2]],psi[cn[3]],psi[cn[4]],psi[cn[5]]])
        local_vals = localelement_function_evaluation(K,theta,K_prime,theta_prime,psi_local,P_El)
        
        
        # if P_El.degree == 1:
        #     for j in range(3):
        #         if np.isnan( local_vals.theta_d_psi[j]):
        #             theta_derivative[cn[j]]= 0
        #         else:
        #             theta_derivative[cn[j]]= local_vals.theta_d_psi[j]
               
            
        
        quadrature = gauss_quadrature_points(order+1)
        quadrature_points = quadrature[:,0:2]
        
        grad_of_phi = P_El.grad_phi_eval(quadrature_points) 
        shapefunc = P_El.phi_eval(quadrature_points)
        Phi = P_El.phi_eval(quadrature_points)
        
        local_vals_in_Q = np.dot(psi_local.T.reshape(len(psi_local)),shapefunc.T)
        #print(shapefunc.shape)
        dPhi = grad_of_phi
        grad_of_phi = grad_of_phi[0]
        
        G  = np.zeros(dPhi.shape)
        Z  =  np.zeros(dPhi.shape)
        Z2 =  np.zeros(dPhi.shape)
        test = np.zeros((3,1,2))
        Z3 =  np.zeros(Phi.shape)
        bn=0
        #YY=0
        
        for k in range(len(dPhi)):
       
            G[k] = dPhi[k] @ J_inv.T
 
            Z[k] = local_vals.K_in_Q[k]*dPhi[k] @ J_inv.T # (K(psi) nabla v @J.inv.T)
            Z2[k] = local_vals.K_prime_Q[k]*local_vals_in_Q[k]*Phi[k]@dPhi[k] @ J_inv.T 
            #Z2[k] = dPhi[k] @ J_inv.T # (K'(psi)*psi nabla v @J.inv.T)
            #test[k] = local_vals.K_prime_Q[k]*local_vals_in_Q[k]*Phi[k]
          
        #print(bn)        
        #print(Z2)
        # compute local stiffnessmatrix
        localStiffness =1/2* np.tensordot(quadrature[:,2],Z@G.swapaxes(1,2),axes=1)*np.abs(jac)
        
        
        localStiffness_prime =1/2* np.tensordot(quadrature[:,2],Z2@G.swapaxes(1,2),axes=1)*np.abs(jac)
        
   
        
        Z4 = np.zeros(Phi.shape)
        for k in range(len(Phi)):
            Z4[k] = local_vals.theta_prime_Q[k]*Phi[k]
            #Z3[k] = local_vals.K_prime_Q[k]*shapefunc[k]
            # if  np.isnan(local_vals.theta_prime_Q[k]):
            #     Z4[k] = 0*shapefunc[k]
            #     YY =1
        #local_grav = 1/2* np.tensordot(quadrature[:,2],Z3@G.swapaxes(1,2),axes=1)*np.abs(jac)
        local_mass_theta = 1/2*Z4@ (np.multiply(quadrature[:,2], Phi.T)) * np.abs(jac)
        
        for k,l in np.ndindex(element.num_dofs,element.num_dofs):
            _i[n] = cn[k]
            _j[n] = cn[l]   
            
            _data_s[n] = localStiffness[k,l]
            _data_f[n] = localStiffness_prime[k,l]
            #_data_g[n] = local_grav[k,l]
            _data_m[n] = local_mass_theta[k,l]
            
            
            n+=1
            
        
                
    
    A =  coo_matrix((_data_s, (_i, _j))).todense() 
    B =  coo_matrix((_data_f, (_i, _j))).todense() 
    C =  1#coo_matrix((_data_g, (_i, _j))).todense() 
    M  = coo_matrix((_data_m, (_i, _j))).todense() 
    #print(bn)
    #print(YY)
    #print(len(dPhi), len(local_vals.K_prime_Q))
    return B
                