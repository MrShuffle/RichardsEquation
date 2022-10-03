# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 18:55:15 2022

@author: jakob
"""

import numpy as np
from RichardsEqFEM.source.localevaluation.local_evaluation import localelement_function_evaluation_L, localelement_function_evaluation,localelement_function_evaluation_P

from scipy.sparse.coo import coo_matrix
import math
from RichardsEqFEM.source.basisfunctions.lagrange_element import global_element_geometry, finite_element
from RichardsEqFEM.source.basisfunctions.Gauss_quadrature_points import *
import porepy as pp
from RichardsEqFEM.source.LocalGlobalMapping.map_local_to_global import Local_to_Global_table
import sympy as sp
from RichardsEqFEM.source.utils.operators import reference_to_local
#from multiprocessing import Process, Pool, cpu_count, Manager, Queue,freeze_support
#import numba as nb



# Local mass matrix assembly
def local_mass_assembly(element_num,d):
    cn = d.mapping[:,element_num]
    corners = d.geometry.nodes[0:2,cn[d.local_dofs_corners]]  # corners of element
    
    
    # PK element
    P_El = global_element_geometry(d.element, corners,d.geometry,d.degree)
    #Map reference element to local element
    [J, c, J_inv] = reference_to_local(P_El.element_coord)
    det_J = np.abs(np.linalg.det(J))
    a =  gauss_quadrature_points(2) # needs a fix works only for P1 elements
    Phi = P_El.phi_eval(a[:,0:2])
    local_mass = np.zeros((P_El.num_dofs,P_El.num_dofs))
    for j in range(P_El.num_dofs):
        for i in range(P_El.num_dofs):
            valalt = np.sum(np.multiply(1/2*np.multiply(a[:,2],Phi[:][i]),Phi[:][j]))
            
                
            
            local_mass[j,i] += valalt*det_J

    return local_mass


# Local assembly of source term at time t and <theta(psi),phi> which is also a vector
def local_source_saturation_assembly(element_num,d,psi_t):
    cn = d.mapping[:,element_num]
    corners = d.geometry.nodes[0:2,cn[d.local_dofs_corners]]  # corners of element
    
    
    # PK element
    P_El = global_element_geometry(d.element, corners,d.geometry,d.degree)
    #Map reference element to local element
    [J, c, J_inv] = reference_to_local(P_El.element_coord)
    det_J = np.abs(np.linalg.det(J))
    # Fetch quadrature points
    a =  gauss_quadrature_points(2) 
    Phi = P_El.phi_eval(a[:,0:2])
    # Initalize local vectors
    local_source = np.zeros((P_El.num_dofs,1))
    local_saturation = np.zeros((P_El.num_dofs,1))
    
    # Local pressure head values
    psi_local = np.array([psi_t[cn[0]],psi_t[cn[1]],psi_t[cn[2]]])
    
    local_vals = localelement_function_evaluation_L(d.K,d.theta,psi_local,P_El)
    for j in range(P_El.num_dofs):
        val1=0
        val2=0
        for k in range(len(Phi)):
        
            val1 += local_vals.theta_in_Q[k]*a[k][2]*Phi[k][j]
            
            vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
            q_i = J2@vec +c2 # transformed quadrature points
            w_i = a[k][2] # weights
            val2 += w_i*d.f(t,q_i[0][0].item(),q_i[1][0].item())*Phi[k][j]

        local_source[j] = 0.5*val2*det_J
        local_saturation[j] = 0.5*val1*det_J

    return local_source, local_saturation
    
# Local assembly of <theta(psi_k),phi>, <K(psi_k)dPhi,dPhi>, <K(psi_k)e_z,dPhi>
def local_Lscheme_assembly(element_num,d,psi_k):
    cn = d.mapping[:,element_num]
    corners = d.geometry.nodes[0:2,cn[d.local_dofs_corners]]  # corners of element
    
    
    # PK element
    P_El = global_element_geometry(d.element, corners,d.geometry,d.degree)
    #Map reference element to local element
    [J, c, J_inv] = reference_to_local(P_El.element_coord)
    det_J = np.abs(np.linalg.det(J))
    # Fetch quadrature points
    a =  gauss_quadrature_points(2) 
    Phi = P_El.phi_eval(a[:,0:2])
    dPhi = P_El.grad_phi_eval(a[:,0:2]) 
    # Local pressure head values
    psi_local = np.array([psi_k[cn[0]],psi_k[cn[1]],psi_k[cn[2]]])
    local_vals = localelement_function_evaluation_L(d.K,d.theta,psi_local,P_El)
    
    local_perm = np.zeros((P_El.num_dofs,P_El.num_dofs))
    local_gravity = np.zeros((P_El.num_dofs,1))
    local_saturation_k = np.zeros((P_El.num_dofs,1))
    
    transform = J_inv@J_inv.T
    for j in range(P_El.num_dofs):
        #val1=0
        val2=0
        val1 = np.sum(local_vals.theta_in_Q*(a[:,2]*Phi[:][j]).reshape(-1,1))
        #valalt2 =  local_vals.K_in_Q*(a[:,2]*np.inner(np.array([0,1]),dPhi[:][j]@J_inv.T)).reshape(-1,1)
        for k in range(len(Phi)):
            #val1 += local_vals.theta_in_Q[k]*a[k][2]*Phi[k][j]
            val2 += local_vals.K_in_Q[k]*a[k][2]*np.inner(np.array([0,1]),dPhi[k][j]@J_inv.T)
            
        #print(valalt2,val2)
        local_saturation_k[j] = 0.5*val1*det_J
        local_gravity[j] = 0.5*val2*det_J 
        for i in range(P_El.num_dofs):
            val3=0
            for l in range(len(Phi)):
                val3 += local_vals.K_in_Q[l]*a[l][2]*dPhi[l][i]@transform@dPhi[l][j].T
            
            local_perm[j,i] += 0.5*val3*det_J

    return local_perm, local_gravity, local_saturation_k


# Local assembly of <theta(psi_k),phi>, <K(psi_k)dPhi,dPhi>, <K(psi_k)e_z,dPhi>, <K'(psi_k)e_z,dPhi> and <K'(psi_k) nabla(psi_k),dPhi>
def local_Newton_assembly(element_num,d,psi_k):
    cn = d.mapping[:,element_num]
    corners = d.geometry.nodes[0:2,cn[d.local_dofs_corners]]  # corners of element
    
    
    # PK element
    P_El = global_element_geometry(d.element, corners,d.geometry,d.degree)
    #Map reference element to local element
    [J, c, J_inv] = reference_to_local(P_El.element_coord)
    det_J = np.abs(np.linalg.det(J))
    # Fetch quadrature points
    a =  gauss_quadrature_points(2) 
    Phi = P_El.phi_eval(a[:,0:2])
    dPhi = P_El.grad_phi_eval(a[:,0:2]) 
    # Local pressure head values
    psi_local = np.array([psi_k[cn[0]],psi_k[cn[1]],psi_k[cn[2]]])
    local_vals = localelement_function_evaluation(d.K,d.theta,d.K_prime,d.theta_prime,psi_local,P_El)
    
    local_perm = np.zeros((P_El.num_dofs,P_El.num_dofs))
    local_gravity = np.zeros((P_El.num_dofs,1))
    local_saturation_k = np.zeros((P_El.num_dofs,1))
    local_J_perm =  np.zeros((P_El.num_dofs,P_El.num_dofs))
    local_J_gravity = np.zeros((P_El.num_dofs,P_El.num_dofs))
    local_J_saturation = np.zeros((P_El.num_dofs,P_El.num_dofs))
    
    transform = J_inv@J_inv.T
    for j in range(P_El.num_dofs):
        #val1=0
        val2=0
        val1 = np.sum(local_vals.theta_in_Q*(a[:,2]*Phi[:][j]).reshape(-1,1))
        #valalt2 =  local_vals.K_in_Q*(a[:,2]*np.inner(np.array([0,1]),dPhi[:][j]@J_inv.T)).reshape(-1,1)
        for k in range(len(Phi)):
            #val1 += local_vals.theta_in_Q[k]*a[k][2]*Phi[k][j]
            val2 += local_vals.K_in_Q[k]*a[k][2]*np.inner(np.array([0,1]),dPhi[k][j]@J_inv.T)
            
        #print(valalt2,val2)
        local_saturation_k[j] = 0.5*val1*det_J
        local_gravity[j] = 0.5*val2*det_J 
        for i in range(P_El.num_dofs):
            val3 = 0
            val4 = 0
            val5 = 0
            val6 = 0
            for l in range(len(Phi)):
                val3 += local_vals.K_in_Q[l]*a[l][2]*dPhi[l][i]@transform@dPhi[l][j].T
                val4 += local_vals.K_prime_Q[l]*a[l][2]*Phi[l][i]*local_vals.valgrad_Q[l]@transform@dPhi[l][j].T  
                val5 += local_vals.K_prime_Q[l]*a[l][2]*Phi[l][i]*np.inner(np.array([0,1]),dPhi[l][j]@J_inv.T)
                val6 += local_vals.theta_prime_Q[l]*a[l][2]*Phi[l][i]*Phi[l][j]
            
            local_perm[j,i]         += 0.5*val3*det_J
            local_J_perm[j,i]       += 0.5*val4*det_J
            local_J_gravity[j,i]    += 0.5*val5*det_J
            local_J_saturation[j,i] += 0.5*val6*det_J

    return local_perm, local_gravity, local_saturation_k, local_J_perm, local_J_gravity, local_J_saturation