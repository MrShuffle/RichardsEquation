# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 20:27:57 2022

@author: jakob
"""

import numpy as np
from RichardsEqFEM.source.localevaluation.local_evaluation import localelement_function_evaluation_L, localelement_function_evaluation

from scipy.sparse.coo import coo_matrix
import math
from RichardsEqFEM.source.basisfunctions.lagrange_element import global_element_geometry, finite_element
from RichardsEqFEM.source.basisfunctions.Gauss_quadrature_points import *
import porepy as pp
from RichardsEqFEM.source.LocalGlobalMapping.map_local_to_global import Local_to_Global_table
import sympy as sp
from RichardsEqFEM.source.utils.operators import reference_to_local

class L_scheme_hetrogenous():
    
    def __init__(self,L,dt,d,g,order,psi, K,theta,f, glob_perm):
        
        self.L     = L
        self.dt    = dt
        self.K     = K
        self.theta = theta
        self.f     = f
        self.glob_perm = glob_perm
        
        self.g     = g
        self.d     = d
        self.order = order
        
        # Initalize matrices
        self.perm_matrix    = np.zeros((d.total_geometric_pts, d.total_geometric_pts)) 
        self.sat_matrix_t   = np.zeros((d.total_geometric_pts, 1)) 
        self.sat_matrix_k   = np.zeros((d.total_geometric_pts, 1)) 
        self.mass_matrix    = np.zeros((d.total_geometric_pts, d.total_geometric_pts)) 
        self.gravity_vector = np.zeros((d.total_geometric_pts, 1)) 
        self.source_vector  = np.zeros((d.total_geometric_pts, 1))
        
        # compute mass matrix
        element = finite_element(self.order)
        
        elements = self.g.cell_nodes()
        points = self.g.nodes
        loc_node_idx = self.d.local_dofs_corners
        
        for e in range(self.g.num_cells):
            
            cn = self.d.mapping[:,e]
            corners = points[0:2,cn[loc_node_idx]]
            
            
            # PK element
            P_El = global_element_geometry(element, corners,self.g,self.order)
            
            # quadrature points
            
            quadrature = gauss_quadrature_points(self.order+1)
            quadrature_points = quadrature[:,0:2]
            dPhi = P_El.grad_phi_eval(quadrature_points) 
            Phi = P_El.phi_eval(quadrature_points)
            quadrature_pt = np.array([quadrature_points]).T
            #Phi = Phi[::-1]
        
            [J2, c2, Jinv] = reference_to_local(P_El.corners)
            jac = np.linalg.det(J2)
          
            for j in range(P_El.num_dofs):
                for i in range(P_El.num_dofs):
                    val=0
                    for k in range(len(Phi)):
                        val = val + quadrature[k][2]*Phi[k][i]*Phi[k][j]
                    #print(val)
                    self.mass_matrix[cn[j]][cn[i]] = self.mass_matrix[cn[j]][cn[i]] + 0.5*np.abs(jac)*val
           
    
    
    def update_at_iteration(self,psi_k):
        # Initalize matrices
        self.perm_matrix    = np.zeros((self.d.total_geometric_pts, self.d.total_geometric_pts)) 
        self.sat_matrix_k   = np.zeros((self.d.total_geometric_pts, 1)) 
        self.gravity_vector = np.zeros((self.d.total_geometric_pts, 1)) 

        
        element = finite_element(self.order)
        
        elements = self.g.cell_nodes()
        points = self.g.nodes
        loc_node_idx = self.d.local_dofs_corners
        
        gravity = np.array([0,1])
        for e in range(self.g.num_cells):
            
            cn = self.d.mapping[:,e]
            corners = points[0:2,cn[loc_node_idx]]
            
            
            # PK element
            P_El = global_element_geometry(element, corners,self.g,self.order)
            
            # quadrature points
            
            quadrature = gauss_quadrature_points(self.order+1)
            quadrature_points = quadrature[:,0:2]
            dPhi = P_El.grad_phi_eval(quadrature_points) 
            Phi = P_El.phi_eval(quadrature_points)
            quadrature_pt = np.array([quadrature_points]).T
            #Phi = Phi[::-1]
            if P_El.degree ==1:
                psi_local = np.array([psi_k[cn[0]],psi_k[cn[1]],psi_k[cn[2]]])
            if P_El.degree ==2:
                
                psi_local = np.array([psi_k[cn[0]],psi_k[cn[1]],psi_k[cn[2]],psi_k[cn[3]],psi_k[cn[4]],psi_k[cn[5]]])
            
            local_vals = localelement_function_evaluation_L(self.K,self.theta,psi_local,P_El)

            [J2, c2, J_inv] = reference_to_local(P_El.corners)
            jac = np.linalg.det(J2)
            transform = J_inv@J_inv.T
            for j in range(P_El.num_dofs):
               
                val1=0
               
                for k in range(len(dPhi)):
                          
                    val1 = val1 +local_vals.theta_in_Q[k]*quadrature[k][2]*Phi[k][j]
           
                    
                self.sat_matrix_k[cn[j]] = self.sat_matrix_k[cn[j]] + 0.5*np.abs(jac)*val1
            
            for j in range(P_El.num_dofs):
               
                #val1=0
                val2=0
                for k in range(len(dPhi)):
                          
                    #val1 = val1 +local_vals.theta_in_Q[k]*quadrature[k][2]*Phi[k][j]
                    vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
                    q_i = J2@vec +c2 # transformed quadrature points
                    w_i = quadrature[k][2] # weights
                    #mod = self.glob_perm@dPhi[k][j]
                    val2 += local_vals.K_in_Q[k]*w_i*np.inner(gravity,self.glob_perm@dPhi[k][j]@J_inv.T)
                    
                self.gravity_vector[cn[j]]= self.gravity_vector[cn[j]]+0.5*np.abs(jac)*val2
            for j in range(P_El.num_dofs):
                for i in range(P_El.num_dofs):
                    val3=0
                    for l in range(len(Phi)):
                        mod = self.glob_perm@dPhi[l][i]
                        val3 += local_vals.K_in_Q[l]*quadrature[l][2]*mod@transform@dPhi[l][j].T
                    
                    self.perm_matrix[cn[j]][cn[i]] = self.perm_matrix[cn[j]][cn[i]] + 0.5*np.abs(jac)*val3
        
                    
    def update_at_newtime(self,psi_t,t):
        
        # Initalize matrices
        self.sat_matrix_t   = np.zeros((self.d.total_geometric_pts, 1)) 
        self.source_vector  = np.zeros((self.d.total_geometric_pts, 1))
        
        element = finite_element(self.order)
        
        elements = self.g.cell_nodes()
        points = self.g.nodes
        loc_node_idx = self.d.local_dofs_corners
        
        for e in range(self.g.num_cells):
            
            cn = self.d.mapping[:,e]
            corners = points[0:2,cn[loc_node_idx]]
            
            
            # PK element
            P_El = global_element_geometry(element, corners,self.g,self.order)
            
            # quadrature points
            
            quadrature = gauss_quadrature_points(self.order+1)
            quadrature_points = quadrature[:,0:2]
            dPhi = P_El.grad_phi_eval(quadrature_points) 
            Phi = P_El.phi_eval(quadrature_points)
            quadrature_pt = np.array([quadrature_points]).T
            #Phi = Phi[::-1]
            if P_El.degree ==1:
                psi_local = np.array([psi_t[cn[0]],psi_t[cn[1]],psi_t[cn[2]]])
            if P_El.degree ==2:
                
                psi_local = np.array([psi_t[cn[0]],psi_t[cn[1]],psi_t[cn[2]],psi_t[cn[3]],psi_t[cn[4]],psi_t[cn[5]]])
            local_vals = localelement_function_evaluation_L(self.K,self.theta,psi_local,P_El)

            [J2, c2, Jinv] = reference_to_local(P_El.corners)
            jac = np.linalg.det(J2)
            transform = Jinv@Jinv.T
            for j in range(P_El.num_dofs):
               
                val1=0
                val2=0
                for k in range(len(Phi)):
                          
                    val1 = val1 +local_vals.theta_in_Q[k]*quadrature[k][2]*Phi[k][j]
                    
                    vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
                    q_i = J2@vec +c2 # transformed quadrature points
                    w_i = quadrature[k][2] # weights
                    val2 += w_i*self.f(t,q_i[0][0].item(),q_i[1][0].item())*Phi[k][j]
                #print(val)
                self.sat_matrix_t[cn[j]] = self.sat_matrix_t[cn[j]] + 0.5*np.abs(jac)*val1
                self.source_vector[cn[j]] = self.source_vector[cn[j]]+0.5*np.abs(jac)*val2
                   
        
    def assemble(self,psi_k):
        
        
        self.lhs = self.L*self.mass_matrix+self.dt*self.perm_matrix
        self.rhs = self.L*self.mass_matrix@psi_k +self.sat_matrix_t-self.sat_matrix_k + self.dt*self.source_vector - self.dt*self.gravity_vector
        
class Newton_scheme_hetr():
    
    def __init__(self,dt,d,g,order,psi, K,theta,K_prime,theta_prime,f,glob):
        
        
        self.dt    = dt #time step
        
        # data
        self.K     = K
        self.theta = theta
        self.f     = f
        self.K_prime =K_prime
        self.theta_prime = theta_prime
        self.glob_perm = glob
        
        self.g     = g
        self.d     = d
        self.order = order
        
        # Initalize matrices
        self.perm_matrix    = np.zeros((d.total_geometric_pts, d.total_geometric_pts)) 
        self.sat_matrix_t   = np.zeros((d.total_geometric_pts, 1)) 
        self.sat_matrix_k   = np.zeros((d.total_geometric_pts, 1)) 
        self.J_perm_matrix    = np.zeros((d.total_geometric_pts, d.total_geometric_pts)) 
        self.J_sat_matrix    = np.zeros((d.total_geometric_pts, d.total_geometric_pts)) 
        self.J_gravity_matrix    = np.zeros((d.total_geometric_pts, d.total_geometric_pts)) 
        self.gravity_vector = np.zeros((d.total_geometric_pts, 1)) 
        self.source_vector  = np.zeros((d.total_geometric_pts, 1))
        
        
    
    def update_at_iteration(self,psi_k):
        # Reset matrices
        self.perm_matrix    = np.zeros((self.d.total_geometric_pts, self.d.total_geometric_pts)) 
        self.sat_matrix_k   = np.zeros((self.d.total_geometric_pts, 1)) 
        self.gravity_vector = np.zeros((self.d.total_geometric_pts, 1)) 
        self.J_perm_matrix    = np.zeros((self.d.total_geometric_pts, self.d.total_geometric_pts))  
        self.J_sat_matrix    = np.zeros((self.d.total_geometric_pts, self.d.total_geometric_pts)) 
        self.J_gravity_matrix    = np.zeros((self.d.total_geometric_pts, self.d.total_geometric_pts))  
        
        element = finite_element(self.order)
        
        elements = self.g.cell_nodes()
        points = self.g.nodes
        loc_node_idx = self.d.local_dofs_corners
        
        gravity = np.array([0,1])
        for e in range(self.g.num_cells):
            
            cn = self.d.mapping[:,e]
            corners = points[0:2,cn[loc_node_idx]]
            
            
            # PK element
            P_El = global_element_geometry(element, corners,self.g,self.order)
            
            # quadrature points
            
            quadrature = gauss_quadrature_points(self.order+1)
            quadrature_points = quadrature[:,0:2]
            dPhi = P_El.grad_phi_eval(quadrature_points) 
            Phi = P_El.phi_eval(quadrature_points)
            quadrature_pt = np.array([quadrature_points]).T
            #Phi = Phi[::-1]
            if P_El.degree ==1:
                psi_local = np.array([psi_k[cn[0]],psi_k[cn[1]],psi_k[cn[2]]])
            if P_El.degree ==2:
                
                psi_local = np.array([psi_k[cn[0]],psi_k[cn[1]],psi_k[cn[2]],psi_k[cn[3]],psi_k[cn[4]],psi_k[cn[5]]])
            
            local_vals = localelement_function_evaluation(self.K,self.theta,self.K_prime,self.theta_prime,psi_local,P_El)

            [J2, c2, J_inv] = reference_to_local(P_El.corners)
            jac = np.linalg.det(J2)
            transform = J_inv@J_inv.T
            for j in range(P_El.num_dofs):
               
                val1=0
               
                for k in range(len(dPhi)):
                          
                    val1 = val1 +local_vals.theta_in_Q[k]*quadrature[k][2]*Phi[k][j]
           
                    
                self.sat_matrix_k[cn[j]] = self.sat_matrix_k[cn[j]] + 0.5*np.abs(jac)*val1
            
            for j in range(P_El.num_dofs):
               
                #val1=0
                val2=0
                for k in range(len(dPhi)):
                          
                    #val1 = val1 +local_vals.theta_in_Q[k]*quadrature[k][2]*Phi[k][j]
                    vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
                    q_i = J2@vec +c2 # transformed quadrature points
                    w_i = quadrature[k][2] # weights
                    val2 += local_vals.K_in_Q[k]*w_i*np.inner(gravity,self.glob_perm@dPhi[k][j]@J_inv.T)
                    
                self.gravity_vector[cn[j]]= self.gravity_vector[cn[j]]+0.5*np.abs(jac)*val2
            for j in range(P_El.num_dofs):
                for i in range(P_El.num_dofs):
                    val3=0
                    val4=0
                    val5=0
                    val6=0
                    for k in range(len(Phi)):
                        
                        w_i = quadrature[k][2] # weights
                        
                        val5 += local_vals.theta_prime_Q[k]*w_i*Phi[k][i]*Phi[k][j]
                        
                        mod = self.glob_perm@local_vals.valgrad_Q[k]
                        val4 += local_vals.K_prime_Q[k]*w_i*Phi[k][i]*mod@transform@dPhi[k][j].T 
                        mod = self.glob_perm@dPhi[k][i]
                        val3 += local_vals.K_in_Q[k]*w_i*mod@transform@dPhi[k][j].T
                        
                        vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
                        q_i = J2@vec +c2 # transformed quadrature points
                        w_i = quadrature[k][2] # weights
                     
                        val6 += local_vals.K_prime_Q[k]*w_i*Phi[k][i]*np.inner(gravity,self.glob_perm@dPhi[k][j]@J_inv.T)
                    
                    self.perm_matrix[cn[j]][cn[i]] = self.perm_matrix[cn[j]][cn[i]] + 0.5*np.abs(jac)*val3
                    self.J_perm_matrix[cn[j]][cn[i]] = self.J_perm_matrix[cn[j]][cn[i]] + 0.5*np.abs(jac)*val4
                    self.J_sat_matrix[cn[j]][cn[i]] = self.J_sat_matrix[cn[j]][cn[i]] + 0.5*np.abs(jac)*val5
                    self.J_gravity_matrix[cn[j]][cn[i]] = self.J_gravity_matrix[cn[j]][cn[i]] + 0.5*np.abs(jac)*val6
                   
                    
                    
    def update_at_newtime(self,psi_t,t):
        
        # Initalize matrices
        self.sat_matrix_t   = np.zeros((self.d.total_geometric_pts, 1)) 
        self.source_vector  = np.zeros((self.d.total_geometric_pts, 1))
        
        element = finite_element(self.order)
        
        elements = self.g.cell_nodes()
        points = self.g.nodes
        loc_node_idx = self.d.local_dofs_corners
        
        for e in range(self.g.num_cells):
            
            cn = self.d.mapping[:,e]
            corners = points[0:2,cn[loc_node_idx]]
            
            
            # PK element
            P_El = global_element_geometry(element, corners,self.g,self.order)
            
            # quadrature points
            
            quadrature = gauss_quadrature_points(self.order+1)
            quadrature_points = quadrature[:,0:2]
            dPhi = P_El.grad_phi_eval(quadrature_points) 
            Phi = P_El.phi_eval(quadrature_points)
            quadrature_pt = np.array([quadrature_points]).T
            #Phi = Phi[::-1]
            if P_El.degree ==1:
                psi_local = np.array([psi_t[cn[0]],psi_t[cn[1]],psi_t[cn[2]]])
            if P_El.degree ==2:
                
                psi_local = np.array([psi_t[cn[0]],psi_t[cn[1]],psi_t[cn[2]],psi_t[cn[3]],psi_t[cn[4]],psi_t[cn[5]]])
            local_vals = localelement_function_evaluation_L(self.K,self.theta,psi_local,P_El)

            [J2, c2, Jinv] = reference_to_local(P_El.corners)
            jac = np.linalg.det(J2)
            transform = Jinv@Jinv.T
            for j in range(P_El.num_dofs):
               
                val1=0
                val2=0
                for k in range(len(Phi)):
                          
                    val1 = val1 +local_vals.theta_in_Q[k]*quadrature[k][2]*Phi[k][j]
                    
                    vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
                    q_i = J2@vec +c2 # transformed quadrature points
                    w_i = quadrature[k][2] # weights
                    #print(q_i[0][0],q_i[1][0].item())
                    val2 += w_i*self.f(t,q_i[0][0].item(),q_i[1][0].item())*Phi[k][j]
                #print(val)
                self.sat_matrix_t[cn[j]] = self.sat_matrix_t[cn[j]] + 0.5*np.abs(jac)*val1
                self.source_vector[cn[j]] = self.source_vector[cn[j]]+0.5*np.abs(jac)*val2
                   
        
    def assemble(self,psi_k):
        
        self.lhs = self.J_sat_matrix+self.dt*self.perm_matrix+self.dt*self.J_perm_matrix+self.dt*self.J_gravity_matrix
        self.rhs = self.J_sat_matrix@psi_k +self.sat_matrix_t-self.sat_matrix_k + self.dt*self.source_vector - self.dt*self.gravity_vector \
            +self.dt*self.J_perm_matrix@psi_k+self.dt*self.J_gravity_matrix@psi_k
        
        
        
        
        
        