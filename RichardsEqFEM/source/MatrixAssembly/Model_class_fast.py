# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:19:54 2022

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
from RichardsEqFEM.source.MatrixAssembly.local_mass_assembly import local_assembly_mass, global_assembly_mass, local_mass_pool
from RichardsEqFEM.source.MatrixAssembly.local_assemblers import local_mass_assembly, local_Lscheme_assembly

class L_scheme_fast():
    
    def __init__(self,L,dt,d,g,order,psi, K,theta,f):
        
        self.L     = L
        self.dt    = dt
        self.K     = K
        self.theta = theta
        self.f     = f
        
        self.g     = g
        self.d     = d
        self.order = order
        
        # Initalize matrices
        #self.perm_matrix    = np.zeros((d.total_geometric_pts, d.total_geometric_pts)) 
        self.sat_matrix_t   = np.zeros((d.total_geometric_pts, 1)) 
        self.sat_matrix_k   = np.zeros((d.total_geometric_pts, 1)) 
        #self.mass_matrix    = np.zeros((d.total_geometric_pts, d.total_geometric_pts)) 
        self.gravity_vector = np.zeros((d.total_geometric_pts, 1)) 
        self.source_vector  = np.zeros((d.total_geometric_pts, 1))
        
        # compute mass matrix
        self.element = finite_element(self.order)
        
        _data_mass = np.empty(d.numDataPts, dtype=np.double)
        n=0
        
        # if para == True:
        #     if __name__ == '__main__':
        #         pool = Pool(6)
        #         elements_list =list(range(g.num_cells))
        #         key = (d)
        #         testy = list(map(lambda e: (e, key), elements_list))
        #         #testy = list(map(lambda e: (e, g), testy))
        #         results = pool.starmap(local_mass_pool,testy)
        #         #pool.close()
        #         #print(results)
        #         for result in results:
            
                    
        #             for k,l in np.ndindex(d.element.num_dofs,d.element.num_dofs):
                 
        #             #print(result)
        #                 _data_mass[n]= result[k,l]
        #                 n+=1
        #         print(_data_mass,'data')
        #         print('-----')
        #     self.mass_matrix = coo_matrix((_data_mass,(d.row,d.col)))
        for e in range(g.num_cells):
            local = local_mass_assembly(e,d)
            
            for k,l in np.ndindex(d.element.num_dofs,d.element.num_dofs):
                 
            
                _data_mass[n] = local[k,l]
                n+=1
        
        self.mass_matrix = coo_matrix((_data_mass,(d.row,d.col)))  .tocsr()
    
    
    def update_at_iteration(self,psi_k):
        # Initalize matrices
        #self.perm_matrix    = np.zeros((self.d.total_geometric_pts, self.d.total_geometric_pts)) 
        self.sat_matrix_k   = np.zeros((self.d.total_geometric_pts, 1)) 
        self.gravity_vector = np.zeros((self.d.total_geometric_pts, 1)) 
        _data_perm = np.empty(self.d.numDataPts, dtype=np.double)
        
        #element = finite_element(self.order)
        n=0
        for e in range(self.g.num_cells):
            cn = self.d.mapping[:,e]
            local_perm, local_gravity, local_saturation_k = local_Lscheme_assembly(e,self.d,psi_k)
            
            for j in range(self.d.num_nodes_per_EL):
                self.sat_matrix_k[cn[j]] = self.sat_matrix_k[cn[j]] + local_saturation_k[j]
       
                    
                self.gravity_vector[cn[j]]= self.gravity_vector[cn[j]]+local_gravity[j]
                
            for k,l in np.ndindex(self.d.element.num_dofs,self.d.element.num_dofs):
                 
            
                _data_perm[n] = local_perm[k,l]
                n+=1
        
        self.perm_matrix = coo_matrix((_data_perm,(self.d.row,self.d.col)))  .tocsr()
      
        
                    
    def update_at_newtime(self,psi_t,t):
        
        # Initalize matrices
        self.sat_matrix_t   = np.zeros((self.d.total_geometric_pts, 1)) 
        self.source_vector  = np.zeros((self.d.total_geometric_pts, 1))
        numDataPts = self.d.global_vals[2]*self.element.num_dofs**2
  
   
        # _row = np.empty(numDataPts, dtype=int)
        # _col = np.empty(numDataPts, dtype=int)
        # _data_source = np.empty(numDataPts, dtype=np.double)
        
        # n=0
        
        #element = finite_element(self.order)
        
        # elements = self.g.cell_nodes()
        points = self.g.nodes
        loc_node_idx = self.d.local_dofs_corners
        
        for e in range(self.g.num_cells):
            
            cn = self.d.mapping[:,e]
            corners = points[0:2,cn[loc_node_idx]]
            
            
            # PK element
            P_El = global_element_geometry(self.element, corners,self.g,self.order)
            
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
                #_row[n]= cn[j]
                #valalt1 =
                for k in range(len(Phi)):
                
                    val1 = val1 +local_vals.theta_in_Q[k]*quadrature[k][2]*Phi[k][j]
                    
                    vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
                    q_i = J2@vec +c2 # transformed quadrature points
                    w_i = quadrature[k][2] # weights
                    val2 += w_i*self.f(t,q_i[0][0].item(),q_i[1][0].item())*Phi[k][j]
                #print(val)
                self.sat_matrix_t[cn[j]] = self.sat_matrix_t[cn[j]] + 0.5*np.abs(jac)*val1
                self.source_vector[cn[j]] += 0.5*np.abs(jac)*val2
              
         
    def assemble(self,psi_k):
        
        
        self.lhs = self.L*self.mass_matrix+self.dt*self.perm_matrix
        self.rhs = self.L*self.mass_matrix@psi_k +self.sat_matrix_t-self.sat_matrix_k + self.dt*self.source_vector - self.dt*self.gravity_vector
        #print(self.rhs)
        