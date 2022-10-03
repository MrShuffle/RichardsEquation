# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 12:41:27 2022

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
#from RichardsEqFEM.source.MatrixAssembly.local_mass_assembly import local_assembly_mass, global_assembly_mass, local_mass_pool
#from RichardsEqFEM.source.MatrixAssembly.local_assemblers import local_mass_assembly, local_Lscheme_assembly, local_Newton_assembly
from multiprocessing import Process, Pool, cpu_count, Manager, Queue,freeze_support
class L_scheme_Parallell():
    
    def __init__(self,L,dt,d,g,order,psi, K,theta,f):
        
        d.L_scheme(K, theta, f)
        
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
        pool = Pool(6)
        elements_list =list(range(g.num_cells))
              
        results = pool.map(self.local_mass_assembly,elements_list)
               
        for result in results:
            
            #_data_m.append(result)
            for k,l in np.ndindex(d.element.num_dofs,d.element.num_dofs):
                 
                #print(result)
                _data_mass[n]= result[k,l]
                n+=1
                # print(_data_mass,'data')
                # print('-----')
                
        mass_m = coo_matrix((_data_mass,(d.row,d.col))).tocsr()
        
        self.mass_matrix = mass_m
     
    
    
    def update_at_iteration(self,psi_k):
        self.psi_k = psi_k
        # Initalize matrices
        #self.perm_matrix    = np.zeros((self.d.total_geometric_pts, self.d.total_geometric_pts)) 
        self.sat_matrix_k   = np.zeros((self.d.total_geometric_pts, 1)) 
        self.gravity_vector = np.zeros((self.d.total_geometric_pts, 1)) 
        _data_perm = np.empty(self.d.numDataPts, dtype=np.double)
        
        #element = finite_element(self.order)
        n=0
        
        pool = Pool(6)
        elements_list =list(range(self.g.num_cells))
        results = pool.map(self.local_Lscheme_assembly,elements_list)
        
        for result in results:
            
            #_data_m.append(result)
            for k,l in np.ndindex(self.d.element.num_dofs,self.d.element.num_dofs):
                 _data_perm[n]= result[0][k,l]
                 n+=1
                 
            for i in range(len(result[-1])):
                self.sat_matrix_k[result[-1][i]] = self.sat_matrix_k[result[-1][i]] + result[2][i]
       
                    
                self.gravity_vector[result[-1][i]]= self.gravity_vector[result[-1][i]]+result[1][i]
        
        self.perm_matrix = coo_matrix((_data_perm,(self.d.row,self.d.col))).tocsr()
      
        
                    
    def update_at_newtime(self,psi_t,t):
        
        # Initalize matrices
        self.sat_matrix_t   = np.zeros((self.d.total_geometric_pts, 1)) 
        self.source_vector  = np.zeros((self.d.total_geometric_pts, 1))
        #numDataPts = self.d.global_vals[2]*self.element.num_dofs**2
        self.time = t
        self.psi_t = psi_t
        #self.psi_k = psi_t
        pool = Pool(6)
        elements_list =list(range(self.g.num_cells))
        results = pool.map(self.local_source_saturation_assembly,elements_list)
        
        for result in results:
            
          
                 
            for i in range(len(result[-1])):
                self.sat_matrix_t[result[-1][i]] = self.sat_matrix_t[result[-1][i]] + result[1][i]
       
                    
                self.source_vector[result[-1][i]]= self.source_vector[result[-1][i]]+result[0][i]
   
    
  
              
         
    def assemble(self,psi_k):
        
        
        self.lhs = self.L*self.mass_matrix+self.dt*self.perm_matrix
        self.rhs = self.L*self.mass_matrix.dot(psi_k) +self.sat_matrix_t-self.sat_matrix_k + self.dt*self.source_vector - self.dt*self.gravity_vector
        #print(self.rhs)

    # Local mass matrix assembly
    def local_mass_assembly(self,element_num):
        cn = self.d.mapping[:,element_num]
        corners = self.d.geometry.nodes[0:2,cn[self.d.local_dofs_corners]]  # corners of element
        
        
        # PK element
        P_El = global_element_geometry(self.d.element, corners,self.d.geometry,self.d.degree)
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
    def local_source_saturation_assembly(self,element_num):
        cn = self.d.mapping[:,element_num]
        corners = self.d.geometry.nodes[0:2,cn[self.d.local_dofs_corners]]  # corners of element
        
        
        # PK element
        P_El = global_element_geometry(self.d.element, corners,self.d.geometry,self.d.degree)
        #Map reference element to local element
        [J, c, J_inv] = reference_to_local(P_El.element_coord)
        det_J = np.abs(np.linalg.det(J))
        # Fetch quadrature points
        a =  gauss_quadrature_points(2) 
        quadrature_points = a[:,0:2]
        quadrature_pt = np.array([quadrature_points]).T
        Phi = P_El.phi_eval(a[:,0:2])
        # Initalize local vectors
        local_source = np.zeros((P_El.num_dofs,1))
        local_saturation = np.zeros((P_El.num_dofs,1))
        
        # Local pressure head values
        psi_local = np.array([self.psi_t[cn[0]],self.psi_t[cn[1]],self.psi_t[cn[2]]])
        
        local_vals = localelement_function_evaluation_L(self.d.K,self.d.theta,psi_local,P_El)
        for j in range(P_El.num_dofs):
            val1=0
            val2=0
            valalt1 = np.sum(local_vals.theta_in_Q*(a[:,2]*Phi[:][j]).reshape(-1,1))
            for k in range(len(Phi)):
            
                #val1 += local_vals.theta_in_Q[k]*a[k][2]*Phi[k][j]
                
                vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
                q_i = J@vec +c # transformed quadrature points
                w_i = a[k][2] # weights
                val2 += w_i*self.d.f(self.time,q_i[0][0].item(),q_i[1][0].item())*Phi[k][j]

            local_source[j] = 0.5*val2*det_J
            local_saturation[j] = 0.5*valalt1*det_J

        return local_source, local_saturation, cn
        
    # Local assembly of <theta(psi_k),phi>, <K(psi_k)dPhi,dPhi>, <K(psi_k)e_z,dPhi>
    def local_Lscheme_assembly(self,element_num):
        cn = self.d.mapping[:,element_num]
        corners = self.d.geometry.nodes[0:2,cn[self.d.local_dofs_corners]]  # corners of element
        
        
        # PK element
        P_El = global_element_geometry(self.d.element, corners,self.d.geometry,self.d.degree)
        #Map reference element to local element
        [J, c, J_inv] = reference_to_local(P_El.element_coord)
        det_J = np.abs(np.linalg.det(J))
        # Fetch quadrature points
        a =  gauss_quadrature_points(2) 
        Phi = P_El.phi_eval(a[:,0:2])
        dPhi = P_El.grad_phi_eval(a[:,0:2]) 
        # Local pressure head values
        psi_local = np.array([self.psi_k[cn[0]],self.psi_k[cn[1]],self.psi_k[cn[2]]])
        local_vals = localelement_function_evaluation_L(self.d.K,self.d.theta,psi_local,P_El)
        
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

        return local_perm, local_gravity, local_saturation_k, cn
    
    
    
    
    
class Newton_scheme_Parallell():
    

    def __init__(self,dt,d,g,order,psi, K,theta,K_prime,theta_prime,f):
        
        d.Newton_method(K, theta,K_prime,theta_prime,f)
        
        
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
        

        
        
    
    
    def update_at_iteration(self,psi_k):
        self.psi_k = psi_k
        # Initalize matrices
        #self.perm_matrix    = np.zeros((self.d.total_geometric_pts, self.d.total_geometric_pts)) 
        self.sat_matrix_k   = np.zeros((self.d.total_geometric_pts, 1)) 
        self.gravity_vector = np.zeros((self.d.total_geometric_pts, 1)) 
        # _data_perm = np.empty(self.d.numDataPts, dtype=np.double)
        # _data_J_perm = np.empty(self.d.numDataPts, dtype=np.double)
        # _data_J_sat =np.empty(self.d.numDataPts, dtype=np.double)
        # _data_J_grav = np.empty(self.d.numDataPts, dtype=np.double)
        #element = finite_element(self.order)
        n=0
        
        _data_perm = []
        _data_J_perm = []
        _data_J_sat = []
        _data_J_grav = []
        pool = Pool()
        elements_list =list(range(self.g.num_cells))
        results = pool.map(self.local_Newtonscheme_assembly,elements_list)
        
        for result in results:
            
            _data_perm.append(result[0].flatten())
            _data_J_perm.append(result[3].flatten())
            _data_J_sat.append(result[5].flatten())
            _data_J_grav.append(result[4].flatten())
            
            # for k,l in np.ndindex(self.d.element.num_dofs,self.d.element.num_dofs):
            #      _data_perm[n]= result[0][k,l]
            #      _data_J_perm[n] = result[3][k,l]
            #      _data_J_sat[n] = result[5][k,l]
            #      _data_J_grav[n] = result[4][k,l]
            #      n+=1
                 
            for i in range(len(result[-1])):
                self.sat_matrix_k[result[-1][i]] = self.sat_matrix_k[result[-1][i]] + result[2][i]
       
                    
                self.gravity_vector[result[-1][i]]= self.gravity_vector[result[-1][i]]+result[1][i]
        
        _data_perm = np.concatenate(_data_perm)
        _data_J_perm = np.concatenate(_data_J_perm)
        _data_J_sat = np.concatenate(_data_J_sat)
        _data_J_grav = np.concatenate(_data_J_grav)
        #print(_data_p.shape)
        self.perm_matrix = coo_matrix((_data_perm,(self.d.row,self.d.col)))#.tocsr()
        self.J_perm_matrix = coo_matrix((_data_J_perm,(self.d.row,self.d.col)))#.tocsr()
        self.J_sat_matrix = coo_matrix((_data_J_sat,(self.d.row,self.d.col)))#.tocsr()
        self.J_gravity_matrix = coo_matrix((_data_J_grav,(self.d.row,self.d.col)))#.tocsr()
        
                    
    def update_at_newtime(self,psi_t,t):
        
        # Initalize matrices
        self.sat_matrix_t   = np.zeros((self.d.total_geometric_pts, 1)) 
        self.source_vector  = np.zeros((self.d.total_geometric_pts, 1))
        #numDataPts = self.d.global_vals[2]*self.element.num_dofs**2
        self.time = t
        self.psi_t = psi_t
        #self.psi_k = psi_t
        pool = Pool(6)
        elements_list =list(range(self.g.num_cells))
        results = pool.map(self.local_source_saturation_assembly,elements_list)
        
        for result in results:
            
          
                 
            for i in range(len(result[-1])):
                self.sat_matrix_t[result[-1][i]] = self.sat_matrix_t[result[-1][i]] + result[1][i]
       
                    
                self.source_vector[result[-1][i]]= self.source_vector[result[-1][i]]+result[0][i]
   
    
              
         
    def assemble(self,psi_k):
            
            
        self.lhs = self.J_sat_matrix+self.dt*self.perm_matrix+self.dt*self.J_perm_matrix+self.dt*self.J_gravity_matrix
        self.rhs = self.J_sat_matrix@psi_k +self.sat_matrix_t-self.sat_matrix_k + self.dt*self.source_vector - self.dt*self.gravity_vector \
                +self.dt*self.J_perm_matrix@psi_k+self.dt*self.J_gravity_matrix@psi_k


    # Local assembly of source term at time t and <theta(psi),phi> which is also a vector
    def local_source_saturation_assembly(self,element_num):
        cn = self.d.mapping[:,element_num]
        corners = self.d.geometry.nodes[0:2,cn[self.d.local_dofs_corners]]  # corners of element
        
        
        # PK element
        P_El = global_element_geometry(self.d.element, corners,self.d.geometry,self.d.degree)
        #Map reference element to local element
        [J, c, J_inv] = reference_to_local(P_El.element_coord)
        det_J = np.abs(np.linalg.det(J))
        # Fetch quadrature points
        a =  gauss_quadrature_points(2) 
        quadrature_points = a[:,0:2]
        quadrature_pt = np.array([quadrature_points]).T
        Phi = P_El.phi_eval(a[:,0:2])
        # Initalize local vectors
        local_source = np.zeros((P_El.num_dofs,1))
        local_saturation = np.zeros((P_El.num_dofs,1))
        
        # Local pressure head values
        psi_local = np.array([self.psi_t[cn[0]],self.psi_t[cn[1]],self.psi_t[cn[2]]])
        
        local_vals = localelement_function_evaluation_L(self.d.K,self.d.theta,psi_local,P_El)
        for j in range(P_El.num_dofs):
            val1=0
            val2=0
            valalt1 = np.sum(local_vals.theta_in_Q*(a[:,2]*Phi[:][j]).reshape(-1,1))
            for k in range(len(Phi)):
            
                #val1 += local_vals.theta_in_Q[k]*a[k][2]*Phi[k][j]
                
                vec = [(quadrature_pt)[0][k],(quadrature_pt)[1][k]]
                q_i = J@vec +c # transformed quadrature points
                w_i = a[k][2] # weights
                val2 += w_i*self.d.f(self.time,q_i[0][0].item(),q_i[1][0].item())*Phi[k][j]

            local_source[j] = 0.5*val2*det_J
            local_saturation[j] = 0.5*valalt1*det_J

        return local_source, local_saturation, cn
        
    # Local assembly of <theta(psi_k),phi>, <K(psi_k)dPhi,dPhi>, <K(psi_k)e_z,dPhi>, <K'(psi_k)e_z,dPhi> and <K'(psi_k) nabla(psi_k),dPhi>
    def local_Newtonscheme_assembly(self,element_num):
        cn = self.d.mapping[:,element_num]
        corners = self.d.geometry.nodes[0:2,cn[self.d.local_dofs_corners]]  # corners of element
        
        
        # PK element
        P_El = global_element_geometry(self.d.element, corners,self.d.geometry,self.d.degree)
        #Map reference element to local element
        [J, c, J_inv] = reference_to_local(P_El.element_coord)
        det_J = np.abs(np.linalg.det(J))
        # Fetch quadrature points
        a =  gauss_quadrature_points(2) 
        Phi = P_El.phi_eval(a[:,0:2])
        dPhi = P_El.grad_phi_eval(a[:,0:2]) 
        # Local pressure head values
        psi_local = np.array([self.psi_k[cn[0]],self.psi_k[cn[1]],self.psi_k[cn[2]]])
        local_vals = localelement_function_evaluation(self.d.K,self.d.theta,self.d.K_prime,self.d.theta_prime,psi_local,P_El)
        
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
            

        return local_perm, local_gravity, local_saturation_k,local_J_perm, local_J_gravity, local_J_saturation, cn