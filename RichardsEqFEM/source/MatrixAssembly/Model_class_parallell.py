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
   
 
        pool = Pool()
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
        
        pool = Pool()
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
        pool = Pool()
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
    
    def linearization_error(self,u_j,u_j_1):
        '''
        Computes lineartization error wrt norm in 

        Parameters
        ----------
        res : u^{j}-u^{j-1}.

        Returns
        -------
        None.

        '''
        self.u_j = u_j
        self.u_j_1 = u_j_1
        val=0
        for e in range(self.g.num_cells):
            val +=self.error_on_element(e)
            
        self.linear_norm = np.sqrt(val)
    def error_on_element(self,element_num):
        cn = self.d.mapping[:,element_num]
        corners = self.d.geometry.nodes[0:2,cn[self.d.local_dofs_corners]]  # corners of element
        
        
        # PK element
        P_El = global_element_geometry(self.d.element, corners,self.d.geometry,self.d.degree)
        #Map reference element to local element
        [J, c, J_inv]     = reference_to_local(P_El.element_coord)
        det_J             = np.abs(np.linalg.det(J))
        # Fetch quadrature points
        a                 =  gauss_quadrature_points(2) 
        quadrature_points = a[:,0:2]
        quadrature_pt     = np.array([quadrature_points]).T
        Phi               = P_El.phi_eval(a[:,0:2])
        dPhi              = P_El.grad_phi_eval(a[:,0:2]) 
        # Local pressure head values
        psi_local   = np.array([self.u_j_1[cn[0]],self.u_j_1[cn[1]],self.u_j_1[cn[2]]])
        psi_local2  = np.array([self.u_j[cn[0]],self.u_j[cn[1]],self.u_j[cn[2]]])
        local_vals  = localelement_function_evaluation_L(self.d.K,self.d.theta,psi_local2,P_El)
        local_vals2 = localelement_function_evaluation_L(self.d.K,self.d.theta,psi_local,P_El)
        R_h         = np.dot((psi_local).T.reshape(len(psi_local)),Phi.T)
        
        
        val =0
        for k in range(len(Phi)):
            val += 0.5*a[k][2]*self.L*R_h[k]**2*det_J+self.dt*0.5*a[k][2]*local_vals.K_in_Q[k]*np.linalg.norm(local_vals2.valgrad_Q[k]@J_inv.T)**2*det_J        
    
        return val
    
    def estimate_eta(self, u):
        self.u = u
        val=0
        for element_num in range(self.g.num_cells):
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
            dPhi = P_El.grad_phi_eval(a[:,0:2]) 
            # Local pressure head values
            psi_local = np.array([self.u[cn[0]],self.u[cn[1]],self.u[cn[2]]])
            
            
            R_h = np.dot((psi_local).T.reshape(len(psi_local)),Phi.T)
            
            
            val =0
            for k in range(len(Phi)):
                val += 0.5*a[k][2]*np.linalg.norm(R_h[k]*dPhi[k][:]@J_inv.T)**2*det_J
            self.gradbound=np.sqrt(val)        
            return np.sqrt(val)
        
    def estimate_CN(self,K,K_prime,theta_prime,u):
        x = sp.symbols('x')
        
        eta=self.gradbound
        # Initialize
        C_array = np.zeros((len(u),1))
        theta_in_Q = np.zeros((len(u),1))
        K_in_Q = np.zeros((len(u),1))
        theta_prime_Q = np.zeros((len(u),1))
        K_d_theta = np.zeros((len(u),1))
        K_prime_Q = np.zeros((len(u),1))
        for k in range(len(u)):
            theta_in_Q[k] = self.theta(u[k].item())
            K_in_Q[k]     = K(theta_in_Q[k].item())
            
            #self.K[k]     = self.K_func(self.val_Q[k].item())
            #r = self.K_prime_func.subs(x,self.val_Q[k].item())
            z = theta_prime.subs(x,u[k].item())
            r = K_prime.subs(x,theta_in_Q[k].item())

            theta_prime_Q[k] = z#self.theta_prime_func.evalf(subs={x: np.asscalar(self.val[k])})
            #print(r)
            # derivative of theta wrt psi
            K_d_theta[k] = r #self.K_prime_func(self.theta[k])
            # derivative of K wrt psi
            K_prime_Q[k] = K_d_theta[k]*theta_prime_Q[k]
            
            C_array[k]=self.dt*np.linalg.norm(K_in_Q[k]**(-1/2)*K_prime_Q[k]*(eta+1))**2/theta_prime_Q[k]
      
        self.CN =np.nanmax(C_array)
        if 0<self.CN<2:
            self.CN=self.CN
        else:
            self.CN=0
        return C_array
    
    def L_to_N_eta(self,w1,w2,K_prime,theta_prime):
        '''
        

        Parameters
        ----------
        w1 : psi^(k).
        w2 : psi^(k-1).
        K_prime : Derivative of permability.
        theta_prime : Derivative of stauration.

        Returns
        -------
        Indicator for switch to Newton's method.

        '''
        self.estimate_CN(self.d.K, K_prime, theta_prime, w1) # estimate C_N^i
        self.theta_prime = theta_prime
        self.K_prime = K_prime
        a = 2/(2-self.CN)
        self.w1=w1
        self.w2=w2
        self.diff = self.w1-self.w2
        pool = Pool()
        elements_list =list(range(self.g.num_cells))
        results = pool.map(self.norm_L_to_N_on_element,elements_list)
        
        val  = 0
        val2 = 0
        val3 = 0
        val4 = 0
        for result in results:
            
            val  += result[0]
            val2 += result[1] 
            val3 += result[2] # L-scheme linearization norm error
            val4 += result[3]
        # for e in range(self.g.num_cells):
        #     q,w,r,o = self.norm_L_to_N_on_element(e)
        #     val += q
        #     val2 +=w
        #     val3+= r
        #     val4+= o
            
        #eta_L1 = np.sqrt(val4)
        self.linear_norm = np.sqrt(val3)
        self.eta_LtoL = a/self.linear_norm*np.sqrt(val4+self.dt*val2)
        self.eta_LtoN = a/self.linear_norm*np.sqrt(val+self.dt*val2)
    
    def norm_L_to_N_on_element(self,element_num):
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
        dPhi = P_El.grad_phi_eval(a[:,0:2]) 
        psi_local   = np.array([self.w2[cn[0]],self.w2[cn[1]],self.w2[cn[2]]])
        psi_local2  = np.array([self.w1[cn[0]],self.w1[cn[1]],self.w1[cn[2]]])
        psi_local3  = np.array([self.diff[cn[0]],self.diff[cn[1]],self.diff[cn[2]]])
        local_vals  = localelement_function_evaluation(self.d.K,self.d.theta,self.K_prime,self.theta_prime,psi_local2,P_El)
        local_vals2 = localelement_function_evaluation(self.d.K,self.d.theta,self.K_prime,self.theta_prime,psi_local,P_El)
        local_vals3 = localelement_function_evaluation_L(self.d.K,self.d.theta,psi_local3,P_El) 
       
        val  = 0
        val2 = 0
        val3 = 0
        val4 = 0
 
        R_h = np.dot((psi_local3).T.reshape(len(psi_local)),Phi.T)
           
        
       
            
        for k in range(len(Phi)):
            if 0== local_vals.theta_prime_Q[k]:
                val+= 0#0.5*a[k][2]*det_J*((np.sqrt(2)/np.pi)/local_vals.K_in_Q[k]**(1/2)*(self.L*(local_vals.val_Q[k]-local_vals2.val_Q[k])-(local_vals.theta_in_Q[k]-local_vals2.theta_in_Q[k])))**2
                
            else:
                val  += 0.5*a[k][2]*det_J*(1/local_vals.theta_prime_Q[k]**(1/2)*(self.L*(local_vals.val_Q[k]-local_vals2.val_Q[k])-(local_vals.theta_in_Q[k]-local_vals2.theta_in_Q[k])))**2
            
            val4 += 0.5*a[k][2]*det_J*(1/self.L**(1/2)*(self.L*(local_vals.val_Q[k]-local_vals2.val_Q[k])-(local_vals.theta_in_Q[k]-local_vals2.theta_in_Q[k])))**2
        
            val2 += 0.5*a[k][2]*det_J*(1/local_vals.K_in_Q[k]**(1/2)*((local_vals.K_in_Q[k]-local_vals2.K_in_Q[k]))*np.linalg.norm(local_vals.valgrad_Q[k]@J_inv.T+np.array([0,1])))**2
            val3 += 0.5*a[k][2]*self.L*R_h[k]**2*det_J+self.dt*0.5*a[k][2]*local_vals2.K_in_Q[k]*np.linalg.norm(local_vals3.valgrad_Q[k]@J_inv.T)**2*det_J        
            
        return val,val2,val3,val4
    def update_L(self,L):
        self.L=L
    # def N_to_L_eta(self):
    #     pass
    
    
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
        pool = Pool()
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
    
    def linearization_error(self,u_j,u_j_1):
        '''
        Computes lineartization error wrt iteration dependent energy norm.

        '''
        self.u_j = u_j
        self.u_j_1 = u_j_1
        val=0
        pool = Pool()
        elements_list =list(range(self.g.num_cells))
        results = pool.map(self.error_on_element,elements_list)
        val= np.sum(results)
        # for e in range(self.g.num_cells):
        #     val +=self.error_on_element(e)
            
        self.linear_norm = np.sqrt(val)
    def error_on_element(self,element_num):
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
        dPhi = P_El.grad_phi_eval(a[:,0:2]) 
        # Local pressure head values
        psi_local = np.array([self.u_j_1[cn[0]],self.u_j_1[cn[1]],self.u_j_1[cn[2]]])
        psi_local2 = np.array([self.u_j[cn[0]],self.u_j[cn[1]],self.u_j[cn[2]]])
        local_vals = localelement_function_evaluation(self.d.K,self.d.theta,self.d.K_prime,self.d.theta_prime,psi_local2,P_El)
        local_vals2 = localelement_function_evaluation(self.d.K,self.d.theta,self.d.K_prime,self.d.theta_prime,psi_local,P_El)
        
        
        R_h = np.dot((psi_local).T.reshape(len(psi_local)),Phi.T)
        
        # if element_num == 5:
        #     print(R_h,'tttt')
        
        val =0
        for k in range(len(Phi)):
            val += 0.5*a[k][2]*local_vals.theta_prime_Q[k]*R_h[k]**2*det_J+0.5*a[k][2]*self.dt*local_vals.K_in_Q[k]*np.linalg.norm(local_vals2.valgrad_Q[k]@J_inv.T)**2*det_J        
    
        return val
    def estimate_eta(self, u):
        self.u = u
        val=0
        for element_num in range(self.g.num_cells):
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
            dPhi = P_El.grad_phi_eval(a[:,0:2]) 
            # Local pressure head values
            psi_local = np.array([self.u[cn[0]],self.u[cn[1]],self.u[cn[2]]])
            
            
            R_h = np.dot((psi_local).T.reshape(len(psi_local)),Phi.T)
            
            
            val =0
            for k in range(len(Phi)):
                val += 0.5*a[k][2]*np.linalg.norm(R_h[k]*dPhi[k][:]@J_inv.T)**2*det_J
                
            return np.sqrt(val)
    
    def estimate_CN(self,K,K_prime,theta_prime,u):
        x = sp.symbols('x')
        
        eta=self.estimate_eta(u)
        # Initialize
        C_array = np.zeros((len(u),1))
        theta_in_Q = np.zeros((len(u),1))
        K_in_Q = np.zeros((len(u),1))
        theta_prime_Q = np.zeros((len(u),1))
        K_d_theta = np.zeros((len(u),1))
        K_prime_Q = np.zeros((len(u),1))
        for k in range(len(u)):
            theta_in_Q[k] = self.theta(u[k].item())
            K_in_Q[k]     = K(theta_in_Q[k].item())
            
            #self.K[k]     = self.K_func(self.val_Q[k].item())
            #r = self.K_prime_func.subs(x,self.val_Q[k].item())
            z = theta_prime.subs(x,u[k].item())
            r = K_prime.subs(x,theta_in_Q[k].item())

            theta_prime_Q[k] = z#self.theta_prime_func.evalf(subs={x: np.asscalar(self.val[k])})
            #print(r)
            # derivative of theta wrt psi
            K_d_theta[k] = r #self.K_prime_func(self.theta[k])
            # derivative of K wrt psi
            K_prime_Q[k] = K_d_theta[k]*theta_prime_Q[k]
            
            C_array[k]=self.dt*np.linalg.norm(K_in_Q[k]**(-1/2)*K_prime_Q[k]*(eta+1))**2/theta_prime_Q[k]
      
        self.CN =np.nanmax(C_array)
        if 0<self.CN<2:
            self.CN=self.CN
        else:
            self.CN=0
        return C_array
    
    def N_to_L_eta(self,w1,w2,K_prime,theta_prime):
        '''
        

        Parameters
        ----------
        w1 : psi^(k).
        w2 : psi^(k-1).
        K_prime : Derivative of permability.
        theta_prime : Derivative of stauration.

        Returns
        -------
        Indicator for switch to Newton's method.

        '''
        #self.estimate_CN(self.d.K, K_prime, theta_prime, w1)
        self.CN=0
        self.theta_prime = theta_prime
        self.K_prime = K_prime
        a = 2/(2-self.CN)
        #self.linearization_error(w2, w1-w2) 
        #b= self.linear_norm
        self.w1=w1
        self.w2=w2
        self.diff = self.w1-self.w2
        pool = Pool()
        elements_list =list(range(self.g.num_cells))
        results = pool.map(self.norm_N_to_L_on_element,elements_list)
        
        val  = 0
        val2 = 0
        val3 = 0
        for result in results:
            
            val  += result[0]
            val2 += result[1]
            val3 += result[2]
            #print(result[2])
        
        self.linear_norm=np.sqrt(val3)
        self.eta_NtoL = a/self.linear_norm*np.sqrt(val+self.dt*val2)
        
    def norm_N_to_L_on_element(self,element_num):
        cn      = self.d.mapping[:,element_num] # Node values of element
        corners = self.d.geometry.nodes[0:2,cn[self.d.local_dofs_corners]]  # corners of element
        
        # PK element
        P_El    = global_element_geometry(self.d.element, corners,self.d.geometry,self.d.degree)
        
        # Map reference element to local element
        [J, c, J_inv]     = reference_to_local(P_El.element_coord)
        det_J             = np.abs(np.linalg.det(J))
        # Fetch quadrature points
        a                 =  gauss_quadrature_points(2) 
        quadrature_points = a[:,0:2]
        quadrature_pt     = np.array([quadrature_points]).T
        
        # Evaluate basis functions
        Phi  = P_El.phi_eval(a[:,0:2])
        dPhi = P_El.grad_phi_eval(a[:,0:2]) 
        
        # Local Values
        psi_local   = np.array([self.w2[cn[0]],self.w2[cn[1]],self.w2[cn[2]]])
        psi_local2  = np.array([self.w1[cn[0]],self.w1[cn[1]],self.w1[cn[2]]])
        psi_local3  = np.array([self.diff[cn[0]],self.diff[cn[1]],self.diff[cn[2]]])
        local_vals  = localelement_function_evaluation(self.d.K,self.d.theta,self.K_prime,self.theta_prime,psi_local2,P_El)
        local_vals2 = localelement_function_evaluation(self.d.K,self.d.theta,self.K_prime,self.theta_prime,psi_local,P_El)
        local_vals3 = localelement_function_evaluation_L(self.d.K,self.d.theta,psi_local3,P_El) 
        
        R_h = np.dot((psi_local3).T.reshape(len(psi_local)),Phi.T)
        #if element_num ==5:
        #print(R_h,'fact')
        val  = 0
        val2 = 0
        val3 = 0
        for k in range(len(Phi)):
            if 0== local_vals.theta_prime_Q[k]:
                val+=0# 0.5*a[k][2]*det_J*((np.sqrt(2)/np.pi)/local_vals.K_in_Q[k]**(1/2)*(self.L*(local_vals.val_Q[k]-local_vals2.val_Q[k])-(local_vals.theta_in_Q[k]-local_vals2.theta_in_Q[k])))**2
         
            else:
                val += 0.5*a[k][2]*det_J*(1/local_vals.theta_prime_Q[k]**(1/2)*(local_vals.theta_prime_Q[k]*(local_vals.val_Q[k]-local_vals2.val_Q[k])-(local_vals.theta_in_Q[k]-local_vals2.theta_in_Q[k])))**2
            #val += 0.5*a[k][2]*det_J*(1/local_vals.theta_prime_Q[k]**(1/2)*((local_vals.val_Q[k]-local_vals2.val_Q[k])-(local_vals.theta_in_Q[k]-local_vals2.theta_in_Q[k])))**2
           
            # if local_vals.K_in_Q[k]==0:
            #     val2+=0
            # else:
            'careful with paranthesis here'
            val2 += 0.5*a[k][2]*det_J*((((local_vals.K_in_Q[k]-local_vals2.K_in_Q[k]))-(local_vals2.K_prime_Q[k]*(local_vals.val_Q[k]-local_vals2.val_Q[k])))*1/local_vals.K_in_Q[k]**(1/2)*np.linalg.norm(local_vals.valgrad_Q[k]@J_inv.T+np.array([0,1])))**2
            val3 += 0.5*a[k][2]*det_J*local_vals2.theta_prime_Q[k]*(R_h[k])**2+self.dt*0.5*a[k][2]*local_vals2.K_in_Q[k]*np.linalg.norm(local_vals3.valgrad_Q[k]@J_inv.T)**2*det_J        
        
        return val,val2,val3
    
    
class LN_alg():
    def __init__(self,L,dt,d,g,order,psi, K,theta,K_prime,theta_prime,f, Switch=False):
        
        d.Newton_method(K, theta,K_prime,theta_prime,f)
        
        self.L     =L
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
        
        
        _data_mass = np.empty(d.numDataPts, dtype=np.double)
        n=0
   
 
        pool = Pool()
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
        

        
      
    
    
    def update_at_iteration(self,psi_k,ind,Switch):
        self.psi_k = psi_k
        # Initalize matrices
        
        if Switch ==True:
            if ind ==1:
                n=0
        
                
                _data_J_perm = []
                _data_J_sat = []
                _data_J_grav = []
                pool = Pool()
                elements_list =list(range(self.g.num_cells))
                results = pool.map(self.local_derivative_assembly,elements_list)
        
                for result in results:
            
                    
                    _data_J_perm.append(result[0].flatten())
                    _data_J_sat.append(result[2].flatten())
                    _data_J_grav.append(result[1].flatten())
        
                _data_J_perm = np.concatenate(_data_J_perm)
                _data_J_sat = np.concatenate(_data_J_sat)
                _data_J_grav = np.concatenate(_data_J_grav)
       
                self.J_perm_matrix = coo_matrix((_data_J_perm,(self.d.row,self.d.col)))#.tocsr()
                self.J_sat_matrix = coo_matrix((_data_J_sat,(self.d.row,self.d.col)))#.tocsr()
                self.J_gravity_matrix = coo_matrix((_data_J_grav,(self.d.row,self.d.col)))#.tocsr()
            
            
            else:
                self.sat_matrix_k   = np.zeros((self.d.total_geometric_pts, 1)) 
                self.gravity_vector = np.zeros((self.d.total_geometric_pts, 1)) 
  
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
       
                self.perm_matrix = coo_matrix((_data_perm,(self.d.row,self.d.col)))#.tocsr()
                self.J_perm_matrix = coo_matrix((_data_J_perm,(self.d.row,self.d.col)))#.tocsr()
                self.J_sat_matrix = coo_matrix((_data_J_sat,(self.d.row,self.d.col)))#.tocsr()
                self.J_gravity_matrix = coo_matrix((_data_J_grav,(self.d.row,self.d.col)))#.tocsr()
            
        else:
            self.psi_k = psi_k
            # Initalize matrices
            #self.perm_matrix    = np.zeros((self.d.total_geometric_pts, self.d.total_geometric_pts)) 
            self.sat_matrix_k   = np.zeros((self.d.total_geometric_pts, 1)) 
            self.gravity_vector = np.zeros((self.d.total_geometric_pts, 1)) 
            _data_perm = np.empty(self.d.numDataPts, dtype=np.double)
            
            #element = finite_element(self.order)
            n=0
            
            pool = Pool()
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
        pool = Pool()
        elements_list =list(range(self.g.num_cells))
        results = pool.map(self.local_source_saturation_assembly,elements_list)
        
        for result in results:
            
          
                 
            for i in range(len(result[-1])):
                self.sat_matrix_t[result[-1][i]] = self.sat_matrix_t[result[-1][i]] + result[1][i]
       
                    
                self.source_vector[result[-1][i]]= self.source_vector[result[-1][i]]+result[0][i]
   
    
              
         
    def assemble(self,psi_k,Switch):
            
        if Switch==True:   
            self.lhs = self.J_sat_matrix+self.dt*self.perm_matrix+self.dt*self.J_perm_matrix+self.dt*self.J_gravity_matrix
            self.rhs = self.J_sat_matrix@psi_k +self.sat_matrix_t-self.sat_matrix_k + self.dt*self.source_vector - self.dt*self.gravity_vector \
                    +self.dt*self.J_perm_matrix@psi_k+self.dt*self.J_gravity_matrix@psi_k
    

        else:
        
            self.lhs = self.L*self.mass_matrix+self.dt*self.perm_matrix
            self.rhs = self.L*self.mass_matrix.dot(psi_k) +self.sat_matrix_t-self.sat_matrix_k + self.dt*self.source_vector - self.dt*self.gravity_vector
     
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
    

    def local_derivative_assembly(self,element_num):
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
        
   
        local_J_perm =  np.zeros((P_El.num_dofs,P_El.num_dofs))
        local_J_gravity = np.zeros((P_El.num_dofs,P_El.num_dofs))
        local_J_saturation = np.zeros((P_El.num_dofs,P_El.num_dofs))
        
        transform = J_inv@J_inv.T
        for j in range(P_El.num_dofs):
            
            for i in range(P_El.num_dofs):
               
                val4 = 0
                val5 = 0
                val6 = 0
                for l in range(len(Phi)):
                  
                    val4 += local_vals.K_prime_Q[l]*a[l][2]*Phi[l][i]*local_vals.valgrad_Q[l]@transform@dPhi[l][j].T  
                    val5 += local_vals.K_prime_Q[l]*a[l][2]*Phi[l][i]*np.inner(np.array([0,1]),dPhi[l][j]@J_inv.T)
                    val6 += local_vals.theta_prime_Q[l]*a[l][2]*Phi[l][i]*Phi[l][j]
                
            
                local_J_perm[j,i]       += 0.5*val4*det_J
                local_J_gravity[j,i]    += 0.5*val5*det_J
                local_J_saturation[j,i] += 0.5*val6*det_J
            

        return local_J_perm, local_J_gravity, local_J_saturation, cn
    
        
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
    
    def linearization_error(self,u_j,u_j_1):
        '''
        Computes lineartization error wrt iteration dependent energy norm.

        '''
        self.u_j = u_j
        self.u_j_1 = u_j_1
        val=0
        pool = Pool()
        elements_list =list(range(self.g.num_cells))
        results = pool.map(self.error_on_element,elements_list)
        val= np.sum(results)
        # for e in range(self.g.num_cells):
        #     val +=self.error_on_element(e)
            
        self.linear_norm = np.sqrt(val)
    def error_on_element(self,element_num):
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
        dPhi = P_El.grad_phi_eval(a[:,0:2]) 
        # Local pressure head values
        psi_local = np.array([self.u_j_1[cn[0]],self.u_j_1[cn[1]],self.u_j_1[cn[2]]])
        psi_local2 = np.array([self.u_j[cn[0]],self.u_j[cn[1]],self.u_j[cn[2]]])
        local_vals = localelement_function_evaluation(self.d.K,self.d.theta,self.d.K_prime,self.d.theta_prime,psi_local2,P_El)
        local_vals2 = localelement_function_evaluation(self.d.K,self.d.theta,self.d.K_prime,self.d.theta_prime,psi_local,P_El)
        
        
        R_h = np.dot((psi_local).T.reshape(len(psi_local)),Phi.T)
        
        # if element_num == 5:
        #     print(R_h,'tttt')
        
        val =0
        for k in range(len(Phi)):
            val += 0.5*a[k][2]*local_vals.theta_prime_Q[k]*R_h[k]**2*det_J+0.5*a[k][2]*self.dt*local_vals.K_in_Q[k]*np.linalg.norm(local_vals2.valgrad_Q[k]@J_inv.T)**2*det_J        
    
        return val
    def estimate_eta(self, u):
        self.u = u
        val=0
        for element_num in range(self.g.num_cells):
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
            dPhi = P_El.grad_phi_eval(a[:,0:2]) 
            # Local pressure head values
            psi_local = np.array([self.u[cn[0]],self.u[cn[1]],self.u[cn[2]]])
            
            
            R_h = np.dot((psi_local).T.reshape(len(psi_local)),Phi.T)
            
            
            val =0
            for k in range(len(Phi)):
                val += 0.5*a[k][2]*np.linalg.norm(R_h[k]*dPhi[k][:]@J_inv.T)**2*det_J
                
            return np.sqrt(val)
    
    def estimate_CN(self,K,K_prime,theta_prime,u):
        x = sp.symbols('x')
        
        eta=self.estimate_eta(u)
        # Initialize
        C_array = np.zeros((len(u),1))
        theta_in_Q = np.zeros((len(u),1))
        K_in_Q = np.zeros((len(u),1))
        theta_prime_Q = np.zeros((len(u),1))
        K_d_theta = np.zeros((len(u),1))
        K_prime_Q = np.zeros((len(u),1))
        for k in range(len(u)):
            theta_in_Q[k] = self.theta(u[k].item())
            K_in_Q[k]     = K(theta_in_Q[k].item())
            
            #self.K[k]     = self.K_func(self.val_Q[k].item())
            #r = self.K_prime_func.subs(x,self.val_Q[k].item())
            z = theta_prime.subs(x,u[k].item())
            r = K_prime.subs(x,theta_in_Q[k].item())

            theta_prime_Q[k] = z#self.theta_prime_func.evalf(subs={x: np.asscalar(self.val[k])})
            #print(r)
            # derivative of theta wrt psi
            K_d_theta[k] = r #self.K_prime_func(self.theta[k])
            # derivative of K wrt psi
            K_prime_Q[k] = K_d_theta[k]*theta_prime_Q[k]
            
            C_array[k]=self.dt*np.linalg.norm(K_in_Q[k]**(-1/2)*K_prime_Q[k]*(eta+1))**2/theta_prime_Q[k]
      
        self.CN =np.nanmax(C_array)
        if 0<self.CN<2:
            self.CN=self.CN
        else:
            self.CN=0
        return C_array
    
    def N_to_L_eta(self,w1,w2,K_prime,theta_prime):
        '''
        

        Parameters
        ----------
        w1 : psi^(k).
        w2 : psi^(k-1).
        K_prime : Derivative of permability.
        theta_prime : Derivative of stauration.

        Returns
        -------
        Indicator for switch to Newton's method.

        '''
        #self.estimate_CN(self.d.K, K_prime, theta_prime, w1)
        self.CN=0
        self.theta_prime = theta_prime
        self.K_prime = K_prime
        a = 2/(2-self.CN)
        #self.linearization_error(w2, w1-w2) 
        #b= self.linear_norm
        self.w1=w1
        self.w2=w2
        self.diff = self.w1-self.w2
        pool = Pool()
        elements_list =list(range(self.g.num_cells))
        results = pool.map(self.norm_N_to_L_on_element,elements_list)
        
        val  = 0
        val2 = 0
        val3 = 0
        for result in results:
            
            val  += result[0]
            val2 += result[1]
            val3 += result[2]
            #print(result[2])
        
        self.linear_norm=np.sqrt(val3)
        self.eta_NtoL = a/self.linear_norm*np.sqrt(val+self.dt*val2)
        
    def norm_N_to_L_on_element(self,element_num):
        cn      = self.d.mapping[:,element_num] # Node values of element
        corners = self.d.geometry.nodes[0:2,cn[self.d.local_dofs_corners]]  # corners of element
        
        # PK element
        P_El    = global_element_geometry(self.d.element, corners,self.d.geometry,self.d.degree)
        
        # Map reference element to local element
        [J, c, J_inv]     = reference_to_local(P_El.element_coord)
        det_J             = np.abs(np.linalg.det(J))
        # Fetch quadrature points
        a                 =  gauss_quadrature_points(2) 
        quadrature_points = a[:,0:2]
        quadrature_pt     = np.array([quadrature_points]).T
        
        # Evaluate basis functions
        Phi  = P_El.phi_eval(a[:,0:2])
        dPhi = P_El.grad_phi_eval(a[:,0:2]) 
        
        # Local Values
        psi_local   = np.array([self.w2[cn[0]],self.w2[cn[1]],self.w2[cn[2]]])
        psi_local2  = np.array([self.w1[cn[0]],self.w1[cn[1]],self.w1[cn[2]]])
        psi_local3  = np.array([self.diff[cn[0]],self.diff[cn[1]],self.diff[cn[2]]])
        local_vals  = localelement_function_evaluation(self.d.K,self.d.theta,self.K_prime,self.theta_prime,psi_local2,P_El)
        local_vals2 = localelement_function_evaluation(self.d.K,self.d.theta,self.K_prime,self.theta_prime,psi_local,P_El)
        local_vals3 = localelement_function_evaluation_L(self.d.K,self.d.theta,psi_local3,P_El) 
        
        R_h = np.dot((psi_local3).T.reshape(len(psi_local)),Phi.T)
        #if element_num ==5:
        #print(R_h,'fact')
        val  = 0
        val2 = 0
        val3 = 0
        for k in range(len(Phi)):
            if 0== local_vals.theta_prime_Q[k]:
                val+=0# 0.5*a[k][2]*det_J*((np.sqrt(2)/np.pi)/local_vals.K_in_Q[k]**(1/2)*(self.L*(local_vals.val_Q[k]-local_vals2.val_Q[k])-(local_vals.theta_in_Q[k]-local_vals2.theta_in_Q[k])))**2
         
            else:
                val += 0.5*a[k][2]*det_J*(1/local_vals.theta_prime_Q[k]**(1/2)*(local_vals.theta_prime_Q[k]*(local_vals.val_Q[k]-local_vals2.val_Q[k])-(local_vals.theta_in_Q[k]-local_vals2.theta_in_Q[k])))**2
            #val += 0.5*a[k][2]*det_J*(1/local_vals.theta_prime_Q[k]**(1/2)*((local_vals.val_Q[k]-local_vals2.val_Q[k])-(local_vals.theta_in_Q[k]-local_vals2.theta_in_Q[k])))**2
           
            # if local_vals.K_in_Q[k]==0:
            #     val2+=0
            # else:
            'careful with paranthesis here'
            val2 += 0.5*a[k][2]*det_J*((((local_vals.K_in_Q[k]-local_vals2.K_in_Q[k]))-(local_vals2.K_prime_Q[k]*(local_vals.val_Q[k]-local_vals2.val_Q[k])))*1/local_vals.K_in_Q[k]**(1/2)*np.linalg.norm(local_vals.valgrad_Q[k]@J_inv.T+np.array([0,1])))**2
            val3 += 0.5*a[k][2]*det_J*local_vals2.theta_prime_Q[k]*(R_h[k])**2+self.dt*0.5*a[k][2]*local_vals2.K_in_Q[k]*np.linalg.norm(local_vals3.valgrad_Q[k]@J_inv.T)**2*det_J        
        
        return val,val2,val3
    
    def L_to_N_eta(self,w1,w2,K_prime,theta_prime):
        '''
        

        Parameters
        ----------
        w1 : psi^(k).
        w2 : psi^(k-1).
        K_prime : Derivative of permability.
        theta_prime : Derivative of stauration.

        Returns
        -------
        Indicator for switch to Newton's method.

        '''
        self.estimate_CN(self.d.K, K_prime, theta_prime, w1) # estimate C_N^i
        self.theta_prime = theta_prime
        self.K_prime = K_prime
        a = 2/(2-self.CN)
        self.w1=w1
        self.w2=w2
        self.diff = self.w1-self.w2
        pool = Pool()
        elements_list =list(range(self.g.num_cells))
        results = pool.map(self.norm_L_to_N_on_element,elements_list)
        
        val  = 0
        val2 = 0
        val3 = 0
        val4 = 0
        for result in results:
            
            val  += result[0]
            val2 += result[1] 
            val3 += result[2] # L-scheme linearization norm error
            val4 += result[3]
        # for e in range(self.g.num_cells):
        #     q,w,r,o = self.norm_L_to_N_on_element(e)
        #     val += q
        #     val2 +=w
        #     val3+= r
        #     val4+= o
            
        #eta_L1 = np.sqrt(val4)
        self.linear_norm = np.sqrt(val3)
        self.eta_LtoL = a/self.linear_norm*np.sqrt(val4+self.dt*val2)
        self.eta_LtoN = a/self.linear_norm*np.sqrt(val+self.dt*val2)
    
    def norm_L_to_N_on_element(self,element_num):
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
        dPhi = P_El.grad_phi_eval(a[:,0:2]) 
        psi_local   = np.array([self.w2[cn[0]],self.w2[cn[1]],self.w2[cn[2]]])
        psi_local2  = np.array([self.w1[cn[0]],self.w1[cn[1]],self.w1[cn[2]]])
        psi_local3  = np.array([self.diff[cn[0]],self.diff[cn[1]],self.diff[cn[2]]])
        local_vals  = localelement_function_evaluation(self.d.K,self.d.theta,self.K_prime,self.theta_prime,psi_local2,P_El)
        local_vals2 = localelement_function_evaluation(self.d.K,self.d.theta,self.K_prime,self.theta_prime,psi_local,P_El)
        local_vals3 = localelement_function_evaluation_L(self.d.K,self.d.theta,psi_local3,P_El) 
       
        val  = 0
        val2 = 0
        val3 = 0
        val4 = 0
 
        R_h = np.dot((psi_local3).T.reshape(len(psi_local)),Phi.T)
           
        
       
            
        for k in range(len(Phi)):
            if 0== local_vals.theta_prime_Q[k]:
                val+= 0#0.5*a[k][2]*det_J*((np.sqrt(2)/np.pi)/local_vals.K_in_Q[k]**(1/2)*(self.L*(local_vals.val_Q[k]-local_vals2.val_Q[k])-(local_vals.theta_in_Q[k]-local_vals2.theta_in_Q[k])))**2
                
            else:
                val  += 0.5*a[k][2]*det_J*(1/local_vals.theta_prime_Q[k]**(1/2)*(self.L*(local_vals.val_Q[k]-local_vals2.val_Q[k])-(local_vals.theta_in_Q[k]-local_vals2.theta_in_Q[k])))**2
            
            val4 += 0.5*a[k][2]*det_J*(1/self.L**(1/2)*(self.L*(local_vals.val_Q[k]-local_vals2.val_Q[k])-(local_vals.theta_in_Q[k]-local_vals2.theta_in_Q[k])))**2
        
            val2 += 0.5*a[k][2]*det_J*(1/local_vals.K_in_Q[k]**(1/2)*((local_vals.K_in_Q[k]-local_vals2.K_in_Q[k]))*np.linalg.norm(local_vals.valgrad_Q[k]@J_inv.T+np.array([0,1])))**2
            val3 += 0.5*a[k][2]*self.L*R_h[k]**2*det_J+self.dt*0.5*a[k][2]*local_vals2.K_in_Q[k]*np.linalg.norm(local_vals3.valgrad_Q[k]@J_inv.T)**2*det_J        
            
        return val,val2,val3,val4
    def update_L(self,L):
        self.L=L