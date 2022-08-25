# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:34:35 2022

@author: jakob
"""

import porepy as pp
from RichardsEqFEM.source.LocalGlobalMapping.map_local_to_global import Local_to_Global_table
from RichardsEqFEM.source.basisfunctions.Gauss_quadrature_points import *
from sympy.tensor.array import derive_by_array
from sympy import Symbol, Matrix, Function, simplify
import numpy as np
import sympy as sp


class localelement_function_evaluation():
    def __init__(self,K,theta,K_prime,theta_prime,u,P_El):
        '''
        

        Parameters
        ----------
        K : Permability -function of theta.
        theta : Saturation -function of psi.
        u : a vector with psi value.
        PK : PK element number of the class global_element_geometry(element,corners,g,order).

        Returns
        -------
        Permability and theta evaluated at psi at the local element.

        '''
        
        #idea define a tring with keywords like {K:'permability', theta:'saturation'}
        # the extract keywords if they are given, both K and theta should be functions, but maybe this should be input 
        # matrix assembly, this would allow for specified functions determining what to do with each
        # they will both be part of the residual
        
        self.val = u
        self.theta_func = theta
        self.theta_prime_func = theta_prime
        self.K_func = K
        self.K_prime_func = K_prime#*theta_prime
        
        # # Derivative of theta wrt psi
        x = sp.symbols('x')
        # self.theta_prime = sp.diff(self.theta_func(x),x)
        # # Derivative of K wrt theta
        # self.K_prime = sp.diff(self.K_func(x),x)
       
        
        
        self.theta = np.zeros((len(u),1))
        self.K = np.zeros((len(u),1))
        self.theta_d_psi = np.zeros((len(u),1))
        self.K_d_theta = np.zeros((len(u),1))
        self.K_d_psi = np.zeros((len(u),1))
        for k in range(len(u)):
            self.theta[k] = self.theta_func(self.val[k])
            self.K[k]     = self.K_func(self.theta[k])
    
            z = self.theta_prime_func.subs(x,self.val[k].item())
            r = self.K_prime_func.subs(x,self.theta[k].item())

            self.theta_d_psi[k] = z#self.theta_prime_func.evalf(subs={x: np.asscalar(self.val[k])})
          
            self.K_d_theta[k] = r #self.K_prime_func(self.theta[k])
            # derivative of K wrt psi
            self.K_d_psi[k] = self.K_d_theta[k]*self.theta_d_psi[k]
            
        self.interpolate_to_quadpts(P_El) # interpolates K and theta on a local element to quadtarure points
    
    def interpolate_to_quadpts(self,P_El):
        quadrature = gauss_quadrature_points(P_El.degree +1)
        quadrature_points = quadrature[:,0:2]
        Phi = P_El.phi_eval(quadrature_points)
        self.K_in_Q = np.dot(self.K.T.reshape(len(self.K)),Phi.T)
        self.theta_in_Q = np.dot(self.theta.T.reshape(len(self.theta)),Phi.T)
        self.K_prime_Q = np.dot(self.K_d_psi.T.reshape(len(self.K)),Phi.T)
        self.theta_prime_Q = np.dot(self.theta_d_psi.T.reshape(len(self.theta)),Phi.T)
        
        
        
        
        