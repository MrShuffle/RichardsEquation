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
        
        #self.val = u
        theta_func = theta
        x = sp.symbols('x')
        theta_prime_func =sp.lambdify([x],theta_prime)
        K_func = K
        K_prime_func = sp.lambdify([x],K_prime)
        
        
        
        
        quadrature = gauss_quadrature_points(P_El.degree +1)
        quadrature_points = quadrature[:,0:2]
        Phi = P_El.phi_eval(quadrature_points)
        dPhi = P_El.grad_phi_eval(quadrature_points)
        # self.val_Q = np.zeros((len(Phi),1))# np.dot(self.val.T.reshape(len(self.val)),Phi.T)
        # self.valgrad_Q = np.zeros((dPhi.shape[1],dPhi.shape[2]))
        # sum over quadrature points q
        self.valgrad_Q = np.tensordot(u,dPhi,axes=((0),(1)))[0]
        self.val_Q=np.sum(u*Phi,axis=0)
        # Initialize
        self.theta_in_Q = np.zeros((len(u),1))
        self.K_in_Q = np.zeros((len(u),1))
        self.theta_prime_Q = np.zeros((len(u),1))
        #self.K_d_theta = np.zeros((len(u),1))
        self.K_prime_Q = np.zeros((len(u),1))
        #self.theta_prime_Q = theta_prime_func(self.val_Q)
        #self.K_prime_Q = K_prime_func(self.theta_in_Q)*self.theta_prime_Q
        for k in range(len(u)):
            self.theta_in_Q[k] = theta_func(self.val_Q[k].item())
            self.K_in_Q[k]     = K_func(self.theta_in_Q[k].item())
            
            #self.K[k]     = self.K_func(self.val_Q[k].item())
            #r = self.K_prime_func.subs(x,self.val_Q[k].item())
            # z = self.theta_prime_func.subs(x,self.val_Q[k].item())
            # r = self.K_prime_func.subs(x,self.theta_in_Q[k].item())

            self.theta_prime_Q[k] = theta_prime_func(self.val_Q[k].item())#self.theta_prime_func.evalf(subs={x: np.asscalar(self.val[k])})
            #print(r)
            # derivative of theta wrt psi
            #self.K_d_theta[k] = r #self.K_prime_func(self.theta[k])
            # derivative of K wrt psi
            self.K_prime_Q[k] = K_prime_func(self.theta_in_Q[k].item())*self.theta_prime_Q[k]
        #self.K_prime_Q = np.multiply(K_prime_func(self.theta_in_Q),self.theta_prime_Q)

class localelement_function_evaluation_newtonnorm():
    def __init__(self,K,theta,theta_prime,u,u2,P_El):
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
        
        #self.val = u
        theta_func = theta
        x = sp.symbols('x')
        theta_prime_func =sp.lambdify([x],theta_prime)
        K_func = K
        #K_prime_func = sp.lambdify([x],K_prime)
        
        
        
        
        quadrature = gauss_quadrature_points(P_El.degree +1)
        quadrature_points = quadrature[:,0:2]
        Phi = P_El.phi_eval(quadrature_points)
        dPhi = P_El.grad_phi_eval(quadrature_points)
        # self.val_Q = np.zeros((len(Phi),1))# np.dot(self.val.T.reshape(len(self.val)),Phi.T)
        # self.valgrad_Q = np.zeros((dPhi.shape[1],dPhi.shape[2]))
        # sum over quadrature points q
        self.valgrad_Q = np.tensordot(u,dPhi,axes=((0),(1)))[0]
        #self.val_Q=np.sum(u*Phi,axis=0)
        self.val_Q2=np.sum(u2*Phi,axis=0)
        # Initialize
        self.theta_in_Q2 = np.zeros((len(u),1))
        self.K_in_Q2 = np.zeros((len(u),1))
        self.theta_prime_Q2 = np.zeros((len(u),1))
      
        for k in range(len(u)):
            self.theta_in_Q2[k] = theta_func(self.val_Q2[k].item())
            self.K_in_Q2[k]     = K_func(self.theta_in_Q2[k].item())
            
            #self.K[k]     = self.K_func(self.val_Q[k].item())
            #r = self.K_prime_func.subs(x,self.val_Q[k].item())
            # z = self.theta_prime_func.subs(x,self.val_Q[k].item())
            # r = self.K_prime_func.subs(x,self.theta_in_Q[k].item())

            self.theta_prime_Q2[k] = theta_prime_func(self.val_Q2[k].item())#self.theta_prime_func.evalf(subs={x: np.asscalar(self.val[k])})

class localelement_function_evaluation_Lnorm():
    def __init__(self,K,theta,u,u2,P_El):
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
        
        #self.val = u
        theta_func = theta
        x = sp.symbols('x')
        #theta_prime_func =sp.lambdify([x],theta_prime)
        K_func = K
        #K_prime_func = sp.lambdify([x],K_prime)
        
        
        
        
        quadrature = gauss_quadrature_points(P_El.degree +1)
        quadrature_points = quadrature[:,0:2]
        Phi = P_El.phi_eval(quadrature_points)
        dPhi = P_El.grad_phi_eval(quadrature_points)
        # self.val_Q = np.zeros((len(Phi),1))# np.dot(self.val.T.reshape(len(self.val)),Phi.T)
        # self.valgrad_Q = np.zeros((dPhi.shape[1],dPhi.shape[2]))
        # sum over quadrature points q
        self.valgrad_Q = np.tensordot(u,dPhi,axes=((0),(1)))[0]
        #self.val_Q=np.sum(u*Phi,axis=0)
        self.val_Q2=np.sum(u2*Phi,axis=0)
        # Initialize
        self.theta_in_Q2 = np.zeros((len(u),1))
        self.K_in_Q2 = np.zeros((len(u),1))
        #self.theta_prime_Q2 = np.zeros((len(u),1))
      
        for k in range(len(u)):
            self.theta_in_Q2[k] = theta_func(self.val_Q2[k].item())
            self.K_in_Q2[k]     = K_func(self.theta_in_Q2[k].item())
            
            #self.K[k]     = self.K_func(self.val_Q[k].item())
            #r = self.K_prime_func.subs(x,self.val_Q[k].item())
            # z = self.theta_prime_func.subs(x,self.val_Q[k].item())
            # r = self.K_prime_func.subs(x,self.theta_in_Q[k].item())

            #self.theta_prime_Q2[k] = theta_prime_func(self.val_Q2[k].item())#self.theta_prime_func.evalf(subs={x: np.asscalar(self.val[k])})

        
class localelement_function_evaluation_P():
    def __init__(self,K,theta,theta_prime,u,P_El):
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
        
        
        x = sp.symbols('x')
        
        quadrature = gauss_quadrature_points(P_El.degree +1)
        quadrature_points = quadrature[:,0:2]
        Phi = P_El.phi_eval(quadrature_points)
        dPhi = P_El.grad_phi_eval(quadrature_points)
        self.val_Q = np.zeros((len(Phi),1))# np.dot(self.val.T.reshape(len(self.val)),Phi.T)
        #self.valgrad_Q = np.zeros((dPhi.shape[1],dPhi.shape[2]))
        # sum over quadrature points q
        for q in range(len(Phi)):
            
            # sum over local index
            rtalt = np.sum(np.multiply(self.val,Phi[:,q].reshape(-1,1)))
           
            self.val_Q[q] = rtalt
            #self.valgrad_Q[q] = gradrt
        # rtalt2 = np.dot(self.val,Phi)
        # print(rtalt2,'valdiff',self.val_Q)
        # Initialize
        self.theta = np.zeros((len(u),1))
        self.K = np.zeros((len(u),1))
        self.theta_d_psi = np.zeros((len(u),1))
        #self.K_d_theta = np.zeros((len(u),1))
        #self.K_d_psi = np.zeros((len(u),1))
        for k in range(len(u)):
            self.theta[k] = self.theta_func(self.val_Q[k].item())
            self.K[k]     = self.K_func(self.theta[k].item())
            
            #self.K[k]     = self.K_func(self.val_Q[k].item())
            #r = self.K_prime_func.subs(x,self.val_Q[k].item())
            z = self.theta_prime_func.subs(x,self.val_Q[k].item())
            #r = self.K_prime_func.subs(x,self.theta[k].item())

            self.theta_d_psi[k] = z#self.theta_prime_func.evalf(subs={x: np.asscalar(self.val[k])})
            #print(r)
            # derivative of theta wrt psi
            #self.K_d_theta[k] = r #self.K_prime_func(self.theta[k])
            # derivative of K wrt psi
            #self.K_d_psi[k] = self.K_d_theta[k]*self.theta_d_psi[k]
        
        # TODO does nothing but keeps naming conventions 
        self.interpolate_to_quadpts(P_El) 
    
    def interpolate_to_quadpts(self,P_El):
        quadrature = gauss_quadrature_points(P_El.degree +1)
        quadrature_points = quadrature[:,0:2]
        Phi = P_El.phi_eval(quadrature_points)
        self.K_in_Q = self.K#np.dot(self.K.T.reshape(len(self.K)),Phi.T)
        self.theta_in_Q = self.theta#np.dot(self.theta.T.reshape(len(self.theta)),Phi.T)
        #self.K_prime_Q = self.K_d_psi#np.dot(self.K_d_psi.T.reshape(len(self.K)),Phi)
        self.theta_prime_Q = self.theta_d_psi#np.dot(self.theta_d_psi.T.reshape(len(self.theta)),Phi.T)
        
                
        
class localelement_function_evaluation_L():
    def __init__(self,K,theta,u,P_El):
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
        self.K_func = K
        
        #print(self.val)
        x = sp.symbols('x')
        
        quadrature = gauss_quadrature_points(P_El.degree +1)
        quadrature_points = quadrature[:,0:2]
        Phi = P_El.phi_eval(quadrature_points)
        dPhi = P_El.grad_phi_eval(quadrature_points)
        # self.val_Q = np.zeros((len(Phi),1))# np.dot(self.val.T.reshape(len(self.val)),Phi.T)
        
        # # sum over quadrature points q
        # for q in range(len(Phi)):
        #     # rt = 0
            
        #     # # sum over local index
        #     # for k in range(len(self.val)):
        #     #     rt += self.val[k]*Phi[q][k].T
        #     rtalt = np.sum(np.multiply(self.val,Phi[:,q].reshape(-1,1)))    
        #     self.val_Q[q] = rtalt
        #  # sum over local index
        self.valgrad_Q = np.zeros((dPhi.shape[1],dPhi.shape[2]))
        for q in range(len(Phi)):
            
            gradrt =0
            # sum over local index
            for k in range(len(self.val)):
                
                gradrt += self.val[k]*dPhi[q][k]
            #rtalt = np.sum(np.multiply(self.val,Phi[:,q].reshape(-1,1)))
            #print(rt,rtalt)
            #print(rtalt)
            
            #self.val_Q[q] = rtalt
            self.valgrad_Q[q] = gradrt
        
     
        #  #self.valgrad_Q[q] = gradrt
        rtalt2 = np.sum(self.val*Phi,axis=0)
        #print(rtalt2,'valdiff',self.val_Q)  
        self.val_Q = rtalt2
        # Initialize
        self.theta_in_Q = np.zeros((len(u),1))
        self.K_in_Q = np.zeros((len(u),1))

        for k in range(len(u)):
            self.theta_in_Q[k] = self.theta_func(self.val_Q[k].item())
            self.K_in_Q[k]     = self.K_func(self.theta_in_Q[k].item())
            
class localelement_function_evaluation_norms():
    def __init__(self,K,theta,K_prime,theta_prime,u,u2,u3,P_El):
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
        self.val2=u2
        self.val3=u3
        self.theta_func = theta
        x = sp.symbols('x')
        self.theta_prime_func = sp.lambdify([x],theta_prime)
        self.K_func = K
        #self.K_prime_func = K_prime
        
        
        
        
        quadrature = gauss_quadrature_points(P_El.degree +1)
        quadrature_points = quadrature[:,0:2]
        Phi = P_El.phi_eval(quadrature_points)
        dPhi = P_El.grad_phi_eval(quadrature_points)
        #self.val_Q = np.zeros((len(Phi),1))# np.dot(self.val.T.reshape(len(self.val)),Phi.T)
        self.valgrad_Q = np.zeros((dPhi.shape[1],dPhi.shape[2]))
        #self.val_Q2 = np.zeros((len(Phi),1))# np.dot(self.val.T.reshape(len(self.val)),Phi.T)
        #self.valgrad_Q2 = np.zeros((dPhi.shape[1],dPhi.shape[2]))
        #self.val_Q3 = np.zeros((len(Phi),1))# np.dot(self.val.T.reshape(len(self.val)),Phi.T)
        self.valgrad_Q3 = np.zeros((dPhi.shape[1],dPhi.shape[2]))
        
        self.val_Q=np.sum(self.val*Phi,axis=0)
        self.val_Q2=np.sum(self.val2*Phi,axis=0)
        self.val_Q3=np.sum(self.val3*Phi,axis=0)
        
        # Initialize
        self.theta_in_Q = np.zeros((len(u),1))
        self.K_in_Q = np.zeros((len(u),1))
        self.theta_prime_Q = np.zeros((len(u),1))

        
        self.theta_in_Q2 = np.zeros((len(u),1))
        self.K_in_Q2 = np.zeros((len(u),1))
        self.theta_prime_Q=  self.theta_prime_func(self.val_Q)
        
        self.valgrad_Q = np.tensordot(self.val,dPhi,axes=((0),(1)))[0]
        
        self.valgrad_Q3 = np.tensordot(self.val3,dPhi,axes=((0),(1)))[0]
        # sum over quadrature points q
        for q in range(len(Phi)):
            self.theta_in_Q[q] = self.theta_func(self.val_Q[q].item())
            self.K_in_Q[q]     = self.K_func(self.theta_in_Q[q].item())
  
   

            #self.theta_prime_func.subs(x,self.val_Q[q].item())#self.theta_prime_func.evalf(subs={x: np.asscalar(self.val[k])})
            #print(r)
     
            
            
            self.theta_in_Q2[q] = self.theta_func(self.val_Q2[q].item())
            self.K_in_Q2[q]     = self.K_func(self.theta_in_Q2[q].item())
            
            # gradrt =0
            # #gradrt2 =0
            # gradrt3 =0
            # # sum over local index
            # for k in range(len(self.val)):
                
            #     gradrt += self.val[k]*dPhi[q][k]
            #     #gradrt2 += self.val2[k]*dPhi[q][k]
            #     gradrt3 += self.val3[k]*dPhi[q][k]
                
            # #self.valgrad_Q[q] = self.val*dPhi[q][:]
            # #print(gradrt,'cor',gradrtba)
            # # rtalt = np.sum(np.multiply(self.val,Phi[:,q].reshape(-1,1)))
            # # rtalt2 = np.sum(np.multiply(self.val2,Phi[:,q].reshape(-1,1)))
            # # rtalt3 = np.sum(np.multiply(self.val3,Phi[:,q].reshape(-1,1)))
            # #print(rt,rtalt)
            # #print(rtalt)
            
            # #self.val_Q[q] = rtalt
            # self.valgrad_Q[q] = gradrt
            # #self.val_Q2[q] = rtalt2
            # #self.valgrad_Q2[q] = gradrt2
           
            # self.valgrad_Q3[q] = gradrt3
            
        
      


        
 

        # for k in range(len(u)):
        #     self.theta_in_Q[k] = self.theta_func(self.val_Q[k].item())
        #     self.K_in_Q[k]     = self.K_func(self.theta_in_Q[k].item())
  
   

        #     self.theta_prime_Q[k] = self.theta_prime_func.subs(x,self.val_Q[k].item())#self.theta_prime_func.evalf(subs={x: np.asscalar(self.val[k])})
        #     #print(r)
     
            
            
        #     self.theta_in_Q2[k] = self.theta_func(self.val_Q2[k].item())
        #     self.K_in_Q2[k]     = self.K_func(self.theta_in_Q2[k].item())
            


class localelement_function_evaluation_norms2():
    def __init__(self,K,theta,K_prime,theta_prime,val,val2,val3,P_El):
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
        
        x = sp.symbols('x',real=True)
        
        
        theta_func = theta
        theta_prime_func = sp.lambdify([x],theta_prime)
        K_func = K
        K_prime_func = sp.lambdify([x],K_prime)
        
        
        
        
        quadrature = gauss_quadrature_points(P_El.degree +1)
        quadrature_points = quadrature[:,0:2]
        Phi = P_El.phi_eval(quadrature_points)
        dPhi = P_El.grad_phi_eval(quadrature_points)
       
        
        self.val_Q=np.sum(val*Phi,axis=0)
        self.val_Q2=np.sum(val2*Phi,axis=0)
   
        # Initialize
        self.theta_in_Q = np.zeros((3,1),dtype=np.float64)
        self.K_in_Q = np.zeros((3,1),dtype=np.float64)
        self.theta_prime_Q = np.zeros((3,1),dtype=np.float64)

        
        self.theta_in_Q2 = np.zeros((3,1),dtype=np.float64)
        self.K_in_Q2 = np.zeros((3,1),dtype=np.float64)
        self.theta_prime_Q2 = np.zeros((3,1),dtype=np.float64)
        self.K_prime_Q2=np.zeros((3,1),dtype=np.float64)
    
        # sum over quadrature points q
        self.valgrad_Q = np.tensordot(val,dPhi,axes=((0),(1)))[0]
       
       
        self.valgrad_Q3 =np.tensordot(val3,dPhi,axes=((0),(1)))[0]
        # self.theta_prime_Q = theta_prime_func(self.val_Q)
        #self.theta_prime_Q2 = theta_prime_func(self.val_Q2)
        for q in range(3):
            self.theta_in_Q[q] = theta_func(self.val_Q[q].item())
            self.K_in_Q[q]     = K_func(self.theta_in_Q[q].item())
  
   

            self.theta_prime_Q[q] = theta_prime_func(self.val_Q[q].item())#self.theta_prime_func.subs(x,self.val_Q[q].item())
     
            
            
            self.theta_in_Q2[q] = theta_func(self.val_Q2[q].item())
            self.K_in_Q2[q]     = K_func(self.theta_in_Q2[q].item())
            self.theta_prime_Q2[q] = theta_prime_func(self.val_Q2[q].item())#self.theta_prime_func.subs(x,self.val_Q2[q].item())#self.theta_prime_func.evalf(subs={x: np.asscalar(self.val[k])})
            

         
            self.K_prime_Q2[q] = K_prime_func(self.theta_in_Q2[q].item())*self.theta_prime_Q2[q]#self.K_prime_func.subs(x,self.theta_in_Q2[q].item())*self.theta_prime_Q2[q]
      
            
        
     
            


            

          


          
