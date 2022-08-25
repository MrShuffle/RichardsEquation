# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:52:30 2022

@author: jakob
"""

import numpy as np
import matplotlib.pyplot as plt
import porepy as pp
from scipy.special import binom
from itertools import combinations_with_replacement



def vandermonde(points, degree):
    '''
    Generates a Vandermonde matrix for polynomials up to degree
    :param = degree.


    Parameters
    ----------
    points : points of a finite element.
    degree : degree of polynomial basis for the Vandermonde matrix.

    Returns
    -------
    V : Vandermonde matrix of size (number of nodes(or points) x 
                                    dimension of polynomial space).

    '''
    itr = 0

    # Dimension of the ploynomial space in 2D
    dim_of_space = int((degree+1)*(degree+2)/2)
    # Initialize the matrix
    V = np.ones((len(points), dim_of_space))
   
    for k in range(degree + 1):
        for j in combinations_with_replacement(range(2), k):
            for i in j:
        
                V[:, itr] *= points[:,i]

            itr += 1
    return V
def vandermonde_Grad(points, degree):
    '''
    Generates a Gradient of a Vandermonde matrix for polynomials up to degree
    :param = degree.


    Parameters
    ----------
    g : geometry generated from porepy.
    degree : degree of polynomial basis for the Vandermonde matrix.

    Returns
    -------
    dV : Gradient of a Vandermonde matrix of size (number of nodes x dimension of polynomial
                                    space x spatial dimension).

    '''
    itr = 0
    # Dimension of the ploynomial space in 2D
    dim_of_space = int((degree+1)*(degree+2)/2)
    
    dV = np.ones((len(points),dim_of_space,2))
    
    
    for k in range(degree+1):
        for j in combinations_with_replacement(range(2), k):
            A = np.zeros(2)
            
            for i in j:
                A[i] += 1
                
            # derive in all spatial dimensions
            for i in range(2):
                d_A = A.copy()
                d_A[i] -= 1
                d_i = A[i]
                
                if d_i <= 0:
                    # derivative in ith direction is 0
                    dV[:,itr,i] = 0
                else:
                    for b in range(len(A)):
                        
                        dV[:,itr,i] *= points[:,b]**d_A[b]
                      
                    dV[:,itr,i] *= d_i
            itr += 1

    return dV

class finite_element():
    def __init__(self,degree):
        self.degree = degree
        self.poly_dim = int((degree+1)*(degree+2)/2) # dimension of mononomial basis of polynomials 
        
        
        # P1 element
        if self.degree == 1:
            
            self.pts = np.array([[0,0],
                                  [1,0],
                                  [0,1]])
            # dictionary with local nodes
            self.local_dofs =  {'nodes': {0: [0],
                                     1: [2],
                                     2: [1]},
                                'faces':{},
                                'elements':{}}
            

            self.num_dofs = 3
        
        # P2 element
        elif self.degree ==2:
            
            self.pts = self.__lagrange_pts(self.degree)
            self.pts = np.array([[0,0],
                                  [0.5,0],
                                  [1,0],
                                  [0.5,0.5],
                                  [0,1],
                                  [0,0.5]])
            self.local_dofs =   {'nodes': {0: [0],
                                     1: [4],
                                     2: [2]},
                                 'faces': {0: [5],
                                     1: [1],
                                     2: [3]},
                                 'elements':{}}
        
            self.num_dofs = 6
        else:
            NotImplementedError()
            
        V = vandermonde(self.pts, self.degree)
       
        self.inv_V = np.linalg.inv(V)
        self.V = V
        
    # Evaluate test functions
    def phi_eval(self,points):
        V = vandermonde(points, self.degree)
        
        return V@self.inv_V
    
    # Evaluate gradient of test functions
    def grad_phi_eval(self,points):
        dV = vandermonde_Grad(points, self.degree)
        gradPhi = np.empty((dV.shape[0], self.poly_dim, 2))
        
        for i in range(dV.shape[0]):  # for each point
        
            gradPhi[i] = np.dot( self.inv_V.T,dV[i] )
            
        return gradPhi
    
    
    def __lagrange_pts(self, degree):
        # generate equally spaced nodes
        #  top to bottom, left to right
        pts = []
        for i in range(degree+1):
            for j in range(degree+1-i):
                pts.append([i/degree, j/degree])

        return np.array(pts, dtype=np.double)
    
class global_element_geometry(finite_element):
    '''This class inherits from the finite element class.
    
    The purpose of this class is to extract the local geometry properties 
    of any particular element from the global triangulation.'''
    def __init__(self,element,corners,geometry,order):
        super().__init__(order)
        self.element = element
        self.corners = corners
        self.geometry = geometry
       
        
        
        # P1 element
        if self.degree == 1:

            self.element_coord = self.corners
        
        # P2 element
        elif self.degree ==2:
 
            self.element_coord = self.mid_point(self.corners)
            #print(self.element_coord)
            #self.element_coord = self.coordinates_sorted(self.geometry,2)
            #print(self.element_coord)

        else:
            NotImplementedError()
            

    # TODO: use porepy's g.face_centers instead.
    def mid_point(self,corners):
        
        x_coord = corners[0]
        y_coord = corners[1]
        #print(y_coord)
        x_M = np.zeros(len(x_coord))
        y_M = np.zeros(len(y_coord))
        itr =0
        
        x_M = (x_coord[1:] + x_coord[:-1]) / 2
        x_M= np.append(x_M,(x_coord[0]+x_coord[-1])/2)
        y_M = (y_coord[1:] + y_coord[:-1]) / 2
        y_M= np.append(y_M,(y_coord[0]+y_coord[-1])/2)
        
        
        
        x_coord2 = np.append(x_coord,x_M)
        y_coord2 = np.append(y_coord,y_M)
        
        idx_new = [0,1,2,3,4,5]
        idx = [0,3,1,4,2,5]
        x_coord2[idx_new]=  x_coord2[idx]  
        y_coord2[idx_new]=  y_coord2[idx] 
        
        points =np.array([x_coord2,y_coord2])
        #points =np.array([x_mod,y_mod])
        return points 


        
        
        
        


            
