# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:39:04 2022

@author: jakob
"""

import numpy as np
from sympy.tensor.array import derive_by_array
from sympy import Symbol, Matrix, Function, simplify
import sympy as sp

def divergence(f, x):
    return sum(fi.diff(xi) for fi, xi in zip(f, x))

def construct_source_function(u,theta,K):
    t = sp.symbols('t')
    x = sp.symbols('x')
    y = sp.symbols('y')
    # time derivative of theta
    theta_t = sp.diff(theta(u(t,x,y)),t)
   
    # gradient of u
    grad_u = derive_by_array(u(t,x,y), (x,y))
    
    # divergence of K(u)*(grad(u)+grad(z))
    K_sym = K(theta(u(t,x,y)))
    
    q = K_sym*grad_u
    div_q =divergence(q, [x, y]) 
    
    # compute source function
    f=theta_t-div_q
    f = sp.lambdify([t,x,y],f)
    return f
def construct_source_function2(u,theta,K):
    t = sp.symbols('t')
    x = sp.symbols('x')
    y = sp.symbols('y')
    # time derivative of theta
    theta_t = sp.diff(theta(u(t,x,y)),t)
   
    # gradient of u
    grad_u = derive_by_array(u(t,x,y), (x,y))
    
    # divergence of K(u)*(grad(u)+grad(z))
    K_sym = K(u(t,x,y))
    
    q = K_sym*grad_u
    div_q =divergence(q, [x, y]) 
    
    # compute source function
    f=theta_t-div_q
    f = sp.lambdify([t,x,y],f)
    print(f(t,x,y))
    return f

def construct_source_function_withGrav(u,theta,K):
    t = sp.symbols('t')
    x = sp.symbols('x')
    y = sp.symbols('y')
    # time derivative of theta
    theta_t = sp.diff(theta(u(t,x,y)),t)
   
    # gradient of u
    grad_u = derive_by_array(u(t,x,y), (x,y))
    
    # divergence of K(u)*(grad(u)+grad(z))
    K_sym = K(theta(u(t,x,y)))
    
    q = K_sym*(grad_u+sp.Array([0,1]))
    div_q =divergence(q, [x, y]) 
    
    # compute source function
    f=theta_t-div_q
    f = sp.lambdify([t,x,y],f)
    return f