# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:54:20 2022

@author: jakob
"""

import numpy as np
import porepy as pp
from RichardsEqFEM.source.basisfunctions.lagrange_element import finite_element
from RichardsEqFEM.source.LocalGlobalMapping.map_local_to_global import Local_to_Global_table
from RichardsEqFEM.source.MatrixAssembly.mass_matrix import mass_matrix, mass_matrix_alternative
from RichardsEqFEM.source.MatrixAssembly.source_term import Source_term_assembly
from RichardsEqFEM.source.MatrixAssembly.Richards_matrices import saturation_matrix_assembly,permability_matrix_assembly
from RichardsEqFEM.source.utils.boundary_conditions import dirichlet_BC
from RichardsEqFEM.source.utils.create_source_term import construct_source_function
from RichardsEqFEM.source.MatrixAssembly.Richards_Jacobian import Jacobian_saturation, Jacobian_matrices
import sympy as sp

def u_fabricated(t,x,y):
    u = -t*(1-x)*(1-y)*x*y 
    
    #u = np.sin(np.pi*x)*np.sin(np.pi*y)/(2*np.pi**2)
    return u

def K(thetaa):
    #val = (thetaa**(-1))*thetaa**(-1)-2*thetaa**(-1)+1
    val =1+np.power(thetaa,4)
    #val=thetaa
    return val

def theta(u):
    #val = 1/(1-u)
    #val=u
    val = 0.125*u+(1.33-0.125)*np.power(u,3)
    #val=u/u
    return val

x = sp.symbols('x')
theta_prime = sp.diff(theta(x),x)
# Derivative of K wrt theta
K_prime = sp.diff(K(x),x)

f = construct_source_function(u_fabricated,theta,K)

x_part=y_part=8
phys_dim = [1,1]
g = pp.StructuredTriangleGrid(np.array([x_part, y_part]),phys_dim)

g.compute_geometry()
#pp.plot_grid(g,info='cf', figsize=(15,12),alpha=0)

order = 1 # Order of polynomial basis
t=0
quad_deg =order+3 # Gaussian quadrature degree

# define lagrange element and local to global map
element = finite_element(order)
d = Local_to_Global_table(g,element)
b_nodes = d.boundary

B = mass_matrix(element,d,order+1, g)

B2 = mass_matrix_alternative(d,g,order)

C = B-B2


f_vect= Source_term_assembly(f,element,d,quad_deg, g, t=0, time=True)

u_h = np.zeros((d.total_geometric_pts,1))
t=0
points1 = g.nodes[0:2]
for k in range(d.total_geometric_pts):
    u_h[k]=u_fabricated(t,points1[0][k],points1[1][k])
    
psi_k = u_h
psi_t = u_h


b_nodes = d.boundary
timesteps = 1

coordinates = g.nodes[0:2]
u_exact = np.zeros([len(psi_t),1])
for i in range(len(u_exact)):
    u_exact[i]=u_fabricated(1,coordinates[0][i],coordinates[1][i])


store = np.zeros((100,1))
corder = np.zeros((100,1))
dt =0.1
L=1.3
TOL = 10**(-15)
ctot=0
for j in range(timesteps):
        count=0
        t=t+dt
        f_vect =Source_term_assembly(f,element,d,quad_deg, g, t, time=True)
        theta_t = theta(psi_t)
        
        #L-scheme iteration
        while True:
            count = count + 1
            A = permability_matrix_assembly(d,g,order,K,theta,K_prime,theta_prime,psi_k)
            C = saturation_matrix_assembly(d, g, order, K, theta, K_prime, theta_prime, psi_k)
            #g_vect_prime = gravity_vector_prime(element, d, quad_deg-2, g,K,theta,K_prime,theta_prime,psi_k)
            Jsat= Jacobian_saturation(g,element,d,K,theta,K_prime,theta_prime,psi_k)
            Jperm = Jacobian_matrices(g,element,d,K,theta,K_prime,theta_prime,psi_k)
            
            #Jsat=C
            
            lhs = Jsat+dt*(A)+dt*Jperm
            
            rhs = Jsat@psi_k + B@theta_t - B@theta(psi_k) + dt*f_vect+dt*Jperm@psi_k
            
          
            
          
            lhs,rhs = dirichlet_BC(0,b_nodes,lhs,rhs,g)
            psi = np.linalg.solve(lhs,rhs)
            print(np.linalg.norm(psi-psi_k))
            store[count-1]= np.linalg.norm(psi-psi_k)
            corder[count-1]= np.linalg.norm(u_exact-psi)
            if np.linalg.norm(psi-psi_k)<=TOL+TOL*np.linalg.norm(psi_k):
                break
            else:
                psi_k = psi
        # err = mesh.compute_error(u,lambda x,y : u_exact(x,y,t))
        # print('error at time ',t," : ",err[0])
        print('Newton-scheme iterations: ',count)
        ctot =ctot+count
        psi_t = psi
        psi_k = psi

psi=psi.squeeze()
# Extract geometric information
coordinates = g.nodes[0:2]
xcoords,ycoords = coordinates
elements = g.cell_nodes()

# local to global map
cn = d.mapping.T


flat_list = d.local_dofs_corners

u_h =psi
