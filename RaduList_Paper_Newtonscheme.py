# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 17:34:08 2022

@author: jakob
"""

import numpy as np
import porepy as pp
import matplotlib.pyplot as plt
from RichardsEqFEM.source.basisfunctions.lagrange_element import finite_element
from RichardsEqFEM.source.LocalGlobalMapping.map_local_to_global import Local_to_Global_table
from RichardsEqFEM.source.MatrixAssembly.mass_matrix import mass_matrix, mass_matrix_alternative
from RichardsEqFEM.source.MatrixAssembly.source_term import Source_term_assembly
from RichardsEqFEM.source.MatrixAssembly.Richards_matrices import saturation_matrix_assembly,permability_matrix_assembly
from RichardsEqFEM.source.utils.boundary_conditions import dirichlet_BC, dirichlet_BC_func
from RichardsEqFEM.source.utils.create_source_term import construct_source_function
from RichardsEqFEM.source.MatrixAssembly.Richards_Jacobian import Jacobian_saturation, Jacobian_matrices, Newton_Assembly_alt
import sympy as sp
from RichardsEqFEM.source.MatrixAssembly.gravitation import gravity_vector, gravity_vector_prime


x_part=10
y_part=15
phys_dim = [2,3]
g = pp.StructuredTriangleGrid(np.array([x_part, y_part]),phys_dim)

g.compute_geometry()
coordinates = g.nodes[0:2]
#pp.plot_grid(g,info='cf', figsize=(15,12),alpha=0)

order = 1 # Order of polynomial basis
t=0
quad_deg =order+3 # Gaussian quadrature degree

# define lagrange element and local to global map
element = finite_element(order)
d = Local_to_Global_table(g,element)

#Van Genuchten parameteres
a_g = 0.423
n_g = 2.06
k_abs = 4.96*10**(-2)
the_r = 0.131
the_s = 0.396
exp_1 = n_g/(n_g-1)
exp_2 = (n_g-1)/n_g

def K(thetaa):
    val = ((k_abs)*((thetaa-the_r)/(the_s-the_r))**(1/2))*(1-(1-((thetaa-the_r)/(the_s-the_r))**exp_1)**exp_2)**2
   
    return val

def theta(u):
    
    val = sp.Piecewise((the_r+(the_s-the_r)*(1+(-a_g*u)**n_g)**(-exp_2),u<0),(the_s,u>=0))
    return val
#f = construct_source_function(u_fabricated,theta,K)
def f(t,x,y):
    return 0
x = sp.symbols('x')
theta_prime = sp.diff(theta(x),x)
# Derivative of K wrt theta
K_prime = sp.diff(K(x),x)



                   

def K(thetaa):
    # if thetaa ==the_s:
    #     val = k_abs
    # else:
    val = ((k_abs)*((thetaa-the_r)/(the_s-the_r))**(1/2))*(1-(1-((thetaa-the_r)/(the_s-the_r))**exp_1)**exp_2)**2
    return val
def thetaglob(u):
    
    #val = sp.Piecewise((the_r+(the_s-the_r)*(1+(-a_g*u)**n_g)**(-exp_2),u<=0),(the_s,u>0))
    #val = sp.lambdify([p],theta)
    val = np.zeros((len(u),1))
    if u.shape[0]>=2:
        u = np.resize(u, (u.shape[0],))
    for i in range(len(u)):
        if u[i]<0:
            
            val[i] = the_r+(the_s-the_r)*(1+(-a_g*u[i])**n_g)**(-exp_2)
        else:
            val[i] = the_s
    return val
points1 = g.nodes[0:2]
# vectorize u_h
u_h = np.zeros((d.total_geometric_pts,1))
t=0
for k in range(d.total_geometric_pts):
    #u_h[k]=u_fabricated(t,points1[0][k],points1[1][k])
    u_h[k]= 1-coordinates[1][k]
    
psi_k = u_h.copy()
psi_t = u_h.copy()
B = mass_matrix(element,d,order+1, g)
#B = matrix_assembly_const(g, order)
#A,B = nonlinear_matrix_assembly(g, element, d, K, theta,psi)



b_nodes = d.boundary
sec_b_nodes = np.zeros((int(len(b_nodes)/8)),dtype=int)
first_b_nodes = np.zeros((int(len(b_nodes)/8)),dtype=int)
j=0
n=0
for i in range(len(b_nodes)):
    
    if coordinates[0][b_nodes[i]]==2 and 1>=coordinates[1][b_nodes[i]]>=0:
        
        sec_b_nodes[j] = b_nodes[i]
        j=j+1
        
    if coordinates[1][b_nodes[i]]==3 and 1>=coordinates[0][b_nodes[i]]>=0:
        first_b_nodes[n] = b_nodes[i]
        n=n+1

first_b_nodes = np.flip(first_b_nodes)


timesteps = 9

dt =1/48
L=3.501*10**(-2)
TOL = 10**(-5)

def boundary_one(x,y,t):
    return -2+2.2*t/(1/16)

def boundary_one_alt(x,y):
    return 0.2

def boundary_two(x,y):
    return 1-y

store = np.zeros((100,1))
count_tot =0
for j in range(timesteps):
        count=0
        
        t=t+dt
        f_vect =Source_term_assembly(f,element,d,quad_deg, g, t, time=True)
        
        theta_t = thetaglob(psi_t)
        theta_t =saturation_matrix_assembly(d, g, order, K, theta, K_prime, theta_prime, psi_t)
        #Newton-scheme iteration
        while True:
            count = count + 1
            #W,Q =gravitation_matrix(g,element,d,K,theta,K_prime,theta_prime,psi_k)
            A = permability_matrix_assembly(d,g,order,K,theta,K_prime,theta_prime,psi_k)
            C =saturation_matrix_assembly(d, g, order, K, theta, K_prime, theta_prime, psi_k)
            #C= Newton_Assembly_alt(g,element,d,K,theta,K_prime,theta_prime,psi_k)
            g_vect = gravity_vector(element, d, 2, g,K,theta,K_prime,theta_prime,psi_k)
            Jsat= Jacobian_saturation(g,element,d,K,theta,K_prime,theta_prime,psi_k)
            Jperm = Jacobian_matrices(g,element,d,K,theta,K_prime,theta_prime,psi_k)
            g_M_prime = gravity_vector_prime(element, d, quad_deg-2, g,K,theta,K_prime,theta_prime,psi_k)
            
            #Jperm =C
            lhs = Jsat+dt*(A)+dt*Jperm+dt*g_M_prime
            rhs = Jsat@psi_k + theta_t - C + dt*f_vect-dt*g_vect+dt*Jperm@psi_k+dt*g_M_prime@psi_k
            
            
            if t <= 1/16:
                lhs,rhs = dirichlet_BC_func(boundary_one,first_b_nodes,lhs,rhs,coordinates,t,time=True)
            else:
                lhs,rhs = dirichlet_BC_func(boundary_one_alt,first_b_nodes,lhs,rhs,coordinates,t,time=False)
            lhs,rhs = dirichlet_BC_func(boundary_two,sec_b_nodes,lhs,rhs,coordinates,t,time=False)  
            psi = np.linalg.solve(lhs,rhs)
            # if count ==20:
            print(np.linalg.norm(psi-psi_k))
            store[count-1]= np.linalg.norm(psi-psi_k)
            if np.linalg.norm(psi-psi_k)<=TOL+TOL*np.linalg.norm(psi):
                break
            else:
                psi_k = psi.copy()
        # err = mesh.compute_error(u,lambda x,y : u_exact(x,y,t))
        # print('error at time ',t," : ",err[0])
        print('Newton-scheme iterations: ',count)
        count_tot = count_tot +count
        psi_t = psi.copy()
        psi_k = psi.copy()
        psi=psi.squeeze()
        # Extract geometric information
        coordinates = g.nodes
        xcoords,ycoords = coordinates[0:2]
        elements = g.cell_nodes()

        # local to global map
        cn = d.mapping.T


        flat_list = d.local_dofs_corners

        #u_h =psi
        
        psi_plot = psi.copy()#np.resize(psi,(psi.shape[1],))
        plt.tricontourf(xcoords, ycoords, cn[:,flat_list], psi_plot,40)
        plt.colorbar()
        plt.show()
print('Total number of iterations', count_tot)
psi=psi.squeeze()
# Extract geometric information
coordinates = g.nodes
xcoords,ycoords = coordinates[0:2]
elements = g.cell_nodes()

# local to global map
cn = d.mapping.T


flat_list = d.local_dofs_corners


psi_plot = psi.copy()#np.resize(psi,(psi.shape[1],))
plt.tricontourf(xcoords, ycoords, cn[:,flat_list], psi_plot,40)
plt.colorbar()
plt.show()