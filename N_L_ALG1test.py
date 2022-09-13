# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 18:58:42 2022

@author: jakob
"""


import numpy as np
import porepy as pp
import matplotlib.pyplot as plt
from RichardsEqFEM.source.basisfunctions.lagrange_element import finite_element
from RichardsEqFEM.source.LocalGlobalMapping.map_local_to_global import Local_to_Global_table

from RichardsEqFEM.source.MatrixAssembly.classLscheme import Newton_scheme, L_scheme
from RichardsEqFEM.source.utils.boundary_conditions import dirichlet_BC, dirichlet_BC_func
from RichardsEqFEM.source.utils.create_source_term import construct_source_function_withGrav
import sympy as sp

# def u_fabricated(t,x,y):
#     u = -t*(1-x)*(1-y)*x*y -1
    
    
#     return u

# def K(thetaa):
#     val = (thetaa**(-1))*thetaa**(-1)-2*thetaa**(-1)+1
#     #val =1+np.power(thetaa,4)
#     #val=1+thetaa**2
#     return val

# def theta(u):
#     val = 1/(1-u)
#     #val=u
#     #val = 0.125*u+(1.33-0.125)*np.power(u,3)
#     #val = u+np.power(u,3)
#     #val=u/u
#     return val

# bcval=-1


x_part=20
y_part=20
phys_dim = [1,1]
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



# x = sp.symbols('x')
# theta_prime = sp.diff(theta(x),x)
# # Derivative of K wrt theta
# K_prime = sp.diff(K(x),x)

# f = construct_source_function_withGrav(u_fabricated,theta,K)

# points1 = g.nodes[0:2]
# # vectorize u_h
# u_h = np.zeros((d.total_geometric_pts,1))
# t=0
# for k in range(d.total_geometric_pts):
#     u_h[k]=u_fabricated(t,points1[0][k],points1[1][k])
#     #u_h[k]= 1-coordinates[1][k]


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

points1 = g.nodes[0:2]
# vectorize u_h
u_h = np.zeros((d.total_geometric_pts,1))
t=0
for k in range(d.total_geometric_pts):
    #u_h[k]=u_fabricated(t,points1[0][k],points1[1][k])
    u_h[k]= -5
    
b_nodes = d.boundary
bcval=0     
u_h[b_nodes]=bcval
 
# psi_k = u_h.copy()
# psi_t = u_h.copy()



psi_k = u_h.copy()
psi_t = u_h.copy()







timesteps = 1

# dt =0.01
# L=1.3
# TOL = 10**(-10)
dt =0.1
L=3.501*10**(-2)
TOL = 10**(-5)


store = np.zeros((100,1))
count_tot =0

Newt = Newton_scheme(dt,d,g,order,psi_t,K,theta,K_prime,theta_prime,f)
Lmet = L_scheme(L,dt,d,g,order,psi_t,K,theta,f)

N_count = 0
L_count =0
for j in range(timesteps):
        count=0
        
        t=t+dt
        
        Newt.update_at_newtime(psi_t,t)
        Lmet.update_at_newtime(psi_t,t)
        #L-scheme iteration
        while True:
            
            if count ==0:
            
                Newt.update_at_iteration(psi_k)
                Newt.assemble(psi_k)
            
                lhs = Newt.lhs
                rhs = Newt.rhs
                

                lhs,rhs = dirichlet_BC(bcval,b_nodes,lhs,rhs,g)
                psiN = np.linalg.solve(lhs,rhs)
                print(np.linalg.norm(psiN-psi_k),'newton')
                r_N = np.linalg.norm(psiN-psi_k)
                
                Lmet.update_at_iteration(psi_k)
                Lmet.assemble(psi_k)
            
                lhs = Lmet.lhs
                rhs = Lmet.rhs
                
                lhs,rhs = dirichlet_BC(bcval,b_nodes,lhs,rhs,g)
                psiL = np.linalg.solve(lhs,rhs)
                print(np.linalg.norm(psiL-psi_k),'lscheme')
                r_L = np.linalg.norm(psiL-psi_k)
                print(r_N<r_L)
                
                r_L_old =r_L
                
                if r_N <r_L:
                    psi = psiN
                    N_count +=1
                    #print('dgdg')
                else:
                    psi = psiL
                    L_count +=1
                    #print('llll')
                
                
            else:
                if r_N< r_L:
                    Newt.update_at_iteration(psi_k)
                    Newt.assemble(psi_k)
         
                    lhs = Newt.lhs
                    rhs = Newt.rhs
            
                    lhs,rhs = dirichlet_BC(bcval,b_nodes,lhs,rhs,g)
                
                    N_count +=1
                    
                    psi = np.linalg.solve(lhs,rhs)
                    
                    r_N = np.linalg.norm(psi-psi_k)
                else:
                    Lmet.update_at_iteration(psi_k)
                    Lmet.assemble(psi_k)
            
                    lhs = Lmet.lhs
                    rhs = Lmet.rhs
                
                    lhs,rhs = dirichlet_BC(bcval,b_nodes,lhs,rhs,g)
                
                    L_count +=1
                    #print('l222')
                
                    psi = np.linalg.solve(lhs,rhs)
                    
                    r_L = np.linalg.norm(psi-psi_k)
                    
                    if r_L<1:
                        r_N=0
            count = count + 1
            # if count ==20:
            print(np.linalg.norm(psi-psi_k))
            store[count-1]= np.linalg.norm(psi-psi_k)
            if np.linalg.norm(psi-psi_k)<=TOL+TOL*np.linalg.norm(psi):
                break
            else:
                psi_k = psi.copy()
        # err = mesh.compute_error(u,lambda x,y : u_exact(x,y,t))
        # print('error at time ',t," : ",err[0])
        print('L-scheme iterations: ',count)
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
print('Newton iterations', N_count, 'L-scheme iterations', L_count)
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