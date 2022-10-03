# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 13:08:02 2022

@author: jakob
"""

from timeit import default_timer as timer
import numpy as np
import porepy as pp
import matplotlib.pyplot as plt
from RichardsEqFEM.source.basisfunctions.lagrange_element import finite_element
from RichardsEqFEM.source.LocalGlobalMapping.map_P1 import Local_to_Global_table

from RichardsEqFEM.source.MatrixAssembly.classLscheme import L_scheme, ModL_scheme
from RichardsEqFEM.source.MatrixAssembly.Model_class_fast import L_scheme_fast
from RichardsEqFEM.source.MatrixAssembly.Model_class_parallell import L_scheme_Parallell, Newton_scheme_Parallell
from RichardsEqFEM.source.utils.boundary_conditions import dirichlet_BC, dirichlet_BC_func
#from RichardsEqFEM.source.MatrixAssembly.local_mass_assembly import local_mass_pool, global_assembly_mass
from multiprocessing import Pool
from scipy.sparse.coo import coo_matrix
#from RichardsEqFEM.source.MatrixAssembly.
import sympy as sp
import scipy as sci

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



if __name__ == '__main__':
    
    x = sp.symbols('x')
    theta_prime = sp.diff(theta(x),x)

    #theta_Ltest = sp.lambdify([x],theta_prime)
    theta_prime_prime = sp.diff(theta(x),x,2)
    theta_Ltest = sp.lambdify([x],theta_prime_prime)


    # Derivative of K wrt theta
    K_prime = sp.diff(K(x),x)
    
    x_part=80
    y_part=120
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
    d = Local_to_Global_table(g,element,x_part,y_part)
    # test = global_assembly_mass(d, g,para=True)
    # bw= test.mass_matrix.todense()

    
    #x = sp.symbols('x')

    points1 = g.nodes[0:2]
    # vectorize u_h
    u_h = np.zeros((d.total_geometric_pts,1))
    t=0
    for k in range(d.total_geometric_pts):
        #u_h[k]=u_fabricated(t,points1[0][k],points1[1][k])
        u_h[k]= 1-coordinates[1][k]
    
    psi_k = u_h.copy()
    psi_t = u_h.copy()





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
    
    start = timer()
    scheme = Newton_scheme_Parallell(dt, d, g, order, psi_t, K, theta, K_prime, theta_prime, f)
    #scheme = L_scheme_Parallell(L,dt,d,g,order,psi_t,K,theta,f)
    #B = scheme.mass_matrix.todense()
    elapsed_time = timer() - start
    print(elapsed_time)



        
    start = timer()
    # if __name__ == "__main__":
        #     test2 = global_assembly_mass(d, g)
        #     bt= test2.todense()
        # elapsed_time = timer() - start
        # print(elapsed_time)
    for j in range(timesteps):
        count=0
        
        t=t+dt
        
        scheme.update_at_newtime(psi_t,t)
        
        #L-scheme iteration
        while True:
            
   
            
            scheme.update_at_iteration(psi_k)
            scheme.assemble(psi_k)
            
            lhs = scheme.lhs.tocsr()
            rhs = scheme.rhs
            #print(lhs)
            #print(rhs)
            if t <= 1/16:
                lhs,rhs = dirichlet_BC_func(boundary_one,first_b_nodes,lhs,rhs,coordinates,t,time=True)
            else:
                lhs,rhs = dirichlet_BC_func(boundary_one_alt,first_b_nodes,lhs,rhs,coordinates,t,time=False)
            lhs,rhs = dirichlet_BC_func(boundary_two,sec_b_nodes,lhs,rhs,coordinates,t,time=False)  
            
            #linsol = timer()
            psi = sci.sparse.linalg.spsolve(lhs,rhs)
            #print(timer()-linsol, 'sol')
            #psi = np.linalg.solve(lhs,rhs)
            psi = np.resize(psi,(psi.shape[0],1))
            #print(psi.shape,'psi')
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
    elapsed_time = timer() - start
    print(elapsed_time)
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