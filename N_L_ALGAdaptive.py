# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:55:34 2022

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
from RichardsEqFEM.source.utils.create_source_term import construct_source_function_withGrav
import sympy as sp
import scipy as sci
#import torchvision.transforms as transforms
import functools

#Van Genuchten parameteres
a_g = 0.423
n_g = 2.06
k_abs = 4.96*10**(-2)
the_r = 0.131
the_s = 0.396
exp_1 = n_g/(n_g-1)
exp_2 = (n_g-1)/n_g

# def K(thetaa):
#     val = ((k_abs)*((thetaa-the_r)/(the_s-the_r))**(1/2))*(1-(1-((thetaa-the_r)/(the_s-the_r))**exp_1)**exp_2)**2

#     return val

# def theta(u):

#     val = sp.Piecewise((the_r+(the_s-the_r)*(1+(-a_g*u)**n_g)**(-exp_2),u<0),(the_s,u>=0))
#     return val
# #f = construct_source_function(u_fabricated,theta,K)
# def f(t,x,y):
#     return 0

def u_fabricated(t,x,y):
    u = -t*(1-x)*(1-y)*x*y -1
    
    
    return u

def K(thetaa):
    val = (thetaa**(-1))*thetaa**(-1)-2*thetaa**(-1)+1
    #val =1+np.power(thetaa,4)
    #val=1+thetaa**2
    return val

def theta(u):
    val = 1/(1-u)
    #val=u
    #val = 0.125*u+(1.33-0.125)*np.power(u,3)
    #val = u+np.power(u,3)
    #val=u/u
    return val
fla = construct_source_function_withGrav(u_fabricated,theta,K)
# def f(t,x,y):
#     return 0
#f = functools.partial(f)
def f(t,x,y):
    return fla(t,x,y)
if __name__ == '__main__':
    
    #f = transforms.Compose([transforms.Lambda(f)])
    x = sp.symbols('x')
    theta_prime = sp.diff(theta(x),x)

    #theta_Ltest = sp.lambdify([x],theta_prime)
    theta_prime_prime = sp.diff(theta(x),x,2)
    theta_Ltest = sp.lambdify([x],theta_prime_prime)


    # Derivative of K wrt theta
    K_prime = sp.diff(K(x),x)
    
    x_part=40
    y_part=40
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
    d = Local_to_Global_table(g,element,x_part,y_part)
    # test = global_assembly_mass(d, g,para=True)
    # bw= test.mass_matrix.todense()
    g_coarse = pp.StructuredTriangleGrid(np.array([int(x_part/2), int(x_part/2)]),phys_dim)
    g_coarse.compute_geometry()
    d_coarse = Local_to_Global_table(g_coarse,element,int(x_part/2),int(x_part/2))

    # Produces elements with nodes from original mesh
    d.coarse_fine_mesh_nodes(x_part, y_part)
    d.coarse_map()
    
    #x = sp.symbols('x')

    points1 = g.nodes[0:2]
    # vectorize u_h
    u_h = np.zeros((d.total_geometric_pts,1))
    t=0
    # for k in range(d.total_geometric_pts):
    #     #u_h[k]=u_fabricated(t,points1[0][k],points1[1][k])
    #     u_h[k]= 1-coordinates[1][k]
    for k in range(d.total_geometric_pts):
        u_h[k]=u_fabricated(t,points1[0][k],points1[1][k])
        #u_h[k]= -5
    psi_k = u_h.copy()
    psi_t = u_h.copy()


    bcval=-1


    b_nodes = d.boundary
    b_nodes_c = d_coarse.boundary


    timesteps = 1

    dt =1
    L=1.3#3.501*10**(-2)
    TOL = 10**(-5)
    

    store = np.zeros((100,1))
    count_tot =0
    
    start = timer()
    scheme = Newton_scheme_Parallell(dt, d, g, order, psi_t, K, theta, K_prime, theta_prime, f)
    #scheme = L_scheme_Parallell(L,dt,d,g,order,psi_t,K,theta,f)
    #B = scheme.mass_matrix.todense()
    elapsed_time = timer() - start
    print(elapsed_time)
    Newt = Newton_scheme_Parallell(dt,d,g,order,psi_t,K,theta,K_prime,theta_prime,f)
    Lmet = L_scheme_Parallell(L,dt,d,g,order,psi_t,K,theta,f)
    Newt_co = Newton_scheme_Parallell(dt,d_coarse,g_coarse,order,psi_t[d.nodes_coarse_ordered],K,theta,K_prime,theta_prime,f)
    Lmet_co = L_scheme_Parallell(L,dt,d_coarse,g_coarse,order,psi_t[d.nodes_coarse_ordered],K,theta,f)


        
    start = timer()
    
    for j in range(timesteps):
        count=0
        L_count=0
        N_count=0
        t=t+dt
        
        Newt.update_at_newtime(psi_t,t)
        Lmet.update_at_newtime(psi_t,t)
        Newt_co.update_at_newtime(psi_t[d.nodes_coarse_ordered],t)
        Lmet_co.update_at_newtime(psi_t[d.nodes_coarse_ordered],t)
        
        #L-scheme iteration
        while True:
            
   
            if count ==0:
            
                Newt_co.update_at_iteration(psi_k[d.nodes_coarse_ordered])
                Newt_co.assemble(psi_k[d.nodes_coarse_ordered])
            
                lhs = Newt_co.lhs
                rhs = Newt_co.rhs
                

                lhs,rhs = dirichlet_BC(bcval,b_nodes_c,lhs,rhs,g)
                psiN_co = sci.sparse.linalg.spsolve(lhs,rhs)
                # #psi = np.linalg.solve(lhs,rhs)
                psiN_co = np.resize(psiN_co,(psiN_co.shape[0],1))
                print(np.linalg.norm(psiN_co-psi_k[d.nodes_coarse_ordered]),'newton')
                r_N_co = np.linalg.norm(psiN_co-psi_k[d.nodes_coarse_ordered])
                
                Lmet_co.update_at_iteration(psi_k[d.nodes_coarse_ordered])
                Lmet_co.assemble(psi_k[d.nodes_coarse_ordered])
            
                lhs = Lmet_co.lhs
                rhs = Lmet_co.rhs
                
                lhs,rhs = dirichlet_BC(bcval,b_nodes_c,lhs,rhs,g)
                psiL_co = sci.sparse.linalg.spsolve(lhs,rhs)
                # #psi = np.linalg.solve(lhs,rhs)
                psiL_co = np.resize(psiL_co,(psiL_co.shape[0],1))
                print(np.linalg.norm(psiL_co-psi_k[d.nodes_coarse_ordered]),'lscheme')
                r_L_co = np.linalg.norm(psiL_co-psi_k[d.nodes_coarse_ordered])
                print(r_N_co<r_L_co)
                print(np.linalg.norm(psiL_co-psiN_co),'runge estimate')
                
                r_L_co_old =r_L_co
                
                if r_N_co <r_L_co+0.1:
                    Newt.update_at_iteration(psi_k)
                    Newt.assemble(psi_k)
         
                    lhs = Newt.lhs
                    rhs = Newt.rhs
            
                    lhs,rhs = dirichlet_BC(bcval,b_nodes,lhs,rhs,g)
                
                    #N_count +=1
                    
                    psi = sci.sparse.linalg.spsolve(lhs,rhs)
                    # #psi = np.linalg.solve(lhs,rhs)
                    psi = np.resize(psi,(psi.shape[0],1))
                    #psi = psiN_co
                    N_count +=1
                    #print('dgdg')
                    Switch = True
                else:
                    Lmet.update_at_iteration(psi_k)
                    Lmet.assemble(psi_k)
            
                    lhs = Lmet.lhs
                    rhs = Lmet.rhs
                
                    lhs,rhs = dirichlet_BC(bcval,b_nodes,lhs,rhs,g)
                
                    L_count +=1
                    #print('l222')
                
                    psi = sci.sparse.linalg.spsolve(lhs,rhs)
                    # #psi = np.linalg.solve(lhs,rhs)
                    psi = np.resize(psi,(psi.shape[0],1))
                    
                    r_L = np.linalg.norm(psi-psi_k)
                    print(r_L)
                    L_count +=1
                    print('llll')
                    Switch = False
                    
            else:
                if Switch == True:
                    Newt.update_at_iteration(psi_k)
                    Newt.assemble(psi_k)
         
                    lhs = Newt.lhs
                    rhs = Newt.rhs
            
                    lhs,rhs = dirichlet_BC(bcval,b_nodes,lhs,rhs,g)
                
                    #N_count +=1
                    
                    psi = sci.sparse.linalg.spsolve(lhs,rhs)
                    # #psi = np.linalg.solve(lhs,rhs)
                    psi = np.resize(psi,(psi.shape[0],1))
                    #psi = psiN_co
                    N_count +=1
                    #print('dgdg')
                    Switch = True
                    
                else:
                    Lmet.update_at_iteration(psi_k)
                    Lmet.assemble(psi_k)
            
                    lhs = Lmet.lhs
                    rhs = Lmet.rhs
                
                    lhs,rhs = dirichlet_BC(bcval,b_nodes,lhs,rhs,g)
                
                    L_count +=1
                    #print('l222')
                
                    psi = sci.sparse.linalg.spsolve(lhs,rhs)
                    # #psi = np.linalg.solve(lhs,rhs)
                    psi = np.resize(psi,(psi.shape[0],1))
                    
                    r_L = np.linalg.norm(psi-psi_k)
                    L_count +=1
                
           
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
        print('Newton iterations', N_count, 'L-scheme iterations', L_count)
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