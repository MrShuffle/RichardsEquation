# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:53:17 2022

@author: jakob
"""

from timeit import default_timer as timer
import numpy as np
import porepy as pp
import matplotlib.pyplot as plt
from RichardsEqFEM.source.basisfunctions.lagrange_element import finite_element
from RichardsEqFEM.source.LocalGlobalMapping.map_P1 import Local_to_Global_table

from RichardsEqFEM.source.MatrixAssembly.classLscheme import L_scheme, ModL_scheme
from RichardsEqFEM.source.MatrixAssembly.Model_class_fast import L_scheme_fast, Newton_method_fast
from RichardsEqFEM.source.MatrixAssembly.Model_class_parallell import LN_alg
#from RichardsEqFEM.source.MatrixAssembly
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

from RichardsEqFEM.source.utils.Optimal_L import compute_L

# #Van Genuchten parameteres
# a_g = 0.423
# n_g = 2.06
# k_abs = 4.96*10**(-2)
# the_r = 0.131
# the_s = 0.396
# exp_1 = n_g/(n_g-1)
# exp_2 = (n_g-1)/n_g

a_g = 0.551
n_g = 2.9
k_abs = 0.12
the_r = 0.026
the_s = 0.42
exp_1 = n_g/(n_g-1)
exp_2 = (n_g-1)/n_g

# a_g = 0.95
# n_g = 2.9
# k_abs = 0.12
# the_r = 0.026
# the_s = 0.42
# exp_1 = n_g/(n_g-1)
# exp_2 = (n_g-1)/n_g

# a_g = 0.95
# n_g = 1.9
# k_abs = 0.12
# the_r = 0.026
# the_s = 0.42
# exp_1 = n_g/(n_g-1)
# exp_2 = (n_g-1)/n_g

def theta_sp(u):

    val = sp.Piecewise((the_r+(the_s-the_r)*(sp.functions.elementary.complexes.Abs(1+(sp.functions.elementary.complexes.Abs(-a_g*u))**n_g))**(-exp_2),u<0),(the_s,u>=0))
    return val

def K_sp(thetaa):
    val = ((k_abs)*((thetaa-the_r)/(the_s-the_r))**(1/2))*(sp.functions.elementary.complexes.Abs(1-(sp.functions.elementary.complexes.Abs(1-(sp.functions.elementary.complexes.Abs((thetaa-the_r)/(the_s-the_r)))**exp_1))**exp_2))**2

    return val

def K(thetaa):
    val = ((k_abs)*((thetaa-the_r)/(the_s-the_r))**(1/2))*(1-(1-((thetaa-the_r)/(the_s-the_r))**exp_1)**exp_2)**2

    return val

def theta(u):

    val = sp.Piecewise((the_r+(the_s-the_r)*(1+(np.abs(-a_g*u))**n_g)**(-exp_2),u<0),(the_s,u>=0))
    return val
# #f = construct_source_function(u_fabricated,theta,K)
# def f(t,x,y):
#     return 0

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
# fla = construct_source_function_withGrav(u_fabricated,theta,K)
# # def f(t,x,y):
# #     return 0
def f(t,x,y):
    if y > 1/4:
        val= 0.06*np.cos(4/3*np.pi*(y))*np.sin(((x)))
    else:
        val=0
    return val
# #f = functools.partial(f)
# def f(t,x,y):
#     return fla(t,x,y)


if __name__ == '__main__':
    
    
    
    #f = transforms.Compose([transforms.Lambda(f)])
    x = sp.symbols('x',real=True)
    theta_prime = sp.diff(theta(x),x)

    #theta_Ltest = sp.lambdify([x],theta_prime)
    theta_prime_prime = sp.diff(theta_sp(x),x,2)
    theta_Ltest = sp.lambdify([x],theta_prime)
    testarray = np.linspace(-100,4,10000)
    F = theta_Ltest(testarray)
    L_theta = np.amax(F)
    theta_min = np.amin(F)
 
    
    

    print(L_theta,theta_min)
    # theta_Ltest2 = sp.lambdify([x],theta_prime_prime)
    # #testarray = np.linspace(-100,3,10000)
    # F = theta_Ltest2(testarray)
    # L2 = np.amax(F)
    # M2 = np.amin(F)
    # print(L2,M2)

    # Derivative of K wrt theta
    K_prime = sp.diff(K_sp(x),x)
    K_Ltest = sp.lambdify([x],K_prime)
    testarray = np.linspace(k_abs,the_s-the_r,100)
    gv = K_Ltest(testarray)
    L_K = np.amax(gv)
    print(L_K)
    

    # i = sp.symbols('i')
    # invK = sp.solve(K(theta(x))-i,x)
    
    x_part=80
    y_part=80
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
    # g_coarse = pp.StructuredTriangleGrid(np.array([int(x_part/2), int(x_part/2)]),phys_dim)
    # g_coarse.compute_geometry()
    # d_coarse = Local_to_Global_table(g_coarse,element,int(x_part/2),int(x_part/2))
    # coordinates_co = g_coarse.nodes[0:2]
    # # Produces elements with nodes from original mesh
    # d.coarse_fine_mesh_nodes(x_part, y_part)
    # d.coarse_map()
    
    #x = sp.symbols('x')

    points1 = g.nodes[0:2]
    # vectorize u_h
    u_h = np.zeros((d.total_geometric_pts,1))
    t=0
    # for k in range(d.total_geometric_pts):
    #     #u_h[k]=u_fabricated(t,points1[0][k],points1[1][k])
    #     u_h[k]= 1-coordinates[1][k]
    # for k in range(d.total_geometric_pts):
    #     u_h[k]=u_fabricated(t,points1[0][k],points1[1][k])

    for k in range(d.total_geometric_pts):
        #u_h[k]=u_fabricated(t,points1[0][k],points1[1][k])
        if coordinates[1][k]>1/4:
            u_h[k]=-4
        else:
            u_h[k]=-coordinates[1][k]-1/4
  
        #u_h[k]= -5
    psi_k     = u_h.copy()
    psi_t     = u_h.copy()
    psi_L_old = u_h.copy()

    


    b_nodes = d.boundary
    
    
    first_b_nodes = np.zeros((int(len(b_nodes)/4+1)),dtype=int)
 
    #j=0
    n=0
    for i in range(len(b_nodes)):
        
        # if coordinates[0][b_nodes[i]]==2 and 1>=coordinates[1][b_nodes[i]]>=0:
            
        #     sec_b_nodes[j] = b_nodes[i]
        #     j=j+1
            
        if coordinates[1][b_nodes[i]]==1 and 1>=coordinates[0][b_nodes[i]]>=0:
            first_b_nodes[n] = b_nodes[i]
            n=n+1


    first_b_nodes = np.flip(first_b_nodes)


    timesteps = 1

    dt =0.01
    #L=3.501*10**(-2)
    L=0.1
    #L=0.15
    # eta = 1
    # Lmet = L_scheme_Parallell(L,dt,d,g,order,psi_t,K,theta,f)
    # eta = Lmet.estimate_eta(psi_t)
    # L = compute_L(L_theta, k_abs, theta_min, L_K, eta, dt)
    # print(eta)
    # print(L,'L')
    TOL = 10**(-7)
    
    testL = np.zeros((100,1))
    store = np.zeros((100,1))
    count_tot =0
    L_count_tot = 0
    N_count_tot =0
    

    scheme = LN_alg(L,dt,d,g,order,psi_t,K,theta,K_prime,theta_prime,f)


    Switch=False   
    print('starting')
    start = timer()
    bcval=-4
    for j in range(timesteps):
        count=0
        L_count=0
        N_count=0
        ind =1
        t=t+dt
        
        scheme.update_at_newtime(psi_t,t)



        #L-scheme iteration
        while True:
            
   
           timeass=timer()
           scheme.update_at_iteration(psi_k,ind,Switch)
           scheme.assemble(psi_k,Switch)
     
           lhs = scheme.lhs
           rhs = scheme.rhs
           print(timeass-timer(),'time assembly')
           lhs,rhs = dirichlet_BC(bcval,first_b_nodes,lhs,rhs,g)
            
           psi = sci.sparse.linalg.spsolve(lhs,rhs)

           psi = np.resize(psi,(psi.shape[0],1))
           
           if Switch ==True:
               N_count+=1
               testyy = timer()
               scheme.N_to_L_eta(psi, psi_k, K_prime, theta_prime)
               print(testyy-timer(),'time Newt')
               print(scheme.eta_NtoL,'eta_NtoL')
               if scheme.eta_NtoL>1:
                   Switch = False
                   ind=1
                   psi=psi_L_old
                   # scheme.assemble(psi_k,Switch)
             
                   # lhs = scheme.lhs
                   # rhs = scheme.rhs
                
                   # lhs,rhs = dirichlet_BC(bcval,first_b_nodes,lhs,rhs,g)
                    
                   # psi = sci.sparse.linalg.spsolve(lhs,rhs)

                   # psi = np.resize(psi,(psi.shape[0],1))
                   # L_count+=1
               else:  
                   Switch = True
                   valstop = scheme.linear_norm
                   ind=0
           else:
               L_count+=1
               testyy = timer()
               scheme.L_to_N_eta(psi, psi_k, K_prime, theta_prime)
               print(testyy-timer(),'time')
               print(scheme.eta_LtoN,'eta L to N')
               print(scheme.eta_LtoL,'eta L to L')
               # if L_count >=2:
               #     Rconv = np.abs(np.log(testL[count])/np.log(testL[count-1]))**(1/(count))
               psi_L_old = psi
               valstop= scheme.linear_norm
               #if a<=1:
                   
               if scheme.eta_LtoL>=1:
                   scheme.update_L(L_theta)
                   print(scheme.L,'L')
               if scheme.eta_LtoN<1.5:
                   Switch= True
               else:
                   Switch = False

           count = count + 1
           print(valstop)
           store[count-1]= np.linalg.norm(psi-psi_k)
           if valstop<=TOL:#+TOL*np.linalg.norm(psi):
               break
           else:
               psi_k = psi.copy()
        # err = mesh.compute_error(u,lambda x,y : u_exact(x,y,t))
        # print('error at time ',t," : ",err[0])
        print('L-scheme iterations: ',count)
        print('Newton iterations', N_count, 'L-scheme iterations', L_count)
        count_tot = count_tot +count
        L_count_tot = L_count_tot +L_count
        N_count_tot = N_count_tot +N_count
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
    print('Total','Newton iterations', N_count_tot, 'L-scheme iterations', L_count_tot)
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