# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:19:02 2022

@author: jakob
"""

import math

import numpy as np
import porepy as pp

from RichardsEqFEM.source.basisfunctions.Gauss_quadrature_points import *
from RichardsEqFEM.source.basisfunctions.lagrange_element import (
    finite_element, global_element_geometry)


# Local -> Global node numbering lookup table
class Local_to_Global_table:
    def __init__(self, geometry, element, x_part, y_part):
        # Geometry is the geometry from porepy generated by pp.StructuredTriangle
        self.geometry = geometry
        self.dim = geometry.dim
        self.element = element
        self.degree = self.element.degree
        self.x_part = x_part
        self.y_part = y_part

        self.local_dofs = self.element.local_dofs

        # Nodes per element
        self.num_nodes_per_EL = 3

        # Initialize list for the mapping.
        self.mapping = np.zeros(
            (self.num_nodes_per_EL, self.geometry.num_cells), dtype=int
        )

        # Create list with global values;
        #         0 : The number of corner nodes
        #         1 : The number of faces
        #         2 : The number of cells
        self.global_vals = np.array(
            [geometry.num_nodes, geometry.num_faces, geometry.num_cells]
        )

        # Create list with nodes per geometric quantity
        #         0 : Number of proper nodes
        #         1 : Number of nodes per face
        #         2 : Number of nodes within the element
        self.dofs_per_quantity = np.zeros(self.dim + 1, dtype=int)  # initialize

        ########################
        tmp_ind = np.arange(0, self.x_part)
        ind_1 = tmp_ind  # Lower left node in quad
        ind_2 = tmp_ind + 1  # Lower right node
        ind_3 = self.x_part + 2 + tmp_ind  # Upper left node
        ind_4 = self.x_part + 1 + tmp_ind  # Upper right node

        # The first triangle is defined by (i1, i4, i2), the next by
        # (i1, i3, i4). Stack these vertically, and reshape so that the
        # first quad is split into cells 0 and 1 and so on
        tri_base = np.vstack((ind_1, ind_3, ind_2, ind_1, ind_4, ind_3)).reshape(
            (3, -1), order="F"
        )
        # Initialize array of triangles. For the moment, we will append the
        # cells here, but we do know how many cells there are in advance,
        # so pre-allocation is possible if this turns out to be a bottleneck
        tri = tri_base

        # Loop over all remaining rows in the y-direction.
        for iter1 in range(self.y_part - 1):
            # The node numbers are increased by nx[0] + 1 for each row
            tri = np.hstack((tri, tri_base + (iter1 + 1) * (self.x_part + 1)))

        self.mapping = tri
        #######################

        # # Total number of nodes for any lagrange finite element

        # Geometric keyword dictionary
        self.geometry_keywords = {0: {"nodes"}, 1: {"faces"}, 2: {"elements"}}

        for i in self.geometry_keywords:
            for j in self.geometry_keywords[i]:
                for k in self.local_dofs[j]:

                    self.dofs_per_quantity[i] = len(self.local_dofs[j][k])

        # Total number of geometric points for any given lagrange finite element.
        self.total_geometric_pts = int(np.dot(self.dofs_per_quantity, self.global_vals))

        self.points_glob = self.geometry.nodes

        self.numDataPts = self.global_vals[2] * self.element.num_dofs**2

        self.row = np.empty(self.numDataPts, dtype=int)
        self.col = np.empty(self.numDataPts, dtype=int)

        n = 0
        # Generate row and col entries for coo_matrix
        for e in range(self.geometry.num_cells):

            cn = self.mapping[:, e]
            for k, l in np.ndindex(self.element.num_dofs, self.element.num_dofs):
                self.row[n] = cn[k]
                self.col[n] = cn[l]
                n += 1

            # self.points_glob[:,self.mapping[:,e] ]

        oo = list(self.local_dofs["nodes"].values())
        flat_list = [item for sublist in oo for item in sublist]

        flat_list[1], flat_list[2] = flat_list[2], flat_list[1]

        self.local_dofs_corners = flat_list

        # Fetch boundary nodes
        self.boundary = geometry.tags["domain_boundary_nodes"].nonzero()[0]

        # Storing quadrature points an weights
        a = gauss_quadrature_points(self.degree + 1)
        self.quad_pts = a[:, 0:2]
        self.quad_weights = 1 / 2 * a[:, 2]

    def L_scheme(self, K, theta, f):
        self.K = K
        self.theta = theta
        self.f = f

    def Newton_method(self, K, theta, K_prime, theta_prime, f):
        self.K = K
        self.theta = theta
        self.K_prime = K_prime
        self.theta_prime = theta_prime
        self.f = f

    def coarse_fine_mesh_nodes(self, x_part, y_part):

        self.x_part = x_part
        self.y_part = y_part
        vab = np.arange(x_part + 1)[0 : int(x_part / 2 + 1)] * 2
        self.dy = self.mapping[1, 1] * 2

        a = np.zeros((int(x_part / 2 + 1), int(y_part / 2 + 1)), dtype=int)
        for j in range(int(x_part / 2 + 1)):
            a[j] = vab + self.dy * j

        self.nodes_coarse = a
        self.nodes_coarse_ordered = a.reshape(
            -1,
        )

    def coarse_map(self):
        x_row1 = self.nodes_coarse[0]
        x_row2 = self.nodes_coarse[1]

        ind_1 = self.nodes_coarse[0][0 : int(self.x_part / 2)]
        ind_2 = self.nodes_coarse[0][1 : int(self.x_part / 2 + 1)]
        ind_3 = self.dy + 2 + ind_1
        ind_4 = self.dy + ind_1

        tri_base = np.vstack((ind_1, ind_3, ind_2, ind_1, ind_4, ind_3)).reshape(
            (3, -1), order="F"
        )

        tri = tri_base

        # Loop over all remaining rows in the y-direction.
        for iter1 in range(int(self.y_part / 2 - 1)):

            # The node numbers are increased by nx[0] + 1 for each row
            tri = np.hstack((tri, tri_base + (iter1 + 1) * (self.dy)))

        self.mapping_coarse = tri
