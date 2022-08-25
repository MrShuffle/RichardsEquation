# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:42:58 2022

@author: jakob
"""

import numpy as np

def reference_to_local(element_num,coordinates, cn):
    # coordinates of the refrence triangle
    mat_coord = np.array([[0, 0, 1],
                          [1, 0, 1],
                          [0, 1, 1]])
    b1 = np.array([[coordinates[0][0]],
                       [coordinates[0][1]],
                       [coordinates[0][2]]])
    b2 = np.array([[coordinates[1][0]],
                       [coordinates[1][1]],
                       [coordinates[1][2]]])
    # solution to the two linear systems
    a1 = np.linalg.solve(mat_coord, b1)
    a2 = np.linalg.solve(mat_coord, b2)

    J = np.matrix([[a1[0][0], a1[1][0]], [a2[0][0], a2[1][0]]])
    c = np.matrix([[a1[2][0]], [a2[2][0]]])
    J_inv = np.linalg.inv(J)
    return [J, c,J_inv]