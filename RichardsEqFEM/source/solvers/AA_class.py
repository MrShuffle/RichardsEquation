# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:47:01 2022

@author: jakob
"""
import numpy as np
import scipy as sp


class AndersonAcceleration:
    def __init__(self, dimension, depth):

        self.dimension = dimension
        self.depth = depth
        self.reset()
        self.fkm1 = np.zeros((self.dimension))
        self.gkm1 = np.zeros((self.dimension))

    def reset(self):
        self.F_k: np.ndarray = np.zeros(
            (self.dimension, self.depth)
        )  # changes in increments
        self.G_k: np.ndarray = np.zeros(
            (self.dimension, self.depth)
        )  # changes in fixed point applications

    def apply(self, gk: np.ndarray, fk: np.ndarray, iteration: int) -> np.ndarray:

        if iteration == 0:
            self.F_k: np.ndarray = np.zeros(
                (self.dimension, self.depth)
            )  # changes in increments
            self.G_k: np.ndarray = np.zeros(
                (self.dimension, self.depth)
            )  # changes in fixed point applications

        mk = min(self.depth, iteration)

        # apply acceleration
        if mk > 0:

            col = (iteration - 1) % self.depth

            self.F_k[:, col : col + 1] = fk - self.fkm1
            self.G_k[:, col : col + 1] = gk - self.gkm1
            # Solve least squares problem
            # [Q,R] = np.linalg.qr(self.F_k[:, range(mk)])
            # W=Q.T*self.F_k[0][col]
            # gamma_k = np.linalg.lstsq(R,W)
            # sp.linalg.lstsq(self.F_k[:, 0:mk], fk)
            lstsq_solution = sp.linalg.lstsq(self.F_k[:, 0:mk], fk)

            gamma_k = lstsq_solution[0]

            xkp1 = gk - np.dot(self.G_k[:, 0:mk], gamma_k)
            # print(xkp1)

        else:
            xkp1 = gk

        self.fkm1 = fk.copy()
        self.gkm1 = gk.copy()
        return xkp1
