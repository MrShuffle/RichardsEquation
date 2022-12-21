# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:34:35 2022

@author: jakob
"""

import copy
from collections import namedtuple

import numpy as np

LocalElementFunctionEvaluation = namedtuple(
    "LocalElementFunctionEvaluation",
    ["valgrad_Q", "val_Q", "theta_in_Q", "K_in_Q", "K_prime_Q", "theta_prime_Q"],
)
LocalElementFunctionEvaluationNorms2 = namedtuple(
    "LocalElementFunctionEvaluationNorms2",
    [
        "val_Q",
        "val_Q2",
        "theta_in_Q",
        "K_in_Q",
        "theta_in_Q2",
        "K_in_Q2",
        "K_prime_Q2",
        "valgrad_Q",
        "valgrad_Q2",
        "valgrad_Q3",
        "theta_prime_Q",
        "theta_prime_Q2",
    ],
)
FunctionEvaluation_L = namedtuple(
    "FunctionEvaluation_L", ["valgrad_Q", "val_Q", "theta_in_Q", "K_in_Q"]
)
FunctionEvaluationNorms = namedtuple(
    "FunctionEvaluation",
    [
        "valgrad_Q",
        "valgrad_Q3",
        "val_Q",
        "val_Q2",
        "val_Q3",
        "theta_in_Q",
        "K_in_Q",
        "K_in_Q2",
        "theta_in_Q2",
        "theta_prime_Q",
    ],
)
FunctionEvaluationNewtonNorms = namedtuple(
    "FunctionEvaluationNewtonNorms",
    ["valgrad_Q", "val_Q2", "theta_in_Q2", "K_in_Q2", "theta_prime_Q2"],
)


class HydraulicsLocalEvaluation:
    def __init__(self, theta, theta_prime, K, K_prime):
        self.theta_func = copy.deepcopy(theta)
        self.theta_prime_func = copy.deepcopy(theta_prime)
        self.K_func = copy.deepcopy(K)
        self.K_prime_func = copy.deepcopy(K_prime)

    def function_evaluation(self, u, Phi, dPhi):
        """
        Evaluation of FE functions and constitutive laws.

        Parameters
        ----------
        u : a vector with psi value.
        Phi, dPhi: Shape function and derivative.

        Returns
        -------
        Permability and theta evaluated at psi at the local element.

        """

        valgrad_Q = np.tensordot(u, dPhi, axes=((0), (1)))[0]
        val_Q = np.sum(u * Phi, axis=0)

        # Derivative of theta wrt psi
        theta_in_Q = self.theta_func(np.ravel(val_Q)).reshape(-1, 1)
        K_in_Q = self.K_func(np.ravel(theta_in_Q)).reshape(-1, 1)

        theta_prime_Q = self.theta_prime_func(np.ravel(val_Q)).reshape(-1, 1)
        K_prime_Q = np.multiply(
            self.K_prime_func(np.ravel(theta_in_Q)).reshape(-1, 1),
            theta_prime_Q.reshape(-1, 1),
        )

        return LocalElementFunctionEvaluation(
            valgrad_Q, val_Q, theta_in_Q, K_in_Q, K_prime_Q, theta_prime_Q
        )

    def function_evaluation_L(self, val, Phi, dPhi):
        """
        Evaluation of FE functions and constitutive laws.

        Parameters
        ----------
        val: a vector with psi value.
        Phi, dPhi: Shape function and derivative.

        Returns
        -------
        Permability and theta evaluated at psi at the local element.

        """
        val_Q = np.sum(val * Phi, axis=0)
        valgrad_Q = np.tensordot(val, dPhi, axes=((0), (1)))[0]
        theta_in_Q = self.theta_func(np.ravel(val_Q)).reshape(-1, 1)
        K_in_Q = self.K_func(np.ravel(theta_in_Q)).reshape(-1, 1)

        return FunctionEvaluation_L(valgrad_Q, val_Q, theta_in_Q, K_in_Q)

    def function_evaluation_norms(self, val, val2, val3, Phi, dPhi):
        """
        Evaluation of FE functions and constitutive laws.

        Parameters
        ----------
        val, val2, val3 : a vector with psi value.
        Phi, dPhi: Shape function and derivative.

        Returns
        -------
        Permability and theta evaluated at psi at the local element.

        """
        val_Q = np.sum(val * Phi, axis=0)
        valgrad_Q = np.tensordot(val, dPhi, axes=((0), (1)))[0]

        theta_in_Q = self.theta_func(np.ravel(val_Q)).reshape(-1, 1)
        theta_prime_Q = self.theta_prime_func(np.ravel(val_Q)).reshape(-1, 1)
        K_in_Q = self.K_func(np.ravel(theta_in_Q)).reshape(-1, 1)

        val_Q2 = np.sum(val2 * Phi, axis=0)
        theta_in_Q2 = self.theta_func(np.ravel(val_Q2)).reshape(-1, 1)
        K_in_Q2 = self.K_func(np.ravel(theta_in_Q2)).reshape(-1, 1)

        val_Q3 = np.sum(val3 * Phi, axis=0)
        valgrad_Q3 = np.tensordot(val3, dPhi, axes=((0), (1)))[0]

        return FunctionEvaluationNorms(
            valgrad_Q,
            valgrad_Q3,
            val_Q,
            val_Q2,
            val_Q3,
            theta_in_Q,
            K_in_Q,
            K_in_Q2,
            theta_in_Q2,
            theta_prime_Q,
        )

    def function_evaluation_norms2(self, val, val2, val3, Phi, dPhi):
        """
        Evaluation of FE functions and constitutive laws.


        Parameters
        ----------
        val, val2, val3 : a vector with psi value.
        Phi, dPhi: Shape function and derivative.

        Returns
        -------
        Permability and theta evaluated at psi at the local element.

        """

        val_Q = np.sum(val * Phi, axis=0)
        valgrad_Q = np.tensordot(val, dPhi, axes=((0), (1)))[0]

        theta_in_Q = self.theta_func(np.ravel(val_Q)).reshape(-1, 1)
        theta_prime_Q = self.theta_prime_func(np.ravel(val_Q)).reshape(-1, 1)
        K_in_Q = self.K_func(np.ravel(theta_in_Q)).reshape(-1, 1)

        val_Q2 = np.sum(val2 * Phi, axis=0)
        valgrad_Q2 = np.tensordot(val2, dPhi, axes=((0), (1)))[0]

        theta_in_Q2 = self.theta_func(np.ravel(val_Q2)).reshape(-1, 1)
        theta_prime_Q2 = self.theta_prime_func(np.ravel(val_Q2)).reshape(-1, 1)
        K_in_Q2 = self.K_func(np.ravel(theta_in_Q2)).reshape(-1, 1)
        K_prime_Q2 = np.multiply(
            self.K_prime_func(np.ravel(theta_in_Q2)).reshape(-1, 1),
            theta_prime_Q2.reshape(-1, 1),
        )

        valgrad_Q3 = np.tensordot(val3, dPhi, axes=((0), (1)))[0]

        return LocalElementFunctionEvaluationNorms2(
            val_Q,
            val_Q2,
            theta_in_Q,
            K_in_Q,
            theta_in_Q2,
            K_in_Q2,
            K_prime_Q2,
            valgrad_Q,
            valgrad_Q2,
            valgrad_Q3,
            theta_prime_Q,
            theta_prime_Q2,
        )

    def function_evaluation_newtonnorm(self, u, u2, Phi, dPhi):
        """
        Evaluation of FE functions and constitutive laws.

        Parameters
        ----------
        u, u2 : a vector with psi value.
        Phi, dPhi: Shape function and derivative.

        Returns
        -------
        Permability and theta evaluated at psi at the local element.

        """
        valgrad_Q = np.tensordot(u, dPhi, axes=((0), (1)))[0]

        val_Q2 = np.sum(u2 * Phi, axis=0)
        theta_in_Q2 = self.theta_func(np.ravel(val_Q2)).reshape(-1, 1)
        theta_prime_Q2 = self.theta_prime_func(np.ravel(val_Q2)).reshape(-1, 1)
        K_in_Q2 = self.K_func(np.ravel(theta_in_Q2)).reshape(-1, 1)

        return FunctionEvaluationNewtonNorms(
            valgrad_Q, val_Q2, theta_in_Q2, K_in_Q2, theta_prime_Q2
        )
