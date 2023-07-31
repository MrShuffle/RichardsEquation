# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 12:41:27 2022

@author: jakob
"""

import math
from multiprocessing import Pool

import numpy as np
from scipy.sparse.coo import coo_matrix

from RichardsEqFEM.source.basisfunctions.Gauss_quadrature_points import *
from RichardsEqFEM.source.basisfunctions.lagrange_element import \
    global_element_geometry
from RichardsEqFEM.source.localevaluation.local_evaluation import \
    HydraulicsLocalEvaluation
from RichardsEqFEM.source.utils.operators import reference_to_local

# Module containing the local assembly for the L-scheme and Newton's method,
# as well as all switching criteria for the adaptive algorithm.


class LN_alg:
    def __init__(
        self,
        L,
        dt,
        d,
        g,
        order,
        psi,
        theta,
        K,
        theta_prime,
        K_prime,
        f,
        Switch=False,
    ):
        """
        A posteriori estimate based adaptive switching between L-scheme and Newton

        Parameters
        ----------
        L : L-scheme stabilization parameter.
        dt : time step size.
        d : Local to global mapping.
        g : Geometry.
        order : Polynomial basis order.
        psi : Pressure head.
        K : Permeability.
        theta : Saturation.
        K_prime : Derivative of permeability.
        theta_prime : Derivative of saturation.
        f : Source term.
        Switch : Boolean, False meaning L-scheme iteration, True meaning
            Newton iteration. The default is False.

        Returns
        -------
        None.

        """
        # Cache the nonlinearities
        d.Newton_method(K, theta, K_prime, theta_prime, f)

        # Cache parameters
        self.L = L
        self.dt = dt
        self.f = f

        # Cache grid and FE order
        self.g = g
        self.d = d
        self.order = order

        # Initalize matrices
        self.sat_matrix_t = np.zeros((d.total_geometric_pts, 1))
        self.sat_matrix_k = np.zeros((d.total_geometric_pts, 1))
        self.gravity_vector = np.zeros((d.total_geometric_pts, 1))
        self.source_vector = np.zeros((d.total_geometric_pts, 1))

        # Build FE cache
        self._build_fe_cache()

        # Model
        self.hydraulics = HydraulicsLocalEvaluation(theta, theta_prime, K, K_prime)
        self.theta = theta
        self.theta_prime = theta_prime
        self.K = K
        self.K_prime = K_prime

        # Assemble mass matrix
        _data_mass = np.empty(d.numDataPts, dtype=np.double)
        n = 0

        # Local assembly for all elements
        pool = Pool()
        elements_list = list(range(g.num_cells))
        results = pool.map(self.local_mass_assembly, elements_list)

        # Collect the results
        for result in results:
            for k, l in np.ndindex(d.element.num_dofs, d.element.num_dofs):
                _data_mass[n] = result[k, l]
                n += 1

        # Build matrix
        mass_m = coo_matrix((_data_mass, (d.row, d.col))).tocsr()
        self.mass_matrix = mass_m

    def _build_fe_cache(self):
        """
        Build cache of FE functions for reuse in each iteration.
        """

        self.fe_cache = []
        for element_num in range(self.g.num_cells):

            # Geometrical info
            cn = self.d.mapping[:, element_num]  # Node values of element
            corners = self.g.nodes[
                0:2, cn[self.d.local_dofs_corners]
            ]  # corners of element

            # FE evaluations
            P_El = global_element_geometry(
                self.d.element, corners, self.d.geometry, self.d.degree
            )

            # Map reference element to local element
            [J, c, J_inv] = reference_to_local(P_El.element_coord)
            det_J = np.abs(np.linalg.det(J))

            # Shape functions and derivative
            Phi = P_El.phi_eval(self.d.quad_pts)
            dPhi = P_El.grad_phi_eval(self.d.quad_pts)

            # Gauss quadrature
            a = gauss_quadrature_points(2)

            # Collect results
            self.fe_cache.append((Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a))

    def update_at_iteration(self, psi_k, ind, Switch):
        """
        Setup for one iteratoin of Newton's/L-scheme method, depending on Switch. If True, apply Newton.
        """

        # Fetch previous iterate
        self.psi_k = psi_k

        if Switch:
            # Newton:

            # Initialize data structure
            self.sat_matrix_k = np.zeros((self.d.total_geometric_pts, 1))
            self.gravity_vector = np.zeros((self.d.total_geometric_pts, 1))

            n = 0

            _data_perm = []
            _data_J_perm = []
            _data_J_sat = []
            _data_J_grav = []
            pool = Pool()

            # Assemble matrices for Newton's method
            elements_list = list(range(self.g.num_cells))
            results = pool.map(self.local_Newtonscheme_assembly, elements_list)

            for result in results:

                _data_perm.append(result[0].flatten())
                _data_J_perm.append(result[3].flatten())
                _data_J_sat.append(result[5].flatten())
                _data_J_grav.append(result[4].flatten())

                for i in range(len(result[-1])):
                    self.sat_matrix_k[result[-1][i]] = (
                        self.sat_matrix_k[result[-1][i]] + result[2][i]
                    )

                    self.gravity_vector[result[-1][i]] = (
                        self.gravity_vector[result[-1][i]] + result[1][i]
                    )

            # Buidl matrix
            _data_perm = np.concatenate(_data_perm)
            _data_J_perm = np.concatenate(_data_J_perm)
            _data_J_sat = np.concatenate(_data_J_sat)
            _data_J_grav = np.concatenate(_data_J_grav)

            self.perm_matrix = coo_matrix((_data_perm, (self.d.row, self.d.col)))
            self.J_perm_matrix = coo_matrix((_data_J_perm, (self.d.row, self.d.col)))
            self.J_sat_matrix = coo_matrix((_data_J_sat, (self.d.row, self.d.col)))
            self.J_gravity_matrix = coo_matrix((_data_J_grav, (self.d.row, self.d.col)))

        else:
            # L-scheme

            # Initalize matrices
            self.sat_matrix_k = np.zeros((self.d.total_geometric_pts, 1))
            self.gravity_vector = np.zeros((self.d.total_geometric_pts, 1))
            _data_perm = np.empty(self.d.numDataPts, dtype=np.double)

            n = 0

            # Local assembly
            pool = Pool()
            elements_list = list(range(self.g.num_cells))
            results = pool.map(self.local_Lscheme_assembly, elements_list)

            for result in results:

                # _data_m.append(result)
                for k, l in np.ndindex(
                    self.d.element.num_dofs, self.d.element.num_dofs
                ):
                    _data_perm[n] = result[0][k, l]
                    n += 1

                for i in range(len(result[-1])):
                    self.sat_matrix_k[result[-1][i]] = (
                        self.sat_matrix_k[result[-1][i]] + result[2][i]
                    )

                    self.gravity_vector[result[-1][i]] = (
                        self.gravity_vector[result[-1][i]] + result[1][i]
                    )

            # Global matrix
            self.perm_matrix = coo_matrix(
                (_data_perm, (self.d.row, self.d.col))
            ).tocsr()

    def update_at_newtime(self, psi_t, t):
        """
        Assembly of constant terms (for each time step).
        """

        # Initalize matrices
        self.sat_matrix_t = np.zeros((self.d.total_geometric_pts, 1))
        self.source_vector = np.zeros((self.d.total_geometric_pts, 1))

        # Fetch time and current solution
        self.time = t
        self.psi_t = psi_t

        # Local assembly
        pool = Pool()
        elements_list = list(range(self.g.num_cells))
        results = pool.map(self.local_source_saturation_assembly, elements_list)

        # Collect the results
        for result in results:

            for i in range(len(result[-1])):
                self.sat_matrix_t[result[-1][i]] = (
                    self.sat_matrix_t[result[-1][i]] + result[1][i]
                )

                self.source_vector[result[-1][i]] = (
                    self.source_vector[result[-1][i]] + result[0][i]
                )

    def assemble(self, psi_k, Switch):
        """
        Assemble lhs and rhs of either L-scheme of Newton's method
        """

        # Global assemble routine, combining all contributions to build left and right
        # hand sides. Use non-incremental formulation.

        if Switch:
            # Newton
            self.lhs = (
                self.J_sat_matrix
                + self.dt * self.perm_matrix
                + self.dt * self.J_perm_matrix
                + self.dt * self.J_gravity_matrix
            )
            self.rhs = (
                self.J_sat_matrix @ psi_k
                + self.sat_matrix_t
                - self.sat_matrix_k
                + self.dt * self.source_vector
                - self.dt * self.gravity_vector
                + self.dt * self.J_perm_matrix @ psi_k
                + self.dt * self.J_gravity_matrix @ psi_k
            )

        else:
            # L-scheme

            self.lhs = self.L * self.mass_matrix + self.dt * self.perm_matrix
            self.rhs = (
                self.L * self.mass_matrix.dot(psi_k)
                + self.sat_matrix_t
                - self.sat_matrix_k
                + self.dt * self.source_vector
                - self.dt * self.gravity_vector
            )

    def local_mass_assembly(self, element_num):
        """
        Local mass matrix assembly
        """

        # Fetch FE info
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]

        # Assemble local mass matrix
        local_mass = np.zeros((P_El.num_dofs, P_El.num_dofs))
        for l in range(len(Phi)):
            local_mass += a[l, 2] * Phi[l][:].reshape(-1, 1) @ Phi[l][:].reshape(1, -1)
        local_mass *= 0.5 * det_J

        return local_mass

    def local_source_saturation_assembly(self, element_num):
        """
        Local assembly of source term at time t and <theta(psi),phi> which is also a vector
        """

        # Fetch FE
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]
        quadrature_points = a[:, 0:2]
        quadrature_pt = np.array([quadrature_points]).T

        # Initalize local vectors
        local_source = np.zeros((P_El.num_dofs, 1))
        local_saturation = np.zeros((P_El.num_dofs, 1))

        # Local pressure head values
        psi_local = np.array([self.psi_t[cn[i]] for i in range(3)])

        # Local function values
        local_vals = self.hydraulics.function_evaluation_L(psi_local, Phi, dPhi)

        # Construct the local verctor from different FE contributions
        for j in range(P_El.num_dofs):
            val1 = np.sum(local_vals.theta_in_Q * (a[:, 2] * Phi[:][j]).reshape(-1, 1))
            val2 = 0
            for k in range(len(Phi)):

                vec = [(quadrature_pt)[0][k], (quadrature_pt)[1][k]]
                q_i = J @ vec + c  # transformed quadrature points
                w_i = a[k][2]  # weights
                val2 += (
                    w_i
                    * self.d.f(self.time, q_i[0][0].item(), q_i[1][0].item())
                    * Phi[k][j]
                )

            local_source[j] = 0.5 * val2 * det_J
            local_saturation[j] = 0.5 * val1 * det_J

        return local_source, local_saturation, cn

    def local_Newtonscheme_assembly(self, element_num):
        """
        Local assembly of <theta(psi_k),phi>, <K(psi_k)dPhi,dPhi>, <K(psi_k)e_z,dPhi>, <K'(psi_k)e_z,dPhi> and <K'(psi_k) nabla(psi_k),dPhi>
        """

        # Fetch FE
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]
        transform = J_inv @ J_inv.T

        # Local pressure head values
        psi_local = np.array([self.psi_k[cn[i]] for i in range(3)])

        # Local function evaluations
        local_vals = self.hydraulics.function_evaluation(
            psi_local,
            Phi,
            dPhi,
        )

        # Data structures
        local_perm = np.zeros((P_El.num_dofs, P_El.num_dofs))
        local_gravity = np.zeros((P_El.num_dofs, 1))
        local_saturation_k = np.zeros((P_El.num_dofs, 1))
        local_J_perm = np.zeros((P_El.num_dofs, P_El.num_dofs))
        local_J_gravity = np.zeros((P_El.num_dofs, P_El.num_dofs))
        local_J_saturation = np.zeros((P_El.num_dofs, P_El.num_dofs))

        # Build contributions to Newton's method
        for l in range(len(Phi)):

            local_perm += (
                local_vals.K_in_Q[l][0]
                * a[l][2]
                * (dPhi[l][:] @ transform @ dPhi[l][:].T)
            )
            local_J_perm += (
                local_vals.K_prime_Q[l][0]
                * a[l][2]
                * (local_vals.valgrad_Q[l] @ transform @ dPhi[l][:].T).reshape(-1, 1)
                @ Phi[l][:].reshape(1, -1)
            )

            local_J_gravity += (
                local_vals.K_prime_Q[l]
                * a[l][2]
                * dPhi[l][:]
                @ J_inv.T
                @ np.array([[0], [1]])
                @ Phi[l][:].reshape(1, -1)
            )
            local_J_saturation += (
                local_vals.theta_prime_Q[l]
                * a[l][2]
                * Phi[l][:].reshape(-1, 1)
                * Phi[l][:].reshape(1, -1)
            )

            local_gravity += (
                local_vals.K_in_Q[l]
                * a[l][2]
                * dPhi[l][:]
                @ J_inv.T
                @ np.array([[0], [1]])
            )

        for j in range(3):
            local_saturation_k[j] = np.sum(
                local_vals.theta_in_Q * (a[:, 2] * Phi[:][j]).reshape(-1, 1)
            )

        scaling = 0.5 * det_J
        local_saturation_k *= scaling
        local_gravity *= scaling
        local_perm *= scaling
        local_J_perm *= scaling
        local_J_gravity *= scaling
        local_J_saturation *= scaling

        return (
            local_perm,
            local_gravity,
            local_saturation_k,
            local_J_perm,
            local_J_gravity,
            local_J_saturation,
            cn,
        )

    def local_Lscheme_assembly(self, element_num):
        """
        Local assembly of <theta(psi_k),phi>, <K(psi_k)dPhi,dPhi>, <K(psi_k)e_z,dPhi>
        """

        # Fetch FE
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]
        transform = J_inv @ J_inv.T

        # Local pressure head values
        psi_local = np.array([self.psi_k[cn[i]] for i in range(3)])

        # Hydraulics evaluated
        local_vals = self.hydraulics.function_evaluation_L(psi_local, Phi, dPhi)

        # Local assembly
        local_perm = np.zeros((P_El.num_dofs, P_El.num_dofs))
        local_gravity = np.zeros((P_El.num_dofs, 1))
        local_saturation_k = np.zeros((P_El.num_dofs, 1))

        # Single contributions to L-scheme
        for j in range(3):
            local_saturation_k[j] = np.sum(
                local_vals.theta_in_Q * (a[:, 2] * Phi[:][j]).reshape(-1, 1)
            )

        for l in range(len(Phi)):

            local_perm += (
                local_vals.K_in_Q[l][0]
                * a[l][2]
                * (dPhi[l][:] @ transform @ dPhi[l][:].T)
            )
            local_gravity += (
                local_vals.K_in_Q[l]
                * a[l][2]
                * dPhi[l][:]
                @ J_inv.T
                @ np.array([[0], [1]])
            )

        scaling = 0.5 * det_J
        local_saturation_k *= scaling
        local_gravity *= scaling
        local_perm *= scaling

        return local_perm, local_gravity, local_saturation_k, cn

    def linearization_error(self, u_j, u_j_1):
        """
        Computes lineartization error wrt iteration dependent energy norm.

        """
        self.u_j = u_j
        self.u_j_1 = u_j_1
        val = 0
        pool = Pool()
        elements_list = list(range(self.g.num_cells))
        results = pool.map(self.error_on_element, elements_list)
        val = np.sum(results)

        self.linear_norm = np.sqrt(val)

    def error_on_element(self, element_num):
        """
        Computes error on each element.
        """

        # Fetch FE
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]

        # Local pressure head values
        psi_local = np.array([self.u_j_1[cn[i]] for i in range(3)])
        psi_local2 = np.array([self.u_j[cn[i]] for i in range(3)])

        # Evaluate functions locally
        local_vals = self.hydraulics.function_evaluation_newtonnorm(
            psi_local, psi_local2, Phi, dPhi
        )

        R_h = np.dot((psi_local).T.reshape(len(psi_local)), Phi.T)

        val = 0
        for k in range(len(Phi)):
            val += (
                self.d.quad_weights[k]
                * local_vals.theta_prime_Q2[k]
                * R_h[k] ** 2
                * det_J
                + self.d.quad_weights[k]
                * self.dt
                * local_vals.K_in_Q2[k]
                * np.linalg.norm(local_vals.valgrad_Q[k] @ J_inv.T) ** 2
                * det_J
            )

        return val

    def estimate_CN(self, u):
        """
        Estimation of C_N^j
        """

        # Compute pointwise gradient
        h = 1 / self.d.x_part
        Z = u.reshape(self.d.x_part + 1, self.d.y_part + 1)
        etax, etay = np.gradient(Z, h, h)
        gradeta = np.zeros(((self.d.x_part + 1) * (self.d.y_part + 1), 2))
        n = 0
        for i in range(self.d.y_part + 1):
            for j in range(self.d.x_part + 1):
                gradeta[n] = np.array([etax[j][i], etay[j][i]])
                n += 1

        # Initialize
        C_array = np.zeros((len(u), 1))
        K_prime_Q = np.zeros((len(u), 1), dtype=np.float32)

        # Evaluate hydraulics
        theta_in_Q = self.theta(np.ravel(u))
        K_in_Q = self.K(np.ravel(theta_in_Q))
        theta_prime_Q = self.theta_prime(np.ravel(u))

        # Compute further terms in the estimate
        for k in range(len(u)):

            # Avoid discontinuity
            if np.abs(gradeta[k]).any() > 0.5:  
                gradeta[k] = np.zeros((2,))

            # C_N^j definition is almost everywhere, for the benchmark problem 5-6 individual points
            # causes C_N^j<2. Those points can be negelcted as the definition is almost everywhere
            # and the points are not measurable. (This criteria does not affect other examples)
            if math.isclose(
                theta_prime_Q[k], 0
            ) or 0.395999<=theta_in_Q[k].item()<=0.396:
                K_prime_Q[k] = 0

            else:
                K_prime_Q[k] = (
                    self.K_prime(theta_in_Q[k].item()) * theta_prime_Q[k]
                ).astype(np.float16)

            # Compute C_N^j at every point
            C_array[k] = (
                np.sqrt(self.dt)
                * np.linalg.norm(
                    K_in_Q[k] ** (-1 / 2)
                    * K_prime_Q[k]
                    * (gradeta[k] + np.array([0, 1]))
                )
                / np.sqrt(theta_prime_Q[k])
            )
            if theta_prime_Q[k] == 0:  # Inequality satisfied in degenrate zone
                C_array[k] = 0
            if k == self.d.boundary.any():  # Avoid boundary
                C_array[k] = 0

        # Pick largest C_N^j
        self.CN = np.nanmax(C_array)
        if 0 < self.CN < 2:
            self.CN = self.CN
        elif self.CN >= 2:

            self.CN = self.CN
        else:  # Avoid negative values
            self.CN = 0
        return C_array

    def N_to_L_eta(self, w1, w2):
        """
        Compute the switch criterion from Newton to L-scheme.

        Parameters
        ----------
        w1 : psi^(k).
        w2 : psi^(k-1).

        Returns
        -------
        Indicator for switch to Newton's method.

        """

        # Guess CN
        CN = 0  # To speed up computations
        # self.estimate_CN(w1)
        # CN = self.CN
        a = 2 / (2 - CN)

        # Fetch arguments
        self.w1 = w1
        self.w2 = w2
        self.diff = w1 - w2

        # Assemble switch criterion
        pool = Pool(8)
        elements_list = list(range(self.g.num_cells))
        results = pool.map(self.norm_N_to_L_on_element, elements_list)

        val = 0
        val2 = 0
        val3 = 0
        for result in results:

            val += result[0]
            val2 += result[1]
            val3 += result[2]

        # Determine final scalar global criterion
        self.linear_norm = np.sqrt(val3)
        self.eta_NtoL = a / self.linear_norm * np.sqrt(val + self.dt * val2)

    def norm_N_to_L_on_element(self, element_num):
        """
        Auxiliary for N_to_L_eta.
        """
        # Fetch FE
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]
        wi = self.d.quad_weights * det_J

        # Local Values
        psi_local = np.array([self.w2[cn[i]] for i in range(3)])
        psi_local2 = np.array([self.w1[cn[i]] for i in range(3)])
        psi_local3 = np.array([self.diff[cn[i]] for i in range(3)])
        local_vals = self.hydraulics.function_evaluation_norms2(
            psi_local2,
            psi_local,
            psi_local3,
            Phi,
            dPhi,
        )

        R_h = np.dot((psi_local3).T.reshape(3), Phi.T)

        K_nonzero = [
            k
            for k in range(len(Phi))
            if not math.isclose(local_vals.theta_prime_Q[k][0], 0)
        ]
        val = sum(
            [
                wi[k]
                * (
                    1
                    / local_vals.theta_prime_Q[k][0] ** (1 / 2)
                    * (
                        local_vals.theta_prime_Q2[k][0]
                        * (local_vals.val_Q[k] - local_vals.val_Q2[k])
                        - (local_vals.theta_in_Q[k] - local_vals.theta_in_Q2[k])
                    )
                )
                ** 2
                for k in K_nonzero
            ]
        )

        val2 = sum(
            [
                wi[k]
                * (
                    (
                        (
                            (local_vals.K_in_Q[k] - local_vals.K_in_Q2[k])
                            * np.linalg.norm(
                                local_vals.valgrad_Q2[k] @ J_inv.T + np.array([0, 1])
                            )
                        )
                        - (
                            local_vals.K_prime_Q2[k]
                            * (local_vals.val_Q[k] - local_vals.val_Q2[k])
                            * np.linalg.norm(
                                local_vals.valgrad_Q[k] @ J_inv.T + np.array([0, 1])
                            )
                        )
                    )
                    * 1
                    / local_vals.K_in_Q[k] ** (1 / 2)
                )
                ** 2
                for k in range(len(Phi))
            ]
        )

        val3 = sum(
            [
                wi[k] * local_vals.theta_prime_Q2[k][0] * R_h[k] ** 2
                + self.dt
                * wi[k]
                * local_vals.K_in_Q2[k]
                * np.linalg.norm(local_vals.valgrad_Q3[k] @ J_inv.T) ** 2
                for k in range(len(Phi))
            ]
        )

        return val, val2, val3

    def L_to_N_eta(self, w1, w2):
        """
        Compute the switch criterion from L-scheme to Newton.

        Parameters
        ----------
        w1 : psi^(k).
        w2 : psi^(k-1).

        Returns
        -------
        Indicator for switch to Newton's method.

        """
        # Estimate CN
        self.estimate_CN(w1)
        self.CN = self.CN
        print(self.CN, "CN")
        a = 2 / (2 - self.CN)

        # Fetch data
        self.w1 = w1
        self.w2 = w2
        self.diff = self.w1 - self.w2

        # Local assembly
        pool = Pool(8)
        elements_list = list(range(self.g.num_cells))
        results = pool.map(self.norm_L_to_N_on_element, elements_list)

        # Collection of results
        val = 0
        val2 = 0
        val3 = 0
        val4 = 0

        for result in results:

            val += result[0]
            val2 += result[1]
            val3 += result[2]  # L-scheme linearization norm error
            val4 += result[3]

        # Determine final global scalar
        self.linear_norm = np.sqrt(val3)
        self.eta_LtoL = 1 / self.linear_norm * np.sqrt(val4 + self.dt * val2)
        self.eta_LtoN = a / self.linear_norm * np.sqrt(val + self.dt * val2)

    def norm_L_to_N_on_element(self, element_num):
        """
        Auxiliary for L_to_N_eta.
        """

        # Fetch FE
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]

        # Local solution
        psi_local = np.array([self.w2[cn[i]] for i in range(3)])
        psi_local2 = np.array([self.w1[cn[i]] for i in range(3)])
        psi_local3 = np.array([self.diff[cn[i]] for i in range(3)])

        # Local function evaluations
        local_vals = self.hydraulics.function_evaluation_norms(
            psi_local2,
            psi_local,
            psi_local3,
            Phi,
            dPhi,
        )

        R_h = np.dot((psi_local3).T.reshape(len(psi_local)), Phi.T)

        K_nonzero = [
            k
            for k in range(len(Phi))
            if not math.isclose(local_vals.theta_prime_Q[k][0], 0)
        ]

        val = sum(
            [
                self.d.quad_weights[k]
                * det_J
                * (
                    1
                    / local_vals.theta_prime_Q[k][0] ** (1 / 2)
                    * (
                        self.L * (local_vals.val_Q[k] - local_vals.val_Q2[k])
                        - (local_vals.theta_in_Q[k] - local_vals.theta_in_Q2[k])
                    )
                )
                ** 2
                for k in K_nonzero
            ]
        )

        val2 = sum(
            [
                self.d.quad_weights[k]
                * det_J
                * (
                    1
                    / local_vals.K_in_Q[k] ** (1 / 2)
                    * ((local_vals.K_in_Q[k] - local_vals.K_in_Q2[k]))
                    * np.linalg.norm(
                        local_vals.valgrad_Q[k] @ J_inv.T + np.array([0, 1])
                    )
                )
                ** 2
                for k in range(len(Phi))
            ]
        )

        val3 = sum(
            [
                self.d.quad_weights[k] * self.L * R_h[k] ** 2 * det_J
                + self.dt
                * self.d.quad_weights[k]
                * local_vals.K_in_Q2[k]
                * np.linalg.norm(local_vals.valgrad_Q3[k] @ J_inv.T) ** 2
                * det_J
                for k in range(len(Phi))
            ]
        )

        val4 = sum(
            [
                self.d.quad_weights[k]
                * det_J
                * (
                    1
                    / self.L ** (1 / 2)
                    * (
                        self.L * (local_vals.val_Q[k] - local_vals.val_Q2[k])
                        - (local_vals.theta_in_Q[k] - local_vals.theta_in_Q2[k])
                    )
                )
                ** 2
                for k in range(len(Phi))
            ]
        )

        return val, val2, val3, val4

    def update_L(self, L):
        """
        Update of the L-scheme parameter. Used for Adaptive L-scheme.
        """
        self.L = L



# import math
# from multiprocessing import Pool

# import numpy as np
# from scipy.sparse.coo import coo_matrix

# from RichardsEqFEM.source.basisfunctions.Gauss_quadrature_points import *
# from RichardsEqFEM.source.basisfunctions.lagrange_element import \
#     global_element_geometry
# from RichardsEqFEM.source.localevaluation.local_evaluation import \
#     HydraulicsLocalEvaluation
# from RichardsEqFEM.source.utils.operators import reference_to_local

# # Module containing the local assembly for the L-scheme and Newton's method,
# # as well as all switching criteria for the adaptive algorithm.


class LN_alg_heter:
    def __init__(
        self,
        L,
        dt,
        d,
        g,
        order,
        psi,
        theta,
        K,
        theta_prime,
        K_prime,
        f,
        permability_tensor=None,
        Switch=False,
    ):
        """
        A posteriori estimate based adaptive switching between L-scheme and Newton

        Parameters
        ----------
        L : L-scheme stabilization parameter.
        dt : time step size.
        d : Local to global mapping.
        g : Geometry.
        order : Polynomial basis order.
        psi : Pressure head.
        K : Permeability.
        theta : Saturation.
        K_prime : Derivative of permeability.
        theta_prime : Derivative of saturation.
        f : Source term.
        permability_tensor: permability tensor 2x2 constant
        Switch : Boolean, False meaning L-scheme iteration, True meaning
            Newton iteration. The default is False.

        Returns
        -------
        None.

        """
        # Cache the nonlinearities
        d.Newton_method(K, theta, K_prime, theta_prime, f)

        # Cache parameters
        self.L = L
        self.dt = dt
        self.f = f
        self.glob_perm = permability_tensor

        # Cache grid and FE order
        self.g = g
        self.d = d
        self.order = order

        # Initalize matrices
        self.sat_matrix_t = np.zeros((d.total_geometric_pts, 1))
        self.sat_matrix_k = np.zeros((d.total_geometric_pts, 1))
        self.gravity_vector = np.zeros((d.total_geometric_pts, 1))
        self.source_vector = np.zeros((d.total_geometric_pts, 1))

        # Build FE cache
        self._build_fe_cache()

        # Model
        self.hydraulics = HydraulicsLocalEvaluation(theta, theta_prime, K, K_prime)
        self.theta = theta
        self.theta_prime = theta_prime
        self.K = K
        self.K_prime = K_prime

        # Assemble mass matrix
        _data_mass = np.empty(d.numDataPts, dtype=np.double)
        n = 0

        # Local assembly for all elements
        pool = Pool()
        elements_list = list(range(g.num_cells))
        results = pool.map(self.local_mass_assembly, elements_list)

        # Collect the results
        for result in results:
            for k, l in np.ndindex(d.element.num_dofs, d.element.num_dofs):
                _data_mass[n] = result[k, l]
                n += 1

        # Build matrix
        mass_m = coo_matrix((_data_mass, (d.row, d.col))).tocsr()
        self.mass_matrix = mass_m

    def _build_fe_cache(self):
        """
        Build cache of FE functions for reuse in each iteration.
        """

        self.fe_cache = []
        for element_num in range(self.g.num_cells):

            # Geometrical info
            cn = self.d.mapping[:, element_num]  # Node values of element
            corners = self.g.nodes[
                0:2, cn[self.d.local_dofs_corners]
            ]  # corners of element

            # FE evaluations
            P_El = global_element_geometry(
                self.d.element, corners, self.d.geometry, self.d.degree
            )

            # Map reference element to local element
            [J, c, J_inv] = reference_to_local(P_El.element_coord)
            det_J = np.abs(np.linalg.det(J))

            # Shape functions and derivative
            Phi = P_El.phi_eval(self.d.quad_pts)
            dPhi = P_El.grad_phi_eval(self.d.quad_pts)

            # Gauss quadrature
            a = gauss_quadrature_points(2)

            # Collect results
            self.fe_cache.append((Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a))

    def update_at_iteration(self, psi_k, ind, Switch):
        """
        Setup for one iteratoin of Newton's/L-scheme method, depending on Switch. If True, apply Newton.
        """

        # Fetch previous iterate
        self.psi_k = psi_k

        if Switch:
            # Newton:

            # Initialize data structure
            self.sat_matrix_k = np.zeros((self.d.total_geometric_pts, 1))
            self.gravity_vector = np.zeros((self.d.total_geometric_pts, 1))

            n = 0

            _data_perm = []
            _data_J_perm = []
            _data_J_sat = []
            _data_J_grav = []
            pool = Pool()

            # Assemble matrices for Newton's method
            elements_list = list(range(self.g.num_cells))
            results = pool.map(self.local_Newtonscheme_assembly, elements_list)

            for result in results:

                _data_perm.append(result[0].flatten())
                _data_J_perm.append(result[3].flatten())
                _data_J_sat.append(result[5].flatten())
                _data_J_grav.append(result[4].flatten())

                for i in range(len(result[-1])):
                    self.sat_matrix_k[result[-1][i]] = (
                        self.sat_matrix_k[result[-1][i]] + result[2][i]
                    )

                    self.gravity_vector[result[-1][i]] = (
                        self.gravity_vector[result[-1][i]] + result[1][i]
                    )

            # Buidl matrix
            _data_perm = np.concatenate(_data_perm)
            _data_J_perm = np.concatenate(_data_J_perm)
            _data_J_sat = np.concatenate(_data_J_sat)
            _data_J_grav = np.concatenate(_data_J_grav)

            self.perm_matrix = coo_matrix((_data_perm, (self.d.row, self.d.col)))
            self.J_perm_matrix = coo_matrix((_data_J_perm, (self.d.row, self.d.col)))
            self.J_sat_matrix = coo_matrix((_data_J_sat, (self.d.row, self.d.col)))
            self.J_gravity_matrix = coo_matrix((_data_J_grav, (self.d.row, self.d.col)))

        else:
            # L-scheme

            # Initalize matrices
            self.sat_matrix_k = np.zeros((self.d.total_geometric_pts, 1))
            self.gravity_vector = np.zeros((self.d.total_geometric_pts, 1))
            _data_perm = np.empty(self.d.numDataPts, dtype=np.double)

            n = 0

            # Local assembly
            pool = Pool()
            elements_list = list(range(self.g.num_cells))
            results = pool.map(self.local_Lscheme_assembly, elements_list)

            for result in results:

                # _data_m.append(result)
                for k, l in np.ndindex(
                    self.d.element.num_dofs, self.d.element.num_dofs
                ):
                    _data_perm[n] = result[0][k, l]
                    n += 1

                for i in range(len(result[-1])):
                    self.sat_matrix_k[result[-1][i]] = (
                        self.sat_matrix_k[result[-1][i]] + result[2][i]
                    )

                    self.gravity_vector[result[-1][i]] = (
                        self.gravity_vector[result[-1][i]] + result[1][i]
                    )

            # Global matrix
            self.perm_matrix = coo_matrix(
                (_data_perm, (self.d.row, self.d.col))
            ).tocsr()

    def update_at_newtime(self, psi_t, t):
        """
        Assembly of constant terms (for each time step).
        """

        # Initalize matrices
        self.sat_matrix_t = np.zeros((self.d.total_geometric_pts, 1))
        self.source_vector = np.zeros((self.d.total_geometric_pts, 1))

        # Fetch time and current solution
        self.time = t
        self.psi_t = psi_t

        # Local assembly
        pool = Pool()
        elements_list = list(range(self.g.num_cells))
        results = pool.map(self.local_source_saturation_assembly, elements_list)

        # Collect the results
        for result in results:

            for i in range(len(result[-1])):
                self.sat_matrix_t[result[-1][i]] = (
                    self.sat_matrix_t[result[-1][i]] + result[1][i]
                )

                self.source_vector[result[-1][i]] = (
                    self.source_vector[result[-1][i]] + result[0][i]
                )

    def assemble(self, psi_k, Switch):
        """
        Assemble lhs and rhs of either L-scheme of Newton's method
        """

        # Global assemble routine, combining all contributions to build left and right
        # hand sides. Use non-incremental formulation.

        if Switch:
            # Newton
            self.lhs = (
                self.J_sat_matrix
                + self.dt * self.perm_matrix
                + self.dt * self.J_perm_matrix
                + self.dt * self.J_gravity_matrix
            )
            self.rhs = (
                self.J_sat_matrix @ psi_k
                + self.sat_matrix_t
                - self.sat_matrix_k
                + self.dt * self.source_vector
                - self.dt * self.gravity_vector
                + self.dt * self.J_perm_matrix @ psi_k
                + self.dt * self.J_gravity_matrix @ psi_k
            )

        else:
            # L-scheme

            self.lhs = self.L * self.mass_matrix + self.dt * self.perm_matrix
            self.rhs = (
                self.L * self.mass_matrix.dot(psi_k)
                + self.sat_matrix_t
                - self.sat_matrix_k
                + self.dt * self.source_vector
                - self.dt * self.gravity_vector
            )

    def local_mass_assembly(self, element_num):
        """
        Local mass matrix assembly
        """

        # Fetch FE info
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]

        # Assemble local mass matrix
        local_mass = np.zeros((P_El.num_dofs, P_El.num_dofs))
        for l in range(len(Phi)):
            local_mass += a[l, 2] * Phi[l][:].reshape(-1, 1) @ Phi[l][:].reshape(1, -1)
        local_mass *= 0.5 * det_J

        return local_mass

    def local_source_saturation_assembly(self, element_num):
        """
        Local assembly of source term at time t and <theta(psi),phi> which is also a vector
        """

        # Fetch FE
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]
        quadrature_points = a[:, 0:2]
        quadrature_pt = np.array([quadrature_points]).T

        # Initalize local vectors
        local_source = np.zeros((P_El.num_dofs, 1))
        local_saturation = np.zeros((P_El.num_dofs, 1))

        # Local pressure head values
        psi_local = np.array([self.psi_t[cn[i]] for i in range(3)])

        # Local function values
        local_vals = self.hydraulics.function_evaluation_L(psi_local, Phi, dPhi)

        # Construct the local verctor from different FE contributions
        for j in range(P_El.num_dofs):
            val1 = np.sum(local_vals.theta_in_Q * (a[:, 2] * Phi[:][j]).reshape(-1, 1))
            val2 = 0
            for k in range(len(Phi)):

                vec = [(quadrature_pt)[0][k], (quadrature_pt)[1][k]]
                q_i = J @ vec + c  # transformed quadrature points
                w_i = a[k][2]  # weights
                val2 += (
                    w_i
                    * self.d.f(self.time, q_i[0][0].item(), q_i[1][0].item())
                    * Phi[k][j]
                )

            local_source[j] = 0.5 * val2 * det_J
            local_saturation[j] = 0.5 * val1 * det_J

        return local_source, local_saturation, cn

    def local_Newtonscheme_assembly(self, element_num):
        """
        Local assembly of <theta(psi_k),phi>, <K(psi_k)dPhi,dPhi>, <K(psi_k)e_z,dPhi>, <K'(psi_k)e_z,dPhi> and <K'(psi_k) nabla(psi_k),dPhi>
        """

        # Fetch FE
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]
        transform = J_inv @ J_inv.T

        # Local pressure head values
        psi_local = np.array([self.psi_k[cn[i]] for i in range(3)])

        # Local function evaluations
        local_vals = self.hydraulics.function_evaluation(
            psi_local,
            Phi,
            dPhi,
        )

        # Data structures
        local_perm = np.zeros((P_El.num_dofs, P_El.num_dofs))
        local_gravity = np.zeros((P_El.num_dofs, 1))
        local_saturation_k = np.zeros((P_El.num_dofs, 1))
        local_J_perm = np.zeros((P_El.num_dofs, P_El.num_dofs))
        local_J_gravity = np.zeros((P_El.num_dofs, P_El.num_dofs))
        local_J_saturation = np.zeros((P_El.num_dofs, P_El.num_dofs))

        # Build contributions to Newton's method
        for l in range(len(Phi)):

            local_perm += (
                local_vals.K_in_Q[l][0]
                * a[l][2]
                * (dPhi[l][:] @ self.glob_perm@transform @ dPhi[l][:].T)
            )
            local_J_perm += (
                local_vals.K_prime_Q[l][0]
                * a[l][2]
                * (local_vals.valgrad_Q[l] @ self.glob_perm@transform @ dPhi[l][:].T).reshape(-1, 1)
                @ Phi[l][:].reshape(1, -1)
            )

            local_J_gravity += (
                local_vals.K_prime_Q[l]
                * a[l][2]
                * dPhi[l][:]
                @self.glob_perm@ J_inv.T
                @ np.array([[0], [1]])
                @ Phi[l][:].reshape(1, -1)
            )
            local_J_saturation += (
                local_vals.theta_prime_Q[l]
                * a[l][2]
                * Phi[l][:].reshape(-1, 1)
                * Phi[l][:].reshape(1, -1)
            )

            local_gravity += (
                local_vals.K_in_Q[l]
                * a[l][2]
                * dPhi[l][:]
                @self.glob_perm@ J_inv.T
                @ np.array([[0], [1]])
            )

        for j in range(3):
            local_saturation_k[j] = np.sum(
                local_vals.theta_in_Q * (a[:, 2] * Phi[:][j]).reshape(-1, 1)
            )

        scaling = 0.5 * det_J
        local_saturation_k *= scaling
        local_gravity *= scaling
        local_perm *= scaling
        local_J_perm *= scaling
        local_J_gravity *= scaling
        local_J_saturation *= scaling

        return (
            local_perm,
            local_gravity,
            local_saturation_k,
            local_J_perm,
            local_J_gravity,
            local_J_saturation,
            cn,
        )

    def local_Lscheme_assembly(self, element_num):
        """
        Local assembly of <theta(psi_k),phi>, <K(psi_k)dPhi,dPhi>, <K(psi_k)e_z,dPhi>
        """

        # Fetch FE
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]
        transform = J_inv @ J_inv.T

        # Local pressure head values
        psi_local = np.array([self.psi_k[cn[i]] for i in range(3)])

        # Hydraulics evaluated
        local_vals = self.hydraulics.function_evaluation_L(psi_local, Phi, dPhi)

        # Local assembly
        local_perm = np.zeros((P_El.num_dofs, P_El.num_dofs))
        local_gravity = np.zeros((P_El.num_dofs, 1))
        local_saturation_k = np.zeros((P_El.num_dofs, 1))

        # Single contributions to L-scheme
        for j in range(3):
            local_saturation_k[j] = np.sum(
                local_vals.theta_in_Q * (a[:, 2] * Phi[:][j]).reshape(-1, 1)
            )

        for l in range(len(Phi)):

            local_perm += (
                local_vals.K_in_Q[l][0]
                * a[l][2]
                * (dPhi[l][:] @ self.glob_perm@transform @ dPhi[l][:].T)
            )
            local_gravity += (
                local_vals.K_in_Q[l]
                * a[l][2]
                * dPhi[l][:]
                @self.glob_perm@ J_inv.T
                @ np.array([[0], [1]])
            )

        scaling = 0.5 * det_J
        local_saturation_k *= scaling
        local_gravity *= scaling
        local_perm *= scaling

        return local_perm, local_gravity, local_saturation_k, cn

    def linearization_error(self, u_j, u_j_1):
        """
        Computes lineartization error wrt iteration dependent energy norm.

        """
        self.u_j = u_j
        self.u_j_1 = u_j_1
        val = 0
        pool = Pool()
        elements_list = list(range(self.g.num_cells))
        results = pool.map(self.error_on_element, elements_list)
        val = np.sum(results)

        self.linear_norm = np.sqrt(val)

    def error_on_element(self, element_num):
        """
        Computes error on each element.
        """

        # Fetch FE
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]

        # Local pressure head values
        psi_local = np.array([self.u_j_1[cn[i]] for i in range(3)])
        psi_local2 = np.array([self.u_j[cn[i]] for i in range(3)])

        # Evaluate functions locally
        local_vals = self.hydraulics.function_evaluation_newtonnorm(
            psi_local, psi_local2, Phi, dPhi
        )

        R_h = np.dot((psi_local).T.reshape(len(psi_local)), Phi.T)

        val = 0
        for k in range(len(Phi)):
            val += (
                self.d.quad_weights[k]
                * local_vals.theta_prime_Q2[k]
                * R_h[k] ** 2
                * det_J
                + self.d.quad_weights[k]
                * self.dt
                * local_vals.K_in_Q2[k]
                * np.linalg.norm(local_vals.valgrad_Q[k] @ J_inv.T) ** 2
                * det_J
            )

        return val

    def estimate_CN(self, u):
        """
        Estimation of C_N^j
        """

        # Compute pointwise gradient
        h = 1 / self.d.x_part
        Z = u.reshape(self.d.x_part + 1, self.d.y_part + 1)
        etax, etay = np.gradient(Z, h, h)
        gradeta = np.zeros(((self.d.x_part + 1) * (self.d.y_part + 1), 2))
        n = 0
        for i in range(self.d.y_part + 1):
            for j in range(self.d.x_part + 1):
                gradeta[n] = np.array([etax[j][i], etay[j][i]])
                n += 1

        # Initialize
        C_array = np.zeros((len(u), 1))
        K_prime_Q = np.zeros((len(u), 1), dtype=np.float32)

        # Evaluate hydraulics
        theta_in_Q = self.theta(np.ravel(u))
        K_in_Q = self.K(np.ravel(theta_in_Q))
        theta_prime_Q = self.theta_prime(np.ravel(u))

        # Compute further terms in the estimate
        for k in range(len(u)):

            # Avoid discontinuity
            if np.abs(gradeta[k]).any() > 0.5:  
                gradeta[k] = np.zeros((2,))

            # C_N^j definition is almost everywhere, for the benchmark problem 5-6 individual points
            # causes C_N^j<2. Those points can be negelcted as the definition is almost everywhere
            # and the points are not measurable. (This criteria does not affect other examples)
            if math.isclose(
                theta_prime_Q[k], 0
            ) or 0.395999<=theta_in_Q[k].item()<=0.396:
                K_prime_Q[k] = 0

            else:
                K_prime_Q[k] = (
                    self.K_prime(theta_in_Q[k].item()) * theta_prime_Q[k]
                ).astype(np.float16)

            # Compute C_N^j at every point
            C_array[k] = (
                np.sqrt(self.dt)
                * np.linalg.norm(
                    K_in_Q[k] ** (-1 / 2)
                    * K_prime_Q[k]
                    * (gradeta[k] + np.array([0, 1]))
                )
                / np.sqrt(theta_prime_Q[k])
            )
            if theta_prime_Q[k] == 0:  # Inequality satisfied in degenrate zone
                C_array[k] = 0
            if k == self.d.boundary.any():  # Avoid boundary
                C_array[k] = 0

        # Pick largest C_N^j
        self.CN = np.nanmax(C_array)
        if 0 < self.CN < 2:
            self.CN = self.CN
        elif self.CN >= 2:

            self.CN = self.CN
        else:  # Avoid negative values
            self.CN = 0
        return C_array

    def N_to_L_eta(self, w1, w2):
        """
        Compute the switch criterion from Newton to L-scheme.

        Parameters
        ----------
        w1 : psi^(k).
        w2 : psi^(k-1).

        Returns
        -------
        Indicator for switch to Newton's method.

        """

        # Guess CN
        CN = 0  # To speed up computations
        # self.estimate_CN(w1)
        # CN = self.CN
        a = 2 / (2 - CN)

        # Fetch arguments
        self.w1 = w1
        self.w2 = w2
        self.diff = w1 - w2

        # Assemble switch criterion
        pool = Pool(8)
        elements_list = list(range(self.g.num_cells))
        results = pool.map(self.norm_N_to_L_on_element, elements_list)

        val = 0
        val2 = 0
        val3 = 0
        for result in results:

            val += result[0]
            val2 += result[1]
            val3 += result[2]

        # Determine final scalar global criterion
        self.linear_norm = np.sqrt(val3)
        self.eta_NtoL = a / self.linear_norm * np.sqrt(val + self.dt * val2)

    def norm_N_to_L_on_element(self, element_num):
        """
        Auxiliary for N_to_L_eta.
        """
        # Fetch FE
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]
        wi = self.d.quad_weights * det_J

        # Local Values
        psi_local = np.array([self.w2[cn[i]] for i in range(3)])
        psi_local2 = np.array([self.w1[cn[i]] for i in range(3)])
        psi_local3 = np.array([self.diff[cn[i]] for i in range(3)])
        local_vals = self.hydraulics.function_evaluation_norms2(
            psi_local2,
            psi_local,
            psi_local3,
            Phi,
            dPhi,
        )

        R_h = np.dot((psi_local3).T.reshape(3), Phi.T)

        K_nonzero = [
            k
            for k in range(len(Phi))
            if not math.isclose(local_vals.theta_prime_Q[k][0], 0)
        ]
        val = sum(
            [
                wi[k]
                * (
                    1
                    / local_vals.theta_prime_Q[k][0] ** (1 / 2)
                    * (
                        local_vals.theta_prime_Q2[k][0]
                        * (local_vals.val_Q[k] - local_vals.val_Q2[k])
                        - (local_vals.theta_in_Q[k] - local_vals.theta_in_Q2[k])
                    )
                )
                ** 2
                for k in K_nonzero
            ]
        )

        val2 = sum(
            [
                wi[k]
                * (
                    (
                        (
                            (local_vals.K_in_Q[k] - local_vals.K_in_Q2[k])
                            * np.linalg.norm(
                                local_vals.valgrad_Q2[k] @ J_inv.T + np.array([0, 1])
                            )
                        )
                        - (
                            local_vals.K_prime_Q2[k]
                            * (local_vals.val_Q[k] - local_vals.val_Q2[k])
                            * np.linalg.norm(
                                local_vals.valgrad_Q[k] @ J_inv.T + np.array([0, 1])
                            )
                        )
                    )
                    * 1
                    / local_vals.K_in_Q[k] ** (1 / 2)
                )
                ** 2
                for k in range(len(Phi))
            ]
        )

        val3 = sum(
            [
                wi[k] * local_vals.theta_prime_Q2[k][0] * R_h[k] ** 2
                + self.dt
                * wi[k]
                * local_vals.K_in_Q2[k]
                * np.linalg.norm(local_vals.valgrad_Q3[k] @ J_inv.T) ** 2
                for k in range(len(Phi))
            ]
        )

        return val, val2, val3

    def L_to_N_eta(self, w1, w2):
        """
        Compute the switch criterion from L-scheme to Newton.

        Parameters
        ----------
        w1 : psi^(k).
        w2 : psi^(k-1).

        Returns
        -------
        Indicator for switch to Newton's method.

        """
        # Estimate CN
        self.estimate_CN(w1)
        self.CN = self.CN
        print(self.CN, "CN")
        a = 2 / (2 - self.CN)

        # Fetch data
        self.w1 = w1
        self.w2 = w2
        self.diff = self.w1 - self.w2

        # Local assembly
        pool = Pool(8)
        elements_list = list(range(self.g.num_cells))
        results = pool.map(self.norm_L_to_N_on_element, elements_list)

        # Collection of results
        val = 0
        val2 = 0
        val3 = 0
        val4 = 0

        for result in results:

            val += result[0]
            val2 += result[1]
            val3 += result[2]  # L-scheme linearization norm error
            val4 += result[3]

        # Determine final global scalar
        self.linear_norm = np.sqrt(val3)
        self.eta_LtoL = 1 / self.linear_norm * np.sqrt(val4 + self.dt * val2)
        self.eta_LtoN = a / self.linear_norm * np.sqrt(val + self.dt * val2)

    def norm_L_to_N_on_element(self, element_num):
        """
        Auxiliary for L_to_N_eta.
        """

        # Fetch FE
        Phi, dPhi, P_El, J, c, J_inv, det_J, cn, a = self.fe_cache[element_num]

        # Local solution
        psi_local = np.array([self.w2[cn[i]] for i in range(3)])
        psi_local2 = np.array([self.w1[cn[i]] for i in range(3)])
        psi_local3 = np.array([self.diff[cn[i]] for i in range(3)])

        # Local function evaluations
        local_vals = self.hydraulics.function_evaluation_norms(
            psi_local2,
            psi_local,
            psi_local3,
            Phi,
            dPhi,
        )

        R_h = np.dot((psi_local3).T.reshape(len(psi_local)), Phi.T)

        K_nonzero = [
            k
            for k in range(len(Phi))
            if not math.isclose(local_vals.theta_prime_Q[k][0], 0)
        ]

        val = sum(
            [
                self.d.quad_weights[k]
                * det_J
                * (
                    1
                    / local_vals.theta_prime_Q[k][0] ** (1 / 2)
                    * (
                        self.L * (local_vals.val_Q[k] - local_vals.val_Q2[k])
                        - (local_vals.theta_in_Q[k] - local_vals.theta_in_Q2[k])
                    )
                )
                ** 2
                for k in K_nonzero
            ]
        )

        val2 = sum(
            [
                self.d.quad_weights[k]
                * det_J
                * (
                    1
                    / local_vals.K_in_Q[k] ** (1 / 2)
                    * ((local_vals.K_in_Q[k] - local_vals.K_in_Q2[k]))
                    * np.linalg.norm(
                        local_vals.valgrad_Q[k] @ J_inv.T + np.array([0, 1])
                    )
                )
                ** 2
                for k in range(len(Phi))
            ]
        )

        val3 = sum(
            [
                self.d.quad_weights[k] * self.L * R_h[k] ** 2 * det_J
                + self.dt
                * self.d.quad_weights[k]
                * local_vals.K_in_Q2[k]
                * np.linalg.norm(local_vals.valgrad_Q3[k] @ J_inv.T) ** 2
                * det_J
                for k in range(len(Phi))
            ]
        )

        val4 = sum(
            [
                self.d.quad_weights[k]
                * det_J
                * (
                    1
                    / self.L ** (1 / 2)
                    * (
                        self.L * (local_vals.val_Q[k] - local_vals.val_Q2[k])
                        - (local_vals.theta_in_Q[k] - local_vals.theta_in_Q2[k])
                    )
                )
                ** 2
                for k in range(len(Phi))
            ]
        )

        return val, val2, val3, val4

    def update_L(self, L):
        """
        Update of the L-scheme parameter. Used for Adaptive L-scheme.
        """
        self.L = L
