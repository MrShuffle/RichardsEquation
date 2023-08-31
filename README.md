# RichardsEquation

This repositoy contains the implementation of all examples shown and discussed in the article:

Stokke, J.S., Storvik, E., Mitra, K., Both, J.W., and Radu, F.A., An adaptive solution strategy for Richards' equation

https://arxiv.org/abs/2301.02055

---

The code in this repository is using PorePy (https://github.com/pmgbergen/porepy)
for managing grids and data structures. For an introduction to PorePy, see the
following reference:

The geometry of the domain is generated from PorePy, see; Keilegavlen, E., Berge, R., Fumagalli, A., Starnoni, M., Stefansson, I., Varela, J., & Berre, I. PorePy: an open-source software for simulation of multiphysics processes in fractured porous media. Computational Geosciences, 25, 243–265 (2021), doi:10.1007/s10596-020-10002-5

---

The code uses FreeFem++ (https://freefem.org/) and connects it to python code by using pyFreeFem https://github.com/odevauchelle/pyFreeFem.git

---
# Test cases

The associated article contains three test case, for which the code is provided in this repository:

* Test 1: Stricly unsaturated medium. Code: './StrictlyUnsaturated_EX1.py'
* Test 2: Variably saturated medium. Code: './VariablySaturated_EX2.py'
* Test 3: Realistic case, a recognized benchmark problem. Code: './BenchmarkProblem_EX3.py'
* Test 4: Heterogeneous and anisotropic medium. Test case is from: https://inria.hal.science/hal-03328944v2. Code: ./Mitra_EX4


Using the codes, the results in the paper can be reproduced. Note that the number of cpus used can be specified in the algorithm, currently set to 8.

In order to run every variation of the examples the mesh size, time step size and L-parameter can be changed. The switch can be set to permanently be True to run only Newton's method or False to only apply the L-scheme. See documentation in script.






