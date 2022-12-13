# RichardsEquation

This repositoy contains the implementation of all examples shown and discussed in the article:

Stokke, J.S., Storvik, E., Mitra, K., Both, J.W., and Radu, F.A., An Adaptive and Robust Algorithm to Switch between Linearization Schemes for Nonlinear Diffusion through Porous Domains.

# TODO add reference to arxiv when published.

---

# Install PorePy
The code in this repository is using PorePy (https://github.com/pmgbergen/porepy)
for managing grids and data structures. For an introduction to PorePy, see the
following reference:

The geometry of the domain is generated from PorePy, see; Keilegavlen, E., Berge, R., Fumagalli, A., Starnoni, M., Stefansson, I., Varela, J., & Berre, I. PorePy: an open-source software for simulation of multiphysics processes in fractured porous media. Computational Geosciences, 25, 243–265 (2021), doi:10.1007/s10596-020-10002-5

---
# Test cases

The associated article contains three test case, for which the code is provided in this repository:

* Test 1: Stricly unsaturated medium. Code: './StrictlyUnsaturated_EX1.py'
* Test 2: Variably saturated medium. Code: './VariablySaturated_EX2.py'
* Test 3: (Benchmark_EX#): Realistic case, a recognized benchmark problem. Code: '*.py'

# TODO add location of code for example 3

Using the codes, the results in the paper can be reproduced.

# TODO Add comment whether the parameters have to be modified to reproduce the results or whether everything is provided.

Comment: All run scripts can be modified to run just Newton or L-scheme by enforcing the Switch to be either false or true. Or it is possible to import the linearization schemes indivually from the Model_class_parallell file, where also the modified L-scheme is possible to import.
