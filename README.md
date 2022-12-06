# RichardsEquation


---

# Install PorePy

The geometry of the domain is generated from PorePy, see; Keilegavlen, E., Berge, R., Fumagalli, A., Starnoni, M., Stefansson, I., Varela, J., & Berre, I. PorePy: an open-source software for simulation of multiphysics processes in fractured porous media. Computational Geosciences, 25, 243–265 (2021), doi:10.1007/s10596-020-10002-5

***Install tutorial***

https://github.com/pmgbergen/porepy/blob/develop/Install.md

---
# Test cases

There are three codes for the following test cases Test 1 (StrictlyUnsaturated_EX1): Stricly unsaturated medium. Test 2 (VariablySaturated_EX2): Variably saturated medium. Test 3 (Benchmark_EX#): Realistic case, a reconized benchmark problem.

Comment: All run scripts can be modified to run just Newton or L-scheme by enforcing the Switch to be either false or true. Or it is possible to import the linearization schemes indivually from the Model_class_parallell file, where also the modified L-scheme is possible to import.
