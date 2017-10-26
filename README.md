# RHMC [![Build Status](https://travis-ci.org/lkeegan/RHMC.svg?branch=master)](https://travis-ci.org/lkeegan/RHMC)
A simple lattice code: RHMC simulation of QCD with n_f unimproved staggered fermions

Requires the eigen C++ header library for matrix operations, and can use multiple openMP threads on a single shared memory machine.

# Current working features
-HMC for pure gauge + n_f=8 staggered fermions
-Leapfrog integrator
-CG and CG-multishift inverter
-Plaquette and Polykov loop observables

# In progress
-Isospin chemical potential
-Gauge field I/O
-Isospin observables
-Brute force inversion of explicitly constructed Dirac operator via Lapack

# To do
-RHMC: remez algorithm, maybe also zolotarev for square roots
-CG block solver for multiple RHS
-Mutiple timescale integration
-Higher order integrators: OM2, OM4
-Even/odd preconditioning
-Input file
-Measurement program