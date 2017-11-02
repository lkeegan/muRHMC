# RHMC [![Build Status](https://travis-ci.org/lkeegan/RHMC.svg?branch=master)](https://travis-ci.org/lkeegan/RHMC)
A simple lattice code: RHMC simulation of (n_f+n_f) flavor QCD with isospin chemical potential, using unimproved staggered fermions and the Wilson gauge action. See the [documentation](../blob/master/doc/RHMC.pdf) for more details.

Requires the [Eigen](http://eigen.tuxfamily.org) C++ template library for matrix operations, and supports openMP.

## Current working features
- HMC for pure gauge + n_f=4+4 isospin chemical potential
- Leapfrog and OMF2 integrator
- CG and CG-multishift inverter
- Gauge observables
	- Plaquette
	- Polyakov loop
- Fermionic observables
	- psibar-psi
- Gauge field I/O
 
## In progress
- Isospin observables
- Brute force inversion of explicitly constructed Dirac operator via Lapack

## To do
- RHMC: remez algorithm, maybe also zolotarev for square roots
- CG block solver for multiple RHS
- Mutiple timescale integration
- OMF4 integrator
- Even/odd preconditioning
- Input file
- Measurement program