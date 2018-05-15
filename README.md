# muRHMC [![Build Status](https://travis-ci.org/lkeegan/muRHMC.svg?branch=master)](https://travis-ci.org/lkeegan/muRHMC)
A simple lattice code: RHMC simulation of (n_f+n_f) flavor QCD with isospin chemical potential, using unimproved staggered fermions and the Wilson gauge action. See the [documentation](doc/muRHMC.pdf) for more details.

Uses the [Eigen](http://eigen.tuxfamily.org) C++ template library for matrix operations, and supports openMP.

## Current working features
- HMC for pure gauge + n_f=4+4 isospin chemical potential
- Leapfrog and OMF2 integrator
- CG, BCG and CG-multishift inverters
- Gauge observables
	- Plaquette
	- Polyakov loop
- Fermionic gaussian noise observables
	- Quark condensate
	- Pion susceptibility
	- Isospin density
- Fermionic exact diagonalisation observables
	- Quark condensate
	- Pion susceptibility
	- Phase of determinant of Dirac op
- Gauge field I/O
 
## In progress
- Isospin observables
- Eigenvalues of Dirac op via Sparse MVM Arpack routines

## To do
- RHMC: remez algorithm, maybe also zolotarev for square roots
- CG block solver for multiple RHS
- Mutiple timescale integration
- OMF4 integrator
- Even/odd preconditioning
