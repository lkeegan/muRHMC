# muRHMC [![Build Status](https://travis-ci.org/lkeegan/muRHMC.svg?branch=master)](https://travis-ci.org/lkeegan/muRHMC) [![codecov](https://codecov.io/gh/lkeegan/muRHMC/branch/master/graph/badge.svg)](https://codecov.io/gh/lkeegan/muRHMC)
A simple C++ lattice code: RHMC simulation of (n_f+n_f) flavor QCD with isospin chemical potential, using unimproved staggered fermions and the Wilson gauge action. See the [documentation](doc/muRHMC.pdf) for more details.

Uses the [Eigen](http://eigen.tuxfamily.org) C++ template library for matrix operations, and supports openMP.

## Current working features
- RHMC for pure gauge + n_f staggered fermions [or n_f+n_f isospin chemical potential staggered fermions]
- Even/odd preconditioning
- Leapfrog and OMF2 integrator
- Mutiple timescale integration
- CG, multishift CG solver
- Block solvers: BCG[A][dQ/dQA][rQ]
- Multishift block solver: SBCGrQ
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
