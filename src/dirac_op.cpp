#include "dirac_op.hpp"
#include "omp.h"
#include <iostream> //FOR DEBUGGING

dirac_op::dirac_op (const lattice& grid) : grid(grid), eta(grid) {
	construct_eta(eta, grid);
}

void dirac_op::construct_eta(field<gamma_matrices>& eta, const lattice& grid) {
	for(int l=0; l<grid.L3; ++l){
		for(int k=0; k<grid.L2; ++k){
			for(int j=0; j<grid.L1; ++j){
				for(int i=0; i<grid.L0; ++i){
					int ix = grid.index(i, j, k, l);
					eta[ix][0] = +1.0;
					eta[ix][1] = +1.0-2.0*(i%2);
					eta[ix][2] = +1.0-2.0*((i+j)%2);
					eta[ix][3] = +1.0-2.0*((i+j+k)%2);
					eta[ix][4] = +1.0-2.0*((i+j+k+l)%2);
				}
			}
		}
	}
}

void dirac_op::gamma5 (field<fermion>& phi) const {
	for(int ix=0; ix<phi.V; ++ix) {
		phi[ix] *= eta[ix][4];
	}
}

void dirac_op::D (field<fermion> &lhs, const field<fermion> &rhs, const field<gauge>& U, double m) const {
	// default static scheduling, with N threads, split loop into N chunks, one per thread 
	#pragma omp parallel for
	for(int ix=0; ix<rhs.V; ++ix) {
		lhs[ix] = m * rhs[ix];
		for(int mu=0; mu<4; ++mu) {
			lhs[ix] += 0.5 * eta[ix][mu] * (U[ix][mu]*rhs.up(ix,mu) - U.dn(ix,mu)[mu].adjoint() * rhs.dn(ix,mu));
		}
	}
}

void dirac_op::DDdagger (field<fermion> &lhs, const field<fermion> &rhs, const field<gauge>& U, double m) const {
	// (D+m)(D+m)^dagger == g5 D g5 D + m^2
	field<fermion> tmp(rhs.grid);
	D(tmp, rhs, U, 0.0);
	gamma5(tmp);
	D(lhs, tmp, U, 0.0);
	gamma5(lhs);
	for(int ix=0; ix<rhs.V; ++ix) {
		lhs[ix] += (m * m) * rhs[ix];
	}
}

int dirac_op::cg(field<fermion>& x, const field<fermion>& b, const field<gauge>& U, double m, double eps) const {
	field<fermion> a(b.grid), p(b.grid), r(b.grid);
	double r2_old = 0;
	double r2_new = 0;
	std::complex<double> alpha;
	double beta;
	int iter = 0;

	// initial guess zero since we'll need to do this for multi-shift CG anyway
	// x = 0, a = Ax = 0
	x.setZero();
	a.setZero();
	// r = b - a
	r = b;
	// p = r
	p = r;
	// r2_new = <r|r>
	r2_new = r.squaredNorm();
	//double b2 = b.squaredNorm();
	while (sqrt(r2_new) > eps)
	{
		// a = A p
		DDdagger(a, p, U, m);
		++iter;
		r2_old = r2_new;
		// alpha = <r|r>/<p|a>
		alpha = r2_old / p.dot(a);
		// x += alpha * p
		x.add(alpha, p);
		// r -= alpha a
		r.add(-alpha, a);
		// r2_new = <r|r>
		r2_new = r.squaredNorm();
		beta = r2_new / r2_old;
		// p = beta p + r
		p.scale_add(beta, 1.0, r);
		//std::cout << r2_new/b2 << std::endl;
	}
	return iter;	
} 
