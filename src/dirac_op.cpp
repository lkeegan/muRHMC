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

void dirac_op::apbs_in_time (field<gauge>& U) const {
	if(ANTI_PERIODIC_BCS) {
		for(int ix=0; ix<U.VOL3; ++ix) {
			int i4x = U.it_ix(U.L0-1, ix);
			U[i4x][0] *= -1;
		}
	}
}

void dirac_op::D (field<fermion>& lhs, const field<fermion>& rhs, field<gauge>& U, double m) const {
	// flip sign of timelike U's at boundary to impose anti-periodic bcs on fermions 
	apbs_in_time(U);
	// default static scheduling, with N threads, split loop into N chunks, one per thread 
	#pragma omp parallel for
	for(int ix=0; ix<rhs.V; ++ix) {
		lhs[ix] = m * rhs[ix];
		// mu=0 terms have extra chemical potential isospin factors exp(+-\mu_I/2): 
		lhs[ix] += 0.5 * eta[ix][0] * (U[ix][0] * exp(0.5*mu_I) * rhs.up(ix,0) - U.dn(ix,0)[0].adjoint() * exp(-0.5*mu_I) * rhs.dn(ix,0));
		for(int mu=1; mu<4; ++mu) {
			lhs[ix] += 0.5 * eta[ix][mu] * (U[ix][mu] * rhs.up(ix,mu) - U.dn(ix,mu)[mu].adjoint() * rhs.dn(ix,mu));
		}
	}
	// undo flip to restore original U's
	apbs_in_time(U);
}

void dirac_op::DDdagger (field<fermion>& lhs, const field<fermion>& rhs, field<gauge>& U, double m) const {
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

int dirac_op::cg(field<fermion>& x, const field<fermion>& b, field<gauge>& U, double m, double eps) const {
	field<fermion> a(b.grid), p(b.grid), r(b.grid);
	double r2_old = 0;
	double r2_new = 0;
	std::complex<double> beta;
	double alpha;
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
		// beta = -<r|r>/<p|a>
		beta = -r2_old / p.dot(a);
		// x -= beta * p
		x.add(-beta, p);
		// r += beta a
		r.add(beta, a);
		// r2_new = <r|r>
		r2_new = r.squaredNorm();
		alpha = r2_new / r2_old;
		// p = alpha p + r
		p.scale_add(alpha, 1.0, r);
		//std::cout << r2_new/b2 << std::endl;
	}
	return iter;	
} 

int dirac_op::cg_multishift(std::vector<field<fermion>>& x, const field<fermion>& b, field<gauge>& U, std::vector<double>& m, double eps) const {
	int n_shifts = x.size();
	std::vector<field<fermion>> p;
	field<fermion> a(b.grid), r(b.grid);
	double r2_old = 0;
	double r2_new = 0;
	std::vector<std::complex<double>> beta(n_shifts), beta_m1(n_shifts);
	std::vector<std::complex<double>> zeta_m1(n_shifts), zeta(n_shifts), zeta_p1(n_shifts);
	std::vector<std::complex<double>> alpha(n_shifts);
	int iter = 0;

	// initial guess zero required for multi-shift CG
	// x_i = 0, a = Ax_0 = 0
	a.setZero();
	// r = b - a
	r = b;
	// p = r
	for(int i_shift=0; i_shift<n_shifts; ++i_shift) {
		p.push_back(r);
		x[i_shift].setZero();
		zeta[i_shift] = 1.0;
		zeta_m1[i_shift] = 1.0;
		alpha[i_shift] = 0.0;		
	}
	beta_m1[0] = 1.0;
	// r2_new = <r|r>
	r2_new = r.squaredNorm();
	//double b2 = b.squaredNorm();
	while (sqrt(r2_new) > eps)
	{
		// a = A p_0
		DDdagger(a, p[0], U, m[0]);
		++iter;
		r2_old = r2_new;
		// beta = -<r|r>/<p|a>
		beta[0] = -r2_old / p[0].dot(a);
		// calculate zeta and beta coefficients for shifted vectors
		// see arXiv:hep-lat/9612014 for derivation
		for(int i_shift=1; i_shift<n_shifts; ++i_shift) {
			zeta_p1[i_shift] = zeta[i_shift] * zeta_m1[i_shift] * beta_m1[0];
			zeta_p1[i_shift] /= (beta[0]*alpha[0]*(zeta_m1[i_shift]-zeta[i_shift]) + zeta_m1[i_shift]*beta_m1[0]*(1.0 - (m[i_shift]*m[i_shift]-m[0]*m[0])*beta[0]));
			beta[i_shift] = beta[0] * zeta_p1[i_shift] / zeta[i_shift];
		}
		// x_i -= beta_i * p_i
		for(int i_shift=0; i_shift<n_shifts; ++i_shift) {
			x[i_shift].add(-beta[i_shift], p[i_shift]);			
		}
		// r += beta_0 a
		r.add(beta[0], a);
		// r2_new = <r|r>
		r2_new = r.squaredNorm();
		// increment timestep:
		// X_m1 <- X
		// X <- X_p1
		alpha[0] = r2_new / r2_old;
		beta_m1[0] = beta[0];
		for(int i_shift=1; i_shift<n_shifts; ++i_shift) {
			beta_m1[i_shift] = beta[i_shift];
			zeta_m1[i_shift] = zeta[i_shift];
			zeta[i_shift] = zeta_p1[i_shift];
		}
		// calculate alpha coeffs for shifted vectors
		for(int i_shift=1; i_shift<n_shifts; ++i_shift) {
			alpha[i_shift] = alpha[0] * (zeta[i_shift] * beta_m1[i_shift]) / (zeta_m1[i_shift] * beta_m1[0]);
		}
		// p_i = alpha_i p_i + zeta_i r
		for(int i_shift=0; i_shift<n_shifts; ++i_shift) {
			p[i_shift].scale_add(alpha[i_shift], zeta[i_shift], r);
		}
	}
	return iter;	

}
