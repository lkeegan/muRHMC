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

// explicitly construct dirac op as dense (3xVOL)x(3xVOL) matrix
Eigen::MatrixXcd dirac_op::D_dense_matrix (field<gauge>& U, double mass, double mu_I) const {
	Eigen::MatrixXcd D_matrix = Eigen::MatrixXcd::Zero(3*U.V, 3*U.V);
	apbs_in_time(U);
	double mu_I_plus_factor = exp(mu_I);
	double mu_I_minus_factor = exp(-mu_I);
	for(int ix=0; ix<U.V; ++ix) {
		D_matrix.block<3,3>(3*ix,3*ix) = mass * SU3mat::Identity();
		// mu=0 terms have extra chemical potential isospin factors exp(+-\mu_I/2):
		// NB eta[ix][0] is just 1 so dropped from this expression
		D_matrix.block<3,3>(3*ix,3*U.iup(ix,0)) = 0.5 * mu_I_plus_factor * U[ix][0];
		D_matrix.block<3,3>(3*ix,3*U.idn(ix,0)) = -0.5 * mu_I_minus_factor * U.dn(ix,0)[0].adjoint();
		for(int mu=1; mu<4; ++mu) {
			D_matrix.block<3,3>(3*ix,3*U.iup(ix,mu)) = 0.5 * eta[ix][mu] * U[ix][mu]; 
			D_matrix.block<3,3>(3*ix,3*U.idn(ix,mu)) = -0.5 * eta[ix][mu] * U.dn(ix,mu)[mu].adjoint();
		}
	}
	// undo flip to restore original U's
	apbs_in_time(U);
	return D_matrix;
}

Eigen::MatrixXcd dirac_op::D_eigenvalues (field<gauge>& U, double mass, double mu_I) const {
	// construct explicit dense dirac matrix
	Eigen::MatrixXcd D_matrix = D_dense_matrix(U, mass, mu_I);
	// find all eigenvalues of dirac operator matrix
	Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
	ces.compute(D_matrix);
	// return complex phase of determinant (product of all eigenvalues) of D
	return ces.eigenvalues();
	// return complex phase of determinant (product of all eigenvalues) of D
}

Eigen::MatrixXcd dirac_op::DDdagger_eigenvalues (field<gauge>& U, double mass, double mu_I) const {
	// construct explicit dense dirac matrix
	Eigen::MatrixXcd D_matrix = D_dense_matrix(U, mass, mu_I);
	D_matrix = D_matrix * D_matrix.adjoint();
	// find all eigenvalues of dirac operator matrix
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> saes;
	saes.compute(D_matrix);
	// return complex phase of determinant (product of all eigenvalues) of D
	return saes.eigenvalues();
	// return complex phase of determinant (product of all eigenvalues) of D
}

void dirac_op::D (field<fermion>& lhs, const field<fermion>& rhs, field<gauge>& U, double mass, double mu_I) const {
	// flip sign of timelike U's at boundary to impose anti-periodic bcs on fermions 
	apbs_in_time(U);

	double mu_I_plus_factor = exp(mu_I);
	double mu_I_minus_factor = exp(-mu_I);
	// default static scheduling, with N threads, split loop into N chunks, one per thread 
	#pragma omp parallel for
	for(int ix=0; ix<rhs.V; ++ix) {
		lhs[ix] = mass * rhs[ix];
		// mu=0 terms have extra chemical potential isospin factors exp(+-\mu_I/2):
		// NB eta[ix][0] is just 1 so dropped from this expression
		lhs[ix].noalias() += 0.5 * mu_I_plus_factor * U[ix][0] * rhs.up(ix,0);
		lhs[ix].noalias() += -0.5 * mu_I_minus_factor * U.dn(ix,0)[0].adjoint() * rhs.dn(ix,0);
		for(int mu=1; mu<4; ++mu) {
			lhs[ix].noalias() += 0.5 * eta[ix][mu] * U[ix][mu] * rhs.up(ix,mu); 
			lhs[ix].noalias() += -0.5 * eta[ix][mu] * U.dn(ix,mu)[mu].adjoint() * rhs.dn(ix,mu);
		}
	}
	// undo flip to restore original U's
	apbs_in_time(U);
}

void dirac_op::DDdagger (field<fermion>& lhs, const field<fermion>& rhs, field<gauge>& U, double mass, double mu_I) {
	// without isospin:
	// (D+m)(D+m)^dagger = g5 D g5 D + m^2 = -D^2 + m^2
	// however with isospin: D(mu)^dag(mu) = -D(-mu) so above no longer holds,
	// instead here we do: 
	// (D(mu)+m)*(D(mu)+m)^dagger = (D(mu)+m)*(-D(-mu)+m) = -D(m,mu)*D(-m,-mu)
    field<fermion> tmp(rhs.grid);
	D(tmp, rhs, U, -mass, -mu_I);
	D(lhs, tmp, U, mass, mu_I);
	lhs *= -1.0;
}

int dirac_op::cg(field<fermion>& x, const field<fermion>& b, field<gauge>& U, double mass, double mu_I, double eps) {
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
		DDdagger(a, p, U, mass, mu_I);
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
		if(iter>1e4)
		{
			std::cout << "CG not converging: iter=" << iter << " error=" << sqrt(r2_new) << std::endl; 
			exit(0); 
		}
	}
	return iter;
} 

int dirac_op::cg_multishift(std::vector<field<fermion>>& x, const field<fermion>& b, field<gauge>& U, double mass, double mu_I, std::vector<double>& sigma, double eps) {
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
		DDdagger(a, p[0], U, mass, mu_I);
		// add first shift to DDdagger explicitly here:
		a.add( (sigma[0]*sigma[0]), p[0]);
		++iter;
		r2_old = r2_new;
		// beta = -<r|r>/<p|a>
		beta[0] = -r2_old / p[0].dot(a);
		// calculate zeta and beta coefficients for shifted vectors
		// see arXiv:hep-lat/9612014 for derivation
		for(int i_shift=1; i_shift<n_shifts; ++i_shift) {
			zeta_p1[i_shift] = zeta[i_shift] * zeta_m1[i_shift] * beta_m1[0];
			zeta_p1[i_shift] /= (beta[0]*alpha[0]*(zeta_m1[i_shift]-zeta[i_shift]) + zeta_m1[i_shift]*beta_m1[0]*(1.0 - (sigma[i_shift]*sigma[i_shift]-sigma[0]*sigma[0])*beta[0]));
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
		if(iter>1e4)
		{
			std::cout << "CG-multishift not converging: iter=" << iter << " error=" << sqrt(r2_new) << std::endl;
			exit(0); 
		}
	}
	return iter;	
}
