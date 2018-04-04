#include "dirac_op.hpp"
#include "omp.h"
#include <iostream> //FOR DEBUGGING

dirac_op::dirac_op (const lattice& grid, double mass, double mu_I) : mass(mass), mu_I(mu_I), eta(grid) {
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

void dirac_op::apbcs_in_time (field<gauge>& U) const {
	if(ANTI_PERIODIC_BCS) {
		for(int ix=0; ix<U.VOL3; ++ix) {
			int i4x = U.it_ix(U.L0-1, ix);
			U[i4x][0] *= -1;
		}
	}
}

void dirac_op::apply_eta_bcs_to_U (field<gauge>& U) {
	if (!GAUGE_LINKS_INCLUDE_ETA_BCS) {
		for(int ix=0; ix<U.V; ++ix) {
			for(int mu=1; mu<4; ++mu) {
				// eta[mu=0] = 1 so skip it
				U[ix][mu] *= eta[ix][mu];
			}	
		} //NB maybe can incorporate apbcs in eta's - need to check that etas are not used anywhere else...
		// probably OK apart from massimo config format, so check this isn't broken by this..
		if(ANTI_PERIODIC_BCS) {
			for(int ix=0; ix<U.VOL3; ++ix) {
				int i4x = U.it_ix(U.L0-1, ix);
				U[i4x][0] *= -1;
			}
		}
		GAUGE_LINKS_INCLUDE_ETA_BCS = true;
	}
}

void dirac_op::remove_eta_bcs_from_U (field<gauge>& U) {
	if (GAUGE_LINKS_INCLUDE_ETA_BCS) {
		for(int ix=0; ix<U.V; ++ix) {
			for(int mu=1; mu<4; ++mu) {
				U[ix][mu] *= eta[ix][mu];
			}	
		}
		if(ANTI_PERIODIC_BCS) {
			for(int ix=0; ix<U.VOL3; ++ix) {
				int i4x = U.it_ix(U.L0-1, ix);
				U[i4x][0] *= -1;
			}
		}
		GAUGE_LINKS_INCLUDE_ETA_BCS = false;
	}
}

void dirac_op::gamma5 (field<fermion>& phi) const {
	for(int ix=0; ix<phi.V; ++ix) {
		phi[ix] *= eta[ix][4];
	}
}

// explicitly construct dirac op as dense (3xVOL)x(3xVOL) matrix
Eigen::MatrixXcd dirac_op::D_dense_matrix (field<gauge>& U){
	Eigen::MatrixXcd D_matrix = Eigen::MatrixXcd::Zero(3*U.V, 3*U.V);
	apply_eta_bcs_to_U(U);
	double mu_I_plus_factor = exp(0.5 * mu_I);
	double mu_I_minus_factor = exp(-0.5 * mu_I);
	for(int ix=0; ix<U.V; ++ix) {
		D_matrix.block<3,3>(3*ix,3*ix) = mass * SU3mat::Identity();
		// mu=0 terms have extra chemical potential isospin factors exp(+-\mu_I/2):
		// NB eta[ix][0] is just 1 so dropped from this expression
		D_matrix.block<3,3>(3*ix,3*U.iup(ix,0)) = 0.5 * mu_I_plus_factor * U[ix][0];
		D_matrix.block<3,3>(3*ix,3*U.idn(ix,0)) = -0.5 * mu_I_minus_factor * U.dn(ix,0)[0].adjoint();
		for(int mu=1; mu<4; ++mu) {
			D_matrix.block<3,3>(3*ix,3*U.iup(ix,mu)) = 0.5 * U[ix][mu]; 
			D_matrix.block<3,3>(3*ix,3*U.idn(ix,mu)) = -0.5 * U.dn(ix,mu)[mu].adjoint();
		}
	}
	remove_eta_bcs_from_U(U);
	return D_matrix;
}

// explicitly construct (dD/d\mu) as dense (3xVOL)x(3xVOL) matrix
Eigen::MatrixXcd dirac_op::dD_dmu_dense_matrix (field<gauge>& U) const {
	Eigen::MatrixXcd D_matrix = Eigen::MatrixXcd::Zero(3*U.V, 3*U.V);
	apbcs_in_time(U);
	double mu_I_plus_factor = exp(0.5 * mu_I);
	double mu_I_minus_factor = exp(-0.5 * mu_I);
	for(int ix=0; ix<U.V; ++ix) {
		D_matrix.block<3,3>(3*ix,3*U.iup(ix,0)) = mu_I * 0.25 * mu_I_plus_factor * U[ix][0];
		D_matrix.block<3,3>(3*ix,3*U.idn(ix,0)) = mu_I * 0.25 * mu_I_minus_factor * U.dn(ix,0)[0].adjoint();
	}
	// undo flip to restore original U's
	apbcs_in_time(U);
	return D_matrix;
}

Eigen::MatrixXcd dirac_op::D_eigenvalues (field<gauge>& U) {
	// construct explicit dense dirac matrix
	Eigen::MatrixXcd D_matrix = D_dense_matrix(U);
	// find all eigenvalues of non-hermitian dirac operator matrix D
	Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
	ces.compute(D_matrix);
	return ces.eigenvalues();
}

Eigen::MatrixXcd dirac_op::DDdagger_eigenvalues (field<gauge>& U) {
	// construct explicit dense dirac matrix
	Eigen::MatrixXcd D_matrix = D_dense_matrix(U);
	D_matrix = D_matrix * D_matrix.adjoint();
	// find all eigenvalues of hermitian dirac operator matrix DDdagger
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> saes;
	saes.compute(D_matrix);
	return saes.eigenvalues();
}

double dirac_op::D_phase_angle (field<gauge>& U) {
	Eigen::MatrixXcd eigenvalues = D_eigenvalues(U);				
	// phase{Det[D]} = phase{\prod_i \lambda_i} = \sum_i phase{lambda_i}
	double sum = 0;
	for (int i=0; i<eigenvalues.size(); ++i) {
		sum += std::arg(eigenvalues(i));
	}
	return sum;
}

double dirac_op::pion_susceptibility_exact (field<gauge>& U) {
	Eigen::MatrixXcd eigenvaluesDDdag = DDdagger_eigenvalues (U);				
	return eigenvaluesDDdag.cwiseInverse().sum().real()/static_cast<double>(3*U.V);
}

// massless even_odd part of dirac op (assumes mu_I=0):
// also assumes that gauge links contain eta matrices and bcs
// rhs is only defined for odd sites, lhs for even sites
void dirac_op::D_eo (field<fermion>& lhs, const field<fermion>& rhs, const field<gauge>& U) const {
	// default static scheduling, with N threads, split loop into N chunks, one per thread 
	//std::cout << "D_EO_" << lhs.eo_storage << rhs.eo_storage << std::endl;
	#pragma omp parallel for
	// loop over even ix_e = ix (no offset from true ix):
	for(int ix=0; ix<lhs.V; ++ix) {
		lhs[ix].setZero();
		for(int mu=0; mu<4; ++mu) {
			lhs[ix].noalias() += 0.5 * U[ix][mu] * rhs.up(ix,mu); 
			lhs[ix].noalias() += -0.5 * U.dn(ix,mu)[mu].adjoint() * rhs.dn(ix,mu);
		}
	}
}

void dirac_op::D_oe (field<fermion>& lhs, const field<fermion>& rhs, const field<gauge>& U) const {
	// default static scheduling, with N threads, split loop into N chunks, one per thread 
	//std::cout << "D_OE_" << lhs.eo_storage << rhs.eo_storage << std::endl;
	#pragma omp parallel for
	// loop over odd ix_o = ix + V:
	for(int ix_o=0; ix_o<lhs.V; ++ix_o) {
		lhs[ix_o].setZero();
		// true ix has V offset from ix_o:
		int ix = ix_o + lhs.V; 
		for(int mu=0; mu<4; ++mu) {
			lhs[ix_o].noalias() += 0.5 * U[ix][mu] * rhs.up(ix,mu); 
			lhs[ix_o].noalias() += -0.5 * U.dn(ix,mu)[mu].adjoint() * rhs.dn(ix,mu);
		}
	}
}

void dirac_op::D (field<fermion>& lhs, const field<fermion>& rhs, field<gauge>& U) {
	apply_eta_bcs_to_U(U);
	double mu_I_plus_factor = exp(0.5 * mu_I);
	double mu_I_minus_factor = exp(-0.5 * mu_I);
	// default static scheduling, with N threads, split loop into N chunks, one per thread 
	#pragma omp parallel for
	for(int ix=0; ix<rhs.V; ++ix) {
		lhs[ix] = mass * rhs[ix];
		// mu=0 terms have extra chemical potential isospin factors exp(+-\mu_I/2):
		// NB eta[ix][0] is just 1 so dropped from this expression
		lhs[ix].noalias() += 0.5 * mu_I_plus_factor * U[ix][0] * rhs.up(ix,0);
		lhs[ix].noalias() += -0.5 * mu_I_minus_factor * U.dn(ix,0)[0].adjoint() * rhs.dn(ix,0);
		for(int mu=1; mu<4; ++mu) {
			lhs[ix].noalias() += 0.5 * U[ix][mu] * rhs.up(ix,mu); 
			lhs[ix].noalias() += -0.5 * U.dn(ix,mu)[mu].adjoint() * rhs.dn(ix,mu);
		}
	}
	remove_eta_bcs_from_U(U);
}

void dirac_op::DDdagger (field<fermion>& lhs, const field<fermion>& rhs, field<gauge>& U, bool LEAVE_ETA_BCS_IN_GAUGE_FIELD) {
	// without isospin:
	// (D+m)(D+m)^dagger = g5 D g5 D + m^2 = -D^2 + m^2
	// however with isospin: D(mu)^dag(mu) = -D(-mu) so above no longer holds,
	// instead here we do: 
	// (D(mu)+m)*(D(mu)+m)^dagger = (D(mu)+m)*(-D(-mu)+m) = -D(m,mu)*D(-m,-mu)
	if(lhs.eo_storage == field<fermion>::EVEN_ONLY && rhs.eo_storage == field<fermion>::EVEN_ONLY) {
		// if lhs and rhs are both even-only fields,
		// then use even-even sub block of operator (and assume mu_I=0)
		apply_eta_bcs_to_U(U);
	    field<fermion> tmp_o(rhs.grid, field<fermion>::ODD_ONLY);
		D_oe(tmp_o, rhs, U);
		D_eo(lhs, tmp_o, U);
		lhs.scale_add(-1.0, mass*mass, rhs);
		remove_eta_bcs_from_U(U);
	} else {
		// otherwise do full operator with non-zero mu_I
	    field<fermion> tmp(rhs.grid);
	    // D^dagger(mass, mu_I) = -D(-mass, -mu_I)
	    mass = -mass;
	    mu_I = -mu_I;
		D(tmp, rhs, U);
		tmp *= -1.0;
	    mass = -mass;
	    mu_I = -mu_I;
		D(lhs, tmp, U);
	}
}

// in-place chebyshev polynomial of DDdag acting on vector of fermion fields
// if supplied X is EVEN_ONLY then DDdagger_ee is used and mu_I is assumed to be zero
// Uses recursive formula from hep-lat/0512021
// c_{n+1}= 2*z*c_{n} - c_{n-1}
// where z = ((v+u) - 2 DDdag) / (u - v)
// c_0 = 1, c_1 = z
void dirac_op::chebyshev (int k, double u, double v, std::vector<field<fermion>>& X, field<gauge>& U) {
	int n_block = X.size();
	field<fermion> c_minus2 (X[0].grid, X[0].eo_storage);
	field<fermion> c_minus1 (X[0].grid, X[0].eo_storage);
	field<fermion> c_minus0 (X[0].grid, X[0].eo_storage);
	double norm = (v + u) / (v - u);
	double inv_vmu = 1.0 / (v - u);
	for(int i_b=0; i_b<n_block; ++i_b) {
		// do each vector separately to minimise memory requirements
		// c_0 = x
		c_minus1 = X[i_b];
		// c_1 = z(x) = norm x - inv_vmu DDdag x
		DDdagger (c_minus0, c_minus1, U);
		c_minus0.scale_add(-inv_vmu, norm, c_minus1);
		// c_k = 2 z(c_{k-1}) - c_{k-2}
		for(int i_k=1; i_k<k; ++i_k) {
			// relabel previous c's
			c_minus2 = c_minus1;
			c_minus1 = c_minus0;
			// calculate current c
			DDdagger (c_minus0, c_minus1, U);
			c_minus0.scale_add(-inv_vmu, norm, c_minus1);
			c_minus0.scale_add(2.0, -1.0, c_minus2);
		}
		X[i_b] = c_minus0;
	}
}

double dirac_op::largest_eigenvalue_bound (field<gauge>& U, field<fermion>::eo_storage_options EO_STORAGE, double rel_err) {
	field<fermion> x (U.grid, EO_STORAGE), x2 (U.grid, EO_STORAGE);
	for(int i=0; i<x.V; ++i) {
		x[i] = fermion::Random();
	}
	double x_norm = x.norm();
	double lambda_max = 1;
	double lambda_max_err = 100;
	int iter = 0;
	while((lambda_max_err/lambda_max) > rel_err) {
		for(int i=0; i<8; ++i) {
			x /= x_norm;
			DDdagger(x2, x, U);
			x2 /= x2.norm();
			DDdagger(x, x2, U);				
			x_norm = x.norm();
			iter += 2;
		}
		lambda_max = x2.dot(x).real();
		lambda_max_err = sqrt(x_norm * x_norm - lambda_max * lambda_max);
		std::cout << "lambda_max" << lambda_max << ", " << lambda_max_err << std::endl;
	}
	return lambda_max + lambda_max_err;
}