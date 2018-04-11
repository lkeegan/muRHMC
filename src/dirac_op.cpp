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