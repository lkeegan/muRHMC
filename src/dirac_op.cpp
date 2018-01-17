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
	double mu_I_plus_factor = exp(0.5 * mu_I);
	double mu_I_minus_factor = exp(-0.5 * mu_I);
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

// explicitly construct (dD/d\mu) as dense (3xVOL)x(3xVOL) matrix
Eigen::MatrixXcd dirac_op::dD_dmu_dense_matrix (field<gauge>& U, double mu_I) const {
	Eigen::MatrixXcd D_matrix = Eigen::MatrixXcd::Zero(3*U.V, 3*U.V);
	apbs_in_time(U);
	double mu_I_plus_factor = exp(0.5 * mu_I);
	double mu_I_minus_factor = exp(-0.5 * mu_I);
	for(int ix=0; ix<U.V; ++ix) {
		D_matrix.block<3,3>(3*ix,3*U.iup(ix,0)) = mu_I * 0.25 * mu_I_plus_factor * U[ix][0];
		D_matrix.block<3,3>(3*ix,3*U.idn(ix,0)) = mu_I * 0.25 * mu_I_minus_factor * U.dn(ix,0)[0].adjoint();
	}
	// undo flip to restore original U's
	apbs_in_time(U);
	return D_matrix;
}

Eigen::MatrixXcd dirac_op::D_eigenvalues (field<gauge>& U, double mass, double mu_I) const {
	// construct explicit dense dirac matrix
	Eigen::MatrixXcd D_matrix = D_dense_matrix(U, mass, mu_I);
	// find all eigenvalues of non-hermitian dirac operator matrix D
	Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
	ces.compute(D_matrix);
	return ces.eigenvalues();
}

Eigen::MatrixXcd dirac_op::DDdagger_eigenvalues (field<gauge>& U, double mass, double mu_I) const {
	// construct explicit dense dirac matrix
	Eigen::MatrixXcd D_matrix = D_dense_matrix(U, mass, mu_I);
	D_matrix = D_matrix * D_matrix.adjoint();
	// find all eigenvalues of hermitian dirac operator matrix DDdagger
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> saes;
	saes.compute(D_matrix);
	return saes.eigenvalues();
}

double dirac_op::D_phase_angle (field<gauge>& U, double mass, double mu_I) const {
	Eigen::MatrixXcd eigenvalues = D_eigenvalues(U, mass, mu_I);				
	// phase{Det[D]} = phase{\prod_i \lambda_i} = \sum_i phase{lambda_i}
	double sum = 0;
	for (int i=0; i<eigenvalues.size(); ++i) {
		sum += std::arg(eigenvalues(i));
	}
	return sum;
}

double dirac_op::pion_susceptibility_exact (field<gauge>& U, double mass, double mu_I) const {
	Eigen::MatrixXcd eigenvaluesDDdag = DDdagger_eigenvalues (U, mass, mu_I);				
	return eigenvaluesDDdag.cwiseInverse().sum().real()/static_cast<double>(3*U.V);
}

void dirac_op::D (field<fermion>& lhs, const field<fermion>& rhs, field<gauge>& U, double mass, double mu_I) const {
	// flip sign of timelike U's at boundary to impose anti-periodic bcs on fermions 
	apbs_in_time(U);

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

// thin QR decomposition of M using Algorithm 2 from arXiv:1710.09745
// input: M is a vector of N fermion fields
// output: R is a NxN complex matrix
// output: Q is a vector of N orthogonal fermion fields
void dirac_op::thinQR(std::vector<field<fermion>>& Q, Eigen::MatrixXcd& R, std::vector<field<fermion>>& M) {
	int N = M.size();
	// Construct H_ij = M_i^dag M_j = hermitian, 
	// so only need to do the dot products needed for lower triangular part of matrix
	Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(N, N);
	for(int i=0; i<N; ++i) {
		for(int j=0; j<=i; ++j) {
			H(i, j) = M[i].dot(M[j]);
		}
	}

	// Find upper triangular R such that R^dag R = H = M^dag M
	// i.e. adjoint of cholesky decomposition L matrix: L L^dag = H
	R = H.llt().matrixL().adjoint();
	// Solve QR = M for Q, where R is upper triangular
	Q = M;
	for(int i=0; i<N; ++i) {
		for(int j=0; j<i; ++j) {
			Q[i].add(-R(j,i), Q[j]);
		}
		Q[i] /= R(i,i);
	}
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
	// b_norm = sqrt(<b|b>)
	double b_norm = sqrt(b.squaredNorm());
	// r2_new = <r|r>
	r2_new = r.squaredNorm();
	while (sqrt(r2_new)/b_norm > eps)
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
		//std::cout << iter << "\t" << sqrt(r2_new)/b_norm << std::endl;
		if(iter>1e5)
		{
			std::cout << "CG not converging: iter= " << iter << " residual= " << sqrt(r2_new)/b_norm << std::endl; 
			exit(1); 
		}
	}
	return iter;
}

int dirac_op::cg_singleshift(field<fermion>& x, const field<fermion>& b, field<gauge>& U, double mass, double mu_I, double shift, double eps) {
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
	// b_norm = sqrt(<b|b>)
	double b_norm = sqrt(b.squaredNorm());
	// r2_new = <r|r>
	r2_new = r.squaredNorm();
	while (sqrt(r2_new)/b_norm > eps)
	{
		// a = (A + shift) p
		DDdagger(a, p, U, mass, mu_I);
		a.add(shift, p);
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
		if(iter>1e5)
		{
			std::cout << "cg_singleshift not converging: iter= " << iter << " residual= " << sqrt(r2_new)/b_norm << std::endl; 
			exit(1); 
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
	// b_norm = sqrt(<b|b>)
	double b_norm = sqrt(b.squaredNorm());
	// r2_new = <r|r>
	r2_new = r.squaredNorm();
	while (sqrt(r2_new)/b_norm > eps)
	{
		// a = A p_0
		DDdagger(a, p[0], U, mass, mu_I);
		// add first shift to DDdagger explicitly here:
		a.add(sigma[0], p[0]);
		++iter;
		r2_old = r2_new;
		// beta = -<r|r>/<p|a>
		beta[0] = -r2_old / p[0].dot(a);
		// calculate zeta and beta coefficients for shifted vectors
		// see arXiv:hep-lat/9612014 for derivation
		for(int i_shift=1; i_shift<n_shifts; ++i_shift) {
			zeta_p1[i_shift] = zeta[i_shift] * zeta_m1[i_shift] * beta_m1[0];
			zeta_p1[i_shift] /= (beta[0]*alpha[0]*(zeta_m1[i_shift]-zeta[i_shift]) + zeta_m1[i_shift]*beta_m1[0]*(1.0 - (sigma[i_shift]-sigma[0])*beta[0]));
			beta[i_shift] = beta[0] * zeta_p1[i_shift] / zeta[i_shift];
			if(beta[i_shift] != beta[i_shift]) {
				// beta is NaN, i.e. we have converged to machine precision for this shift,
				// so stop updating it
				if(i_shift+1 == n_shifts) {
					// If shifts are in increasing order, we can just reduce n_shifts by one
					// to stop updating this shift
					--n_shifts;
				}
				else {
					std::cout << "NaN in cg_multishift for i_shift: " << i_shift << std::endl;
					std::cout << "But n_shifts = " << n_shifts << "; check that shifts are in increasing order!" << std::endl;
					exit(1);
				}
			}
//			std::cout << i_shift << "\t" << beta[i_shift] << std::endl;
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
		if(iter>1e5)
		{
			std::cout << "CG-multishift not converging: iter= " << iter << " residual= " << sqrt(r2_new)/b_norm << std::endl;
			exit(1); 
		}
	}
	return iter;	
}

int dirac_op::cg_block(std::vector<field<fermion>>& X, const std::vector<field<fermion>>& B, field<gauge>& U, double mass, double mu_I, double eps) {
	int N = static_cast<int>(B.size());
	// S = 1 [NxN]
	Eigen::MatrixXcd S = Eigen::MatrixXcd::Identity(N, N);
	// C = 0 [NxN]
	Eigen::MatrixXcd C = Eigen::MatrixXcd::Zero(N, N);
	// beta = 0 [NxN]
	Eigen::MatrixXcd beta = Eigen::MatrixXcd::Zero(N, N);
	// betaC = 0 [NxN]
	Eigen::MatrixXcd betaC = Eigen::MatrixXcd::Zero(N, N);
	// AP, P, Q, R are [NxVOL]
	std::vector<field<fermion>> AP, P, R, Q, tmpQ;
	for(int i=0; i<N; ++i) {
 		AP.push_back(field<fermion>(grid));
 		P.push_back(field<fermion>(grid));
 		Q.push_back(field<fermion>(grid));
 		tmpQ.push_back(field<fermion>(grid));
	}
	// start from X=0 initial guess, so R = B [NxVOL]
	R = B;
	// QR decomposition of R[N][VOL] into Q[N][VOL] and C[NxN]
	thinQR(Q, C, R);
	// P = 0 [NxVOL]
	for(int i=0; i<N; ++i) {
		P[i].setZero(); 
	}

	// main loop
	int iter = 0;
	// get norm of each vector b in matrix B,
	// which for X=0 is also the norm of initial residual vectors r in matrix R:
	// NB: v.norm() is sqrt(\sum_i v_i v_i^*) = l2 norm of vector v
	// or sqrt(\sum_i\sum_j v_ij v_ij^*) = frobenius norm of matrix
	Eigen::ArrayXd b_norm = C.rowwise().norm().array();
	double residual = 1.0;
	while(residual > eps) {
		// P <- Q + P S^dag:
		// S is upper triangular, so S^dag is lower triangular
		// so if we do it in the right order no aliasing required
		for(int i=0; i<N; ++i) {
			P[i] *= std::conj(S(i,i));
			for(int j=i+1; j<N; ++j) {
				P[i].add(std::conj(S(i,j)), P[j]);
			}
			P[i] += Q[i];
		}

		// Apply dirac op to P:
		for(int i=0; i<N; ++i) {
			DDdagger(AP[i], P[i], U, mass, mu_I);
			++iter;
		}
		// construct NxN beta matrix:
		// beta^-1 = P^dag AP [NxN hermitian matrix]
		// note this is hermitian since A is hermitian we
		// only need to calculate lower triangular elements of matrix
		for(int i=0; i<N; ++i) {
			for(int j=0; j<=i; ++j) {
				beta(i,j) = P[i].dot(AP[j]);
			}
		}
		// Find inverse of beta via cholesky decomposition
		// and solving beta beta^-1 = I
		beta = beta.llt().solve(Eigen::MatrixXcd::Identity(N, N));

		// TODO: output condition number of C, maybe also beta?

		// X = X + P beta C
		betaC = beta * C;
		for(int i=0; i<N; ++i) {
			for(int j=0; j<N; ++j) {
				X[i].add(betaC(j,i), P[j]);
			}
		}

		//tmpQ = Q - AP beta
		tmpQ = Q;
		for(int i=0; i<N; ++i) {
			for(int j=0; j<N; ++j) {
				tmpQ[i].add(-beta(j,i), AP[j]);
			}
		}

		//QS = tmpQ
		thinQR(Q, S, tmpQ);

		C = S * C;
		// use maximum over vectors of residual/b_norm as stopping crit
		// worst vector should be equal to CG with same eps on that vector, others will be better
		residual = (C.rowwise().norm().array()/b_norm).maxCoeff();

		// [debugging] find eigenvalues of C
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> saes;
		saes.compute(C);
		Eigen::ArrayXd evals = saes.eigenvalues();
		std::cout << "#BLOCK-CG " << iter << "\t" << residual << "\t" << evals.maxCoeff()/evals.minCoeff() << "\t" << evals.matrix().transpose() << std::endl;
	}
	return iter;
}