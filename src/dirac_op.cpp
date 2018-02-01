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

// in-place thin QR decomposition of Q using Algorithm 2 from arXiv:1710.09745
// input: Q is a vector of N fermion fields
// output: R is a NxN complex hermitian matrix such that Q_input R = Q_output
// output: Q is a now a vector of N orthogonal fermion fields
void dirac_op::thinQR(std::vector<field<fermion>>& Q, Eigen::MatrixXcd& R) {
	int N = Q.size();
	// Construct H_ij (using R for storage) = Q_i^dag Q_j = hermitian, 
	// so only need to do the dot products needed for lower triangular part of matrix
	R.setZero();
	for(int i=0; i<N; ++i) {
		for(int j=0; j<=i; ++j) {
			R(i, j) = Q[i].dot(Q[j]);
		}
	}	
	// Find upper triangular R such that R^dag R = H (i.e. previous contents of R) = Q^dag Q
	// i.e. adjoint of cholesky decomposition L matrix: L L^dag = H
	R = R.llt().matrixL().adjoint();
	// Solve Q_new R = Q for Q_new, where R is upper triangular
	for(int i=0; i<N; ++i) {
		for(int j=0; j<i; ++j) {
			Q[i].add(-R(j,i), Q[j]);
		}
		Q[i] /= R(i,i);
	}
}

// in-place A-orthonormalisation of V and AV:
// input/output: V is a vector of N fermion fields
// input/ouput: AV is the result of hermitian matrix A acting on V
// output: R is upper triangular such that V_new R = V_old, AV_new R = AV_old
// we want PR = V, and P^dag AP = I, with upper triangular R,
// so [V^dag AV] = R^dag [P^dag AP] R = R^dag R i.e.
// can do cholesky decomposition to find R, then back substitution:
// V <- P = V R^-1
// AV <- AP = AV R^-1
void dirac_op::thinQRA(std::vector<field<fermion>>& V, std::vector<field<fermion>>& AV, Eigen::MatrixXcd& R) {
	int N = V.size();
	// Construct [V^dag AV]_ij (using R for storage) - hermitian, 
	// so only need to do the dot products needed for lower triangular part of matrix
	R.setZero();
	for(int i=0; i<N; ++i) {
		for(int j=0; j<=i; ++j) {
			R(i, j) = V[i].dot(AV[j]);
		}
	}
	// Find upper triangular R such that R^dag R = [V^dag AV]
	// i.e. adjoint of cholesky decomposition L matrix
	R = R.llt().matrixL().adjoint();
	// Solve V_new R = V and AV_new R = AV where R is upper triangular
	for(int i=0; i<N; ++i) {
		for(int j=0; j<i; ++j) {
			V[i].add(-R(j,i), V[j]);
			AV[i].add(-R(j,i), AV[j]);
		}
		V[i] /= R(i,i);
		AV[i] /= R(i,i);
	}
}

// Bartels–Stewart: O(N^3) algorithm to solve Sylvester equation for X:
// AX + XB = C
// (NB there also exists the Hessenberg-Schur method that only requires one Schur decomposition)
void dirac_op::bartels_stewart(Eigen::MatrixXcd& X, const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B, const Eigen::MatrixXcd& C) {
	int N = X.rows();
	// Compute Schur form U T U^dag of A and B
	Eigen::ComplexSchur<Eigen::MatrixXcd> schurA(A);
	Eigen::ComplexSchur<Eigen::MatrixXcd> schurB(B);

	// Can now transform equation to the form
	// TA X_tilde + X_tilde TB = C_tilde
	// where TA, TB are upper triangular,
	// X_tilde = UA^dag X UB
	// C_tilde = UA^dag C UB
	Eigen::MatrixXcd C_tilde = schurA.matrixU().adjoint() * C * schurB.matrixU();
	Eigen::MatrixXcd X_tilde = Eigen::MatrixXcd::Zero(N, N);

	// Solve triangular system by back substitution
	// consider k-th vector of transformed equation
	for(int k=0; k<N; ++k) {
		// do subtractions from C_tilde
		Eigen::MatrixXcd C_sub = C_tilde.col(k);
		for(int l=0; l<k; ++l) {
			C_sub -= schurB.matrixT()(l,k) * X_tilde.col(l);
		}
		// do back substitution to solve for X_tilde^(k)
		for(int i=N-1; i>=0 ; --i) {
			X_tilde(i,k) = C_sub(i);
			for(int j=i+1; j<N; j++) {
				X_tilde(i,k) -= schurA.matrixT()(i,j) * X_tilde(j,k);
			}
			X_tilde(i,k) /= (schurA.matrixT()(i,i) + schurB.matrixT()(k,k));
		} 
	}
	// Transform solution X_tilde back to X
	X = schurA.matrixU() * X_tilde * schurB.matrixU().adjoint();
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

int dirac_op::cg_block(std::vector<field<fermion>>& X, const std::vector<field<fermion>>& B, field<gauge>& U, double mass, double mu_I, double eps, bool BCGA, bool dQ, bool dQA, bool rQ, const field<fermion>& x0_star) {
	int N = static_cast<int>(B.size());
	// S = 1 [NxN]
	Eigen::MatrixXcd S = Eigen::MatrixXcd::Identity(N, N);
	// C = 1 [NxN]
	Eigen::MatrixXcd C = Eigen::MatrixXcd::Identity(N, N);
	// beta = 0 [NxN]
	Eigen::MatrixXcd beta = Eigen::MatrixXcd::Zero(N, N);
	// betaC = 0 [NxN]
	Eigen::MatrixXcd betaC = Eigen::MatrixXcd::Zero(N, N);
	// R = 1 [NxN]
	Eigen::MatrixXcd R = Eigen::MatrixXcd::Identity(N, N);
	// PQ, PAQ (for BCGA dot products) [NxN]
	Eigen::MatrixXcd mPAPinv = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd mPQ = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd mPAQ = Eigen::MatrixXcd::Identity(N, N);
	// AP, P, Q are [NxVOL]
	std::vector<field<fermion>> AP, P, Q;
	for(int i=0; i<N; ++i) {
 		AP.push_back(field<fermion>(grid));
 		P.push_back(field<fermion>(grid));
 		Q.push_back(field<fermion>(grid));
	}
	// for debugging (error norms of first vector):
	field<fermion> tmpE0(grid), tmpAE0(grid);
	// get error norms for X=0 to normalise all to 1 intially
	double norm0_x0_star = sqrt(x0_star.squaredNorm());
	DDdagger(tmpAE0, x0_star, U, mass, mu_I);
	double norm1_x0_star = sqrt(x0_star.dot(tmpAE0).real());
	// note norm2 is just the residual so we already have the normalisation

	// start from X=0 initial guess, so residual Q = B [NxVOL]
	Q = B;
	if(rQ) {
		// in place thinQR decomposition of residual Q[N][VOL] into orthonormal Q[N][VOL] and triangular C[NxN]
		thinQR(Q, C);
	} else if(BCGA) {
		// set diagonal values of C to residuals Q^dag Q, only needed for residual stopping criterion
		for(int i=0; i<N; ++i) {
			C(i, i) = Q[i].dot(Q[i]);
		}
	} else {
		// set C to hermitian matrix Q^dag Q
		for(int i=0; i<N; ++i) {
			for(int j=0; j<=i; ++j) {
				C(i, j) = Q[i].dot(Q[j]);
				C(j, i) = conj(C(i, j));
			}
		}
		// S = C_old^-1 C_new = C since C_old=1.
		S = C;
	}

	// P = 0 [NxVOL]
	for(int i=0; i<N; ++i) {
		P[i].setZero(); 
	}

	// main loop
	int iter = 0;
	// get norm of each vector b in matrix B
	// NB: v.norm() == sqrt(\sum_i v_i v_i^*) = l2 norm of vector v
	Eigen::ArrayXd b_norm = Eigen::ArrayXd::Zero(N);
	if(rQ) {
		// residual_i = sqrt(\sum_j Q_i^dag Q_j) = sqrt(\sum_j C_ij)
		b_norm = C.rowwise().norm().array();
	} else {
		// residual_i = sqrt(Q_i^dag Q_i) = sqrt(C_ii)
		b_norm = C.diagonal().real().array().sqrt();		
	}
	double residual = 1.0;
	while(residual > eps) {

		// P <- Q + PS
		for(int i=0; i<N; ++i) {
			AP[i] = Q[i];
			for(int j=0; j<N; ++j) {
				AP[i].add((S(j,i)), P[j]);
			}
		}
		P = AP;

		/*
		// NOTE: for special case of BCGrQ S is triangular
		// so we could do above P <- Q + P S without temporary storage
		// and with less opertions:
		if(rQ and !dQ) {
			// P <- Q + P S or lower triangular S
			for(int i=0; i<N; ++i) {
				P[i] *= S(i,i);
				for(int j=i+1; j<N; ++j) {
					P[i].add(S(j,i), P[j]);
				}
				P[i] += Q[i];
			}
		}
		*/

		if(dQ) {
			// in-place thinQR decomposition of descent matrix P
			// such that P^dag P = I
			thinQR(P, R);
		}

		// Apply dirac op to P:
		for(int i=0; i<N; ++i) {
			DDdagger(AP[i], P[i], U, mass, mu_I);
			++iter;
		}

		if(dQA) {
			// in-place thinQRA decomposition of descent matrix P and AP
			// such that P^dag AP = I
			thinQRA(P, AP, R);			
		}

		// construct NxN beta matrix:
		// beta^-1 = P^dag AP [NxN hermitian matrix]
		if(dQA) {
			// P^dag AP = I by construction
			beta = Eigen::MatrixXcd::Identity(N, N);
		} else {
			// note beta is hermitian since A is hermitian so we
			// only need to calculate lower triangular elements of matrix
			for(int i=0; i<N; ++i) {
				for(int j=0; j<=i; ++j) {
					beta(i,j) = P[i].dot(AP[j]);
				}
			}
			// Find inverse of beta via LDLT cholesky decomposition
			// and solving beta beta^-1 = I
			beta = beta.ldlt().solve(Eigen::MatrixXcd::Identity(N, N));
		}

		if((dQ || dQA) && !BCGA) {
			// beta <- beta (R^dag)^-1
			// Solve X (R^dag) = beta for X, then beta <- X
			// Can be done in-place by back-substitution since R is upper-triangular
			//std::cout << "beta\n" << beta << std::endl;
			for(int i=0; i<N; ++i) {
				for(int j=N-1; j>=0; --j) {
					for(int k=N-1; k>j; --k) {
						beta(i,j) -= beta(i,k) * conj(R(j,k));
					}
					beta(i,j) /= R(j,j);
				}
			}
			//std::cout << "new beta\n" << beta << std::endl;
			//std::cout << "beta R^dag\n" << beta*R.adjoint() << std::endl;
		}

		if(BCGA) {
			mPAPinv = beta;
			for(int i=0; i<N; ++i) {
				for(int j=0; j<N; ++j) {
					mPQ(i, j) = P[i].dot(Q[j]);
				}
			}
			beta = beta * mPQ;
			if(rQ) {
				betaC = beta * C;
			} else {
				betaC = beta;
			}			
		} else {
			betaC = beta * C;
			if(!rQ) {
				beta = betaC;
			}
		}
		// X = X + P beta C
		for(int i=0; i<N; ++i) {
			for(int j=0; j<N; ++j) {
				X[i].add(betaC(j,i), P[j]);
			}
		}

		//Q -= AP beta
		for(int i=0; i<N; ++i) {
			for(int j=0; j<N; ++j) {
				Q[i].add(-beta(j,i), AP[j]);
			}
		}

		if(BCGA) {
			if(rQ) {
				// in-place thinQR decomposition of residuals matrix Q
				thinQR(Q, S);
				C = S * C;
			} else {
				// update diagonal values of C for residual
				for(int i=0; i<N; ++i) {
					C(i, i) = Q[i].dot(Q[i]);
				}				
			}
			// S <- -[P^dag AP]^-1 [Q^dag AP] = - PQ [Q^dag AP]
			for(int i=0; i<N; ++i) {
				for(int j=0; j<N; ++j) {
					mPAQ(i, j) = Q[i].dot(AP[j]);
				}
			}
			S = -mPAPinv * mPAQ.adjoint();
		} else {
			if(rQ) {
				// in-place thinQR decomposition of residuals matrix Q
				thinQR(Q, S);
				C = S * C;
				// S <- S^dag:
				S.adjointInPlace();
			} else {
				// find inverse of C = Q_old^dagQ_old by cholesky decomposition
				S = C.ldlt().solve(Eigen::MatrixXcd::Identity(N, N));
				// update C to hermitian matrix Q^dag Q
				for(int i=0; i<N; ++i) {
					for(int j=0; j<=i; ++j) {
						C(i, j) = Q[i].dot(Q[j]);
						C(j, i) = conj(C(i, j));
					}
				}
				// S = [Q_old^dag Q_old]^-1 [Q_new^dag Q_new]
				S = S * C;
			}
			if(dQ || dQA) {
				// S <- RS:
				S = R * S;
			}
		}

		// use maximum over vectors of residual/b_norm as stopping crit
		// worst vector should be equal to CG with same eps on that vector, others will be better
		if(rQ) {
			residual = (C.rowwise().norm().array()/b_norm).maxCoeff();
		}
		else {
			// C_ii = Q_i^dag Q_i
			residual = (C.diagonal().real().array().sqrt()/b_norm).maxCoeff();
		}

		// debugging: use known solution to get error norms for first block vector
		tmpE0 = X[0];
		tmpE0.add(-1.0, x0_star);
		DDdagger(tmpAE0, tmpE0, U, mass, mu_I);
		double norm0 = sqrt(tmpE0.squaredNorm())/norm0_x0_star;
		double norm1 = sqrt(tmpE0.dot(tmpAE0).real())/norm1_x0_star;
		double norm2 = sqrt(tmpAE0.squaredNorm())/b_norm[0];
		std::cout << "#Error-norms <(x-x*)|(1,sqrt(A),A)|(x-x*)> " << norm0 << "\t" << norm1 << "\t" << norm2 << std::endl;

		// [debugging] find eigenvalues of C
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> saes;
		saes.compute(C);
		Eigen::ArrayXd evals = saes.eigenvalues();
		std::cout << "#BCG";
		if(BCGA) {
			std::cout << "A";
		}
		if(dQ) {
			std::cout << "dQ";
		}
		if(dQA) {
			std::cout << "dQA";
		}
		if(rQ) {
			std::cout << "rQ";
		}
		std::cout << " " << iter << "\t" << residual << "\t" << evals.maxCoeff()/evals.minCoeff() << std::endl;
		// "\t" << evals.matrix().transpose() << std::endl;
	}
	return iter;
}