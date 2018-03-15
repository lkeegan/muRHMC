#include "inverters.hpp"
#include "omp.h"
#include <iostream> //FOR DEBUGGING

// in-place thin QR decomposition of Q using Algorithm 2 from arXiv:1710.09745
// input: Q is a vector of N fermion fields
// output: R is a NxN complex hermitian matrix such that Q_input R = Q_output
// output: Q is a now a vector of N orthogonal fermion fields
void thinQR(std::vector<field<fermion>>& Q, Eigen::MatrixXcd& R) {
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
void thinQRA(std::vector<field<fermion>>& V, std::vector<field<fermion>>& AV, Eigen::MatrixXcd& R) {
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

// A-orthogonalise X in place, without having AX already, i.e. does a bunch of DDdag operations
// Return the eigenvalues of the hermitian matrix <X_i|A|X_j> in first column of Evals [eigenvalues]
// And the square root of the diagonal elemets only of the hermitian matrix <X_i|A^2|X_j> in 2nd [optimistic error estimates]
// (Does not calculate or return the square root of the largest eigenvalue of <X_i|A^2|X_j> [conservative error estimate])
void thinQRA_evals(std::vector<field<fermion>>& X, Eigen::MatrixXd& Evals, field<gauge>& U, dirac_op& D) {
	int N = X.size();
	field<fermion> x (X[0].grid, X[0].eo_storage);
	// Construct lower triangular part of hermitian matrix <X_i|A|X_j> and diagonal part of <X_i|A^2|X_j> 
	Eigen::MatrixXcd R = Eigen::MatrixXcd::Zero(N, N);
	for(int i=0; i<N; ++i) {
		// x = A X[i]
		D.DDdagger (x, X[i], U);
		Evals.col(1)[i] = x.norm();
		for(int j=0; j<=i; ++j) {
			R(i, j) = x.dot(X[j]);
		}
	}
	//find eigensystem of R - only references lower triangular part
	//NB also finding eigenvectors here, but no need to..
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> R_eigen_system(R);
	Evals.col(0) = R_eigen_system.eigenvalues().col(0);
	// A-orthonormalise X
	R = R.llt().matrixL().adjoint();
	for(int i=0; i<N; ++i) {
		for(int j=0; j<i; ++j) {
			X[i].add(-R(j,i), X[j]);
		}
		X[i] /= R(i,i);
	}
}

int cg(field<fermion>& x, const field<fermion>& b, field<gauge>& U, dirac_op& D, double eps) {
	field<fermion> a(b.grid, b.eo_storage), p(b.grid, b.eo_storage), r(b.grid, b.eo_storage);
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
	double b_norm = b.norm();
	// r2_new = <r|r>
	r2_new = r.squaredNorm();
	while (sqrt(r2_new)/b_norm > eps)
	{
		// a = A p
		D.DDdagger(a, p, U);
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
			//exit(1); 
			return iter;
		}
	}
	return iter;
}

int cg_singleshift(field<fermion>& x, const field<fermion>& b, field<gauge>& U, double shift, dirac_op& D, double eps) {
	field<fermion> a(b.grid, b.eo_storage), p(b.grid, b.eo_storage), r(b.grid, b.eo_storage);
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
	double b_norm = b.norm();
	// r2_new = <r|r>
	r2_new = r.squaredNorm();
	while (sqrt(r2_new)/b_norm > eps)
	{
		// a = (A + shift) p
		D.DDdagger(a, p, U);
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

int cg_multishift(std::vector<field<fermion>>& x, const field<fermion>& b, field<gauge>& U, std::vector<double>& sigma, dirac_op& D, double eps, double eps_shifts) {
	int n_shifts = x.size();
	int n_unconverged_shifts = n_shifts;
	std::vector<field<fermion>> p;
	field<fermion> a(b.grid, b.eo_storage), r(b.grid, b.eo_storage);
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
	double b_norm = b.norm();
	// r2_new = <r|r>
	r2_new = r.squaredNorm();
	//std::cout << "INIT res " << sqrt(r2_new)/b_norm << " x00: " << x[0][0](0) << std::endl;	
	while (sqrt(r2_new)/b_norm > eps)
	{
		// a = A p_0
		D.DDdagger(a, p[0], U);
		// add first shift to DDdagger explicitly here:
		a.add(sigma[0], p[0]);
		++iter;
		r2_old = r2_new;
		// beta = -<r|r>/<p|a>
		beta[0] = -r2_old / p[0].dot(a);
		// calculate zeta and beta coefficients for shifted vectors
		// see arXiv:hep-lat/9612014 for derivation
		for(int i_shift=1; i_shift<n_unconverged_shifts; ++i_shift) {
			zeta_p1[i_shift] = zeta[i_shift] * zeta_m1[i_shift] * beta_m1[0];
			zeta_p1[i_shift] /= (beta[0]*alpha[0]*(zeta_m1[i_shift]-zeta[i_shift]) + zeta_m1[i_shift]*beta_m1[0]*(1.0 - (sigma[i_shift]-sigma[0])*beta[0]));
			beta[i_shift] = beta[0] * zeta_p1[i_shift] / zeta[i_shift];
		}
		// if largest shift has converged (to machine precision), i.e. normalised residual < ulp, stop updating it
		if(sqrt((zeta[n_unconverged_shifts-1]*conj(zeta[n_unconverged_shifts-1])).real()*r2_old)/b_norm < eps_shifts) {
			--n_unconverged_shifts;
			//std::cout << "iter " << iter << "converged " << n_unconverged_shifts+1 << std::endl;
		}
		// x_i -= beta_i * p_i
		for(int i_shift=0; i_shift<n_unconverged_shifts; ++i_shift) {
			x[i_shift].add(-beta[i_shift], p[i_shift]);			
		}
		//std::cout << "iter " << iter << "res " << sqrt(r2_new)/b_norm << " x00: " << x[0][0](0) << std::endl;
		// r += beta_0 a
		r.add(beta[0], a);
		// r2_new = <r|r>
		r2_new = r.squaredNorm();
		// increment timestep:
		// X_m1 <- X
		// X <- X_p1
		alpha[0] = r2_new / r2_old;
		beta_m1[0] = beta[0];
		for(int i_shift=1; i_shift<n_unconverged_shifts; ++i_shift) {
			beta_m1[i_shift] = beta[i_shift];
			zeta_m1[i_shift] = zeta[i_shift];
			zeta[i_shift] = zeta_p1[i_shift];
		}
		// calculate alpha coeffs for shifted vectors
		for(int i_shift=1; i_shift<n_unconverged_shifts; ++i_shift) {
			alpha[i_shift] = alpha[0] * (zeta[i_shift] * beta_m1[i_shift]) / (zeta_m1[i_shift] * beta_m1[0]);
		}
		// p_i = alpha_i p_i + zeta_i r
		for(int i_shift=0; i_shift<n_unconverged_shifts; ++i_shift) {
			p[i_shift].scale_add(alpha[i_shift], zeta[i_shift], r);
		}
		if(iter>1e5)
		{
			std::cout << "CG-multishift not converging: iter= " << iter << " residual= " << sqrt(r2_new)/b_norm << std::endl;
			exit(1); 
		}
	}
	//std::cout << "CONV iter " << iter << "res " << sqrt(r2_new)/b_norm << " x00: " << x[0][0](0) << std::endl;
	return iter;	
}

// return rational approx x = \alpha_0 + \sum_i \alpha_{i+1} [A + \beta_i]^{-1} b using cg_multishift for inversions
// avoids storing shifted solution vectors, instead does above sum for every CG iteration and only stores result
// still need the shifted search vectors p of course 
int rational_approx_cg_multishift(field<fermion>& x, const field<fermion>& b, field<gauge>& U, std::vector<double>& rational_alpha, std::vector<double>& rational_beta, dirac_op& D, double eps) {
	int n_shifts = rational_beta.size();
	int n_unconverged_shifts = n_shifts;
	std::vector<field<fermion>> p;
	field<fermion> a(b.grid, b.eo_storage), r(b.grid, b.eo_storage);
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
		zeta[i_shift] = 1.0;
		zeta_m1[i_shift] = 1.0;
		alpha[i_shift] = 0.0;		
	}
	beta_m1[0] = 1.0;

	// Initial value for our x is \rational_alpha_0 b:
	x = b;
	x *= rational_alpha[0];

	// b_norm = sqrt(<b|b>)
	double b_norm = b.norm();
	// r2_new = <r|r>
	r2_new = r.squaredNorm();
	while (sqrt(r2_new)/b_norm > eps)
	{
		// a = A p_0
		D.DDdagger(a, p[0], U);
		// add first shift to DDdagger explicitly here:
		a.add(rational_beta[0], p[0]);
		++iter;
		r2_old = r2_new;
		// beta = -<r|r>/<p|a>
		beta[0] = -r2_old / p[0].dot(a);
		// calculate zeta and beta coefficients for shifted vectors
		// see arXiv:hep-lat/9612014 for derivation
		for(int i_shift=1; i_shift<n_unconverged_shifts; ++i_shift) {
			zeta_p1[i_shift] = zeta[i_shift] * zeta_m1[i_shift] * beta_m1[0];
			zeta_p1[i_shift] /= (beta[0]*alpha[0]*(zeta_m1[i_shift]-zeta[i_shift]) + zeta_m1[i_shift]*beta_m1[0]*(1.0 - (rational_beta[i_shift]-rational_beta[0])*beta[0]));
			beta[i_shift] = beta[0] * zeta_p1[i_shift] / zeta[i_shift];
		}
		// if largest shift has converged, i.e. normalised residual < eps, stop updating it
		if(sqrt((zeta[n_unconverged_shifts-1]*conj(zeta[n_unconverged_shifts-1])).real()*r2_old)/b_norm < eps) {
			--n_unconverged_shifts;
		}
		// usual CG solver would do: x_i -= beta_i * p_i
		// instead we do: x -= \sum_i \rational_alpha_i beta_i * p_i
		for(int i_shift=0; i_shift<n_unconverged_shifts; ++i_shift) {
			x.add(-rational_alpha[i_shift+1] * beta[i_shift], p[i_shift]);			
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
		for(int i_shift=1; i_shift<n_unconverged_shifts; ++i_shift) {
			beta_m1[i_shift] = beta[i_shift];
			zeta_m1[i_shift] = zeta[i_shift];
			zeta[i_shift] = zeta_p1[i_shift];
		}
		// calculate alpha coeffs for shifted vectors
		for(int i_shift=1; i_shift<n_unconverged_shifts; ++i_shift) {
			alpha[i_shift] = alpha[0] * (zeta[i_shift] * beta_m1[i_shift]) / (zeta_m1[i_shift] * beta_m1[0]);
		}
		// p_i = alpha_i p_i + zeta_i r
		for(int i_shift=0; i_shift<n_unconverged_shifts; ++i_shift) {
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

int cg_block(std::vector<field<fermion>>& X, const std::vector<field<fermion>>& B, field<gauge>& U, dirac_op& D, double eps, bool BCGA, bool dQ, bool dQA, bool rQ, const field<fermion>& x0_star) {
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
 		AP.push_back(field<fermion>(B[0].grid, B[0].eo_storage));
 		P.push_back(field<fermion>(B[0].grid, B[0].eo_storage));
 		Q.push_back(field<fermion>(B[0].grid, B[0].eo_storage));
	}
	// for debugging (error norms of first vector):
	field<fermion> tmpE0(B[0].grid, B[0].eo_storage), tmpAE0(B[0].grid, B[0].eo_storage);
	// get error norms for X=0 to normalise all to 1 intially
	double norm0_x0_star = sqrt(x0_star.squaredNorm());
	D.DDdagger(tmpAE0, x0_star, U);
	double norm1_x0_star = sqrt(x0_star.dot(tmpAE0).real());
	// note norm2 is just the residual so we already have the normalisation

	// start from X=0 initial guess, so residual Q = B [NxVOL]
	Q = B;
	for(int i=0; i<N; ++i) {
		X[i].setZero();
	}
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

		if(dQ) {
			// in-place thinQR decomposition of descent matrix P
			// such that P^dag P = I
			thinQR(P, R);
		}

		// Apply dirac op to P:
		for(int i=0; i<N; ++i) {
			D.DDdagger(AP[i], P[i], U);
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
		D.DDdagger(tmpAE0, tmpE0, U);
		
		double norm0 = sqrt(tmpE0.squaredNorm())/norm0_x0_star;
		double norm1 = sqrt(tmpE0.dot(tmpAE0).real())/norm1_x0_star;
		double norm2 = sqrt(tmpAE0.squaredNorm())/b_norm[0];
		std::cout << "#Error-norms <(x-x*)|(1,sqrt(A),A)|(x-x*)> " << iter << "\t" << norm0 << "\t" << norm1 << "\t" << norm2 << std::endl;
		/*
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
		*/
	}
	return iter;
}

int SBCGrQ(std::vector< std::vector< field<fermion> > >& X, const std::vector<field<fermion>>& B, field<gauge>& U, std::vector<double>& input_shifts, dirac_op& D, double eps, double eps_shifts) {
	int N = static_cast<int>(B.size());
	// count shifts (not including first one that is included in the dirac op)
	int N_shifts = static_cast<int>(input_shifts.size()) - 1;
	int N_unconverged_shifts = N_shifts;

	// Check shift parameters are consistent with size of supplied solution vector X:
	if (N_shifts+1 != static_cast<int>(X.size())) {
		std::cout << "Error in SBCGrQ: N_shifts+1 (" << N_shifts+1 << ") not consistent with size of X (" << static_cast<int>(X.size()) << ")" << std::endl;
		exit(1);
	}

	// subtract first shift from remaining set of shifts
	std::vector<double> shifts(N_shifts);
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
		shifts[i_shift] = input_shifts[i_shift+1] - input_shifts[0];
	}


	// std::cout << "#Shifted BCGrQ:\t" << std::endl;
	// std::cout << "#mass:\t" << D.mass << std::endl;
	// std::cout << "#input_shifts:\t";
	// for(int i_shift=0; i_shift<N_shifts+1; ++i_shift) {
	//		std::cout << "\t" << input_shifts[i_shift];
	// }
	// std::cout << std::endl;
	// std::cout << "#remaining_rescaled_shifts:\t";
	// for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
	// 		std::cout << "\t" << shifts[i_shift];
	// }
	// std::cout << std::endl;

	// Unshifted matrices:
	Eigen::MatrixXcd S = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd C = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd beta = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd betaC = Eigen::MatrixXcd::Zero(N, N);
	// AP, Q are [NxVOL]
	std::vector< field<fermion> > AP, Q;
	// start from X=0 initial guess, so residual Q = B [NxVOL]
	// X = 0 for unshifted + all shifts
	for(int i_shift=0; i_shift<N_shifts+1; ++i_shift) {
		for(int i=0; i<N; ++i) {
			X[i_shift][i].setZero();
		}
	}
	Q = B;
	// in place thinQR decomposition of residual Q[N][VOL] into orthonormal Q[N][VOL] and triangular C[NxN]
	thinQR(Q, C);
	S = C;
	AP = Q;
	// P has one NxVOL block per shift
	std::vector< std::vector< field<fermion> > > P;
	// P = Q for all shifts
	for(int i_shift=0; i_shift<N_shifts+1; ++i_shift) {
		P.push_back(Q);
	}

	// Shifted matrices:
	// previous / inverted versions of beta
	Eigen::MatrixXcd beta_inv = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd beta_inv_m1 = Eigen::MatrixXcd::Identity(N, N);
	// Construct decomposition of S: can then do S^-1 M using S_inv.solve(M)
	Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> S_inv (S);
	Eigen::MatrixXcd S_m1 = Eigen::MatrixXcd::Zero(N, N);
	// These are temporary matrices used for each shift
	Eigen::MatrixXcd tmp_betaC, tmp_Sdag; 
	// ksi_s are the scale factors related shifted and unshifted residuals
	// 3-term recurrence so need _k, _k-1, _k-2 for each shift
	// initially ksi_s_m2 = I, ksi_s_m1 = S
	std::vector<Eigen::MatrixXcd> ksi_s, ksi_s_m1, ksi_s_m2; 
	std::vector< Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> > ksi_s_inv_m1, ksi_s_inv_m2;
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
 		ksi_s.push_back(Eigen::MatrixXcd::Identity(N, N));
 		ksi_s_m1.push_back(S);
 		ksi_s_inv_m1.push_back(S_inv);
 		ksi_s_m2.push_back(Eigen::MatrixXcd::Identity(N, N));
 		ksi_s_inv_m2.push_back(Eigen::MatrixXcd::Identity(N, N).colPivHouseholderQr());
	}

	// main loop
	int iter = 0;
	// get norm of each vector b in matrix B
	// NB: v.norm() == sqrt(\sum_i v_i v_i^*) = l2 norm of vector v
	// residual_i = sqrt(\sum_j Q_i^dag Q_j) = sqrt(\sum_j C_ij)
	Eigen::ArrayXd b_norm = C.rowwise().norm().array();
	double residual = 1.0;
	while(residual > eps) {

		// Apply dirac op (with lowest shift) to P[0]:
		for(int i=0; i<N; ++i) {
			D.DDdagger(AP[i], P[0][i], U);
			// do lowest shift as part of dirac op
			AP[i].add(input_shifts[0], P[0][i]);
			++iter;
		}

		beta_inv_m1 = beta_inv;
		for(int i=0; i<N; ++i) {
			for(int j=0; j<=i; ++j) {
				beta_inv(i,j) = P[0][i].dot(AP[j]);
				beta_inv(j,i) = conj(beta_inv(i,j));
			}
		}
		// Find inverse of beta_inv via LDLT cholesky decomposition
		// and solving beta beta_inv = I
		beta = beta_inv.ldlt().solve(Eigen::MatrixXcd::Identity(N, N));
		betaC = beta * C;

		// X[0] = X[0] + P[0] beta C
		for(int i=0; i<N; ++i) {
			for(int j=0; j<N; ++j) {
				X[0][i].add(betaC(j,i), P[0][j]);
			}
		}

		//Q -= AP beta
		for(int i=0; i<N; ++i) {
			for(int j=0; j<N; ++j) {
				Q[i].add(-beta(j,i), AP[j]);
			}
		}

		S_m1 = S;
		// in-place thinQR decomposition of residuals matrix Q
		thinQR(Q, S);
		C = S * C;
		if(N_unconverged_shifts>0) {
			// update decomposition of S for S_inv operations
			S_inv.compute(S);
		}

		// P <- Q + P S^dag for lower triangular S
		for(int i=0; i<N; ++i) {
			P[0][i] *= S(i,i);
			for(int j=i+1; j<N; ++j) {
				P[0][i].add(conj(S(i,j)), P[0][j]);
			}
			P[0][i] += Q[i];
		}
		// calculate shifted X and P
		// note X[0] and P[0] are the unshifted ones, so first shift has index 1 in X and P
		for(int i_shift=0; i_shift<N_unconverged_shifts; ++i_shift) {
			// calculate shifted coefficients
			// ksi_s:
			tmp_betaC = S_m1 * beta_inv_m1 - ksi_s_m1[i_shift] * ksi_s_inv_m2[i_shift].solve(beta_inv_m1);
			tmp_Sdag = Eigen::MatrixXcd::Identity(N, N) + shifts[i_shift] * beta + tmp_betaC * S_m1.adjoint() * beta;
			ksi_s[i_shift] = S * tmp_Sdag.colPivHouseholderQr().solve(ksi_s_m1[i_shift]);
			// tmp_betaC == "alpha^sigma" in paper:
			tmp_betaC = beta * S_inv.solve(ksi_s[i_shift]);
			// tmp_Sdag == "beta^sigma" in paper:
			tmp_Sdag = tmp_betaC * ksi_s_inv_m1[i_shift].solve(beta_inv * S.adjoint());
			// update shifted X and P
			// X_s = X_s + P_s tmp_betaC
			for(int i=0; i<N; ++i) {
				for(int j=0; j<N; ++j) {
					X[i_shift+1][i].add(tmp_betaC(j,i), P[i_shift+1][j]);
				}
			}
			// P_s <- Q + P_s tmp_Sdag (using AP as temporary storage)
			for(int i=0; i<N; ++i) {
				AP[i] = Q[i];
				for(int j=0; j<N; ++j) {
					AP[i].add(tmp_Sdag(j,i), P[i_shift+1][j]);
				}
			}
			P[i_shift+1] = AP;
			// update inverse ksi's for next iteration
			ksi_s_m2[i_shift] = ksi_s_m1[i_shift];
			ksi_s_inv_m2[i_shift] = ksi_s_inv_m1[i_shift];
			ksi_s_m1[i_shift] = ksi_s[i_shift];
			ksi_s_inv_m1[i_shift].compute(ksi_s[i_shift]);
			// check if largest unconverged shift has converged
			if((ksi_s[N_unconverged_shifts-1].rowwise().norm().array()/b_norm).maxCoeff() < eps_shifts) {
				--N_unconverged_shifts;
			}
		}

		// use maximum over vectors of residuals/b_norm for unshifted solution as stopping crit
		// assumes that all shifts are positive i.e. better conditioned and converge faster
		// worst vector should be equal to CG with same eps on that vector, others will be better
		residual = (C.rowwise().norm().array()/b_norm).maxCoeff();

		// [debugging] find eigenvalues of C
		/*
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> saes;
		saes.compute(C);
		Eigen::ArrayXd evals = saes.eigenvalues();
		*/
			// Output iteration number and worst residual for each shift
		/*
		std::cout << "#SBCGrQ " << iter << "\t" << residual;
		for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
			std::cout << "\t" << (ksi_s[i_shift].rowwise().norm().array()/b_norm).maxCoeff();
		}
		std::cout << std::endl;
		*/
	}
	return iter;
}

int rational_approx_SBCGrQ(std::vector< field<fermion> >& X, const std::vector<field<fermion>>& B, field<gauge>& U, std::vector<double>& rational_alpha, std::vector<double>& rational_beta, dirac_op& D, double eps) {
	int N = static_cast<int>(B.size());
	// count shifts (not including first one that is included in the dirac op)
	int N_shifts = static_cast<int>(rational_beta.size()) - 1;
	int N_unconverged_shifts = N_shifts;

	// subtract first shift from remaining set of shifts
	std::vector<double> shifts(N_shifts);
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
		shifts[i_shift] = rational_beta[i_shift+1] - rational_beta[0];
	}

//	std::cout << "#Shifted BCGrQ:\t" << std::endl;
//	std::cout << "#mass:\t" << D.mass << std::endl;
//	std::cout << "#input_shifts:\t";
//	for(int i_shift=0; i_shift<N_shifts+1; ++i_shift) {
//			std::cout << "\t" << rational_beta[i_shift];
//	}
//	std::cout << std::endl;
//	std::cout << "#remaining_rescaled_shifts:\t";
//	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
//			std::cout << "\t" << shifts[i_shift];
//	}
//	std::cout << std::endl;

	// Unshifted matrices:
	Eigen::MatrixXcd S = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd C = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd beta = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd betaC = Eigen::MatrixXcd::Zero(N, N);
	// AP, Q are [NxVOL]
	std::vector< field<fermion> > AP, Q;
	// start from X=0 initial guess, so residual Q = B [NxVOL]
	Q = B;
	// then set X = \alpha_0 B
	X = B;
	for(int i=0; i<N; ++i) {
		X[i] *= rational_alpha[0];
	}
	// in place thinQR decomposition of residual Q[N][VOL] into orthonormal Q[N][VOL] and triangular C[NxN]
	thinQR(Q, C);
	S = C;
	AP = Q;
	// P has one NxVOL block per shift
	std::vector< std::vector< field<fermion> > > P;
	// P = Q for all shifts
	for(int i_shift=0; i_shift<N_shifts+1; ++i_shift) {
		P.push_back(Q);
	}

	// Shifted matrices:
	// previous / inverted versions of beta
	Eigen::MatrixXcd beta_inv = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd beta_inv_m1 = Eigen::MatrixXcd::Identity(N, N);
	// Construct decomposition of S: can then do S^-1 M using S_inv.solve(M)
	Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> S_inv (S);
	Eigen::MatrixXcd S_m1 = Eigen::MatrixXcd::Zero(N, N);
	// These are temporary matrices used for each shift
	Eigen::MatrixXcd tmp_betaC, tmp_Sdag; 
	// ksi_s are the scale factors related shifted and unshifted residuals
	// 3-term recurrence so need _k, _k-1, _k-2 for each shift
	// initially ksi_s_m2 = I, ksi_s_m1 = S
	std::vector<Eigen::MatrixXcd> ksi_s, ksi_s_m1, ksi_s_m2; 
	std::vector< Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> > ksi_s_inv_m1, ksi_s_inv_m2;
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
 		ksi_s.push_back(Eigen::MatrixXcd::Identity(N, N));
 		ksi_s_m1.push_back(S);
 		ksi_s_inv_m1.push_back(S_inv);
 		ksi_s_m2.push_back(Eigen::MatrixXcd::Identity(N, N));
 		ksi_s_inv_m2.push_back(Eigen::MatrixXcd::Identity(N, N).colPivHouseholderQr());
	}

	// main loop
	int iter = 0;
	// get norm of each vector b in matrix B
	// NB: v.norm() == sqrt(\sum_i v_i v_i^*) = l2 norm of vector v
	// residual_i = sqrt(\sum_j Q_i^dag Q_j) = sqrt(\sum_j C_ij)
	Eigen::ArrayXd b_norm = C.rowwise().norm().array();
	double residual = 1.0;
	while(residual > eps) {

		// Apply dirac op (with lowest shift) to P[0]:
		for(int i=0; i<N; ++i) {
			D.DDdagger(AP[i], P[0][i], U);
			// do lowest shift as part of dirac op
			AP[i].add(rational_beta[0], P[0][i]);
			++iter;
		}

		beta_inv_m1 = beta_inv;
		for(int i=0; i<N; ++i) {
			for(int j=0; j<=i; ++j) {
				beta_inv(i,j) = P[0][i].dot(AP[j]);
				beta_inv(j,i) = conj(beta_inv(i,j));
			}
		}
		// Find inverse of beta_inv via LDLT cholesky decomposition
		// and solving beta beta_inv = I
		beta = beta_inv.ldlt().solve(Eigen::MatrixXcd::Identity(N, N));
		betaC = beta * C;

		// X += alpha[1] P[0] beta C
		for(int i=0; i<N; ++i) {
			for(int j=0; j<N; ++j) {
				X[i].add(rational_alpha[1] * betaC(j,i), P[0][j]);
			}
		}

		//Q -= AP beta
		for(int i=0; i<N; ++i) {
			for(int j=0; j<N; ++j) {
				Q[i].add(-beta(j,i), AP[j]);
			}
		}

		S_m1 = S;
		// in-place thinQR decomposition of residuals matrix Q
		thinQR(Q, S);
		C = S * C;
		if(N_unconverged_shifts>0) {
			// update decomposition of S for S_inv operations
			S_inv.compute(S);
		}

		// P <- Q + P S^dag for lower triangular S
		for(int i=0; i<N; ++i) {
			P[0][i] *= S(i,i);
			for(int j=i+1; j<N; ++j) {
				P[0][i].add(conj(S(i,j)), P[0][j]);
			}
			P[0][i] += Q[i];
		}
		// calculate shifted X and P
		// note X[0] and P[0] are the unshifted ones, so first shift has index 1 in X and P
		for(int i_shift=0; i_shift<N_unconverged_shifts; ++i_shift) {
			// calculate shifted coefficients
			// ksi_s:
			tmp_betaC = S_m1 * beta_inv_m1 - ksi_s_m1[i_shift] * ksi_s_inv_m2[i_shift].solve(beta_inv_m1);
			tmp_Sdag = Eigen::MatrixXcd::Identity(N, N) + shifts[i_shift] * beta + tmp_betaC * S_m1.adjoint() * beta;
			ksi_s[i_shift] = S * tmp_Sdag.colPivHouseholderQr().solve(ksi_s_m1[i_shift]);
			// tmp_betaC == "alpha^sigma" in paper:
			tmp_betaC = beta * S_inv.solve(ksi_s[i_shift]);
			// tmp_Sdag == "beta^sigma" in paper:
			tmp_Sdag = tmp_betaC * ksi_s_inv_m1[i_shift].solve(beta_inv * S.adjoint());
			// update shifted X and P
			// rational approx solution X is linear sum
			// X += \alpha[i_shift+1] (P_s tmp_betaC)[i_shift]
			for(int i=0; i<N; ++i) {
				for(int j=0; j<N; ++j) {
					X[i].add(rational_alpha[i_shift+2] * tmp_betaC(j,i), P[i_shift+1][j]);
				}
			}
			// P_s <- Q + P_s tmp_Sdag (using AP as temporary storage)
			for(int i=0; i<N; ++i) {
				AP[i] = Q[i];
				for(int j=0; j<N; ++j) {
					AP[i].add(tmp_Sdag(j,i), P[i_shift+1][j]);
				}
			}
			P[i_shift+1] = AP;
			// update inverse ksi's for next iteration
			ksi_s_m2[i_shift] = ksi_s_m1[i_shift];
			ksi_s_inv_m2[i_shift] = ksi_s_inv_m1[i_shift];
			ksi_s_m1[i_shift] = ksi_s[i_shift];
			ksi_s_inv_m1[i_shift].compute(ksi_s[i_shift]);
			// check if largest unconverged shift has converged
			if((ksi_s[N_unconverged_shifts-1].rowwise().norm().array()/b_norm).maxCoeff() < eps) {
				--N_unconverged_shifts;
			}
		}

		// use maximum over vectors of residuals/b_norm for unshifted solution as stopping crit
		// assumes that all shifts are positive i.e. better conditioned and converge faster
		// worst vector should be equal to CG with same eps on that vector, others will be better
		residual = (C.rowwise().norm().array()/b_norm).maxCoeff();

		// [debugging] find eigenvalues of C
		/*
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> saes;
		saes.compute(C);
		Eigen::ArrayXd evals = saes.eigenvalues();
		*/
			// Output iteration number and worst residual for each shift
		/*
		std::cout << "#SBCGrQ " << iter << "\t" << residual;
		for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
			std::cout << "\t" << (ksi_s[i_shift].rowwise().norm().array()/b_norm).maxCoeff();
		}
		std::cout << std::endl;
		*/
	}
	return iter;
}

/*
int SBCGAdQArQ(std::vector< std::vector< field<fermion> > >& X, const std::vector<field<fermion>>& B, field<gauge>& U, std::vector<double>& input_shifts, dirac_op& D, double eps) {
	int N = static_cast<int>(B.size());
	// count shifts (not including first one that is just a mass rescaling in the dirac op)
	int N_shifts = static_cast<int>(input_shifts.size()) - 1;
	int N_unconverged_shifts = N_shifts;

	// Check shift parameters are consistent with size of supplied solution vector X:
	if (N_shifts+1 != static_cast<int>(X.size())) {
		std::cout << "Error in SBCGAdQArQ: N_shifts+1 (" << N_shifts+1 << ") not consistent with size of X (" << static_cast<int>(X.size()) << ")" << std::endl;
		exit(1);
	}

	// absorb first shift as rescaling of mass
	double mass = sqrt(input_mass*input_mass + input_shifts[0]);
	// subtract first shift from remaining set of shifts
	std::vector<double> shifts(N_shifts);
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
		shifts[i_shift] = input_shifts[i_shift+1] - input_shifts[0];
	}

	std::cout << "#Shifted BCGAdQArQ:\t" << input_mass << std::endl;
	std::cout << "#input_mass:\t" << input_mass << std::endl;
	std::cout << "#input_shifts:\t";
	for(int i_shift=0; i_shift<N_shifts+1; ++i_shift) {
			std::cout << "\t" << input_shifts[i_shift];
	}
	std::cout << std::endl;
	std::cout << "#rescaled_mass:\t" << mass << std::endl;
	std::cout << "#remaining_rescaled_shifts:\t";
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
			std::cout << "\t" << shifts[i_shift];
	}
	std::cout << std::endl;

	// Unshifted matrices:
	Eigen::MatrixXcd S = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd C = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd beta = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd betaC = Eigen::MatrixXcd::Zero(N, N);
	Eigen::MatrixXcd R = Eigen::MatrixXcd::Identity(N, N);
	// AP, Q are [NxVOL]
	std::vector< field<fermion> > AP, Q;
	// P has one NxVOL block per shift
	std::vector< std::vector< field<fermion> > > P;
	// start from X=0 initial guess, so residual Q = B [NxVOL]
	Q = B;
	// in place thinQR decomposition of residual Q[N][VOL] into orthonormal Q[N][VOL] and triangular C[NxN]
	thinQR(Q, C);
	S = C;
	AP = Q;
	P.push_back(Q);

	// Shifted matrices:
	// previous / inverted versions of beta
	Eigen::MatrixXcd beta_inv = Eigen::MatrixXcd::Identity(N, N);
	Eigen::MatrixXcd beta_inv_m1 = Eigen::MatrixXcd::Identity(N, N);
	// Construct decomposition of S: can then do S^-1 A using S_inv.solve(A)
	Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> S_inv (S);
	Eigen::MatrixXcd S_m1 = Eigen::MatrixXcd::Zero(N, N);
	// These are temporary matrices used for each shift
	Eigen::MatrixXcd tmp_betaC, tmp_Sdag; 
	// add a block to P for each shift s
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
		// P = Q initially for all shifts
 		P.push_back(Q);
	}
	// ksi_s are the scale factors related shifted and unshifted residuals
	// 3-term recurrence so need _k, _k-1, _k-2 for each shift
	// initially ksi_s_m2 = I, ksi_s_m1 = S
	std::vector<Eigen::MatrixXcd> ksi_s, ksi_s_m1, ksi_s_m2; 
	std::vector< Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> > ksi_s_inv_m1, ksi_s_inv_m2;
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
 		ksi_s.push_back(Eigen::MatrixXcd::Identity(N, N));
 		ksi_s_m1.push_back(S);
 		ksi_s_inv_m1.push_back(S_inv);
 		ksi_s_m2.push_back(Eigen::MatrixXcd::Identity(N, N));
 		ksi_s_inv_m2.push_back(Eigen::MatrixXcd::Identity(N, N).colPivHouseholderQr());
	}

	// X = 0 for unshifted + all shifts
	for(int i_shift=0; i_shift<N_shifts+1; ++i_shift) {
		for(int i=0; i<N; ++i) {
			X[i_shift][i].setZero();
		}
	}

	// main loop
	int iter = 0;
	// get norm of each vector b in matrix B
	// NB: v.norm() == sqrt(\sum_i v_i v_i^*) = l2 norm of vector v
	// residual_i = sqrt(\sum_j Q_i^dag Q_j) = sqrt(\sum_j C_ij)
	Eigen::ArrayXd b_norm = C.rowwise().norm().array();
	double residual = 1.0;
	while(residual > eps) {

		// P[0] <- Q + P[0] S (using AP as temporary storage)
		for(int i=0; i<N; ++i) {
			AP[i] = Q[i];
			for(int j=0; j<N; ++j) {
				AP[i].add(S(j,i), P[0][j]);
			}
		}
		P[0] = AP;

		// Apply dirac op (with lowest shift absorbed into mass) to P[0]:
		for(int i=0; i<N; ++i) {
			D.DDdagger(AP[i], P[0][i], U);
			++iter;
		}

		// in-place thinQRA decomposition of descent matrix P[0] and AP
		// such that P[0]^dag AP = I
		thinQRA(P[0], AP, R);			

		beta_inv_m1 = beta_inv;
		for(int i=0; i<N; ++i) {
			for(int j=0; j<=i; ++j) {
				beta_inv(i,j) = P[0][i].dot(Q[j]);
				beta_inv(j,i) = conj(beta_inv(i,j));
			}
		}
		// Find inverse of beta_inv via LDLT cholesky decomposition
		// and solving beta beta_inv = I
		beta = beta_inv.ldlt().solve(Eigen::MatrixXcd::Identity(N, N));
		betaC = beta * C;

		// X[0] = X[0] + P[0] beta C
		for(int i=0; i<N; ++i) {
			for(int j=0; j<N; ++j) {
				X[0][i].add(betaC(j,i), P[0][j]);
			}
		}

		//Q -= AP beta
		for(int i=0; i<N; ++i) {
			for(int j=0; j<N; ++j) {
				Q[i].add(-beta(j,i), AP[j]);
			}
		}

		S_m1 = S;
		// in-place thinQR decomposition of residuals matrix Q
		thinQR(Q, S);
		C = S * C;
		if(N_unconverged_shifts>0) {
			// update decomposition of S for S_inv operations
			S_inv.compute(S);
		}

		// P <- Q + P S^dag for lower triangular S
		for(int i=0; i<N; ++i) {
			P[0][i] *= S(i,i);
			for(int j=i+1; j<N; ++j) {
				P[0][i].add(conj(S(i,j)), P[0][j]);
			}
			P[0][i] += Q[i];
		}

		// calculate shifted X and P
		// note X[0] and P[0] are the unshifted ones, so +1 to i_shift in X and P
		for(int i_shift=0; i_shift<N_unconverged_shifts; ++i_shift) {
			// calculate shifted coefficients
			// ksi_s:
			tmp_betaC = S_m1 * beta_inv_m1 - ksi_s_m1[i_shift] * ksi_s_inv_m2[i_shift].solve(beta_inv_m1);
			tmp_Sdag = Eigen::MatrixXcd::Identity(N, N) + shifts[i_shift] * beta + tmp_betaC * S_m1.adjoint() * beta;
			ksi_s[i_shift] = S * tmp_Sdag.colPivHouseholderQr().solve(ksi_s_m1[i_shift]);
			// tmp_betaC == "alpha^sigma" in paper:
			tmp_betaC = beta * S_inv.solve(ksi_s[i_shift]);
			// tmp_Sdag == "beta^sigma" in paper:
			tmp_Sdag = tmp_betaC * ksi_s_inv_m1[i_shift].solve(beta_inv * S.adjoint());
			// update shifted X and P
			// X_s = X_s + P_s tmp_betaC
			for(int i=0; i<N; ++i) {
				for(int j=0; j<N; ++j) {
					X[i_shift+1][i].add(tmp_betaC(j,i), P[i_shift+1][j]);
				}
			}
			// P_s <- Q + P_s tmp_Sdag (using AP as temporary storage)
			for(int i=0; i<N; ++i) {
				AP[i] = Q[i];
				for(int j=0; j<N; ++j) {
					AP[i].add(tmp_Sdag(j,i), P[i_shift+1][j]);
				}
			}
			P[i_shift+1] = AP;
			// update inverse ksi's for next iteration
			ksi_s_m2[i_shift] = ksi_s_m1[i_shift];
			ksi_s_inv_m2[i_shift] = ksi_s_inv_m1[i_shift];
			ksi_s_m1[i_shift] = ksi_s[i_shift];
			ksi_s_inv_m1[i_shift].compute(ksi_s[i_shift]);
			// check if largest unconverged shift has converged
			if((ksi_s[N_unconverged_shifts-1].rowwise().norm().array()/b_norm).maxCoeff() < eps) {
				--N_unconverged_shifts;
			}
		}

		// use maximum over vectors of residuals/b_norm for unshifted solution as stopping crit
		// assumes that all shifts are positive i.e. better conditioned and converge faster
		// worst vector should be equal to CG with same eps on that vector, others will be better
		residual = (C.rowwise().norm().array()/b_norm).maxCoeff();

		// [debugging] find eigenvalues of C
		
		//Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> saes;
		//saes.compute(C);
		//Eigen::ArrayXd evals = saes.eigenvalues();
		
			// Output iteration number and worst residual for each shift
		std::cout << "#SBCGrQ " << iter << "\t" << residual;
		for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
			std::cout << "\t" << (ksi_s[i_shift].rowwise().norm().array()/b_norm).maxCoeff();
		}
		std::cout << std::endl;
	}
	return iter;
}
*/
// Bartels–Stewart: O(N^3) algorithm to solve Sylvester equation for X:
// AX + XB = C
// (NB there also exists the Hessenberg-Schur method that only requires one Schur decomposition)
void bartels_stewart(Eigen::MatrixXcd& X, const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B, const Eigen::MatrixXcd& C) {
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