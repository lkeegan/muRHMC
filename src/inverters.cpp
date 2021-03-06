#include "inverters.hpp"
#include <chrono>
#include <iostream>  //FOR DEBUGGING
#include "omp.h"

// Construct R_ij = X_i^dag Y_j assuming that the result is hermitian (e.g. Y =
// X, or Y = MX with M hermitian), so only does the dot products needed for the
// lower triangular part of matrix
void hermitian_dot(const field<block_fermion>& X, const field<block_fermion>& Y,
                   block_matrix& R) {
  // construct lower-triangular part of matrix
  R.setZero();
  for (int ix = 0; ix < X.V; ++ix) {
    for (int i = 0; i < N_rhs; ++i) {
      for (int j = 0; j <= i; ++j) {
        R(i, j) += X[ix].col(i).dot(Y[ix].col(j));
      }
    }
  }
  // upper triangular part from conjugate of lower triangular elements
  for (int i = 1; i < N_rhs; ++i) {
    for (int j = 0; j < i; ++j) {
      R(j, i) = std::conj(R(i, j));
    }
  }
}

// In-place Multiply field X on RHS by inverse of triangular matrix R, i.e.
// X <- X R^{-1}
void multiply_triangular_inverse_RHS(field<block_fermion>& X, block_matrix& R) {
#pragma omp parallel for
  for (int ix = 0; ix < X.V; ++ix) {
    for (int i = 0; i < N_rhs; ++i) {
      for (int j = 0; j < i; ++j) {
        X[ix].col(i) -= R(j, i) * X[ix].col(j);
      }
      X[ix].col(i) /= R(i, i);
    }
  }
  // NOTE: this could instead be done using eigen in one line:
  // R.triangularView<Eigen::Upper>().solveInPlace<Eigen::OnTheRight>(Q[ix]);
  // however the above must use temporaries, as it is much slower than my simple
  // in-place implementation
}

// in-place thin QR decomposition of Q using Algorithm 2 from arXiv:1710.09745
// input: Q is a vector of N fermion fields
// output: R is a NxN complex hermitian matrix such that Q_input R = Q_output
// output: Q is a now a vector of N orthogonal fermion fields
void thinQR(field<block_fermion>& Q, block_matrix& R) {
  // Construct R_ij = Q_i^dag Q_j = hermitian,
  hermitian_dot(Q, Q, R);
  // Find upper triangular R such that R^dag R = H (i.e. previous contents of R)
  // = Q^dag Q i.e. adjoint of cholesky decomposition L matrix: L L^dag = H
  R = R.llt().matrixL().adjoint();

  // Q <- Q R^-1
  multiply_triangular_inverse_RHS(Q, R);
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
void thinQRA(field<block_fermion>& V, field<block_fermion>& AV,
             block_matrix& R) {
  hermitian_dot(V, AV, R);

  // Find upper triangular R such that R^dag R = [V^dag AV]
  // i.e. adjoint of cholesky decomposition L matrix
  R = R.llt().matrixL().adjoint();

  // Solve V_new R = V and AV_new R = AV where R is upper triangular
  multiply_triangular_inverse_RHS(V, R);
  multiply_triangular_inverse_RHS(AV, R);
}

// A-orthogonalise X in place, without having AX already, i.e. does a bunch of
// DDdag operations Return the eigenvalues of the hermitian matrix <X_i|A|X_j>
// in first column of Evals [eigenvalues] And the square root of the diagonal
// elemets only of the hermitian matrix <X_i|A^2|X_j> in 2nd [optimistic error
// estimates] (Does not calculate or return the square root of the largest
// eigenvalue of <X_i|A^2|X_j> [conservative error estimate])
void thinQRA_evals(field<block_fermion>& X, Eigen::MatrixXd& Evals,
                   field<gauge>& U, dirac_op& D) {
  field<block_fermion> AX(X);
  // Construct lower triangular part of hermitian matrix <X_i|A|X_j> and
  // diagonal part of <X_i|A^2|X_j>
  block_matrix R = block_matrix::Zero();
  D.DDdagger(AX, X, U);
  hermitian_dot(X, AX, R);
  Evals.col(1) = R.diagonal().real().array().sqrt();
  // find eigensystem of R - only references lower triangular part
  // NB also finding eigenvectors here, but no need to..
  Eigen::SelfAdjointEigenSolver<block_matrix> R_eigen_system(R);
  Evals.col(0) = R_eigen_system.eigenvalues().col(0);
  // A-orthonormalise X
  R = R.llt().matrixL().adjoint();
  multiply_triangular_inverse_RHS(X, R);
}

int cg(field<fermion>& x, const field<fermion>& b, field<gauge>& U, dirac_op& D,
       double eps) {
  field<fermion> a(b.grid, b.eo_storage), p(b.grid, b.eo_storage),
      r(b.grid, b.eo_storage);
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
  while (sqrt(r2_new) / b_norm > eps) {
    // a = A p
    D.DDdagger(a, p, U, true);
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
    // std::cout << iter << "\t" << sqrt(r2_new)/b_norm << std::endl;
    if (iter > 1e5) {
      std::cout << "CG not converging: iter= " << iter
                << " residual= " << sqrt(r2_new) / b_norm << std::endl;
      // exit(1);
      return iter;
    }
  }
  D.remove_eta_bcs_from_U(U);
  return iter;
}

int cg_singleshift(field<fermion>& x, const field<fermion>& b, field<gauge>& U,
                   double shift, dirac_op& D, double eps) {
  field<fermion> a(b.grid, b.eo_storage), p(b.grid, b.eo_storage),
      r(b.grid, b.eo_storage);
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
  while (sqrt(r2_new) / b_norm > eps) {
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
    if (iter > 1e5) {
      std::cout << "cg_singleshift not converging: iter= " << iter
                << " residual= " << sqrt(r2_new) / b_norm << std::endl;
      exit(1);
    }
  }
  return iter;
}

int cg_multishift(std::vector<field<fermion>>& x, const field<fermion>& b,
                  field<gauge>& U, std::vector<double>& sigma, dirac_op& D,
                  double eps, double eps_shifts, int max_iter) {
  int n_shifts = x.size();
  int n_unconverged_shifts = n_shifts;
  std::vector<field<fermion>> p;
  field<fermion> a(b.grid, b.eo_storage), r(b.grid, b.eo_storage);
  double r2_old = 0;
  double r2_new = 0;
  std::vector<std::complex<double>> beta(n_shifts), beta_m1(n_shifts);
  std::vector<std::complex<double>> zeta_m1(n_shifts), zeta(n_shifts),
      zeta_p1(n_shifts);
  std::vector<std::complex<double>> alpha(n_shifts);
  int iter = 0;

  // initial guess zero required for multi-shift CG
  // x_i = 0, a = Ax_0 = 0
  a.setZero();
  // r = b - a
  r = b;
  // p = r
  for (int i_shift = 0; i_shift < n_shifts; ++i_shift) {
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
  // std::cout << "INIT res " << sqrt(r2_new)/b_norm << " x00: " << x[0][0](0)
  // << std::endl;
  while (sqrt(r2_new) / b_norm > eps && iter < max_iter) {
    // a = A p_0
    D.DDdagger(a, p[0], U, true);
    // add first shift to DDdagger explicitly here:
    a.add(sigma[0], p[0]);
    ++iter;
    r2_old = r2_new;
    // beta = -<r|r>/<p|a>
    beta[0] = -r2_old / p[0].dot(a);
    // calculate zeta and beta coefficients for shifted vectors
    // see arXiv:hep-lat/9612014 for derivation
    for (int i_shift = 1; i_shift < n_unconverged_shifts; ++i_shift) {
      zeta_p1[i_shift] = zeta[i_shift] * zeta_m1[i_shift] * beta_m1[0];
      zeta_p1[i_shift] /=
          (beta[0] * alpha[0] * (zeta_m1[i_shift] - zeta[i_shift]) +
           zeta_m1[i_shift] * beta_m1[0] *
               (1.0 - (sigma[i_shift] - sigma[0]) * beta[0]));
      beta[i_shift] = beta[0] * zeta_p1[i_shift] / zeta[i_shift];
    }
    // if largest shift has converged (to machine precision), i.e. normalised
    // residual < ulp, stop updating it
    if (sqrt((zeta[n_unconverged_shifts - 1] *
              conj(zeta[n_unconverged_shifts - 1]))
                 .real() *
             r2_old) /
            b_norm <
        eps_shifts) {
      --n_unconverged_shifts;
      // std::cout << "iter " << iter << "converged " << n_unconverged_shifts+1
      // << std::endl;
    }
    // x_i -= beta_i * p_i
    for (int i_shift = 0; i_shift < n_unconverged_shifts; ++i_shift) {
      x[i_shift].add(-beta[i_shift], p[i_shift]);
    }
    // std::cout << "iter " << iter << "res " << sqrt(r2_new)/b_norm << " x00: "
    // << x[0][0](0) << std::endl;
    // r += beta_0 a
    r.add(beta[0], a);
    // r2_new = <r|r>
    r2_new = r.squaredNorm();
    // increment timestep:
    // X_m1 <- X
    // X <- X_p1
    alpha[0] = r2_new / r2_old;
    beta_m1[0] = beta[0];
    for (int i_shift = 1; i_shift < n_unconverged_shifts; ++i_shift) {
      beta_m1[i_shift] = beta[i_shift];
      zeta_m1[i_shift] = zeta[i_shift];
      zeta[i_shift] = zeta_p1[i_shift];
    }
    // calculate alpha coeffs for shifted vectors
    for (int i_shift = 1; i_shift < n_unconverged_shifts; ++i_shift) {
      alpha[i_shift] = alpha[0] * (zeta[i_shift] * beta_m1[i_shift]) /
                       (zeta_m1[i_shift] * beta_m1[0]);
    }
    // p_i = alpha_i p_i + zeta_i r
    for (int i_shift = 0; i_shift < n_unconverged_shifts; ++i_shift) {
      p[i_shift].scale_add(alpha[i_shift], zeta[i_shift], r);
    }
    if (iter > 1e5) {
      std::cout << "CG-multishift not converging: iter= " << iter
                << " residual= " << sqrt(r2_new) / b_norm << std::endl;
      exit(1);
    }
  }
  // std::cout << "CONV iter " << iter << "res " << sqrt(r2_new)/b_norm << "
  // x00: " << x[0][0](0) << std::endl;
  D.remove_eta_bcs_from_U(U);

  return iter;
}

// return rational approx x = \alpha_0 + \sum_i \alpha_{i+1} [A + \beta_i]^{-1}
// b using cg_multishift for inversions avoids storing shifted solution vectors,
// instead does above sum for every CG iteration and only stores result still
// need the shifted search vectors p of course
int rational_approx_cg_multishift(field<fermion>& x, const field<fermion>& b,
                                  field<gauge>& U,
                                  std::vector<double>& rational_alpha,
                                  std::vector<double>& rational_beta,
                                  dirac_op& D, double eps) {
  int n_shifts = rational_beta.size();
  int n_unconverged_shifts = n_shifts;
  std::vector<field<fermion>> p;
  field<fermion> a(b.grid, b.eo_storage), r(b.grid, b.eo_storage);
  double r2_old = 0;
  double r2_new = 0;
  std::vector<std::complex<double>> beta(n_shifts), beta_m1(n_shifts);
  std::vector<std::complex<double>> zeta_m1(n_shifts), zeta(n_shifts),
      zeta_p1(n_shifts);
  std::vector<std::complex<double>> alpha(n_shifts);
  int iter = 0;

  // initial guess zero required for multi-shift CG
  // x_i = 0, a = Ax_0 = 0
  a.setZero();
  // r = b - a
  r = b;
  // p = r
  for (int i_shift = 0; i_shift < n_shifts; ++i_shift) {
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
  while (sqrt(r2_new) / b_norm > eps) {
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
    for (int i_shift = 1; i_shift < n_unconverged_shifts; ++i_shift) {
      zeta_p1[i_shift] = zeta[i_shift] * zeta_m1[i_shift] * beta_m1[0];
      zeta_p1[i_shift] /=
          (beta[0] * alpha[0] * (zeta_m1[i_shift] - zeta[i_shift]) +
           zeta_m1[i_shift] * beta_m1[0] *
               (1.0 - (rational_beta[i_shift] - rational_beta[0]) * beta[0]));
      beta[i_shift] = beta[0] * zeta_p1[i_shift] / zeta[i_shift];
    }
    // if largest shift has converged, i.e. normalised residual < eps, stop
    // updating it
    if (sqrt((zeta[n_unconverged_shifts - 1] *
              conj(zeta[n_unconverged_shifts - 1]))
                 .real() *
             r2_old) /
            b_norm <
        eps) {
      --n_unconverged_shifts;
    }
    // usual CG solver would do: x_i -= beta_i * p_i
    // instead we do: x -= \sum_i \rational_alpha_i beta_i * p_i
    for (int i_shift = 0; i_shift < n_unconverged_shifts; ++i_shift) {
      x.add(-rational_alpha[i_shift + 1] * beta[i_shift], p[i_shift]);
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
    for (int i_shift = 1; i_shift < n_unconverged_shifts; ++i_shift) {
      beta_m1[i_shift] = beta[i_shift];
      zeta_m1[i_shift] = zeta[i_shift];
      zeta[i_shift] = zeta_p1[i_shift];
    }
    // calculate alpha coeffs for shifted vectors
    for (int i_shift = 1; i_shift < n_unconverged_shifts; ++i_shift) {
      alpha[i_shift] = alpha[0] * (zeta[i_shift] * beta_m1[i_shift]) /
                       (zeta_m1[i_shift] * beta_m1[0]);
    }
    // p_i = alpha_i p_i + zeta_i r
    for (int i_shift = 0; i_shift < n_unconverged_shifts; ++i_shift) {
      p[i_shift].scale_add(alpha[i_shift], zeta[i_shift], r);
    }
    if (iter > 1e5) {
      std::cout << "CG-multishift not converging: iter= " << iter
                << " residual= " << sqrt(r2_new) / b_norm << std::endl;
      exit(1);
    }
  }
  return iter;
}

int cg_block(field<block_fermion>& X, const field<block_fermion>& B,
             field<gauge>& U, dirac_op& D, double eps, bool BCGA, bool dQ,
             bool dQA, bool rQ, bool OUTPUT_ERROR_NORMS,
             field<fermion>* x0_star) {
  int N = N_rhs;
  // S = 1 [NxN]
  block_matrix S = block_matrix::Identity();
  // C = 1 [NxN]
  block_matrix C = block_matrix::Identity();
  // beta = 0 [NxN]
  block_matrix beta = block_matrix::Zero();
  // betaC = 0 [NxN]
  block_matrix betaC = block_matrix::Zero();
  // R = 1 [NxN]
  block_matrix R = block_matrix::Identity();
  // PQ, PAQ (for BCGA dot products) [NxN]
  block_matrix mPAPinv = block_matrix::Identity();
  block_matrix mPQ = block_matrix::Identity();
  block_matrix mPAQ = block_matrix::Identity();
  // AP, P, Q are [NxVOL]
  field<block_fermion> AP(X), P(X), Q(B);

  // for debugging (error norms of first vector):
  double norm0_x0_star = 1.0;
  double norm1_x0_star = 1.0;
  if (OUTPUT_ERROR_NORMS) {
    // get error norms for X=0 to normalise all to 1 intially
    field<fermion> tmpAE0(*x0_star);
    norm0_x0_star = sqrt(x0_star->squaredNorm());
    D.DDdagger(tmpAE0, *x0_star, U);
    norm1_x0_star = sqrt(x0_star->dot(tmpAE0).real());
    // note norm2 is just the residual so we already have the normalisation
  }

  // start from X=0 initial guess, so residual Q = B [NxVOL]
  // Q = B;
  X.setZero();
  if (rQ) {
    // in place thinQR decomposition of residual Q[N][VOL] into orthonormal
    // Q[N][VOL] and triangular C[NxN]
    thinQR(Q, C);
  } else if (BCGA) {
    // set diagonal values of C to residuals Q^dag Q, only needed for residual
    // stopping criterion
    for (int i = 0; i < N; ++i) {
      C.setZero();
      for (int ix = 0; ix < Q.V; ++ix) {
        for (int i = 0; i < N_rhs; ++i) {
          C(i, i) += Q[ix].col(i).squaredNorm();
          //				C += Q[ix].adjoint()*Q[ix];
        }
      }
      // C(i, i) = Q[i].dot(Q[i]);
    }
  } else {
    // set C to hermitian matrix Q^dag Q
    hermitian_dot(Q, Q, C);
    // S = C_old^-1 C_new = C since C_old=1.
    S = C;
  }

  // P = 0 [NxVOL]
  P.setZero();

  // main loop
  int iter = 0;
  // get norm of each vector b in matrix B
  // NB: v.norm() == sqrt(\sum_i v_i v_i^*) = l2 norm of vector v
  Eigen::ArrayXd b_norm = Eigen::ArrayXd::Zero(N);
  if (rQ) {
    // residual_i = sqrt(\sum_j Q_i^dag Q_j) = sqrt(\sum_j C_ij)
    b_norm = C.rowwise().norm().array();
  } else {
    // residual_i = sqrt(Q_i^dag Q_i) = sqrt(C_ii)
    b_norm = C.diagonal().real().array().sqrt();
  }
  double residual = 1.0;
  while (residual > eps) {
    // P <- Q + PS
    for (int ix = 0; ix < P.V; ++ix) {
      for (int i = 0; i < N; ++i) {
        AP[ix].col(i) = Q[ix].col(i);
        for (int j = 0; j < N; ++j) {
          AP[ix].col(i) += S(j, i) * P[ix].col(j);
        }
      }
    }
    P = AP;

    if (dQ) {
      // in-place thinQR decomposition of descent matrix P
      // such that P^dag P = I
      thinQR(P, R);
    }

    // Apply dirac op to P:
    D.DDdagger(AP, P, U);
    iter++;

    if (dQA) {
      // in-place thinQRA decomposition of descent matrix P and AP
      // such that P^dag AP = I
      thinQRA(P, AP, R);
    }

    // construct NxN beta matrix:
    // beta^-1 = P^dag AP [NxN hermitian matrix]
    if (dQA) {
      // P^dag AP = I by construction
      beta = block_matrix::Identity();
    } else {
      // note beta is hermitian since A is hermitian so we
      // only need to calculate lower triangular elements of matrix
      hermitian_dot(P, AP, beta);
      // Find inverse of beta via LDLT cholesky decomposition
      // and solving beta beta^-1 = I
      beta = beta.ldlt().solve(block_matrix::Identity());
    }

    if ((dQ || dQA) && !BCGA) {
      // beta <- beta (R^dag)^-1
      // Solve X (R^dag) = beta for X, then beta <- X
      // Can be done in-place by back-substitution since R is upper-triangular
      // std::cout << "beta\n" << beta << std::endl;
      for (int i = 0; i < N; ++i) {
        for (int j = N - 1; j >= 0; --j) {
          for (int k = N - 1; k > j; --k) {
            beta(i, j) -= beta(i, k) * conj(R(j, k));
          }
          beta(i, j) /= R(j, j);
        }
      }
      // std::cout << "new beta\n" << beta << std::endl;
      // std::cout << "beta R^dag\n" << beta*R.adjoint() << std::endl;
    }

    if (BCGA) {
      mPAPinv = beta;
      mPQ.setZero();
      for (int ix = 0; ix < Q.V; ++ix) {
        mPQ += P[ix].adjoint() * Q[ix];
      }
      beta = beta * mPQ;
      if (rQ) {
        betaC = beta * C;
      } else {
        betaC = beta;
      }
    } else {
      betaC = beta * C;
      if (!rQ) {
        beta = betaC;
      }
    }
    // X = X + P beta C
    X.add(P, betaC);
    /*		for(int ix=0; ix<X.V; ++ix) {
                            for(int i=0; i<N; ++i) {
                                    for(int j=0; j<N; ++j) {
                                            X[ix].col(i) += betaC(j,i) *
       P[ix].col(j);
                                    }
                            }
                    }
    */		//Q -= AP beta
    betaC = -beta;
    Q.add(AP, betaC);
    /*		for(int ix=0; ix<Q.V; ++ix) {
                            for(int i=0; i<N; ++i) {
                                    for(int j=0; j<N; ++j) {
                                            Q[ix].col(i) -= beta(j,i) *
       AP[ix].col(j);
                                    }
                            }
                    }
    */
    if (BCGA) {
      if (rQ) {
        // in-place thinQR decomposition of residuals matrix Q
        thinQR(Q, S);
        C = S * C;
      } else {
        // update diagonal values of C for residual
        C.setZero();
        for (int ix = 0; ix < Q.V; ++ix) {
          C += Q[ix].adjoint() * Q[ix];
        }
      }
      // S <- -[P^dag AP]^-1 [Q^dag AP] = - PQ [Q^dag AP]
      hermitian_dot(Q, AP, mPAQ);
      /*			mPAQ.setZero();
                              for(int ix=0; ix<Q.V; ++ix) {
                                      mPAQ += Q[ix].adjoint()*AP[ix];
                              }
      */
      S = -mPAPinv * mPAQ.adjoint();
    } else {
      if (rQ) {
        // in-place thinQR decomposition of residuals matrix Q
        thinQR(Q, S);
        C = S * C;
        // S <- S^dag:
        S.adjointInPlace();
      } else {
        // find inverse of C = Q_old^dagQ_old by cholesky decomposition
        S = C.ldlt().solve(block_matrix::Identity());
        // update C to hermitian matrix Q^dag Q
        hermitian_dot(Q, Q, C);
        /*				C.setZero();
                                        for(int ix=0; ix<Q.V; ++ix) {
                                                C += Q[ix].adjoint()*Q[ix];
                                        }
        */				// S = [Q_old^dag Q_old]^-1 [Q_new^dag Q_new]
        S = S * C;
      }
      if (dQ || dQA) {
        // S <- RS:
        S = R * S;
      }
    }

    // use maximum over vectors of residual/b_norm as stopping crit
    // worst vector should be equal to CG with same eps on that vector, others
    // will be better
    if (rQ) {
      residual = (C.rowwise().norm().array() / b_norm).maxCoeff();
    } else {
      // C_ii = Q_i^dag Q_i
      residual = (C.diagonal().real().array().sqrt() / b_norm).maxCoeff();
    }

    if (OUTPUT_ERROR_NORMS) {
      // debugging: use known solution to get error norms for first block vector
      field<fermion> tmpE0(*x0_star), tmpAE0(*x0_star);
      for (int ix = 0; ix < Q.V; ++ix) {
        tmpE0[ix] = X[ix].col(0);
      }
      tmpE0.add(-1.0, *x0_star);
      D.DDdagger(tmpAE0, tmpE0, U);

      double norm0 = sqrt(tmpE0.squaredNorm()) / norm0_x0_star;
      double norm1 = sqrt(tmpE0.dot(tmpAE0).real()) / norm1_x0_star;
      double norm2 = sqrt(tmpAE0.squaredNorm()) / b_norm[0];
      std::cout << "#Error-norms <(x-x*)|(1,sqrt(A),A)|(x-x*)> " << iter << "\t"
                << norm0 << "\t" << norm1 << "\t" << norm2 << std::endl;
    }
    /*
    // [debugging] find eigenvalues of C
    Eigen::SelfAdjointEigenSolver<block_matrix> saes;
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
    std::cout << " " << iter << "\t" << residual << "\t" <<
    evals.maxCoeff()/evals.minCoeff() << std::endl;
    */
  }
  return iter;
}

int SBCGrQ_old(std::vector<field<block_fermion>>& X,
               const field<block_fermion>& B, field<gauge>& U,
               std::vector<double>& input_shifts, dirac_op& D, double eps,
               double eps_shifts, int max_iter) {
  int N = N_rhs;
  // count shifts (not including first one that is included in the dirac op)
  int N_shifts = static_cast<int>(input_shifts.size()) - 1;
  int N_unconverged_shifts = N_shifts;

  // Check shift parameters are consistent with size of supplied solution vector
  // X:
  if (N_shifts + 1 != static_cast<int>(X.size())) {
    std::cout << "Error in SBCGrQ: N_shifts+1 (" << N_shifts + 1
              << ") not consistent with size of X ("
              << static_cast<int>(X.size()) << ")" << std::endl;
    exit(1);
  }

  // subtract first shift from remaining set of shifts
  std::vector<double> shifts(N_shifts);
  for (int i_shift = 0; i_shift < N_shifts; ++i_shift) {
    shifts[i_shift] = input_shifts[i_shift + 1] - input_shifts[0];
  }

  // Unshifted matrices:
  block_matrix S = block_matrix::Identity();
  block_matrix C = block_matrix::Identity();
  block_matrix beta = block_matrix::Identity();
  block_matrix betaC = block_matrix::Zero();
  // AP, Q are [NxVOL]
  field<block_fermion> AP(B), Q(B);
  // start from X=0 initial guess, so residual Q = B [NxVOL]
  // X = 0 for unshifted + all shifts
  for (int i_shift = 0; i_shift < N_shifts + 1; ++i_shift) {
    X[i_shift].setZero();
  }
  // Q = B;
  // in place thinQR decomposition of residual Q[N][VOL] into orthonormal
  // Q[N][VOL] and triangular C[NxN]
  thinQR(Q, C);
  S = C;
  AP = Q;
  // P has one NxVOL block per shift
  std::vector<field<block_fermion>> P;
  // P = Q for all shifts
  for (int i_shift = 0; i_shift < N_shifts + 1; ++i_shift) {
    P.push_back(Q);
  }

  // Shifted matrices:
  // previous / inverted versions of beta
  block_matrix beta_inv = block_matrix::Identity();
  block_matrix beta_inv_m1 = block_matrix::Identity();
  // Construct decomposition of S: can then do S^-1 M using S_inv.solve(M)
  Eigen::FullPivLU<block_matrix> S_inv(S);
  block_matrix S_m1 = block_matrix::Zero();
  // These are temporary matrices used for each shift
  block_matrix tmp_betaC, tmp_Sdag;
  // ksi_s are the scale factors related shifted and unshifted residuals
  // 3-term recurrence so need _k, _k-1, _k-2 for each shift
  // initially ksi_s_m2 = I, ksi_s_m1 = S
  std::vector<block_matrix> ksi_s, ksi_s_m1, ksi_s_m2;
  std::vector<Eigen::FullPivLU<block_matrix>> ksi_s_inv_m1, ksi_s_inv_m2;
  for (int i_shift = 0; i_shift < N_shifts; ++i_shift) {
    ksi_s.push_back(block_matrix::Identity());
    ksi_s_m1.push_back(S);
    ksi_s_inv_m1.push_back(S_inv);
    ksi_s_m2.push_back(block_matrix::Identity());
    ksi_s_inv_m2.push_back(block_matrix::Identity().fullPivLu());
  }

  // main loop
  int iter = 0;
  // get norm of each vector b in matrix B
  // NB: v.norm() == sqrt(\sum_i v_i v_i^*) = l2 norm of vector v
  // residual_i = sqrt(\sum_j Q_i^dag Q_j) = sqrt(\sum_j C_ij)
  Eigen::ArrayXd b_norm = C.rowwise().norm().array();
  double residual = 1.0;

  while (residual > eps && iter < max_iter) {
    // auto timer_start = std::chrono::high_resolution_clock::now();

    // Apply dirac op (with lowest shift) to P[0]:
    D.DDdagger(AP, P[0], U, true);
    // do lowest shift as part of dirac op
    AP.add(input_shifts[0], P[0]);
    ++iter;

    // auto timer_stop = std::chrono::high_resolution_clock::now();
    // auto timer_count =
    // std::chrono::duration_cast<std::chrono::milliseconds>(timer_stop-timer_start).count();
    // std::cout << "#SBCG_DiracOp_Runtime " << timer_count << std::endl;
    // timer_start = std::chrono::high_resolution_clock::now();

    beta_inv_m1 = beta_inv;
    hermitian_dot(P[0], AP, beta_inv);
    //		beta_inv.setZero();
    //		for(int ix=0; ix<AP.V; ++ix) {
    //			beta_inv += P[0][ix].adjoint()*AP[ix];
    //		}
    // Find inverse of beta_inv via LDLT cholesky decomposition
    // and solving beta beta_inv = I
    beta = beta_inv.fullPivLu().solve(block_matrix::Identity());
    betaC = beta * C;

    // X[0] = X[0] + P[0] beta C
    X[0].add(P[0], betaC);

    // Q -= AP beta
    betaC = -beta;
    Q.add(AP, betaC);

    S_m1 = S;
    // in-place thinQR decomposition of residuals matrix Q
    thinQR(Q, S);
    C = S * C;
    if (N_unconverged_shifts > 0) {
      // update decomposition of S for S_inv operations
      S_inv.compute(S);
    }

    // P <- Q + P S^dag [in-place] for lower triangular S
    for (int ix = 0; ix < AP.V; ++ix) {
      for (int i = 0; i < N; ++i) {
        P[0][ix].col(i) *= S(i, i);
        for (int j = i + 1; j < N; ++j) {
          P[0][ix].col(i) += conj(S(i, j)) * P[0][ix].col(j);
        }
        P[0][ix].col(i) += Q[ix].col(i);
      }
    }
    // timer_stop = std::chrono::high_resolution_clock::now();
    // timer_count =
    // std::chrono::duration_cast<std::chrono::milliseconds>(timer_stop-timer_start).count();
    // std::cout << "#SBCG_Remaining_Unshifted_Runtime " << timer_count <<
    // std::endl; timer_start = std::chrono::high_resolution_clock::now();

    // calculate shifted X and P
    // note X[0] and P[0] are the unshifted ones, so first shift has index 1 in
    // X and P
    for (int i_shift = 0; i_shift < N_unconverged_shifts; ++i_shift) {
      // calculate shifted coefficients
      // ksi_s:
      tmp_betaC = S_m1 * beta_inv_m1 -
                  ksi_s_m1[i_shift] * ksi_s_inv_m2[i_shift].solve(beta_inv_m1);
      tmp_Sdag = block_matrix::Identity() + shifts[i_shift] * beta +
                 tmp_betaC * S_m1.adjoint() * beta;
      ksi_s[i_shift] = S * tmp_Sdag.fullPivLu().solve(ksi_s_m1[i_shift]);
      // tmp_betaC == "alpha^sigma" in paper:
      tmp_betaC = beta * S_inv.solve(ksi_s[i_shift]);
      // tmp_Sdag == "beta^sigma" in paper:
      tmp_Sdag =
          tmp_betaC * ksi_s_inv_m1[i_shift].solve(beta_inv * S.adjoint());
      // update shifted X and P
      // X_s = X_s + P_s tmp_betaC
      X[i_shift + 1].add(P[i_shift + 1], tmp_betaC);

      // P_s <- Q + P_s tmp_Sdag (using AP as temporary storage)
      AP = Q;
      AP.add(P[i_shift + 1], tmp_Sdag);
      P[i_shift + 1] = AP;

      // update inverse ksi's for next iteration
      ksi_s_m2[i_shift] = ksi_s_m1[i_shift];
      ksi_s_inv_m2[i_shift] = ksi_s_inv_m1[i_shift];
      ksi_s_m1[i_shift] = ksi_s[i_shift];
      ksi_s_inv_m1[i_shift].compute(ksi_s[i_shift]);
      // check if largest unconverged shift has converged
      if ((ksi_s[N_unconverged_shifts - 1].rowwise().norm().array() / b_norm)
              .maxCoeff() < eps_shifts) {
        --N_unconverged_shifts;
      }
    }

    // use maximum over vectors of residuals/b_norm for unshifted solution as
    // stopping crit assumes that all shifts are positive i.e. better
    // conditioned and converge faster worst vector should be equal to CG with
    // same eps on that vector, others will be better
    residual = (C.rowwise().norm().array() / b_norm).maxCoeff();

    // timer_stop = std::chrono::high_resolution_clock::now();
    // timer_count =
    // std::chrono::duration_cast<std::chrono::milliseconds>(timer_stop-timer_start).count();
    // std::cout << "#SBCG_Shifts_Runtime " << timer_count << std::endl;
    // timer_start = std::chrono::high_resolution_clock::now();

    // [debugging] find eigenvalues of C
    /*
    Eigen::SelfAdjointEigenSolver<block_matrix> saes;
    saes.compute(C);
    Eigen::ArrayXd evals = saes.eigenvalues();
    */
    // Output iteration number and worst residual for each shift
    /*
    std::cout << "#SBCGrQ " << iter << "\t" << residual;
    for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
            std::cout << "\t" <<
    (ksi_s[i_shift].rowwise().norm().array()/b_norm).maxCoeff();
    }
    std::cout << std::endl;
    */
  }
  D.remove_eta_bcs_from_U(U);

  return iter;
}

int SBCGrQ(std::vector<field<block_fermion>>& X, const field<block_fermion>& B,
           field<gauge>& U, std::vector<double>& sigma, dirac_op& D, double eps,
           double eps_shifts, int max_iter) {
  int n_shifts = static_cast<int>(sigma.size());
  int n_unconverged_shifts = n_shifts;

  // Check shift parameters are consistent
  if (n_shifts != static_cast<int>(X.size())) {
    std::cout << "Error in SBCGrQ: n_shifts (" << n_shifts
              << ") not consistent with size of X ("
              << static_cast<int>(X.size()) << ")" << std::endl;
    exit(1);
  }

  // Unshifted matrices:
  block_matrix Identity = block_matrix::Identity();
  block_matrix alpha, rho, delta;
  block_matrix alpha_inv(Identity), alpha_inv_old, rho_old;
  field<block_fermion> T(B), Q(B);
  // start from X=0 initial guess, so residual Q = B [NxVOL]
  // X = 0 for unshifted + all shifts
  for (int i_shift = 0; i_shift < n_shifts; ++i_shift) {
    X[i_shift].setZero();
  }
  // Q = B;
  // in place thinQR decomposition of residual Q
  thinQR(Q, delta);
  rho = delta;
  std::vector<field<block_fermion>> P(n_shifts, Q);

  // matrices for shifted residuals
  block_matrix beta_s_inv;
  using bm_alloc = Eigen::aligned_allocator<block_matrix>;
  std::vector<block_matrix, bm_alloc> alpha_s(n_shifts, Identity);
  std::vector<block_matrix, bm_alloc> beta_s(n_shifts, Identity);

  // main loop
  int iter = 0;
  // get norm of each vector b in matrix B
  // NB: v.norm() == sqrt(\sum_i v_i v_i^*) = l2 norm of vector v
  // residual_i = sqrt(\sum_j Q_i^dag Q_j) = sqrt(\sum_j delta_ij)
  Eigen::ArrayXd b_norm = delta.rowwise().norm().array();
  double residual = 1.0;

  while (residual > eps && iter < max_iter) {
    // Apply dirac op (with lowest shift) to P[0]:
    D.DDdagger(T, P[0], U, true);
    // do lowest shift as part of dirac op
    T.add(sigma[0], P[0]);
    ++iter;

    alpha_inv_old = alpha_inv;
    hermitian_dot(P[0], T, alpha_inv);
    // Find inverse of alpha^{-1} via fullPivLu decomposition
    alpha = alpha_inv.fullPivLu().solve(Identity);
    // X[0] = X[0] + P[0] alpha delta
    X[0].add(P[0], (alpha * delta).eval());
    // Q -= T alpha
    Q.add(T, (-alpha).eval());
    rho_old = rho;
    // in-place thinQR decomposition of residuals matrix Q
    thinQR(Q, rho);
    delta = rho * delta;
    // use maximum over relative residual of lowest shift as residual
    residual = (delta.rowwise().norm().array() / b_norm).maxCoeff();

    // P <- Q + P rho^dag [in-place] for lower triangular rho
    for (int ix = 0; ix < Q.V; ++ix) {
      for (int i = 0; i < N_rhs; ++i) {
        P[0][ix].col(i) *= rho(i, i);
        for (int j = i + 1; j < N_rhs; ++j) {
          P[0][ix].col(i) += conj(rho(i, j)) * P[0][ix].col(j);
        }
        P[0][ix].col(i) += Q[ix].col(i);
      }
    }

    // calculate shifted X and P
    for (int i_shift = n_unconverged_shifts - 1; i_shift > 0; --i_shift) {
      // calculate shifted coefficients
      beta_s_inv = Identity + (sigma[i_shift] - sigma[0]) * alpha +
                   alpha * rho_old * alpha_inv_old *
                       (Identity - beta_s[i_shift]) * rho_old.adjoint();
      beta_s[i_shift] = beta_s_inv.fullPivLu().solve(Identity);
      alpha_s[i_shift] =
          beta_s[i_shift] * alpha * rho_old * alpha_inv_old * alpha_s[i_shift];
      double residual_shift =
          ((rho * alpha_inv * alpha_s[i_shift]).rowwise().norm().array() /
           b_norm)
              .maxCoeff();
      // update shifted X and P
      // X_s = X_s + P_s tmp_betaC
      X[i_shift].add(P[i_shift], alpha_s[i_shift]);
      // P_s <- P_s tmp_Sdag + R
      T = Q;
      T.add(P[i_shift], (beta_s[i_shift] * rho.adjoint()).eval());
      P[i_shift] = T;
      // if shift has converged stop updating it
      if (residual_shift < eps_shifts) {
        --n_unconverged_shifts;
      }
    }
    D.remove_eta_bcs_from_U(U);
  }
  return iter;
}

int rational_approx_SBCGrQ(field<block_fermion>& X,
                           const field<block_fermion>& B, field<gauge>& U,
                           std::vector<double>& rational_alpha,
                           std::vector<double>& rational_beta, dirac_op& D,
                           double eps, double eps_shifts, int max_iter) {
  int N = N_rhs;
  // count shifts (not including first one that is included in the dirac op)
  int N_shifts = static_cast<int>(rational_beta.size()) - 1;
  int N_unconverged_shifts = N_shifts;

  // subtract first shift from remaining set of shifts
  std::vector<double> shifts(N_shifts);
  for (int i_shift = 0; i_shift < N_shifts; ++i_shift) {
    shifts[i_shift] = rational_beta[i_shift + 1] - rational_beta[0];
  }

  // Unshifted matrices:
  block_matrix S = block_matrix::Identity();
  block_matrix C = block_matrix::Identity();
  block_matrix beta = block_matrix::Identity();
  block_matrix betaC = block_matrix::Zero();
  // AP, Q are [NxVOL]
  field<block_fermion> AP(B), Q(B);
  // start from X=0 initial guess, so residual Q = B [NxVOL]
  // then set X = \alpha_0 B
  X = B;
  X *= rational_alpha[0];
  // Q = B;
  // in place thinQR decomposition of residual Q[N][VOL] into orthonormal
  // Q[N][VOL] and triangular C[NxN]
  thinQR(Q, C);
  S = C;
  AP = Q;
  // P has one NxVOL block per shift
  std::vector<field<block_fermion>> P;
  // P = Q for all shifts
  for (int i_shift = 0; i_shift < N_shifts + 1; ++i_shift) {
    P.push_back(Q);
  }

  // Shifted matrices:
  // previous / inverted versions of beta
  block_matrix beta_inv = block_matrix::Identity();
  block_matrix beta_inv_m1 = block_matrix::Identity();
  // Construct decomposition of S: can then do S^-1 M using S_inv.solve(M)
  Eigen::ColPivHouseholderQR<block_matrix> S_inv(S);
  block_matrix S_m1 = block_matrix::Zero();
  // These are temporary matrices used for each shift
  block_matrix tmp_betaC, tmp_Sdag;
  // ksi_s are the scale factors related shifted and unshifted residuals
  // 3-term recurrence so need _k, _k-1, _k-2 for each shift
  // initially ksi_s_m2 = I, ksi_s_m1 = S
  std::vector<block_matrix> ksi_s, ksi_s_m1, ksi_s_m2;
  std::vector<Eigen::ColPivHouseholderQR<block_matrix>> ksi_s_inv_m1,
      ksi_s_inv_m2;
  for (int i_shift = 0; i_shift < N_shifts; ++i_shift) {
    ksi_s.push_back(block_matrix::Identity());
    ksi_s_m1.push_back(S);
    ksi_s_inv_m1.push_back(S_inv);
    ksi_s_m2.push_back(block_matrix::Identity());
    ksi_s_inv_m2.push_back(block_matrix::Identity().colPivHouseholderQr());
  }

  // main loop
  int iter = 0;
  // get norm of each vector b in matrix B
  // NB: v.norm() == sqrt(\sum_i v_i v_i^*) = l2 norm of vector v
  // residual_i = sqrt(\sum_j Q_i^dag Q_j) = sqrt(\sum_j C_ij)
  Eigen::ArrayXd b_norm = C.rowwise().norm().array();
  double residual = 1.0;

  while (residual > eps && iter < max_iter) {
    // auto timer_start = std::chrono::high_resolution_clock::now();

    // Apply dirac op (with lowest shift) to P[0]:
    D.DDdagger(AP, P[0], U, true);
    // do lowest shift as part of dirac op
    AP.add(rational_beta[0], P[0]);
    ++iter;

    // auto timer_stop = std::chrono::high_resolution_clock::now();
    // auto timer_count =
    // std::chrono::duration_cast<std::chrono::milliseconds>(timer_stop-timer_start).count();
    // std::cout << "#SBCG_DiracOp_Runtime " << timer_count << std::endl;
    // timer_start = std::chrono::high_resolution_clock::now();

    beta_inv_m1 = beta_inv;
    beta_inv.setZero();
    for (int ix = 0; ix < AP.V; ++ix) {
      beta_inv += P[0][ix].adjoint() * AP[ix];
    }
    // Find inverse of beta_inv via LDLT cholesky decomposition
    // and solving beta beta_inv = I
    beta = beta_inv.ldlt().solve(block_matrix::Identity());
    betaC = beta * C;

    // X[0] = X[0] + P[0] beta C
    betaC *= rational_alpha[1];
    X.add(P[0], betaC);

    // Q -= AP beta
    betaC = -beta;
    Q.add(AP, betaC);

    S_m1 = S;
    // in-place thinQR decomposition of residuals matrix Q
    thinQR(Q, S);
    C = S * C;
    if (N_unconverged_shifts > 0) {
      // update decomposition of S for S_inv operations
      S_inv.compute(S);
    }

    // P <- Q + P S^dag for lower triangular S
    for (int ix = 0; ix < AP.V; ++ix) {
      for (int i = 0; i < N; ++i) {
        P[0][ix].col(i) *= S(i, i);
        for (int j = i + 1; j < N; ++j) {
          P[0][ix].col(i) += conj(S(i, j)) * P[0][ix].col(j);
        }
        P[0][ix].col(i) += Q[ix].col(i);
      }
    }
    // timer_stop = std::chrono::high_resolution_clock::now();
    // timer_count =
    // std::chrono::duration_cast<std::chrono::milliseconds>(timer_stop-timer_start).count();
    // std::cout << "#SBCG_Remaining_Unshifted_Runtime " << timer_count <<
    // std::endl; timer_start = std::chrono::high_resolution_clock::now();

    // calculate shifted X and P
    // note X[0] and P[0] are the unshifted ones, so first shift has index 1
    // in X and P
    for (int i_shift = 0; i_shift < N_unconverged_shifts; ++i_shift) {
      // calculate shifted coefficients
      // ksi_s:
      tmp_betaC = S_m1 * beta_inv_m1 -
                  ksi_s_m1[i_shift] * ksi_s_inv_m2[i_shift].solve(beta_inv_m1);
      tmp_Sdag = block_matrix::Identity() + shifts[i_shift] * beta +
                 tmp_betaC * S_m1.adjoint() * beta;
      ksi_s[i_shift] =
          S * tmp_Sdag.colPivHouseholderQr().solve(ksi_s_m1[i_shift]);
      // tmp_betaC == "alpha^sigma" in paper:
      tmp_betaC = beta * S_inv.solve(ksi_s[i_shift]);
      // tmp_Sdag == "beta^sigma" in paper:
      tmp_Sdag =
          tmp_betaC * ksi_s_inv_m1[i_shift].solve(beta_inv * S.adjoint());
      // update shifted X and P
      // rational approx solution X is linear sum
      // X += \alpha[i_shift+1] (P_s tmp_betaC)[i_shift]
      tmp_betaC *= rational_alpha[i_shift + 2];
      X.add(P[i_shift + 1], tmp_betaC);

      // P_s <- Q + P_s tmp_Sdag (using AP as temporary storage)
      AP = Q;
      AP.add(P[i_shift + 1], tmp_Sdag);
      P[i_shift + 1] = AP;

      // update inverse ksi's for next iteration
      ksi_s_m2[i_shift] = ksi_s_m1[i_shift];
      ksi_s_inv_m2[i_shift] = ksi_s_inv_m1[i_shift];
      ksi_s_m1[i_shift] = ksi_s[i_shift];
      ksi_s_inv_m1[i_shift].compute(ksi_s[i_shift]);
      // check if largest unconverged shift has converged
      if ((ksi_s[N_unconverged_shifts - 1].rowwise().norm().array() / b_norm)
              .maxCoeff() < eps_shifts) {
        --N_unconverged_shifts;
      }
    }

    // use maximum over vectors of residuals/b_norm for unshifted solution as
    // stopping crit assumes that all shifts are positive i.e. better
    // conditioned and converge faster worst vector should be equal to CG with
    // same eps on that vector, others will be better
    residual = (C.rowwise().norm().array() / b_norm).maxCoeff();

    // timer_stop = std::chrono::high_resolution_clock::now();
    // timer_count =
    // std::chrono::duration_cast<std::chrono::milliseconds>(timer_stop-timer_start).count();
    // std::cout << "#SBCG_Shifts_Runtime " << timer_count << std::endl;
    // timer_start = std::chrono::high_resolution_clock::now();
  }
  D.remove_eta_bcs_from_U(U);

  return iter;
}

/*
int SBCGAdQArQ(std::vector< std::vector< field<fermion> > >& X, const
std::vector<field<fermion>>& B, field<gauge>& U, std::vector<double>&
input_shifts, dirac_op& D, double eps) { int N = static_cast<int>(B.size());
        // count shifts (not including first one that is just a mass rescaling
in the dirac op) int N_shifts = static_cast<int>(input_shifts.size()) - 1; int
N_unconverged_shifts = N_shifts;

        // Check shift parameters are consistent with size of supplied
solution vector X: if (N_shifts+1 != static_cast<int>(X.size())) { std::cout
<< "Error in SBCGAdQArQ: N_shifts+1 (" << N_shifts+1 << ") not consistent with
size of X ("
<< static_cast<int>(X.size()) << ")" << std::endl; exit(1);
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
        block_matrix S = block_matrix::Identity(N, N);
        block_matrix C = block_matrix::Identity(N, N);
        block_matrix beta = block_matrix::Identity(N, N);
        block_matrix betaC = block_matrix::Zero(N, N);
        block_matrix R = block_matrix::Identity(N, N);
        // AP, Q are [NxVOL]
        std::vector< field<fermion> > AP, Q;
        // P has one NxVOL block per shift
        std::vector< std::vector< field<fermion> > > P;
        // start from X=0 initial guess, so residual Q = B [NxVOL]
        Q = B;
        // in place thinQR decomposition of residual Q[N][VOL] into
orthonormal Q[N][VOL] and triangular C[NxN] thinQR(Q, C); S = C; AP = Q;
P.push_back(Q);

        // Shifted matrices:
        // previous / inverted versions of beta
        block_matrix beta_inv = block_matrix::Identity(N, N);
        block_matrix beta_inv_m1 = block_matrix::Identity(N, N);
        // Construct decomposition of S: can then do S^-1 A using
S_inv.solve(A) Eigen::ColPivHouseholderQR<block_matrix> S_inv (S);
        block_matrix S_m1 = block_matrix::Zero(N, N);
        // These are temporary matrices used for each shift
        block_matrix tmp_betaC, tmp_Sdag;
        // add a block to P for each shift s
        for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
                // P = Q initially for all shifts
                P.push_back(Q);
        }
        // ksi_s are the scale factors related shifted and unshifted residuals
        // 3-term recurrence so need _k, _k-1, _k-2 for each shift
        // initially ksi_s_m2 = I, ksi_s_m1 = S
        std::vector<block_matrix> ksi_s, ksi_s_m1, ksi_s_m2;
        std::vector< Eigen::ColPivHouseholderQR<block_matrix> > ksi_s_inv_m1,
ksi_s_inv_m2; for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
                ksi_s.push_back(block_matrix::Identity(N, N));
                ksi_s_m1.push_back(S);
                ksi_s_inv_m1.push_back(S_inv);
                ksi_s_m2.push_back(block_matrix::Identity(N, N));
                ksi_s_inv_m2.push_back(block_matrix::Identity(N,
N).colPivHouseholderQr());
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

                // Apply dirac op (with lowest shift absorbed into mass) to
P[0]: for(int i=0; i<N; ++i) { D.DDdagger(AP[i], P[0][i], U);
                        ++iter;
                }

                // in-place thinQRA decomposition of descent matrix P[0] and
AP
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
                beta = beta_inv.ldlt().solve(block_matrix::Identity(N, N));
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
                // note X[0] and P[0] are the unshifted ones, so +1 to i_shift
in X and P for(int i_shift=0; i_shift<N_unconverged_shifts; ++i_shift) {
                        // calculate shifted coefficients
                        // ksi_s:
                        tmp_betaC = S_m1 * beta_inv_m1 - ksi_s_m1[i_shift] *
ksi_s_inv_m2[i_shift].solve(beta_inv_m1); tmp_Sdag = block_matrix::Identity(N,
N) + shifts[i_shift] * beta + tmp_betaC * S_m1.adjoint() * beta;
ksi_s[i_shift] = S * tmp_Sdag.colPivHouseholderQr().solve(ksi_s_m1[i_shift]);
                        // tmp_betaC == "alpha^sigma" in paper:
                        tmp_betaC = beta * S_inv.solve(ksi_s[i_shift]);
                        // tmp_Sdag == "beta^sigma" in paper:
                        tmp_Sdag = tmp_betaC *
ksi_s_inv_m1[i_shift].solve(beta_inv * S.adjoint());
                        // update shifted X and P
                        // X_s = X_s + P_s tmp_betaC
                        for(int i=0; i<N; ++i) {
                                for(int j=0; j<N; ++j) {
                                        X[i_shift+1][i].add(tmp_betaC(j,i),
P[i_shift+1][j]);
                                }
                        }
                        // P_s <- Q + P_s tmp_Sdag (using AP as temporary
storage) for(int i=0; i<N; ++i) { AP[i] = Q[i]; for(int j=0; j<N; ++j) {
                                        AP[i].add(tmp_Sdag(j,i),
P[i_shift+1][j]);
                                }
                        }
                        P[i_shift+1] = AP;
                        // update inverse ksi's for next iteration
                        ksi_s_m2[i_shift] = ksi_s_m1[i_shift];
                        ksi_s_inv_m2[i_shift] = ksi_s_inv_m1[i_shift];
                        ksi_s_m1[i_shift] = ksi_s[i_shift];
                        ksi_s_inv_m1[i_shift].compute(ksi_s[i_shift]);
                        // check if largest unconverged shift has converged
                        if((ksi_s[N_unconverged_shifts-1].rowwise().norm().array()/b_norm).maxCoeff()
< eps) {
                                --N_unconverged_shifts;
                        }
                }

                // use maximum over vectors of residuals/b_norm for unshifted
solution as stopping crit
                // assumes that all shifts are positive i.e. better
conditioned and converge faster
                // worst vector should be equal to CG with same eps on that
vector, others will be better residual =
(C.rowwise().norm().array()/b_norm).maxCoeff();

                // [debugging] find eigenvalues of C

                //Eigen::SelfAdjointEigenSolver<block_matrix> saes;
                //saes.compute(C);
                //Eigen::ArrayXd evals = saes.eigenvalues();

                        // Output iteration number and worst residual for each
shift std::cout << "#SBCGrQ " << iter << "\t" << residual; for(int i_shift=0;
i_shift<N_shifts; ++i_shift) { std::cout << "\t" <<
(ksi_s[i_shift].rowwise().norm().array()/b_norm).maxCoeff();
                }
                std::cout << std::endl;
        }
        return iter;
}

// Bartels–Stewart: O(N^3) algorithm to solve Sylvester equation for X:
// AX + XB = C
// (NB there also exists the Hessenberg-Schur method that only requires one
Schur decomposition) void bartels_stewart(block_matrix& X, const block_matrix&
A, const block_matrix& B, const block_matrix& C) { int N = X.rows();
        // Compute Schur form U T U^dag of A and B
        Eigen::ComplexSchur<block_matrix> schurA(A);
        Eigen::ComplexSchur<block_matrix> schurB(B);

        // Can now transform equation to the form
        // TA X_tilde + X_tilde TB = C_tilde
        // where TA, TB are upper triangular,
        // X_tilde = UA^dag X UB
        // C_tilde = UA^dag C UB
        block_matrix C_tilde = schurA.matrixU().adjoint() * C *
schurB.matrixU(); block_matrix X_tilde = block_matrix::Zero(N, N);

        // Solve triangular system by back substitution
        // consider k-th vector of transformed equation
        for(int k=0; k<N; ++k) {
                // do subtractions from C_tilde
                block_matrix C_sub = C_tilde.col(k);
                for(int l=0; l<k; ++l) {
                        C_sub -= schurB.matrixT()(l,k) * X_tilde.col(l);
                }
                // do back substitution to solve for X_tilde^(k)
                for(int i=N-1; i>=0 ; --i) {
                        X_tilde(i,k) = C_sub(i);
                        for(int j=i+1; j<N; j++) {
                                X_tilde(i,k) -= schurA.matrixT()(i,j) *
X_tilde(j,k);
                        }
                        X_tilde(i,k) /= (schurA.matrixT()(i,i) +
schurB.matrixT()(k,k));
                }
        }
        // Transform solution X_tilde back to X
        X = schurA.matrixU() * X_tilde * schurB.matrixU().adjoint();
}
*/