#ifndef LKEEGAN_MURHMC_INVERTERS_H
#define LKEEGAN_MURHMC_INVERTERS_H
#include "4d.hpp"
#include "dirac_op.hpp"
#include "su3.hpp"

// Construct R_ij = X_i^dag Y_j assuming that the result is hermitian (e.g. Y =
// X, or Y = MX with M hermitian), so only does the dot products needed for the
// lower triangular part of matrix
void hermitian_dot(const field<block_fermion>& X, const field<block_fermion>& Y,
                   block_matrix& R);

// in-place thinQR decomposition of Q, making Q orthonormal, R is upper
// triangular matrix such that Q^{new} R = Q^{old}
void thinQR(field<block_fermion>& Q, block_matrix& R);

// in-place QR A-orthonormalisation of V and AV, rotates V and AV such that
// [V^dag AV] = I:
void thinQRA(field<block_fermion>& V, field<block_fermion>& AV,
             block_matrix& R);

// A-orthogonalise X in place, without having AX already, i.e. does a bunch of
// DDdag operations Return the eigenvalues of the hermitian matrix <X_i|A|X_j>
// in first column of Evals [eigenvalues] And the square root of the diagonal
// elemets only of the hermitian matrix <X_i|A^2|X_j> in 2nd column [optimistic
// error estimates]
void thinQRA_evals(field<block_fermion>& X, Eigen::MatrixXd& Evals,
                   field<gauge>& U, dirac_op& D);

// CG inversion of D D^{\dagger} x = b: given b solves for x
// returns number of times Dirac operator was called
int cg(field<fermion>& x, const field<fermion>& b, field<gauge>& U, dirac_op& D,
       double eps);

// As above but inverts shifted (DD^dag + shift): for debugging only
int cg_singleshift(field<fermion>& x, const field<fermion>& b, field<gauge>& U,
                   double shift, dirac_op& D, double eps);

// CG multishift inversion of (D D^{\dagger} + sigma_a^2) x_a = b: given b
// solves for x_a for each shift sigma_a returns number of times Dirac operator
// was called note larger shifts converge faster: currently only stop iterating
// a given shift if residual is zero to avoid NaN issues could add vector of eps
// values for each shift in the future
int cg_multishift(std::vector<field<fermion>>& x, const field<fermion>& b,
                  field<gauge>& U, std::vector<double>& sigma, dirac_op& D,
                  double eps, double eps_shifts = 1.e-15, int max_iter = 1e9);

int rational_approx_cg_multishift(field<fermion>& x, const field<fermion>& b,
                                  field<gauge>& U,
                                  std::vector<double>& rational_alpha,
                                  std::vector<double>& rational_beta,
                                  dirac_op& D, double eps);

// BlockCG[A][dQ/dQA][rQ] as described in arXiv:1710.09745
int cg_block(field<block_fermion>& X, const field<block_fermion>& B,
             field<gauge>& U, dirac_op& D, double eps, bool BCGA, bool dQ,
             bool dQA, bool rQ, bool OUTPUT_ERROR_NORMS = false,
             field<fermion>* x0_star = nullptr);

// Shifted Block CG rQ:
// NB assumes vector of shifts are positive and in ascending order
int SBCGrQ(std::vector<field<block_fermion>>& X, const field<block_fermion>& B,
           field<gauge>& U, std::vector<double>& sigma, dirac_op& D, double eps,
           double eps_shifts = 1.e-15, int max_iter = 1e9);

// Shifted Block CG rQ:
// NB assumes vector of shifts are positive and in ascending order
int SBCGrQ_old(std::vector<field<block_fermion>>& X,
               const field<block_fermion>& B, field<gauge>& U,
               std::vector<double>& input_shifts, dirac_op& D, double eps,
               double eps_shifts = 1.e-15, int max_iter = 1e9);

int rational_approx_SBCGrQ(field<block_fermion>& X,
                           const field<block_fermion>& B, field<gauge>& U,
                           std::vector<double>& rational_alpha,
                           std::vector<double>& rational_beta, dirac_op& D,
                           double eps, double eps_shifts = 1.e-14,
                           int max_iter = 1e9);

int SBCGAdQArQ(std::vector<std::vector<field<fermion>>>& X,
               const std::vector<field<fermion>>& B, field<gauge>& U,
               std::vector<double>& input_shifts, dirac_op& D, double eps);

// Bartels–Stewart - O(N^3) algorithm to solve Sylvester equation for X:
// AX + XB = C
// Computes Schur form of A and B, then transforms equation to triangular form,
// solves by back substitution, then transforms solution back to give X
// void bartels_stewart (block_matrix& X, const block_matrix& A, const
// block_matrix& B, const block_matrix& C);

#endif  // LKEEGAN_MURHMC_INVERTERS_H