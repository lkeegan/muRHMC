#include <iostream>
#include <random>
#include "dirac_op.hpp"
#include "hmc.hpp"
#include "inverters.hpp"
#include "io.hpp"
#include "stats.hpp"

int main(int argc, char *argv[]) {
  std::cout.precision(17);

  constexpr int n_args = 4;
  if (argc - 1 != n_args) {
    std::cout << "This program requires " << n_args
              << " arguments:" << std::endl;
    std::cout << "L, mass, base_name, config_number" << std::endl;
    std::cout << "e.g. ./eigenvalues 8 0.002 conf 1" << std::endl;
    return 1;
  }

  int L = static_cast<int>(atof(argv[1]));
  double mass = atof(argv[2]);
  std::string base_name(argv[3]);
  int config_number = static_cast<int>(atof(argv[4]));
  int N_eigenvalues = N_rhs;

  lattice grid(L, L, true);
  // field<fermion>::eo_storage_options eo_storage = field<fermion>::FULL;
  field<fermion>::eo_storage_options eo_storage = field<fermion>::EVEN_ONLY;

  log("Eigenvalues measurement run with parameters:");
  log("T", grid.L0);
  log("L", grid.L1);
  log("mass", mass);
  log("N_eigenvalues", N_eigenvalues);

  field<gauge> U(grid);
  read_gauge_field(U, base_name, config_number);
  dirac_op D(grid, mass);
  rhmc_params rhmc_pars;
  rhmc_pars.seed = 123;
  rhmc rhmc(rhmc_pars, grid);

  // Power method gives strict lower bound on lambda_max
  // Iterate until relative error < eps

  field<fermion> x(grid, eo_storage), x2(grid, eo_storage);
  rhmc.gaussian_fermion(x);
  double x_norm = x.norm();
  double lambda_max = 1;
  double lambda_max_err = 100;
  int iter = 0;
  while ((lambda_max_err / lambda_max) > 1e-6) {
    for (int i = 0; i < 8; ++i) {
      x /= x_norm;
      D.DDdagger(x2, x, U);
      x2 /= x2.norm();
      D.DDdagger(x, x2, U);
      x_norm = x.norm();
      iter += 2;
    }
    lambda_max = x2.dot(x).real();
    lambda_max_err = sqrt(x_norm * x_norm - lambda_max * lambda_max);
    // log("lambda_max", lambda_max, lambda_max_err);
  }
  // since lambda_max is a lower bound, and lambda_max + lamda_max_err is an
  // upper bound:
  // log("iterations", iter);
  // log("final_lambda_max", lambda_max + 0.5 * lambda_max_err,
  //  0.5 * lambda_max_err);

  // Find N lowest eigenvalues of DDdagger.
  // Uses chebyshev acceleration as described in Appendix A of hep-lat/0512021
  block_matrix R = block_matrix::Zero();
  Eigen::MatrixXd Evals = Eigen::MatrixXd::Zero(N_eigenvalues, 2);

  // make initial fermion vector basis of gaussian noise vectors
  field<block_fermion> X(grid, field<block_fermion>::EVEN_ONLY);
  rhmc.gaussian_fermion(X);
  // orthonormalise X
  thinQR(X, R);
  // make X A-orthormal and get eigenvalues of matrix <X_i AX_j>
  thinQRA_evals(X, Evals, U, D);

  double delta = 1.0;  // change from last estimate of nth eigenvalue
  // v is the upper bound on possible eigenvalues
  // use 50% margin of safety on estimate of error to get safe upper bound
  double v = lambda_max + 1.5 * lambda_max_err;
  while (delta > 1.e-10) {
    // get current estimates of min/max eigenvalues
    double lambda_min = Evals.col(0)[0];
    double lambda_N = Evals.col(0)[N_eigenvalues - 1];
    // find optimal chebyshev order k
    int k = 2 * std::ceil(0.25 * sqrt(((v - lambda_min) * 8.33633354 -
                                       (v - lambda_N) * 3.1072776) /
                                      (lambda_N - lambda_min)));
    // find chebyshev lower bound
    double u = lambda_N + (v - lambda_N) * (tanh(1.76275 / (2.0 * k))) *
                              tanh(1.76275 / (2.0 * k));
    log("Chebyshev order k:", k);
    log("Chebyshev range u:", u);
    log("Chebyshev range v:", v);
    // apply chebyshev polynomial in DDdagger
    D.chebyshev(k, u, v, X, U);
    // orthonormalise X
    thinQR(X, R);
    // make X A-orthormal and get eigenvalues of matrix <X_i AX_j>
    thinQRA_evals(X, Evals, U, D);
    // note the estimated errors for the eigenvalues are only correct
    // if the eigevalues are separated more than the errors
    // i.e. assumes residuals matrix is diagonal
    // if this is not the case then errors are underestimated
    delta = fabs((lambda_N - Evals.col(0)[N_eigenvalues - 1]) / lambda_N);
    log("relative change:", delta);
    // std::cout << Evals << std::endl;
  }

  std::cout << 0 << "\t" << sqrt(lambda_max + 0.5 * lambda_max_err)
            << std::endl;
  for (int i = 0; i < N_eigenvalues; ++i) {
    std::cout << i + 1 << "\t" << sqrt(Evals.col(0)[i]) << std::endl;
  }
  return (0);
}
