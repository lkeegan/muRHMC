#include <iostream>
#include <random>
#include "dirac_op.hpp"
#include "hmc.hpp"
#include "inverters.hpp"
#include "io.hpp"
#include "stats.hpp"

int main(int argc, char *argv[]) {
  std::cout.precision(17);

  if (argc - 1 != 5) {
    std::cout << "This program requires 5 arguments:" << std::endl;
    std::cout << "mass mu_I base_name config_number relative_error"
              << std::endl;
    std::cout << "e.g. ./hmc 0.14 0.25 mu0.25_sus_3.1_3.3 1 1e-4" << std::endl;
    return 1;
  }

  double mass = atof(argv[1]);
  double mu_I = atof(argv[2]);
  std::string base_name(argv[3]);
  int config_number = static_cast<int>(atof(argv[4]));
  int N_eigenvalues = N_rhs;  // static_cast<int>(atof(argv[5]));
  double eps = atof(argv[5]);

  lattice grid(8, 8, true);
  // field<fermion>::eo_storage_options eo_storage = field<fermion>::FULL;
  field<fermion>::eo_storage_options eo_storage = field<fermion>::EVEN_ONLY;

  log("Eigenvalues measurement run with parameters:");
  log("T", grid.L0);
  log("L", grid.L1);
  log("mass", mass);
  log("mu_I", mu_I);
  log("N_eigenvalues", N_eigenvalues);
  log("relative_error", eps);

  field<gauge> U(grid);
  read_gauge_field(U, base_name, config_number);
  dirac_op D(grid, mass, mu_I);
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
  while ((lambda_max_err / lambda_max) > eps) {
    for (int i = 0; i < 8; ++i) {
      x /= x_norm;
      D.DDdagger(x2, x, U);
      x2 /= x2.norm();
      D.DDdagger(x, x2, U);
      x_norm = x.norm();
      iter += 2;
    }
    lambda_max = x2.real_dot(x);
    lambda_max_err = sqrt(x_norm * x_norm - lambda_max * lambda_max);
  }
  // since lambda_max is a lower bound, and lambda_max + lamda_max_err is an
  // upper bound:
  std::cout << "iterations" << iter << std::endl;
  std::cout << "final_lambda_max" << lambda_max + 0.5 * lambda_max_err << "\t"
            << 0.5 * lambda_max_err << std::endl;

  return (0);
}
