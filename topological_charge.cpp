#include <iostream>
#include "hmc.hpp"
#include "io.hpp"
#include "stats.hpp"

int main(int argc, char *argv[]) {
  if (argc - 1 != 4) {
    std::cout << "This program requires 4 arguments:" << std::endl;
    std::cout << "base_name initial_config rho n_smear" << std::endl;
    std::cout << "e.g. ./topological_charge beta5.2_m0.002_npf3 1 0.02 50"
              << std::endl;
    return 1;
  }

  std::string base_name(argv[1]);
  int n_initial = static_cast<int>(atof(argv[2]));
  double rho = atof(argv[3]);
  int n_smear = static_cast<int>(atof(argv[4]));

  hmc_params hmc_pars;
  hmc_pars.seed = 123;

  lattice grid(8);
  hmc hmc(hmc_pars);
  field<gauge> U(grid);
  std::cout.precision(12);

  log("Topological charge measurements with parameters:");
  log("L", grid.L0);
  log("rho", rho);
  log("n_smear", n_smear);

  for (int i = n_initial;; i += 1) {
    read_gauge_field(U, base_name, i);
    for (int i_smear = 0; i_smear < n_smear; ++i_smear) {
      hmc.stout_smear(rho, U);
    }
    std::cout << hmc.topological_charge(U) << std::endl;
  }
  return (0);
}
