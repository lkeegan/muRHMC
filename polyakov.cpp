#include "hmc.hpp"
#include "io.hpp"
#include "stats.hpp"
#include <iostream>

int main(int argc, char *argv[]) {

    if (argc-1 != 4) {
        std::cout << "This program requires 4 arguments:" << std::endl;
        std::cout << "base_name initial_config rho n_smear" << std::endl;
        std::cout << "e.g. ./polyakov mu0.25_sus_3.1_3.3 1 0.05 10" << std::endl;
        return 1;
    }

	std::string base_name(argv[1]);
	int n_initial = static_cast<int>(atof(argv[2]));
	double rho = atof(argv[3]);
	int n_smear = static_cast<int>(atof(argv[4]));

	hmc_params hmc_pars;
	hmc_pars.seed = 123;

	// make 12^4 lattice
	lattice grid (12);
	hmc hmc (hmc_pars);
	field<gauge> U (grid);
	std::cout.precision(12);

	log("Polyakov loop measurements with parameters:");
	log("L", grid.L0);
	log("rho", rho);
	log("n_smear", n_smear);

	for(int i=n_initial; ; i+=1) {
		read_gauge_field(U, base_name, i);
		std::complex<double> pT = hmc.polyakov_loop(U);
		std::complex<double> pX = hmc.polyakov_loop_spatial(U);
		std::cout << "polyTIME\t" << i << "\tn_smear =\t" << 0 << "\t" << std::scientific << pT.real() << "\t" << pT.imag() << std::endl;
		std::cout << "polySPACE\t" << i << "\tn_smear =\t" << 0 << "\t" << std::scientific << pX.real() << "\t" << pX.imag() << std::endl;
		for(int i_smear=1; i_smear<n_smear; ++i_smear) {
			hmc.stout_smear(rho, U);
			std::complex<double> pT = hmc.polyakov_loop(U);
			std::complex<double> pX = hmc.polyakov_loop_spatial(U);
			std::cout << "polyTIME\t" << i << "\tn_smear =\t" << i_smear << "\t" << std::scientific << pT.real() << "\t" << pT.imag() << std::endl;
			std::cout << "polySPACE\t" << i << "\tn_smear =\t" << i_smear << "\t" << std::scientific << pX.real() << "\t" << pX.imag() << std::endl;
		}
	}
	return(0);
}
