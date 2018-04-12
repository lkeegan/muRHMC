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
	log("max_n_smear", n_smear);

	log("Data format: config number,\tsmearing_steps,\tunsmeared plaquette,\tre,im part of smeared polyakov loop in time, ditto for 3x spatial directions");
	for(int i=n_initial; ; i+=1) {
		read_gauge_field(U, base_name, i);
		double plq = hmc.plaq(U);
		for(int i_smear=0; i_smear<n_smear; ++i_smear) {
			std::cout << i << "\tn_smear= " << i_smear << "\t" << std::scientific << plq << "\t";
			for(int mu=0; mu<4; ++mu) {
				std::complex<double> ply = hmc.polyakov_loop(U, mu);
			  	std::cout << std::scientific << ply.real() << "\t" << ply.imag() << "\t";
			}
			std::cout << std::endl;
			hmc.stout_smear(rho, U);
		}
	}
	return(0);
}
