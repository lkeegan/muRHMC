#include "hmc.hpp"
#include "dirac_op.hpp"
#include "io.hpp"
#include "stats.hpp"
#include <iostream>
#include <random>
#include <Eigen/Eigenvalues>

int main(int argc, char *argv[]) {

    if (argc-1 != 4) {
        std::cout << "This program requires 4 arguments:" << std::endl;
        std::cout << "mass mu_I base_name initial_config" << std::endl;
        std::cout << "e.g. ./hmc 0.14 0.25 mu0.25_sus_3.1_3.3 23" << std::endl;
        return 1;
    }

	double mass = atof(argv[1]);
	double mu_I = atof(argv[2]);
	std::string base_name(argv[3]);
	int n_initial = static_cast<int>(atof(argv[4]));

	// make 4^4 lattice
	lattice grid (4);

	std::cout.precision(12);

	std::cout << "# Eigenvalues measurement run with parameters:" << std::endl;
	std::cout << "# L\t" << grid.L0 << std::endl;
	std::cout << "# mass\t" << mass << std::endl;
	std::cout << "# mu_I\t" << mu_I << std::endl;
	// make U[mu] field on lattice
	field<gauge> U (grid);
	// initialise Dirac Op
	dirac_op D (grid);

	for(int i_config=n_initial; ; ++i_config) {
		read_gauge_field(U, base_name, i_config);

		// Calculate all eigenvalues lambda_i of Dirac op:
		Eigen::MatrixXcd eigenvalues = D.D_eigenvalues (U, mass, mu_I);				
		for (int i=0; i<eigenvalues.size(); ++i) {
			std::cout << i_config << "\t" << eigenvalues(i).real() << "\t" << eigenvalues(i).imag() << std::endl;
		}
	}
	return(0);
}
