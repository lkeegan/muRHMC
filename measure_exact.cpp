#include "hmc.hpp"
#include "dirac_op.hpp"
#include "io.hpp"
#include "stats.hpp"
#include <iostream>
#include <random>

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

	std::cout << "# Exact measurement run with parameters:" << std::endl;
	std::cout << "# L\t" << grid.L0 << std::endl;
	std::cout << "# mass\t" << mass << std::endl;
	std::cout << "# mu_I\t" << mu_I << std::endl;
	// make U[mu] field on lattice
	field<gauge> U (grid);
	// initialise Dirac Op
	dirac_op D (grid, mass, mu_I);

	// read philippe gauge config
	// Initialise HMC
	/*
	hmc_params hmc_pars;
	hmc hmc (hmc_pars);
	read_fortran_gauge_field(U, "fort.1");
	std::cout << "# FORTRAN GAUGE CONFIG PLAQ: " << hmc.plaq(U) << std::endl;
	std::cout << "# FORTRAN GAUGE CONFIG POLY: " << hmc.polyakov_loop(U) << std::endl;
	*/

	for(int i=n_initial; ; ++i) {
		read_gauge_field(U, base_name, i);

		// Calculate all eigenvalues lambda_i of Dirac op:
		Eigen::MatrixXcd eigenvalues = D.D_eigenvalues(U);
		// Det[D] = \prod_i \lambda_i
		// phase{D} = \sum_i phase{lambda_i}
		double sum = 0;
		for (int i=0; i<eigenvalues.size(); ++i) {
			//std::cout << std::arg(eigenvalues(i)) << std::endl;
			sum += std::arg(eigenvalues(i));
		}
		log("[evals] det-phase", sum);
		//log("[evals]phase complex-det", eigenvalues.prod()/static_cast<double>(3*U.V));
		//double phase_det = std::arg(eigenvalues.prod());
		//log("[evals] det-phase", phase_det);

		// Trace[D^-1] = \sum_i \lambda_i^-1:
		log("[evals] psibar-psi", eigenvalues.cwiseInverse().sum()/static_cast<double>(3*U.V));

		// Calculate all eigenvalues of DDdagger op:
		Eigen::MatrixXcd eigenvaluesDDdag = D.DDdagger_eigenvalues(U);
		log("[evals] pion-suscept", eigenvaluesDDdag.cwiseInverse().sum().real()/static_cast<double>(3*U.V));
		log("[evals] mineval-DDdag", eigenvaluesDDdag.real().minCoeff());
		log("[evals] maxeval-DDdag", eigenvaluesDDdag.real().maxCoeff());
	}
	return(0);
}
