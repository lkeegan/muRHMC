#include "hmc.hpp"
#include "dirac_op.hpp"
#include "io.hpp"
#include "stats.hpp"
#include <iostream>
#include <random>
#include <Eigen/Eigenvalues>
// some default parameters for the HMC
hmc_params hmc_pars = {
	5.4, 	// beta
	0.292, 	// mass
	0.157, 	// mu_I
	1.0, 	// tau
	7, 		// n_steps
	1.e-6,	// MD_eps
	1234,	// seed
	false,	// constrained HMC (fixed allowed range for pion susceptibility)
	3.0, 	// suscept_central
	0.05	// suscept_eps
};

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

	hmc hmc (hmc_pars);

	for(int i_config=n_initial; ; ++i_config) {
		read_gauge_field(U, base_name, i_config);
/*
		// Calculate all eigenvalues lambda_i of Dirac op:
		Eigen::MatrixXcd eigenvalues = D.D_eigenvalues (U, mass, mu_I);				
		for (int i=0; i<eigenvalues.size(); ++i) {
			std::cout << i_config << "\t" << eigenvalues(i).real() << "\t" << eigenvalues(i).imag() << std::endl;
		}
		std::cout << eigenvalues.imag().minCoeff() << "\t" << eigenvalues.imag().maxCoeff() << std::endl;
*/
		//Noise vector approach to min/max eigenvalue bounds:
		field<fermion> psi1 (grid), psi2(grid);
		hmc.gaussian_fermion(psi1);

		psi1 /= sqrt(psi1.squaredNorm());
		double lambda_max = 0;
		for(int i=0; i<100; ++i) {
			D.DDdagger(psi2, psi1, U, mass, mu_I);
			//lambda_max = psi1.dot(psi2).real();
			psi2 /= sqrt(psi2.squaredNorm());
			D.DDdagger(psi1, psi2, U, mass, mu_I);
			lambda_max = psi2.dot(psi1).real();
			psi1 /= sqrt(psi1.squaredNorm());
			std::cout << "MAX: " << lambda_max << std::endl;
		}

		double c = 0.55*lambda_max + 0.55*mass*mass; // need c > 0.5(lambda_max + lambda_min)
		hmc.gaussian_fermion(psi1);
		psi1 /= sqrt(psi1.squaredNorm());
		for(int i=0; i<1000000; ++i) {
			// psi2 = (c - A) psi1
			D.DDdagger(psi2, psi1, U, mass, mu_I);
			psi2.scale_add(-1.0, c, psi1);
			lambda_max = psi1.dot(psi2).real();
			// normalise psi2
			std::cout << "MIN: " << c - lambda_max << std::endl;
			psi2 /= sqrt(psi2.squaredNorm());
			// psi1 = (c - A) psi2
			D.DDdagger(psi1, psi2, U, mass, mu_I);
			psi1.scale_add(-1.0, c, psi2);
			lambda_max = psi1.dot(psi2).real();
			// normalise psi1
			psi1 /= sqrt(psi1.squaredNorm());
			std::cout << "MIN: " << c - lambda_max << std::endl;
		}

	}
	return(0);
}
