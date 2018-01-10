#include "hmc.hpp"
#include "dirac_op.hpp"
#include "io.hpp"
#include "stats.hpp"
#include <iostream>
#include <random>

// returns sqrt(\sum_i |lhs_i-rhs_i|^2)
double is_field_equal (const field<fermion>& lhs, const field<fermion>& rhs) {
	double norm = 0.0;
	for(int ix=0; ix<lhs.V; ++ix) {
		norm += (lhs[ix]-rhs[ix]).squaredNorm();
	}
	return sqrt(norm);
}

int main(int argc, char *argv[]) {

    if (argc-1 != 1) {
        std::cout << "This program requires 1 argument:" << std::endl;
        std::cout << "number of rhs's in block CG" << std::endl;
        std::cout << "e.g. ./CG 8" << std::endl;
        return 1;
    }

	int N = static_cast<int>(atof(argv[1]));
	hmc_params hmc_params = {
		5.4, 	// beta
		0.01, 	// mass
		0.000, 	// mu_I
		1.0, 	// tau
		7, 		// n_steps
		1.e-6,	// MD_eps
		1234,	// seed
		false,	// constrained HMC (fixed allowed range for pion susceptibility)
		3.0, 	// suscept_central
		0.05	// suscept_eps
	};

	// make 4^4 lattice
	lattice grid (4);
	// make U[mu] field on lattice
	field<gauge> U (grid);
	hmc hmc (hmc_params);
	hmc.random_U(U, 10.0);
	// initialise Dirac Op
	dirac_op D (grid);
	double eps = 1.e-15;

	std::cout.precision(17);
	log("CG test program");
	log("L", grid.L0);
	log("mass", hmc_params.mass);
	log("mu_I", hmc_params.mu_I);
	log("eps", eps);
	log("N_block", N);

	// make vector of N fermion fields for random chi and empty Q, x
	std::vector<field<fermion>> x, Q, chi;
	field<fermion> tmp_chi (grid);
	for(int i=0; i<N; ++i) {
		x.push_back(field<fermion>(grid));
		Q.push_back(field<fermion>(grid));
		hmc.gaussian_fermion (tmp_chi);
		chi.push_back(tmp_chi);
	}
	field<fermion> y (grid), z(grid);

	// x_i = (DD')^-1 chi_i
	int iterBLOCK = D.cg_block(x, chi, U, hmc_params.mass, hmc_params.mu_I, eps);

	Eigen::MatrixXcd R = Eigen::MatrixXcd::Zero(N, N);
	D.thinQR(Q, R, chi);

	int iterCG = 0;
	for(int i=0; i<N; ++i) {
		// y = (DD')^-1 chi_i
		iterCG += D.cg(y, chi[i], U, hmc_params.mass, hmc_params.mu_I, eps);

		D.DDdagger(z, y, U, hmc_params.mass, hmc_params.mu_I);
		double devCG = is_field_equal(z, chi[i]);

		D.DDdagger(z, x[i], U, hmc_params.mass, hmc_params.mu_I);
		double devBLOCK = is_field_equal(z, chi[i]);
		log("CG dev", devCG);
		log("BlockCG dev", devBLOCK);
	}

	log("CG iterations", iterCG);
	log("BlockCG iterations", iterBLOCK);
	return(0);
}
