#include "hmc.hpp"
#include "dirac_op.hpp"
#include "io.hpp"
#include "stats.hpp"
#include <iostream>
#include <random>
#include <chrono>

// returns sqrt(\sum_i |lhs_i-rhs_i|^2)
double is_field_equal (const field<fermion>& lhs, const field<fermion>& rhs) {
	double norm = 0.0;
	for(int ix=0; ix<lhs.V; ++ix) {
		norm += (lhs[ix]-rhs[ix]).squaredNorm();
	}
	return sqrt(norm);
}

int main(int argc, char *argv[]) {

    if (argc-1 != 2) {
        std::cout << "This program requires 2 arguments:" << std::endl;
        std::cout << "config_name n_RHS" << std::endl;
        std::cout << "e.g. ./CG conf_20_40_m0.002_b5.144 12" << std::endl;
        return 1;
    }

	std::string config_name(argv[1]);
	int N = static_cast<int>(atof(argv[2]));

	hmc_params hmc_params = {
		5.144, 	// beta
		0.0134,	// mass
		0.000, 	// mu_I
		1.0, 	// tau
		7, 		// n_steps
		1.e-6,	// MD_eps
		1234,	// seed
		false,	// constrained HMC
		3.0, 	// suscept_central
		0.05	// suscept_eps
	};
	hmc hmc (hmc_params);

	lattice grid (32, 16, 16, 16);
	field<gauge> U (grid);
	//read_massimo_gauge_field(U, config_name);
	read_gauge_field(U, config_name, 1);
	log("Spatial plaquette", hmc.plaq_spatial(U));
	log("Timelike plaquette", hmc.plaq_timelike(U));

	// initialise Dirac Op
	dirac_op D (grid);
	double eps = 1.e-15;

	std::cout.precision(17);
	log("CG test program");
	log("L0", grid.L0);
	log("L1", grid.L1);
	log("L2", grid.L2);
	log("L3", grid.L3);
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

    auto timer_start = std::chrono::high_resolution_clock::now();
	// x_i = (DD')^-1 chi_i
	int iterBLOCK = D.cg_block(x, chi, U, hmc_params.mass, hmc_params.mu_I, eps);
    auto timer_stop = std::chrono::high_resolution_clock::now();
    auto timer_count = std::chrono::duration_cast<std::chrono::milliseconds>(timer_stop-timer_start).count();

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
	std::cout << N << "\t" << iterCG << "\t" << iterBLOCK << "\t" << timer_count << std::endl;
	return(0);
}
