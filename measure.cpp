#include "hmc.hpp"
#include "dirac_op.hpp"
#include "io.hpp"
#include <iostream>
#include <random>

// return average of values in vector
double av(std::vector<double> &vec) {
	double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
	return sum/static_cast<double>(vec.size());
}

// return std error of average
double std_err(std::vector<double> &vec) {
	double mean = av(vec);

	double s = 0.;
	for (unsigned int i=0; i<vec.size(); i++) {
		s += (vec[i] - mean)*(vec[i] - mean);
	}
	return sqrt(s)/static_cast<double>(vec.size());
}

int main(int argc, char *argv[]) {

    if (argc != 5) {
        std::cout << "This program requires 4 arguments:" << std::endl;
        std::cout << "beta mass mu_I seed" << std::endl;
        std::cout << "e.g. ./hmc 4.4 0.14 0.25 12345" << std::endl;
        return 1;
    }

	// HMC parameters
	hmc_params hmc_pars;
	hmc_pars.beta = atof(argv[1]);
	hmc_pars.mass = atof(argv[2]);
	hmc_pars.mu_I = atof(argv[3]);
	hmc_pars.tau = 1.0;
	hmc_pars.n_steps = 1;
	hmc_pars.MD_eps = 1.e-6;
	hmc_pars.seed = atoi(argv[4]);

	// make 4^4 lattice
	lattice grid (4);

	std::cout.precision(17);

	std::cout << "# Measurement run with parameters:" << std::endl;
	std::cout << "# L\t" << grid.L0 << std::endl;
	std::cout << "# beta\t" << hmc_pars.beta << std::endl;
	std::cout << "# mass\t" << hmc_pars.mass << std::endl;
	std::cout << "# mu_I\t" << hmc_pars.mu_I << std::endl;
	std::cout << "# seed\t" << hmc_pars.seed << std::endl;

	// make U[mu] field on lattice
	field<gauge> U (grid);
	// initialise Dirac Op
	dirac_op D (grid);

	field<fermion> phi (U.grid);
	field<fermion> chi(U.grid);
	field<fermion> psi(U.grid);

	// Initialise HMC
	hmc hmc (hmc_pars);

	// read philippe gauge config
	read_fortran_gauge_field(U, "fort.1");
	std::cout << "FORTRAN GAUGE CONFIG PLAQ: " << hmc.plaq(U) << std::endl;
	std::cout << "FORTRAN GAUGE CONFIG POLY: " << hmc.polyakov_loop(U) << std::endl;

	int n_meas = 10000; //10000;
	int n_block = 1;
	int acc = 0;
	double eps = 1.e-10;
	std::vector<double> pbp(n_meas);
	std::vector<double> pbp_block(n_block);
	std::vector<double> actionF(n_meas);
	std::vector<double> actionF_block(n_block);

	for(int i_meas=0; i_meas<n_meas; ++i_meas) {
		for(int i_block=0; i_block<n_block; ++i_block) {
			hmc.gaussian_fermion(phi);
			// chi = [D(mu,m)D(mu,m)^dag]-1 phi	
			D.cg(chi, phi, U, hmc_pars.mass, hmc_pars.mu_I, eps);
			actionF_block[i_block] = phi.dot(chi).real() / static_cast<double>(phi.V);
			// psi = D(mu,m)^dag chi = - D(-mu,-m) chi = D(mu)^-1 phi
			D.D(psi, chi, U, -hmc_pars.mass, -hmc_pars.mu_I);
			pbp_block[i_block] = -phi.dot(psi).real() / static_cast<double>(phi.V);
		}
		pbp[i_meas] = av(pbp_block);
		actionF[i_meas] = av(actionF_block);
	}

	std::cout 	<< hmc_pars.mu_I << "\t" 
				<< av(pbp) << "\t" << std_err(pbp) << "\t"
				<< av(actionF) << "\t" << std_err(actionF) << std::endl;

	return(0);
}
