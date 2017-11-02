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

	double s = std::accumulate(vec.begin(), vec.end(), 0.0,
		[mean](double partial_result, double value){ 
			return partial_result + (value - mean) * (value - mean);
		} );
	/*
	// above is equivalent to:
	double s = 0.;
	for (unsigned int i=0; i<vec.size(); i++) {
		s += (vec[i] - mean)*(vec[i] - mean);
	}
	*/
	return sqrt(s)/static_cast<double>(vec.size());
}

int main(int argc, char *argv[]) {

    if (argc != 6) {
        std::cout << "This program requires 5 arguments:" << std::endl;
        std::cout << "beta mass mu_I n_integrator_steps seed" << std::endl;
        std::cout << "e.g. ./hmc 4.4 0.25 0.10 17 12345" << std::endl;
        return 1;
    }

	// HMC parameters
	hmc_params hmc_pars;
	hmc_pars.beta = atof(argv[1]);
	hmc_pars.mass = atof(argv[2]);
	hmc_pars.mu_I = atof(argv[3]);
	hmc_pars.tau = 1.0;
	hmc_pars.n_steps = atof(argv[4]);
	hmc_pars.MD_eps = 1.e-6;
	hmc_pars.seed = atoi(argv[5]);

	// make 4^4 lattice
	lattice grid (4);

	std::cout.precision(17);

	std::cout << "# RHMC run with parameters:" << std::endl;
	std::cout << "# L\t" << grid.L0 << std::endl;
	std::cout << "# beta\t" << hmc_pars.beta << std::endl;
	std::cout << "# mass\t" << hmc_pars.mass << std::endl;
	std::cout << "# mu_I\t" << hmc_pars.mu_I << std::endl;
	std::cout << "# tau\t" << hmc_pars.tau << std::endl;
	std::cout << "# n_steps\t" << hmc_pars.n_steps << std::endl;
	std::cout << "# MD epsilon\t" << hmc_pars.MD_eps << std::endl;
	std::cout << "# seed\t" << hmc_pars.seed << std::endl;

	// make U[mu] field on lattice
	field<gauge> U (grid);
	// initialise Dirac Op
	dirac_op D (grid);

	// Initialise HMC
	hmc hmc (hmc_pars);

	// start from random gauge field: 2nd param = roughness
	// so 0.0 gives unit gauge links, large value = random
	hmc.random_U (U, 0.5);

	int n_traj = 1000; //10000;
	int n_therm = 100; //500
	int n_block = 100;
	int acc = 0;
	std::vector<double> plq;
	std::vector<double> tmpplq;
	std::vector<double> dE;
	std::vector<double> tmpdE;
	std::vector<double> expdE;
	std::vector<double> tmpexpdE;
	std::vector<double> poly;
	std::vector<double> tmppoly;
	std::vector<double> pbp;
	std::vector<double> tmppbp;

	for(int i=0; i<n_traj+n_therm; ++i) {

		acc += hmc.trajectory (U, D);
		if(i>n_therm) {
			tmpplq.push_back(hmc.plaq(U));
			tmppoly.push_back(hmc.polyakov_loop(U));
			tmpdE.push_back(hmc.deltaE);
			tmpexpdE.push_back(exp(-hmc.deltaE));
			tmppbp.push_back(hmc.chiral_condensate(U, D));
//			std::cout << "#dE:\t" << new_ac - old_ac << std::endl;
			if(i%n_block==0) {
				plq.push_back(av(tmpplq));
				tmpplq.clear();
				poly.push_back(av(tmppoly));
				tmppoly.clear();
				dE.push_back(av(tmpdE));
				tmpdE.clear();
				expdE.push_back(av(tmpexpdE));
				tmpexpdE.clear();
				pbp.push_back(av(tmppbp));
				tmppbp.clear();
				//std::cout << deltaE << std::endl;	
			}
		}
		//std::cout << "#action U, P, total:\t" << action_U(U, beta) << "\t" << action_P(P) << "\t" << action(U, P, beta) << std::endl;
		std::cout << "iter " << i-n_therm << "/" << n_traj<< "\tplaq: " << hmc.plaq(U) << "\t acc: " << acc/static_cast<double>(i+1) << std::endl;
		//std::cout << hmc.plaq(U) << "\t" << hmc.chiral_condensate(U, D) << std::endl;
	}
	//std::cout << "UU^dagger: " << U[34][2]*U[34][2].adjoint() << std::endl;
	//std::cout << "Det[U]: " << U[34][2].determinant() << std::endl;

	std::cout 	<< hmc_pars.tau/hmc_pars.n_steps << "\t" 
				<< 1.0-av(expdE) << "\t" << std_err(expdE) << "\t" 
				<< av(dE) << "\t" << std_err(dE) << "\t" 
				<< av(plq) << "\t" << std_err(plq) << "\t" 
				<< av(poly) << "\t" << std_err(poly) << "\t"
				<< av(pbp) << "\t" << std_err(pbp) << "\t"
		 		<< acc/static_cast<double>(n_traj+n_therm) << std::endl;

	return(0);
}
