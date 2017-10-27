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

    if (argc != 2) {
        std::cout << "This program requires one argument:" << std::endl;
        std::cout << "seed" << std::endl;
        std::cout << "e.g. ./hmc 123" << std::endl;
        return 1;
    }

	unsigned int seed = atoi(argv[1]);
	std::cout.precision(17);

	// make 4^4 lattice
	lattice grid (4);
	// make U[mu] field on lattice
	field<gauge> U (grid);
	// initialise HMC and Dirac Op
	hmc hmc (seed);
	dirac_op D (grid);
	D.mu_I = 0.02;

	read_fortran_gauge_field(U, "fort.1");
	std::cout << "PLAQ " << hmc.plaq(U) << std::endl;

	// start from random gauge field: 2nd param = roughness
	// so 0.0 gives unit gauge links
	hmc.random_U (U, 0.5);

	std::cout << "PLAQ rand1 " << hmc.plaq(U) << std::endl;
	write_gauge_field(U, "tmp.dat");
	hmc.random_U (U, 0.5);
	std::cout << "PLAQ rand2 " << hmc.plaq(U) << std::endl;

	read_gauge_field(U, "tmp.dat");
	std::cout << "PLAQ rand1 " << hmc.plaq(U) << std::endl;

	hmc_params hmc_pars;
	hmc_pars.beta = 5.0;
	hmc_pars.mass = 0.025;
	hmc_pars.tau = 1.0;
	hmc_pars.n_steps = 23;

	int n_traj = 5000;
	int n_therm = 500;
	int n_block = 50;
	int acc = 0;
	std::vector<double> plq;
	std::vector<double> tmpplq;
	std::vector<double> dE;
	std::vector<double> tmpdE;
	std::vector<double> expdE;
	std::vector<double> tmpexpdE;
	std::vector<double> poly;
	std::vector<double> tmppoly;

	for(int i=0; i<n_traj+n_therm; ++i) {

		acc += hmc.trajectory (U, hmc_pars, D);
		if(i>n_therm) {
			//std::cout << "#new action U, P, total:\t" << action_U(U, beta) << "\t" << action_P(P) << "\t" << old_ac << std::endl << std::endl;
			tmpplq.push_back(hmc.plaq(U));
			tmppoly.push_back(hmc.polyakov_loop(U));
			tmpdE.push_back(hmc.deltaE);
			tmpexpdE.push_back(exp(-hmc.deltaE));
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
				//std::cout << deltaE << std::endl;	
			}
		}
		//std::cout << "#action U, P, total:\t" << action_U(U, beta) << "\t" << action_P(P) << "\t" << action(U, P, beta) << std::endl;
		std::cout << hmc.plaq(U) << "\t" << hmc.polyakov_loop(U) << std::endl;
	}
	//std::cout << "UU^dagger: " << U[34][2]*U[34][2].adjoint() << std::endl;
	//std::cout << "Det[U]: " << U[34][2].determinant() << std::endl;

	std::cout 	<< hmc_pars.tau/hmc_pars.n_steps << "\t" 
				<< 1.0-av(expdE) << "\t" << std_err(expdE) << "\t" 
				<< av(dE) << "\t" << std_err(dE) << "\t" 
				<< av(plq) << "\t" << std_err(plq) << "\t" 
				<< av(poly) << "\t" << std_err(poly) << "\t"
		 		<< acc/static_cast<double>(n_traj+n_therm) << std::endl;

	return(0);
}