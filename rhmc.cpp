#include "hmc.hpp"
#include "dirac_op.hpp"
#include "io.hpp"
#include "stats.hpp"
#include <iostream>
#include <iomanip>
#include <random>

int main(int argc, char *argv[]) {
    if (argc-1 != 1) {
        std::cout << "Input file not specified, e.g." << std::endl;
        std::cout << "./rhmc input_file.txt" << std::endl;
        return 1;
    }

	std::cout.precision(14);

	// HMC parameters
	hmc_params hmc_pars;
	run_params run_pars;

	read_input_file(argv[1], hmc_pars, run_pars);

	// make L^4 lattice
	lattice grid (run_pars.L);

	log("");
	if(hmc_pars.constrained) {
		log("Constrained HMC with parameters:");
		log("suscept_central", hmc_pars.suscept_central);
		log("suscept_delta", hmc_pars.suscept_delta);
	}
	else {
		log("Unconstrained HMC with parameters:");
	}
	log("");
	log("beta", hmc_pars.beta);
	log("mass", hmc_pars.mass);
	log("mu_I", hmc_pars.mu_I);
	log("tau", hmc_pars.tau);
	log("n_steps", hmc_pars.n_steps);
	log("MD epsilon", hmc_pars.MD_eps);
	log("RNG seed", hmc_pars.seed);

	log("");
	log("Run parameters:");
	log("");
	log("base_name", run_pars.base_name);
	log("L", grid.L0);
	log("n_traj", run_pars.n_traj);
	log("n_therm", run_pars.n_therm);
	log("n_save", run_pars.n_save);
	log("initial_config", run_pars.initial_config);

	// Initialise HMC
	hmc hmc (hmc_pars);
	if(!hmc_pars.constrained) {
		hmc.suscept = 0;
	}

	// make U[mu] field on lattice
	field<gauge> U (grid);

	// initialise Dirac Op
	dirac_op D (grid);

	if(run_pars.initial_config < 0) {
		log("initial config negative i.e. RANDOM");
		run_pars.initial_config = 0;
		// start from random gauge field: 2nd param = roughness
		// so 0.0 gives unit gauge links, large value = random
		hmc.random_U (U, 0.5);
	}
	else {
		// load specified gauge config
		read_gauge_field (U, run_pars.base_name, run_pars.initial_config);
	}

	// observables to measure
	std::vector<double> dE;
	std::vector<double> expdE;
	std::vector<double> plq;
	std::vector<double> poly_re;
	std::vector<double> poly_im;
/*
	if (hmc_pars.constrained) {
		// pre-thermalisation: find a random U matrix with 
		// pion susceptibility in desired range
		log("");
		log("Pre-pre-thermalisation (find good initial random gauge field):");
		log("");
		double roughness = 0.5;
		double epsilon = 0.05;
		hmc.random_U (U, roughness);
		double sus = D.pion_susceptibility_exact(U, hmc_pars.mass, hmc_pars.mu_I);
		log("Plaquette", hmc.plaq(U));
		log("Pion susceptibility", sus);
		double sus_dev = sus - hmc_pars.suscept_central;
		while( (fabs(sus_dev) > hmc_pars.suscept_delta) ) {
			roughness -= epsilon * (sus_dev / sus);
			hmc.random_U (U, roughness);
			sus = D.pion_susceptibility_exact(U, hmc_pars.mass, hmc_pars.mu_I);
			sus_dev = sus - hmc_pars.suscept_central;
			hmc.suscept = sus;
			log("Plaquette", hmc.plaq(U));
			log("Pion susceptibility", sus);
		}
	}
*/

		// ALTERNATIVE pre-thermalisation: adjust suscept range to force suscept 
		// to go in the right direction until we get a value in the desired range
		// obviously this is only efficient if we are moving towards the value
		// that minimises the action, i.e. the susceptibility is initially too low
	if (hmc_pars.constrained) {
		hmc.suscept = D.pion_susceptibility_exact(U, hmc_pars.mass, hmc_pars.mu_I);
		if (run_pars.n_therm > 0) {
			log("");
			log("Pre-thermalisation (until we get a pion suscept in the desired range):");
			log("");
			// make random U with suscept below desired value
			hmc.random_U (U, 0.5);
			// use larger tau for pre-therm
			hmc.params.tau = 1.0;
			hmc.params.n_steps = 19;
			std::cout << hmc.suscept << std::endl;
			while(hmc.suscept < (hmc_pars.suscept_central - hmc_pars.suscept_delta)) {
				hmc.params.suscept_central = 0.5 * (hmc_pars.suscept_central + hmc_pars.suscept_delta + hmc.suscept);
				hmc.params.suscept_delta = 0.5 * (hmc_pars.suscept_central + hmc_pars.suscept_delta - hmc.suscept);
				hmc.trajectory (U, D);
				std::cout << hmc.suscept << std::endl;
			}
			/*
			while(hmc.suscept > (hmc_pars.suscept_central + hmc_pars.suscept_delta)) {
				hmc.params.suscept_central = 0.0;
				hmc.params.suscept_delta = hmc.suscept;
				hmc.trajectory (U, D);
				std::cout << hmc.suscept << std::endl;
			}
			*/
			std::cout << hmc.suscept << std::endl;
			// reset suscept bounds to original desired values
			hmc.params.suscept_central = hmc_pars.suscept_central;
			hmc.params.suscept_delta = hmc_pars.suscept_delta;
			hmc.params.tau = hmc_pars.tau;
			hmc.params.n_steps = hmc_pars.n_steps;
		}
	}

	// thermalisation
	log("");
	log("Thermalisation:");
	log("");
	int acc = 0;
	// use smaller tau for therm
	//hmc.params.tau *= 0.25;
	for(int i=1; i<=run_pars.n_therm; ++i) {
		acc += hmc.trajectory (U, D);

		//Eigen::MatrixXcd eigenvaluesDDdag = D.DDdagger_eigenvalues (U, hmc_pars.mass, hmc_pars.mu_I);				
		//double effective_mass = sqrt(eigenvaluesDDdag.real().minCoeff());

		std::cout << "# therm " << i << "/" << run_pars.n_therm 
				  << "\tplaq: " << hmc.plaq(U) 
				  << "\t acc: " << acc/static_cast<double>(i)
				  << "\t dE: " << hmc.deltaE
				  << "\t suscept: " << hmc.suscept
 				  << std::endl;
 	}
	log("Thermalisation acceptance", static_cast<double>(acc)/static_cast<double>(run_pars.n_therm));

	// gauge config generation
	// reset tau
	//hmc.params.tau = hmc_pars.tau;
	log("");
	log("Main run:");
	log("");
	acc = 0;
	int i_save = run_pars.initial_config;
	for(int i=1; i<=run_pars.n_traj; ++i) {
		acc += hmc.trajectory (U, D);
		dE.push_back(hmc.deltaE);
		expdE.push_back(exp(-hmc.deltaE));
		plq.push_back(hmc.plaq(U));
		std::complex<double> tmp_poly = hmc.polyakov_loop(U);
		poly_re.push_back(tmp_poly.real());
		poly_im.push_back(tmp_poly.imag());
		std::cout << "# iter " << i << "/" << run_pars.n_traj 
				  << "\tplaq: " << hmc.plaq(U) 
				  << "\t acc: " << static_cast<double>(acc)/static_cast<double>(i)
				  << "\t dE: " << hmc.deltaE
				  << "\t suscept: " << hmc.suscept
				  << std::endl;
		if(i%run_pars.n_save==0) {
			// save gauge config
			++i_save;
			write_gauge_field(U, run_pars.base_name, i_save);
		}
	}

	// print average, error & integrated autocorrelation time of measured observables
	log("Acceptance", static_cast<double>(acc)/static_cast<double>(run_pars.n_traj));
	log("");
	std::cout << "# " << std::left << std::setw(20) << "observable" << "average\t\terror\t\t\ttau_int\t\terror" << std::endl;
	print_av(expdE, "<exp(-dE)>");
	print_av(dE, "<dE>");
	print_av(plq, "<plaq>");
	print_av(poly_re, "Re<polyakov>");
	print_av(poly_im, "Im<polyakov>");

	return(0);
}
