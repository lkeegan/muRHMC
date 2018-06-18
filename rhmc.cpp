#include "rhmc.hpp"
#include "rational_approx.hpp"
#include "dirac_op.hpp"
#include "inverters.hpp"
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

	rhmc_params rhmc_pars;
	run_params run_pars;

	read_input_file(argv[1], rhmc_pars, run_pars);

	// make TxL^3 lattice
	lattice grid (run_pars.T, run_pars.L, rhmc_pars.EE);

	log("");
	log("Unconstrained RHMC with parameters:");
	log("");
	log("beta", rhmc_pars.beta);
	log("mass", rhmc_pars.mass);
	log("mu_I", rhmc_pars.mu_I);
	log("n_f", rhmc_pars.n_f);
	log("n_pf", rhmc_pars.n_pf);
	log("tau", rhmc_pars.tau);
	log("n_steps_fermion", rhmc_pars.n_steps_fermion);
	log("n_steps_gauge", rhmc_pars.n_steps_gauge);
	log("MD epsilon", rhmc_pars.MD_eps);
	log("HB epsilon", rhmc_pars.HB_eps);
	log("RNG seed", rhmc_pars.seed);
	log("EO-preconditioning", rhmc_pars.EE);
	log("block", rhmc_pars.block);

	log("");
	log("Run parameters:");
	log("");
	log("base_name", run_pars.base_name);
	log("T", grid.L0);
	log("L1", grid.L1);
	log("L2", grid.L2);
	log("L3", grid.L3);
	log("n_traj", run_pars.n_traj);
	log("n_therm", run_pars.n_therm);
	log("n_save", run_pars.n_save);
	log("initial_config", run_pars.initial_config);

	// Initialise RHMC
	rhmc hmc (rhmc_pars, grid);

	// make U[mu] field on lattice
	field<gauge> U (grid);

	// initialise Dirac Op
	dirac_op D (grid, rhmc_pars.mass, rhmc_pars.mu_I);

	if(run_pars.initial_config < 0) {
		log("initial config negative i.e. RANDOM");
		run_pars.initial_config = 0;
		// start from random gauge field: 2nd param = roughness
		// so 0.0 gives unit gauge links, large value = random
		hmc.random_U (U, 0.5);
	} else {
		// load specified gauge config
		//	read_massimo_gauge_field(U, run_pars.base_name);
		read_gauge_field (U, run_pars.base_name, run_pars.initial_config);
	}

	// just do force measurements:
	/*
	for(int i=0; i<=run_pars.n_traj; ++i) {
		read_gauge_field (U, run_pars.base_name, run_pars.initial_config+i);
		hmc.trajectory (U, D, false, true);
	}
	exit(0);
	*/

	/*
	// also repeat force measurements for other (not)blocked case:
	rhmc_pars.block = !rhmc_pars.block;
	rhmc rhmc_block (rhmc_pars, grid);
	rhmc_block.trajectory (U, D, false, true);
	exit(0);
	*/
	
	// observables to measure
	std::vector<double> dE;
	std::vector<double> expdE;
	std::vector<double> plq;
	std::vector<double> poly_re;
	std::vector<double> poly_im;

	// thermalisation
	log("");
	log("Thermalisation:");
	log("");
	int acc = 0;
	for(int i=1; i<=run_pars.n_therm; ++i) {
		acc += hmc.trajectory (U, D);

		std::cout << "# therm " << i << "/" << run_pars.n_therm 
				  << "\tplaq: " << hmc.plaq(U) 
				  << "\t acc: " << acc/static_cast<double>(i)
				  << "\t dE: " << hmc.deltaE
 				  << std::endl;
 	}
	log("Thermalisation acceptance", static_cast<double>(acc)/static_cast<double>(run_pars.n_therm));

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
