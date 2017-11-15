#include "hmc.hpp"
#include "dirac_op.hpp"
#include "io.hpp"
#include "stats.hpp"
#include <iostream>
#include <iomanip>
#include <random>

int main(int argc, char *argv[]) {
    if (argc-1 != 8) {
        std::cout << "This program requires 8 arguments:" << std::endl;
        std::cout << "beta mass mu_I n_integrator_steps n_therm n_traj n_save seed, e.g." << std::endl;
        std::cout << "./rhmc 4.4 0.14 0.25 17 500 1000 100 12345" << std::endl;
        return 1;
    }

	std::cout.precision(14);

	// HMC parameters
	hmc_params hmc_pars;
	hmc_pars.beta = atof(argv[1]);
	hmc_pars.mass = atof(argv[2]);
	hmc_pars.mu_I = atof(argv[3]);
	hmc_pars.tau = 1.0;	
	hmc_pars.n_steps = static_cast<int>(atof(argv[4]));
	hmc_pars.MD_eps = 1.e-7;
	hmc_pars.seed = static_cast<int>(atof(argv[8]));

	// run parameters
	int n_therm = static_cast<int>(atof(argv[5]));
	int n_traj = static_cast<int>(atof(argv[6]));
	int n_save = static_cast<int>(atof(argv[7]));
	std::string str_mu(argv[3]);
	std::string base_name = "mu" + str_mu;	

	// make 4^4 lattice
	lattice grid (4);

	log("RHMC run with parameters:");
	log("L", grid.L0);
	log("beta", hmc_pars.beta);
	log("mass", hmc_pars.mass);
	log("mu_I", hmc_pars.mu_I);
	log("tau", hmc_pars.tau);
	log("n_steps", hmc_pars.n_steps);
	log("MD epsilon", hmc_pars.MD_eps);
	log("RNG seed", hmc_pars.seed);
	log("n_traj", n_traj);
	log("n_therm", n_therm);
	log("n_save", n_save);

	// make U[mu] field on lattice
	field<gauge> U (grid);
	// initialise Dirac Op
	dirac_op D (grid);

	// Initialise HMC
	hmc hmc (hmc_pars);

	// start from random gauge field: 2nd param = roughness
	// so 0.0 gives unit gauge links, large value = random
	hmc.random_U (U, 0.5);

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
	for(int i=1; i<=n_therm; ++i) {
		acc += hmc.trajectory (U, D);

		//Eigen::MatrixXcd eigenvaluesDDdag = D.DDdagger_eigenvalues (U, hmc_pars.mass, hmc_pars.mu_I);				
		//double effective_mass = sqrt(eigenvaluesDDdag.real().minCoeff());

		std::cout << "# therm " << i << "/" << n_therm 
				  << "\tplaq: " << hmc.plaq(U) 
				  << "\t acc: " << acc/static_cast<double>(i)
				  << "\t dE: " << hmc.deltaE
				  //<< "\t m_eff: " << effective_mass
 				  << std::endl;
 	}
	log("Thermalisation acceptance", static_cast<double>(acc)/static_cast<double>(n_therm));

	// gauge config generation
	log("");
	log("Main run:");
	log("");
	acc = 0;
	for(int i=1; i<=n_traj; ++i) {
		acc += hmc.trajectory (U, D);
		dE.push_back(hmc.deltaE);
		expdE.push_back(exp(-hmc.deltaE));
		plq.push_back(hmc.plaq(U));
		std::complex<double> tmp_poly = hmc.polyakov_loop(U);
		poly_re.push_back(tmp_poly.real());
		poly_im.push_back(tmp_poly.imag());
		std::cout << "# iter " << i << "/" << n_traj 
				  << "\tplaq: " << hmc.plaq(U) 
				  << "\t acc: " << static_cast<double>(acc)/static_cast<double>(i)
				  << "\t dE: " << hmc.deltaE
				  << std::endl;
		if(i%n_save==0) {
			// save gauge config
			write_gauge_field(U, base_name, i/n_save);
		}
	}

	// print average, error & integrated autocorrelation time of measured observables
	log("Acceptance", static_cast<double>(acc)/static_cast<double>(n_traj));
	log("");
	std::cout << "# " << std::left << std::setw(20) << "observable" << "average\t\terror\t\t\ttau_int\t\terror" << std::endl;
	print_av(expdE, "<exp(-dE)>");
	print_av(dE, "<dE>");
	print_av(plq, "<plaq>");
	print_av(poly_re, "Re<polyakov>");
	print_av(poly_im, "Im<polyakov>");

	return(0);
}
