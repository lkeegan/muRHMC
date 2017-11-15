#include "hmc.hpp"
#include "dirac_op.hpp"
#include "io.hpp"
#include "stats.hpp"
#include <iostream>
#include <random>
#include <Eigen/Eigenvalues>

int main(int argc, char *argv[]) {

    if (argc-1 != 4) {
        std::cout << "This program requires 4 arguments:" << std::endl;
        std::cout << "mass mu_I initial_config noise_vectors" << std::endl;
        std::cout << "e.g. ./hmc 0.14 0.25 23 100" << std::endl;
        return 1;
    }

	hmc_params hmc_pars;
	hmc_pars.mass = atof(argv[1]);
	hmc_pars.mu_I = atof(argv[2]);
	hmc_pars.seed = 123;
	int n_initial = static_cast<int>(atof(argv[3]));
	int noise_vectors = static_cast<int>(atof(argv[4]));

	std::string str_mu(argv[2]);
	std::string base_name = "mu" + str_mu;	

	// make 4^4 lattice
	lattice grid (4);

	std::cout.precision(12);
	hmc hmc (hmc_pars);

	double eps = 1.e-10;
	log("Gaussian noise vector measurement run");
	log("");
	log("L", grid.L0);
	log("mass", hmc_pars.mass);
	log("mu_I", hmc_pars.mu_I);
	log("inverter precision", eps);
	log("noise vectors / cnfg", noise_vectors);
	log("seed", hmc_pars.seed);
	log("");
	// make U[mu] field on lattice
	field<gauge> U (grid);
	// initialise Dirac Op
	dirac_op D (grid);

	// read philippe gauge config
	/*
	read_fortran_gauge_field(U, "fort.1");
	std::cout << "# FORTRAN GAUGE CONFIG PLAQ: " << hmc.plaq(U) << std::endl;
	std::cout << "# FORTRAN GAUGE CONFIG POLY: " << hmc.polyakov_loop(U) << std::endl;
	*/

	// Gaussian noise observables:
	field<fermion> phi(U.grid);
	field<fermion> chi(U.grid);
	field<fermion> psi(U.grid);

	std::vector<double> psibar_psi, pion_susceptibility, isospin_density;
	
	for(int i=n_initial; ; ++i) {
		read_gauge_field(U, base_name, i);
		for(int i_noise=0; i_noise<noise_vectors; ++i_noise) {
			// phi = gaussian noise vector
			// unit norm: <phi|phi> = 1
			hmc.gaussian_fermion(phi);
			phi /= sqrt(phi.squaredNorm());

			// chi = [D(mu,m)D(mu,m)^dag]-1 phi	
			D.cg(chi, phi, U, hmc_pars.mass, hmc_pars.mu_I, eps);
			// pion_susceptibility = Tr[{D(mu,m)D(mu,m)^dag}^-1] = <phi|chi>
			pion_susceptibility.push_back(phi.dot(chi).real());
			
			// psi = -D(mu,m)^dag chi = D(-mu,-m) chi = -D(mu)^-1 phi
			// psibar_psi = Tr[D(mu,nu)^-1] = -<phi|psi>
			D.D(psi, chi, U, -hmc_pars.mass, -hmc_pars.mu_I);
			psibar_psi.push_back(-phi.dot(psi).real());

			// isospin_density = (d/dmu)Tr[{D(mu,m)D(mu,m)^dag}^-1]
			// 2 Re{ <chi|(dD/dmu)|psi> }
			double mu_I_plus_factor = exp(0.5 * hmc_pars.mu_I);
			double mu_I_minus_factor = exp(-0.5 * hmc_pars.mu_I);
			double sum_iso = 0;
			for(int ix=0; ix<U.V; ++ix) {
				sum_iso += mu_I_plus_factor * chi[ix].dot( U[ix][0] * psi.up(ix,0) ).real();
				sum_iso += mu_I_minus_factor * chi[ix].dot( U.dn(ix,0)[0].adjoint() * psi.dn(ix,0) ).real();
			}
			isospin_density.push_back(0.5 * hmc_pars.mu_I * sum_iso);
		}
		print_av(psibar_psi, "[noise] psibar-psi");
		print_av(pion_susceptibility, "[noise] pion-suscept");
		print_av(isospin_density, "[noise] iso-density");
	}

	return(0);
}
