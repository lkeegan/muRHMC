#include "hmc.hpp"
#include "dirac_op.hpp"
#include "io.hpp"
#include "stats.hpp"
#include <iostream>
#include <random>
#include <Eigen/Eigenvalues>

int main(int argc, char *argv[]) {

    if (argc != 6) {
        std::cout << "This program requires 5 arguments:" << std::endl;
        std::cout << "beta mass mu_I n_measurements seed" << std::endl;
        std::cout << "e.g. ./hmc 4.4 0.14 0.25 1e3 12345" << std::endl;
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
	hmc_pars.seed = atoi(argv[5]);

	int n_meas = static_cast<int>(atof(argv[4])); //10000;
	double eps = 1.e-10;

	// make 4^4 lattice
	lattice grid (4);

	std::cout.precision(17);

	std::cout << "# Measurement run with parameters:" << std::endl;
	std::cout << "# L\t" << grid.L0 << std::endl;
	std::cout << "# beta\t" << hmc_pars.beta << std::endl;
	std::cout << "# mass\t" << hmc_pars.mass << std::endl;
	std::cout << "# mu_I\t" << hmc_pars.mu_I << std::endl;
	std::cout << "# inverter precision\t" << eps << std::endl;
	std::cout << "# number of measurements\t" << n_meas << std::endl;
	std::cout << "# seed\t" << hmc_pars.seed << std::endl;

	// make U[mu] field on lattice
	field<gauge> U (grid);
	// initialise Dirac Op
	dirac_op D (grid);

	field<fermion> phi(U.grid);
	field<fermion> chi(U.grid);
	field<fermion> psi(U.grid);

	// Initialise HMC
	hmc hmc (hmc_pars);

	// read philippe gauge config
	read_fortran_gauge_field(U, "fort.1");
	std::cout << "# FORTRAN GAUGE CONFIG PLAQ: " << hmc.plaq(U) << std::endl;
	std::cout << "# FORTRAN GAUGE CONFIG POLY: " << hmc.polyakov_loop(U) << std::endl;

	std::vector<double> psibar_psi, pion_susceptibility, isospin_density;
	
	for(int i_meas=0; i_meas<n_meas; ++i_meas) {
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
		double mu_I_plus_factor = exp(hmc_pars.mu_I);
		double mu_I_minus_factor = exp(-hmc_pars.mu_I);
		double sum_iso = 0;
		for(int ix=0; ix<U.V; ++ix) {
			sum_iso += mu_I_plus_factor * chi[ix].dot( U[ix][0] * psi.up(ix,0) ).real();
			sum_iso += mu_I_minus_factor * chi[ix].dot( U.dn(ix,0)[0].adjoint() * psi.dn(ix,0) ).real();
		}
		isospin_density.push_back(hmc_pars.mu_I * sum_iso);
	}

	std::cout << "# mu_I\t<psibar_psi>\t\terror\t\t\t<pion_susceptibility>\terror\t\t\t<isospin_density>\terror" << std::endl;
	std::cout 	<< hmc_pars.mu_I << "\t" 
				<< av(psibar_psi) << "\t" << std_err(psibar_psi) << "\t"
				<< av(pion_susceptibility) << "\t" << std_err(pion_susceptibility) << "\t"
				<< av(isospin_density) << "\t" << std_err(isospin_density) << std::endl;

	// Calculate all eigenvalues of Dirac op:
	Eigen::MatrixXcd eigenvalues = D.D_eigenvalues (U, hmc_pars.mass, hmc_pars.mu_I);				
	// phase of determinant:
	// Det[D] = \prod_i \lambda_i
	double phase_det = std::arg(eigenvalues.prod());
	std::cout << "phase of determinant: " << phase_det << std::endl;

	// Trace[D^-1] = \sum_i \lambda_i^-1:
	std::cout << "pbp exact: " << eigenvalues.cwiseInverse().sum()/static_cast<double>(3*U.V) << std::endl;

	// Calculate all eigenvalues of DDdagger op:
	Eigen::MatrixXcd eigenvaluesDDdag = D.DDdagger_eigenvalues (U, hmc_pars.mass, hmc_pars.mu_I);				

	// Trace[D^-1] = \sum_i \lambda_i^-1:
	std::cout << "psuscept_hermitian exact: " << ((eigenvalues.cwiseInverse()).adjoint() * (eigenvalues.cwiseInverse()))/static_cast<double>(3*U.V) << std::endl;
	std::cout << "psuscept exact: " << eigenvaluesDDdag.cwiseInverse().sum()/static_cast<double>(3*U.V) << std::endl;
	return(0);
}
