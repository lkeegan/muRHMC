#include "hmc.hpp"
#include "dirac_op.hpp"
#include "inverters.hpp"
#include "io.hpp"
#include "stats.hpp"
#include <iostream>
#include <random>
#include <math.h>

int main(int argc, char *argv[]) {

    if (argc-1 != 5) {
        std::cout << "This program requires 5 arguments:" << std::endl;
        std::cout << "mass mu_I base_name initial_config noise_vectors" << std::endl;
        std::cout << "e.g. ./measure 0.14 0.25 mu0.25_sus_3.1_3.3 23 100" << std::endl;
        return 1;
    }

	double mass = atof(argv[1]);
	double mu_I = atof(argv[2]);
	std::string base_name(argv[3]);
	int n_initial = static_cast<int>(atof(argv[4]));
	int noise_vectors = static_cast<int>(atof(argv[5]));

	hmc_params hmc_pars;
	hmc_pars.mass = mass;
	hmc_pars.mu_I = mu_I;
	hmc_pars.seed = 123;
	double eps = 1.e-10;

	// make 4^4 lattice
	lattice grid (4);
	hmc hmc (hmc_pars);

	std::cout.precision(12);

	log("Exact & noise vector observable measurements with parameters:");
	log("L", grid.L0);
	log("mass", hmc_pars.mass);
	log("mu_I", hmc_pars.mu_I);
	log("inverter precision", eps);
	log("noise vectors / cnfg", noise_vectors);
	log("seed", hmc_pars.seed);
	// make U[mu] field on lattice
	field<gauge> U (grid);
	// initialise Dirac Op
	dirac_op D (grid, hmc_pars.mass, hmc_pars.mu_I);

	// Gaussian noise observables:
	field<fermion> phi(U.grid);
	field<fermion> chi(U.grid);
	field<fermion> psi(U.grid);
	std::vector<double> psibar_psi, pion_susceptibility, isospin_density, quark_density_real, quark_density_imag;

	for(int i=n_initial; ; i+=1) {

		read_gauge_field(U, base_name, i);

		// Exact observables
		double phase, pbp, sus, mineval, maxeval, density_real, density_imag;

		// Calculate all eigenvalues lambda_i of Dirac op:
		Eigen::MatrixXcd eigenvalues = D.D_eigenvalues(U);				
		// Det[D] = \prod_i \lambda_i
		// phase{D} = \sum_i phase{lambda_i}
		phase = 0;
		for (int j=0; j<eigenvalues.size(); ++j) {
			//std::cout << std::arg(eigenvalues(i)) << std::endl;
			phase += std::arg(eigenvalues(j));
		}
		log("[evals] det-phase", phase);

		// Trace[D^-1] = \sum_i \lambda_i^-1:
		pbp = eigenvalues.cwiseInverse().sum().real()/static_cast<double>(3*U.V);
		log("[evals] psibar-psi", pbp);

		// Calculate all eigenvalues of DDdagger op:
		Eigen::MatrixXcd eigenvaluesDDdag = D.DDdagger_eigenvalues(U);
		sus = eigenvaluesDDdag.cwiseInverse().sum().real()/static_cast<double>(3*U.V);
		mineval = eigenvaluesDDdag.real().minCoeff();
		maxeval = eigenvaluesDDdag.real().maxCoeff();
		log("[evals] pion-suscept", sus);
		log("[evals] mineval-DDdag", mineval);
		log("[evals] maxeval-DDdag", maxeval);

		// NO SUPPORT for COMPLEX MATRICES in EIGEN generalised eigenvalue
/*		// Calculate eigenvalues of [D^-1 (dD/dmu)]
		// Generalised eigenvalue problem: 
		// (dD/dmu) |lambda_i> = lambda_i D |lambda_i>

		Eigen::GeneralizedEigenSolver<Eigen::MatrixXcd> ges;
		Eigen::MatrixXcd A = D.dD_dmu_dense_matrix(U, mu_I);
		Eigen::MATRICESatrixXcd B = D.D_dense_matrix(U, mass, mu_I);
		ges.compute(A, B, false);
		std::cout << "The (complex) numerators of the generalzied eigenvalues are: " << ges.alphas().transpose() << std::endl;
		std::cout << "The (real) denominatore of the generalzied eigenvalues are: " << ges.betas().transpose() << std::endl;
		std::cout << "The (complex) generalzied eigenvalues are (alphas./beta): " << ges.eigenvalues().transpose() << std::endl;
		std::complex<double> density = 2.0*ges.eigenvalues().sum()/static_cast<double>(3*U.V);
		density_real = density.real();
		density_imag = density.imag();
		log("[evals] Re quark-density", density_real);
		log("[evals] Im quark-density", density_imag);
*/
		// isospin density noisy estimate
		for(int i_noise=0; i_noise<noise_vectors; ++i_noise) {
			// phi = gaussian noise vector
			// unit norm: <phi|phi> = 1
			hmc.gaussian_fermion(phi);
			phi /= sqrt(phi.squaredNorm());

			// chi = [D(mu,m)D(mu,m)^dag]-1 phi	
			cg(chi, phi, U, D, eps);
			// pion_susceptibility = Tr[{D(mu,m)D(mu,m)^dag}^-1] = <phi|chi>
			pion_susceptibility.push_back(phi.dot(chi).real());
			
			// psi = -D(mu,m)^dag chi = D(-mu,-m) chi = -D(mu)^-1 phi
			// psibar_psi = Tr[D(mu,nu)^-1] = -<phi|psi>
			D.mass = -D.mass;
			D.mu_I = -D.mu_I;
			D.D(psi, chi, U);
			D.mass = -D.mass;
			D.mu_I = -D.mu_I;
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

			std::complex<double> sum = 0;
			for(int ix=0; ix<U.V; ++ix) {
				sum += mu_I_plus_factor * phi[ix].dot( U[ix][0] * psi.up(ix,0) );
				sum += mu_I_minus_factor * phi[ix].dot( U.dn(ix,0)[0].adjoint() * psi.dn(ix,0) );
			}
			quark_density_real.push_back(-hmc_pars.mu_I * sum.real());
			quark_density_imag.push_back(-hmc_pars.mu_I * sum.imag());

		}
		print_av(psibar_psi, "[noise] psibar-psi");
		print_av(pion_susceptibility, "[noise] pion-suscept");
		print_av(isospin_density, "[noise] iso-density");
		print_av(quark_density_real, "[noise] Re quark-density");
		print_av(quark_density_imag, "[noise] Im quark-density");

		// output all obs on single line for later analysis
		log("phase, sus, pbp, isodensity, mineval, maxeval, plaq, poly");
		std::cout << phase << "\t"
				  << sus << "\t"
				  << pbp << "\t"
				  << av(isospin_density) << "\t"
				  << mineval << "\t"
				  << maxeval << "\t"
				  << hmc.plaq(U) << "\t"
				  << hmc.polyakov_loop(U).real() << "\t"
				  << std::endl;
	}
	return(0);
}
