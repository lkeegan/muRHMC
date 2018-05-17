#include "hmc.hpp"
#include "dirac_op.hpp"
#include "inverters.hpp"
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

	// For block CG(A)(dQ/dQA)(rQ):
	
    if (argc-1 != 7) {
        std::cout << "This program requires 7 arguments:" << std::endl;
        std::cout << "config_name n_RHS BCGA dQ dQA rQ fermion_filename" << std::endl;
        std::cout << "e.g. ./CG conf_20_40_m0.002_b5.144 12 1 0 1 0 gaussian.fermion" << std::endl;
        return 1;
    }
	std::string config_name(argv[1]);
	int N = N_rhs; //static_cast<int>(atof(argv[2]));
	bool BCGA = static_cast<bool>(atoi(argv[3]));
	bool dQ = static_cast<bool>(atoi(argv[4]));
	bool dQA = static_cast<bool>(atoi(argv[5]));
	bool rQ = static_cast<bool>(atoi(argv[6]));
	std::string fermion_filename(argv[7]);
	int N_shifts = 0;
	
    // For shifted blockCGrQ
    /*
    if (argc-1 != 3) {
        std::cout << "This program requires 3 arguments:" << std::endl;
        std::cout << "config_name n_RHS n_shifts" << std::endl;
        std::cout << "e.g. ./CG conf_20_40_m0.002_b5.144 12 10" << std::endl;
        return 1;
    }
	std::string config_name(argv[1]);
	int N = static_cast<int>(atof(argv[2]));
	int N_shifts = static_cast<int>(atof(argv[3]));
    */

	bool source_point = false;
	bool source_multiplied_by_D = false;

	rhmc_params rhmc_params;
	rhmc_params.mass = 0.00227;
	rhmc_params.mu_I = 0.0;
	rhmc_params.seed = 123;
	std::cout.precision(12);

	lattice grid (12, 12, true);
	rhmc rhmc (rhmc_params, grid);
	field<gauge> U (grid);
//	read_massimo_gauge_field(U, config_name);
//	write_gauge_field(U, config_name, 1);
	read_gauge_field(U, config_name, 1);
	log("Average plaquette", rhmc.plaq(U));
	//log("Spatial plaquette", rhmc.plaq_spatial(U));
	//log("Timelike plaquette", rhmc.plaq_timelike(U));
	//log("1x2 plaquette", rhmc.plaq_1x2(U));
/*
	for(int i=0; i<2; ++i){
		double rho = 0.15;
		log("Stout smearing with rho:", rho);
		hmc.stout_smear(rho, U);
		log("Average plaquette", hmc.plaq(U));
		log("Spatial plaquette", hmc.plaq_spatial(U));
		log("Timelike plaquette", hmc.plaq_timelike(U));
		log("1x2 plaquette", hmc.plaq_1x2(U));
	}
*/
	// initialise Dirac Op
	dirac_op D (grid, rhmc_params.mass, rhmc_params.mu_I);
	double eps = 1.e-14;

	log("CG test program");
	log("L0", grid.L0);
	log("L1", grid.L1);
	log("L2", grid.L2);
	log("L3", grid.L3);
	log("mass", rhmc_params.mass);
	log("mu_I", rhmc_params.mu_I);
	log("eps", eps);
	log("N_block", N);
//	log("N_shifts", N_shifts);

	// make vector of N fermion fields for random chi and empty Q, x
	field<block_fermion> chi (grid, field<block_fermion>::EVEN_ONLY);
	field<block_fermion> x (chi);
	rhmc.gaussian_fermion (chi);
	// Single Inversion Block CG:

	// We want to have the solution for the first vector to calculate the error norm
	field<fermion> x0_star(grid, field<fermion>::EVEN_ONLY);
	field<fermion> chi0 (x0_star), tmp_phi(x0_star);
	for(int ix=0; ix<chi0.V; ++ix) {
		chi0[ix] = chi[ix].col(0);
	}
	// Try to load stored x0_star fermion field from file: 
	if (read_fermion_field(x0_star, fermion_filename)) {
		// check it is actually the solution!
		D.DDdagger(tmp_phi, x0_star, U);
		double x0_star_res = is_field_equal(tmp_phi, chi0)/chi0.norm();	
		log("Residual of x0_star:", x0_star_res);
	} else {
		// If it fails, then DO CG inversion of first source: x0_star = (DD')^-1 chi[0]
		log("Inverting first source using CG...");
		cg(x0_star, chi0, U, D, eps);
		// and write to file for next time
		write_fermion_field(x0_star, fermion_filename);
	}

    auto timer_start = std::chrono::high_resolution_clock::now();
	// x_i = (DD')^-1 chi_i
	int iterBLOCK = cg_block(x, chi, U, D, eps, BCGA, dQ, dQA, rQ, true, &x0_star);
    auto timer_stop = std::chrono::high_resolution_clock::now();
    auto timer_count = std::chrono::duration_cast<std::chrono::seconds>(timer_stop-timer_start).count();
	log("BlockCG_runtime_sec", timer_count);


// shifted
/*
	std::vector<double> shifts (N_shifts);
	std::vector< std::vector< field<fermion> > > x_s;
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
		//shifts[i_shift] = possible_shifts[i_shift] * possible_shifts[i_shift];
 		shifts[i_shift] = beta[i_shift+1];
 		x_s.push_back(x);
	}
    auto timer_start = std::chrono::high_resolution_clock::now();
	// x_i = (DD')^-1 chi_i
	int iterBLOCK = SBCGrQ(x_s, chi, U, shifts, D, eps);
    auto timer_stop = std::chrono::high_resolution_clock::now();
    auto timer_count = std::chrono::duration_cast<std::chrono::seconds>(timer_stop-timer_start).count();
	log("SBCGrQ_runtime_sec", timer_count);

	// Calculate and output true residuals for first solution vector for each shift
	double norm = chi[0].norm();
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
		D.DDdagger(tmp_phi, x_s[i_shift][0], U);
		tmp_phi.add(shifts[i_shift], x_s[i_shift][0]);
		tmp_phi.add(-1.0, chi[0]);
		log("true-res-shift", shifts[i_shift], tmp_phi.norm()/norm);
	}
*/
	return(0);
}
