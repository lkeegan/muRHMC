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
	/*
    if (argc-1 != 7) {
        std::cout << "This program requires 7 arguments:" << std::endl;
        std::cout << "config_name n_RHS BCGA dQ dQA rQ fermion_filename" << std::endl;
        std::cout << "e.g. ./CG conf_20_40_m0.002_b5.144 12 1 0 1 0 gaussian.fermion" << std::endl;
        return 1;
    }
	std::string config_name(argv[1]);
	int N = static_cast<int>(atof(argv[2]));
	bool BCGA = static_cast<bool>(atoi(argv[3]));
	bool dQ = static_cast<bool>(atoi(argv[4]));
	bool dQA = static_cast<bool>(atoi(argv[5]));
	bool rQ = static_cast<bool>(atoi(argv[6]));
	std::string fermion_filename(argv[7]);
	int N_shifts = 0;
	*/
    // For shifted blockCGrQ
    
    if (argc-1 != 3) {
        std::cout << "This program requires 3 arguments:" << std::endl;
        std::cout << "config_name n_RHS n_shifts" << std::endl;
        std::cout << "e.g. ./CG conf_20_40_m0.002_b5.144 12 10" << std::endl;
        return 1;
    }
	std::string config_name(argv[1]);
	int N = static_cast<int>(atof(argv[2]));
	int N_shifts = static_cast<int>(atof(argv[3]));

	bool source_point = false;
	bool source_multiplied_by_D = true;

	// shifts for RHMC:
	std::vector<double> beta(20);
	if(N==1) {
		N_shifts = 17;
		beta[1] = 5.7196274795000087e-07;
		beta[2] = 3.9257377305342023e-06;
		beta[3] = 1.3301158853547672e-05;
		beta[4] = 3.7393407519062250e-05;
		beta[5] = 9.8544678312844367e-05;
		beta[6] = 2.5346657415502191e-04;
		beta[7] = 6.4584955192068633e-04;
		beta[8] = 1.6397333005274731e-03;
		beta[9] = 4.1578735648366257e-03;
		beta[10] = 1.0542461386902711e-02;
		beta[11] = 2.6759403567453634e-02;
		beta[12] = 6.8139481912911928e-02;
		beta[13] = 1.7496169229706912e-01;
		beta[14] = 4.5903661428494158e-01;
		beta[15] = 1.2750224192997055e+00;
		beta[16] = 4.1648621103493628e+00;
		beta[17] = 2.4273206211980053e+01;
	} else if (N==2) { 
		N_shifts = 16;
		beta[1] = 7.0788344445058014e-07;
		beta[2] = 4.7361072373618120e-06;
		beta[3] = 1.6446848350721826e-05;
		beta[4] = 4.8162794903292892e-05;
		beta[5] = 1.3325534214687028e-04;
		beta[6] = 3.6126395719731302e-04;
		beta[7] = 9.7215315013625186e-04;
		beta[8] = 2.6091336074840856e-03;
		beta[9] = 6.9978710224380897e-03;
		beta[10] = 1.8779786314478436e-02;
		beta[11] = 5.0522834750990842e-02;
		beta[12] = 1.3687290931440266e-01;
		beta[13] = 3.7795481564310507e-01;
		beta[14] = 1.1006650325109624e+00;
		beta[15] = 3.7557864547788964e+00;
		beta[16] = 2.3188479121149765e+01;
	} else if (N==4) { 
		N_shifts = 15;
		beta[1] = 8.4309309440785416e-07;
		beta[2] = 5.6616811944512074e-06;
		beta[3] = 2.0389243936896897e-05;
		beta[4] = 6.2769578869437148e-05;
		beta[5] = 1.8386125820915182e-04;
		beta[6] = 5.2956500163927180e-04;
		beta[7] = 1.5165196335325012e-03;
		beta[8] = 4.3350734125012589e-03;
		beta[9] = 1.2391813861968942e-02;
		beta[10] = 3.5483099142700779e-02;
		beta[11] = 1.0217037216598845e-01;
		beta[12] = 2.9901964316032592e-01;
		beta[13] = 9.1822644800690834e-01;
		beta[14] = 3.2792679148529253e+00;
		beta[15] = 2.1167790755086941e+01;
	} else if (N==8) { 
		N_shifts = 14;
		beta[1] = 9.9179731399929969e-07;
		beta[2] = 6.7962953413258091e-06;
		beta[3] = 2.5645064987228604e-05;
		beta[4] = 8.3814974708916045e-05;
		beta[5] = 2.6240031085836350e-04;
		beta[6] = 8.1039967608426903e-04;
		beta[7] = 2.4922013741293641e-03;
		beta[8] = 7.6567333548349503e-03;
		beta[9] = 2.3545777999569466e-02;
		beta[10] = 7.2710822383871179e-02;
		beta[11] = 2.2755598720561176e-01;
		beta[12] = 7.4287428893477436e-01;
		beta[13] = 2.7921170066148329e+00;
		beta[14] = 1.8764951930188825e+01;
	} else if (N==16) { 
		N_shifts = 14;
		beta[1] = 1.0025012724456462e-06;
		beta[2] = 6.8358104948963599e-06;
		beta[3] = 2.5768448959732921e-05;
		beta[4] = 8.4194148210958169e-05;
		beta[5] = 2.6356368297148072e-04;
		beta[6] = 8.1396887853478820e-04;
		beta[7] = 2.5031547484427544e-03;
		beta[8] = 7.6903785965781648e-03;
		beta[9] = 2.3649417284756426e-02;
		beta[10] = 7.3032846270360732e-02;
		beta[11] = 2.2858363235938853e-01;
		beta[12] = 7.4643739373142659e-01;
		beta[13] = 2.8082453218019161e+00;
		beta[14] = 1.8963687232832950e+01;
	} else if (N==32) { 
		N_shifts = 14;
		beta[1] = 1.0078709065789627e-06;
		beta[2] = 6.8556216222461477e-06;
		beta[3] = 2.5830315408918393e-05;
		beta[4] = 8.4384294098241309e-05;
		beta[5] = 2.6414714485195644e-04;
		beta[6] = 8.1575909971277308e-04;
		beta[7] = 2.5086492005763185e-03;
		beta[8] = 7.7072574183771119e-03;
		beta[9] = 2.3701415591011375e-02;
		beta[10] = 7.3194435632507965e-02;
		beta[11] = 2.2909941873203618e-01;
		beta[12] = 7.4822672301165893e-01;
		beta[13] = 2.8163575145888777e+00;
		beta[14] = 1.9064310089505259e+01;
	} else {
		std::cout << "No shifts for given N_rhs, exiting." << std::endl; 
	}

	hmc_params hmc_params = {
		5.144, 	// beta
		0.00227,// mass
		0.000, 	// mu_I
		1.0, 	// tau
		7, 		// n_steps
		1.e-6,	// MD_eps
		1234,	// seed
		false, 	// EE
		false,	// constrained HMC
		3.0, 	// suscept_central
		0.05	// suscept_eps
	};
	hmc hmc (hmc_params);
	std::cout.precision(12);

	lattice grid (12);
	field<gauge> U (grid);
//	read_massimo_gauge_field(U, config_name);
//	write_gauge_field(U, config_name, 1);
	read_gauge_field(U, config_name, 1);
	log("Average plaquette", hmc.plaq(U));
	log("Spatial plaquette", hmc.plaq_spatial(U));
	log("Timelike plaquette", hmc.plaq_timelike(U));
	log("1x2 plaquette", hmc.plaq_1x2(U));
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
	dirac_op D (grid, hmc_params.mass, hmc_params.mu_I);
	double eps = 1.e-14;

	log("CG test program");
	log("L0", grid.L0);
	log("L1", grid.L1);
	log("L2", grid.L2);
	log("L3", grid.L3);
	log("mass", hmc_params.mass);
	log("mu_I", hmc_params.mu_I);
	log("eps", eps);
	log("N_block", N);
	log("N_shifts", N_shifts);

	// make vector of N fermion fields for random chi and empty Q, x
	std::vector<field<fermion>> x, chi;
	field<fermion> tmp_chi (grid), tmp_phi(grid);
	for(int i=0; i<N; ++i) {
		x.push_back(field<fermion>(grid));
		if(source_point) {
			// point source
			log("point source");
			tmp_chi.setZero();
			tmp_chi[i][0] = 1.0;
		} else {
			// gaussian noise source
			log("gaussian noise source");
			hmc.gaussian_fermion (tmp_chi);
		}
		if(source_multiplied_by_D) {
			// multiply source by D, as in HMC
			log("source <- D * source");
			D.D(tmp_phi, tmp_chi, U);	
			chi.push_back(tmp_phi);
		} else {
			chi.push_back(tmp_chi);
		}
	}

	// Single Inversion Block CG:
/*
	// We want to have the solution for the first vector to calculate the error norm
	field<fermion> x0_star(grid);
	// Try to load stored x0_star fermion field from file: 
	if (read_fermion_field(x0_star, fermion_filename)) {
		// check it is actually the solution!
		D.DDdagger(tmp_phi, x0_star, U, hmc_params.mass, hmc_params.mu_I);
		double x0_star_res = is_field_equal(tmp_phi, chi[0])/sqrt(chi[0].squaredNorm());	
		log("Residual of x0_star:", x0_star_res);
	} else {
		// If it fails, then DO CG inversion of first source: x0_star = (DD')^-1 chi[0]
		log("Inverting first source using CG...");
		D.cg(x0_star, chi[0], U, hmc_params.mass, hmc_params.mu_I, eps);
		// and write to file for next time
		write_fermion_field(x0_star, fermion_filename);
	}

    auto timer_start = std::chrono::high_resolution_clock::now();
	// x_i = (DD')^-1 chi_i
	int iterBLOCK = D.cg_block(x, chi, U, hmc_params.mass, hmc_params.mu_I, eps, BCGA, dQ, dQA, rQ, x0_star);
    auto timer_stop = std::chrono::high_resolution_clock::now();
    auto timer_count = std::chrono::duration_cast<std::chrono::seconds>(timer_stop-timer_start).count();
	log("BlockCG_runtime_sec", timer_count);
*/
	//std::vector<double> possible_shifts = 
		//{0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7123, 0.812, 1.1, 1.8, 2.1, 2.4, 8.0, 74.0, 177.3, 212, 1853.0};
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

	return(0);
}
