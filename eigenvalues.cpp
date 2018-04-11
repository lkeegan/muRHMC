#include "hmc.hpp"
#include "dirac_op.hpp"
#include "inverters.hpp"
#include "io.hpp"
#include "stats.hpp"
#include <iostream>
#include <random>
#include <Eigen/Eigenvalues>

int main(int argc, char *argv[]) {

	std::cout.precision(17);

    if (argc-1 != 6) {
        std::cout << "This program requires 5 arguments:" << std::endl;
        std::cout << "mass mu_I base_name config_number relative_error" << std::endl;
        std::cout << "e.g. ./hmc 0.14 0.25 mu0.25_sus_3.1_3.3 1 1e-4" << std::endl;
        return 1;
    }

	double mass = atof(argv[1]);
	double mu_I = atof(argv[2]);
	std::string base_name(argv[3]);
	int config_number = static_cast<int>(atof(argv[4]));
	int N_eigenvalues = N_rhs; //static_cast<int>(atof(argv[5]));
	double eps = atof(argv[6]);

	lattice grid (8, 24, true);
	//field<fermion>::eo_storage_options eo_storage = field<fermion>::FULL;
	field<fermion>::eo_storage_options eo_storage = field<fermion>::EVEN_ONLY;

	log("Eigenvalues measurement run with parameters:");
	log("T", grid.L0);
	log("L", grid.L1);
	log("mass", mass);
	log("mu_I", mu_I);
	log("N_eigenvalues", N_eigenvalues);
	log("relative_error", eps);

	field<gauge> U (grid);
	read_gauge_field(U, base_name, config_number);
	dirac_op D (grid, mass, mu_I);
	rhmc_params rhmc_pars;
	rhmc_pars.seed = 123;
	rhmc rhmc (rhmc_pars, grid);

	// Power method gives strict lower bound on lambda_max
	// Iterate until relative error < eps

	field<fermion> x (grid, eo_storage), x2 (grid, eo_storage);
	rhmc.gaussian_fermion(x);
	double x_norm = x.norm();
	double lambda_max = 1;
	double lambda_max_err = 100;
	int iter = 0;
	while((lambda_max_err/lambda_max) > eps) {
		for(int i=0; i<8; ++i) {
			x /= x_norm;
			D.DDdagger(x2, x, U);
			x2 /= x2.norm();
			D.DDdagger(x, x2, U);				
			x_norm = x.norm();
			iter += 2;
		}
		lambda_max = x2.dot(x).real();
		lambda_max_err = sqrt(x_norm * x_norm - lambda_max * lambda_max);
		log("lambda_max", lambda_max, lambda_max_err);
	}
	// since lambda_max is a lower bound, and lambda_max + lamda_max_err is an upper bound:
	log("iterations", iter);
	log("final_lambda_max", lambda_max + 0.5 * lambda_max_err, 0.5 * lambda_max_err);

	// Find N lowest eigenvalues of DDdagger.
	// Uses chebyshev acceleration as described in Appendix A of hep-lat/0512021
	block_matrix R = block_matrix::Zero();	
	Eigen::MatrixXd Evals = Eigen::MatrixXd::Zero(N_eigenvalues, 2);	

	// make initial fermion vector basis of gaussian noise vectors
	field<block_fermion> X (grid, field<block_fermion>::EVEN_ONLY);
	rhmc.gaussian_fermion (X);
	// orthonormalise X
	thinQR(X, R);
	// make X A-orthormal and get eigenvalues of matrix <X_i AX_j>
	thinQRA_evals(X, Evals, U, D);

	// v is the upper bound on possible eigenvalues
	// use 50% margin of safety on estimate of error to get safe upper bound
	double v = lambda_max + 1.5 * lambda_max_err;
	while((Evals.col(1).array()/Evals.col(0).array()).maxCoeff() > eps) {
		// get current estimates of min/max eigenvalues
		double lambda_min = Evals.col(0)[0]; 
		double lambda_N = Evals.col(0)[N_eigenvalues-1];
		// find optimal chebyshev order k 
        int k = 2 * std::ceil(0.25*sqrt(((v-lambda_min)*8.33633354 - (v-lambda_N)*3.1072776)/(lambda_N - lambda_min)));
		// find chebyshev lower bound
		double u = lambda_N + (v - lambda_N)*(tanh(1.76275/(2.0*k)))*tanh(1.76275/(2.0*k));
		log("Chebyshev order k:", k);
		log("Chebyshev range u:", u);
		log("Chebyshev range v:", v);
		// apply chebyshev polynomial in DDdagger
		D.chebyshev(k, u, v, X, U);		
		// orthonormalise X
		thinQR(X, R);
		// make X A-orthormal and get eigenvalues of matrix <X_i AX_j>
		thinQRA_evals(X, Evals, U, D);
		// note the estimated errors for the eigenvalues are only correct
		// if the eigevalues are separated more than the errors
		// i.e. assumes residuals matrix is diagonal
		// if this is not the case then errors are underestimated
		std::cout << Evals << std::endl;
	}

	return(0);
}
