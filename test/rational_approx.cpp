#include "catch.hpp"
#include "su3.hpp"
#include "4d.hpp"
#include "hmc.hpp"
#include "dirac_op.hpp"
#include "rational_approx.hpp"
#include "inverters.hpp"
#include "io.hpp"

// if true, tests should run quickly, for use with every compile
// if false, does extended tests that may be quite slow
constexpr bool QUICK_TESTS_ONLY = true;
// tolerance for numbers that should be zero:
constexpr double EPS = 5.e-13;

TEST_CASE( "Rational Approximations", "[rational_approx]") {

	std::vector<int> n_values;
	if(QUICK_TESTS_ONLY) {
		n_values = {2, 3, 4};
	} else {
		n_values = {2, 3, 4, 5, 6, 12, 16, 24, 32, 48, 64};
	}

	double eps = 1.e-15;

	rhmc_params rhmc_pars;
	rhmc_pars.mass = 0.292;
	rhmc_pars.mu_I = 0.0557;
	rhmc_pars.seed = 123;

	// Loop over 3 setups: LEXI (default), EO, EO w/EO preconditioning
	field<fermion>::eo_storage_options eo_storage_e = field<fermion>::FULL;
	field<block_fermion>::eo_storage_options block_eo_storage_e = field<block_fermion>::FULL;
	std::string lt ("LEXI_FULL");
	bool isEO = false;
	for (int lattice_type : {0, 1, 2}) {
		if(lattice_type==1) {
			isEO = true;
			lt = "EO_FULL";
		} else if(lattice_type==2) {
			isEO = true;
			eo_storage_e = field<fermion>::EVEN_ONLY;				
			block_eo_storage_e = field<block_fermion>::EVEN_ONLY;				
			lt = "EO_EVEN_ONLY";
		}

		int n_rhs = 3;
		lattice grid (4, isEO);
		field<gauge> U (grid);
		rhmc rhmc (rhmc_pars, grid);
		rhmc.random_U(U, 0.4);
		dirac_op D (grid, rhmc_pars.mass, rhmc_pars.mu_I);

		rational_approx RA(rhmc_pars.mass*rhmc_pars.mass, D.largest_eigenvalue_bound(U, eo_storage_e));

		field<fermion> x (grid, eo_storage_e);
		field<fermion> y (grid, eo_storage_e);

		std::vector< field<fermion> > X, Y, B;
		for(int i=0; i<n_rhs; ++i) {
			rhmc.gaussian_fermion(x);
			X.push_back(y);
			Y.push_back(y);
			B.push_back(x);
		}

		field<block_fermion> block_B(grid, block_eo_storage_e);
		rhmc.gaussian_fermion(block_B);
		field<block_fermion> block_X(block_B);
		field<block_fermion> block_Y(block_B);

		int iter = 0;
		for(int n : n_values) {
			SECTION( std::string("[A^{1/(n)}]^{n}_") + lt + " n=" + std::to_string(n) ) {
				y = B[0];
				// y = [A^{1/n}]^{n} B = A B:
				for(int i=0; i<n; ++i) {
					// x = A^{1/n} B:
					iter += rational_approx_cg_multishift(x, y, U, RA.alpha_hi[n], RA.beta_hi[n], D, eps);
					y = x;
				}
				// x = A^-1 y = B
				int iter = cg(x, y, U, D, eps);
				x -= B[0];
				double res = x.norm()/B[0].norm();
				INFO("Lattice-type: " << lattice_type << "\t|| b - A^-1 [A^{1/n}]^n b ||_{n=" << n << "} = " << res);
				REQUIRE( res < EPS );				
			}

			SECTION( std::string("[A^{-1/(n)}]^{n}_") + lt + " n=" + std::to_string(n) ) {
				y = B[0];
				// y = [A^{-1/n}]^{n} B = A^{-1} B:
				for(int i=0; i<n; ++i) {
					// x = A^{-1/2n} B:
					iter += rational_approx_cg_multishift(x, y, U, RA.alpha_inv_hi[n], RA.beta_inv_hi[n], D, eps);
					y = x;
				}
				// x = A y = B
				D.DDdagger(x, y, U);
				x -= B[0];
				double res = x.norm()/B[0].norm();
				INFO("Lattice-type: " << lattice_type << "\t|| b - A [A^{-1/n}]^n b ||_{n=" << n << "} = " << res);
				REQUIRE( res < EPS );				
			}

			SECTION( std::string("A^{+1/(n)}A^{-1/(n)}_") + lt + " n=" + std::to_string(n) ) {
				// y = [A^{+1/n}] [A^{-1/n}] B = B:
				iter += rational_approx_cg_multishift(x, B[0], U, RA.alpha_inv_hi[n], RA.beta_inv_hi[n], D, eps);
				iter += rational_approx_cg_multishift(y, x, U, RA.alpha_hi[n], RA.beta_hi[n], D, eps);

				y -= B[0];
				double res = y.norm()/B[0].norm();
				INFO("Lattice-type: " << lattice_type << "\t|| b - [A^{+1/n} A^{-1/n}] b ||_{n=" << n << "} = " << res);
				REQUIRE( res < EPS );				
			}

			SECTION( std::string("A^{-1/(n)}A^{+1/(n)}_") + lt + " n=" + std::to_string(n) ) {
				// y = [A^{-1/n}] [A^{+1/n}] B = B:
				iter += rational_approx_cg_multishift(x, B[0], U, RA.alpha_hi[n], RA.beta_hi[n], D, eps);
				iter += rational_approx_cg_multishift(y, x, U, RA.alpha_inv_hi[n], RA.beta_inv_hi[n], D, eps);

				y -= B[0];
				double res = y.norm()/B[0].norm();
				INFO("Lattice-type: " << lattice_type << "\t|| b - [A^{-1/n} A^{+1/n}] b ||_{n=" << n << "} = " << res);
				REQUIRE( res < EPS );				
			}

			SECTION( std::string("Block A^{-1/(n)}A^{+1/(n)}_") + lt + " n=" + std::to_string(n) ) {
				// Y = [A^{-1/n}] [A^{+1/n}] B = B:
				iter += rational_approx_SBCGrQ(block_X, block_B, U, RA.alpha_hi[n], RA.beta_hi[n], D, eps);
				iter += rational_approx_SBCGrQ(block_Y, block_X, U, RA.alpha_inv_hi[n], RA.beta_inv_hi[n], D, eps);

				block_Y -= block_B;
				double res = block_Y.norm()/block_B.norm();
				INFO("Lattice-type: " << lattice_type << "\t|| B - [A^{-1/n} A^{+1/n}] B ||_{n=" << n << "} = " << res);
				REQUIRE( res < EPS );
			}

			SECTION( std::string("Block A^{+1/(n)}A^{-1/(n)}_") + lt + " n=" + std::to_string(n) ) {
				// Y = [A^{+1/n}] [A^{-1/n}] B = B:
				iter += rational_approx_SBCGrQ(block_X, block_B, U, RA.alpha_inv_hi[n], RA.beta_inv_hi[n], D, eps);
				iter += rational_approx_SBCGrQ(block_Y, block_X, U, RA.alpha_hi[n], RA.beta_hi[n], D, eps);

				block_Y -= block_B;
				double res = block_Y.norm()/block_B.norm();
				INFO("Lattice-type: " << lattice_type << "\t|| B - [A^{+1/n} A^{-1/n}] B ||_{n=" << n << "} = " << res);
				REQUIRE( res < EPS );
			}

		}
	}
}