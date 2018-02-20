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
		n_values = {1, 2, 3};
	} else {
		n_values = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};
	}

	double eps = 1.e-15;

	hmc_params hmc_pars = {
		5.4, 	// beta
		0.2, 	// mass
		0.0557, // mu_I
		1.0, 	// tau
		3, 		// n_steps
		1.e-6,	// MD_eps
		1234,	// seed
		false, 	// EE: only simulate even-even sub-block (requires mu_I=0)
		false,	// constrained HMC (fixed allowed range for pion susceptibility)
		3.0, 	// suscept_central
		0.05	// suscept_eps
	};

	rational_approx RA(hmc_pars.mass*hmc_pars.mass, 16.0);

	// Loop over 3 setups: LEXI (default), EO, EO w/EO preconditioning
	field<fermion>::eo_storage_options eo_storage_e = field<fermion>::FULL;
	std::string lt ("LEXI_FULL");
	bool isEO = false;
	for (int lattice_type : {0, 1, 2}) {
		if(lattice_type==1) {
			isEO = true;
			lt = "EO_FULL";
		} else if(lattice_type==2) {
			isEO = true;
			eo_storage_e = field<fermion>::EVEN_ONLY;				
			lt = "EO_EVEN_ONLY";
		}

		lattice grid (4, isEO);
		field<gauge> U (grid);
		hmc rhmc (hmc_pars);
		rhmc.random_U(U, 10.0);
		dirac_op D (grid, hmc_pars.mass, hmc_pars.mu_I);

		field<fermion> x (grid, eo_storage_e);
		field<fermion> y (grid, eo_storage_e);

		field<fermion> B (grid, eo_storage_e);
		rhmc.gaussian_fermion(B);

		int iter = 0;
		for(int n : n_values) {
			SECTION( std::string("[A^{1/(2n)}]^{2n}_") + lt + " n=" + std::to_string(n) ) {
				y = B;
				// y = [A^{1/2n}]^{2n} B = A B:
				for(int i=0; i<n; ++i) {
					// x = A^{1/2n} B:
					iter += rational_approx_cg_multishift(x, y, U, RA.alpha_hi[n], RA.beta_hi[n], D, eps);
					// y = A^{1/2n} x = A^{1/n} B:
					iter += rational_approx_cg_multishift(y, x, U, RA.alpha_hi[n], RA.beta_hi[n], D, eps);
				}
				// x = A^-1 y = B
				int iter = cg(x, y, U, D, eps);
				x -= B;
				double res = x.norm()/B.norm();
				INFO("Lattice-type: " << lattice_type << "\t|| b - A^-1 [A^{1/2m} A^{1/2m}]^n b ||_{n=" << n << "} = " << res);
				REQUIRE( res < EPS );				
			}

			SECTION( std::string("[A^{-1/(2n)}]^{2n}_") + lt + " n=" + std::to_string(n) ) {
				y = B;
				// y = [A^{-1/2n}]^{2n} B = A^{-1} B:
				for(int i=0; i<n; ++i) {
					// x = A^{-1/2n} B:
					iter += rational_approx_cg_multishift(x, y, U, RA.alpha_inv_hi[n], RA.beta_inv_hi[n], D, eps);
					// y = A^{-1/2n} x = A^{-1/n} B:
					iter += rational_approx_cg_multishift(y, x, U, RA.alpha_inv_hi[n], RA.beta_inv_hi[n], D, eps);
				}
				// x = A y = B
				D.DDdagger(x, y, U);

				x -= B;
				double res = x.norm()/B.norm();
				INFO("Lattice-type: " << lattice_type << "\t|| b - A [A^{-1/2n} A^{-1/2n}]^n b ||_{n=" << n << "} = " << res);
				REQUIRE( res < EPS );				
			}

			SECTION( std::string("A^{+1/(2n)}A^{-1/(2n)}_") + lt + " n=" + std::to_string(n) ) {
				// y = [A^{+1/2n}] [A^{-1/2n}] B = B:
				iter += rational_approx_cg_multishift(x, B, U, RA.alpha_inv_hi[n], RA.beta_inv_hi[n], D, eps);
				iter += rational_approx_cg_multishift(y, x, U, RA.alpha_hi[n], RA.beta_hi[n], D, eps);

				y -= B;
				double res = y.norm()/B.norm();
				INFO("Lattice-type: " << lattice_type << "\t|| b - [A^{+1/2n} A^{-1/2n}] b ||_{n=" << n << "} = " << res);
				REQUIRE( res < EPS );				
			}

			SECTION( std::string("A^{-1/(2n)}A^{+1/(2n)}_") + lt + " n=" + std::to_string(n) ) {
				// y = [A^{-1/2n}] [A^{+1/2n}] B = B:
				iter += rational_approx_cg_multishift(x, B, U, RA.alpha_hi[n], RA.beta_hi[n], D, eps);
				iter += rational_approx_cg_multishift(y, x, U, RA.alpha_inv_hi[n], RA.beta_inv_hi[n], D, eps);

				y -= B;
				double res = y.norm()/B.norm();
				INFO("Lattice-type: " << lattice_type << "\t|| b - [A^{-1/2n} A^{+1/2n}] b ||_{n=" << n << "} = " << res);
				REQUIRE( res < EPS );				
			}
		}
	}
}