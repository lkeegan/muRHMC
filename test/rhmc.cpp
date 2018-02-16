#include "catch.hpp"
#include "su3.hpp"
#include "4d.hpp"
#include "rhmc.hpp"
#include "rational_approx.hpp"
#include "dirac_op.hpp"
#include "io.hpp"
#include <iostream>

constexpr double EPS = 9.e-10;

TEST_CASE( "RHMC Gauge action self consistency", "[rhmc]" ) {
	// create 4^4 lattice with random U[mu] at each site
	// construct gauge action from staples, compare to plaquette expression
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<gauge> U (grid);
		rational_approx RA(0.0025, 16.0);
		rhmc_params rhmc_pars = {
			5.4, 	// beta
			0.05, 	// mass
			0.0, 	// mu_I
			2, 	// n_f
			4, 	// n_pf
			1.0, 	// tau
			3, 		// n_steps
			1.e-6,	// MD_eps
			1234,	// seed
			false, 	// EE: only simulate even-even sub-block (requires mu_I=0)
			RA		// set of rational approximations
		};
		rhmc rhmc (rhmc_pars);
		rhmc.random_U(U, 12.0);
		double ac_plaq = rhmc.action_U(U);
		double ac_staple = 0;
		for(int ix=0; ix<U.V; ++ix) {
			for(int mu=0; mu<4; ++mu) {
				SU3mat P = U[ix][mu]*rhmc.staple(ix, mu, U);
				ac_staple -= P.trace().real() * rhmc_pars.beta / 12.0;
			}
		}
		REQUIRE( ac_plaq == Approx(ac_staple) );
	}
}

TEST_CASE( "RHMC Momenta P have expected mean < Tr[P^2] > = 4 * VOL", "[rhmc]" ) {
	// P = \sum_i p_i T_i
	// where p_i are normally distributed real numbers, mean=0, variance=1, therefore
	// < Tr[P^2] > = < 0.5 \sum_i (p_i)^2 > = 0.5 * 8 * < variance of p_i > = 4
	// NB: set n very large to test this properly 
	int n = 10;
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<gauge> P (grid);
		rational_approx RA(0.0025, 16.0);
		rhmc_params rhmc_pars = {
			5.4, 	// beta
			0.05, 	// mass
			0.0, 	// mu_I
			2, 	// n_f
			4, 	// n_pf
			1.0, 	// tau
			3, 		// n_steps
			1.e-6,	// MD_eps
			1234,	// seed
			false, 	// EE: only simulate even-even sub-block (requires mu_I=0)
			RA		// set of rational approximations
		};
		rhmc rhmc (rhmc_pars);
		double eps = 10.0/sqrt(static_cast<double>(n*P.V));
		double av = 0;
		for(int i=0; i<n; ++i) {
			rhmc.gaussian_P(P);
			av += rhmc.action_P(P) / static_cast<double>(4*P.V);
		}
		av /= static_cast<double>(n);
		REQUIRE( av == Approx(4.0).margin(eps) );
	}
}

TEST_CASE( "RHMC Gaussian pseudofermions have expected mean < |chi^2| > = 3 * VOL", "[rhmc]" ) {
	// chi = \sum_a r_a + i i_a
	// where r_a, i_a are normally distributed real numbers, mean=0, variance=1/2, therefore
	// < |chi^2| > = < \sum_a (r_a)^2 + (i_a)^2 > = 2 * 3 * < variance of r_a > * VOL = 3 * VOL
	// NB: set n very large to test this properly 
	int n = 10;
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<fermion> chi (grid);
		rational_approx RA(0.0025, 16.0);
		rhmc_params rhmc_pars = {
			5.4, 	// beta
			0.05, 	// mass
			0.0, 	// mu_I
			2, 	// n_f
			4, 	// n_pf
			1.0, 	// tau
			3, 		// n_steps
			1.e-6,	// MD_eps
			1234,	// seed
			false, 	// EE: only simulate even-even sub-block (requires mu_I=0)
			RA		// set of rational approximations
		};
		rhmc rhmc (rhmc_pars);
		double eps = 10.0/sqrt(static_cast<double>(n*chi.V));
		double av = 0;
		for(int i=0; i<n; ++i) {
			rhmc.gaussian_fermion(chi);
			av += chi.squaredNorm() / static_cast<double>(chi.V);
		}
		av /= static_cast<double>(n);
		REQUIRE( av == Approx(3.0).margin(eps) );
	}
}

TEST_CASE( "RHMC EE Gaussian pseudofermions have expected mean < |chi^2| > = 3 * VOL", "[rhmc_EE]" ) {
	// chi = \sum_a r_a + i i_a
	// where r_a, i_a are normally distributed real numbers, mean=0, variance=1/2, therefore
	// < |chi^2| > = < \sum_a (r_a)^2 + (i_a)^2 > = 2 * 3 * < variance of r_a > * VOL = 3 * VOL
	// NB: set n very large to test this properly 
	int n = 10;
	lattice grid (4, true);
	field<fermion> chi (grid, field<fermion>::EVEN_ONLY);
	rational_approx RA(0.0025, 16.0);
	rhmc_params rhmc_pars = {
		5.4, 	// beta
		0.05, 	// mass
		0.0, 	// mu_I
		2, 	// n_f
		4, 	// n_pf
		1.0, 	// tau
		3, 		// n_steps
		1.e-6,	// MD_eps
		1234,	// seed
		false, 	// EE: only simulate even-even sub-block (requires mu_I=0)
		RA		// set of rational approximations
	};
	rhmc_pars.EE = true;
	rhmc rhmc (rhmc_pars);
	double eps = 10.0/sqrt(static_cast<double>(n*chi.V));
	double av = 0;
	for(int i=0; i<n; ++i) {
		rhmc.gaussian_fermion(chi);
		av += chi.squaredNorm() / static_cast<double>(chi.V);
	}
	av /= static_cast<double>(n);
	REQUIRE( av == Approx(3.0).margin(eps) );
}

// returns average deviation from unit determinant and unitarity per matrix, should be ~1e-15
double rhmc_is_field_SU3 (const field<gauge>& U) {
	double norm = 0.0;
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			norm += fabs(U[ix][mu].determinant() - 1.0);
			norm += (U[ix][mu]*U[ix][mu].adjoint() - SU3mat::Identity()).norm();
		}
	}
	return norm / static_cast<double>(U.V*4*2);
}

TEST_CASE( "Reversibility of RHMC", "[rhmc]" ) {

	double mass = 0.055;
	rational_approx RA(mass*mass, 16.0);
	for(int n_f : {2}) {
		for(int n_pf : {4}) {
			rhmc_params rhmc_pars = {
				5.4, 	// beta
				mass, 	// mass
				0.0, 	// mu_I
				n_f, 	// n_f
				n_pf, 	// n_pf
				1.0, 	// tau
				3, 		// n_steps
				1.e-10,	// MD_eps NOTE: reversiblity seems to be linked to this value, which should not be the case!
				1234,	// seed
				false, 	// EE: only simulate even-even sub-block (requires mu_I=0)
				RA		// set of rational approximations
			};

			// create 4^4 lattice with random U[mu] at each site, random gaussian P
			// integrate by tau, P -> -P, integrate by tau, compare to original U
			SECTION( "nf = " + std::to_string(rhmc_pars.n_f) + ", n_pf = " + std::to_string(rhmc_pars.n_pf) + " LEXI - FULL" ) {
				lattice grid (4);
				field<gauge> U (grid);
				field<gauge> U_old (grid);
				rhmc rhmc (rhmc_pars);
				dirac_op D (grid);
				rhmc.random_U(U, 10.0);
				U_old = U;
				rhmc.trajectory(U, D, true);
				U_old -= U;
				double dev = U_old.norm();
				INFO("RHMC reversibility violation: " << dev);
				REQUIRE( dev < EPS);
				REQUIRE( rhmc_is_field_SU3(U) < EPS );
			}

			SECTION( "nf = " + std::to_string(rhmc_pars.n_f) + ", n_pf = " + std::to_string(rhmc_pars.n_pf) + " EO - FULL" ) {
				lattice grid (4, true);
				field<gauge> U (grid);
				field<gauge> U_old (grid);
				rhmc rhmc (rhmc_pars);
				dirac_op D (grid);
				rhmc.random_U(U, 10.0);
				U_old = U;
				rhmc.trajectory(U, D, true);
				U_old -= U;
				double dev = U_old.norm();
				INFO("RHMC reversibility violation: " << dev);
				REQUIRE( dev < EPS);
				REQUIRE( rhmc_is_field_SU3(U) < EPS );
			}

			SECTION( "nf = " + std::to_string(rhmc_pars.n_f) + ", n_pf = " + std::to_string(rhmc_pars.n_pf) + " EO - EVEN_ONLY" ) {
				rhmc_pars.EE = true;
				lattice grid (4, true);
				field<gauge> U (grid);
				field<gauge> U_old (grid);
				rhmc rhmc (rhmc_pars);
				dirac_op D (grid);
				rhmc.random_U(U, 10.0);
				U_old = U;
				rhmc.trajectory(U, D, true);
				U_old -= U;
				double dev = U_old.norm();
				INFO("RHMC reversibility violation: " << dev);
				REQUIRE( dev < EPS);
				REQUIRE( rhmc_is_field_SU3(U) < EPS );
			}
		}
	}
}

TEST_CASE( "Energy conservation of RHMC", "[rhmc]" ) {

	double mass = 0.055;
	rational_approx RA(mass*mass, 16.0);
	for(int n_f : {2}) {
		for(int n_pf : {4}) {
			rhmc_params rhmc_pars = {
				5.4, 	// beta
				mass, 	// mass
				0.0, 	// mu_I
				n_f, 	// n_f
				n_pf, 	// n_pf
				0.03, 	// tau
				4, 	// n_steps
				1.e-12,	// MD_eps
				1234,	// seed
				false, 	// EE: only simulate even-even sub-block (requires mu_I=0)
				RA		// set of rational approximations
			};

			// create 4^4 lattice with random U[mu] at each site, random gaussian P
			// integrate by tau, P -> -P, integrate by tau, compare to original U
			SECTION( "nf = " + std::to_string(rhmc_pars.n_f) + ", n_pf = " + std::to_string(rhmc_pars.n_pf) + " LEXI - FULL" ) {
				lattice grid (4);
				field<gauge> U (grid);
				field<gauge> U_old (grid);
				rhmc rhmc (rhmc_pars);
				dirac_op D (grid);
				rhmc.random_U(U, 10.0);
				rhmc.trajectory(U, D);
				CAPTURE(rhmc.deltaE);
				REQUIRE( fabs(rhmc.deltaE) < 1.e-6 * U.V );		
			}

			SECTION( "nf = " + std::to_string(rhmc_pars.n_f) + ", n_pf = " + std::to_string(rhmc_pars.n_pf) + " EO - FULL" ) {
				lattice grid (4, true);
				field<gauge> U (grid);
				field<gauge> U_old (grid);
				rhmc rhmc (rhmc_pars);
				dirac_op D (grid);
				rhmc.random_U(U, 10.0);
				rhmc.trajectory(U, D);
				CAPTURE(rhmc.deltaE);
				REQUIRE( fabs(rhmc.deltaE) < 1.e-6 * U.V );		
			}

			SECTION( "nf = " + std::to_string(rhmc_pars.n_f) + ", n_pf = " + std::to_string(rhmc_pars.n_pf) + " EO - EVEN_ONLY" ) {
				rhmc_pars.EE = true;
				lattice grid (4, true);
				field<gauge> U (grid);
				field<gauge> U_old (grid);
				rhmc rhmc (rhmc_pars);
				dirac_op D (grid);
				rhmc.random_U(U, 10.0);
				rhmc.trajectory(U, D);
				CAPTURE(rhmc.deltaE);
				REQUIRE( fabs(rhmc.deltaE) < 1.e-6 * U.V );		
			}
		}
	}
}
