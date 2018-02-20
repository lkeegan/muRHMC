#include "catch.hpp"
#include "su3.hpp"
#include "4d.hpp"
#include "hmc.hpp"
#include "dirac_op.hpp"
#include "io.hpp"
#include <iostream>

constexpr double EPS = 5.e-13;

TEST_CASE( "Gauge action self consistency", "[hmc]" ) {
	// create 4^4 lattice with random U[mu] at each site
	// construct gauge action from staples, compare to plaquette expression
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<gauge> U (grid);
		hmc_params hmc_pars;
		hmc_pars.beta = 6.0;
		hmc_pars.seed = 123;
		hmc hmc (hmc_pars);
		hmc.random_U(U, 12.0);
		double ac_plaq = hmc.action_U(U);
		double ac_plaq_local = 0;
		double ac_staple = 0;
		for(int ix=0; ix<U.V; ++ix) {
			ac_plaq_local += hmc.plaq(ix, U);
			for(int mu=0; mu<4; ++mu) {
				SU3mat P = U[ix][mu]*hmc.staple(ix, mu, U);
				ac_staple -= P.trace().real() * hmc_pars.beta / 12.0;
			}
		}
		ac_plaq_local *= (-hmc_pars.beta / 3.0);
		REQUIRE( ac_plaq == Approx(ac_staple) );
		REQUIRE( ac_plaq_local == Approx(ac_staple) );
	}
}

TEST_CASE( "Momenta P have expected mean < Tr[P^2] > = 4 * VOL", "[hmc]" ) {
	// P = \sum_i p_i T_i
	// where p_i are normally distributed real numbers, mean=0, variance=1, therefore
	// < Tr[P^2] > = < 0.5 \sum_i (p_i)^2 > = 0.5 * 8 * < variance of p_i > = 4
	// NB: set n very large to test this properly 
	int n = 10;
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<gauge> P (grid);
		hmc_params hmc_pars;
		hmc_pars.seed = 123;
		hmc hmc (hmc_pars);
		double eps = 10.0/sqrt(static_cast<double>(n*P.V));
		double av = 0;
		for(int i=0; i<n; ++i) {
			hmc.gaussian_P(P);
			av += hmc.action_P(P) / static_cast<double>(4*P.V);
		}
		av /= static_cast<double>(n);
		REQUIRE( av == Approx(4.0).margin(eps) );
	}
}

TEST_CASE( "Gaussian pseudofermions have expected mean < |chi^2| > = 3 * VOL", "[hmc]" ) {
	// chi = \sum_a r_a + i i_a
	// where r_a, i_a are normally distributed real numbers, mean=0, variance=1/2, therefore
	// < |chi^2| > = < \sum_a (r_a)^2 + (i_a)^2 > = 2 * 3 * < variance of r_a > * VOL = 3 * VOL
	// NB: set n very large to test this properly 
	int n = 10;
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<fermion> chi (grid);
		hmc_params hmc_pars;
		hmc_pars.seed = 123;
		hmc hmc (hmc_pars);
		double eps = 10.0/sqrt(static_cast<double>(n*chi.V));
		double av = 0;
		for(int i=0; i<n; ++i) {
			hmc.gaussian_fermion(chi);
			av += chi.squaredNorm() / static_cast<double>(chi.V);
		}
		av /= static_cast<double>(n);
		REQUIRE( av == Approx(3.0).margin(eps) );
	}
}

TEST_CASE( "EE Gaussian pseudofermions have expected mean < |chi^2| > = 3 * VOL", "[hmc_EE]" ) {
	// chi = \sum_a r_a + i i_a
	// where r_a, i_a are normally distributed real numbers, mean=0, variance=1/2, therefore
	// < |chi^2| > = < \sum_a (r_a)^2 + (i_a)^2 > = 2 * 3 * < variance of r_a > * VOL = 3 * VOL
	// NB: set n very large to test this properly 
	int n = 10;
	lattice grid (4, true);
	field<fermion> chi (grid, field<fermion>::EVEN_ONLY);
	hmc_params hmc_pars;
	hmc_pars.seed = 123;
	hmc_pars.EE = true;
	hmc hmc (hmc_pars);
	double eps = 10.0/sqrt(static_cast<double>(n*chi.V));
	double av = 0;
	for(int i=0; i<n; ++i) {
		hmc.gaussian_fermion(chi);
		av += chi.squaredNorm() / static_cast<double>(chi.V);
	}
	av /= static_cast<double>(n);
	REQUIRE( av == Approx(3.0).margin(eps) );
}

// returns average deviation from hermitian per matrix, should be ~1e-15
double is_field_hermitian (const field<gauge>& P) {
	double norm = 0.0;
	for(int ix=0; ix<P.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			norm += (P[ix][mu].adjoint() - P[ix][mu]).norm();
		}
	}
	return norm / static_cast<double>(P.V*4);
}

// returns average deviation from unit determinant and unitarity per matrix, should be ~1e-15
double is_field_SU3 (const field<gauge>& U) {
	double norm = 0.0;
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			norm += fabs(U[ix][mu].determinant() - 1.0);
			norm += (U[ix][mu]*U[ix][mu].adjoint() - SU3mat::Identity()).norm();
		}
	}
	return norm / static_cast<double>(U.V*4*2);
}

TEST_CASE( "Reversibility of pure gauge HMC", "[hmc]" ) {

	hmc_params hmc_pars = {
		5.4, 	// beta
		0.100, 	// mass
		0.005, // mu_I
		1.0, 	// tau
		3, 		// n_steps
		1.e-7,	// MD_eps
		1234,	// seed
		false, 	// EE: only simulate even-even sub-block (requires mu_I=0)
		false,	// constrained HMC (fixed allowed range for pion susceptibility)
		3.0, 	// suscept_central
		0.05	// suscept_eps
	};

	// create 4^4 lattice with random U[mu] at each site, random gaussian P
	// integrate by tau, P -> -P, integrate by tau, compare to original U
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<gauge> U (grid);
		field<gauge> P (grid);
		field<gauge> U_old (grid);
		field<gauge> P_old (grid);
		hmc hmc (hmc_pars);
		hmc.random_U(U, 0.2);
		U_old = U;
		hmc.gaussian_P(P);
		P_old = P;
		// make gaussian fermion field
		field<fermion> phi (grid);
		hmc.gaussian_fermion (phi);

		SECTION( "Leapfrog integrator isEO" + std::to_string(isEO)) {
			hmc.leapfrog_pure_gauge (U, P);
			// P <- -P
			P *= -1;
			hmc.leapfrog_pure_gauge (U, P);

			U_old -= U;
			double dev = U_old.norm();
			P_old += P;
			double devP = P_old.norm();
			INFO("HMC reversibility violation: " << dev << "\t P_dev: " << devP);
			REQUIRE( dev < EPS );
			REQUIRE( devP < EPS );
			REQUIRE( is_field_hermitian(P) < EPS );
			REQUIRE( is_field_SU3(U) < EPS );	
		}
	}
}

TEST_CASE( "Reversibility of HMC", "[hmc]" ) {

	hmc_params hmc_pars = {
		5.4, 	// beta
		0.100, 	// mass
		0.005, // mu_I
		1.0, 	// tau
		3, 		// n_steps
		1.e-7,	// MD_eps
		1234,	// seed
		false, 	// EE: only simulate even-even sub-block (requires mu_I=0)
		false,	// constrained HMC (fixed allowed range for pion susceptibility)
		3.0, 	// suscept_central
		0.05	// suscept_eps
	};

	// create 4^4 lattice with random U[mu] at each site, random gaussian P
	// integrate by tau, P -> -P, integrate by tau, compare to original U
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<gauge> U (grid);
		field<gauge> P (grid);
		field<gauge> U_old (grid);
		field<gauge> P_old (grid);
		hmc hmc (hmc_pars);
		dirac_op D (grid, hmc_pars.mass, hmc_pars.mu_I);
		hmc.random_U(U, 0.2);
		U_old = U;
		hmc.gaussian_P(P);
		P_old = P;
		// make gaussian fermion field
		field<fermion> phi (grid);
		hmc.gaussian_fermion (phi);

		SECTION( "Leapfrog integrator isEO" + std::to_string(isEO)) {
			int iter = hmc.leapfrog (U, phi, P, D);
			// P <- -P
			P *= -1;
			iter += hmc.leapfrog (U, phi, P, D);

			U_old -= U;
			double dev = U_old.norm();
			P_old += P;
			double devP = P_old.norm();
			INFO("HMC reversibility violation: " << dev << "\t P_dev: " << devP << "\t MD_eps: " << hmc_pars.MD_eps << "\t CG iter: " << iter);
			REQUIRE( dev < EPS );
			REQUIRE( devP < EPS );
			REQUIRE( is_field_hermitian(P) < EPS );
			REQUIRE( is_field_SU3(U) < EPS );	
		}

		SECTION( "OMF2 integrator isEO" + std::to_string(isEO)) {
			int iter = hmc.OMF2 (U, phi, P, D);
			// P <- -P
			P *= -1;
			iter += hmc.OMF2 (U, phi, P, D);

			U_old -= U;
			double dev = U_old.norm();
			P_old += P;
			double devP = P_old.norm();
			INFO("HMC reversibility violation: " << dev << "\t P_dev: " << devP << "\t MD_eps: " << hmc_pars.MD_eps << "\t CG iter: " << iter);
			REQUIRE( dev < EPS );
			REQUIRE( devP < EPS );
			REQUIRE( is_field_hermitian(P) < EPS );
			REQUIRE( is_field_SU3(U) < EPS );		
		}
	}
}

TEST_CASE( "Reversibility of EE HMC", "[hmc_EE]" ) {

	hmc_params hmc_pars = {
		5.4, 	// beta
		0.05, 	// mass
		0.0157, // mu_I
		1.0, 	// tau
		3, 		// n_steps
		1.e-3,	// MD_eps
		1234,	// seed
		true, 	// EE: only simulate even-even sub-block (requires mu_I=0)
		false,	// constrained HMC (fixed allowed range for pion susceptibility)
		3.0, 	// suscept_central
		0.05	// suscept_eps
	};

	// create 4^4 lattice with random U[mu] at each site, random gaussian P
	// integrate by tau, P -> -P, integrate by tau, compare to original U
	lattice grid (4, true);
	field<gauge> U (grid);
	field<gauge> P (grid);
	field<gauge> U_old (grid);
	field<gauge> P_old (grid);
	hmc hmc (hmc_pars);
	dirac_op D (grid, hmc_pars.mass, hmc_pars.mu_I);
	hmc.random_U(U, 0.2);
	U_old = U;
	hmc.gaussian_P(P);
	P_old = P;
	// make gaussian fermion field
	field<fermion> phi (grid, field<fermion>::EVEN_ONLY);
	hmc.gaussian_fermion (phi);

	SECTION( "EE Leapfrog integrator") {
		int iter = hmc.leapfrog (U, phi, P, D);
		// P <- -P
		P *= -1;
		iter += hmc.leapfrog (U, phi, P, D);

		U_old -= U;
		double dev = U_old.norm();
		P_old += P;
		double devP = P_old.norm();
		INFO("HMC reversibility violation: " << dev << "\t P_dev: " << devP << "\t MD_eps: " << hmc_pars.MD_eps << "\t CG iter: " << iter);
		REQUIRE( dev < EPS );
		REQUIRE( devP < EPS );
		REQUIRE( is_field_hermitian(P) < EPS );
		REQUIRE( is_field_SU3(U) < EPS );		
	}

	SECTION( "EE OMF2 integrator") {
		int iter = hmc.OMF2 (U, phi, P, D);
		// P <- -P
		P *= -1;
		iter += hmc.OMF2 (U, phi, P, D);

		U_old -= U;
		double dev = U_old.norm();
		P_old += P;
		double devP = P_old.norm();
		INFO("HMC reversibility violation: " << dev << "\t P_dev: " << devP << "\t MD_eps: " << hmc_pars.MD_eps << "\t CG iter: " << iter);
		REQUIRE( dev < EPS );
		REQUIRE( devP < EPS );
		REQUIRE( is_field_hermitian(P) < EPS );
		REQUIRE( is_field_SU3(U) < EPS );		
	}
}

TEST_CASE( "HMC EE force term matches full HMC term with even phi sites -> 0", "[hmc_EE]" ) {
	hmc_params hmc_pars = {
		5.4, 	// beta
		0.292, 	// mass
		0.00,   // mu_I
		0.05, 	// tau
		20, 	// n_steps
		1.e-12,	// MD_eps
		1234,	// seed
		false, 	// EE: only simulate even-even sub-block (requires mu_I=0)
		false,	// constrained HMC (fixed allowed range for pion susceptibility)
		3.0, 	// suscept_central
		0.05	// suscept_eps
	};

	hmc_params hmc_EE_pars = hmc_pars;
	hmc_EE_pars.EE = true;

	// create 4^4 lattice with random U[mu] at each site, random gaussian P
	// integrate by a small amount, check action is conserved within some eps
	lattice grid (4, true);
	field<gauge> U (grid);
	hmc hmc_FULL (hmc_pars);
	hmc hmc_EE (hmc_EE_pars);
	dirac_op D (grid);
	hmc_FULL.random_U(U, 0.2);
	field<gauge> F (grid);
	F.setZero();
	field<gauge> F_EE (grid);
	F_EE.setZero();

	field<fermion> phi (grid);
	hmc_FULL.gaussian_fermion(phi);
	// set odd sites to zero
	for(int i=grid.V/2; i<grid.V; ++i) {
		phi[i].setZero();
	}
	field<fermion> phi_e (grid, field<fermion>::EVEN_ONLY);
	for(int i=0; i<grid.V/2; ++i) {
		phi_e[i] = phi[i];
	}

	hmc_FULL.force_fermion(F, U, phi, D);
	hmc_EE.force_fermion(F_EE, U, phi_e, D);

	F_EE -= F;
	REQUIRE( F_EE.norm() < EPS );		
}
/*
TEST_CASE( "HMC conserves action for small tau", "[hmc]" ) {
	hmc_params hmc_pars = {
		5.4, 	// beta
		0.292, 	// mass
		0.0157, // mu_I
		0.05, 	// tau
		20, 	// n_steps
		1.e-10,	// MD_eps
		1234,	// seed
		false, 	// EE: only simulate even-even sub-block (requires mu_I=0)
		false,	// constrained HMC (fixed allowed range for pion susceptibility)
		3.0, 	// suscept_central
		0.05	// suscept_eps
	};
	// create 4^4 lattice with random U[mu] at each site, random gaussian P
	// integrate by a small amount, check action is conserved within some eps
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<gauge> U (grid);
		hmc hmc (hmc_pars);
		dirac_op D (grid);
		hmc.random_U(U, 0.2);
		hmc.trajectory(U, D);
		CAPTURE(hmc.deltaE);
		CAPTURE(hmc_pars.tau);
		CAPTURE(hmc_pars.n_steps);
		CAPTURE(hmc_pars.MD_eps);
		REQUIRE( fabs(hmc.deltaE) < 1.e-6 * U.V );		
	}
}

TEST_CASE( "EE HMC conserves action for small tau", "[hmc_EE]" ) {
	hmc_params hmc_pars = {
		5.4, 	// beta
		0.292, 	// mass
		0.0, 	// mu_I
		0.05, 	// tau
		20, 	// n_steps
		1.e-10,	// MD_eps
		1234,	// seed
		true, 	// EE: only simulate even-even sub-block (requires mu_I=0)
		false,	// constrained HMC (fixed allowed range for pion susceptibility)
		3.0, 	// suscept_central
		0.05	// suscept_eps
	};
	// create 4^4 lattice with random U[mu] at each site, random gaussian P
	// integrate by a small amount, check action is conserved within some eps
	lattice grid (4, true);
	field<gauge> U (grid);
	field<gauge> P (grid);
	hmc hmc (hmc_pars);
	dirac_op D (grid);
	hmc.random_U(U, 0.2);
	hmc.trajectory(U, D);
	CAPTURE(hmc.deltaE);
	CAPTURE(hmc_pars.tau);
	CAPTURE(hmc_pars.n_steps);
	CAPTURE(hmc_pars.MD_eps);
	REQUIRE( fabs(hmc.deltaE) < 1.e-6 * U.V );		
}
*/