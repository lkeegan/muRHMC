#include "catch.hpp"
#include "su3.hpp"
#include "4d.hpp"
#include "hmc.hpp"
#include "dirac_op.hpp"
#include "io.hpp"

constexpr double EPS = 5.e-14;

TEST_CASE( "D(m, mu) = -eta_5 D(-m, mu) eta_5", "[staggered]") {
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<gauge> U (grid);
		hmc hmc (hmc_params);
		hmc_params.mass = 0.1322;
		hmc_params.mu_I = 0.219;
		hmc.random_U(U, 10.0);
		dirac_op D (grid);

		// make random fermion field chi
		field<fermion> chi (grid);
		hmc.gaussian_fermion(chi);

		// psi1 = -D(mass, mu) chi
		field<fermion> psi1 (grid);
		D.D(psi1, chi, U, hmc_params.mass, hmc_params.mu_I);
		psi1 *= -1;

		// psi2 = eta_5 D(-mass, m) eta_5 chi
		field<fermion> psi2 (grid);
		D.gamma5(chi);
		D.D(psi2, chi, U, -hmc_params.mass, hmc_params.mu_I);
		D.gamma5(psi2);
		REQUIRE( is_field_equal(psi1, psi2) == approx(0) );
	}
}

TEST_CASE( "D(m=0,mu=0) phi (phi_O = 0) = D_OE phi_E, and E<->O", "[EO]") {
	// set mass and mu_I to zero
	hmc_params.mass = 0.0;
	hmc_params.mu_I = 0.0;

	// use EO-ordered lattice
	lattice grid (4, true);
	field<gauge> U (grid);
	hmc hmc (hmc_params);
	hmc.random_U(U, 10.0);
	dirac_op D (grid);

	// make random fermion field chi
	field<fermion> chi (grid);
	hmc.gaussian_fermion(chi);
	// set odd sites to zero
	for(int i=grid.V/2; i<grid.V; ++i) {
		chi[i].setZero();
	}
	REQUIRE( norm_odd(chi) == approx(0) );

	// psi1 = D(mass=0, mu=0) chi
	field<fermion> psi1(grid), psi2(grid);
	D.D(psi1, chi, U, 0.0, 0.0);
	// check that psi1 is odd:
	REQUIRE( norm_even(psi1) == approx(0) );
	// chi = D(mass=0, mu=0) psi1
	D.D(psi2, psi1, U, 0.0, 0.0);
	// check that psi2 is even:
	REQUIRE( norm_odd(psi2) == approx(0) );

	// check that D_eo gives same result:
	field<fermion> psi_e(grid, field<fermion>::EVEN_ONLY), psi_o(grid, field<fermion>::ODD_ONLY);
	psi_e = chi;
	REQUIRE( is_field_equal(psi_e, chi) == approx(0) );
	D.apply_eta_bcs_to_U(U);
	D.D_oe(psi_o, psi_e, U);
	D.D_eo(psi_e, psi_o, U);
	D.apply_eta_bcs_to_U(U);
	REQUIRE( is_field_equal(psi_e, psi2) == approx(0) );
}

TEST_CASE( "DDdagger phi (phi_O = 0) = [m^2 - D_EO D_OE] phi_E", "[EO]") {
	// set mu_I to zero
	hmc_params.mu_I = 0.0;
	hmc_params.mass = 0.128;

	// use EO-ordered lattice
	lattice grid (4, true);
	field<gauge> U (grid);
	hmc hmc (hmc_params);
	hmc.random_U(U, 1.0);
	dirac_op D (grid);

	// make random fermion field chi
	field<fermion> chi (grid);
	hmc.gaussian_fermion(chi);
	// set odd sites to zero
	for(int i=grid.V/2; i<grid.V; ++i) {
		chi[i].setZero();
	}
	REQUIRE( norm_odd(chi) == approx(0) );

	// psi1 = DDdagger(mu=0) chi
	field<fermion> psi1(grid);
	D.DDdagger(psi1, chi, U, hmc_params.mass, 0.0);
	// check that psi1 is even:
	REQUIRE( norm_odd(psi1) == approx(0) );

	// check that D_eo gives same result:
	field<fermion> psi_e(grid, field<fermion>::EVEN_ONLY), psi_o(grid, field<fermion>::ODD_ONLY);
	psi_e = chi;
	REQUIRE( is_field_equal(psi_e, chi) == approx(0) );
	D.apply_eta_bcs_to_U(U);
	D.D_oe(psi_o, psi_e, U);
	D.D_eo(psi_e, psi_o, U);
	psi_e.scale_add(-1.0, hmc_params.mass*hmc_params.mass, chi);
	D.remove_eta_bcs_from_U(U);
	REQUIRE( is_field_equal(psi_e, psi1) == approx(0) );
}

TEST_CASE( "DDdagger phi (phi_O = 0) = DDdagger_ee phi_e", "[EO]") {
	// set mu_I to zero
	hmc_params.mu_I = 0.0;
	hmc_params.mass = 0.128;

	// use EO-ordered lattice
	lattice grid (4, true);
	field<gauge> U (grid);
	hmc hmc (hmc_params);
	hmc.random_U(U, 2.0);
	dirac_op D (grid);

	// make random fermion field chi
	field<fermion> chi (grid);
	hmc.gaussian_fermion(chi);
	// set odd sites to zero
	for(int i=grid.V/2; i<grid.V; ++i) {
		chi[i].setZero();
	}
	REQUIRE( norm_odd(chi) == approx(0) );

	// psi1 = DDdagger(mu=0) chi
	field<fermion> psi1(grid);
	D.DDdagger(psi1, chi, U, hmc_params.mass, 0.0);
	// check that psi1 is even:
	REQUIRE( norm_odd(psi1) == approx(0) );

	// check that DDdagger_ee gives same result:
	field<fermion> chi_e(grid, field<fermion>::EVEN_ONLY), psi_e(grid, field<fermion>::EVEN_ONLY);
	chi_e = chi;
	REQUIRE( is_field_equal(chi_e, chi) == approx(0) );
	D.DDdagger(psi_e, chi_e, U, hmc_params.mass, 0.0);
	REQUIRE( is_field_equal(psi_e, psi1) == approx(0) );
}

TEST_CASE( "Gauge covariance of D(m, mu)", "[staggered]") {
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<gauge> U (grid);
		field<gauge> V (grid);
		hmc hmc (hmc_params);
		hmc_params.mass = 0.1322;
		hmc_params.mu_I = 0.219;
		hmc.random_U(U, 10.0);
		// V[ix][0] is a random SU(3) matrix at each site
		hmc.random_U(V, 2.7);
		dirac_op D (grid);

		// make random fermion field chi
		field<fermion> chi (grid);
		hmc.gaussian_fermion(chi);

		// psi1 = D(mass, mu) chi
		field<fermion> psi1 (grid);
		D.D(psi1, chi, U, hmc_params.mass, hmc_params.mu_I);

		// chi[x] -> V[x] chi[x]
		for(int ix=0; ix<U.V; ++ix) {
			chi[ix] = V[ix][0] * chi[ix];
		}
		// psi1[x] -> V[x] psi1[x]
		for(int ix=0; ix<U.V; ++ix) {
			psi1[ix] = V[ix][0] * psi1[ix];
		}
		// U_mu[x] -> V[x] U_mu[x] V[x+mu]^dagger
		for(int ix=0; ix<U.V; ++ix) {
			for(int mu=0; mu<4; ++mu) {
				U[ix][mu] = V[ix][0] * U[ix][mu] * (V.up(ix,mu)[0].adjoint());
			}
		}

		// psi2 = D(mass, m) chi
		field<fermion> psi2 (grid);
		D.D(psi2, chi, U, hmc_params.mass, hmc_params.mu_I);

		REQUIRE( is_field_equal(psi1, psi2) == approx(0) );
	}
}

TEST_CASE( "Explicit Dirac op matrix equivalent to sparse MVM op", "[Eigenvalues]") {
	for(bool isEO : {false, true}) {
			lattice grid (4, isEO);
		field<gauge> U (grid);
		hmc hmc (hmc_params);
		dirac_op D (grid);
		hmc.random_U(U, 10.0);
		field<fermion> phi (grid);
		field<fermion> chi (grid);
		field<fermion> psi (grid);
		// construct explicit dense dirac matrix
		Eigen::MatrixXcd D_matrix = D.D_dense_matrix(U, hmc_params.mass, hmc_params.mu_I);
		// chi = D_dense phi
		Eigen::MatrixXcd D_vec = Eigen::VectorXcd::Random(3*U.V);
		Eigen::MatrixXcd Dvec_vec = D_matrix * D_vec;
		for(int ix=0; ix<U.V; ++ix) {
			phi[ix] = D_vec.block<3,1>(3*ix,0);
			chi[ix] = Dvec_vec.block<3,1>(3*ix,0);
		}
		// psi = D phi
		D.D(psi, phi, U, hmc_params.mass, hmc_params.mu_I);
		// psi ?= chi
		double dev = is_field_equal(psi, chi);
		INFO ("matrix Dirac op deviation from sparse MVM op: " << dev);	
		REQUIRE ( dev == approx(0) );
	}
}

// Note: this routine is not required any more
TEST_CASE( "Bartels-Stewart Matrix solution of AX + XB = C", "[Matrices]") {

	int N = 12;
	Eigen::MatrixXcd A = Eigen::MatrixXcd::Random(N, N);
	Eigen::MatrixXcd B = Eigen::MatrixXcd::Random(N, N);
	Eigen::MatrixXcd C = Eigen::MatrixXcd::Random(N, N);
	Eigen::MatrixXcd X = Eigen::MatrixXcd::Zero(N, N);

	lattice grid (4);
	field<gauge> U (grid);
	dirac_op D (grid);
	D.bartels_stewart(X, A, B, C);

	double dev = (A*X + X*B - C).norm();
	INFO ("|| A*X + X*B - C || = " << dev);	
	REQUIRE ( dev == approx(0).margin(1e-14*N*N) );
}