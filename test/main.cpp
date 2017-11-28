#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "su3.hpp"
#include "4d.hpp"
#include "hmc.hpp"
#include "dirac_op.hpp"
#include "io.hpp"
#include <complex>
#include <iostream>

// default tolerance for floating-point equality tests
Approx approx = Approx::custom().epsilon( 5e-14 );

// some default parameters for the HMC
hmc_params hmc_params = {
	5.4, 	// beta
	0.292, 	// mass
	0.157, 	// mu_I
	1.0, 	// tau
	7, 		// n_steps
	1.e-6,	// MD_eps
	1234,	// seed
	false,	// constrained HMC (fixed allowed range for pion susceptibility)
	3.0, 	// suscept_central
	0.05	// suscept_eps
};

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

double is_field_equal (const field<gauge>& U_lhs, const field<gauge>& U_rhs) {
	double norm = 0.0;
	for(int ix=0; ix<U_lhs.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			norm += (U_lhs[ix][mu] - U_rhs[ix][mu]).norm();
		}
	}
	return norm / static_cast<double>(U_lhs.V*4);
}

// returns sqrt(\sum_i |lhs_i-rhs_i|^2)
double is_field_equal (const field<fermion>& lhs, const field<fermion>& rhs) {
	double norm = 0.0;
	for(int ix=0; ix<lhs.V; ++ix) {
		norm += (lhs[ix]-rhs[ix]).squaredNorm();
	}
	return sqrt(norm);
}

TEST_CASE( "Nearest Neigbours: 4^4 lattice of U[mu]", "[lattice]" ) {
	// create 4^4 lattice with random U[mu] at each site
	lattice grid (4);
	field<gauge> U (grid);
	hmc hmc (hmc_params);
	hmc.random_U (U, 10.0);
	// check up/dn neighbours consistent with 4-vec neighbours
	int i = grid.index(2, 3, 0, 2);
	REQUIRE( U.at(3, 3, 0, 2) == U.up(i, 0) );
	REQUIRE( U.at(1, 3, 0, 2) == U.dn(i, 0) );
	REQUIRE( U.at(2, 0, 0, 2) == U.up(i, 1) );
	REQUIRE( U.at(2, 2, 0, 2) == U.dn(i, 1) );
	REQUIRE( U.at(2, 3, 1, 2) == U.up(i, 2) );
	REQUIRE( U.at(2, 3, 3, 2) == U.dn(i, 2) );
	REQUIRE( U.at(2, 3, 0, 3) == U.up(i, 3) );
	REQUIRE( U.at(2, 3, 0, 1) == U.dn(i, 3) );

	i = grid.index(0, 3, 3, 0);
	REQUIRE( U.at(1, 3, 3, 0) == U.up(i, 0) );
	REQUIRE( U.at(3, 3, 3, 0) == U.dn(i, 0) );
	REQUIRE( U.at(0, 0, 3, 0) == U.up(i, 1) );
	REQUIRE( U.at(0, 2, 3, 0) == U.dn(i, 1) );
	REQUIRE( U.at(0, 3, 0, 0) == U.up(i, 2) );
	REQUIRE( U.at(0, 3, 2, 0) == U.dn(i, 2) );
	REQUIRE( U.at(0, 3, 3, 1) == U.up(i, 3) );
	REQUIRE( U.at(0, 3, 3, 3) == U.dn(i, 3) );
}

TEST_CASE( "Nearest Neighbours: 12x6x3x2 lattice of U[mu]", "[lattice]" ) {
	// create 12x6x3x2 lattice with random U[mu] at each site
	lattice grid (12,6,3,2);
	field<gauge> U (grid);
	hmc hmc (hmc_params);
	hmc.random_U (U, 10.0);
	// check up/dn neighbours consistent with 4-vec neighbours
	int i = grid.index(7, 5, 1, 1);
	REQUIRE( U.at(8, 5, 1, 1) == U.up(i, 0) );
	REQUIRE( U.at(6, 5, 1, 1) == U.dn(i, 0) );
	REQUIRE( U.at(7, 0, 1, 1) == U.up(i, 1) );
	REQUIRE( U.at(7, 4, 1, 1) == U.dn(i, 1) );
	REQUIRE( U.at(7, 5, 2, 1) == U.up(i, 2) );
	REQUIRE( U.at(7, 5, 0, 1) == U.dn(i, 2) );
	REQUIRE( U.at(7, 5, 1, 0) == U.up(i, 3) );
	REQUIRE( U.at(7, 5, 1, 0) == U.dn(i, 3) );
}

TEST_CASE( "Time-slices: 12x6x4x2 lattice of U[mu]", "[lattice]" ) {
	// create 12x6x4x2 lattice with random U[mu] at each site
	// compare av plaquette at each timeslice using it_ix index vs
	// with explicit slow indexing of time slice using "at"
	int L0 = 12;
	int L1 = 6;
	int L2 = 4;
	int L3 = 2;
	lattice grid (L0, L1, L2, L3);
	field<gauge> U (grid);
	hmc hmc (hmc_params);
	hmc.random_U (U, 10.0);

	double plaq_slice_sum = 0;
	for(int x0=0; x0<U.L0; ++x0) {
		double plaq_slice = 0;
		double plaq_at = 0;
		// construct plaq using timeslice
		for(int ix3=0; ix3<U.VOL3; ++ix3) {
			plaq_slice += hmc.plaq(U.it_ix(x0, ix3), U);
		}
		// construct plaq using grid debugging indexing
		for(int x1=0; x1<L1; ++x1) {
			for(int x2=0; x2<L2; ++x2) {
				for(int x3=0; x3<L3; ++x3) {
					plaq_at += hmc.plaq(grid.index(x0, x1, x2, x3), U);
				}
			}
		}
		REQUIRE( plaq_slice == approx(plaq_at) );
		plaq_slice_sum += plaq_slice;
	}
	// check that sum over time slices agrees with normal plaquette
	REQUIRE( plaq_slice_sum == approx(static_cast<double>(6*3*U.V) * hmc.plaq(U)) );
}

TEST_CASE( "SU3 Generators T_a", "[su3]" ) {
	// T_a are hermitian & traceless
	// Tr[T_a T_b] = 0.5 delta_ab
	SU3_Generators T;
	for (int a=0; a<8; ++a) {
		REQUIRE( (T[a]).trace().real() == approx(0) );
		REQUIRE( (T[a]).trace().imag() == approx(0) );
		REQUIRE( (T[a]-T[a].adjoint()).norm() == approx(0) );
		for (int b=0; b<8; ++b) {
			if(a==b) {
				REQUIRE( (T[a]*T[b]).trace().real() == approx(0.5) );
				REQUIRE( (T[a]*T[b]).trace().imag() == approx(0) );
			}
			else {
				REQUIRE( (T[a]*T[b]).trace().real() == approx(0) );				
				REQUIRE( (T[a]*T[b]).trace().imag() == approx(0) );				
			}
			
		}
	}
}

TEST_CASE( "Gauge action self consistency", "[hmc]" ) {
	// create 4^4 lattice with random U[mu] at each site
	// construct gauge action from staples, compare to plaquette expression
	lattice grid (4);
	field<gauge> U (grid);
	hmc hmc (hmc_params);
	hmc.random_U(U, 12.0);
	double ac_plaq = hmc.action_U(U);
	double ac_plaq_local = 0;
	double ac_staple = 0;
	for(int ix=0; ix<U.V; ++ix) {
		ac_plaq_local += hmc.plaq(ix, U);
		for(int mu=0; mu<4; ++mu) {
			SU3mat P = U[ix][mu]*hmc.staple(ix, mu, U);
			ac_staple -= P.trace().real() * hmc_params.beta / 12.0;
		}
	}
	ac_plaq_local *= (-hmc_params.beta / 3.0);
	REQUIRE( ac_plaq == approx(ac_staple) );
	REQUIRE( ac_plaq_local == approx(ac_staple) );
}

TEST_CASE( "Momenta P have expected mean < Tr[P^2] > = 4 * VOL", "[hmc]" ) {
	// P = \sum_i p_i T_i
	// where p_i are normally distributed real numbers, mean=0, variance=1, therefore
	// < Tr[P^2] > = < 0.5 \sum_i (p_i)^2 > = 0.5 * 8 * < variance of p_i > = 4
	// NB: set n very large to test this properly 
	int n = 10;
	lattice grid (4);
	field<gauge> P (grid);
	hmc hmc (hmc_params);
	double eps = 1.0/sqrt(static_cast<double>(n*P.V));
	double av = 0;
	for(int i=0; i<n; ++i) {
		hmc.gaussian_P(P);
		av += hmc.action_P(P) / static_cast<double>(4*P.V);
	}
	av /= static_cast<double>(n);
	REQUIRE( av == approx(4.0).epsilon(eps) );
}

TEST_CASE( "Gaussian pseudofermions have expected mean < |chi^2| > = 3 * VOL", "[hmc]" ) {
	// chi = \sum_a r_a + i i_a
	// where r_a, i_a are normally distributed real numbers, mean=0, variance=1/2, therefore
	// < |chi^2| > = < \sum_a (r_a)^2 + (i_a)^2 > = 2 * 3 * < variance of r_a > * VOL = 3 * VOL
	// NB: set n very large to test this properly 
	int n = 10;
	lattice grid (4);
	field<fermion> chi (grid);
	hmc hmc (hmc_params);
	double eps = 1.0/sqrt(static_cast<double>(n*chi.V));
	double av = 0;
	for(int i=0; i<n; ++i) {
		hmc.gaussian_fermion(chi);
		av += chi.squaredNorm() / static_cast<double>(chi.V);
	}
	av /= static_cast<double>(n);
	REQUIRE( av == approx(3.0).epsilon(eps) );
}

TEST_CASE( "D(m, mu) = -eta_5 D(-m, mu) eta_5", "[staggered]") {
	lattice grid (4);
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

TEST_CASE( "Gauge covariance of D(m, mu)", "[staggered]") {
	lattice grid (4);
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

TEST_CASE( "CG inversion of (D+m)(D+m)^dagger", "[inverters]") {
	lattice grid (4);
	field<gauge> U (grid);
	hmc hmc (hmc_params);
	hmc.random_U(U, 10.0);
	dirac_op D (grid);

	// make random fermion field chi
	field<fermion> chi (grid);
	hmc.gaussian_fermion (chi);

	field<fermion> x (grid);
	field<fermion> y (grid);

	for (double eps=1e-3; eps>1.e-14; eps*=1.e-3) {
		// x = (DD')^-1 chi
		int iter = D.cg(x, chi, U, hmc_params.mass, hmc_params.mu_I, eps);

		// y = (DD') x ?= chi
		D.DDdagger(y, x, U, hmc_params.mass, hmc_params.mu_I);
		double dev = is_field_equal(y, chi);
		INFO("CG: eps = " << eps << "\t iterations = " << iter << "\t error = " << dev);
		REQUIRE( dev == Approx(0).epsilon(5e-13 + 10.0*eps) );
	}
}

TEST_CASE( "Reversibility of HMC", "[hmc]" ) {
	// create 4^4 lattice with random U[mu] at each site, random gaussian P
	// integrate by tau, P -> -P, integrate by tau, compare to original U

	lattice grid (4);
	field<gauge> U (grid);
	field<gauge> P (grid);
	field<gauge> U_old (grid);
	hmc_params.n_steps = 3;
	hmc hmc (hmc_params);
	dirac_op D (grid);
	hmc.random_U(U, 10.0);
	U_old = U;
	hmc.gaussian_P(P);
	// make gaussian fermion field
	field<fermion> phi (grid);
	hmc.gaussian_fermion (phi);

	SECTION( "Leapfrog integrator") {
		int iter = hmc.leapfrog (U, phi, P, D);
		// P <- -P
		P *= -1;
		iter += hmc.leapfrog (U, phi, P, D);

		double dev = is_field_equal(U, U_old);
		INFO("HMC reversibility violation: " << dev << "\t MD_eps: " << hmc_params.MD_eps << "\t CG iter: " << iter);
		REQUIRE( dev == approx(0).epsilon(10.0*hmc_params.MD_eps) );
		REQUIRE( is_field_hermitian(P) == approx(0) );
		REQUIRE( is_field_SU3(U) == approx(0) );		
	}

	SECTION( "OMF2 integrator") {
		int iter = hmc.OMF2 (U, phi, P, D);
		// P <- -P
		P *= -1;
		iter += hmc.OMF2 (U, phi, P, D);

		double dev = is_field_equal(U, U_old);
		INFO("HMC reversibility violation: " << dev << "\t MD_eps: " << hmc_params.MD_eps << "\t CG iter: " << iter);
		REQUIRE( dev == approx(0).epsilon(10.0*hmc_params.MD_eps) );
		REQUIRE( is_field_hermitian(P) == approx(0) );
		REQUIRE( is_field_SU3(U) == approx(0) );
		
	}
}

TEST_CASE( "HMC conserves action for small tau", "[hmc]" ) {
	// create 4^4 lattice with random U[mu] at each site, random gaussian P
	// integrate by a small amount, check action is conserved within some eps

	lattice grid (4);
	field<gauge> U (grid);
	field<gauge> P (grid);
	hmc_params.tau = 0.05;
	hmc_params.n_steps = 7;
	hmc_params.MD_eps = 1.e-10;
	hmc hmc (hmc_params);
	dirac_op D (grid);
	hmc.random_U(U, 0.2);
	hmc.trajectory(U, D);
	INFO("HMC deltaE: " << hmc.deltaE << "\t MD_eps: " << hmc_params.MD_eps 
		 << "\t tau: " << hmc_params.tau << "\t n_steps: " << hmc_params.n_steps);
	REQUIRE( hmc.deltaE == approx(0).epsilon(1.e-6 * U.V) );		
}

/*
TEST_CASE( "CG vs CG-multishift inversion", "[inverters]") {
	double eps = 1.e-10;
	lattice grid (4);
	field<gauge> U (grid);
	hmc hmc (hmc_params);
	hmc.random_U(U, 10.0);
	std::vector<double> sigma = {0.1285, 0.158, 0.201};
	dirac_op D (grid);

	// make random fermion field chi
	field<fermion> chi (grid);
	hmc.gaussian_fermion (chi);

	// make vector of x's to store shifted inversions
	std::vector<field<fermion>> x;
	for(int i_m=0; i_m<static_cast<int>(sigma.size()); ++i_m) {
		x.push_back(field<fermion>(grid));
	}	
	field<fermion> y (grid), z(grid);

	// x_i = (DD' + m_i^2)^-1 chi
	int iter = D.cg_multishift(x, chi, U, hmc_params.mass, hmc_params.mu_I, sigma, eps);
	INFO("CG_multishift: eps = " << eps << "\t iterations = " << iter << "\t lowest shift = " << sigma[0]);

	for(int i_m=0; i_m<static_cast<int>(sigma.size()); ++i_m) {
		// y = (DD' + m^2)^-1 chi
		double effective_mass = sqrt(hmc_params.mass*hmc_params.mass + sigma[i_m]*sigma[i_m]);
		int iter = D.cg(y, chi, U, effective_mass, hmc_params.mu_I, eps);
		INFO("CG: eps = " << eps << "\t\t iterations = " << iter << "\t shift" << sigma[i_m]);

		D.DDdagger(z, y, U, effective_mass, hmc_params.mu_I);
		double devCG = is_field_equal(z, chi);

		D.DDdagger(z, x[i_m], U, effective_mass, hmc_params.mu_I);
		double devCG_multishift = is_field_equal(z, chi);

		INFO("shift: " << sigma[i_m] << "\tCG deviation: " << devCG 
		 	 << "\t CG_multishift deviation: " << devCG_multishift);
		REQUIRE( devCG == Approx(0).epsilon(5e-13 + 10.0*eps) );
		REQUIRE( devCG_multishift == Approx(0).epsilon(5e-13 + 10.0*eps) );
	}
}
*/
TEST_CASE( "Read/Write gauge fields to file", "[IO]") {
	// create random U, measure plaquette, write to file
	// load U from file, check plaquette matches
	lattice grid (4);
	field<gauge> U1 (grid);
	field<gauge> U2 (grid);
	hmc hmc (hmc_params);
	hmc.random_U(U1, 0.7);

	double plaq1 = hmc.plaq(U1);
	write_gauge_field(U1, "tmp_test_data", 666);
	read_gauge_field(U2, "tmp_test_data", 666);
	double plaq2 = hmc.plaq(U2);
	REQUIRE ( plaq1 == approx(plaq2));
}

TEST_CASE( "Explicit Dirac op matrix equivalent to sparse MVM op", "[Eigenvalues]") {
	lattice grid (4);
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