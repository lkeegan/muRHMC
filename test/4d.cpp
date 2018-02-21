#include "catch.hpp"
#include "su3.hpp"
#include "4d.hpp"
#include "hmc.hpp"

constexpr double EPS = 5.e-14;

TEST_CASE( "moving forwards then backwards does nothing: 4^4 lattice of U[mu]", "[lattice]" ) {
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		for(int i=0; i<grid.V; ++i) {
			int i_0 = grid.iup(i, 0);
			int i_1 = grid.idn(i_0, 2);
			int i_2 = grid.iup(i_1, 2);
			int i_n = grid.idn(i_2, 0);
			INFO ( i << "-> " << i_0 << "-> " << i_1 << "-> " << i_2 << "-> " << i_n);
			REQUIRE ( i == i_n);
		}
	}
}

TEST_CASE( "indexing gives one index per site: 4^4 lattice of U[mu]", "[lattice]" ) {
	for(bool isEO : {false, true}) {
		// create 4^4 lattice with random U[mu] at each site
		lattice grid (4, isEO);
		std::vector<int> sites(grid.V, 0);
		for(int x3=0; x3<grid.L3; ++x3) {
			for(int x2=0; x2<grid.L2; ++x2) {
				for(int x1=0; x1<grid.L1; ++x1) {
					for(int x0=0; x0<grid.L0; ++x0) {
						int i = grid.index(x0, x1, x2, x3);
						int iup = grid.iup(i, 0);
						++sites[grid.iup(iup, 1)];
					}	
				}
			}
		}
		for(int i=0; i<grid.V; ++i) {
			REQUIRE ( sites[i] == 1);
		}
	}
}

TEST_CASE( "Nearest Neigbours: 4^4 lattice of U[mu]", "[lattice]" ) {
	for(bool isEO : {false, true}) {
		// create 4^4 lattice with random U[mu] at each site
		lattice grid (4, isEO);
		field<gauge> U (grid);
		hmc_params hmc_pars;
		hmc_pars.seed = 123;
		hmc rhmc (hmc_pars);
		rhmc.random_U (U, 10.0);
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
}

TEST_CASE( "Nearest Neighbours: 12x6x3x2 lattice of U[mu]", "[lattice]" ) {
	for(bool isEO : {false, true}) {
		// create 12x6x3x2 lattice with random U[mu] at each site
		lattice grid (12, 6, 3, 2, isEO);
		field<gauge> U (grid);
		hmc_params hmc_pars;
		hmc_pars.seed = 123;
		hmc rhmc (hmc_pars);
		rhmc.random_U (U, 10.0);
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
}

TEST_CASE( "Time-slices: 12x6x4x2 lattice of U[mu]", "[lattice]" ) {
	// create 12x6x4x2 lattice with random U[mu] at each site
	// compare av plaquette at each timeslice using it_ix index vs
	// with explicit slow indexing of time slice using "at"
	int L0 = 12;
	int L1 = 6;
	int L2 = 4;
	int L3 = 2;
	for(bool isEO : {false, true}) {
		lattice grid (L0, L1, L2, L3, isEO);
		field<gauge> U (grid);
		hmc_params hmc_pars;
		hmc_pars.seed = 123;
		hmc rhmc (hmc_pars);
		rhmc.random_U (U, 10.0);

		double plaq_slice_sum = 0;
		for(int x0=0; x0<U.L0; ++x0) {
			double plaq_slice = 0;
			double plaq_at = 0;
			// construct plaq using timeslice
			for(int ix3=0; ix3<U.VOL3; ++ix3) {
				plaq_slice += rhmc.plaq(U.it_ix(x0, ix3), U);
			}
			// construct plaq using grid debugging indexing
			for(int x1=0; x1<L1; ++x1) {
				for(int x2=0; x2<L2; ++x2) {
					for(int x3=0; x3<L3; ++x3) {
						plaq_at += rhmc.plaq(grid.index(x0, x1, x2, x3), U);
					}
				}
			}
			REQUIRE( plaq_slice == Approx(plaq_at) );
			plaq_slice_sum += plaq_slice;
		}
		// check that sum over time slices agrees with normal plaquette
		REQUIRE( plaq_slice_sum == Approx(static_cast<double>(6*3*U.V) * rhmc.plaq(U)) );
	}
}

TEST_CASE( "Converting EO<->LEXI indices", "[lattice]" ) {
	lattice grid (4);
	for(int i=0; i<grid.V; ++i) {
		int i_eo = grid.EO_from_LEXI(i);
		int i_lexi = grid.LEXI_from_EO(i_eo);
		INFO ( i << "-> " << i_eo << "-> " << i_lexi);
		REQUIRE ( i_eo < grid.V);
		REQUIRE ( i_lexi < grid.V);
		REQUIRE ( i_lexi == i);
	}
}

TEST_CASE( "x.squaredNorm() and x.dot(x) equivalent", "[4d]" ) {
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<fermion> chi (grid);
		hmc_params hmc_pars;
		hmc_pars.seed = 123;
		hmc rhmc (hmc_pars);
		rhmc.gaussian_fermion(chi);
		REQUIRE( chi.dot(chi).real() == Approx(chi.squaredNorm()) );
		}
}

TEST_CASE( "sqrt(x.squaredNorm()) and x.norm equivalent", "[4d]" ) {
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<fermion> chi (grid);
		hmc_params hmc_pars;
		hmc_pars.seed = 123;
		hmc rhmc (hmc_pars);
		rhmc.gaussian_fermion(chi);
		REQUIRE( chi.norm() == Approx(sqrt(chi.squaredNorm())) );
	}
}
