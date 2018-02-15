#include "catch.hpp"
#include "su3.hpp"
#include "4d.hpp"
#include "hmc.hpp"
#include "io.hpp"

TEST_CASE( "IO Tests", "[IO]") {
	// HMC for generating random fields
	hmc_params hmc_pars;
	hmc_pars.seed = 123;
	hmc hmc (hmc_pars);

	SECTION( "Read/Write gauge fields to file") {
		// create random U, measure plaquette, write to file
		// load U from file, check plaquette matches
		for(bool isEO_in : {false, true}) {
			for(bool isEO_out : {false, true}) {
				lattice grid_in (4, isEO_out);
				lattice grid_out (4, isEO_in);
				field<gauge> U1 (grid_out);
				field<gauge> U2 (grid_in);
				hmc.random_U(U1, 0.7);
				double plaq1 = hmc.plaq(U1);
				write_gauge_field(U1, "tmp_test_data", 666);
				read_gauge_field(U2, "tmp_test_data", 666);
				double plaq2 = hmc.plaq(U2);
				REQUIRE ( plaq1 == Approx(plaq2));
			}
		}
	}

	SECTION( "Read/Write fermion fields to file") {
		// create random fermion field, write to file
		// load fermion field from file, compare to original
		for(bool isEO : {false, true}) {
			lattice grid (4, isEO);
			field<fermion> f1 (grid);
			field<fermion> f2 (grid);
			hmc.gaussian_fermion(f1);

			write_fermion_field(f1, "tmp_test_data_fermions");
			read_fermion_field(f2, "tmp_test_data_fermions");
			f2 -= f1;
			REQUIRE ( f2.norm() == Approx(0));
		}
	}
}