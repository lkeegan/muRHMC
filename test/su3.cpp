#include "catch.hpp"
#include "su3.hpp"

TEST_CASE( "SU3 Generators T_a", "[su3]" ) {
	// T_a are hermitian & traceless
	// Tr[T_a T_b] = 0.5 delta_ab
	SU3_Generators T;
	for (int a=0; a<8; ++a) {
		REQUIRE( (T[a]).trace().real() == Approx(0) );
		REQUIRE( (T[a]).trace().imag() == Approx(0) );
		REQUIRE( (T[a]-T[a].adjoint()).norm() == Approx(0) );
		for (int b=0; b<8; ++b) {
			if(a==b) {
				REQUIRE( (T[a]*T[b]).trace().real() == Approx(0.5) );
				REQUIRE( (T[a]*T[b]).trace().imag() == Approx(0) );
			}
			else {
				REQUIRE( (T[a]*T[b]).trace().real() == Approx(0) );				
				REQUIRE( (T[a]*T[b]).trace().imag() == Approx(0) );				
			}
			
		}
	}
}
