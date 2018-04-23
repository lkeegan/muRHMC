#include "catch.hpp"
#include "su3.hpp"
#include <unsupported/Eigen/MatrixFunctions> //for matrix exponential
#include <iostream>
#include <limits>

constexpr double EPS = 5.e-14;

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

TEST_CASE( "anti-hermitian traceless projector", "[su3]" ) {
	SU3mat X = SU3mat::Random();
	project_traceless_antihermitian_part(X);

	REQUIRE( std::abs(X.trace()) < EPS );
	REQUIRE( (X + X.adjoint()).squaredNorm() < EPS );
}

TEST_CASE( "Cayley-Hamilton form of exp(X)", "[su3]" ) {
	// Make random traceless anti-hermitian complex 3x3 matrix X
	SU3mat X = SU3mat::Random();
	project_traceless_antihermitian_part(X);
	// X <- i eps * X
	X *= std::complex<double>(0.0, 0.1);
	INFO( "eigen exp(X)" << X.exp() );

	// Choose N = max value of n such that 1/(N-1)! << ULP
	// c_n = 1/n!
	std::vector<double> c_n(2, 1.0);
	int N = 1;
	double c_N = 1;
	while(c_n[N-1] > 1e-4*std::numeric_limits<double>::epsilon()) {
		c_N /= static_cast<double>(++N);
		c_n.push_back(c_N);
	}
	CAPTURE (std::numeric_limits<double>::epsilon());
	CAPTURE (N);

	SU3mat XX = X*X;
	double t = -0.5 * XX.trace().real();
	std::complex<double> d = X.determinant();
	CAPTURE (t);
	CAPTURE (d);

	std::complex<double> q0 = c_n[N];
	std::complex<double> q1 = 0.0;
	std::complex<double> q2 = 0.0;
	std::complex<double> q0_old, q1_old;
	while(N > 0) {
		q0_old = q0;
		q1_old = q1;
		q0 = c_n[--N] + d * q2;
		q1 = q0_old - t * q2;
		q2 = q1_old;
	}
	SU3mat expX_ch = q0*SU3mat::Identity() + q1*X + q2*XX;

	INFO( "eigen exp(X)" << expX_ch );

	INFO( "|| exp_ch(X) - X.exp()[eigen] || = " << (expX_ch - X.exp()).norm() );

	REQUIRE( (X.exp() - exp_ch(X)).norm() < EPS );

	REQUIRE( (expX_ch - X.exp()).norm() < EPS );
}