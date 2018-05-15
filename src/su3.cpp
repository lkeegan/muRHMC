#include "su3.hpp"

SU3mat exp_ch (const SU3mat& X) {
	SU3mat XX = X*X;
	double t = -0.5 * XX.trace().real();
	std::complex<double> d = X.determinant();
	int N = exp_ch_N;
	std::complex<double> q0 = exp_ch_c_n[N];
	std::complex<double> q1 = 0.0;
	std::complex<double> q2 = 0.0;
	std::complex<double> q0_old, q1_old;
	while(N > 0) {
		q0_old = q0;
		q1_old = q1;
		q0 = exp_ch_c_n[--N] + d * q2;
		q1 = q0_old - t * q2;
		q2 = q1_old;
	}
	return q0*SU3mat::Identity() + q1*X + q2*XX;
}

//In-place F <- (F-F^dag) - Tr(F-F^dag)/3
void project_traceless_antihermitian_part(SU3mat& F) {
	double F00i = 2.0*F(0,0).imag();
	double F11i = 2.0*F(1,1).imag();
	double F22i = 2.0*F(2,2).imag();
	double tr = (1.0/3.0)*(F00i + F11i + F22i);
	F(0,1) -= std::conj(F(1,0));
	F(0,2) -= std::conj(F(2,0));
	F(1,2) -= std::conj(F(2,1));
	F(1,0) = -std::conj(F(0,1));
	F(2,0) = -std::conj(F(0,2));
	F(2,1) = -std::conj(F(1,2));
	F(0,0) = std::complex<double>(0.0, F00i - tr);
	F(1,1) = std::complex<double>(0.0, F11i - tr);
	F(2,2) = std::complex<double>(0.0, F22i - tr);
}
