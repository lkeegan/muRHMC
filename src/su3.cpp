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
