#ifndef LATTICE_RATIONAL_APPROX_H
#define LATTICE_RATIONAL_APPROX_H
#include <vector>

class rational_approx {
private:
	void initialise_rational_approx_LARGE();
	void initialise_rational_approx_SMALL();
	int N_MAX_SHIFTS = 128;

public:
	std::vector< std::vector<double> > alpha_hi, beta_hi;
	std::vector< std::vector<double> > alpha_inv_hi, beta_inv_hi;
	std::vector< std::vector<double> > alpha_lo, beta_lo;
	std::vector< std::vector<double> > alpha_inv_lo, beta_inv_lo;
	
	rational_approx(double lower_bound, double upper_bound);
};

#endif //LATTICE_RATIONAL_APPROX_H