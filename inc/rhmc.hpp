#ifndef LATTICE_RHMC_H
#define LATTICE_RHMC_H
#include "su3.hpp"
#include "4d.hpp"
#include "dirac_op.hpp"
#include "inverters.hpp"
#include "rational_approx.hpp"
#include <random>
#include <string>

struct rhmc_params {
	double beta = 5.4;
	double mass = 0.05;
	double mu_I = 0.0;
	int n_f = 2;
	int n_pf = 4;
	double tau = 1.0;
	int n_steps_fermion = 3;
	int n_steps_gauge = 2;
	double lambda_OMF2 = 0.19;
	double MD_eps = 1.e-6;
	double HB_eps = 1.e-14;
	int seed = 123;
	bool EE = false;
	bool block = false;
};

class rhmc {

private:
	rational_approx RA;
	int n_rational; //rational approx A^(1/(2*n_rational))
	field<fermion>::eo_storage_options eo_storage_e = field<fermion>::FULL;
	field<fermion>::eo_storage_options eo_storage_o = field<fermion>::FULL;
	field<block_fermion>::eo_storage_options block_eo_storage_e = field<block_fermion>::FULL;
	field<block_fermion>::eo_storage_options block_eo_storage_o = field<block_fermion>::FULL;

public:
	std::ranlux48 rng;
	rhmc_params params;
	const lattice& grid;
	double deltaE; // dE of proposed change at end of last trajectory
	std::vector< field<fermion> > phi; //pseudofermion field(s)
	field<block_fermion> block_phi; //block pseudofermion field(s)

	explicit rhmc (const rhmc_params& params, const lattice& grid);

	// Does a full RHMC trajectory using parameters in params, returns 1 if update was accepted
	int trajectory (field<gauge>& U, dirac_op& D, bool do_reversibility_test = false, bool MEASURE_FORCE_ERROR_NORMS = false);

	// HMC accept/reject step on gauge field U, returns 1 if proposed change accepted, 0 if rejected
	int accept_reject (field<gauge>& U, const field<gauge>& U_old, double dE);

	// do leapfrog integration of fields (returns # calls of dirac op)
	int leapfrog (field<gauge>& U, field<gauge>& P, dirac_op& D);
	void leapfrog_pure_gauge (field<gauge>& U, field<gauge>& P);

	// 2nd order OMF integrator: 2x more forces per iteration than leapfrog but smaller errors
	int OMF2 (field<gauge>& U, field<gauge>& P, dirac_op& D);
	void OMF2_pure_gauge (field<gauge>& U, field<gauge>& P);

	// total action
	double action (field<gauge>& U, const field<gauge>& P, dirac_op& D);

	// Set gauge links to random values, eps is distance from unity or "roughness"
	// So eps=0 gives all unit gauge links, eps=large is random
	void random_U (field<gauge>& U, double eps);

	// action of gauge field
	double action_U (const field<gauge>& U);

	// action of momenta
	double action_P (const field<gauge>& P);

	// action of pseudofermion field 
	double action_F (field<gauge>& U, dirac_op& D);

	// do a single HMC integration step of length eps for the momenta
	void step_P_pure_gauge (field<gauge>& P, field<gauge> &U, double eps, bool MEASURE_FORCE_NORM = false);

	// do a single HMC integration step of length eps for the momenta (returns # calls of dirac op)
	int step_P_fermion (field<gauge>& P, field<gauge> &U, dirac_op& D, double eps, bool MEASURE_FORCE_NORM = true);

	// do a single HMC integration step of length eps for the gauge links
	void step_U (const field<gauge>& P, field<gauge> &U, double eps);

	// Set momenta to random gaussian
	void gaussian_P (field<gauge>& P);

	// Set pseudofermion field to random gaussian, return squared norm of field 
	//double gaussian_fermion (field<fermion>& chi);
	double gaussian_fermion (std::vector< field<fermion> >& chi);
	
	template<int N>
	double gaussian_fermion (field< block_fermion_matrix<N> >& chi) {
		// normal distribution p(x) ~ exp(-x^2), i.e. mu=0, sigma=1/sqrt(2):
		// NO openMP: rng not threadsafe!
		double norm = 0;
		std::normal_distribution<double> randdist_gaussian (0, 1.0/sqrt(2.0));
		for(int ix=0; ix<chi.V; ++ix) {
			for(int j=0; j<N; ++j) {
				for(int i=0; i<3; ++i) {
					double re = randdist_gaussian (rng);
					double im = randdist_gaussian (rng);
					norm += re*re + im*im;
					chi[ix](i,j) = std::complex<double> (re, im);
				}
			}
		}
		return norm;
	}

	// HMC gauge force: replaces existing force 
	void force_gauge (field<gauge>& force, const field<gauge>& U);

	// HMC fermionic force: adds to existing force (returns # calls of dirac op)
	int force_fermion (field<gauge> &force, field<gauge> &U, dirac_op& D);
	int force_fermion_block (field<gauge> &force, field<gauge> &U, dirac_op& D);

	// debugging
	int force_fermion_norms (field<gauge> &force, field<gauge> &U, dirac_op& D);

	// staple for link U_{\mu}(ix)
	SU3mat staple (int ix, int mu, const field<gauge>& U);

	double plaq (const field<gauge> &U);

	// average of polyakov loops in time (x_0) direction
	std::complex<double> polyakov_loop (const field<gauge> &U);
};

#endif //LATTICE_RHMC_H