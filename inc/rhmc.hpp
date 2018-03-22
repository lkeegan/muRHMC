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
	double beta;
	double mass;
	double mu_I;
	int n_f;
	int n_pf;
	double tau;
	int n_steps_fermion;
	int n_steps_gauge;
	double MD_eps;
	int seed;
	bool EE;
};

class rhmc {

private:
	double lambda = 1.0/6.0; //tunable parameter for OMF2 integrator
	rational_approx RA;
	int n_rational; //rational approx A^(1/(2*n_rational))
	field<fermion>::eo_storage_options eo_storage_e = field<fermion>::FULL;
	field<fermion>::eo_storage_options eo_storage_o = field<fermion>::FULL;

public:

	std::ranlux48 rng;
	rhmc_params params;
	double deltaE; // dE of proposed change at end of last trajectory
	std::vector< field<fermion> > phi; //pseudofermion field(s)

	explicit rhmc (const rhmc_params& params);

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

	// Set pseudofermion field to random gaussian 
	void gaussian_fermion (field<fermion>& chi);

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