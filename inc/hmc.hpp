#ifndef LATTICE_HMC_H
#define LATTICE_HMC_H
#include "su3.hpp"
#include "4d.hpp"
#include "dirac_op.hpp"
#include <random>
#include <unsupported/Eigen/MatrixFunctions>

struct hmc_params {
	double beta;
	double mass;
	double tau;
	int n_steps;
	double MD_eps;
};

class hmc {

private:

public:

	std::ranlux48 rng;
	hmc_params par;
	double deltaE; //for debugging: deltaE from accept/reject step of last trajectory
	explicit hmc (unsigned int seed);

	// Does a full HMC trajectory using supplied parameters, returns 1 if update was accepted
	int trajectory (field<gauge>& U, hmc_params par, const dirac_op& D);

	// HMC accept/reject step on gauge field U, returns 1 if proposed change accepted, 0 if rejected
	int accept_reject (field<gauge>& U, const field<gauge>& U_old, double dE);

	// do leapfrog integration of fields (returns # calls of dirac op)
	int leapfrog (field<gauge>& U, field<fermion>& phi, field<gauge>& P, double tau, int n_steps, double beta, double m, const dirac_op& D);

	// total action
	double action (const field<gauge>& U, const field<fermion>& phi, const field<gauge>& P, double beta, double m, const dirac_op& D);

	// action of gauge field
	double action_U (const field<gauge>& U, double beta);

	// action of momenta
	double action_P (const field<gauge>& P);

	// action of pseudofermion field 
	double action_F (const field<gauge>& U, const field<fermion>& phi, double m, const dirac_op& D);

	// do a single HMC integration step of length eps for the momenta (returns # calls of dirac op)
	int step_P (field<gauge>& P, const field<gauge> &U, const field<fermion>& phi, double beta, double m, const dirac_op& D, double eps);

	// do a single HMC integration step of length eps for the gauge links
	void step_U (const field<gauge>& P, field<gauge> &U, double eps);

	// Set gauge links to random values, eps is distance from unity or "roughness"
	// So eps=0 gives all unit gauge links, eps=large is random
	void random_U (field<gauge>& U, double eps);

	// Set momenta to random gaussian
	void gaussian_P (field<gauge>& P);

	// Set pseudofermion field to random gaussian 
	void gaussian_fermion (field<fermion>& chi);

	// HMC gauge force: replaces existing force 
	void force_gauge (field<gauge>& force, const field<gauge>& U, double beta);

	// HMC fermionic force: adds to existing force (returns # calls of dirac op)
	int force_fermion (field<gauge> &force, const field<gauge> &U, const field<fermion>& phi, double m, const dirac_op& D);

	// staple for link U_{\mu}(ix) 
	SU3mat staple (int ix, int mu, const field<gauge>& U);

	// local plaquette Re Tr { P_{\mu\nu}(i) }
	double plaq (int i, int mu, int nu, const field<gauge>& U);

	// average of plaquette normalised to 1
	double plaq (const field<gauge>& U);

};

#endif //LATTICE_HMC_H