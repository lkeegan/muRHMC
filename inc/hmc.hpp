#ifndef LKEEGAN_MURHMC_HMC_H
#define LKEEGAN_MURHMC_HMC_H
#include <random>
#include <string>
#include "4d.hpp"
#include "dirac_op.hpp"
#include "inverters.hpp"
#include "su3.hpp"

struct hmc_params {
  double beta = 5.4;
  double mass = 0.05;
  double mu_I = 0.0;
  double tau = 0.5;
  int n_steps_fermion = 1;
  int n_steps_gauge = 1;
  double lambda_OMF2 = 0.19;
  double MD_eps = 1.e-6;
  double HB_eps = 1.e-14;
  int seed = 123;
  bool EE = false;
  bool constrained = false;
  double suscept_central = 2.0;
  double suscept_delta = 0.5;
};

class hmc {
 private:
 public:
  std::ranlux48 rng;
  hmc_params params;
  double deltaE;            // dE of proposed change at end of last trajectory
  double suscept_proposed;  // proposed pion susceptibility from last trajectory
  double suscept;           // pion susceptibility after last trajectory

  explicit hmc(const hmc_params& params);

  // Does a full HMC trajectory using parameters in params, returns 1 if update
  // was accepted
  int trajectory(field<gauge>& U, dirac_op& D,
                 bool MEASURE_FORCE_ERROR_NORMS = false);

  // Does a full (pure gauge) HMC trajectory using parameters in params, returns
  // 1 if update was accepted
  int trajectory_pure_gauge(field<gauge>& U);

  // HMC accept/reject step on gauge field U, returns 1 if proposed change
  // accepted, 0 if rejected
  int accept_reject(field<gauge>& U, const field<gauge>& U_old, double dE);

  // do leapfrog integration of fields (returns # calls of dirac op)
  int leapfrog(field<gauge>& U, field<fermion>& phi, field<gauge>& P,
               dirac_op& D);
  void leapfrog_pure_gauge(field<gauge>& U, field<gauge>& P);

  // 2nd order OMF integrator: 2x more forces per iteration than leapfrog but
  // smaller errors
  int OMF2(field<gauge>& U, field<fermion>& phi, field<gauge>& P, dirac_op& D);
  void OMF2_pure_gauge(field<gauge>& U, field<gauge>& P);

  // total action
  double action(field<gauge>& U, const field<fermion>& phi,
                const field<gauge>& P, dirac_op& D);

  // action of gauge field
  double action_U(const field<gauge>& U);

  // action of momenta
  double action_P(const field<gauge>& P);

  // action of pseudofermion field
  double action_F(field<gauge>& U, const field<fermion>& phi, dirac_op& D);

  // do a single HMC integration step of length eps for the momenta
  void step_P_pure_gauge(field<gauge>& P, field<gauge>& U, double eps,
                         bool MEASURE_FORCE_NORM = false);

  // do a single HMC integration step of length eps for the momenta (returns #
  // calls of dirac op)
  int step_P_fermion(field<gauge>& P, field<gauge>& U,
                     const field<fermion>& phi, dirac_op& D, double eps,
                     bool MEASURE_FORCE_NORM = true);

  // do a single HMC integration step of length eps for the gauge links
  void step_U(const field<gauge>& P, field<gauge>& U, double eps);

  // Set gauge links to random values, eps is distance from unity or "roughness"
  // So eps=0 gives all unit gauge links, eps=large is random
  void random_U(field<gauge>& U, double eps);

  // Set momenta to random gaussian
  void gaussian_P(field<gauge>& P);

  // Set pseudofermion field to random gaussian
  void gaussian_fermion(field<fermion>& chi);

  // HMC gauge force: replaces existing force
  void force_gauge(field<gauge>& force, const field<gauge>& U);

  // HMC fermionic force: adds to existing force (returns # calls of dirac op)
  int force_fermion(field<gauge>& force, field<gauge>& U,
                    const field<fermion>& phi, dirac_op& D);

  // staple for link U_{\mu}(ix)
  SU3mat staple(int ix, int mu, const field<gauge>& U);

  // Stout smearing as defined in arxiv:0311018 eqs (1-3) with rho_munu = rho =
  // constant real
  void stout_smear(double rho, field<gauge>& U);

  // local plaquette Re Tr { P_{\mu\nu}(i) }: range [0, 3]
  double plaq(int i, int mu, int nu, const field<gauge>& U);

  // local plaquette summed over \sum_{\mu<\nu} Re Tr { P_{\mu\nu}(i) } : range
  // [0, 18]
  double plaq(int ix, const field<gauge>& U);

  // average of plaquette over N_c, mu,nu and volume: range [0, 1]
  double plaq(const field<gauge>& U);

  // average of 1x2 plaquette over N_c, mu,nu and volume: range [0, 1]
  double plaq_1x2(const field<gauge>& U);

  // average of spatial plaquette over N_c, mu,nu and volume: range [0, 1]
  double plaq_spatial(const field<gauge>& U);

  // average of timelike plaquette over N_c, mu,nu and volume: range [0, 1]
  double plaq_timelike(const field<gauge>& U);

  // average of polyakov loops in mu direction
  std::complex<double> polyakov_loop(const field<gauge>& U, int mu = 0);

  // topological charge
  double topological_charge(const field<gauge>& U);

  double topological_charge(const field<gauge>& U, int mu, int nu, int rho,
                            int sig);

  // returns estimate of chiral_condensate using 1 random gaussian fermion
  // vector
  double chiral_condensate(field<gauge>& U, dirac_op& D);
};

#endif  // LKEEGAN_MURHMC_HMC_H