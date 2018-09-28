#include "hmc.hpp"
#include <chrono>
#include <complex>
#include <iostream>             //FOR DEBUGGING
#include "io.hpp"               //DEBUGGING
#include "rational_approx.hpp"  //FOR square root in EE PSEUDOFERMION HEATBATH

hmc::hmc(const hmc_params& params) : rng(params.seed), params(params) {}

int hmc::trajectory(field<gauge>& U, dirac_op& D,
                    bool MEASURE_FORCE_ERROR_NORMS) {
  auto timer_trajectory_start = std::chrono::high_resolution_clock::now();
  // set Dirac op mass and isospin mu values
  D.mass = params.mass;
  D.mu_I = params.mu_I;
  // make random gaussian momentum field P
  field<gauge> P(U.grid);
  gaussian_P(P);
  // make gaussian fermion field
  field<fermion>::eo_storage_options eo_storage_e = field<fermion>::FULL;
  if (params.EE) {
    // only use even part of pseudofermions and even-even sub-block of dirac op
    eo_storage_e = field<fermion>::EVEN_ONLY;
  }
  field<fermion> chi(U.grid, eo_storage_e);
  gaussian_fermion(chi);
  // construct phi_(e) = D_(eo) chi_(o)
  field<fermion> phi(U.grid, eo_storage_e);
  if (params.EE) {
    // Use rational approx for (m^2 - D_eo D_oe)^{1/2}:
    // double max = D.largest_eigenvalue_bound(U);
    // rational_approx RA(D.mass*D.mass, max);

    // For now just hard code one with the right spectral range:
    // Approx to x^(1/2) in range [4.000000e-06,1.000000e+01] with degree 32 and
    // relative error 4.846799e-16
    std::vector<double> alpha = {
        3.6890641698138090e+01,  -1.0561012211778129e-10,
        -5.0290884785481546e-10, -1.4893043873142169e-09,
        -3.7857803364223508e-09, -9.0366581199976822e-09,
        -2.0947036800274181e-08, -4.7841931736674990e-08,
        -1.0841084668803320e-07, -2.4459660795632532e-07,
        -5.5051064324919543e-07, -1.2373025102634825e-06,
        -2.7786969358748370e-06, -6.2375418778515783e-06,
        -1.3998623100912925e-05, -3.1413674980792612e-05,
        -7.0495943812265722e-05, -1.5822508225685187e-04,
        -3.5524318538524512e-04, -7.9805022196072738e-04,
        -1.7946475593422942e-03, -4.0429235370409832e-03,
        -9.1354811567293404e-03, -2.0750787741912004e-02,
        -4.7559443531606085e-02, -1.1070294282398256e-01,
        -2.6467546517944529e-01, -6.6307169269270883e-01,
        -1.8044042558746902e+00, -5.7095320003236596e+00,
        -2.4204388167882527e+01, -1.9784900580824535e+02,
        -1.6442888944241899e+04};
    std::vector<double> beta = {
        2.9716703733938721e-07, 1.2769764410245246e-06, 3.2305954737123776e-06,
        6.7385760049491410e-06, 1.2843376392974904e-05, 2.3359148745684882e-05,
        4.1410854018296570e-05, 7.2362923650595078e-05, 1.2541346908569054e-04,
        2.1632788476453539e-04, 3.7212447235720212e-04, 6.3910535422472892e-04,
        1.0966217098551470e-03, 1.8806712847496514e-03, 3.2243622356695097e-03,
        5.5273314696792312e-03, 9.4749333138598405e-03, 1.6243167205269585e-02,
        2.7851859697874980e-02, 4.7775821012012366e-02, 8.2009669916180375e-02,
        1.4094451609143652e-01, 2.4273893822882503e-01, 4.1956324324506633e-01,
        7.2973711037915645e-01, 1.2830890807447892e+00, 2.2996743689191512e+00,
        4.2662231184512791e+00, 8.4410397767685961e+00, 1.9010796060417171e+01,
        5.8057123846762863e+01, 5.4823859356097728e+02};
    rational_approx_cg_multishift(phi, chi, U, alpha, beta, D, params.HB_eps);
  } else {
    D.D(phi, chi, U);
  }
  // make copy of current gauge config in case update is rejected
  field<gauge> U_old(U.grid);
  U_old = U;

  double action_old = chi.squaredNorm() + action_U(U) + action_P(P);

  // FORCE MEASUREMENT: measure force norms once and exit:
  if (MEASURE_FORCE_ERROR_NORMS) {
    std::cout.precision(12);
    // get "exact" high precision solve force vector
    auto timer_start = std::chrono::high_resolution_clock::now();
    field<gauge> force_star(U.grid);
    force_star.setZero();
    params.MD_eps = 1.e-12;
    int iter = force_fermion(force_star, U, phi, D);
    force_star *= -1.0;
    double f_star_norm = force_star.norm();
    auto timer_stop = std::chrono::high_resolution_clock::now();
    auto timer_count = std::chrono::duration_cast<std::chrono::milliseconds>(
                           timer_stop - timer_start)
                           .count();
    std::cout << std::scientific << "# correct force iter " << iter << "\t norm"
              << f_star_norm << "\t runtime: " << timer_count << std::endl;

    // get approx force vector, output norm of difference
    field<gauge> force_eps(U.grid);
    std::vector<double> residuals = {1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7};
    std::cout << "# eps, CGiter, CGerror-norm, CGruntime, true-force-norm"
              << std::endl;
    for (int i_eps = 0; i_eps < static_cast<int>(residuals.size()); ++i_eps) {
      params.MD_eps = residuals[i_eps];
      auto timer_start = std::chrono::high_resolution_clock::now();
      force_eps = force_star;
      int CGiter = force_fermion(force_eps, U, phi, D);
      auto timer_stop = std::chrono::high_resolution_clock::now();
      auto CGtimer = std::chrono::duration_cast<std::chrono::milliseconds>(
                         timer_stop - timer_start)
                         .count();
      double CGerr = force_eps.norm();
      std::cout << std::scientific << residuals[i_eps] << "\t" << CGiter << "\t"
                << CGerr << "\t" << CGtimer << "\t" << f_star_norm << std::endl;
    }
    exit(0);
  }
  // DEBUGGING: these should be the same:
  /*
  std::cout.precision(17);
  std::cout << "chidag.chi " << chi.squaredNorm() << std::endl;
  std::cout << "fermion_ac " << action_F(U, phi, D) << std::endl;
  std::cout << "full_ac " << action(U, phi, P, D) << std::endl;
  std::cout << "full_ac " << action_old << std::endl;
  */

  // do integration
  OMF2(U, phi, P, D);
  // calculate change in action
  double action_new = action(U, phi, P, D);
  deltaE = action_new - action_old;

  auto timer_trajectory_stop = std::chrono::high_resolution_clock::now();
  std::cout << "#RHMC_TrajectoryRuntime "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   timer_trajectory_stop - timer_trajectory_start)
                   .count()
            << std::endl;

  if (params.constrained) {
    // this value is stored from the last trajectory
    // log("[HMC] old-sus", suscept);
    double new_suscept = D.pion_susceptibility_exact(U);
    suscept_proposed = new_suscept;
    if (new_suscept > (params.suscept_central + params.suscept_delta)) {
      // suscept too high: reject proposed update, restore old U
      U = U_old;
      // log("[HMC] sus too high - rejected: upper bound:",
      // params.suscept_central + params.suscept_delta);
      return 0;
    } else if (new_suscept < (params.suscept_central - params.suscept_delta)) {
      // suscept too low: reject proposed update, restore old U
      U = U_old;
      // log("[HMC] sus too low - rejected: lower bound:",
      // params.suscept_central - params.suscept_delta);
      return 0;
    } else {
      // susceptibility is in acceptable range: do standard accept/reject step
      // on dE
      if (accept_reject(U, U_old, action_new - action_old)) {
        // if accepted update cached susceptibility and return 1
        suscept = new_suscept;
        return 1;
      } else {
        return 0;
      }
    }
  } else {
    return accept_reject(U, U_old, action_new - action_old);
  }
}

int hmc::trajectory_pure_gauge(field<gauge>& U) {
  // make random gaussian momentum field P
  field<gauge> P(U.grid);
  gaussian_P(P);
  // make copy of current gauge config in case update is rejected
  field<gauge> U_old(U.grid);
  U_old = U;
  double action_old = action_U(U) + action_P(P);
  // do integration
  leapfrog_pure_gauge(U, P);
  // calculate change in action
  double action_new = action_U(U) + action_P(P);
  deltaE = action_new - action_old;
  return accept_reject(U, U_old, action_new - action_old);
}

int hmc::accept_reject(field<gauge>& U, const field<gauge>& U_old, double dE) {
  std::uniform_real_distribution<double> randdist_double_0_1(0, 1);
  double r = randdist_double_0_1(rng);
  if (r > exp(-dE)) {
    // reject proposed update, restore old U
    U = U_old;
    return 0;
  }
  // otherwise accept proposed update, i.e. do nothing
  return 1;
}

void hmc::leapfrog_pure_gauge(field<gauge>& U, field<gauge>& P) {
  double eps = params.tau / static_cast<double>(params.n_steps_fermion *
                                                params.n_steps_gauge);
  // leapfrog integration:
  step_P_pure_gauge(P, U, 0.5 * eps);
  for (int i = 0; i < params.n_steps_gauge - 1; ++i) {
    step_U(P, U, eps);
    step_P_pure_gauge(P, U, eps);
  }
  step_U(P, U, eps);
  step_P_pure_gauge(P, U, 0.5 * eps);
}

int hmc::leapfrog(field<gauge>& U, field<fermion>& phi, field<gauge>& P,
                  dirac_op& D) {
  double eps = params.tau / static_cast<double>(params.n_steps_fermion);
  int iter = 0;
  // leapfrog integration:
  iter += step_P_fermion(P, U, phi, D, 0.5 * eps);
  for (int i = 0; i < params.n_steps_fermion - 1; ++i) {
    // step_U(P, U, eps);
    leapfrog_pure_gauge(U, P);
    iter += step_P_fermion(P, U, phi, D, eps);
  }
  // step_U(P, U, eps);
  leapfrog_pure_gauge(U, P);
  iter += step_P_fermion(P, U, phi, D, 0.5 * eps);
  return iter;
}

void hmc::OMF2_pure_gauge(field<gauge>& U, field<gauge>& P) {
  double eps =
      0.5 * params.tau /
      static_cast<double>(params.n_steps_fermion * params.n_steps_gauge);
  // OMF2 integration:
  step_P_pure_gauge(P, U, (params.lambda_OMF2) * eps, true);
  for (int i = 0; i < params.n_steps_gauge - 1; ++i) {
    step_U(P, U, 0.5 * eps);
    step_P_pure_gauge(P, U, (1.0 - 2.0 * params.lambda_OMF2) * eps);
    step_U(P, U, 0.5 * eps);
    step_P_pure_gauge(P, U, (2.0 * params.lambda_OMF2) * eps);
  }
  step_U(P, U, 0.5 * eps);
  step_P_pure_gauge(P, U, (1.0 - 2.0 * params.lambda_OMF2) * eps);
  step_U(P, U, 0.5 * eps);
  step_P_pure_gauge(P, U, (params.lambda_OMF2) * eps);
}

int hmc::OMF2(field<gauge>& U, field<fermion>& phi, field<gauge>& P,
              dirac_op& D) {
  double eps = params.tau / static_cast<double>(params.n_steps_fermion);
  std::cout << params.lambda_OMF2 << std::endl;
  int iter = 0;
  // OMF2 integration:
  iter += step_P_fermion(P, U, phi, D, (params.lambda_OMF2) * eps);
  for (int i = 0; i < params.n_steps_fermion - 1; ++i) {
    // step_U(P, U, 0.5*eps);
    OMF2_pure_gauge(U, P);
    iter +=
        step_P_fermion(P, U, phi, D, (1.0 - 2.0 * params.lambda_OMF2) * eps);
    // step_U(P, U, 0.5*eps);
    OMF2_pure_gauge(U, P);
    iter += step_P_fermion(P, U, phi, D, (2.0 * params.lambda_OMF2) * eps);
  }
  // step_U(P, U, 0.5*eps);
  OMF2_pure_gauge(U, P);
  iter += step_P_fermion(P, U, phi, D, (1.0 - 2.0 * params.lambda_OMF2) * eps);
  // step_U(P, U, 0.5*eps);
  OMF2_pure_gauge(U, P);
  iter += step_P_fermion(P, U, phi, D, (params.lambda_OMF2) * eps);
  return iter;
}

double hmc::action(field<gauge>& U, const field<fermion>& phi,
                   const field<gauge>& P, dirac_op& D) {
  return action_U(U) + action_P(P) + action_F(U, phi, D);
}

double hmc::action_U(const field<gauge>& U) {
  return -params.beta * 6.0 * U.V * plaq(U);
}

double hmc::action_P(const field<gauge>& P) {
  double ac = 0.0;
  ////#pragma omp parallel for reduction (+:ac)
  for (int ix = 0; ix < P.V; ++ix) {
    for (int mu = 0; mu < 4; ++mu) {
      ac += (P[ix][mu] * P[ix][mu]).trace().real();
    }
  }
  return ac;
}

double hmc::action_F(field<gauge>& U, const field<fermion>& phi, dirac_op& D) {
  field<fermion> D2inv_phi(phi.grid, phi.eo_storage);
  int iter = cg(D2inv_phi, phi, U, D, params.HB_eps);
  // std::cout << "Action CG iterations: " << iter << std::endl;
  return phi.dot(D2inv_phi).real();
}

void hmc::step_P_pure_gauge(field<gauge>& P, field<gauge>& U, double eps,
                            bool MEASURE_FORCE_NORM) {
  std::complex<double> ibeta_12(0.0, -eps * params.beta / 12.0);
  double force_norm = 0;
  if (MEASURE_FORCE_NORM) {
    for (int ix = 0; ix < U.V; ++ix) {
      for (int mu = 0; mu < 4; ++mu) {
        SU3mat A = staple(ix, mu, U);
        SU3mat F = U[ix][mu] * A;
        project_traceless_antihermitian_part(F);
        F *= ibeta_12;
        // A = F - F.adjoint();
        // F = ( A - (A.trace()/3.0)*SU3mat::Identity() ) * ibeta_12;
        force_norm += F.squaredNorm();
        P[ix][mu] -= F;
      }
    }
    force_norm /= eps * eps;
    std::cout << "gauge_force_norm: "
              << sqrt(force_norm / static_cast<double>(4 * U.V)) << std::endl;
  } else {
    ////#pragma omp parallel for
    for (int ix = 0; ix < U.V; ++ix) {
      for (int mu = 0; mu < 4; ++mu) {
        SU3mat A = staple(ix, mu, U);
        SU3mat F = U[ix][mu] * A;
        project_traceless_antihermitian_part(F);
        P[ix][mu] -= F * ibeta_12;
      }
    }
  }
}

int hmc::step_P_fermion(field<gauge>& P, field<gauge>& U,
                        const field<fermion>& phi, dirac_op& D, double eps,
                        bool MEASURE_FORCE_NORM) {
  field<gauge> force(U.grid);
  force.setZero();
  // force_gauge (force, U);
  int iter = force_fermion(force, U, phi, D);
  if (MEASURE_FORCE_NORM) {
    std::cout << "fermion_force_norm: "
              << sqrt(force.squaredNorm() / static_cast<double>(4 * U.V))
              << std::endl;
  }
  ////#pragma omp parallel for
  for (int ix = 0; ix < U.V; ++ix) {
    for (int mu = 0; mu < 4; ++mu) {
      P[ix][mu] -= eps * force[ix][mu];
    }
  }
  return iter;
}

void hmc::step_U(const field<gauge>& P, field<gauge>& U, double eps) {
  ////#pragma omp parallel for
  for (int ix = 0; ix < U.V; ++ix) {
    for (int mu = 0; mu < 4; ++mu) {
      // U[ix][mu] = ((std::complex<double> (0.0, eps) * P[ix][mu]).exp()) *
      // U[ix][mu];
      U[ix][mu] =
          exp_ch((std::complex<double>(0.0, eps) * P[ix][mu])) * U[ix][mu];
    }
  }
}

void hmc::random_U(field<gauge>& U, double eps) {
  // NO openMP: rng not threadsafe!
  field<gauge> P(U.grid);
  gaussian_P(P);
  if (eps > 1.0) {
    eps = 1.0;
  }
  for (int ix = 0; ix < U.V; ++ix) {
    for (int mu = 0; mu < 4; ++mu) {
      U[ix][mu] = exp_ch((std::complex<double>(0.0, eps) * P[ix][mu]));
    }
  }
}

void hmc::gaussian_fermion(field<fermion>& chi) {
  // normal distribution p(x) ~ exp(-x^2), i.e. mu=0, sigma=1/sqrt(2):
  // NO openMP: rng not threadsafe!
  std::normal_distribution<double> randdist_gaussian(0, 1.0 / sqrt(2.0));
  for (int ix = 0; ix < chi.V; ++ix) {
    for (int i = 0; i < 3; ++i) {
      std::complex<double> r(randdist_gaussian(rng), randdist_gaussian(rng));
      chi[ix](i) = r;
    }
  }
}

void hmc::gaussian_P(field<gauge>& P) {
  // normal distribution p(x) ~ exp(-x^2/2), i.e. mu=0, sigma=1:
  // could replace sum of c*T with single function assigning c
  // to the appropriate matrix elements if this needs to be faster
  // NO openMP: rng not threadsafe!
  std::normal_distribution<double> randdist_gaussian(0, 1.0);
  SU3_Generators T;
  for (int ix = 0; ix < P.V; ++ix) {
    for (int mu = 0; mu < 4; ++mu) {
      P[ix][mu].setZero();
      for (int i = 0; i < 8; ++i) {
        std::complex<double> c(randdist_gaussian(rng), 0.0);
        P[ix][mu] += c * T[i];
      }
    }
  }
}

void hmc::force_gauge(field<gauge>& force, const field<gauge>& U) {
  ////#pragma omp parallel for
  for (int ix = 0; ix < U.V; ++ix) {
    for (int mu = 0; mu < 4; ++mu) {
      SU3mat A = staple(ix, mu, U);
      SU3mat F = U[ix][mu] * A;
      std::complex<double> ibeta_12(0.0, -params.beta / 12.0);
      A = F - F.adjoint();
      force[ix][mu] = (A - (A.trace() / 3.0) * SU3mat::Identity()) * ibeta_12;
    }
  }
}

void hmc::stout_smear(double rho, field<gauge>& U) {
  // construct Q: arxiv:0311018 eq (2)
  // with rho_munu = rho = constant real
  field<gauge> Q(U.grid);
  ////#pragma omp parallel for
  for (int ix = 0; ix < U.V; ++ix) {
    for (int mu = 0; mu < 4; ++mu) {
      SU3mat A = staple(ix, mu, U);
      SU3mat F = rho * U[ix][mu] * A;
      A = F - F.adjoint();
      Q[ix][mu] = (A - (A.trace() / 3.0) * SU3mat::Identity()) *
                  std::complex<double>(0.0, 0.5);
    }
  }
  // arxiv:0311018 eq (3)
  // U <- exp(i Q) U
  step_U(Q, U, 1.0);
}

int hmc::force_fermion(field<gauge>& force, field<gauge>& U,
                       const field<fermion>& phi, dirac_op& D) {
  // chi = (D(mu)D^dagger(mu))^-1 phi
  field<fermion> chi(phi.grid, phi.eo_storage);
  double fermion_force_norm = 0.0;
  int iter = cg(chi, phi, U, D, params.MD_eps);
  // psi = -D^dagger(mu,m) chi = D(-mu, -m) chi
  field<fermion>::eo_storage_options eo_storage = field<fermion>::FULL;
  if (params.EE) {
    eo_storage = field<fermion>::ODD_ONLY;
  }
  field<fermion> psi(phi.grid, eo_storage);
  if (params.EE) {
    D.apply_eta_bcs_to_U(U);
    D.D_oe(psi, chi, U);
    D.remove_eta_bcs_from_U(U);
  } else {
    // apply -Ddagger = D(-m, -mu)
    // NB: should put this in a function
    D.mass = -D.mass;
    D.mu_I = -D.mu_I;
    D.D(psi, chi, U);
    D.mass = -D.mass;
    D.mu_I = -D.mu_I;
  }

  D.apply_eta_bcs_to_U(U);
  SU3_Generators T;
  if (params.EE) {
    // for even-even version half of the terms are zero:
    // even ix: ix = ix_e
    ////#pragma omp parallel for
    for (int ix = 0; ix < chi.V; ++ix) {
      for (int mu = 0; mu < 4; ++mu) {
        for (int a = 0; a < 8; ++a) {
          double Fa = (chi[ix].dot(T[a] * U[ix][mu] * psi.up(ix, mu))).imag();
          fermion_force_norm += 0.5 * Fa * Fa;
          force[ix][mu] -= Fa * T[a];
        }
      }
    }
    // odd ix: ix = ix_o + chi.V
    ////#pragma omp parallel for
    for (int ix_o = 0; ix_o < chi.V; ++ix_o) {
      int ix = ix_o + chi.V;
      for (int mu = 0; mu < 4; ++mu) {
        for (int a = 0; a < 8; ++a) {
          double Fa =
              (chi.up(ix, mu).dot(U[ix][mu].adjoint() * T[a] * psi[ix_o]))
                  .imag();
          fermion_force_norm += 0.5 * Fa * Fa;
          force[ix][mu] -= Fa * T[a];
        }
      }
    }
  } else {
    // mu=0 terms have extra chemical potential isospin factors exp(+-\mu_I/2):
    double mu_I_plus_factor = exp(0.5 * params.mu_I);
    double mu_I_minus_factor = exp(-0.5 * params.mu_I);
    ////#pragma omp parallel for
    for (int ix = 0; ix < U.V; ++ix) {
      for (int a = 0; a < 8; ++a) {
        double Fa = chi[ix]
                        .dot(T[a] * mu_I_plus_factor * U[ix][0] * psi.up(ix, 0))
                        .imag();
        Fa += chi.up(ix, 0)
                  .dot(U[ix][0].adjoint() * mu_I_minus_factor * T[a] * psi[ix])
                  .imag();
        fermion_force_norm += 0.5 * Fa * Fa;
        force[ix][0] -= Fa * T[a];
      }
      for (int mu = 1; mu < 4; ++mu) {
        for (int a = 0; a < 8; ++a) {
          double Fa = (chi[ix].dot(T[a] * U[ix][mu] * psi.up(ix, mu))).imag();
          Fa +=
              (chi.up(ix, mu).dot(U[ix][mu].adjoint() * T[a] * psi[ix])).imag();
          fermion_force_norm += 0.5 * Fa * Fa;
          force[ix][mu] -= Fa * T[a];
        }
      }
    }
  }
  D.remove_eta_bcs_from_U(U);
  // std::cout << "#FFnorms " <<
  // sqrt(fermion_force_norm/static_cast<double>(4*U.V)) << std::endl;
  std::cout << "#HMC_ForceCGIter " << iter << std::endl;

  return iter;
}

SU3mat hmc::staple(int ix, int mu, const field<gauge>& U) {
  SU3mat A = SU3mat::Zero();
  int ix_plus_mu = U.iup(ix, mu);
  for (int nu = 0; nu < 4; ++nu) {
    if (nu != mu) {
      A += U[ix_plus_mu][nu] * U.up(ix, nu)[mu].adjoint() * U[ix][nu].adjoint();
      A += U.dn(ix_plus_mu, nu)[nu].adjoint() * U.dn(ix, nu)[mu].adjoint() *
           U.dn(ix, nu)[nu];
    }
  }
  return A;
}

double hmc::plaq(int ix, int mu, int nu, const field<gauge>& U) {
  return ((U[ix][mu] * U.up(ix, mu)[nu]) *
          ((U[ix][nu] * U.up(ix, nu)[mu]).adjoint()))
      .trace()
      .real();
}

double hmc::plaq(int ix, const field<gauge>& U) {
  double p = 0;
  for (int mu = 1; mu < 4; ++mu) {
    for (int nu = 0; nu < mu; nu++) {
      p += plaq(ix, mu, nu, U);
    }
  }
  return p;
}

double hmc::plaq(const field<gauge>& U) {
  double p = 0;
  ////#pragma omp parallel for reduction (+:p)
  for (int ix = 0; ix < U.V; ++ix) {
    for (int mu = 1; mu < 4; ++mu) {
      for (int nu = 0; nu < mu; nu++) {
        p += plaq(ix, mu, nu, U);
      }
    }
  }
  return p / static_cast<double>(3 * 6 * U.V);
}

double hmc::plaq_1x2(const field<gauge>& U) {
  double p = 0;
  ////#pragma omp parallel for reduction (+:p)
  for (int ix = 0; ix < U.V; ++ix) {
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; nu++) {
        if (mu != nu) {
          int ix_mu = U.iup(ix, mu);
          int ix_nu = U.iup(ix, nu);
          int ix_mu_mu = U.iup(ix_mu, mu);
          int ix_nu_mu = U.iup(ix_nu, mu);
          p += ((U[ix][mu] * U[ix_mu][mu] * U[ix_mu_mu][nu]) *
                ((U[ix][nu] * U[ix_nu][mu] * U[ix_nu_mu][mu]).adjoint()))
                   .trace()
                   .real();
        }
      }
    }
  }
  return p / static_cast<double>(3 * 12 * U.V);
}

double hmc::plaq_spatial(const field<gauge>& U) {
  double p = 0;
  ////#pragma omp parallel for reduction (+:p)
  for (int ix = 0; ix < U.V; ++ix) {
    for (int mu = 1; mu < 4; ++mu) {
      for (int nu = 1; nu < mu; nu++) {
        p += plaq(ix, mu, nu, U);
      }
    }
  }
  return p / static_cast<double>(3 * 3 * U.V);
}

double hmc::plaq_timelike(const field<gauge>& U) {
  double p = 0;
  ////#pragma omp parallel for reduction (+:p)
  for (int ix = 0; ix < U.V; ++ix) {
    for (int mu = 1; mu < 4; ++mu) {
      p += plaq(ix, mu, 0, U);
    }
  }
  return p / static_cast<double>(3 * 3 * U.V);
}

/*
std::complex<double> hmc::polyakov_loop (const field<gauge> &U) {
        std::complex<double> p = 0;
        // NB: no openmp reduction for std::complex, would have to split into
real and imag parts for(int ix3=0; ix3<U.VOL3; ++ix3) { int ix = U.it_ix(0,
ix3); SU3mat P = U[ix][0]; for(int x0=1; x0<U.L0; x0++) { ix = U.iup(ix, 0); P
*= U[ix][0];
                }
                p += P.trace();
        }
        return p / static_cast<double>(3 * U.VOL3);
}
*/

std::complex<double> hmc::polyakov_loop(const field<gauge>& U, int mu) {
  std::complex<double> p = 0;
  int L[4] = {U.L0, U.L1, U.L2, U.L3};
  // NB: no openmp reduction for std::complex, would have to split into real and
  // imag parts
  for (int ix = 0; ix < U.V; ++ix) {
    SU3mat P = U[ix][mu];
    int ixmu = ix;
    for (int n_mu = 1; n_mu < L[mu]; n_mu++) {
      ixmu = U.iup(ixmu, mu);
      P *= U[ixmu][mu];
    }
    p += P.trace();
  }
  return p / static_cast<double>(3 * U.V);
}

double hmc::topological_charge(const field<gauge>& U) {
  double q = 0;
  for (int ix = 0; ix < U.V; ++ix) {
    for (int sig = 3; sig < 4; ++sig) {
      for (int rho = 2; rho < sig; ++rho) {
        for (int nu = 1; nu < rho; ++nu) {
          for (int mu = 0; mu < nu; mu++) {
            SU3mat Pmunu = (U[ix][mu] * U.up(ix, mu)[nu]) *
                           ((U[ix][nu] * U.up(ix, nu)[mu]).adjoint());
            SU3mat Prhosig = (U[ix][rho] * U.up(ix, rho)[sig]) *
                             ((U[ix][sig] * U.up(ix, sig)[rho]).adjoint());
            q += (Pmunu.imag() * Prhosig.imag()).trace();
          }
        }
      }
    }
  }
  return q / static_cast<double>(3 * U.V);
}

double hmc::chiral_condensate(field<gauge>& U, dirac_op& D) {
  // inverter precision
  double eps = 1.e-12;
  // phi = gaussian noise vector
  field<fermion> phi(U.grid);
  gaussian_fermion(phi);
  // chi = [D(mu,m)D(mu,m)^dag]-1 phi
  field<fermion> chi(U.grid);
  cg(chi, phi, U, D, eps);
  // psi = D(mu,m)^dag chi = - D(-mu,-m) chi = D(mu)^-1 phi
  field<fermion> psi(U.grid);
  D.mass = -D.mass;
  D.mu_I = -D.mu_I;
  D.D(psi, chi, U);
  D.mass = -D.mass;
  D.mu_I = -D.mu_I;
  return -phi.dot(psi).real() / static_cast<double>(phi.V);
}
