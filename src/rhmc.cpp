#include "rhmc.hpp"
#include "rational_approx.hpp"
#include <complex>
#include <iostream> //FOR DEBUGGING
#include "io.hpp" //DEBUGGING

rhmc::rhmc (const rhmc_params& params) : rng(params.seed), params(params) {
}

int rhmc::trajectory (field<gauge>& U, dirac_op& D, bool do_reversibility_test) {
	// set Dirac op mass and isospin mu values
	D.mass = params.mass;
	D.mu_I = params.mu_I;
	// make random gaussian momentum field P
	field<gauge> P (U.grid);
	gaussian_P (P);
	// make gaussian fermion field
	field<fermion>::eo_storage_options eo_storage_e = field<fermion>::FULL;
	field<fermion>::eo_storage_options eo_storage_o = field<fermion>::FULL;
	// rational approx for initial phi: M^{n_f/(16 params.n_pf)} = M^{1/(2*n_rational)}
	n_rational = 8 * params.n_pf / params.n_f;
	if(params.EE) {
		// only use even part of pseudofermions and even-even sub-block of dirac op
		eo_storage_e = field<fermion>::EVEN_ONLY;
		eo_storage_o = field<fermion>::ODD_ONLY;
		// rational approx for initial phi: M^{n_f/(8 params.n_pf)} = M^{1/(2*n_rational)}
		n_rational = 4 * params.n_pf / params.n_f;
	}
	//std::cout << "#RHMC: n_f=" << params.n_f << ", n_pf=" << params.n_pf << ", n_rational=" << n_rational << ", n_shifts hi/lo=" << params.RA.beta_hi[n_rational].size() << "/" << params.RA.beta_lo[n_rational].size() << std::endl;
	field<fermion> chi_e (U.grid, eo_storage_e);
	phi.clear();
	for(int i_pf=0; i_pf<params.n_pf; ++i_pf) {
		phi.push_back(chi_e); //TODO: don't do this everytime, just intialise once
		gaussian_fermion (chi_e);
		// construct phi[i] = M^{1/(2*n_rational) chi_e
		int iter = rational_approx_cg_multishift(phi[i_pf], chi_e, U, params.RA.alpha_hi[n_rational], params.RA.beta_hi[n_rational], D, 1.e-15);
	}
	// make copy of current gauge config in case update is rejected
	field<gauge> U_old (U.grid);
	U_old = U;

	double action_old = action(U, P, D); //NB check that action_F given by chi.dot(chi), and avoid recalculating here...
	// do integration
	OMF2 (U, P, D);

	// DEBUGGING reversibility test: P->-P, repeat integration, no accept/reject step
	// U should equal to inital U to ~1e-15
	if(do_reversibility_test) {
		P *= -1;
		OMF2 (U, P, D);
		return 1;
	}

	// calculate change in action
	double action_new = action(U, P, D);
	deltaE = action_new - action_old;

	return accept_reject (U, U_old, action_new - action_old);
}

int rhmc::accept_reject (field<gauge>& U, const field<gauge>& U_old, double dE) {
	std::uniform_real_distribution<double> randdist_double_0_1 (0,1);
	double r = randdist_double_0_1 (rng);
	if (r > exp(-dE)) {
		// reject proposed update, restore old U
		U = U_old;
		return 0;
	}
	// otherwise accept proposed update, i.e. do nothing
	return 1;
}

int rhmc::leapfrog (field<gauge>& U, field<gauge>& P, dirac_op& D) {
	double eps = params.tau / static_cast<double>(params.n_steps);
	int iter = 0;
	// leapfrog integration:
	iter += step_P(P, U, D, 0.5*eps);
	for(int i=0; i<params.n_steps-1; ++i) {
		step_U(P, U, eps);
		iter += step_P(P, U, D, eps);
	}
	step_U(P, U, eps);
	iter += step_P(P, U, D, 0.5*eps);
	return iter;
}

int rhmc::OMF2 (field<gauge>& U, field<gauge>& P, dirac_op& D) {
	double constexpr lambda = 0.19318; //tunable parameter
	double eps = params.tau / static_cast<double>(params.n_steps);
	int iter = 0;
	// OMF2 integration:
	iter += step_P(P, U, D, (lambda)*eps);
	for(int i=0; i<params.n_steps-1; ++i) {
		step_U(P, U, 0.5*eps);
		iter += step_P(P, U, D, (1.0 - 2.0*lambda)*eps);
		step_U(P, U, 0.5*eps);
		iter += step_P(P, U, D, (2.0*lambda)*eps);
	}
	step_U(P, U, 0.5*eps);
	iter += step_P(P, U, D, (1.0 - 2.0*lambda)*eps);
	step_U(P, U, 0.5*eps);
	iter += step_P(P, U, D, (lambda)*eps);
	return iter;
}

double rhmc::action (field<gauge>& U, const field<gauge>& P, dirac_op& D) {
	return action_U (U) + action_P (P) + action_F (U, D);
}

double rhmc::action_U (const field<gauge>& U) {
	return - params.beta * 6.0 * U.V * plaq(U);
}

double rhmc::action_P (const field<gauge>& P) {
	double ac = 0.0;
	#pragma omp parallel for reduction (+:ac)
	for(int ix=0; ix<P.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			ac += (P[ix][mu]*P[ix][mu]).trace().real();
		}
	}
	return ac;
}

double rhmc::action_F (field<gauge>& U, dirac_op& D) {
	//std::cout << "#RHMC fermion action: n_f=" << params.n_f << ", n_pf=" << params.n_pf << ", n_rational/2=" << n_rational/2 << ", n_shifts hi=" << params.RA.beta_inv_hi[n_rational/2].size() << std::endl;
	double ac_F = 0;
	field<fermion> Ainv_phi (phi[0].grid, phi[0].eo_storage);
	for(int i_pf=0; i_pf<params.n_pf; ++i_pf) {
		// construct Ainv_phi = M^{-1/(2*(n_rational/2)) phi[i]
		int iter = rational_approx_cg_multishift(Ainv_phi, phi[i_pf], U, params.RA.alpha_inv_hi[n_rational/2], params.RA.beta_inv_hi[n_rational/2], D, 1.e-15);
		ac_F += phi[i_pf].dot(Ainv_phi).real();
	}
	return ac_F;
}

int rhmc::step_P (field<gauge>& P, field<gauge> &U, dirac_op& D, double eps) {
	field<gauge> force (U.grid);
	force_gauge (force, U);
	int iter = force_fermion (force, U, D);	
	#pragma omp parallel for
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			P[ix][mu] -= eps * force[ix][mu];
		}
	}
	return iter;
}

void rhmc::step_U (const field<gauge>& P, field<gauge> &U, double eps) {
	#pragma omp parallel for
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			U[ix][mu] = ((std::complex<double> (0.0, eps) * P[ix][mu]).exp()) * U[ix][mu];
		}
	}
}

void rhmc::random_U (field<gauge> &U, double eps) {
	// NO openMP: rng not threadsafe!
	gaussian_P(U);
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			U[ix][mu] = ((std::complex<double> (0.0, eps) * U[ix][mu]).exp()).eval();
		}
	}
}

void rhmc::gaussian_fermion (field<fermion>& chi) {
	// normal distribution p(x) ~ exp(-x^2), i.e. mu=0, sigma=1/sqrt(2):
	// NO openMP: rng not threadsafe!
	std::normal_distribution<double> randdist_gaussian (0, 1.0/sqrt(2.0));
	for(int ix=0; ix<chi.V; ++ix) {
		for(int i=0; i<3; ++i) {
			std::complex<double> r (randdist_gaussian (rng), randdist_gaussian (rng));
			chi[ix](i) = r;
		}
	}
}

void rhmc::gaussian_P (field<gauge>& P) {
	// normal distribution p(x) ~ exp(-x^2/2), i.e. mu=0, sigma=1:
	// could replace sum of c*T with single function assigning c 
	// to the appropriate matrix elements if this needs to be faster 
	// NO openMP: rng not threadsafe!
	std::normal_distribution<double> randdist_gaussian (0, 1.0);
	SU3_Generators T;
	for(int ix=0; ix<P.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			P[ix][mu].setZero();
			for(int i=0; i<8; ++i) {
				std::complex<double> c (randdist_gaussian (rng), 0.0);
				P[ix][mu] += c * T[i];
			}
		}
	}
}

void rhmc::force_gauge (field<gauge> &force, const field<gauge> &U) {
	#pragma omp parallel for
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			SU3mat A = staple (ix, mu, U);
			SU3mat F = U[ix][mu]*A;
			std::complex<double> ibeta_12 (0.0, -params.beta / 12.0);
			A = F - F.adjoint();
			force[ix][mu] = ( A - (A.trace()/3.0)*SU3mat::Identity() ) * ibeta_12;
		}
	}
}

int rhmc::force_fermion (field<gauge> &force, field<gauge> &U, dirac_op& D) {
	//std::cout << "#RHMC force: n_f=" << params.n_f << ", n_pf=" << params.n_pf << ", n_rational/2=" << n_rational/2 << ", n_shifts lo=" << params.RA.beta_inv_lo[n_rational/2].size() << std::endl;
	int n_shifts = params.RA.beta_inv_lo[n_rational/2].size();
	std::vector< field<fermion> > chi;
	for(int i_s=0; i_s<n_shifts; ++i_s) {
		chi.push_back(phi[0]);
	}

	field<fermion>::eo_storage_options eo_storage = field<fermion>::FULL;
	if(params.EE) {
		eo_storage = field<fermion>::ODD_ONLY;
	}
	field<fermion> psi (phi[0].grid, eo_storage);

	int iter=0;
	for(int i_pf=0; i_pf<params.n_pf; ++i_pf) {
		// chi_{i_s} = [(DD^dagger + beta_{i_s}]^-1 phi[i_pf]
		iter += cg_multishift (chi, phi[i_pf], U, params.RA.beta_inv_lo[n_rational/2], D, params.MD_eps);
		for(int i_s=0; i_s<n_shifts; ++i_s) {
			// psi_i = -D^dagger(mu,m) chi or Doe chi
			// psi = -D^dagger(mu,m) chi = D(-mu, -m) chi
			if(params.EE) {
				D.apply_eta_bcs_to_U(U);
				D.D_oe(psi, chi[i_s], U);
				D.remove_eta_bcs_from_U(U);
			} else {
				// apply -Ddagger = D(-m, -mu)
				// NB: should put this in a function
				D.mass = -D.mass;
				D.mu_I = -D.mu_I;
				D.D(psi, chi[i_s], U);
				D.mass = -D.mass;
				D.mu_I = -D.mu_I;
			}

			// usual force term expressions
			// but with multiplicative alpha_inv factor:
			double a_rational = params.RA.alpha_inv_lo[n_rational/2][i_s+1];
			D.apply_eta_bcs_to_U(U);
			SU3_Generators T;
			if(params.EE) {
				// for even-even version half of the terms are zero:
				// even ix: ix = ix_e
				#pragma omp parallel for
				for(int ix=0; ix<chi[i_s].V; ++ix) {
					for(int mu=0; mu<4; ++mu) {
						for(int a=0; a<8; ++a) {
							double Fa = a_rational * (chi[i_s][ix].dot(T[a] * U[ix][mu] * psi.up(ix,mu))).imag();
							force[ix][mu] -= Fa * T[a];
						}
					}
				}
				// odd ix: ix = ix_o + chi.V
				#pragma omp parallel for
				for(int ix_o=0; ix_o<chi[i_s].V; ++ix_o) {
					int ix = ix_o + chi[i_s].V;
					for(int mu=0; mu<4; ++mu) {
						for(int a=0; a<8; ++a) {
							double Fa = a_rational * (chi[i_s].up(ix,mu).dot(U[ix][mu].adjoint() * T[a] * psi[ix_o])).imag();
							force[ix][mu] -= Fa * T[a];
						}
					}
				}
			} else {
				// mu=0 terms have extra chemical potential isospin factors exp(+-\mu_I/2):
				double mu_I_plus_factor = exp(0.5 * params.mu_I);
				double mu_I_minus_factor = exp(-0.5 * params.mu_I);
				#pragma omp parallel for
				for(int ix=0; ix<U.V; ++ix) {
					for(int a=0; a<8; ++a) {
						double Fa = a_rational * chi[i_s][ix].dot(T[a] * mu_I_plus_factor * U[ix][0] * psi.up(ix,0)).imag();
						Fa += a_rational * chi[i_s].up(ix,0).dot(U[ix][0].adjoint() * mu_I_minus_factor * T[a] * psi[ix]).imag();
						force[ix][0] -= Fa * T[a];
					}
					for(int mu=1; mu<4; ++mu) {
						for(int a=0; a<8; ++a) {
							double Fa = a_rational * (chi[i_s][ix].dot(T[a] * U[ix][mu] * psi.up(ix,mu))).imag();
							Fa += a_rational * (chi[i_s].up(ix,mu).dot(U[ix][mu].adjoint() * T[a] * psi[ix])).imag();
							force[ix][mu] -= Fa * T[a];
						}
					}
				}
			}
			D.remove_eta_bcs_from_U(U);
		} // loop over shifts
	} // loop over pseudo fermion flavours
	return iter;
}

SU3mat rhmc::staple (int ix, int mu, const field<gauge> &U) {
	SU3mat A = SU3mat::Zero();
	int ix_plus_mu = U.iup(ix,mu);
	for(int nu=0; nu<4; ++nu) {
		if(nu!=mu) {
			A += U[ix_plus_mu][nu] * U.up(ix,nu)[mu].adjoint() * U[ix][nu].adjoint();
			A += U.dn(ix_plus_mu,nu)[nu].adjoint() * U.dn(ix,nu)[mu].adjoint() * U.dn(ix,nu)[nu];
		}
	}
	return A;
}

double rhmc::plaq (const field<gauge> &U) {
	double p = 0;
	#pragma omp parallel for reduction (+:p)
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=1; mu<4; ++mu) {
			for(int nu=0; nu<mu; nu++) {
				p += ((U[ix][mu]*U.up(ix,mu)[nu])*((U[ix][nu]*U.up(ix,nu)[mu]).adjoint())).trace().real();
			}
		}
	}
	return p / static_cast<double>(3*6*U.V);
}

std::complex<double> rhmc::polyakov_loop (const field<gauge> &U) {
	std::complex<double> p = 0;
	// NB: no openmp reduction for std::complex, would have to split into real and imag parts
	for(int ix3=0; ix3<U.VOL3; ++ix3) {
		int ix = U.it_ix(0, ix3);
		SU3mat P = U[ix][0];
		for(int x0=1; x0<U.L0; x0++) {
			ix = U.iup(ix, 0);
			P *= U[ix][0];
		}
		p += P.trace();
	}
	return p / static_cast<double>(3 * U.VOL3);
}
