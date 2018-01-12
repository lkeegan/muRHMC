#include "hmc.hpp"
#include <complex>
#include <iostream> //FOR DEBUGGING
#include "io.hpp" //DEBUGGING

hmc::hmc (const hmc_params& params) : rng(params.seed), params(params) {
}

int hmc::trajectory (field<gauge>& U, dirac_op& D) {
	// make random gaussian momentum field P
	field<gauge> P (U.grid);
	gaussian_P (P);
	// make gaussian fermion field
	field<fermion> chi (U.grid);
	gaussian_fermion (chi);
	// construct phi = D chi
	field<fermion> phi (U.grid);
	D.D (phi, chi, U, params.mass, params.mu_I);

	// make copy of current gauge config in case update is rejected
	field<gauge> U_old (U.grid);
	U_old = U;

	double action_old = action(U, phi, P, D);
	// do integration
	OMF2 (U, phi, P, D);
	// calculate change in action
	double action_new = action(U, phi, P, D);
	deltaE = action_new - action_old;

	if(params.constrained) {
		// this value is stored from the last trajectory
		//log("[HMC] old-sus", suscept);
		double new_suscept = D.pion_susceptibility_exact(U, params.mass, params.mu_I);
		suscept_proposed = new_suscept;
		if (new_suscept > (params.suscept_central + params.suscept_delta)) {
			// suscept too high: reject proposed update, restore old U
			U = U_old;
			//log("[HMC] sus too high - rejected: upper bound:", params.suscept_central + params.suscept_delta);
			return 0;
		}
		else if (new_suscept < (params.suscept_central - params.suscept_delta)) {
			// suscept too low: reject proposed update, restore old U
			U = U_old;
			//log("[HMC] sus too low - rejected: lower bound:", params.suscept_central - params.suscept_delta);
			return 0;
		}
		else {
			// susceptibility is in acceptable range: do standard accept/reject step on dE
			if(accept_reject (U, U_old, action_new - action_old)) {
				// if accepted update cached susceptibility and return 1
				suscept = new_suscept;
				return 1;
			}
			else {
				return 0;
			}
		}
	}
	else {
		return accept_reject (U, U_old, action_new - action_old);
	}
}

int hmc::accept_reject (field<gauge>& U, const field<gauge>& U_old, double dE) {
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

int hmc::leapfrog (field<gauge>& U, field<fermion>& phi, field<gauge>& P, dirac_op& D) {
	double eps = params.tau / static_cast<double>(params.n_steps);
	int iter = 0;
	// leapfrog integration:
	iter += step_P(P, U, phi, D, 0.5*eps);
	for(int i=0; i<params.n_steps-1; ++i) {
		step_U(P, U, eps);
		iter += step_P(P, U, phi, D, eps);
	}
	step_U(P, U, eps);
	iter += step_P(P, U, phi, D, 0.5*eps);
	return iter;
}

int hmc::OMF2 (field<gauge>& U, field<fermion>& phi, field<gauge>& P, dirac_op& D) {
	double constexpr lambda = 0.19318; //tunable parameter
	double eps = params.tau / static_cast<double>(params.n_steps);
	int iter = 0;
	// OMF2 integration:
	iter += step_P(P, U, phi, D, (lambda)*eps);
	for(int i=0; i<params.n_steps-1; ++i) {
		step_U(P, U, 0.5*eps);
		iter += step_P(P, U, phi, D, (1.0 - 2.0*lambda)*eps);
		step_U(P, U, 0.5*eps);
		iter += step_P(P, U, phi, D, (2.0*lambda)*eps);
	}
	step_U(P, U, 0.5*eps);
	iter += step_P(P, U, phi, D, (1.0 - 2.0*lambda)*eps);
	step_U(P, U, 0.5*eps);
	iter += step_P(P, U, phi, D, (lambda)*eps);
	return iter;
}

double hmc::action (field<gauge>& U, const field<fermion>& phi, const field<gauge>& P, dirac_op& D) {
	return action_U (U) + action_P (P) + action_F (U, phi, D);
}

double hmc::action_U (const field<gauge>& U) {
	return - params.beta * 6.0 * U.V * plaq(U);
}

double hmc::action_P (const field<gauge>& P) {
	double ac = 0.0;
	#pragma omp parallel for reduction (+:ac)
	for(int ix=0; ix<P.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			ac += (P[ix][mu]*P[ix][mu]).trace().real();
		}
	}
	return ac;
}

double hmc::action_F (field<gauge>& U, const field<fermion>& phi, dirac_op& D) {
	field<fermion> D2inv_phi (phi.grid);
	int iter = D.cg(D2inv_phi, phi, U, params.mass, params.mu_I, 1.e-15);
	//std::cout << "Action CG iterations: " << iter << std::endl;
	return phi.dot(D2inv_phi).real();
}

int hmc::step_P (field<gauge>& P, field<gauge> &U, const field<fermion>& phi, dirac_op& D, double eps) {
	field<gauge> force (U.grid);
	force_gauge (force, U);
	int iter = force_fermion (force, U, phi, D);	
	#pragma omp parallel for
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			P[ix][mu] -= eps * force[ix][mu];
		}
	}
	return iter;
}

void hmc::step_U (const field<gauge>& P, field<gauge> &U, double eps) {
	#pragma omp parallel for
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			U[ix][mu] = ((std::complex<double> (0.0, eps) * P[ix][mu]).exp()) * U[ix][mu];
		}
	}
}

void hmc::random_U (field<gauge> &U, double eps) {
	// NO openMP: rng not threadsafe!
	gaussian_P(U);
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			U[ix][mu] = ((std::complex<double> (0.0, eps) * U[ix][mu]).exp()).eval();
		}
	}
}

void hmc::gaussian_fermion (field<fermion>& chi) {
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

void hmc::gaussian_P (field<gauge>& P) {
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

void hmc::force_gauge (field<gauge> &force, const field<gauge> &U) {
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

int hmc::force_fermion (field<gauge> &force, field<gauge> &U, const field<fermion>& phi, dirac_op& D) {
	// anti-periodic boundary conditions:
	// want to set F -> -F at end of this for [x0=T-1, mu=0]
	// but we are incrementing existing F, so first set existing to -itself
	// then minus the whole thing at the end
	D.apbs_in_time(force);

	// chi = (D(mu)D^dagger(mu))^-1 phi
	field<fermion> chi (phi.grid);
	int iter = D.cg (chi, phi, U, params.mass, params.mu_I, params.MD_eps);
	// psi = -D^dagger(mu,m) chi = D(-mu, -m) chi
	field<fermion> psi (phi.grid);
	D.D(psi, chi, U, -params.mass, -params.mu_I);

	// mu=0 terms have extra chemical potential isospin factors exp(+-\mu_I/2):
	double mu_I_plus_factor = exp(0.5 * params.mu_I);
	double mu_I_minus_factor = exp(-0.5 * params.mu_I);
	SU3_Generators T;
	#pragma omp parallel for
	for(int ix=0; ix<U.V; ++ix) {
		for(int a=0; a<8; ++a) {
			double Fa = chi[ix].dot(T[a] * mu_I_plus_factor * U[ix][0] * psi.up(ix,0)).imag();
			Fa += chi.up(ix,0).dot(U[ix][0].adjoint() * mu_I_minus_factor * T[a] * psi[ix]).imag();
			force[ix][0] -= Fa * T[a];
		}
		for(int mu=1; mu<4; ++mu) {
			for(int a=0; a<8; ++a) {
				double Fa = D.eta[ix][mu] * (chi[ix].dot(T[a] * U[ix][mu] * psi.up(ix,mu))).imag();
				Fa += D.eta[ix][mu] * (chi.up(ix,mu).dot(U[ix][mu].adjoint() * T[a] * psi[ix])).imag();
				force[ix][mu] -= Fa * T[a];
			}
		}
	}

	D.apbs_in_time(force);

	return iter;
}

SU3mat hmc::staple (int ix, int mu, const field<gauge> &U) {
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

double hmc::plaq (int ix, int mu, int nu, const field<gauge> &U) {
	return ((U[ix][mu]*U.up(ix,mu)[nu])*((U[ix][nu]*U.up(ix,nu)[mu]).adjoint())).trace().real();
}

double hmc::plaq (int ix, const field<gauge> &U) {
	double p = 0;
	for(int mu=1; mu<4; ++mu) {
		for(int nu=0; nu<mu; nu++) {
			p += plaq (ix, mu, nu, U);
		}
	}
	return p;
}

double hmc::plaq (const field<gauge> &U) {
	double p = 0;
	#pragma omp parallel for reduction (+:p)
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=1; mu<4; ++mu) {
			for(int nu=0; nu<mu; nu++) {
				p += plaq (ix, mu, nu, U);
			}
		}
	}
	return p / static_cast<double>(3*6*U.V);
}

double hmc::plaq_spatial (const field<gauge> &U) {
	double p = 0;
	#pragma omp parallel for reduction (+:p)
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=1; mu<4; ++mu) {
			for(int nu=1; nu<mu; nu++) {
				p += plaq (ix, mu, nu, U);
			}
		}
	}
	return p / static_cast<double>(3*3*U.V);
}

double hmc::plaq_timelike (const field<gauge> &U) {
	double p = 0;
	#pragma omp parallel for reduction (+:p)
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=1; mu<4; ++mu) {
			p += plaq (ix, mu, 0, U);
		}
	}
	return p / static_cast<double>(3*3*U.V);
}

std::complex<double> hmc::polyakov_loop (const field<gauge> &U) {
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

double hmc::chiral_condensate (field<gauge> &U, dirac_op& D) {
	// inverter precision
	double eps = 1.e-12;
	// phi = gaussian noise vector
	field<fermion> phi(U.grid);
	gaussian_fermion(phi);
	// chi = [D(mu,m)D(mu,m)^dag]-1 phi
	field<fermion> chi(U.grid);
	D.cg(chi, phi, U, params.mass, params.mu_I, eps);
	// psi = D(mu,m)^dag chi = - D(-mu,-m) chi = D(mu)^-1 phi
	field<fermion> psi(U.grid);
	D.D(psi, chi, U, -params.mass, -params.mu_I);
	return -phi.dot(psi).real() / static_cast<double>(phi.V);
}
