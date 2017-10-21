#include "hmc.hpp"
#include <complex>
#include <iostream> //FOR DEBUGGING

hmc::hmc (unsigned int seed) : rng(seed) {
	// default value
	par.MD_eps = 1.e-7;
}

int hmc::trajectory (field<gauge>& U, hmc_params par, const dirac_op& D) {
	// make random gaussian momentum field P
	field<gauge> P (U.grid);
	gaussian_P (P);

	// make gaussian fermion field
	field<fermion> chi (U.grid);
	gaussian_fermion (chi);
	// construct phi = D chi
	field<fermion> phi (U.grid);
	D.D (phi, chi, U, par.mass);

	// save current gauge config in case update is rejected
	field<gauge> U_old (U.grid);
	U_old = U;

	double action_old = action(U, phi, P, par.beta, par.mass, D);
	leapfrog (U, phi, P, par.tau, par.n_steps, par.beta, par.mass, D);
	double action_new = action(U, phi, P, par.beta, par.mass, D);
	deltaE = action_new - action_old; //debugging
	return accept_reject (U, U_old, action_new - action_old);
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

int hmc::leapfrog (field<gauge>& U, field<fermion>& phi, field<gauge>& P, double tau, int n_steps, double beta, double m, const dirac_op& D) {
	double eps = tau/static_cast<double>(n_steps);
	int iter = 0;
	// leapfrog integration:
	iter += step_P(P, U, phi, beta, m, D, 0.5*eps);
	for(int i=0; i<n_steps-1; ++i) {
		step_U(P, U, eps);
		iter += step_P(P, U, phi, beta, m, D, eps);
	}
	step_U(P, U, eps);
	iter += step_P(P, U, phi, beta, m, D, 0.5*eps);
	return iter;
}

double hmc::action (const field<gauge>& U, const field<fermion>& phi, const field<gauge>& P, double beta, double m, const dirac_op& D) {
	return action_U (U, beta) + action_P (P) + action_F (U, phi, m, D);
}

double hmc::action_U (const field<gauge>& U, double beta) {
	return - beta * 6.0 * U.V * plaq(U);
}

double hmc::action_P (const field<gauge>& P) {
	double ac = 0.0;
	for(int ix=0; ix<P.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			ac += (P[ix][mu]*P[ix][mu]).trace().real();
		}
	}
	return ac;
}

double hmc::action_F (const field<gauge>& U, const field<fermion>& phi, double m, const dirac_op& D) {
	field<fermion> D2inv_phi (phi.grid);
	int iter = D.cg(D2inv_phi, phi, U, m, 1.e-15);
	//std::cout << "Action CG iterations: " << iter << std::endl;
	return phi.dot(D2inv_phi).real();
}

int hmc::step_P (field<gauge>& P, const field<gauge> &U, const field<fermion>& phi, double beta, double m, const dirac_op& D, double eps) {
	field<gauge> force (U.grid);
	force_gauge (force, U, beta);
	int iter = force_fermion (force, U, phi, m, D);	
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			P[ix][mu] -= eps * force[ix][mu];
		}
	}
	return iter;
}

void hmc::step_U (const field<gauge>& P, field<gauge> &U, double eps) {
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			U[ix][mu] = ((std::complex<double> (0.0, eps) * P[ix][mu]).exp()) * U[ix][mu];
		}
	}
}

void hmc::random_U (field<gauge> &U, double eps) {
	gaussian_P(U);
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			U[ix][mu] = ((std::complex<double> (0.0, eps) * U[ix][mu]).exp()).eval();
		}
	}
}

void hmc::gaussian_fermion (field<fermion>& chi) {
	// normal distribution p(x) ~ exp(-x^2), i.e. mu=0, sigma=1/sqrt(2):
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

void hmc::force_gauge (field<gauge> &force, const field<gauge> &U, double beta) {
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			SU3mat A = staple (ix, mu, U);
			SU3mat F = U[ix][mu]*A;
			std::complex<double> ibeta_12 (0.0, -beta / 12.0);
			A = F - F.adjoint();
			force[ix][mu] = ( A - (A.trace()/3.0)*SU3mat::Identity() ) * ibeta_12;
		}
	}
}

int hmc::force_fermion (field<gauge> &force, const field<gauge> &U, const field<fermion>& phi, double m, const dirac_op& D) {
	// chi = (DD)^-1 phi
	field<fermion> chi (phi.grid);
	int iter = D.cg (chi, phi, U, m, par.MD_eps);
	// psi = g5D chi
	field<fermion> psi (phi.grid);
	D.D(psi, chi, U, m);
	D.gamma5(psi);

	SU3_Generators T;
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			for(int a=0; a<8; ++a) {
				double Fa = D.eta[ix][4]*D.eta[ix][mu] * (chi[ix].dot(T[a] * U[ix][mu] * psi.up(ix,mu))).imag();
				Fa += D.eta.up(ix,mu)[4]*D.eta.up(ix,mu)[mu] * (chi.up(ix,mu).dot(U[ix][mu].adjoint() * T[a] * psi[ix])).imag();
				force[ix][mu] += Fa * T[a];
			}
		}
	}
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

double hmc::plaq (const field<gauge> &U) {
	double p = 0;
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=1; mu<4; ++mu) {
			for(int nu=0; nu<mu; nu++) {
				p += plaq (ix, mu, nu, U);
			}
		}
	}
	return p / static_cast<double>(3*6*U.V);
}