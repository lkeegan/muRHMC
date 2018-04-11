#include "rhmc.hpp"
#include "rational_approx.hpp"
#include <complex>
#include <iostream> //FOR DEBUGGING
#include "io.hpp" //DEBUGGING
#include <chrono>

rhmc::rhmc (const rhmc_params& params, const lattice& grid) : RA(1.0, 1.0), rng(params.seed), params(params), grid(grid), block_phi(grid) {
	// rational approx for action: M^{n_f/(8 params.n_pf)} = M^{1/(n_rational)}
	n_rational = 8 * params.n_pf / params.n_f;
	if(params.EE) {
		// only use even part of pseudofermions and even-even sub-block of dirac op
		eo_storage_e = field<fermion>::EVEN_ONLY;
		eo_storage_o = field<fermion>::ODD_ONLY;
		block_eo_storage_e = field<block_fermion>::EVEN_ONLY;
		block_eo_storage_o = field<block_fermion>::ODD_ONLY;
		// rational approx for action: M^{n_f/(4 params.n_pf)} = M^{1/(n_rational)}
		n_rational = 4 * params.n_pf / params.n_f;
	}
	if(n_rational<2) {
		std::cout << "Invalid combination of n_f, n_pf: need n_pf >= n_f/2 for EE, or n_f/4 without e/o predconditioning" << std::endl;
		exit(1);
	}
	if(params.block && (params.n_pf != N_rhs)) {
		std::cout << "n_pf must be equal to N_rhs to use block operations" << std::endl;
		exit(1);
	}
}

int rhmc::trajectory (field<gauge>& U, dirac_op& D, bool do_reversibility_test, bool MEASURE_FORCE_ERROR_NORMS) {
	// set Dirac op mass and isospin mu values
	D.mass = params.mass;
	D.mu_I = params.mu_I;

	// set rational approximations
	RA = rational_approx(params.mass*params.mass, D.largest_eigenvalue_bound(U, eo_storage_e));
	
	//log("RHMC: RA lower_bound", RA.lower_bound);
	//log("RHMC: RA upper_bound", RA.upper_bound);

	//log("RHMC: HB 2n_rational", 2*n_rational);
	//log("RHMC: HB alpha_hi", RA.alpha_hi[2*n_rational]);
	//log("RHMC: HB beta_hi", RA.beta_hi[2*n_rational]);

	//log("RHMC: action n_rational", n_rational);
	//log("RHMC: action alpha_inv_hi", RA.alpha_inv_hi[n_rational]);
	//log("RHMC: action beta_inv_hi", RA.beta_inv_hi[n_rational]);

	//log("RHMC: force n_rational", n_rational);
	//log("RHMC: force alpha_inv_lo", RA.alpha_inv_lo[n_rational]);
	//log("RHMC: force beta_inv_lo", RA.beta_inv_lo[n_rational]);

	// make random gaussian momentum field P
	field<gauge> P (U.grid);
	gaussian_P (P);
	//std::cout << "#RHMC: n_f=" << params.n_f << ", n_pf=" << params.n_pf << ", n_rational=" << n_rational << ", n_shifts hi/lo=" << RA.beta_hi[n_rational].size() << "/" << RA.beta_lo[n_rational].size() << std::endl;

	double chi_norm = 0;
	if(params.block) {
		field<block_fermion> chi (U.grid, block_eo_storage_e);
		block_phi = chi;
	    auto timer_start = std::chrono::high_resolution_clock::now();
		chi_norm = gaussian_fermion (chi);
		// construct phi[i] = M^{1/(2*n_rational) chi_e
		int iter = rational_approx_SBCGrQ(block_phi, chi, U, RA.alpha_hi[2*n_rational], RA.beta_hi[2*n_rational], D, params.HB_eps);
	    auto timer_stop = std::chrono::high_resolution_clock::now();
	    auto timer_count = std::chrono::duration_cast<std::chrono::seconds>(timer_stop-timer_start).count();
		std::cout << "#RHMC_InitBCGRuntime " << timer_count << std::endl;
		std::cout << "#RHMC_InitBCGIter " << iter << std::endl;
	} else {
		std::vector< field<fermion> > chi (params.n_pf, field<fermion>(U.grid, eo_storage_e));
		phi.resize(params.n_pf, chi[0]);
	    auto timer_start = std::chrono::high_resolution_clock::now();
		int iter = 0;
		chi_norm = gaussian_fermion (chi);
		for(int i_pf=0; i_pf<params.n_pf; ++i_pf) {
			// construct phi[i] = M^{1/(2*n_rational) chi_e
			iter += rational_approx_cg_multishift(phi[i_pf], chi[i_pf], U, RA.alpha_hi[2*n_rational], RA.beta_hi[2*n_rational], D, params.HB_eps);
		}
	    auto timer_stop = std::chrono::high_resolution_clock::now();
	    auto timer_count = std::chrono::duration_cast<std::chrono::seconds>(timer_stop-timer_start).count();
		std::cout << "#RHMC_InitCGRuntime " << timer_count << std::endl;
		std::cout << "#RHMC_InitCGIter " << iter << std::endl;
	}

	// make copy of current gauge config in case update is rejected
	field<gauge> U_old (U.grid);
	U_old = U;

	// debugging
	field<gauge> P_old (U.grid);
	if(do_reversibility_test) {
		P_old = P;
	}

	double action_old = chi_norm + action_U (U) + action_P (P);

	// FORCE MEASUREMENT: measure force norms once and exit:
	if(MEASURE_FORCE_ERROR_NORMS) {

		// just measure one force term using supplied MD_eps for inversion
		std::cout.precision(15);
		field<gauge> force (U.grid);
		force.setZero();
		if(params.block) {
			int iterBCG = force_fermion_block (force, U, D);
			std::cout << std::scientific << "fermion_force_normBCG: " << force.squaredNorm() << std::endl;
		} else {
			force.setZero();
			int iterCG = force_fermion (force, U, D);
			std::cout << std::scientific << "fermion_force_normCG: " << force.squaredNorm() << std::endl;		
		}
		return 1;

		// do full force error measurements:
		/*
		std::cout.precision(12);
		// get "exact" high precision solve force vector
	    auto timer_start = std::chrono::high_resolution_clock::now();
		field<gauge> force_star (U.grid);
		force_star.setZero();
		params.MD_eps = 1.e-12;
		int iter = force_fermion_block (force_star, U, D);
		force_star *= -1.0;
		double f_star_norm = force_star.norm();
	    auto timer_stop = std::chrono::high_resolution_clock::now();
		auto timer_count = std::chrono::duration_cast<std::chrono::seconds>(timer_stop-timer_start).count();
		std::cout << std::scientific << "# correct force iter " << iter << "\t norm" << f_star_norm << "\t runtime: " << timer_count << std::endl;

		// get approx force vector, output norm of difference
		field<gauge> force_eps (U.grid);
		std::vector<double> residuals = {1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7};
		std::cout << "# eps, CGiter, CGerror-norm, CGruntime, BLOCKiter, BLOCKerror-norm, BLOCKruntime, true-force-norm" << std::endl;
		for(int i_eps=0; i_eps<static_cast<int>(residuals.size()); ++i_eps) {
			params.MD_eps = residuals[i_eps];
		    auto timer_start = std::chrono::high_resolution_clock::now();
			force_eps = force_star;
			int CGiter = force_fermion (force_eps, U, D);
		    auto timer_stop = std::chrono::high_resolution_clock::now();
			auto CGtimer = std::chrono::duration_cast<std::chrono::seconds>(timer_stop-timer_start).count();
			double CGerr = force_eps.norm();
		    timer_start = std::chrono::high_resolution_clock::now();
			force_eps = force_star;
			int BLOCKiter = force_fermion_block (force_eps, U, D);
		    timer_stop = std::chrono::high_resolution_clock::now();
			auto BLOCKtimer = std::chrono::duration_cast<std::chrono::seconds>(timer_stop-timer_start).count();
			double BLOCKerr = force_eps.norm();
			std::cout << std::scientific << residuals[i_eps] << "\t" << CGiter << "\t" << CGerr << "\t" << CGtimer << "\t"
					  << BLOCKiter << "\t" << BLOCKerr << "\t" << BLOCKtimer << "\t" << f_star_norm << std::endl;
		}
		exit(0);
		*/
	}

	// DEBUGGING
	/*
	std::cout << "chidag.chi " << chi_norm << std::endl;
	std::cout << "fermion_ac " << action_F(U, D) << std::endl;
	std::cout << "full_ac " << action(U, P, D) << std::endl;
	std::cout << "full_ac " << action_old << std::endl;
	*/
	// do integration
	OMF2 (U, P, D);

	// DEBUGGING reversibility test: P->-P, repeat integration, no accept/reject step
	// U should equal to inital U to ~1e-15
	if(do_reversibility_test) {
		P *= -1;
		OMF2 (U, P, D);
		P_old += P;
		std::cout << "P reserve dev: " << P_old.norm() << std::endl;
		U_old -= U;
		std::cout << "U reserve dev: " << U_old.norm() << std::endl;
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

void rhmc::leapfrog_pure_gauge (field<gauge>& U, field<gauge>& P) {
	double eps = params.tau / static_cast<double>(params.n_steps_fermion * params.n_steps_gauge);
	// leapfrog integration:
	step_P_pure_gauge(P, U, 0.5*eps, true);
	for(int i=0; i<params.n_steps_gauge-1; ++i) {
		step_U(P, U, eps);
		step_P_pure_gauge(P, U, eps);
	}
	step_U(P, U, eps);
	step_P_pure_gauge(P, U, 0.5*eps);
}

int rhmc::leapfrog (field<gauge>& U, field<gauge>& P, dirac_op& D) {
	double eps = params.tau / static_cast<double>(params.n_steps_fermion);
	int iter = 0;
	// leapfrog integration:
	iter += step_P_fermion(P, U, D, 0.5*eps);
	for(int i=0; i<params.n_steps_fermion-1; ++i) {
		//step_U(P, U, eps);
		leapfrog_pure_gauge(U, P);
		iter += step_P_fermion(P, U, D, eps);
	}
	//step_U(P, U, eps);
	leapfrog_pure_gauge(U, P);
	iter += step_P_fermion(P, U, D, 0.5*eps);
	return iter;
}

void rhmc::OMF2_pure_gauge (field<gauge>& U, field<gauge>& P) {
	double eps = 0.5 * params.tau / static_cast<double>(params.n_steps_fermion * params.n_steps_gauge);
	// OMF2 integration:
	step_P_pure_gauge(P, U, (params.lambda_OMF2)*eps, true);
	for(int i=0; i<params.n_steps_gauge-1; ++i) {
		step_U(P, U, 0.5*eps);
		step_P_pure_gauge(P, U, (1.0 - 2.0*params.lambda_OMF2)*eps);
		step_U(P, U, 0.5*eps);
		step_P_pure_gauge(P, U, (2.0*params.lambda_OMF2)*eps);
	}
	step_U(P, U, 0.5*eps);
	step_P_pure_gauge(P, U, (1.0 - 2.0*params.lambda_OMF2)*eps);
	step_U(P, U, 0.5*eps);
	step_P_pure_gauge(P, U, (params.lambda_OMF2)*eps);
}

int rhmc::OMF2 (field<gauge>& U, field<gauge>& P, dirac_op& D) {
	double eps = params.tau / static_cast<double>(params.n_steps_fermion);
	int iter = 0;
	// OMF2 integration:
	iter += step_P_fermion(P, U, D, (params.lambda_OMF2)*eps);
	for(int i=0; i<params.n_steps_fermion-1; ++i) {
		//step_U(P, U, 0.5*eps);
		OMF2_pure_gauge(U, P);
		iter += step_P_fermion(P, U, D, (1.0 - 2.0*params.lambda_OMF2)*eps);
		//step_U(P, U, 0.5*eps);
		OMF2_pure_gauge(U, P);
		iter += step_P_fermion(P, U, D, (2.0*params.lambda_OMF2)*eps);
	}
	//step_U(P, U, 0.5*eps);
	OMF2_pure_gauge(U, P);
	iter += step_P_fermion(P, U, D, (1.0 - 2.0*params.lambda_OMF2)*eps);
	//step_U(P, U, 0.5*eps);
	OMF2_pure_gauge(U, P);
	iter += step_P_fermion(P, U, D, (params.lambda_OMF2)*eps);
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
	//std::cout << "#RHMC fermion action: n_f=" << params.n_f << ", n_pf=" << params.n_pf << ", n_rational/2=" << n_rational/2 << ", n_shifts hi=" << RA.beta_inv_hi[n_rational/2].size() << std::endl;
	if(params.block) {
		double ac_F = 0;
		field<block_fermion> Ainv_phi (block_phi);
		// construct Ainv_phi = M^{-1/(n_rational)) phi[i]
		int iter = rational_approx_SBCGrQ(Ainv_phi, block_phi, U, RA.alpha_inv_hi[n_rational], RA.beta_inv_hi[n_rational], D, params.HB_eps);
		for(int i=0; i<block_phi.V; ++i) {
			for(int i_pf=0; i_pf<params.n_pf; ++i_pf) {
				ac_F += block_phi[i].col(i_pf).dot(Ainv_phi[i].col(i_pf)).real();			
			}
		}
		std::cout << "#RHMC_ActionBCGIter " << iter << std::endl;
		return ac_F;		
	} else {
		double ac_F = 0;
		field<fermion> Ainv_phi (phi[0].grid, phi[0].eo_storage);
		int iter = 0;
		for(int i_pf=0; i_pf<params.n_pf; ++i_pf) {
			// construct Ainv_phi = M^{-1/(n_rational)) phi[i]
			iter += rational_approx_cg_multishift(Ainv_phi, phi[i_pf], U, RA.alpha_inv_hi[n_rational], RA.beta_inv_hi[n_rational], D, params.HB_eps);
			ac_F += phi[i_pf].dot(Ainv_phi).real();
		}
		std::cout << "#RHMC_ActionCGIter " << iter << std::endl;
		return ac_F;		
	}
}

void rhmc::step_P_pure_gauge (field<gauge>& P, field<gauge> &U, double eps, bool MEASURE_FORCE_NORM) {
	std::complex<double> ibeta_12 (0.0, -eps * params.beta / 12.0);
	double force_norm = 0;
	if(MEASURE_FORCE_NORM) {
		for(int ix=0; ix<U.V; ++ix) {
			for(int mu=0; mu<4; ++mu) {
				SU3mat A = staple (ix, mu, U);
				SU3mat F = U[ix][mu]*A;
				A = F - F.adjoint();
				F = ( A - (A.trace()/3.0)*SU3mat::Identity() ) * ibeta_12;
				force_norm += F.squaredNorm();
				P[ix][mu] -= F;
			}
		}
		force_norm /= eps*eps;
		std::cout << "gauge_force_norm: " << sqrt(force_norm/static_cast<double>(4*U.V)) << std::endl;
	} else {
		#pragma omp parallel for
		for(int ix=0; ix<U.V; ++ix) {
			for(int mu=0; mu<4; ++mu) {
				SU3mat A = staple (ix, mu, U);
				SU3mat F = U[ix][mu]*A;
				A = F - F.adjoint();
				P[ix][mu] -= ( A - (A.trace()/3.0)*SU3mat::Identity() ) * ibeta_12;
			}
		}
	}
}

int rhmc::step_P_fermion (field<gauge>& P, field<gauge> &U, dirac_op& D, double eps, bool MEASURE_FORCE_NORM) {
	field<gauge> force (U.grid);
	force.setZero();
	int iter = 0;
	if (params.block) {
		iter = force_fermion_block (force, U, D);	
	} else {
		iter = force_fermion (force, U, D);	
	}
	//std::cout << "F_g: " << force.norm() << std::endl;
	if(MEASURE_FORCE_NORM) {
		std::cout << "fermion_force_norm: " << sqrt(force.squaredNorm()/static_cast<double>(4*U.V)) << std::endl;
	}
	//std::cout << "F_g + F_pf: " << force.norm() << std::endl;
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
			U[ix][mu] = exp_ch((std::complex<double> (0.0, eps) * P[ix][mu])) * U[ix][mu];
			//U[ix][mu] = ((std::complex<double> (0.0, eps) * P[ix][mu]).exp()) * U[ix][mu];
		}
	}
}

void rhmc::random_U (field<gauge> &U, double eps) {
	// NO openMP: rng not threadsafe!
	gaussian_P(U);
	if(eps > 1.0) {
		eps = 1.0;
	}
	for(int ix=0; ix<U.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			U[ix][mu] = exp_ch((std::complex<double> (0.0, eps) * U[ix][mu]));
		}
	}
}

double rhmc::gaussian_fermion (std::vector< field<fermion> >& chi) {
	// normal distribution p(x) ~ exp(-x^2), i.e. mu=0, sigma=1/sqrt(2):
	// NO openMP: rng not threadsafe!
	int N = static_cast<int>(chi.size());
	double norm = 0;
	std::normal_distribution<double> randdist_gaussian (0, 1.0/sqrt(2.0));
	for(int ix=0; ix<chi[0].V; ++ix) {
		for(int j=0; j<N; ++j) {
			for(int i=0; i<3; ++i) {
				double re = randdist_gaussian (rng);
				double im = randdist_gaussian (rng);
				norm += re*re + im*im;
				chi[j][ix](i) = std::complex<double> (re, im);
			}
		}
	}
	return norm;
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

// DEBUGGING: output fermion norms and error vs high precision fermion force term versus 
// residual for each shift
int rhmc::force_fermion_norms (field<gauge> &force, field<gauge> &U, dirac_op& D) {
	//std::cout << "#RHMC force: n_f=" << params.n_f << ", n_pf=" << params.n_pf << ", n_rational/2=" << n_rational/2 << ", n_shifts lo=" << RA.beta_inv_lo[n_rational/2].size() << std::endl;
	int n_shifts = RA.beta_inv_lo[n_rational].size();
	std::vector< field<fermion> > chi_star(n_shifts, phi[0]);
	field<fermion> psi_star (phi[0].grid, eo_storage_o);

    auto timer_start = std::chrono::high_resolution_clock::now();
	field<fermion> chi (phi[0].grid, eo_storage_e);
	field<fermion> psi (phi[0].grid, eo_storage_o);

	std::vector<double> residuals = {1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7};
	std::vector<double> ff_norms(n_shifts, 0.0);
	std::vector< std::vector<double> > ff_error_norms(residuals.size(), ff_norms);
	std::vector< std::vector<double> > ff_error_cgiter(residuals.size(), ff_norms);
	int iter = 0;
	for(int i_pf=0; i_pf<params.n_pf; ++i_pf) {
		// chi_{i_s} = [(DD^dagger + beta_{i_s}]^-1 phi[i_pf]
		iter += cg_multishift (chi_star, phi[i_pf], U, RA.beta_inv_lo[n_rational], D, 1.e-12);
		//std::cout << "ipf " << i_pf << "cum iter " << iter << " inv0 norm " << chi[0].norm() << " Unorm " << U.norm() << " phinorm " << phi[i_pf].norm() << std::endl;
		for(int i_s=0; i_s<n_shifts; ++i_s) {
			// psi_i = -D^dagger(mu,m) chi or Doe chi
			// psi = -D^dagger(mu,m) chi = D(-mu, -m) chi
			D.apply_eta_bcs_to_U(U);
			D.D_oe(psi_star, chi_star[i_s], U);
			D.remove_eta_bcs_from_U(U);

			for(int i_eps=0; i_eps<static_cast<int>(residuals.size()); ++i_eps) {
				double eps = residuals[i_eps];
				// approx chi & psi:
				ff_error_cgiter[i_eps][i_s] += cg_singleshift (chi, phi[i_pf], U, RA.beta_inv_lo[n_rational][i_s], D, eps);
				D.apply_eta_bcs_to_U(U);
				D.D_oe(psi, chi, U);
				D.remove_eta_bcs_from_U(U);

				// usual force term expressions
				// but with multiplicative alpha_inv factor:
				double a_rational = RA.alpha_inv_lo[n_rational][i_s+1];
				D.apply_eta_bcs_to_U(U);
				SU3_Generators T;
					// for even-even version half of the terms are zero:
					// even ix: ix = ix_e
					#pragma omp parallel for
					for(int ix=0; ix<chi_star[i_s].V; ++ix) {
						for(int mu=0; mu<4; ++mu) {
							for(int a=0; a<8; ++a) {
								double Fa_star = a_rational * (chi_star[i_s][ix].dot(T[a] * U[ix][mu] * psi_star.up(ix,mu))).imag();
								double Fa = a_rational * (chi[ix].dot(T[a] * U[ix][mu] * psi.up(ix,mu))).imag();
								ff_norms[i_s] += 0.5 * Fa * Fa;
								ff_error_norms[i_eps][i_s] += 0.5 * (Fa - Fa_star) * (Fa - Fa_star);
								force[ix][mu] -= Fa * T[a];
							}
						}
					}
					// odd ix: ix = ix_o + chi.V
					#pragma omp parallel for
					for(int ix_o=0; ix_o<chi_star[i_s].V; ++ix_o) {
						int ix = ix_o + chi_star[i_s].V;
						for(int mu=0; mu<4; ++mu) {
							for(int a=0; a<8; ++a) {
								double Fa_star = a_rational * (chi_star[i_s].up(ix,mu).dot(U[ix][mu].adjoint() * T[a] * psi_star[ix_o])).imag();
								double Fa = a_rational * (chi.up(ix,mu).dot(U[ix][mu].adjoint() * T[a] * psi[ix_o])).imag();
								ff_norms[i_s] += 0.5 * Fa * Fa;
								ff_error_norms[i_eps][i_s] += 0.5 * (Fa - Fa_star) * (Fa - Fa_star);
								force[ix][mu] -= Fa * T[a];
							}
						}
					}
				D.remove_eta_bcs_from_U(U);
			} // end of loop over residuals
		} // end of loop over shifts
	} // end of loop over pseudo fermion flavours
	// output fermion force norms

	std::cout.precision(12);

	std::cout << "#i_shift, shift, force_norm, error, iter, epsilon" << std::endl;	
	for(int i_eps=0; i_eps<static_cast<int>(residuals.size()); ++i_eps) {
		for(int i_s=0; i_s<n_shifts; ++i_s) {
			std::cout 	<< std::scientific << i_s << "\t"
						<< RA.beta_inv_lo[n_rational][i_s] << "\t"
						<< sqrt(ff_norms[i_s]/static_cast<double>(4*residuals.size()*U.V)) << "\t"
						<< sqrt(ff_error_norms[i_eps][i_s]/static_cast<double>(4*U.V)) << "\t"
						<< ff_error_cgiter[i_eps][i_s] << "\t"
						<< residuals[i_eps] << std::endl;
		}
		std::cout << std::endl;
	}

	std::cout << "#i_shift #shift, force_norm, ";	
	for(int i_eps=0; i_eps<static_cast<int>(residuals.size()); ++i_eps) {
		std::cout << std::scientific << "res: " << residuals[i_eps] << ",\t cgiter,\t";
	}
	std::cout << std::endl;
	for(int i_s=0; i_s<n_shifts; ++i_s) {
		std::cout << "" << i_s << "\t" << RA.beta_inv_lo[n_rational][i_s] << "\t"
		<< sqrt(ff_norms[i_s]/static_cast<double>(4*residuals.size()*U.V)) << "\t";
		for(int i_eps=0; i_eps<static_cast<int>(residuals.size()); ++i_eps) {
			std::cout << std::scientific << sqrt(ff_error_norms[i_eps][i_s]/static_cast<double>(4*U.V)) << "\t"
			<< ff_error_cgiter[i_eps][i_s] << "\t";
		}
		std::cout << std::endl;
	}

    auto timer_stop = std::chrono::high_resolution_clock::now();
    auto timer_count = std::chrono::duration_cast<std::chrono::seconds>(timer_stop-timer_start).count();
	std::cout << "#RHMC_InitCGRuntime " << timer_count << std::endl;
	std::cout << "#RHMC_ForceCGIter " << iter << std::endl;

	return iter;
}

int rhmc::force_fermion (field<gauge> &force, field<gauge> &U, dirac_op& D) {
	//std::cout << "#RHMC force: n_f=" << params.n_f << ", n_pf=" << params.n_pf << ", n_rational/2=" << n_rational/2 << ", n_shifts lo=" << RA.beta_inv_lo[n_rational/2].size() << std::endl;
    auto timer_start = std::chrono::high_resolution_clock::now();
	int n_shifts = RA.beta_inv_lo[n_rational].size();
	std::vector< field<fermion> > chi(n_shifts, phi[0]);

	field<fermion> psi (phi[0].grid, eo_storage_o);

	int iter = 0;
	for(int i_pf=0; i_pf<params.n_pf; ++i_pf) {
		// chi_{i_s} = [(DD^dagger + beta_{i_s}]^-1 phi[i_pf]
		iter += cg_multishift (chi, phi[i_pf], U, RA.beta_inv_lo[n_rational], D, params.MD_eps);
		//std::cout << "ipf " << i_pf << "cum iter " << iter << " inv0 norm " << chi[0].norm() << " Unorm " << U.norm() << " phinorm " << phi[i_pf].norm() << std::endl;
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
			double a_rational = RA.alpha_inv_lo[n_rational][i_s+1];
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
		} // end of loop over shifts
	} // end of loop over pseudo fermion flavours
	// output fermion force norms
	
    auto timer_stop = std::chrono::high_resolution_clock::now();
    auto timer_count = std::chrono::duration_cast<std::chrono::seconds>(timer_stop-timer_start).count();
	std::cout << "#RHMC_ForceCGRuntime " << timer_count << std::endl;
	std::cout << "#RHMC_ForceCGIter " << iter << std::endl;

	return iter;
}

int rhmc::force_fermion_block (field<gauge> &force, field<gauge> &U, dirac_op& D) {
	//std::cout << "#RHMC force: n_f=" << params.n_f << ", n_pf=" << params.n_pf << ", n_rational/2=" << n_rational/2 << ", n_shifts lo=" << RA.beta_inv_lo[n_rational/2].size() << std::endl;
    auto timer_start = std::chrono::high_resolution_clock::now();
	int n_shifts = RA.beta_inv_lo[n_rational].size();
	std::vector< field<block_fermion> > chi(n_shifts, block_phi);
	field<block_fermion> psi (block_phi.grid, block_eo_storage_o);

	int iter = SBCGrQ (chi, block_phi, U, RA.beta_inv_lo[n_rational], D, params.MD_eps);
	for(int i_s=0; i_s<n_shifts; ++i_s) {
		// chi_{i_s} = [(DD^dagger + beta_{i_s}]^-1 phi[i_pf]
		//std::cout << "ipf " << i_pf << "cum iter " << iter << " inv0 norm " << chi[0].norm() << " Unorm " << U.norm() << " phinorm " << phi[i_pf].norm() << std::endl;
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

		for(int i_pf=0; i_pf<params.n_pf; ++i_pf) {
			// usual force term expressions
			// but with multiplicative alpha_inv factor:
			double a_rational = RA.alpha_inv_lo[n_rational][i_s+1];
			D.apply_eta_bcs_to_U(U);
			SU3_Generators T;
			if(params.EE) {
				// for even-even version half of the terms are zero:
				// even ix: ix = ix_e
				#pragma omp parallel for
				for(int ix=0; ix<chi[i_s].V; ++ix) {
					for(int mu=0; mu<4; ++mu) {
						for(int a=0; a<8; ++a) {
							double Fa = a_rational * (chi[i_s][ix].col(i_pf).dot(T[a] * U[ix][mu] * psi.up(ix,mu).col(i_pf))).imag();
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
							double Fa = a_rational * (chi[i_s].up(ix,mu).col(i_pf).dot(U[ix][mu].adjoint() * T[a] * psi[ix_o].col(i_pf))).imag();
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
						double Fa = a_rational * chi[i_s][ix].col(i_pf).dot(T[a] * mu_I_plus_factor * U[ix][0] * psi.up(ix,0).col(i_pf)).imag();
						Fa += a_rational * chi[i_s].up(ix,0).col(i_pf).dot(U[ix][0].adjoint() * mu_I_minus_factor * T[a] * psi[ix].col(i_pf)).imag();
						force[ix][0] -= Fa * T[a];
					}
					for(int mu=1; mu<4; ++mu) {
						for(int a=0; a<8; ++a) {
							double Fa = a_rational * (chi[i_s][ix].col(i_pf).dot(T[a] * U[ix][mu] * psi.up(ix,mu).col(i_pf))).imag();
							Fa += a_rational * (chi[i_s].up(ix,mu).col(i_pf).dot(U[ix][mu].adjoint() * T[a] * psi[ix].col(i_pf))).imag();
							force[ix][mu] -= Fa * T[a];
						}
					}
				}
			}
			D.remove_eta_bcs_from_U(U);
		} // end of loop over shifts
	} // end of loop over pseudo fermion flavours
    auto timer_stop = std::chrono::high_resolution_clock::now();
    auto timer_count = std::chrono::duration_cast<std::chrono::seconds>(timer_stop-timer_start).count();
	std::cout << "#RHMC_ForceBCGRuntime " << timer_count << std::endl;
	std::cout << "#RHMC_ForceBCGIter " << iter << std::endl;

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
