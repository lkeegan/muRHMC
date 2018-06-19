#ifndef LKEEGAN_MURHMC_DIRAC_OP_H
#define LKEEGAN_MURHMC_DIRAC_OP_H
#include "4d.hpp"
#include "su3.hpp"
#include "Eigen3/Eigen/Eigenvalues"
#include "omp.h"

// staggered space-dependent gamma matrices
// for now stored as 5x doubles per site but they are just +/- signs, and g[0] is just + everywhere
// g[4] is gamma_5
class gamma_matrices {
private:
	double g_[5];

public:
	double& operator[](int i) { return g_[i]; }
	double operator[](int i) const { return g_[i]; }
};

// Staggered dirac operator
class dirac_op {
private:
	// Construct staggered eta (gamma) matrices
	void construct_eta(field<gamma_matrices>& eta, const lattice& grid);

public:
	double mass;
	double mu_I;
	field<gamma_matrices> eta;
	bool ANTI_PERIODIC_BCS = true;
	bool GAUGE_LINKS_INCLUDE_ETA_BCS = false;

	dirac_op (const lattice& grid, double mass, double mu_I = 0.0);

	explicit dirac_op (const lattice& grid) : dirac_op::dirac_op(grid, 0.0, 0.0) {}

	void apbcs_in_time (field<gauge>& U) const;

	// Applies eta matrices and apbcs in time to the gauge links U
	// Required before and after using EO versions of dirac op
	// Toggles flag GAUGE_LINKS_INCLUDE_ETA_BCS
	void apply_eta_bcs_to_U (field<gauge>& U);
	void remove_eta_bcs_from_U (field<gauge>& U);

	// Applies staggered gamma_5 operator to fermion field (in place).
	template<int N>
	void gamma5 (field< block_fermion_matrix<N> >& phi) const {
		for(int ix=0; ix<phi.V; ++ix) {
			phi[ix] *= eta[ix][4];
		}
	}

	// massless mu_I-less off-diagonal parts of dirac operator
	template<int N>
	void D_eo (field< block_fermion_matrix<N> >& lhs, const field< block_fermion_matrix<N> >& rhs, const field<gauge>& U) const {
		// massless even_odd part of dirac op (assumes mu_I=0):
		// also assumes that gauge links contain eta matrices and bcs
		// rhs is only defined for odd sites, lhs for even sites
		#pragma omp parallel for
		for(int ix=0; ix<lhs.V; ++ix) {
			lhs[ix].setZero();
			for(int mu=0; mu<4; ++mu) {
				lhs[ix].noalias() += 0.5 * U[ix][mu] * rhs.up(ix,mu); 
				lhs[ix].noalias() += -0.5 * U.dn(ix,mu)[mu].adjoint() * rhs.dn(ix,mu);
			}
		}	
	}
	template<int N>
	void D_oe (field< block_fermion_matrix<N> >& lhs, const field< block_fermion_matrix<N> >& rhs, const field<gauge>& U) const {
		// loop over odd ix_o = ix + V:
		#pragma omp parallel for
		for(int ix_o=0; ix_o<lhs.V; ++ix_o) {
			lhs[ix_o].setZero();
			// true ix has V offset from ix_o:
			int ix = ix_o + lhs.V; 
			for(int mu=0; mu<4; ++mu) {
				lhs[ix_o].noalias() += 0.5 * U[ix][mu] * rhs.up(ix,mu); 
				lhs[ix_o].noalias() += -0.5 * U.dn(ix,mu)[mu].adjoint() * rhs.dn(ix,mu);
			}
		}
	}

	// Staggered massive Dirac operator D with isospin chemical potential mu_I: lhs = D(rhs)
	template<int N>
	void D (field< block_fermion_matrix<N> >& lhs, const field< block_fermion_matrix<N> >& rhs, field<gauge>& U) {
		apply_eta_bcs_to_U(U);
		double mu_I_plus_factor = exp(0.5 * mu_I);
		double mu_I_minus_factor = exp(-0.5 * mu_I);
		// default static scheduling, with N threads, split loop into N chunks, one per thread 
		////#pragma omp parallel for
		for(int ix=0; ix<rhs.V; ++ix) {
			lhs[ix] = mass * rhs[ix];
			// mu=0 terms have extra chemical potential isospin factors exp(+-\mu_I/2):
			// NB eta[ix][0] is just 1 so dropped from this expression
			lhs[ix].noalias() += 0.5 * mu_I_plus_factor * U[ix][0] * rhs.up(ix,0);
			lhs[ix].noalias() += -0.5 * mu_I_minus_factor * U.dn(ix,0)[0].adjoint() * rhs.dn(ix,0);
			for(int mu=1; mu<4; ++mu) {
				lhs[ix].noalias() += 0.5 * U[ix][mu] * rhs.up(ix,mu); 
				lhs[ix].noalias() += -0.5 * U.dn(ix,mu)[mu].adjoint() * rhs.dn(ix,mu);
			}
		}
		remove_eta_bcs_from_U(U);
	}

	// Hermitian D D^{\dagger} operator: lhs = DDdagger(rhs)
	// If lhs and rhs are EVEN_ONLY fermion fields then this is the even-even sub-block of D Ddagger for mu_I=0 
	template<int N>
	void DDdagger (field< block_fermion_matrix<N> >& lhs, const field< block_fermion_matrix<N> >& rhs, field<gauge>& U, field< block_fermion_matrix<N> >& tmp, bool LEAVE_ETA_BCS_IN_GAUGE_FIELD = false) {
		// NOTE: tmp must be odd if EO sub-block used
		if(lhs.eo_storage == field< block_fermion_matrix<N> >::EVEN_ONLY && rhs.eo_storage == field< block_fermion_matrix<N> >::EVEN_ONLY) {
			// if lhs and rhs are both even-only fields,
			// then use even-even sub block of operator (and assume mu_I=0)
			apply_eta_bcs_to_U(U);
			D_oe(tmp, rhs, U);
			D_eo(lhs, tmp, U);
			lhs.scale_add(-1.0, mass*mass, rhs);
			if (!LEAVE_ETA_BCS_IN_GAUGE_FIELD) {
				remove_eta_bcs_from_U(U);			
			}
		} else {
			// otherwise do full operator with non-zero mu_I
		    // D^dagger(mass, mu_I) = -D(-mass, -mu_I)
		    mass = -mass;
		    mu_I = -mu_I;
			D(tmp, rhs, U);
			tmp *= -1.0;
		    mass = -mass;
		    mu_I = -mu_I;
			D(lhs, tmp, U);
		}
	}

	// As above, but allocates its own temporary fermion field storage
	template<int N>
	void DDdagger (field< block_fermion_matrix<N> >& lhs, const field< block_fermion_matrix<N> >& rhs, field<gauge>& U, bool LEAVE_ETA_BCS_IN_GAUGE_FIELD = false) {
		if(lhs.eo_storage == field< block_fermion_matrix<N> >::EVEN_ONLY && rhs.eo_storage == field< block_fermion_matrix<N> >::EVEN_ONLY) {
			// if lhs and rhs are both even-only fields,
			// then use even-even sub block of operator (and assume mu_I=0)
		    field< block_fermion_matrix<N> > tmp_o(rhs.grid, field< block_fermion_matrix<N> >::ODD_ONLY);
		    DDdagger(lhs, rhs, U, tmp_o, LEAVE_ETA_BCS_IN_GAUGE_FIELD);
		} else {
			// otherwise do full operator with non-zero mu_I
		    field< block_fermion_matrix<N> > tmp(rhs.grid);
		    DDdagger(lhs, rhs, U, tmp, LEAVE_ETA_BCS_IN_GAUGE_FIELD);
		}
	}

	// Returns eigenvalues of Dirac op
	// Explicitly constructs dense (3*VOL)x(3*VOL) matrix Dirac op and finds all eigenvalues 
	Eigen::MatrixXcd D_eigenvalues (field<gauge>& U);

	// Same for DDdagger, but much faster since we can use a hermitian solver.
	Eigen::MatrixXcd DDdagger_eigenvalues (field<gauge>& U);

	// Returns phase angle of determinant of D by explicitly constructing and diagonalising D
	double D_phase_angle (field<gauge>& U);

	// Returns real part of isospin density by explicitly constructing and diagonalising DDdagger
	double pion_susceptibility_exact (field<gauge>& U);

	// explicitly construct dirac op as dense (3*VOL)x(3*VOL) matrix
	Eigen::MatrixXcd D_dense_matrix (field<gauge>& U);

	// explicitly construct (dD/d\mu) as dense (3xVOL)x(3xVOL) matrix
	Eigen::MatrixXcd dD_dmu_dense_matrix (field<gauge>& U) const;

	// in-place chebyshev polynomial of DDdag acting on fermion or block_fermion fields
	// Uses recursive formula from hep-lat/0512021
	// c_{n+1}= 2*z*c_{n} - c_{n-1}
	// where z = ((v+u) - 2 DDdag) / (u - v)
	// c_0 = 1, c_1 = z
	template<typename fermion_type>
	void chebyshev (int k, double u, double v, field<fermion_type>& X, field<gauge>& U) {
		field<fermion_type> c_minus2 (X);
		field<fermion_type> c_minus1 (X);
		field<fermion_type> c_minus0 (X);
		double norm = (v + u) / (v - u);
		double inv_vmu = 1.0 / (v - u);
		// c_0 = x
		c_minus1 = X;
		// c_1 = z(x) = norm x - inv_vmu DDdag x
		DDdagger (c_minus0, c_minus1, U);
		c_minus0.scale_add(-inv_vmu, norm, c_minus1);
		// c_k = 2 z(c_{k-1}) - c_{k-2}
		for(int i_k=1; i_k<k; ++i_k) {
			// relabel previous c's
			c_minus2 = c_minus1;
			c_minus1 = c_minus0;
			// calculate current c
			DDdagger (c_minus0, c_minus1, U);
			c_minus0.scale_add(-inv_vmu, norm, c_minus1);
			c_minus0.scale_add(2.0, -1.0, c_minus2);
		}
		X = c_minus0;
	}		

	// returns lower bound on largest eigenvalue of DDdagger operator
	// for use in constructing rational approximations
	double largest_eigenvalue_bound (field<gauge>& U, field<fermion>::eo_storage_options EO_STORAGE = field<fermion>::EVEN_ONLY, double rel_err = 0.01);
};
 
#endif //LKEEGAN_MURHMC_DIRAC_OP_H