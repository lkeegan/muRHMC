#ifndef LATTICE_DIRAC_OP_H
#define LATTICE_DIRAC_OP_H
#include "4d.hpp"
#include "su3.hpp"

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
	void gamma5 (field<fermion>& phi) const;

	// massless mu_I-less off-diagonal parts of dirac operator
	void D_eo (field<fermion>& lhs, const field<fermion>& rhs, const field<gauge>& U) const;
	void D_oe (field<fermion>& lhs, const field<fermion>& rhs, const field<gauge>& U) const;

	// Staggered massive Dirac operator D with isospin chemical potential mu_I: lhs = D(rhs)
	void D (field<fermion>& lhs, const field<fermion>& rhs, field<gauge>& U);

	// Hermitian D D^{\dagger} operator: lhs = DDdagger(rhs)
	// If lhs and rhs are EVEN_ONLY fermion fields then this is the even-even sub-block of D Ddagger for mu_I=0 
	void DDdagger (field<fermion>& lhs, const field<fermion>& rhs, field<gauge>& U);

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

	// in-place chebyshev polynomial of DDdag acting on vector of fermion fields
	// See appendix A of hep-lat/0512021 for details of implementation
	void chebyshev (int k, double u, double v, std::vector<field<fermion>>& X, field<gauge>& U);

	// returns lower bound on largest eigenvalue of DDdagger operator
	// for use in constructing rational approximations
	double largest_eigenvalue_bound (field<gauge>& U, double rel_err = 0.01, field<fermion>::eo_storage_options EO_STORAGE = field<fermion>::EVEN_ONLY);
};
 
#endif //LATTICE_DIRAC_OP_H