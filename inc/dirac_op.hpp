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
	const lattice& grid;
	field<gamma_matrices> eta;
	bool ANTI_PERIODIC_BCS = true;

	explicit dirac_op (const lattice& grid);

	// Applies staggered gamma_5 operator to fermion field (in place).
	void gamma5 (field<fermion>& phi) const;

	// Give timelike gauge links or force links at x_0=L0-1 boundary a minus sign
	// If called before and after applying Dirac op, or calculating fermionic force term,
	// this gives antiperiodic bcs in time direction to the fermions
	void apbs_in_time (field<gauge>& U) const;

	// Staggered massive Dirac operator D with isospin chemical potential mu_I: lhs = D(rhs)
	void D (field<fermion>& lhs, const field<fermion>& rhs, field<gauge>& U, double m, double mu_I) const;

	// Hermitian D D^{\dagger} operator: lhs = DDdagger(rhs)
	void DDdagger (field<fermion>& lhs, const field<fermion>& rhs, field<gauge>& U, double m, double mu_I);

	// Returns eigenvalues of Dirac op
	// Explicitly constructs dense (3*VOL)x(3*VOL) matrix Dirac op and finds all eigenvalues 
	Eigen::MatrixXcd D_eigenvalues (field<gauge>& U, double mass, double mu_I) const;

	// Same for DDdagger, but much faster since we can use a hermitian solver.
	Eigen::MatrixXcd DDdagger_eigenvalues (field<gauge>& U, double mass, double mu_I) const;

	// Returns phase angle of determinant of D by explicitly constructing and diagonalising D
	double D_phase_angle (field<gauge>& U, double mass, double mu_I) const;

	// Returns real part of isospin density by exlpicitly constructing and diagonalising DDdagger
	double pion_susceptibility_exact (field<gauge>& U, double mass, double mu_I) const;

	// explicitly construct dirac op as dense (3*VOL)x(3*VOL) matrix
	Eigen::MatrixXcd D_dense_matrix (field<gauge>& U, double mass, double mu_I) const;

	// CG inversion of D D^{\dagger} x = b: given b solves for x
	// returns number of times Dirac operator was called
	int cg(field<fermion>& x, const field<fermion>& b, field<gauge>& U, double m, double mu_I, double eps);

	// CG multishift inversion of (D D^{\dagger} + sigma_a^2) x_a = b: given b solves for x_a for each shift sigma_a
	// returns number of times Dirac operator was called
	// without isospin sigma is a shift in the mass parameter, so should take same number of inversions as
	// CG inverting DD^dagger with mass^2 = m^2 + sigma[0]^2
	int cg_multishift(std::vector<field<fermion>>& x, const field<fermion>& b, field<gauge>& U, double m, double mu_I, std::vector<double>& sigma, double eps);
};
 
#endif //LATTICE_DIRAC_OP_H