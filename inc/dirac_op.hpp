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

	explicit dirac_op (const lattice& grid);

	// Applies staggered gamma_5 operator to fermion field (in place).
	void gamma5 (field<fermion> &phi) const;

	// Staggered massive Dirac operator D: lhs = D(rhs)
	void D (field<fermion> &lhs, const field<fermion> &rhs, const field<gauge>& U, double m) const;

	// Hermitian Dirac operator squared: (D+m)(D+m)^\dagger == (g5 D) (g5 D) + m^2
	void DDdagger (field<fermion> &lhs, const field<fermion> &rhs, const field<gauge>& U, double m) const;

	// CG inversion of (g5 D g5 D + m^2) x = b: given b solves for x
	// returns number of times Dirac operator was called
	int cg(field<fermion>& x, const field<fermion>& b, const field<gauge>& U, double m, double eps) const;
};
 
#endif //LATTICE_DIRAC_OP_H