#include "catch.hpp"
#include "su3.hpp"
#include "4d.hpp"
#include "hmc.hpp"
#include "dirac_op.hpp"
#include "inverters.hpp"
#include "io.hpp"
#include <iostream>

constexpr double EPS = 5.e-14;

TEST_CASE( "CG inversions of isospin (D+m)(D+m)^dagger", "[inverters]") {

	double eps = 1.e-9;
	int N_rhs = 3;
	int N_shifts = N_rhs;
	std::vector<double> shifts = {0.12986, 0.421, 0.88};

	hmc_params hmc_pars = {
		5.4, 	// beta
		0.292, 	// mass
		0.0557, // mu_I
		1.0, 	// tau
		3, 		// n_steps
		1.e-6,	// MD_eps
		1234,	// seed
		false, 	// EE: only simulate even-even sub-block (requires mu_I=0)
		false,	// constrained HMC (fixed allowed range for pion susceptibility)
		3.0, 	// suscept_central
		0.05	// suscept_eps
	};

	// Loop over 3 setups: LEXI (default), EO, EO w/EO preconditioning
	field<fermion>::eo_storage_options eo_storage_e = field<fermion>::FULL;
	field<fermion>::eo_storage_options eo_storage_o = field<fermion>::FULL;
	std::string lt ("LEXI_FULL");
	bool isEO = false;
	for (int lattice_type : {0, 1, 2}) {
		if(lattice_type==1) {
			isEO = true;
			lt = "EO_FULL";
		} else if(lattice_type==2) {
			isEO = true;
			eo_storage_e = field<fermion>::EVEN_ONLY;			
			eo_storage_o = field<fermion>::ODD_ONLY;			
			lt = "EO_EVEN_ONLY";
		}

		lattice grid (4, isEO);
		field<gauge> U (grid);
		hmc rhmc (hmc_pars);
		rhmc.random_U(U, 10.0);
		dirac_op D (grid, hmc_pars.mass, hmc_pars.mu_I);

		field<fermion> Ax (grid, eo_storage_e);

		std::vector< field<fermion> > B;
		for(int i=0; i<N_rhs; ++i) {
			rhmc.gaussian_fermion(Ax);
			B.push_back(Ax);
		}

		std::vector< std::vector< field<fermion> > > X;
		for(int i=0; i<N_shifts; ++i) {
			X.push_back(B);
		}

		SECTION( std::string("CG_") + lt ) {
			int iter = cg(X[0][0], B[0], U, D, eps);
			D.DDdagger(Ax, X[0][0], U);
			Ax -= B[0];
			double res = Ax.norm()/B[0].norm();
			INFO("Lattice-type: " << lattice_type << "CG: eps = " << eps << "\t iterations = " << iter << "\t residual = " << res);
			REQUIRE( res == Approx(0).margin(5e-13 + 50.0*eps) );
		}

		SECTION( std::string("CG-singleshift_") + lt ) {
			int iter = cg_singleshift(X[0][0], B[0], U, shifts[0], D, eps);
			D.DDdagger(Ax, X[0][0], U);
			Ax.add(shifts[0], X[0][0]);
			Ax -= B[0];
			double res = Ax.norm()/B[0].norm();
			INFO("Lattice-type: " << lattice_type << "\tCG-singleshift: eps = " << eps << "\t iterations = " << iter << "\t residual = " << res);
			REQUIRE( res == Approx(0).margin(5e-13 + 50.0*eps) );
		}

		SECTION( std::string("CG-multishift_") + lt ) {
			int iter = cg_multishift(X[0], B[0], U, shifts, D, eps);
			for(int i=0; i<N_shifts; ++i) {
				D.DDdagger(Ax, X[0][i], U);
				Ax.add(shifts[i], X[0][i]);
				Ax -= B[0];
				double res = Ax.norm()/B[0].norm();
				INFO("Lattice-type: " << lattice_type << "\tCG-multishift: eps = " << eps << "\t iterations = " << iter << "\tshift = " << shifts[i] << "\t residual = " << res);
				REQUIRE( res == Approx(0).margin(5e-13 + 50.0*eps) );				
			}
		}

		SECTION( std::string("BCG(A)(dQ/dQA)(rQ)_") + lt ) {
			for (bool A: {false, true}) {
				for (bool dQ: {false, true}) {
					for (bool dQA: {false, true}) {
						for (bool rQ: {false, true}) {
							// can't have both dQ and dQA
							if(!(dQ && dQA)) {
								int iter = cg_block(X[0], B, U, D, eps, A, dQ, dQA, rQ, B[0]);
								for(int i=0; i<N_rhs; ++i) {
									D.DDdagger(Ax, X[0][i], U);
									Ax -= B[i];
									double residual = Ax.norm();
									CAPTURE(A);
									CAPTURE(dQ);
									CAPTURE(dQA);
									CAPTURE(rQ);
									INFO("Lattice-type: " << lattice_type << "\tBCG[" << i << "]: eps = " << eps << "\t iterations = " << iter << "\t residual = " << residual);
									REQUIRE( residual == Approx(0).margin(5e-13 + 50.0*eps) );
								}
							}
						}
					}
				}
			}
		}

		SECTION( std::string("SBCGrQ_") + lt ) {
			int iter = SBCGrQ(X, B, U, shifts, D, eps);
			for(int i_s=0; i_s<N_shifts; ++i_s) {
				for(int i_rhs=0; i_rhs<N_rhs; ++i_rhs) {
					D.DDdagger(Ax, X[i_s][i_rhs], U);
					Ax.add(shifts[i_s], X[i_s][i_rhs]);
					Ax -= B[i_rhs];
					double residual = Ax.norm();
					INFO("Lattice-type: " << lattice_type << "\tSBCGrQ[" << i_s << "][" << i_rhs << "]: eps = " << eps << "\t iterations = " << iter << "\t residual = " << residual);
					REQUIRE( residual == Approx(0).margin(5e-13 + 50.0*eps) );
				}
			}
		}

		SECTION( std::string("thinQR_") + lt ) {
			X[0] = B;
			Eigen::MatrixXcd R = Eigen::MatrixXcd::Zero(N_rhs, N_rhs);
			thinQR(B, R);

			// B should be orthornormal
			for(int i=N_rhs-1; i>=0; --i) {
				for(int j=N_rhs-1; j>=0; --j) {
					if(i==j) {
						REQUIRE( std::abs(B[i].dot(B[j])) == Approx(1) );
					}
					else {
						REQUIRE( std::abs(B[i].dot(B[j])) < EPS );				
					}
				}
			}

			// R should be upper triangular
			double norm = 0;
			for(int i=0; i<N_rhs; ++i) {
				for(int j=i+1; j<i; ++j) {
					norm += std::abs(R(j, i));
				}
			}
			REQUIRE( norm < EPS );

			// BR should reconstruct original B = X[0]
			for(int i=0; i<N_rhs; ++i) {
				field<fermion> M_reconstructed (grid);
				Ax.setZero();
				for(int j=0; j<=i; ++j) {
					Ax.add(R(j,i), B[j]);
				}
				Ax -= X[0][i];
				REQUIRE( Ax.norm() < EPS );
			}
		}
	}
}
