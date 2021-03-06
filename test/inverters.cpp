#include "catch.hpp"
#include "su3.hpp"
#include "4d.hpp"
#include "hmc.hpp"
#include "dirac_op.hpp"
#include "inverters.hpp"
#include "io.hpp"
#include <iostream>

// default lattice size for tests
constexpr int L = 2;
// tolerance for values that should be zero
constexpr double EPS = 5.e-14;

TEST_CASE( "CG inversions of isospin (D+m)(D+m)^dagger", "[inverters]") {

	double eps = 1.e-9;
	int N_shifts = 3;
	std::vector<double> shifts = {0.2986, 0.421, 0.88};

	rhmc_params rhmc_pars;
	rhmc_pars.mass = 0.292;
	rhmc_pars.mu_I = 0.0157;
	rhmc_pars.seed = 123;

	// Loop over 3 setups: LEXI (default), EO, EO w/EO preconditioning
	field<fermion>::eo_storage_options eo_storage_e = field<fermion>::FULL;
	field<fermion>::eo_storage_options eo_storage_o = field<fermion>::FULL;
	field<block_fermion>::eo_storage_options block_eo_storage_e = field<block_fermion>::FULL;
	field<block_fermion>::eo_storage_options block_eo_storage_o = field<block_fermion>::FULL;
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
			block_eo_storage_e = field<block_fermion>::EVEN_ONLY;			
			block_eo_storage_o = field<block_fermion>::ODD_ONLY;			
			lt = "EO_EVEN_ONLY";
		}

		lattice grid (L, isEO);
		field<gauge> U (grid);
		rhmc rhmc (rhmc_pars, grid);
		rhmc.random_U(U, 0.2);
		dirac_op D (grid, rhmc_pars.mass, rhmc_pars.mu_I);

		field<fermion> Ax (grid, eo_storage_e);
		field<block_fermion> block_Ax (grid, block_eo_storage_e);

		field<fermion> B(Ax);
		field<block_fermion> block_B(block_Ax);

		rhmc.gaussian_fermion(B);
		rhmc.gaussian_fermion(block_B);

		std::vector< field<fermion> > X;
		for(int i=0; i<N_shifts; ++i) {
			X.push_back(B);
		}

		std::vector< field<block_fermion> > block_X;
		for(int i=0; i<N_shifts; ++i) {
			block_X.push_back(block_B);
		}

		SECTION( std::string("CG_") + lt ) {
			int iter = cg(X[0], B, U, D, eps);
			D.DDdagger(Ax, X[0], U);
			Ax -= B;
			double res = Ax.norm()/B.norm();
			INFO("Lattice-type: " << lattice_type << "CG: eps = " << eps << "\t iterations = " << iter << "\t residual = " << res);
			REQUIRE( res < 1e2 * eps );
		}

		SECTION( std::string("CG-singleshift_") + lt ) {
			int iter = cg_singleshift(X[0], B, U, shifts[0], D, eps);
			D.DDdagger(Ax, X[0], U);
			Ax.add(shifts[0], X[0]);
			Ax -= B;
			double res = Ax.norm()/B.norm();
			INFO("Lattice-type: " << lattice_type << "\tCG-singleshift: eps = " << eps << "\t iterations = " << iter << "\t residual = " << res);
			REQUIRE( res < 1e2 * eps );
		}

		SECTION( std::string("CG-multishift_") + lt ) {
			int iter = cg_multishift(X, B, U, shifts, D, eps);
			for(int i=0; i<N_shifts; ++i) {
				D.DDdagger(Ax, X[i], U);
				Ax.add(shifts[i], X[i]);
				Ax -= B;
				double res = Ax.norm()/B.norm();
				INFO("Lattice-type: " << lattice_type << "\tCG-multishift: eps = " << eps << "\t iterations = " << iter << "\tshift = " << shifts[i] << "\t residual = " << res);
				REQUIRE( res < 1e2 * eps );				
			}
		}

		SECTION( std::string("BCG(A)(dQ/dQA)(rQ)_") + lt ) {
			for (bool A: {false, true}) {
				for (bool dQ: {false, true}) {
					for (bool dQA: {false, true}) {
						for (bool rQ: {false, true}) {
							// can't have both dQ and dQA
							if(!(dQ && dQA)) {
								int iter = cg_block(block_X[0], block_B, U, D, eps, A, dQ, dQA, rQ);
								D.DDdagger(block_Ax, block_X[0], U);
								block_Ax -= block_B;
								double residual = Ax.norm();
								CAPTURE(A);
								CAPTURE(dQ);
								CAPTURE(dQA);
								CAPTURE(rQ);
								INFO("Lattice-type: " << lattice_type << "\tBCG[" << N_rhs << "]: eps = " << eps << "\t iterations = " << iter << "\t residual = " << residual);
								REQUIRE( residual < 1e2 * eps );
							}
						}
					}
				}
			}
		}

		SECTION( std::string("SBCGrQ_") + lt ) {
			int iter = SBCGrQ(block_X, block_B, U, shifts, D, eps);
			for(int i_s=0; i_s<N_shifts; ++i_s) {
				D.DDdagger(block_Ax, block_X[i_s], U);
				block_Ax.add(shifts[i_s], block_X[i_s]);
				block_Ax -= block_B;
				double residual = block_Ax.norm();
				INFO("Lattice-type: " << lattice_type << "\tSBCGrQ[" << i_s << "][" << N_rhs << "]: eps = " << eps << "\t iterations = " << iter << "\t residual = " << residual);
				REQUIRE( residual < 1e2 * eps );
			}
		}

		SECTION( std::string("thinQR_") + lt ) {
			block_X[0] = block_B;
			block_matrix R = block_matrix::Zero();
			thinQR(block_B, R);

			// B should be orthornormal
			Eigen::MatrixXcd Bdot = block_matrix::Zero();
			for(int ix=0; ix<block_B.V; ++ix) {
				for(int i=N_rhs-1; i>=0; --i) {
					for(int j=N_rhs-1; j>=0; --j) {
						Bdot(i,j) += block_B[ix].col(i).dot(block_B[ix].col(j));
					}
				}
			}
			REQUIRE( (Bdot - block_matrix::Identity()).norm() < EPS );

			// R should be upper triangular
			double norm = 0;
			for(int i=0; i<N_rhs; ++i) {
				for(int j=i+1; j<i; ++j) {
					norm += std::abs(R(j, i));
				}
			}
			REQUIRE( norm < EPS );

			// BR should reconstruct original B = X[0]
			block_Ax.setZero();
			for(int ix=0; ix<block_B.V; ++ix) {
				for(int i=0; i<N_rhs; ++i) {
					for(int j=0; j<=i; ++j) {
						block_Ax[ix].col(i) += R(j,i) * block_B[ix].col(j);
					}
					//block_Ax[ix].col(i) -= X[0][ix].col(i);
				}
			}
			block_Ax -= block_X[0];
			REQUIRE( block_Ax.norm() < EPS );
		}
	}
}


TEST_CASE( "U perturbations: CG inversions of isospin (D+m)(D+m)^dagger", "[inverters]") {

	double eps = 1.e-5;
	int N_shifts = 3;
	std::vector<double> shifts = {0.005, 0.010, 0.015};

	rhmc_params rhmc_pars;
	rhmc_pars.mass = 0.292;
	rhmc_pars.mu_I = 0.0557;
	rhmc_pars.seed = 123;

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

		lattice grid (L, isEO);
		field<gauge> U (grid);
		field<gauge> Uprime (grid);
		field<gauge> P (grid);
		field<gauge> dP (grid);
		rhmc rhmc (rhmc_pars, grid);

		rhmc.gaussian_P(P);
		rhmc.gaussian_P(dP);
		for(int ix=0; ix<Uprime.V; ++ix) {
			for(int mu=0; mu<4; ++mu) {
				U[ix][mu] = exp_ch(std::complex<double> (0.0, 0.2) * P[ix][mu]);
				Uprime[ix][mu] = exp_ch(std::complex<double> (0.0, 0.2) * P[ix][mu] + (std::complex<double> (0.0, 1e-14) * dP[ix][mu]));
			}
		}

		dirac_op D (grid, rhmc_pars.mass, rhmc_pars.mu_I);

		field<fermion> Ax (grid, eo_storage_e);
		field<fermion> Aprimex (grid, eo_storage_e);

		field<fermion> B (grid, eo_storage_e);
		rhmc.gaussian_fermion(B);

		std::vector< field<fermion> > X;
		std::vector< field<fermion> > Xprime;
		for(int i=0; i<N_shifts; ++i) {
			X.push_back(B);
			Xprime.push_back(B);
		}

		SECTION( std::string("U: CG_") + lt ) {
			int iter = cg(X[0], B, U, D, eps);
			D.DDdagger(Ax, X[0], U);
			Ax -= B;
			iter = cg(Xprime[0], B, Uprime, D, eps);
			D.DDdagger(Aprimex, Xprime[0], Uprime);
			Aprimex -= B;
			double res = Ax.norm()/B.norm();
			double resprime = Aprimex.norm()/B.norm();
			Xprime[0] -= X[0];
			double deltaX = Xprime[0].norm()/X[0].norm();
			Uprime -= U;
			double deltaU = Uprime.norm()/U.norm();			
			CAPTURE( deltaU );
			CAPTURE( deltaX );
			CAPTURE( eps );
			INFO("Lattice-type: " << lattice_type << "CG: eps = " << eps << "\t iterations = " << iter << "\t residual = " << res);
			REQUIRE( res < 1e2 * eps );
		}
/*
		SECTION( std::string("CG-singleshift_") + lt ) {
			int iter = cg_singleshift(X[0][0], B[0], U, shifts[0], D, eps);
			D.DDdagger(Ax, X[0][0], U);
			Ax.add(shifts[0], X[0][0]);
			Ax -= B[0];
			double res = Ax.norm()/B[0].norm();
			INFO("Lattice-type: " << lattice_type << "\tCG-singleshift: eps = " << eps << "\t iterations = " << iter << "\t residual = " << res);
			REQUIRE( res < 1e2 * eps );
		}

		SECTION( std::string("CG-multishift_") + lt ) {
			int iter = cg_multishift(X[0], B[0], U, shifts, D, eps);
			for(int i=0; i<N_shifts; ++i) {
				D.DDdagger(Ax, X[0][i], U);
				Ax.add(shifts[i], X[0][i]);
				Ax -= B[0];
				double res = Ax.norm()/B[0].norm();
				INFO("Lattice-type: " << lattice_type << "\tCG-multishift: eps = " << eps << "\t iterations = " << iter << "\tshift = " << shifts[i] << "\t residual = " << res);
				REQUIRE( res < 1e2 * eps );				
			}
		}
		*/
	}
}