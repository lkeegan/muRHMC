#include "catch.hpp"
#include "su3.hpp"
#include "4d.hpp"
#include "hmc.hpp"
#include "rhmc.hpp"

constexpr double EPS = 9.e-14;

TEST_CASE( "moving forwards then backwards does nothing: 4^4 lattice of U[mu]", "[lattice]" ) {
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		for(int i=0; i<grid.V; ++i) {
			int i_0 = grid.iup(i, 0);
			int i_1 = grid.idn(i_0, 2);
			int i_2 = grid.iup(i_1, 2);
			int i_n = grid.idn(i_2, 0);
			INFO ( i << "-> " << i_0 << "-> " << i_1 << "-> " << i_2 << "-> " << i_n);
			REQUIRE ( i == i_n);
		}
	}
}

TEST_CASE( "indexing gives one index per site: 4^4 lattice of U[mu]", "[lattice]" ) {
	for(bool isEO : {false, true}) {
		// create 4^4 lattice with random U[mu] at each site
		lattice grid (4, isEO);
		std::vector<int> sites(grid.V, 0);
		for(int x3=0; x3<grid.L3; ++x3) {
			for(int x2=0; x2<grid.L2; ++x2) {
				for(int x1=0; x1<grid.L1; ++x1) {
					for(int x0=0; x0<grid.L0; ++x0) {
						int i = grid.index(x0, x1, x2, x3);
						int iup = grid.iup(i, 0);
						++sites[grid.iup(iup, 1)];
					}	
				}
			}
		}
		for(int i=0; i<grid.V; ++i) {
			REQUIRE ( sites[i] == 1);
		}
	}
}

TEST_CASE( "Nearest Neigbours: 4^4 lattice of U[mu]", "[lattice]" ) {
	for(bool isEO : {false, true}) {
		// create 4^4 lattice with random U[mu] at each site
		lattice grid (4, isEO);
		field<gauge> U (grid);
		hmc_params hmc_pars;
		hmc_pars.seed = 123;
		hmc rhmc (hmc_pars);
		rhmc.random_U (U, 10.0);
		// check up/dn neighbours consistent with 4-vec neighbours
		int i = grid.index(2, 3, 0, 2);
		REQUIRE( U.at(3, 3, 0, 2) == U.up(i, 0) );
		REQUIRE( U.at(1, 3, 0, 2) == U.dn(i, 0) );
		REQUIRE( U.at(2, 0, 0, 2) == U.up(i, 1) );
		REQUIRE( U.at(2, 2, 0, 2) == U.dn(i, 1) );
		REQUIRE( U.at(2, 3, 1, 2) == U.up(i, 2) );
		REQUIRE( U.at(2, 3, 3, 2) == U.dn(i, 2) );
		REQUIRE( U.at(2, 3, 0, 3) == U.up(i, 3) );
		REQUIRE( U.at(2, 3, 0, 1) == U.dn(i, 3) );

		i = grid.index(0, 3, 3, 0);
		REQUIRE( U.at(1, 3, 3, 0) == U.up(i, 0) );
		REQUIRE( U.at(3, 3, 3, 0) == U.dn(i, 0) );
		REQUIRE( U.at(0, 0, 3, 0) == U.up(i, 1) );
		REQUIRE( U.at(0, 2, 3, 0) == U.dn(i, 1) );
		REQUIRE( U.at(0, 3, 0, 0) == U.up(i, 2) );
		REQUIRE( U.at(0, 3, 2, 0) == U.dn(i, 2) );
		REQUIRE( U.at(0, 3, 3, 1) == U.up(i, 3) );
		REQUIRE( U.at(0, 3, 3, 3) == U.dn(i, 3) );
	}
}

TEST_CASE( "Nearest Neighbours: 12x6x3x2 lattice of U[mu]", "[lattice]" ) {
	for(bool isEO : {false, true}) {
		// create 12x6x3x2 lattice with random U[mu] at each site
		lattice grid (12, 6, 3, 2, isEO);
		field<gauge> U (grid);
		hmc_params hmc_pars;
		hmc_pars.seed = 123;
		hmc rhmc (hmc_pars);
		rhmc.random_U (U, 10.0);
		// check up/dn neighbours consistent with 4-vec neighbours
		int i = grid.index(7, 5, 1, 1);
		REQUIRE( U.at(8, 5, 1, 1) == U.up(i, 0) );
		REQUIRE( U.at(6, 5, 1, 1) == U.dn(i, 0) );
		REQUIRE( U.at(7, 0, 1, 1) == U.up(i, 1) );
		REQUIRE( U.at(7, 4, 1, 1) == U.dn(i, 1) );
		REQUIRE( U.at(7, 5, 2, 1) == U.up(i, 2) );
		REQUIRE( U.at(7, 5, 0, 1) == U.dn(i, 2) );
		REQUIRE( U.at(7, 5, 1, 0) == U.up(i, 3) );
		REQUIRE( U.at(7, 5, 1, 0) == U.dn(i, 3) );
	}
}

TEST_CASE( "Time-slices: 12x6x4x2 lattice of U[mu]", "[lattice]" ) {
	// create 12x6x4x2 lattice with random U[mu] at each site
	// compare av plaquette at each timeslice using it_ix index vs
	// with explicit slow indexing of time slice using "at"
	int L0 = 12;
	int L1 = 6;
	int L2 = 4;
	int L3 = 2;
	for(bool isEO : {false, true}) {
		lattice grid (L0, L1, L2, L3, isEO);
		field<gauge> U (grid);
		hmc_params hmc_pars;
		hmc_pars.seed = 123;
		hmc rhmc (hmc_pars);
		rhmc.random_U (U, 10.0);

		double plaq_slice_sum = 0;
		for(int x0=0; x0<U.L0; ++x0) {
			double plaq_slice = 0;
			double plaq_at = 0;
			// construct plaq using timeslice
			for(int ix3=0; ix3<U.VOL3; ++ix3) {
				plaq_slice += rhmc.plaq(U.it_ix(x0, ix3), U);
			}
			// construct plaq using grid debugging indexing
			for(int x1=0; x1<L1; ++x1) {
				for(int x2=0; x2<L2; ++x2) {
					for(int x3=0; x3<L3; ++x3) {
						plaq_at += rhmc.plaq(grid.index(x0, x1, x2, x3), U);
					}
				}
			}
			REQUIRE( plaq_slice == Approx(plaq_at) );
			plaq_slice_sum += plaq_slice;
		}
		// check that sum over time slices agrees with normal plaquette
		REQUIRE( plaq_slice_sum == Approx(static_cast<double>(6*3*U.V) * rhmc.plaq(U)) );
	}
}

TEST_CASE( "Converting EO<->LEXI indices", "[lattice]" ) {
	lattice grid (4);
	for(int i=0; i<grid.V; ++i) {
		int i_eo = grid.EO_from_LEXI(i);
		int i_lexi = grid.LEXI_from_EO(i_eo);
		INFO ( i << "-> " << i_eo << "-> " << i_lexi);
		REQUIRE ( i_eo < grid.V);
		REQUIRE ( i_lexi < grid.V);
		REQUIRE ( i_lexi == i);
	}
}

TEST_CASE( "x.squaredNorm() and x.dot(x) equivalent", "[4d]" ) {
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<fermion> chi (grid);
		hmc_params hmc_pars;
		hmc_pars.seed = 123;
		hmc rhmc (hmc_pars);
		rhmc.gaussian_fermion(chi);
		REQUIRE( chi.dot(chi).real() == Approx(chi.squaredNorm()) );
		}
}

TEST_CASE( "sqrt(x.squaredNorm()) and x.norm equivalent", "[4d]" ) {
	for(bool isEO : {false, true}) {
		lattice grid (4, isEO);
		field<fermion> chi (grid);
		hmc_params hmc_pars;
		hmc_pars.seed = 123;
		hmc rhmc (hmc_pars);
		rhmc.gaussian_fermion(chi);
		REQUIRE( chi.norm() == Approx(sqrt(chi.squaredNorm())) );
	}
}

#ifdef EIGEN_USE_MKL_ALL
TEST_CASE( "ZGEMM vs eigen: X beta", "[mkl]" ) {

	block_fermion A = block_fermion::Random();
	block_fermion B = block_fermion::Random();
	block_matrix alpha = block_matrix::Random();
	block_fermion A_mkl = A;

	A += B * alpha;
	std::complex<double> one (1.0, 0.0);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
        3, N_rhs, N_rhs, &one, B.data(), 3, alpha.data(), N_rhs, &one, A_mkl.data(), 3);

    CAPTURE(A);
    CAPTURE(A_mkl);
    REQUIRE ( (A - A_mkl).norm() < EPS );
}
#endif

#ifdef EIGEN_USE_MKL_ALL
TEST_CASE( "ZGEMM vs eigen U^dagger X", "[mkl]" ) {

	block_fermion A = block_fermion::Random();
	block_fermion B = block_fermion::Random();
	SU3mat U = SU3mat::Random();
	block_fermion A_mkl = A;

	A += 0.5 * U.adjoint() * B;
	std::complex<double> one (1.0, 0.0);
	std::complex<double> phalf (0.5, 0.0);
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 
		 3, N_rhs, 3, &phalf, U.data(), 3, B.data(), 3, &one, A_mkl.data(), 3);

    CAPTURE(A);
    CAPTURE(A_mkl);
    REQUIRE ( (A - A_mkl).norm() < EPS );
}
#endif

#ifdef EIGEN_USE_MKL_ALL
TEST_CASE( "add_mkl vs add", "[mkl]" ) {

	lattice grid (4, true);
	rhmc_params rhmc_pars;
	rhmc rhmc (rhmc_pars, grid);
	field<block_fermion> A (grid);
	field<block_fermion> B (grid);
	block_matrix alpha = block_matrix::Random();
	rhmc.gaussian_fermion(A);
	rhmc.gaussian_fermion(B);
	field<block_fermion> A_mkl (A);

	A.add(B, alpha);
	CAPTURE(A[0]);
	A_mkl.add_mkl(B, alpha);
	CAPTURE(A_mkl[0]);
	A -= A_mkl;
    REQUIRE ( A.norm() < EPS );
}
#endif

TEST_CASE( "add_eigen_bigmat vs add", "[mkl]" ) {

	lattice grid (4, true);
	rhmc_params rhmc_pars;
	rhmc rhmc (rhmc_pars, grid);
	field<block_fermion> A (grid);
	field<block_fermion> B (grid);
	block_matrix alpha = block_matrix::Random();
	rhmc.gaussian_fermion(A);
	rhmc.gaussian_fermion(B);

	// make big eigem matrix containg copy of A,B
	typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, N_rhs> fermion_block_field;
	fermion_block_field A_big = fermion_block_field::Zero(3*A.V, N_rhs);
	fermion_block_field B_big = fermion_block_field::Zero(3*A.V, N_rhs);
	for(int in=0; in<N_rhs; ++in) {
		for(int ix=0; ix<A.V; ++ix) {
			for(int ic=0; ic<3; ++ic) {
				A_big(ic + 3*ix, in) = A[ix](ic,in);
				B_big(ic + 3*ix, in) = B[ix](ic,in);
			}
		}
	}

	A.add(B, alpha);
	CAPTURE(A[0]);

	A_big.noalias() += B_big * alpha;

    // A -= A_big
	CAPTURE(A_big(0,0));
	for(int ix=0; ix<A.V; ++ix) {
		for(int ic=0; ic<3; ++ic) {
			for(int in=0; in<N_rhs; ++in) {
				A[ix](ic,in) -= A_big(ic + 3*ix, in);
			}
		}
	}

    REQUIRE ( A.norm() < EPS );
}

#ifdef EIGEN_USE_MKL_ALL
TEST_CASE( "add_mkl_bigmat vs add", "[mkl]" ) {

	lattice grid (4, true);
	rhmc_params rhmc_pars;
	rhmc rhmc (rhmc_pars, grid);
	field<block_fermion> A (grid);
	field<block_fermion> B (grid);
	block_matrix alpha = block_matrix::Random();
	rhmc.gaussian_fermion(A);
	rhmc.gaussian_fermion(B);

	// make mkl_alloc'ed array containg copy of A
	int sizeofA = 3*A.V*N_rhs;
	std::complex<double> *A_mkl = (std::complex<double>*)mkl_malloc(sizeofA*sizeof(std::complex<double>), 64);
	std::complex<double> *B_mkl = (std::complex<double>*)mkl_malloc(sizeofA*sizeof(std::complex<double>), 64);
	for(int in=0; in<N_rhs; ++in) {
		for(int ix=0; ix<A.V; ++ix) {
			for(int ic=0; ic<3; ++ic) {
				A_mkl[ic + 3*(ix + A.V*in)] = A[ix](ic,in);
				B_mkl[ic + 3*(ix + A.V*in)] = B[ix](ic,in);
			}
		}
	}

	A.add(B, alpha);
	CAPTURE(A[0]);

	// A_mkl += B alpha
	std::complex<double> one (1.0, 0.0);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		 3*A.V, N_rhs, N_rhs, &one, B_mkl, 3*A.V, alpha.data(), N_rhs, &one, A_mkl, 3*A.V);

    // A -= A_mkl
	CAPTURE(A_mkl[0]);
	for(int ix=0; ix<A.V; ++ix) {
		for(int ic=0; ic<3; ++ic) {
			for(int in=0; in<N_rhs; ++in) {
				A[ix](ic,in) -= A_mkl[ic + 3*(ix + A.V*in)];
			}
		}
	}
    REQUIRE ( A.norm() < EPS );
    mkl_free(A_mkl);
    mkl_free(B_mkl);
}
#endif

#ifdef EIGEN_USE_MKL_ALL
TEST_CASE( "add_mkl_compact vs add", "[mkl]" ) {

	lattice grid (4, true);
	rhmc_params rhmc_pars;
	rhmc rhmc (rhmc_pars, grid);
	field<block_fermion> A (grid);
	field<block_fermion> B (grid);
	field<block_fermion> AB (grid);
	block_matrix alpha = block_matrix::Random();
	rhmc.gaussian_fermion(A);
	rhmc.gaussian_fermion(B);

	int nm = A.V;

	// make mkl_alloc'ed array containg copy of A, B, alpha
	int sizeofA = 3*nm*N_rhs;
	std::complex<double> *A_mkl = (std::complex<double>*)mkl_malloc(sizeofA*sizeof(std::complex<double>), 64);
	std::complex<double> *B_mkl = (std::complex<double>*)mkl_malloc(sizeofA*sizeof(std::complex<double>), 64);
	std::complex<double> *alpha_mkl = (std::complex<double>*)mkl_malloc(N_rhs*N_rhs*sizeof(std::complex<double>), 64);
	for(int ix=0; ix<nm; ++ix) {
		for(int in=0; in<N_rhs; ++in) {
			for(int ic=0; ic<3; ++ic) {
				A_mkl[ic + 3*(in + N_rhs*ix)] = A[ix](ic,in);
				B_mkl[ic + 3*(in + N_rhs*ix)] = B[ix](ic,in);
			}
		}
	}

	for(int in=0; in<N_rhs; ++in) {
		for(int ic=0; ic<N_rhs; ++ic) {
			alpha_mkl[ic + N_rhs*in] = alpha(ic,in);
		}
	}

	// make array of pointers to each of nm matrices
	MKL_Complex16 *A_mkl_arr[nm];
	MKL_Complex16 *B_mkl_arr[nm];
	MKL_Complex16 *alpha_mkl_arr[nm];
	for(int ix=0; ix<nm; ++ix) {
		A_mkl_arr[ix] = A[ix].data(); //&A_mkl[ix*3*N_rhs];
		B_mkl_arr[ix] = &B_mkl[ix*3*N_rhs];
		alpha_mkl_arr[ix] = &alpha_mkl[0];
	}

	// A += B\alpha 
	AB = A;
	CAPTURE(AB[0]);
	AB.add(B, alpha);
	CAPTURE(AB[0]);

    // Set up Compact arrays
    MKL_COMPACT_PACK format = mkl_get_format_compact();

    MKL_INT a_buffer_size = mkl_zget_size_compact(3, N_rhs, format, nm);
    MKL_INT alpha_buffer_size = mkl_zget_size_compact(N_rhs, N_rhs, format, nm);

	CAPTURE(A_mkl_arr[0][0]);
	CAPTURE(A_mkl_arr[0][1]);

    double *a_compact = (double *)mkl_malloc(a_buffer_size, 64);
    double *b_compact = (double *)mkl_malloc(a_buffer_size, 64);
    double *alpha_compact = (double *)mkl_malloc(alpha_buffer_size, 64);

    /* Pack from P2P to Compact format */
    mkl_zgepack_compact(MKL_COL_MAJOR, 3, N_rhs, reinterpret_cast<const MKL_Complex16* const*>(A_mkl_arr), 3, a_compact, 3, format, nm);
    mkl_zgepack_compact(MKL_COL_MAJOR, 3, N_rhs, reinterpret_cast<const MKL_Complex16* const*>(B_mkl_arr), 3, b_compact, 3, format, nm);
    mkl_zgepack_compact(MKL_COL_MAJOR, N_rhs, N_rhs, reinterpret_cast<const MKL_Complex16* const*>(alpha_mkl_arr), N_rhs, alpha_compact, N_rhs, format, nm);

	// a_compact += b_compact alpha_compact
	std::complex<double> one (1.0, 0.0);
	mkl_zgemm_compact( MKL_COL_MAJOR, (MKL_TRANSPOSE)CblasNoTrans, (MKL_TRANSPOSE)CblasNoTrans, 3, N_rhs, N_rhs, reinterpret_cast<const MKL_Complex16*>(&one), b_compact, 3, alpha_compact, N_rhs, reinterpret_cast<const MKL_Complex16*>(&one), a_compact, 3, format, nm );

    // Unpack from a_compact to A_mkl_arr
    mkl_zgeunpack_compact(MKL_COL_MAJOR, 3, N_rhs, reinterpret_cast<MKL_Complex16* const*>(A_mkl_arr), 3, a_compact, 3, format, nm);

	CAPTURE(A_mkl_arr[0][0]);
	CAPTURE(A_mkl_arr[0][1]);

    /* Deallocate arrays */
    mkl_free(a_compact);
    mkl_free(b_compact);
    mkl_free(alpha_compact);

    mkl_free(A_mkl);
    mkl_free(B_mkl);

	CAPTURE(A[0]);

	A -= AB;
    REQUIRE ( A.norm() < EPS );
}
#endif