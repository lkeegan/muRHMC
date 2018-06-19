#include "dirac_op.hpp"
//#include "inverters.hpp"
#include <iostream>
#include <chrono>

std::chrono::time_point<std::chrono::high_resolution_clock> timer_start;
// start timer
void tick() {
    timer_start = std::chrono::high_resolution_clock::now();
}
// stop timer and return elapsed time in milliseconds
int tock() {
    auto timer_stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(timer_stop-timer_start).count();
}

int main(int argc, char *argv[]) {

    if (argc-1 != 1) {
        std::cout << "This program requires 1 argument, the lattice size L:" << std::endl;
        std::cout << "e.g. ./benchmark 16" << std::endl;
        return 1;
    }

	int L = static_cast<int>(atof(argv[1]));
	int N_dirac_ops = 1 + static_cast<int>(10000000/(L*L*L*L*N_rhs));
	int N_adds = 20 * N_dirac_ops;

	// make L^4 EO lattice
	lattice grid (L, true);

	// initialise Dirac Op
	dirac_op D (grid, 0.002, 0.0);

	std::cout.precision(12);
	std::cout << "# Benchmark run" << std::endl;
	std::cout << "# L = " << L << std::endl;
	std::cout << "# N_rhs = " << N_rhs << std::endl;
	std::cout << "# N_dirac_ops = " << 2*N_dirac_ops << std::endl;
	std::cout << "# N_adds = " << 2*N_adds << std::endl;

	// make U[mu] field on lattice with random 3x3 complex matrices
	field<gauge> U (grid);
	for(int ix=0; ix<grid.V; ++ix) {
		for(int mu=0; mu<4; ++mu) {
			U[ix][mu] = SU3mat::Random();
		}
	}

	block_matrix alpha, beta;
	alpha.setRandom();
	beta.setRandom();

	//
	// BLOCK
	//
	
	// make random block fermion fields:
	field<block_fermion> bphi (U.grid, field<block_fermion>::EVEN_ONLY);
	for(int ix=0; ix<bphi.V; ++ix) {
		bphi[ix] = block_fermion::Random();
	}
	field<block_fermion> bchi = bphi;

	// dirac op	
	tick();
    for(int i=0; i<N_dirac_ops; ++i) {
	    D.DDdagger(bchi, bphi, U);
		D.DDdagger(bphi, bchi, U);    	
    }
    int block_dirac_count = tock();
	std::cout << "# BLOCKDiracOpRunTime: " << block_dirac_count << std::endl;

	// vector add EIGEN
	tick();
    for(int i=0; i<N_adds; ++i) {
		bchi.add(bphi, alpha);
		bphi.add(bchi, beta);
    }
    int block_add_count = tock();
	std::cout << "# BLOCKAddRunTime: " << block_add_count << std::endl;
	exit(0);

	// vector add EIGEN-bigmat
	tick();
	// make big eigem matrix containg copy of A,B
	typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, N_rhs> fermion_block_field;
	fermion_block_field A_big = fermion_block_field::Zero(3*bphi.V, N_rhs);
	fermion_block_field B_big = fermion_block_field::Zero(3*bphi.V, N_rhs);
	for(int in=0; in<N_rhs; ++in) {
		for(int ix=0; ix<bphi.V; ++ix) {
			for(int ic=0; ic<3; ++ic) {
				A_big(ic + 3*ix, in) = bphi[ix](ic,in);
				B_big(ic + 3*ix, in) = bchi[ix](ic,in);
			}
		}
	}
    for(int i=0; i<N_adds; ++i) {
		A_big.noalias() += B_big * alpha;
		B_big.noalias() += A_big * beta;
    }
    int block_add_bigmat_count = tock();
	std::cout << "# BLOCKAddBigmatRunTime: " << block_add_bigmat_count << std::endl;

#ifdef EIGEN_USE_MKL_ALL

	// vector add MKL-ZGEMM
	tick();
	// make mkl_alloc'ed array containg copy of A
	int sizeofA = 3*bphi.V*N_rhs;
	std::complex<double> *A_mkl = (std::complex<double>*)mkl_malloc(sizeofA*sizeof(std::complex<double>), 64);
	std::complex<double> *B_mkl = (std::complex<double>*)mkl_malloc(sizeofA*sizeof(std::complex<double>), 64);
	for(int in=0; in<N_rhs; ++in) {
		for(int ix=0; ix<bphi.V; ++ix) {
			for(int ic=0; ic<3; ++ic) {
				A_mkl[ic + 3*(ix + bphi.V*in)] = bchi[ix](ic,in);
				B_mkl[ic + 3*(ix + bphi.V*in)] = bphi[ix](ic,in);
			}
		}
	}

	std::complex<double> *alpha_mkl = (std::complex<double>*)mkl_malloc(N_rhs*N_rhs*sizeof(std::complex<double>), 64);
	std::complex<double> *beta_mkl = (std::complex<double>*)mkl_malloc(N_rhs*N_rhs*sizeof(std::complex<double>), 64);
	for(int i=0; i<N_rhs; ++i) {
		for(int j=0; j<N_rhs; ++j) {
			alpha_mkl[i+N_rhs*j] = alpha(i,j);
			beta_mkl[i+N_rhs*j] = beta(i,j);
		}
	}

	std::complex<double> one (1.0, 0.0);
    for(int i=0; i<N_adds; ++i) {
		// A_mkl += B_mkl alpha
	    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
			 3*bphi.V, N_rhs, N_rhs, &one, B_mkl, 3*bphi.V, alpha_mkl, N_rhs, &one, A_mkl, 3*bphi.V);
		// B_mkl += A_mkl beta
	    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
			 3*bphi.V, N_rhs, N_rhs, &one, A_mkl, 3*bphi.V, beta_mkl, N_rhs, &one, B_mkl, 3*bphi.V);
	}
    mkl_free(alpha_mkl);
    mkl_free(beta_mkl);
    mkl_free(A_mkl);
    mkl_free(B_mkl);

    int block_add_mkl_bigmat_count = tock();
	std::cout << "# BLOCKAdd_mkl_bigmat_RunTime: " << block_add_mkl_bigmat_count << std::endl;

	// vector add MKL-ZGEMM_COMPACT
	tick();
	// make array of pointers to each of nm matrices
	int nm = bchi.V;
	MKL_Complex16 *A_mkl_arr[nm];
	MKL_Complex16 *B_mkl_arr[nm];
	MKL_Complex16 *alpha_mkl_arr[nm];
	MKL_Complex16 *beta_mkl_arr[nm];
	for(int ix=0; ix<nm; ++ix) {
		A_mkl_arr[ix] = bchi[ix].data();
		B_mkl_arr[ix] = bphi[ix].data();
		alpha_mkl_arr[ix] = alpha.data();
		beta_mkl_arr[ix] = beta.data();
	}

    // Set up Compact arrays
    MKL_COMPACT_PACK format = mkl_get_format_compact();

    MKL_INT a_buffer_size = mkl_zget_size_compact(3, N_rhs, format, nm);
    MKL_INT alpha_buffer_size = mkl_zget_size_compact(N_rhs, N_rhs, format, nm);

    double *a_compact = (double *)mkl_malloc(a_buffer_size, 64);
    double *b_compact = (double *)mkl_malloc(a_buffer_size, 64);
    double *alpha_compact = (double *)mkl_malloc(alpha_buffer_size, 64);
    double *beta_compact = (double *)mkl_malloc(alpha_buffer_size, 64);

    /* Pack from P2P to Compact format */
    mkl_zgepack_compact(MKL_COL_MAJOR, 3, N_rhs, reinterpret_cast<const MKL_Complex16* const*>(A_mkl_arr), 3, a_compact, 3, format, nm);
    mkl_zgepack_compact(MKL_COL_MAJOR, 3, N_rhs, reinterpret_cast<const MKL_Complex16* const*>(B_mkl_arr), 3, b_compact, 3, format, nm);
    mkl_zgepack_compact(MKL_COL_MAJOR, N_rhs, N_rhs, reinterpret_cast<const MKL_Complex16* const*>(alpha_mkl_arr), N_rhs, alpha_compact, N_rhs, format, nm);
    mkl_zgepack_compact(MKL_COL_MAJOR, N_rhs, N_rhs, reinterpret_cast<const MKL_Complex16* const*>(beta_mkl_arr), N_rhs, beta_compact, N_rhs, format, nm);

    for(int i=0; i<N_adds; ++i) {
		// A_mkl += B_mkl alpha
		mkl_zgemm_compact( MKL_COL_MAJOR, (MKL_TRANSPOSE)CblasNoTrans, (MKL_TRANSPOSE)CblasNoTrans, 3, N_rhs, N_rhs, reinterpret_cast<const MKL_Complex16*>(&one), b_compact, 3, alpha_compact, N_rhs, reinterpret_cast<const MKL_Complex16*>(&one), a_compact, 3, format, nm );
		// B_mkl += A_mkl beta
		mkl_zgemm_compact( MKL_COL_MAJOR, (MKL_TRANSPOSE)CblasNoTrans, (MKL_TRANSPOSE)CblasNoTrans, 3, N_rhs, N_rhs, reinterpret_cast<const MKL_Complex16*>(&one), a_compact, 3, beta_compact, N_rhs, reinterpret_cast<const MKL_Complex16*>(&one), b_compact, 3, format, nm );
	}

    mkl_zgeunpack_compact(MKL_COL_MAJOR, 3, N_rhs, reinterpret_cast<MKL_Complex16* const*>(A_mkl_arr), 3, a_compact, 3, format, nm);
    mkl_zgeunpack_compact(MKL_COL_MAJOR, 3, N_rhs, reinterpret_cast<MKL_Complex16* const*>(B_mkl_arr), 3, b_compact, 3, format, nm);

    /* Deallocate arrays */
    mkl_free(a_compact);
    mkl_free(b_compact);
    mkl_free(alpha_compact);
    mkl_free(beta_compact);

    int block_add_mkl_compact_count = tock();
	std::cout << "# BLOCKAdd_mkl_compact_RunTime: " << block_add_mkl_compact_count << std::endl;


	// data output
	std::cout << "# N_rhs\tN_dirac\tN_add\tdirac\tadd\tadd_mkl_bigmat\tadd_mkl_compact: " << std::endl;
	std::cout << N_rhs << "\t" << 2*N_dirac_ops << "\t" << 2*N_adds << "\t"
			  << block_dirac_count << "\t" 
			  << block_add_count << "\t"
			  << block_add_mkl_bigmat_count << "\t" << block_add_mkl_compact_count << "\t"
		      << std::endl;

#endif

	exit(0);


	//
	// NON BLOCK
	//

	// make random non-block fermion fields:
	std::vector< field<fermion> > phi (N_rhs, field<fermion>(U.grid, field<fermion>::EVEN_ONLY));
	for(int i_p=0; i_p<N_rhs; ++i_p) {
		for(int ix=0; ix<phi[0].V; ++ix) {
			phi[i_p][ix] = fermion::Random();
		}
	}
	std::vector< field<fermion> > chi = phi;

	// dirac op
	tick();
    for(int i=0; i<N_dirac_ops; ++i) {
		for(int i_p=0; i_p<N_rhs; ++i_p) {
		    D.DDdagger(chi[i_p], phi[i_p], U);
		}
		for(int i_p=0; i_p<N_rhs; ++i_p) {
			D.DDdagger(phi[i_p], chi[i_p], U);    	
		}
    }
    int dirac_count = tock();
	std::cout << "# DiracOpRunTime: " << dirac_count << std::endl;

	// vector add
	tick();
    for(int i=0; i<N_adds; ++i) {
		//beta.setRandom();
		for(int i_p=0; i_p<N_rhs; ++i_p) {
			for(int j_p=0; j_p<N_rhs; ++j_p) {
				chi[i_p].add(alpha(j_p, i_p), phi[j_p]);
			}
		}
		//beta.setRandom();
		for(int i_p=0; i_p<N_rhs; ++i_p) {
			for(int j_p=0; j_p<N_rhs; ++j_p) {
				phi[i_p].add(beta(j_p, i_p), chi[j_p]);
			}
		}
    }
    int add_count = tock();
	std::cout << "# AddRunTime: " << add_count << std::endl;


	// data output
	std::cout << "# N_rhs\tL\tdirac_time\tblock_time\tadd_time\tblock_time: " << block_dirac_count << std::endl;
	std::cout << N_rhs << "\t" << L << "\t" 
			  << dirac_count << "\t" << block_dirac_count << "\t"
			  << add_count << "\t" << block_add_count << "\t"
		      << std::endl;

	return(0);
}
