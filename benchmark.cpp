#include "dirac_op.hpp"
#include "inverters.hpp"
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
	int N_dirac_ops = 1 + static_cast<int>(500000/(L*L*L*L*N_rhs));
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
	
	// vector add
    for(int i=0; i<N_adds; ++i) {
		bchi.add(bphi, alpha);
		bphi.add(bchi, beta);
    }
    int block_add_count = tock();
	std::cout << "# BLOCKAddRunTime: " << block_add_count << std::endl;

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
