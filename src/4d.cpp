#include "4d.hpp"

// constructor
lattice::lattice (int L0, int L1, int L2, int L3) :
 L0(L0), L1(L1), L2(L2), L3(L3) {
	VOL3 = L1 * L2 * L3;
	V = L0 * VOL3;
	// Initialise nearest neighbours vector with indices of neighbours
	neighbours_up_.reserve(4*V);
	neighbours_dn_.reserve(4*V);
	for(int l=0; l<L3; ++l){
		for(int k=0; k<L2; ++k){
			for(int j=0; j<L1; ++j){
				for(int i=0; i<L0; ++i){
					neighbours_up_.push_back(index(i+1,j,k,l));
					neighbours_up_.push_back(index(i,j+1,k,l));
					neighbours_up_.push_back(index(i,j,k+1,l));
					neighbours_up_.push_back(index(i,j,k,l+1));
					neighbours_dn_.push_back(index(i-1,j,k,l));
					neighbours_dn_.push_back(index(i,j-1,k,l));
					neighbours_dn_.push_back(index(i,j,k-1,l));
					neighbours_dn_.push_back(index(i,j,k,l-1));
				}
			}
		}
	}
}

int lattice::index(int x0, int x1, int x2, int x3) const {
	return pbcs(x0, L0) + L0 * pbcs(x1, L1) + L0 * L1 * pbcs(x2, L2) + L0 * L1 * L2 * pbcs(x3, L3);
}

int lattice::pbcs(int x, int L) const {
	return (x + 2*L) % L;
}