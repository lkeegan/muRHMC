#include "4d.hpp"

// constructor
lattice::lattice (int L0, int L1, int L2, int L3, bool isEO) : 
	neighbours_up_(4*L0*L1*L2*L3), neighbours_dn_(4*L0*L1*L2*L3), 
	L0(L0), L1(L1), L2(L2), L3(L3), VOL3(L1*L2*L3), V(L0*L1*L2*L3), 
	isEO(isEO) {
	// Fill nearest neighbours vector with indices of neighbours
	for(int x3=0; x3<L3; ++x3) {
		for(int x2=0; x2<L2; ++x2) {
			for(int x1=0; x1<L1; ++x1) {
				for(int x0=0; x0<L0; ++x0) {
					int i = index(x0, x1, x2, x3);
					neighbours_up_[4*i+0] = index(x0+1, x1, x2, x3);
					neighbours_up_[4*i+1] = index(x0, x1+1, x2, x3);
					neighbours_up_[4*i+2] = index(x0, x1, x2+1, x3);
					neighbours_up_[4*i+3] = index(x0, x1, x2, x3+1);
					neighbours_dn_[4*i+0] = index(x0-1, x1, x2, x3);
					neighbours_dn_[4*i+1] = index(x0, x1-1, x2, x3);
					neighbours_dn_[4*i+2] = index(x0, x1, x2-1, x3);
					neighbours_dn_[4*i+3] = index(x0, x1, x2, x3-1);
				}
			}
		}
	}
}

int lattice::pbcs (int x, int L) const {
	return (x + 2*L) % L;
}

int lattice::it_ix (int x0, int ix3) const {
	if(isEO) {
		// NB: slow routine - for debugging!
		int x1 = ix3 % L1;
		int tmp = (ix3 - x1) / L1;
		int x2 = tmp % L2;
		int x3 = (tmp - x2) / L2; 
		return index_EO(x0, x1, x2, x3);
	} else {
		return x0 + L0 * ix3;
	}
}

int lattice::index (int x0, int x1, int x2, int x3) const {
	if(isEO) {
		return index_EO (x0, x1, x2, x3);
	} else {
		return index_LEXI (x0, x1, x2, x3);
	}
}

int lattice::index_LEXI (int x0, int x1, int x2, int x3) const {
	return pbcs(x0, L0) + L0 * pbcs(x1, L1) + L0 * L1 * pbcs(x2, L2) + L0 * L1 * L2 * pbcs(x3, L3);
}

int lattice::index_EO (int x0, int x1, int x2, int x3) const {
	int i = index_LEXI (x0, x1, x2, x3);
	int eo_offset = (V / 2) * ( (pbcs(x0, L0) + pbcs(x1, L1) + pbcs(x2, L2) + pbcs(x3, L3)) % 2 );
	return i/2 + eo_offset;
}

int lattice::EO_from_LEXI (int ix_lexi) const {
	int x0 = ix_lexi % L0;
	int tmp0 = (ix_lexi - x0) / L0;
	int x1 = tmp0 % L1;
	int tmp1 = (tmp0 - x1) / L1;
	int x2 = tmp1 % L2;
	int x3 = (tmp1 - x2) / L2; 
	return index_EO (x0, x1, x2, x3);
}

int lattice::LEXI_from_EO (int ix_eo) const {
	int ix_lexi = 2*ix_eo;
	if (ix_lexi >= V) {
		ix_lexi -= V;
	}
	// ix_lexi is defined up to +0/1
	// so just check by brute force which one it is:
	if (EO_from_LEXI(ix_lexi) == ix_eo) {
		return ix_lexi;
	} else {
		return ix_lexi + 1;
	}
}