#ifndef LATTICE_4D_H
#define LATTICE_4D_H

#include<vector>
#include<complex>
#include "omp.h"

// 4d lattice with pbcs
// pair of integer vectors store indices of nearest up/dn neighbours
class lattice {
protected:
	std::vector<int> neighbours_up_;
	std::vector<int> neighbours_dn_;
	int pbcs (int x, int L) const;
	int index_LEXI (int x0, int x1, int x2, int x3) const;
	int index_EO (int x0, int x1, int x2, int x3) const;
	
public:
	int L0;
	int L1;
	int L2;
	int L3;
	int VOL3;
	int V;
	bool isEO;

	// constructor for L0xL1xL2xL3 lattice, optionally with EO data layout
	lattice (int L0, int L1, int L2, int L3, bool isEO=false);

	// assume TxL^3 if only two lengths specified
	lattice (int T, int L, bool isEO=false) : lattice::lattice (T, L, L, L, isEO) {}

	// assume L^4 if only one length specified
	lattice (int L, bool isEO=false) : lattice::lattice (L, L, L, L, isEO) {}

	// return index to data from 4-vector coordinates (SLOW: for debugging!)
	int index (int x0, int x1, int x2, int x3) const;

	// returns index of nearest neighbour in mu direction
	int iup (int i, int mu) const { return neighbours_up_[4*i+mu]; }
	int idn (int i, int mu) const { return neighbours_dn_[4*i+mu]; }

	// returns 4-index i in [0, V) corresponding to the point (x0, ix3) 
	// where ix3 is an index in the range [0, VOL3)
	// so looping over all ix3 in [0, VOL3) for fixed x0 loops over x0 timeslice
	int it_ix (int x0, int ix3) const; //NB SLOW for EO layout

	// convert ix between EO and LEXI, useful for reading files
	// stored in one format to another format
	int EO_from_LEXI (int ix_lexi) const;
	int LEXI_from_EO (int ix_eo) const;
};

template<typename T> class field {
protected:
	// data
	std::vector<T> data_;
	// even-odd offset to subtract
	int eo_offset = 0;

public:
	const lattice& grid;
	int V, VOL3, L0;
	int L1, L2, L3;
	enum eo_storage_options {FULL, EVEN_ONLY, ODD_ONLY};
	eo_storage_options eo_storage;
	explicit field (const lattice& latt, eo_storage_options eo_storage = FULL) : grid(latt), V(latt.V), VOL3(latt.VOL3), 
										   L0(latt.L0), L1(latt.L1), L2(latt.L2), L3(latt.L3), eo_storage(eo_storage) {
		if (eo_storage == FULL) {
			// store entire field
			data_.resize(V);
		} else if (eo_storage == EVEN_ONLY) {
			// only store even part of field
			V /= 2;
			data_.resize(V);			
		} else {
			// only store odd part of field
			V /= 2;
			data_.resize(V);
			// need to subtract V from array indices that come from grid.iup etc
			// since we don't store the even part which is the first V elements
			eo_offset = V;
		} 
	}

	// assignment op just copies data, no check that lattices are consistent - TODO 
	field& operator=(const field& rhs) {
		for(int ix=0; ix<V; ++ix) {
			data_[ix] = rhs[ix];
		}
		return *this;
	}

	field& operator+=(const field& rhs)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] += rhs[ix];
		}
	    return *this;
	}

	field& operator-=(const field& rhs)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] -= rhs[ix];
		}
	    return *this;
	}

	field& operator*=(double scalar)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] *= scalar;
		}
	    return *this;
	}

	field& operator*=(std::complex<double> scalar)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] *= scalar;
		}
	    return *this;
	}

	field& operator/=(double scalar)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] /= scalar;
		}
	    return *this;
	}

	field& operator/=(std::complex<double> scalar)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] /= scalar;
		}
	    return *this;
	}

	// *this += rhs_multiplier * rhs
	field& add(std::complex<double> rhs_multiplier, const field& rhs)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] += rhs_multiplier * rhs[ix];
		}
	    return *this;
	}
	field& add(double rhs_multiplier, const field& rhs)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] += rhs_multiplier * rhs[ix];
		}
	    return *this;
	}
	// *this = scale * (*this) + rhs_multiplier * rhs
	field& scale_add(double scale, double rhs_multiplier, const field& rhs)
	{
		#pragma omp parallel for
		for(int ix=0; ix<V; ++ix) {
			data_[ix] = scale * data_[ix] + rhs_multiplier * rhs[ix];
		}
	    return *this;
	}
	field& scale_add(std::complex<double> scale, std::complex<double> rhs_multiplier, const field& rhs)
	{
		#pragma omp parallel for
		for(int ix=0; ix<V; ++ix) {
			data_[ix] = scale * data_[ix] + rhs_multiplier * rhs[ix];
		}
	    return *this;
	}

	// these are some eigen routines for individual vectors or matrices
	// trivially extended by applying them to each element in the vector in turn

	void setZero() {
		for(int ix=0; ix<V; ++ix) {
			data_[ix].setZero();
		}		
	}
	// equivalent to real part of dot with itself i.e. l2-norm squared
	double squaredNorm() const {
		double norm = 0.0;
		#pragma omp parallel for reduction(+:norm)
		for(int ix=0; ix<V; ++ix) {
			norm += data_[ix].squaredNorm();
		}
		return norm;		
	}
	// returns square root of squaredNorm() above i.e. l2-norm
	double norm() const {
		return sqrt(squaredNorm());		
	}
	//complex conjugate of this dotted with rhs
	std::complex<double> dot (const field& rhs) const {
		std::complex<double> sum (0.0, 0.0);
		//#pragma omp parallel for reduction(+:sum)
		//note: openMP can't reduce std::complex<double> type
		for(int ix=0; ix<V; ++ix) {
			sum += data_[ix].dot(rhs[ix]);
		}
		return sum;		
	}

	// return reference to data at point (x0, x1, x2, x3) (SLOW: for debugging!)
	T& at (int x0, int x1, int x2, int x3) { return data_[grid.index(x0,x1,x2,x3) - eo_offset]; }
	const T& at (int x0, int x1, int x2, int x3) const { return data_[grid.index(x0,x1,x2,x3) - eo_offset]; }

	// [i] operator returns data with index i
	T& operator[](int i) { return data_[i]; }
	const T& operator[](int i) const { return data_[i]; }

	// returns reference to nearest neighbour in mu direction
	T& up (int i, int mu) { return data_[grid.iup (i, mu) - eo_offset]; }
	const T& up (int i, int mu) const { return data_[grid.iup (i, mu) - eo_offset]; }
	T& dn (int i, int mu) { return data_[grid.idn (i, mu) - eo_offset]; }
	const T& dn (int i, int mu) const { return data_[grid.idn (i, mu) - eo_offset]; }

	// returns index of nearest neighbour in mu direction
	int iup (int i, int mu) const { return grid.iup (i, mu) - eo_offset; }
	int idn (int i, int mu) const { return grid.idn (i, mu) - eo_offset; }

	// returns 4-index i in [0, V) corresponding to the point (x0, ix) 
	// where ix is a point in the spatial volume [0, VOL3)
	// and x0 is the timeslice in [0, L0)
	int it_ix (int x0, int ix) const { return grid.it_ix (x0, ix) - eo_offset; }

};

#endif //LATTICE_4D_H