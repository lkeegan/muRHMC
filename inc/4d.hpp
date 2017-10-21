#ifndef LATTICE_4D_H
#define LATTICE_4D_H

#include<vector>
#include<complex>

// 4d lattice with pbcs
// pair of integer vectors store indices of nearest up/dn neighbours
class lattice {
protected:
	std::vector<int> neighbours_up_;
	std::vector<int> neighbours_dn_;
	int pbcs (int x, int L) const;
	
public:
	int L0;
	int L1;
	int L2;
	int L3;
	int V;

	// constructor for L0xL1xL2xL3 lattice
	lattice (int L0, int L1, int L2, int L3);

	// assume L^4 if only one length specified
	explicit lattice (int L) : lattice::lattice (L, L, L, L) {}

	// default destructor is fine:
	//~lattice ();

	// return index to data from 4-vector coordinates (SLOW: for debugging!)
	int index (int x0, int x1, int x2, int x3) const;

	// returns index of nearest neighbour in mu direction
	int iup (int i, int mu) const { return neighbours_up_[4*i+mu]; }
	int idn (int i, int mu) const { return neighbours_dn_[4*i+mu]; }

};

template<typename T> class field {
protected:
	// data
	std::vector<T> data_;

public:
	const lattice& grid;
	int V;
	explicit field (const lattice& latt) : grid(latt), V(latt.V) { data_.resize(V); }

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

	// *this += rhs_multiplier * rhs
	field& add(std::complex<double> rhs_multiplier, field& rhs)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] += rhs_multiplier * rhs[ix];
		}
	    return *this;
	}
	field& add(double rhs_multiplier, field& rhs)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] += rhs_multiplier * rhs[ix];
		}
	    return *this;
	}
	// *this = scale * (*this) + rhs_multiplier * rhs
	field& scale_add(double scale, double rhs_multiplier, field& rhs)
	{
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

	// equivalent to real part of dot with itself
	double squaredNorm() const {
		double norm = 0.0;
		for(int ix=0; ix<V; ++ix) {
			norm += data_[ix].squaredNorm();
		}
		return norm;		
	}
	//complex conjugate of this dotted with rhs
	std::complex<double> dot (const field& rhs) const {
		std::complex<double> sum (0.0, 0.0);
		for(int ix=0; ix<V; ++ix) {
			sum += data_[ix].dot(rhs[ix]);
		}
		return sum;		
	}

	// return reference to data at point (x0, x1, x2, x3) (SLOW: for debugging!)
	T& at (int x0, int x1, int x2, int x3) { return data_[grid.index(x0,x1,x2,x3)]; }
	const T& at (int x0, int x1, int x2, int x3) const { return data_[grid.index(x0,x1,x2,x3)]; }

	// [i] operator returns data with index i
	T& operator[](int i) { return data_[i]; }
	const T& operator[](int i) const { return data_[i]; }

	// returns reference to nearest neighbour in mu direction
	T& up (int i, int mu) { return data_[grid.iup (i, mu)]; }
	const T& up (int i, int mu) const { return data_[grid.iup (i, mu)]; }
	T& dn (int i, int mu) { return data_[grid.idn (i, mu)]; }
	const T& dn (int i, int mu) const { return data_[grid.idn (i, mu)]; }

	// returns index of nearest neighbour in mu direction
	int iup (int i, int mu) const { return grid.iup (i, mu); }
	int idn (int i, int mu) const { return grid.idn (i, mu); }
};

#endif //LATTICE_4D_H