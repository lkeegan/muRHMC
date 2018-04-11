#ifndef LATTICE_SU3_H
#define LATTICE_SU3_H
#include <complex>
#include <Eigen/Dense>
#include <Eigen/StdVector>
// hard code (for now) block fermion RHS
constexpr int N_rhs = 6;
constexpr int N_gauge = 3;
// define types for gauge links and fermion fields
typedef Eigen::Matrix<std::complex<double>, N_gauge, N_gauge> SU3mat;
template <int N>
using block_fermion_matrix = Eigen::Matrix<std::complex<double>, N_gauge, N>;
typedef block_fermion_matrix<1> fermion;
typedef block_fermion_matrix<N_rhs> block_fermion;
typedef Eigen::Matrix<std::complex<double>, N_rhs, N_rhs>  block_matrix;

// the following is required to be able to use STL vectors of
// these objects with correct alignment (i.e. otherwise segfaults!) 
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(SU3mat)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(fermion)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(block_fermion)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(block_matrix)

//template<int N_rhs>
//using block_fermion = Eigen::Matrix<std::complex<double>, 3, N_rhs>;

// return exp(X)
// where X is 3x3 complex anti-hermitian traceless matrix
// so exp(X) is an element of SU(3)
// Cayley-Hamilton form of matrix exponential
// https://luscher.web.cern.ch/luscher/notes/su3fcts.pdf
constexpr int exp_ch_N = 21;
// factorial 1/n! up to n = 21:
// NOTE! requires || X || <~ 1, if this norm is too large the expansion fails
constexpr double exp_ch_c_n[exp_ch_N+1] = {1.0, 1.0, 0.5, 0.16666666666666666,
 0.041666666666666664, 0.0083333333333333332, 0.0013888888888888889,
 0.00019841269841269841, 2.4801587301587302e-05, 2.7557319223985893e-06, 
 2.7557319223985894e-07, 2.505210838544172e-08, 2.08767569878681e-09, 
 1.6059043836821616e-10, 1.1470745597729726e-11, 7.6471637318198174e-13, 
 4.7794773323873859e-14, 2.811457254345521e-15, 1.5619206968586228e-16, 
 8.220635246624331e-18, 4.1103176233121653e-19, 1.9572941063391263e-20};
SU3mat exp_ch (const SU3mat& X);

// SU3 generators: complex 3x3 traceless hermitian matrices T_a
class SU3_Generators {
private:
	SU3mat T_[8];

public:
	const SU3mat& operator[](int i) const { return T_[i]; }

	SU3_Generators() {
		std::complex<double> I (0.0, 1.0);
		T_[0]<<	0, 		0.5, 	0,
				0.5,	0, 		0,
				0, 		0, 		0;

		T_[1]<<	0, 		-0.5*I, 0,
				0.5*I, 	0, 		0,
				0, 		0, 		0;

		T_[2]<<	0.5, 	0, 		0,
				0, 		-0.5, 	0,
				0, 		0, 		0;

		T_[3]<<	0, 		0, 		0.5,
				0, 		0,	 	0,
				0.5, 	0, 		0;

		T_[4]<<	0, 		0, 		-0.5*I,
				0, 		0, 		0,
				0.5*I, 	0, 		0;

		T_[5]<<	0, 		0, 		0,
				0, 		0, 		0.5,
				0, 		0.5, 	0;

		T_[6]<<	0, 		0, 		0,
				0, 		0, 		-0.5*I,
				0, 		0.5*I, 	0;

		T_[7]<<	0.5/sqrt(3.0), 0, 0,
				0, 0.5/sqrt(3.0), 0,
				0, 0, -1.0/sqrt(3.0);
	}
};

// class that contains U[mu], i.e. 4 x SU(3) matrices
class gauge {
private:
	SU3mat U_[4];
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// [mu] operator returns U[mu]
	SU3mat& operator[](int i) { return U_[i]; }
	const SU3mat& operator[](int i) const { return U_[i]; }

	// equality operator returns true if U[mu] are equal for all mu
	bool operator==(const gauge& other) const {
		return (U_[0]==other[0]) &&
			   (U_[1]==other[1]) &&
			   (U_[2]==other[2]) &&
			   (U_[3]==other[3]);
	}

	gauge& operator-=(const gauge& rhs)
	{
		for(int i=0; i<4; ++i) {
			U_[i] -= rhs[i];
		}
	    return *this;
	}

	gauge& operator+=(const gauge& rhs)
	{
		for(int i=0; i<4; ++i) {
			U_[i] += rhs[i];
		}
	    return *this;
	}

	gauge& operator*=(double scalar)
	{
		for(int i=0; i<4; ++i) {
			U_[i] *= scalar;
		}
	    return *this;
	}

	void setZero() {
		for(int i=0; i<4; ++i) {
			U_[i].setZero();
		}		
	}

	double squaredNorm() const {
		double norm = 0.0;
		for(int i=0; i<4; ++i) {
			norm += U_[i].squaredNorm();
		}
		return norm;		
	}

};

#endif //LATTICE_SU3_H
