#ifndef LATTICE_SU3_H
#define LATTICE_SU3_H
#include <complex>
#include <Eigen/Dense>
#include <Eigen/StdVector>

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3cd)
typedef Eigen::Matrix3cd SU3mat;
typedef Eigen::Vector3cd fermion;

// SU3 generators
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
