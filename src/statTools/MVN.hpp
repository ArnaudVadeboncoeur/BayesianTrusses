#ifndef MVN_HPP_
#define MVN_HPP_

#include <utility>
#include <functional>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>

using Mat   = Eigen::MatrixXd;
using Vect  = Eigen::VectorXd;

class MVN{
public:

	//! constructors
	MVN( );// { }
	MVN( Vect mean, Mat covariance );

	Mat sampleMVN( int N  );

	//! destructor
	~MVN( ) { }

private:
	Mat covariance_;
	Vect mean_;
	Mat samples_;



	int N_;


};

MVN::MVN( Vect mean, Mat covariance) {

	covariance_ = covariance;
	mean_ 	    = mean;

}

Mat MVN::sampleMVN( int N ){

	N_ = N;
	Vect z(mean_.size());
	Vect v(mean_.size());
	samples_.resize(N_, mean_.size());

	Eigen::LLT<Mat> lltOfC(covariance_); // compute the Cholesky decomposition of A
	Mat L = lltOfC.matrixL();

	std::normal_distribution<double> Normal( 0, 1 );
	std::random_device rd;
	std::mt19937 engine( rd() );

	for(int i = 0; i < N_ ; ++i){

		for(int j = 0; j < mean_.size(); j++){

			z[j] = Normal( engine );

		}

		v = L * z;
		samples_.row(i) = (v + mean_).transpose();

	}

	return samples_;
}


#endif // MVN_HPP_


