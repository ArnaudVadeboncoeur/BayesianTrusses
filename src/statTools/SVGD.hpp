#ifndef SVGD_HPP_
#define SVGD_HPP_

#include <utility>
#include <functional>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>


using Mat  = Eigen::MatrixXd;
using Vect = Eigen::VectorXd;

template< typename FUNC >
class SVGD{
public:

	//! constructors
	SVGD( );// { }
	SVGD( FUNC delLogP );

	void InitSamples( Mat samples  );
	void InitSamples( Vect means, Mat covariance, int numSamples );

	void gradOptim( double nesterovVMu , int iter );


	//! destructor
	~SVGD( ) { }

private:

	Mat    gradSVGD( const Mat samples   );

	Mat    kernMat    ( const Mat X );
	Mat    sumDerKern ( const Mat X, const Mat kernX );

	FUNC  delLogP_;

	Mat   delLogPMat_;
	Mat   Xn_;
	Mat   grad_;
	Mat   pertubation_;

	Mat kernMat_;
	Mat sumDerKernMat_;

	double nesterovMu_;
	double nesterovAlpha_;

	double h_;

	int iter_;

};

template< typename FUNC >
SVGD< FUNC >::SVGD(FUNC delLogP) {

	FUNC delLogP_ = delLogP;
}

template< typename FUNC >
Mat SVGD< FUNC >::kernMat( const Mat& X ){

	Mat kernOfX (X.rows(), X.rows());

	for(int i = 0; i < X.rows(); ++i){
		for(int j = 0; j<= i; ++j){

			kernOfX(i, j) = std::exp( -1/ h_ * ( X.row(i) - X.row(j) ).norm()  );
			kernOfX(j, i) = kernOfX(i, j);
		}
	}

	return kernOfX;
}

template< typename FUNC >
Mat SVGD< FUNC >::sumDerKern( const Mat& X, const Mat& kernX ){

	Mat sumDerkernOfX(Xn_.rows(), Xn_.cols());
	sumDerkernOfX.setZero();

	for(int i = 0; i < Xn_.rows(); ++i){

		sumDerkernOfX += (  (- Xn_).colwise() +  Xn_.row(i) ).array() * kernX.row(i).transpose().array();
	}
	sumDerkernOfX = sumDerkernOfX * 1./ Xn.rows();

	return sumDerkernOfX ;
}

template< typename FUNC >
Mat SVGD< FUNC >::gradSVGD( ){

	Eigen::MatrixXd gradient (Xn_.rows(), Xn_.cols() );
	delLogPMat_ = delLogP_( Xn_ );

	kernMat_        = kernMat( Xn_ );
	sumDerKernMat_  = sumDerKern( Xn_, kernMat_);

	gradient = 1 / Xn_.rows() * ( kernMat * delLogPMat_ + sumDerKern );

	return gradient;
}



template< typename FUNC >
void SVGD< FUNC >::gradOptim( double nesterovMu = 0.9, int iter ) {

	nesterovMu_ = nesterovMu;
	iter_ 		= iter;

	Mat vt (Xn_.rows(), Xn_.cols() ); vt.setZero();

	for(int i = 0; i < iter; ++i){

		grad_ = gradSVGD( Xn_ + nesterovMu_ * vt_);

		vt    = nesterovMu_ * vt + nesterovAlpha_ * grad_ ;

		Xn_ += vt;

	}

	return;
}

#endif // SVGD_HPP_


