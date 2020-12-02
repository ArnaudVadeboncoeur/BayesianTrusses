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

#include <Eigen/Dense>

using Mat       = Eigen::MatrixXd;
using Vect      = Eigen::VectorXd;
using tupMatMat = std::tuple< Eigen::MatrixXd, Eigen::MatrixXd > ;

template< typename FUNC >
class SVGD{
public:

	//! constructors
	SVGD( );// { }
	SVGD( FUNC delLogP );

	void InitSamples( const Mat& samples  );

	void gradOptim( int iter, double nesterovMu = 0.9, double nesterovAlpha_ = 1e-3);


	//! destructor
	~SVGD( ) { }

private:

	Mat    gradSVGD( const Mat& X );

	Mat    kernMat    ( const Mat& X );
	Mat    sumDerKern ( const Mat& X, const Mat& kernX );

	FUNC  delLogP_;

	tupMatMat returnDelLogPFUNC_;
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

	delLogP_ = delLogP;
}


template< typename FUNC >
void SVGD< FUNC >::InitSamples( const Mat& X ){

	Xn_ = X;

	return;
}





template< typename FUNC >
Mat SVGD< FUNC >::kernMat( const Mat& X ){

	Mat kernOfX (X.rows(), X.rows());
	double median;

	for(int i = 0; i < X.rows(); ++i){
		for(int j = 0; j<= i; ++j){

			kernOfX(i, j) =  ( X.row(i) - X.row(j) ).norm() ;
			kernOfX(j, i) = kernOfX(i, j);

		}
	}

	Eigen::Map < Eigen::RowVectorXd > vOrig ( kernOfX.data(), kernOfX.size() ) ;
	Eigen::VectorXd v = vOrig;
	std::sort( v.data(), v.data() + v.size() );

	if( X.rows() % 2 == 0 ){
		median = v[(int) v.size()/2];
	}
	else{

		median =  ( v[(int) v.size()/2] + v[(int) (v.size()/2 + 1) ] ) / 2.;
	}

	h_ = median * median / std::log(X.rows());

	std::cout << "h_\n" << h_ << std::endl;
	kernOfX = ( 1./ h_ *  kernOfX.array() ).exp() ;


	return kernOfX;
}

template< typename FUNC >
Mat SVGD< FUNC >::sumDerKern( const Mat& X, const Mat& kernX ){

	Mat avgDerkernOfX(X.rows(), X.cols());
	avgDerkernOfX.setZero();
	Eigen::MatrixXd matx_iMinX(X.rows(), X.cols());

	for(int i = 0; i < X.rows(); ++i){

		matx_iMinX =  ( (-X).rowwise() + X.row(i) );

		for(int j = 0; j< kernX.rows(); ++j){

			avgDerkernOfX.row(j) += matx_iMinX.row(j) * kernX.row(i)[j] ;
		}

	}
	avgDerkernOfX = avgDerkernOfX * (double) 1./ X.rows();

	return avgDerkernOfX ;
}

template< typename FUNC >
Mat SVGD< FUNC >::gradSVGD( const Mat& X ){

	Mat gradient (X.rows(), X.cols() );

	returnDelLogPFUNC_ = delLogP_( X );
	delLogPMat_        = std::get<1>( returnDelLogPFUNC_ ) ;

	kernMat_        = kernMat( X );

	sumDerKernMat_  = sumDerKern( X, kernMat_);

	gradient = (double) 1. / X.rows() * ( kernMat_ * delLogPMat_ + sumDerKernMat_ );

	return gradient;
}



template< typename FUNC >
void SVGD< FUNC >::gradOptim(  int iter, double nesterovMu, double nesterovAlpha_ ) {

	nesterovMu_ = nesterovMu;
	iter_ 		= iter;

	Mat vt (Xn_.rows(), Xn_.cols() ); vt.setZero();

	for(int i = 0; i < iter; ++i){
		std::cout << "i " << i<< std::endl;
		grad_ = gradSVGD( Xn_ + nesterovMu_ * vt);


		//std::cout << "grad_\n"<<  grad_ << std::endl;
		std::cout << "grad_.norm()\n"<<  grad_.norm() << std::endl;


		vt    = nesterovMu_ * vt + nesterovAlpha_ * grad_ ;

		//std::cout << "vt\n"<<  vt << std::endl;

		Xn_ += vt;

	}

	return;
}

#endif // SVGD_HPP_


