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

template< typename FUNC >
class SVGD{
public:

	//! constructors
	SVGD( );// { }
	SVGD( FUNC delLogP ) {delLogP_ = delLogP;}

	void InitSamples( const Mat& samples  ) { Xn_ = samples; return; }

	void gradOptim( int iter,  double nesterovAlpha = 1e-3, double nesterovMu = 0.9);

	Mat getSamples( ) { return Xn_;}

	//creat matrix of hard limits or constraints for SVGD

	Mat gradNormHistory;
	Mat pertNormHistory;

	//! destructor
	~SVGD( ) { }

private:

	Mat    gradSVGD( const Mat& X );

	Mat    kernMat    ( const Mat& X );
	Mat    sumDerKern ( const Mat& X, const Mat& kernX );

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

	//std::cout << "h_\n" << h_ << std::endl;
	kernOfX = (- 1./ h_ *  kernOfX.array() *  kernOfX.array() ).matrix();
	//std::cout << "kernOfX\n"<<  kernOfX << std::endl;
	kernOfX = kernOfX.array().exp().matrix() ;

//	std::cout << "median\n"<<  median << std::endl;
//	std::cout << "h_\n"<<  h_ << std::endl;
//	std::cout << "kernOfX.nomr()\n"<<  kernOfX.norm() << std::endl;
//	std::cout << "X.nomr()\n"<<  X.norm() << std::endl;



	return kernOfX;
}

template< typename FUNC >
Mat SVGD< FUNC >::sumDerKern( const Mat& X, const Mat& kernX ){

	Mat sumDerKernMat(X.rows(), X.cols());
	sumDerKernMat.setZero();
	Eigen::MatrixXd matx_iMinX(X.rows(), X.cols());

	for(int i = 0; i < X.rows(); ++i){

		matx_iMinX =  ( (-X).rowwise() + X.row(i) );

		for(int j = 0; j< kernX.rows(); ++j){

			sumDerKernMat.row(j) += matx_iMinX.row(j) * kernX.row(i)[j] ;
		}

	}
	//sumDerKernMat = sumDerKernMat * (double) 1./ X.rows(); dont avg twice!
	sumDerKernMat = -2. / h_ * sumDerKernMat;
	return sumDerKernMat ;
}

template< typename FUNC >
Mat SVGD< FUNC >::gradSVGD( const Mat& X ){

	Mat gradient (X.rows(), X.cols() );

	delLogPMat_ = delLogP_( X );

	kernMat_        = kernMat( X );

	sumDerKernMat_  = sumDerKern( X, kernMat_);

	gradient = (double) 1. / X.rows() * ( kernMat_ * delLogPMat_ + sumDerKernMat_ );

	return gradient;
}



template< typename FUNC >
void SVGD< FUNC >::gradOptim(  int iter,  double nesterovAlpha, double nesterovMu ){

	nesterovAlpha_ = nesterovAlpha;
	nesterovMu_    = nesterovMu;
	iter_ 		   = iter;

	gradNormHistory.resize(iter, 1);
	pertNormHistory.resize(iter, 1);

	Mat vt      (Xn_.rows(), Xn_.cols() ); vt.setZero();
	Mat temp_vt (Xn_.rows(), Xn_.cols() ); temp_vt.setZero();

	double unstableNudge = 1e-8;

	for(int i = 0; i < iter; ++i){
		std::cout << "iteration: " << i<< std::endl;

		grad_ = gradSVGD( Xn_ + nesterovMu_ * vt );

		gradNormHistory(i, 0) = grad_.norm();

		std::cout << "gradNormHistory(i, 0)\n"<<  gradNormHistory(i, 0) << std::endl;

		vt    = nesterovMu_ * vt + nesterovAlpha_ * grad_ ;


		pertNormHistory(i, 0) = vt.norm();

		std::cout << "pertNormHistory(i, 0)\n"<<  pertNormHistory(i, 0) << std::endl;

		std::cout << "avg pertNormHistory(i, 0)\n"<< ( vt.colwise().norm() ).sum() * (double) 1./vt.rows() << std::endl;



		if(vt.array().isNaN().any() == true || vt.array().isInf().any() == true){
			std::cout << "\n\ncontains inf or nan" << std::endl;
			std::cout << "vt + unstableNudge\n" <<std::endl;
			vt = (temp_vt.array() + unstableNudge).matrix();
		}

		if(pertNormHistory(i, 0) > 0.1){
			std::cout << "\n\n\n pertb > 0.1\n" << pertNormHistory(i, 0) << std::endl;
			vt = 0.0001 * vt / vt.norm();
			pertNormHistory(i, 0) = vt.norm();
			std::cout << "\n\n\n corrected = \n" << pertNormHistory(i, 0) << std::endl;
		}

		Xn_ += vt;

		Xn_ = Xn_.unaryExpr([](double v) { return v > 0 ? v : 1e-6; });

		temp_vt = vt;

		std::cout << "\n\n";

	}

	return;
}


#endif // SVGD_HPP_


