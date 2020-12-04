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

//	std::cout << "\n\n\n\n"<<  std::endl;
//	std::cout << "kernMat_\n"<<  kernMat_ << std::endl;
//	std::cout << "delLogPMat_\n"<<  delLogPMat_ << std::endl;
//	std::cout << "sumDerKernMat_\n"<<  sumDerKernMat_ << std::endl;
//	std::cout << "gradient\n"<<  gradient << std::endl;
//	std::cout << "Xn_\n"<<  Xn_ << std::endl;
//	std::cout << "\n\n\n\n"<<  std::endl;


	return gradient;
}



template< typename FUNC >
void SVGD< FUNC >::gradOptim(  int iter,  double nesterovAlpha, double nesterovMu ){

	nesterovAlpha_ = nesterovAlpha;
	nesterovMu_    = nesterovMu;
	iter_ 		   = iter;

	gradNormHistory.resize(iter, 1);
	pertNormHistory.resize(iter, 1);

	Mat vt (Xn_.rows(), Xn_.cols() ); vt.setZero();

	for(int i = 0; i < iter; ++i){
		std::cout << "iteration: " << i<< std::endl;

		grad_ = gradSVGD( Xn_ + nesterovMu_ * vt);

		gradNormHistory(i, 0) = grad_.norm();
		pertNormHistory(i, 0) = grad_.norm();

		std::cout << "gradNormHistory(i, 0)\n"<<  gradNormHistory(i, 0) << std::endl;

		vt    = nesterovMu_ * vt + nesterovAlpha_ * grad_ ;

		//std::cout << "vt\n"<< vt << std::endl;

		Xn_ += vt;

		//Xn_ mus be > 0;
		//std::cout << "Before Xn_\n" << Xn_ << std::endl;
		Xn_ = Xn_.unaryExpr([](double v) { return v > 0? v : 1e-6; });
		//std::cout << "After Xn_\n" << Xn_ << std::endl;



	}

	return;
}


#endif // SVGD_HPP_


