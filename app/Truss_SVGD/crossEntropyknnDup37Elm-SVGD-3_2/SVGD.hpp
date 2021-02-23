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

//for CE eval
using tupMatMat = std::tuple< Eigen::MatrixXd, Eigen::MatrixXd > ;
using FUNC2 	= std::function < tupMatMat (const Eigen::MatrixXd) >;

template< typename FUNC >
class SVGD{
public:

	//! constructors
	SVGD( );// { }
	SVGD( FUNC delLogP ) {delLogP_ = delLogP;}

	void InitSamples( const Mat& samples  ) { Xn = samples; return; }

	void gradOptim_AdaMaxCE(  FUNC2 logPdelLogP, int iter,  double alpha = 1e-3, double ceRatio = 0.05, double crossEntropyValue = -1.,
							  double PertNormRatioStop = 1e-9, double gradNormStop = -1.);

	Mat getSamples( ) { return Xn;}

	//creat matrix of hard limits or constraints for SVGD

	Mat gradNormHistory;
	Mat pertNormHistory;
	Mat XNormHistory;
	Mat CrossEntropyHistory;

	Mat   Xn;

	//! destructor
	~SVGD( ) { }

private:

	Mat    gradSVGD( const Mat& X );

	Mat    kernMat    ( const Mat& X );
	Mat    sumDerKern ( const Mat& X, const Mat& kernX );

	FUNC  delLogP_;

	Mat   delLogPMat_;

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

	kernOfX = (- 1./ h_ *  kernOfX.array() *  kernOfX.array() ).matrix();
	kernOfX = kernOfX.array().exp().matrix() ;

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
void SVGD< FUNC >::gradOptim_AdaMaxCE( FUNC2 logPdelLogP, int iter,  double alpha, double ceRatio, double crossEntropyValue ,
									   double PertNormRatioStop, double gradNormStop){

	std::cout << "Init Params --  AdaMax Opt" << std::endl;

	iter_ 		   = iter;

	double beta_1 = 0.9;
	double beta_2 = 0.999;
	double epsi   = 1e-8;

	gradNormHistory.resize(0, 1);
	pertNormHistory.resize(0, 1);
	XNormHistory.resize(0, 1);

	CrossEntropyHistory.resize(0, 1);

	Mat mt        (Xn.rows(), Xn.cols() ); mt.setZero();
	Mat ut        (Xn.rows(), Xn.cols() ); ut.setZero();

	Mat pert      (Xn.rows(), Xn.cols() ); pert.setZero();
	Mat temp_pert (Xn.rows(), Xn.cols() ); temp_pert.setZero();

	Mat logP;
	Mat dellogP;
	tupMatMat logPdelLogPMat;

	double unstableNudge = 1e-8;

	double ceDiff = 1e9;

	for(int i = 0; i < iter; ++i){

		std::cout << "iteration: " << i<< std::endl;

		logPdelLogPMat = logPdelLogP( Xn );

		//grad_ =  - gradSVGD( Xn ); before

		dellogP = std::get<1>( logPdelLogPMat );
		logP  = std::get<0>( logPdelLogPMat );

		kernMat_        = kernMat( Xn );

		sumDerKernMat_  = sumDerKern( Xn, kernMat_);

		grad_ = - (double) 1. / Xn.rows() * ( kernMat_ * dellogP + sumDerKernMat_ );



		gradNormHistory.conservativeResize(gradNormHistory.rows() + 1, 1);
		gradNormHistory(gradNormHistory.rows() - 1, 0) = grad_.norm();

		std::cout << "gradNormHistory(i, 0)\n"<<  gradNormHistory(i, 0) << std::endl;

		CrossEntropyHistory.conservativeResize(CrossEntropyHistory.rows() + 1, 1);
		CrossEntropyHistory(CrossEntropyHistory.rows() - 1, 0) = - logP.mean();

		std::cout << "CrossEntropyHistory(i, 0)\n"<<  CrossEntropyHistory(i, 0) << std::endl;


		mt     = beta_1 * mt + (1. - beta_1) * grad_ ;

		ut     = (beta_2 * ut).cwiseMax(grad_.cwiseAbs());

		pert   = alpha / (1 - std::pow(beta_1, i+1)) * ( mt.array() / ut.array() ).matrix();

		pertNormHistory.conservativeResize(pertNormHistory.rows() + 1, 1);
		pertNormHistory(pertNormHistory.rows() - 1, 0) = pert.norm();


		if(pert.array().isNaN().any() == true || pert.array().isInf().any() == true){
			std::cout << "\n\ncontains inf or nan" << std::endl;
			std::cout << "AdaMax stopped\n" <<std::endl;
			return;
			pert = (temp_pert.array() + unstableNudge).matrix();
		}else{
			temp_pert = pert;
		}

		std::cout << "pertNormHistory(i, 0)\n"<<  pertNormHistory(i, 0) << std::endl;

		std::cout << "avg pertNormHistory(i, 0)\n"<< ( pert.colwise().norm() ).sum() * (double) 1./pert.rows() << std::endl;

		if(i > 0){
			ceDiff = std::abs( (CrossEntropyHistory(i, 0) - CrossEntropyHistory(i - 1, 0) )/ CrossEntropyHistory(i - 1, 0) );
		}


		if(CrossEntropyHistory(i, 0)  <  crossEntropyValue    ){
			std::cout << "CrossEntropyHistory(i, 0)  <  crossEntropyValue \n\n";
			std::cout << "crossEntropyValue  =" << crossEntropyValue << "\n";
			std::cout << "CrossEntropyHistory(i, 0)  =" << CrossEntropyHistory(i, 0) << "\n";
			break;
		}
		else if( gradNormStop < 0. && ceDiff  <  ceRatio    ){

			std::cout << "ceDiff  <  ceRatio \n\n";
			std::cout << "ceDiff  =" << ceDiff << "\n";
			std::cout << "ceRatio  =" << ceRatio << "\n";
			break;
		}else if( gradNormStop < 0. && pertNormHistory(i, 0)  <  PertNormRatioStop * pertNormHistory(0, 0)    ){
			std::cout << "pertNormHistory(i, 0)  <  PertNormRatioStop * pertNormHistory(0, 0) \n\n";
			break;
		}else if( gradNormHistory(i, 0)  <  gradNormStop  ){
			std::cout << "gradNormHistory(i, 0)  <  gradNormStop \n\n";
			break;
		}


		Xn -= pert;

		Xn = Xn.unaryExpr([](double v) { return v > 0 ? v : 1e-6; });


		XNormHistory.conservativeResize(XNormHistory.rows() + 1, 1);
		XNormHistory(XNormHistory.rows() - 1, 0) = Xn.norm();

		//XNormHistory(i, 0) = Xn.norm();

		std::cout << "XNormHistory(i, 0)\n"<<  XNormHistory(i, 0) << std::endl;

		std::cout << "\n\n";

	}

	return;
}

#endif // SVGD_HPP_


