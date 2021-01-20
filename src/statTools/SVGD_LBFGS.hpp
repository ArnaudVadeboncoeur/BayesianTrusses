#ifndef SVGD_LBGFS_HPP_
#define SVGD_LBGFS_HPP_


#include <utility>
#include <functional>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>
#include <LBFGSB.h>


using Mat       = Eigen::MatrixXd;
using Vect      = Eigen::VectorXd;

//for CE eval
using tupMatMat 		= std::tuple< Eigen::MatrixXd, Eigen::MatrixXd > ;
using FUNC_tupMatMat 	= std::function < tupMatMat (const Eigen::MatrixXd) >;

class SVGD_LBFGS{

public:
	SVGD_LBFGS() {}


	void InitSamples( const Mat& samples  ) { Xn = samples; return; }


	Mat getSamples( ) { return Xn;}

	Mat gradNormHistory;
	Mat pertNormHistory;
	Mat XNormHistory;
	Mat CrossEntropyHistory;

	Mat   Xn;

	//! destructor
	~SVGD_LBFGS( ) { }


private:
	Mat    kernMat    ( const Mat& X );
	Mat    sumDerKern ( const Mat& X, const Mat& kernX );


	double h_;

};




Mat SVGD_LBFGS::kernMat( const Mat& X ){

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


Mat SVGD_LBFGS::sumDerKern( const Mat& X, const Mat& kernX ){

	Mat sumDerKernMat(X.rows(), X.cols());
	sumDerKernMat.setZero();
	Mat matx_iMinX(X.rows(), X.cols());

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


























#endif // SVGD_LBGFS_HPP_
