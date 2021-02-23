#ifndef SVGD_LBGFS2_HPP_
#define SVGD_LBGFS2_HPP_


#include <utility>
#include <functional>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>

#include "../matTools.hpp"



using Mat       = Eigen::MatrixXd;
using Vect      = Eigen::VectorXd;

using tupMatMat 		= std::tuple< Eigen::MatrixXd, Eigen::MatrixXd > ;
using FUNC_tupMatMat 	= std::function < tupMatMat (const Eigen::MatrixXd) >;

class SVGD_LBFGS{

public:
	//Constructors and member initializer lists
	SVGD_LBFGS( FUNC_tupMatMat logPdelLogPFunc_, int i_, int j_, std::ofstream& CEHistFile_, std::ofstream& stateOfX_)
	: FowardModelFunc( logPdelLogPFunc_ ),
	  matRows(i_), matCols(j_) , CEHistFile(CEHistFile_), stateOfX(stateOfX_)
	{}
	double operator()( Vect& vX, Vect& grad){

		Mat gradMatrix;

		Mat X = Eigen::Map<Eigen::MatrixXd> (vX.data(), matCols, matRows );
	    X.transposeInPlace();

		Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision,
									 Eigen::DontAlignCols,
									 " ", "\n",
									 "", "", "", "");
		stateOfX.open("statofX.dat", std::ios::out | std::ios::trunc);
		stateOfX << X.format(CommaInitFmt);
		stateOfX.close();

	    //std::cout << "X\n" << X << std::endl;

		double fx = 0.;

		logPdelLogPMat_ = FowardModelFunc( X );

		FMlogP			= std::get<0>( logPdelLogPMat_ );
		FMdellogP		= std::get<1>( logPdelLogPMat_ );

		kernMat_        = kernMat( X );

		sumDerKernMat_  = sumDerKern( X, kernMat_);

		gradMatrix      = - (double) 1. / X.rows() * ( kernMat_ * FMdellogP + sumDerKernMat_ );

		grad = matTools::ravelMatrixXdRowWiseToVectorXd(gradMatrix);


		crossEntropy_ = - FMlogP.mean();
		fx = crossEntropy_;

		std::cout << "f(x) = " << fx << std::endl;

		CEHistFile.open("CEHist.dat", std::ios::out | std::ios::app);
		CEHistFile << fx << "\n";
		CEHistFile.close();


		return fx;
	}

	Mat gradNormHistory;
	Mat pertNormHistory;
	Mat XNormHistory;
	Mat CrossEntropyHistory;

	//! destructor
	~SVGD_LBFGS( ) { }


private:

	int matRows;
	int matCols;
	std::ofstream& CEHistFile;
	std::ofstream& stateOfX;

	FUNC_tupMatMat FowardModelFunc;

	Mat    kernMat    ( const Mat& X );
	Mat    sumDerKern ( const Mat& X, const Mat& kernX );

	tupMatMat logPdelLogPMat_;
	Mat FMlogP;
	Mat FMdellogP;

	Mat kernMat_;
	Mat sumDerKernMat_;


	double h_ = 0;

	double crossEntropy_ = 0;

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


#endif // SVGD_LBGFS2_HPP_
