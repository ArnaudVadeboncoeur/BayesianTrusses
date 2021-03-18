#ifndef SVGD_7_0_HPP_
#define SVGD_7_0_HPP_


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
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include "../../../src/matTools.hpp"


using Mat       = Eigen::MatrixXd;
using Vect      = Eigen::VectorXd;

using tup3Mat 		= std::tuple< Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd > ;
using FUNC_tup3Mat 	= std::function < tup3Mat (const Eigen::MatrixXd) >;

template<unsigned DIM>
class SVGD_7{

public:
	//Constructors and member initializer lists
	SVGD_7( FUNC_tup3Mat logPdelLogPFunc_){

		FowardModelFunc = logPdelLogPFunc_;

		kernVal = 0;

		diag_bandwidth.resize(DIM,DIM);
		diag_bandwidth.setIdentity();
		diag_bandwidth *= DIM;

		bandwidth = DIM;

	};



	void SVGDGrad(const Mat& X, Mat& gradMatrix)
	{

		logPdelLogPMatHess_ = FowardModelFunc( X );

		FMlogP			= std::get<0>( logPdelLogPMatHess_ );
		FMdellogP		= std::get<1>( logPdelLogPMatHess_ );
		HessAppFm		= std::get<2>( logPdelLogPMatHess_ );

		//New code -- SVGD -- an-iso kern SVN
		for(int i = 0; i < X.rows(); ++i){
			for(int j = 0; j < X.rows(); ++j){

				diff   = X.row(j) - X.row(i);
				kernM  = diff * HessAppFm;
				kernVal= std::exp(  (-1./(2. * X.cols() ) * kernM * diff.transpose())(0,0)  );
				dkern  = -2./(2. * X.cols() ) * kernM * kernVal;
				gradMatrix.row(i)  += kernVal * FMdellogP.row(j) + dkern;
		}}

		gradMatrix *=  (double) 1./X.rows();

		CrossEntropyHistory.conservativeResize(CrossEntropyHistory.rows() + 1, 1);
		CrossEntropyHistory(CrossEntropyHistory.rows() - 1, 0) = - FMlogP.mean();
		std::cout << "CrossEntropyHistory(i, 0)\t"<<  CrossEntropyHistory(CrossEntropyHistory.rows() - 1, 0) << std::endl;


	};

	void mixture_avg_SVGDGrad(const Mat& X, Mat& gradMatrix, bool diagonal_bd = false, bool compute_unbiased_KSD = true)
		{

			logPdelLogPMatHess_ = FowardModelFunc( X );

			FMlogP			= std::get<0>( logPdelLogPMatHess_ );
			FMdellogP		= std::get<1>( logPdelLogPMatHess_ );
			Mat Q	        = std::get<2>( logPdelLogPMatHess_ );
			Q = Q.setIdentity(Q.rows(),  Q.cols());

			Mat Qsqrt		= Q.sqrt();
			Mat Qinv		= Q.inverse();

			double unbiased_KSD = 0;
			double traceDellxx  = 0;
			Mat diffsvgd;
			Mat dkernSvgd;
			double kernSvgd;
			double h = DIM;

			double temp_bandwidth      = 0;
			Mat  temp_diag_bandwidth(DIM, DIM);
			temp_diag_bandwidth.setZero();

			Mat inv_diag_bandwidth = diag_bandwidth.inverse();

			avgDist = 0;

			//bandwidth = DIM;
			//New code -- SVGD -- an-iso kern SVN
			for(int i = 0; i < X.rows(); ++i){
				for(int j = 0; j < X.rows(); ++j){

					diff     = Qsqrt * X.row(j).transpose() - Qsqrt * X.row(i).transpose(); //n x 1

					diffsvgd =  X.row(j).transpose() -  X.row(i).transpose(); //n x 1
					// k(x, x') => k(X[j], X[i])

					//scalar bandwidth mean of norm of sample diff
					if( diagonal_bd == false){
						//std::cout << "this" << std::endl;
						temp_bandwidth      +=  (X.row(j) - X.row(i)).norm();
						avgDist             += (X.row(j) - X.row(i)).norm();
						kernVal= std::exp( -1./(2. * bandwidth ) * std::pow(diff.norm(), 2) )  ;
						dkern  = -1./bandwidth * diff.transpose() * kernVal;  //1xn x scalar
					}

					//diagonal matrix bandwidth mean of sample diff per dim
					if(diagonal_bd == true){
						temp_diag_bandwidth += ( X.row(j) - X.row(i) ).cwiseAbs().asDiagonal();
						avgDist             += (X.row(j) - X.row(i)).norm();
						//std::cout << temp_diag_bandwidth << "\n" <<std::endl;
						kernM = diff.transpose() * inv_diag_bandwidth;// nx1^T x nxn = 1xn
						kernVal= std::exp(  (-1./(2. * DIM ) * kernM * diff )(0,0)  ); // scalar
						dkern  = -1./DIM * kernM * kernVal;   // 1xn x scalar = 1xn
					}

					gradMatrix.row(i)  += kernVal * FMdellogP.row(j) + dkern;  //scalar x 1xn x 1xn = 1xn

					if(compute_unbiased_KSD == true ){//&& i != j){
						h = bandwidth;
						kernSvgd      = std::exp( -1./(2. * bandwidth ) * std::pow(diffsvgd.norm(), 2) )  ;
						dkernSvgd     = -1./ h * diffsvgd.transpose() * kernSvgd;

						traceDellxx   = -1. / h * kernSvgd * ( 1./h * (diffsvgd.transpose() * diffsvgd)(0,0) - (double) X.cols() );

						unbiased_KSD +=    (FMdellogP.row(j)  * kernSvgd * FMdellogP.row(i).transpose()
										 +  FMdellogP.row(j)  * -1.*dkernSvgd.transpose()
										 +  dkernSvgd * FMdellogP.row(i).transpose())(0,0)
										 +  traceDellxx ;
					}
			}}

			//unbiased_KSD *= (double) 1./( X.rows()*(X.rows() - 1.));
			unbiased_KSD *= (double) 1./( X.rows()*X.rows());

			std::cout << "D(mu||nu)^(-1) = " << 1./unbiased_KSD << std::endl;
			//alphaGlobal = 1./unbiased_KSD;

			std::cout << "unbiased_KSD\n" << unbiased_KSD << std::endl;

			bandwidth       = std::pow( (double) temp_bandwidth / ( X.rows() * X.rows() ), 2) / std::log(X.rows());
			diag_bandwidth  = (temp_diag_bandwidth * (double) 1. / ( X.rows() * X.rows() )).array().pow(2) * 1 / std::log(X.rows());

			avgDist = avgDist * (double) 1. / ( X.rows() * X.rows() );

			gradMatrix = ( Qinv * gradMatrix.transpose() * (double) 1./X.rows() ).transpose();

			CrossEntropyHistory.conservativeResize(CrossEntropyHistory.rows() + 1, 1);
			CrossEntropyHistory(CrossEntropyHistory.rows() - 1, 0) = - FMlogP.mean();
			std::cout << "CrossEntropyHistory(i, 0)\t"<<  CrossEntropyHistory(CrossEntropyHistory.rows() - 1, 0) << std::endl;

			KSDHistory.conservativeResize(KSDHistory.rows() + 1, 1);
			//KSDHistory(KSDHistory.rows() - 1, 0) =  unbiased_KSD ;
			KSDHistory(KSDHistory.rows() - 1, 0) =  std::log( unbiased_KSD );
			std::cout << "KSDHistory(i, 0)\t"<<  KSDHistory(KSDHistory.rows() - 1, 0) << std::endl;

		};




	double meanMagnitude(const Mat& matrix){

		double result;
		result = matrix.rowwise().norm().matrix().mean();

		return result;
	}


	void AdaMaxOptim(Mat & Xn, double alpha, double stopping_criteria, int maxIter = 10000){

		Mat svgdGrad  (Xn.rows(), Xn.cols() ); svgdGrad.setZero();
		Mat mt        (Xn.rows(), Xn.cols() ); mt.setZero();
		Mat ut        (Xn.rows(), Xn.cols() ); ut.setZero();

		Mat pert      (Xn.rows(), Xn.cols() ); pert.setZero();
		Mat temp_pert (Xn.rows(), Xn.cols() ); temp_pert.setZero();

		double avgNormLast = 9e9;
		double avgNormCurrent;

		double diff;
		alphaGlobal = alpha;

		double beta_1 = 0.9;
		double beta_2 = 0.999;
		double epsi   = 1e-8;

		for(int i = 0; i < Xn.rows(); ++i){
			for(int j = 0; j < Xn.rows(); ++j){
				bandwidth +=  (Xn.row(j) - Xn.row(i)).norm();
		}}
		bandwidth       = std::pow( (double) bandwidth / ( Xn.rows() * Xn.rows() ), 2) / std::log(Xn.rows());

		for(int i = 0; i < maxIter; ++i){

			std::cout << "Iteration : " << i << std::endl;

			//SVGDGrad(Xn, svgdGrad);
			mixture_avg_SVGDGrad(Xn, svgdGrad, false);

			if(svgdGrad.array().isNaN().any() == true || svgdGrad.array().isInf().any() == true){
				std::cout << "\n\ncontains inf or nan" << std::endl;
				std::cout << "AdaMax stopped\n" <<std::endl;
				std::cout << "svgdGrad\n" << svgdGrad << std::endl;
				svgdGrad = Previouse_svgdGrad;
				//break;
			}
			Previouse_svgdGrad = svgdGrad;

			mt     = beta_1 * mt + (1. - beta_1) * svgdGrad ;

			ut     = (beta_2 * ut).cwiseMax(svgdGrad.cwiseAbs());

			pert   = alphaGlobal / (1 - std::pow(beta_1, i+1)) * ( mt.array() / ut.array() ).matrix();



			Xn    += pert;

			//Xn    += 10 * svgdGrad;

			Xn = Xn.unaryExpr([](double v) { return v > 0 ? v : 1e-6; });

			critVal = meanMagnitude(svgdGrad);
			avgNormCurrent = critVal;
			diff = std::abs( (avgNormCurrent - avgNormLast) / avgNormLast * 100 );
			avgNormLast = avgNormCurrent;

//			if(i > 2){
//				diff = std::abs( (CrossEntropyHistory(i,0) - CrossEntropyHistory(i - 1,0)) / CrossEntropyHistory(i - 1,0) * 100 );
//			}
			if(i > 2){
				diff = std::abs( std::exp(KSDHistory(i,0)) - std::exp(KSDHistory(i - 1,0)) );
			}
			std::cout << "diff\n" << diff << std::endl;

			if(i > 2){
				std::cout << "% of change " << std::abs( (std::exp(KSDHistory(i  ,0)) - std::exp(KSDHistory(i - 1,0)))/ std::exp(KSDHistory(i - 1,0)))*100
				<< std::endl;
			}
			//store behaviour in matrices
			gradNormHistory.conservativeResize(gradNormHistory.rows() + 1, 1);
			gradNormHistory(gradNormHistory.rows() - 1, 0) = svgdGrad.norm();
			std::cout << "gradNormHistory(i, 0)\t"<<  gradNormHistory(i, 0) << std::endl;

			avgGradNormHistory.conservativeResize(avgGradNormHistory.rows() + 1, 1);
			avgGradNormHistory(avgGradNormHistory.rows() - 1, 0) = critVal ;
			std::cout << "avgGradNormHistory(i, 0)\t"<<  avgGradNormHistory(i, 0) << std::endl;

			pertNormHistory.conservativeResize(pertNormHistory.rows() + 1, 1);
			pertNormHistory(pertNormHistory.rows() - 1, 0) = pert.norm();
			std::cout << "pertNormHistory(i, 0)\t"<<  pertNormHistory(i, 0) << std::endl;

			XMeanHistory.conservativeResize(XMeanHistory.rows() + 1, 1);
			XMeanHistory(XMeanHistory.rows() - 1, 0) = Xn.mean();
			std::cout << "XMeanHistory(i, 0)\t"<<  XMeanHistory(i, 0) << std::endl;

			std::cout << "\n\n\n" << std::endl;

//			if( i > 2 && diff < stopping_criteria){
//				std::cout << "diff < stopping_criteria\n"
//				<< diff << " < " << stopping_criteria << std::endl;
//				return;
//			}
			if( i > 3
			&& std::abs( (std::exp(KSDHistory(i  ,0)) - std::exp(KSDHistory(i - 1,0)))/ std::exp(KSDHistory(i - 1,0)))*100 < stopping_criteria
			&& std::abs( (std::exp(KSDHistory(i-1,0)) - std::exp(KSDHistory(i - 2,0)))/ std::exp(KSDHistory(i - 2,0)))*100 < stopping_criteria
			&& std::abs( (std::exp(KSDHistory(i-2,0)) - std::exp(KSDHistory(i - 3,0)))/ std::exp(KSDHistory(i - 3,0)))*100 < stopping_criteria
				){
					std::cout << "diff < stopping_criteria\n for 3 consectuive iterations\n"
					<< "3 consecutive relative diff" << " < " << stopping_criteria << std::endl;
					return;
			}

		}

	}

	Mat gradNormHistory;
	Mat avgGradNormHistory;
	Mat pertNormHistory;
	Mat XMeanHistory;
	Mat CrossEntropyHistory;
	Mat KSDHistory;

	double alphaGlobal;


	Mat HessAppFm;
	double bandwidth ;
	Mat diag_bandwidth;
	double avgDist;

	//! destructor
	~SVGD_7( ) { };


private:

	FUNC_tup3Mat FowardModelFunc;

	tup3Mat logPdelLogPMatHess_;
	Mat FMlogP;
	Mat FMdellogP;





	Mat diff;
	double kernVal;
	Mat kernM;
	Mat dkern;

	Mat Previouse_svgdGrad;

	double critVal = -9e20;

};



#endif // SVGD_7_0_HPP_
