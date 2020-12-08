/*
 * BTrussMCMC_MH.cpp
 *
 *  Created on: 12 Dec 2019
 *      Author: arnaudv
 */

#include "../../../src/FEMClass.hpp"
#include "../../../src/statTools/SVGD.hpp"
#include "../../../src/statTools/MVN.hpp"

#include "trueModelDataGen.hpp"
#include "../Truss37Elm.hpp"

#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <math.h>


int main(){

	using vecMat = std::vector< Eigen::MatrixXd > ;


    constexpr unsigned DimK       =  30 ;
    constexpr unsigned DimObs     =  20 ;
    constexpr unsigned DimPara    =  10 ;

    constexpr unsigned NumTotPara =  37;
    //these worked well --           {12, 13,14, 15, 16, 17  };
    //std::vector<int> paraIndex     { 0, 1, 2,3,4, 5};//, 7, 8, 9, 10, 11 };
    std::vector<int> paraIndex     { 12, 13,14, 15, 16, 17, 18, 19, 20, 21};// DimParam = 10
    //std::vector<int> paraIndex     { 13 , 16 };
    bool plot                      = false;
    bool             plot_1_dim    = false;
    std::vector<int> plotParaIndex {0, 1};

    //Index of dofs observed -- 2 = x and y only
    int Numxyz = 2;//1, 2, 3
    Eigen::MatrixXi nodesObs(1,  10 ); nodesObs <<   1, 2, 3,4,5,8, 9, 10, 11, 12;
        Eigen::VectorXi ObsIndex( nodesObs.size() * Numxyz );
        for(int j = 0; j < nodesObs.size(); ++j){

            ObsIndex[ j*Numxyz + 0] = nodesObs(0, j)*3 + 0;   //x
            ObsIndex[ j*Numxyz + 1] = nodesObs(0, j)*3 + 1;   //y
          //ObsIndex[ j*Numxyz + 2] = nodesObs(0, j)*3 + 2;   //z
        }


	DataCont trueSamplesTupleContainer = trueSampleGen( ObsIndex );

	Eigen::MatrixXd trueSampleDispC    = std::get<0>( trueSamplesTupleContainer );
	Eigen::MatrixXi ytL                = std::get<1>( trueSamplesTupleContainer );
	vecMat          trueForcingC       = std::get<2>( trueSamplesTupleContainer );
	double          sigma_n            = std::get<3>( trueSamplesTupleContainer );


//---------------------------------------Noise----------------------------------

	Eigen::MatrixXd cov_n (DimObs, DimObs);
	cov_n.setZero();
	for(int i = 0; i < DimObs; ++i){
		cov_n(i,i) = sigma_n * sigma_n;
	}
	Eigen::MatrixXd cov_n_Inv = cov_n.inverse();
//---------------------------------------Prior----------------------------------

    Eigen::VectorXd priorMeans(DimPara);
    priorMeans.setConstant(0.015);

    Eigen::MatrixXd PriorCovMatrix (DimPara,DimPara); PriorCovMatrix.setZero();
    //double sigma_p = 0.0025;
    double sigma_p = 0.0025;
    //97.5% += 3 * sigma_p
    Eigen::VectorXd priorStdVec(DimPara); priorStdVec.setConstant( sigma_p );
    for(int i = 0; i < priorStdVec.size(); ++i){

        PriorCovMatrix(i, i) = pow(priorStdVec[i], 2) ;//* 0.1;
    }

    Eigen::MatrixXd PriorCovMatrixInv = PriorCovMatrix.inverse();


//------------------------------------------------------------------------------
    //init FEM model
    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();
    FEMClass TrussFem(false, TrussDef );
    TrussFem.assembleS();
    std::vector<int> dofK = TrussFem.getFreeDof();
    TrussFem.FEMClassReset(false);

//--------------------------------------------------------------------------------------

   //Lambda function to compute u(Theta)
    std::function < Eigen::VectorXd ( const Eigen::MatrixXd, int) > uTheta;
    uTheta = [ &TrussFem, paraIndex, trueForcingC ](const Eigen::MatrixXd& X, int forcingIndex ){

    	TrussFem.FEMClassReset(false);
        for(int j = 0; j < paraIndex.size(); ++j){
            TrussFem.modA(paraIndex[j], X(j,0) );
        }

        TrussFem.modForce( trueForcingC[forcingIndex] );

        TrussFem.assembleS();

        TrussFem.computeDisp();
        Eigen::VectorXd u = TrussFem.getDisp();
        TrussFem.FEMClassReset(false);

        return u;
    };

   //Lambda function to compute K(Theta)
    std::function < Eigen::MatrixXd ( const Eigen::MatrixXd) > KThetaFunc;
    KThetaFunc = [ &TrussFem, paraIndex ](  const Eigen::MatrixXd& X ){
        Eigen::MatrixXd K;
        //produce k(theta)
        for(int j = 0; j < paraIndex.size(); ++j){
            TrussFem.modA(paraIndex[j], X(j,0));
        }
        TrussFem.assembleS();
        K = TrussFem.getK();
        TrussFem.FEMClassReset(false);

        return K;
    };

    //Lambda function to compute dK/dTheta_i
    std::function < Eigen::MatrixXd (Eigen::VectorXd, int) > dKdTheta_iFunc;
    dKdTheta_iFunc = [ &TrussFem, NumTotPara, paraIndex ]( Eigen::VectorXd X, int index ){
        TrussFem.FEMClassReset(false);
        Eigen::MatrixXd dKdtheta_i;
        //produce  dKdTheta_i
        for(int j = 0; j < NumTotPara; ++j){
            TrussFem.modA(j, 0 );
        }
        TrussFem.modA(paraIndex[index], 1 );
        TrussFem.assembleS( );
        dKdtheta_i = TrussFem.getK();
        TrussFem.FEMClassReset(false);

        return dKdtheta_i;
    };

   //Labmda function to compute dudtheta_i
   std::function < Eigen::VectorXd      ( const Eigen::VectorXd, const Eigen::MatrixXd,
                                          const Eigen::MatrixXd, const Eigen::VectorXd ) > dudtheta_iFunc;

   dudtheta_iFunc = [ ](   const Eigen::VectorXd& X,          const Eigen::MatrixXd& K_inv,
                           const Eigen::MatrixXd& dKdtheta_b, const Eigen::VectorXd& u ){

        Eigen::VectorXd dudtheta_b( X.rows() );
        dudtheta_b = - K_inv * dKdtheta_b * u ;

        return dudtheta_b;
    };

//Labmda function to compute dudTheta
   std::function < Eigen::MatrixXd ( const Eigen::VectorXd,
                                     const Eigen::MatrixXd, const Eigen::VectorXd ) > dudThetaFunc;

   dudThetaFunc = [ &dudtheta_iFunc, &dKdTheta_iFunc ]( const Eigen::VectorXd& X, const Eigen::MatrixXd& K_inv,
                                                        const Eigen::VectorXd& u){

       Eigen::MatrixXd dudTheta( K_inv.rows(),  X.rows() );

       Eigen::VectorXd dudtheta_i( X.rows() );

       for(int i = 0; i < X.rows(); ++i ){

           dudtheta_i = dudtheta_iFunc( X, K_inv, dKdTheta_iFunc( X, i), u );
           for(int j = 0; j < u.rows(); ++j ){
               dudTheta(j, i) = dudtheta_i(j);
               }
           }

       return dudTheta;
   };


	Eigen::MatrixXd L( DimObs , DimK ); L.setZero();
	   for(int i = 0; i < ObsIndex.size(); ++i ){
	           for( int j = 0; j < dofK.size(); ++j ){
	               if( dofK[j] == ObsIndex[i] ){
	                   L(i, j) = 1;
	                   break;
	               }
	           }
	       }

	//compute del_xLogP(x)

    using tupMatMat = std::tuple< Eigen::MatrixXd, Eigen::MatrixXd > ;
    using FUNCtupMatMat = std::function < tupMatMat (const Eigen::MatrixXd) >;
    using FUNC          = std::function < Eigen::MatrixXd (const Eigen::MatrixXd) >;

    FUNCtupMatMat delLogPtupMatMat;
    delLogPtupMatMat =
    		  [&TrussFem,    &KThetaFunc,     &dudThetaFunc,    &uTheta,
			   paraIndex,    DimK,DimObs,     DimPara,          ObsIndex,
			   trueForcingC, trueSampleDispC, ytL, L,
			   cov_n_Inv,    priorMeans,      PriorCovMatrixInv ]
			   (const  Eigen::MatrixXd& X ){

		Eigen::MatrixXd delLogPVar (X.rows(), X.cols()); delLogPVar.setZero();
		Eigen::MatrixXd LogPVar    (X.rows(), 1 );       LogPVar.setZero();

		Eigen::MatrixXd X_i (1, X.cols());

		Eigen::MatrixXd k(DimK, DimK);
		Eigen::MatrixXd k_inv(DimK, DimK);

		Eigen::MatrixXd u(DimObs, 1);
		Eigen::MatrixXd u_n(DimK , 1 );
		Eigen::MatrixXd du_dTheta ( DimObs, DimPara );

		bool firstEval = true;
		for(int i = 0; i < delLogPVar.rows(); ++i){

		  X_i                =     X.row(i).transpose() ;
		  k                  =     KThetaFunc ( X_i ) ;
		  k_inv              =     k.inverse();
		  delLogPVar.row(i) += - ( X_i - priorMeans ).transpose() * PriorCovMatrixInv;

		  LogPVar(i,0)    += -1./2. * std::log( std::pow((2. * M_PI), X.cols()) * PriorCovMatrixInv.inverse().determinant() )
						   - (1./2. * ( X_i - priorMeans ).transpose() * PriorCovMatrixInv * ( X_i - priorMeans )) (0,0)
						   - (1./2. * ytL.rows() * std::log( std::pow((2. * M_PI), X.cols()) * cov_n_Inv.inverse().determinant() ));

		  for(int j = 0 ; j < ytL.rows() ; ++j){

			  if(j > 0 ){ if(ytL(j, 0) == ytL(j - 1, 0)){ firstEval = false; } }

			  if(firstEval){
				  u_n         = uTheta(X_i, ytL(j, 0) ) ;
				  u           = L * uTheta(X_i, ytL(j, 0) );
				  du_dTheta   = L * dudThetaFunc(X_i, k_inv, u_n );
			  }
			  delLogPVar.row(i) -=  (trueSampleDispC.block(j,0,1, DimObs ) - u.transpose() ) * cov_n_Inv * -1. * du_dTheta;

			  LogPVar.row(i)    +=    -1./2. * (trueSampleDispC.block(j,0,1, DimObs ) - u.transpose() ) * cov_n_Inv *
					                	       (trueSampleDispC.block(j,0,1, DimObs ) - u.transpose() ).transpose();
			  firstEval = true;
			  if(X_i.minCoeff() <=0 ){

				  std::cout << "\n*Xval<0*\n" << X_i << std::endl;
				  //std::cout << "delLogPVar.row(i)\n" << delLogPVar.row(i) << std::endl;
			  }
		  }
		}
	tupMatMat results = std::make_tuple(LogPVar, delLogPVar);
	return results;
	};

    FUNC delLogPSVGD;
    delLogPSVGD = [ &delLogPtupMatMat ]
			      (const  Eigen::MatrixXd& X  ){

    	return std::get<1>(delLogPtupMatMat(X));

    };


//	Eigen::MatrixXd testX (1, DimPara); testX.setConstant(0.03);
//	Eigen::MatrixXd delLogPMat = std::get<1>( delLogP(testX) );
//	std::cout << delLogPMat << std::endl;
//	Eigen::MatrixXd  diff(1,1) ; diff << 1e-5;
//	std::cout << " Numderivative = "<< ( std::get<0>( delLogP(testX + diff) ) - std::get<0>( delLogP(testX - diff) ) ) * 1./2. * diff.inverse()
//		      << std::endl;


//	Eigen::IOFormat spaceSep(
//			int _precision=Eigen::StreamPrecision,
//			int _flags=0,
//			const std::string &_coeffSeparator=" ",
//			const std::string &_rowSeparator="\n",
//			const std::string &_rowPrefix="",
//			const std::string &_rowSuffix="",
//			const std::string &_matPrefix="",
//			const std::string &_matSuffix="");

	Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision,
								 Eigen::DontAlignCols,
								 " ", "\n",
								 "", "", "", "");

	MVN mvn( priorMeans , PriorCovMatrix  );
	Eigen::MatrixXd Xinit = mvn.sampleMVN( 100 );

	std::ofstream myFilePriorSamples;
	myFilePriorSamples.open("priorSamples.dat", std::ios::trunc);
	myFilePriorSamples << Xinit.format(CommaInitFmt) ;
	myFilePriorSamples.close();

	Eigen::MatrixXd delLogPMat = std::get<1>( delLogPtupMatMat(Xinit) );

	SVGD< FUNC > svgd(delLogPSVGD);
	svgd.InitSamples( Xinit );
	//svgd.gradOptim(75, 5*1e-8);
	//svgd.gradOptim(300, 1e-8);
	svgd.gradOptim(1000, 1 * 1e-12, 0.9);
	Mat X = svgd.getSamples();

	std::ofstream myFilePostSamples;
	myFilePostSamples.open("postSamples.dat", std::ios::trunc);

	myFilePostSamples << X.format(CommaInitFmt) ;
	myFilePostSamples.close();

	//Mat gradHist = svgd.gradNormHistory;

	std::ofstream gradHist;
	gradHist.open("gradHist.dat", std::ios::trunc);

	gradHist << svgd.gradNormHistory.format(CommaInitFmt) ;
	gradHist.close();

	std::ofstream pertHist;
	pertHist.open("pertist.dat", std::ios::trunc);

	pertHist << svgd.pertNormHistory.format(CommaInitFmt) ;
	pertHist.close();




//Eval True Pdf to plot ---------------------------------------------------------
	//bool plot = true;
	if( ! plot ){ return 0;}


	std::cout << "Computing scatter points true pdf " <<
			"\n-----------------------------------------------" << '\n';

    Eigen::VectorXd xPost( DimPara ); xPost.setZero();

	std::ofstream myEvalFile;
	myEvalFile.open("pdfResults.dat");

	double a = 0.0025;//-0.08;
	double b = 0.01;

	double c = 0.0075;
	double d = 0.015;

	int samplesX = 1 * 1e2;

	int samplesY = 1 * 1e2;
	if (plot_1_dim) {samplesY = 1;}


	double dx = (double) (b - a) / samplesX;
	double dy = (double) (d - c) / samplesY;
	if (plot_1_dim) {dy = 1;}

	double bottomLim = 1e-3;

	Eigen::MatrixXd evalX( samplesX * samplesY, DimPara );


	for(int i = 0; i < DimPara; ++i){

		evalX.col(i).setConstant( X.col(i).maxCoeff() );
	}

	int ctr = 0;
	for(int i = 0; i < samplesX ; ++i){
		for(int j = 0; j < samplesY; ++j){

			evalX(ctr, plotParaIndex[0] ) = a + i * dx;
			if (!plot_1_dim) { evalX(ctr, plotParaIndex[1]) = c + j * dy; }
			ctr ++;

		}
	}

	Eigen::MatrixXd delLogPEvals = std::get<0>(  delLogPtupMatMat(evalX) );

	double max = delLogPEvals.maxCoeff();

	delLogPEvals = (delLogPEvals.array() - max).matrix();


	delLogPEvals = delLogPEvals.array().exp().matrix();

	double Vol = (delLogPEvals * dx * dy).sum();

	delLogPEvals = delLogPEvals / Vol ;

	for(int i = 0; i < evalX.rows(); ++i){

		for(int j = 0; j < plotParaIndex.size(); j++){

			myEvalFile << evalX(i, plotParaIndex[j]) << " " ;
		}
		if( delLogPEvals(i, 0) < bottomLim ) { delLogPEvals(i, 0) = 0 ; }

		myEvalFile << delLogPEvals(i, 0) << "\n";
	}



   return 0;

}
