/*
 * BTrussMCMC_MH.cpp
 *
 *  Created on: 12 Dec 2019
 *      Author: arnaudv
 */

#include "../../../src/FEMClass.hpp"
#include "../../../src/statTools/KLDiv.hpp"

#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>


#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <math.h>

#include "ThreeDTruss37Elm.hpp"
//#include "ThreeDTruss23Elm.hpp"
//#include "ThreeDTruss3Elm.hpp"

#include "sampleGen.hpp"
#include "PdfEval.hpp"

int main(){

    std::tuple<Eigen::MatrixXd, std::vector<double> > trueSamples;
    trueSampleGen( trueSamples );

    Eigen::MatrixXd trueSampleDisp = std::get<0>( trueSamples );

    std::cout << "Here -- DTRUSSDEF37Elm_HPP_ worked" << std::endl;

    constexpr unsigned DimK       =  30;
    //constexpr unsigned DimK       =  3;
    constexpr unsigned DimObs     =  12 ;//1 node 3->x,y,z
    constexpr unsigned DimPara    =  2 ;

    constexpr unsigned NumTotPara =  37;

    std::vector<int> paraIndex     {13, 16};
    std::vector<int> plotParaIndex {0, 1};


    std::cout << "Here-main" << std::endl;

    //Index of dofs observed
    Eigen::VectorXi nodesObs( 4 ); nodesObs <<   2, 4, 9, 11;
    //Eigen::VectorXi nodesObs( 1 ); nodesObs <<  1;
        Eigen::VectorXi ObsIndex( nodesObs.size() * 3 );
        for(int j = 0; j < nodesObs.size(); ++j){

            ObsIndex[ j*3 + 0] = nodesObs[j]*3 + 0;
            ObsIndex[ j*3 + 1] = nodesObs[j]*3 + 1;
            ObsIndex[ j*3 + 2] = nodesObs[j]*3 + 2;
        }

    //Eigen::VectorXi ObsIndex( DimObs ); ObsIndex << 6, 7, 8, 27, 28, 29;
    //Eigen::VectorXi ObsIndex( 3 ); ObsIndex << 9, 10, 11;

    std::cout << "Here-2" << std::endl;

    //init prior information
    double noiseLikStd = 0.0003;

    Eigen::VectorXd priorMeans(DimPara); priorMeans.setConstant(0.06);
    double priorStd = 0.1;

    Eigen::MatrixXd PriorCovMatrix (DimPara,DimPara); PriorCovMatrix.setZero();
    //std::cout << PriorCovMatrix << std::endl;
        Eigen::VectorXd priorStdVec(DimPara); priorStdVec.setConstant(0.1);
        for(int i = 0; i < priorStdVec.size(); ++i){
            PriorCovMatrix(i, i) = pow(priorStdVec[i], 2) ;//* 0.1;
        }

    PdfEval< DimObs, DimPara , Eigen::VectorXd> PostFunc ( noiseLikStd, trueSampleDisp, paraIndex , ObsIndex, priorMeans, PriorCovMatrix );

    Eigen::MatrixXd CovMatrixNoise (DimObs,DimObs);
    CovMatrixNoise.setIdentity();
    CovMatrixNoise = CovMatrixNoise * std::pow(noiseLikStd, 2);
    Eigen::MatrixXd CovMatrixNoiseInv = CovMatrixNoise.inverse();

    std::cout << CovMatrixNoise << std::endl;

    Eigen::MatrixXd PriorCovMatrixInv = PriorCovMatrix.inverse();

    //init FEM model
    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();
    //List of dofs of reduced system
    FEMClass TrussFem(false, TrussDef );
    TrussFem.assembleS();
    std::vector<int> dofK = TrussFem.getFreeDof();
    Eigen::MatrixXd f = TrussFem.getForce();
    TrussFem.FEMClassReset(false);
    std::cout << "Here2_4" << std::endl;

//--------------------------------------------------------------------------------------

   //Lambda function to compute u(Theta)
    std::function < Eigen::VectorXd ( const Eigen::VectorXd) > uTheta;
    uTheta = [ &TrussFem, paraIndex ](const Eigen::VectorXd& X ){
        for(int j = 0; j < paraIndex.size(); ++j){
            TrussFem.modA(paraIndex[j], X[j] );
        }
        TrussFem.assembleS();
        TrussFem.computeDisp();
        Eigen::VectorXd u = TrussFem.getDisp();
        TrussFem.FEMClassReset(false);

        return u;
    };

   //Lambda function to compute K(Theta)
    std::function < Eigen::MatrixXd (Eigen::VectorXd) > KThetaFunc;
    KThetaFunc = [ &TrussFem, paraIndex ]( Eigen::VectorXd X ){
        Eigen::MatrixXd K;
        //produce k(theta)
        for(int j = 0; j < paraIndex.size(); ++j){
            TrussFem.modA(paraIndex[j], X[j]);
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
   std::function < Eigen::VectorXd      ( Eigen::VectorXd, Eigen::MatrixXd,
                                          Eigen::MatrixXd, Eigen::VectorXd ) > dudtheta_iFunc;
   dudtheta_iFunc = [ ](   const Eigen::VectorXd& X,          const Eigen::MatrixXd& K_inv,
                           const Eigen::MatrixXd& dKdtheta_b, const Eigen::VectorXd& u ){

        Eigen::VectorXd dudtheta_b( X.rows() );
        dudtheta_b = - K_inv * dKdtheta_b * u ;

        return dudtheta_b;
    };

//Labmda function to compute dudTheta
   std::function < Eigen::MatrixXd    ( const Eigen::VectorXd,
                                         const Eigen::MatrixXd, const Eigen::VectorXd ) > dudThetaFunc;

   dudThetaFunc = [ &dudtheta_iFunc, &dKdTheta_iFunc ]( const Eigen::VectorXd& X, const Eigen::MatrixXd& K_inv,
                                                        const Eigen::VectorXd& u){

       Eigen::MatrixXd dudTheta( K_inv.rows(),  X.rows() );

       Eigen::VectorXd dudtheta_i( X.rows() );

       for(int i = 0; i < X.rows(); ++i ){

           //std::cout << "dKdTheta_iFunc( X, i) \n" <<dKdTheta_iFunc( X, i) << "\n\n";
           dudtheta_i = dudtheta_iFunc( X, K_inv, dKdTheta_iFunc( X, i), u );
           for(int j = 0; j < u.rows(); ++j ){
               dudTheta(j, i) = dudtheta_i(j);
               }
           }

       return dudTheta;
   };


   //Labmda function to compute du2_dthetab_Theta
	std::function < Eigen::MatrixXd    ( const Eigen::VectorXd,
									   const Eigen::MatrixXd, const Eigen::VectorXd, int, int) > du2_dthetai_dthetajFunc;

	du2_dthetai_dthetajFunc = [ &dudtheta_iFunc, &dKdTheta_iFunc ]( const Eigen::VectorXd& X, const Eigen::MatrixXd& K_inv,
																                   const Eigen::VectorXd& u, int index_i, int index_j ){
	 Eigen::MatrixXd du2_dthetai_dthetaj( X.rows(), 1 );

	 Eigen::MatrixXd dudtheta_i(K_inv.rows(), 1 );
	 Eigen::MatrixXd dudtheta_j(K_inv.rows(), 1 );

	 Eigen::MatrixXd dK_dtheta_i(K_inv.rows(), K_inv.rows());
	 Eigen::MatrixXd dK_dtheta_j(K_inv.rows(), K_inv.rows());

	 dK_dtheta_i = dKdTheta_iFunc( X, index_i);
	 dK_dtheta_j = dKdTheta_iFunc( X, index_j);

	 dudtheta_i = dudtheta_iFunc( X, K_inv, dK_dtheta_i, u );
	 dudtheta_j = dudtheta_iFunc( X, K_inv, dK_dtheta_j, u );

	 du2_dthetai_dthetaj = - K_inv * ( dK_dtheta_i * dudtheta_j + dK_dtheta_j * dudtheta_i );

	 return du2_dthetai_dthetaj;
	};



	//Labmda function to compute du2_dthetab_Theta
	std::function < Eigen::MatrixXd    ( const Eigen::VectorXd,
										          const Eigen::MatrixXd, const Eigen::VectorXd, int) > du2_dthetab_ThetaFunc;

	du2_dthetab_ThetaFunc = [ &du2_dthetai_dthetajFunc ]( const Eigen::VectorXd& X, const Eigen::MatrixXd& K_inv,
																         const Eigen::VectorXd& u, int index_b ){

	  Eigen::MatrixXd du2_dthetab_Theta( K_inv.rows(),  X.rows() );
	  Eigen::MatrixXd du2_dthetai_dthetaj(  K_inv.rows(), 1 );

	  for(int i = 0; i < X.rows(); ++i ){

		  du2_dthetai_dthetaj = du2_dthetai_dthetajFunc( X, K_inv, u, index_b, i );
		  for(int j = 0; j < u.rows(); ++j ){
			  du2_dthetab_Theta(j, i) = du2_dthetai_dthetaj(j, 0);

			  }
		  }

	  return   du2_dthetab_Theta;
	};

////Labmda function to compute du2_dthetab_Theta
//    std::function < Eigen::MatrixXd    ( const Eigen::VectorXd,
//                                                  const Eigen::MatrixXd, const Eigen::VectorXd, int) > du2_dthetab_ThetaFunc;
//
//    du2_dthetab_ThetaFunc = [ &du2_dthetai_dthetajFunc ]( const Eigen::VectorXd& X, const Eigen::MatrixXd& K_inv,
//                                                                         const Eigen::VectorXd& u, int index_b ){
//        Eigen::MatrixXd
//
//
//    }

//Lambda function to compute d^2/dTheta^2 log P(Theta | y, Sig)


//Newton Ralphson to find MAP--------------------------------------------


    Eigen::VectorXd X(DimPara); X.setConstant(0.075);
    double Null = 1e-14 ;
    Eigen::MatrixXd k(DimK, DimK);
    Eigen::MatrixXd k_inv(DimK, DimK);
    Eigen::MatrixXd dk_dtheta(DimPara, DimPara);
    Eigen::MatrixXd y_i(trueSampleDisp.cols(), 1);
    Eigen::MatrixXd u  ( DimObs,  1 );
    Eigen::MatrixXd u_n( DimK  ,  1 );
    Eigen::MatrixXd du_dTheta( DimObs, DimPara );

    Eigen::MatrixXd du2_dthetab_dTheta( DimObs, DimPara  );
    Eigen::MatrixXd thetaHat_b ( DimPara, 1);
    Eigen::MatrixXd du_dtheta_b ( DimObs, 1  );

    Eigen::MatrixXd grad(1, DimPara);
    grad.setZero();


    Eigen::MatrixXd hess(DimPara, DimPara);
    Eigen::MatrixXd LaplaceHess_inv(DimPara, DimPara);
    hess.setZero();

    Eigen::MatrixXd hess_b( DimPara, 1 );
    hess.setZero();


    std::cout << "Not Done Creating L" << std::endl;
    //std::cout << KThetaFunc ( X ) << std::endl;
    std::cout <<"(KThetaFunc ( X )).rows() " << (KThetaFunc ( X )).rows() << std::endl;
    std::cout <<"(KThetaFunc ( X )).cols() " << (KThetaFunc ( X )).cols() << std::endl;
    //creat K to Obsversed Matrix L
    Eigen::MatrixXd L( DimObs , DimK ); L.setZero();



//    std::cout << "ObsIndex" << std::endl;
//    for(int i = 0; i < ObsIndex.size(); ++i){
//            std::cout  << ObsIndex[i]<<" " <<i << std::endl;
//        }
//    for(int i = 0; i < dofK.size(); ++i){
//        std::cout  << dofK[i]<<" "<< i << std::endl;
//    }


    //std::cout << "dofK" << std::endl;

    for(int i = 0; i < ObsIndex.size(); ++i ){
        for( int j = 0; j < dofK.size(); ++j ){
            if( dofK[j] == ObsIndex[i] ){
                L(i, j) = 1;
                break;
            }
        }
    }
    std::cout << "Done Creating L" << std::endl;
    std::cout << L << std::endl;
    std::cout << "L.rows() " << L.rows() << std::endl;
    std::cout << "L.cols() " << L.cols() << std::endl;

//    std::cout << "dofK\n";
//    for(int i = 0; i < dofK.size(); ++i){ std::cout << dofK[i] << std::endl;}
//    std::cout << "ObsIndex\n" << ObsIndex << std::endl;
//    std::cout << "L\n" << L << std::endl;


    std::ofstream NRFile;
    NRFile.open("Newton-RalphsonOpt.dat");
    for(int d = 0; d < plotParaIndex.size(); ++d){
               NRFile << X[ plotParaIndex[d] ] << " ";
           } NRFile << "\n";


    //N-R iterations
    int maxIter = 1e4;
    for(int i = 0; i < maxIter; ++i){

//        X[0] = 0.0528428;
//        X[1] = 0.0356403;

        k           = KThetaFunc ( X );
        //std::cout << "Computed K" << std::endl;
        k_inv       = k.inverse();
        u_n         = uTheta(X) ;
        //std::cout << "Computed u_n" << std::endl;
        //std::cout << "L \n" << L << std::endl;
        //std::cout << "\nu_n \n" << u_n << std::endl;

        u           = L * uTheta(X);
        //std::cout << "Computed u reduced" << std::endl;


        du_dTheta   = L * dudThetaFunc(X, k_inv, u_n );
        //std::cout << "Computed u etc." << std::endl;


        grad = - ( X - priorMeans ).transpose() * PriorCovMatrixInv;

        //std::cout << grad << " " << std::endl;
        //std::cout << "Computed grad prior term" << std::endl;

        for(int j = 0; j < trueSampleDisp.rows(); ++j){

            for(int k = 0; k <trueSampleDisp.cols();++k ){
                y_i(k,0)= trueSampleDisp(j, k);
            }
            //std::cout << u << "\n\n " << y_i<< "\n\n " << CovMatrixNoiseInv<< "\n\n " << du_dTheta << std::endl;
            grad -= (y_i - u).transpose() * CovMatrixNoiseInv * -1. * du_dTheta ;
        }

        bool gradNull; gradNull = true;
        for(int j = 0; j < grad.size(); ++j){
            if( std::abs( grad(0, j) ) > Null ){ gradNull = false; break; }
        }
        if( gradNull ){
            std::cout << "gradNull\n" << grad << std::endl;
            grad.setConstant(1e-10);
            break ;
        }

        //std::cout << "Computed grad" << std::endl;

//       X = X + hess.inverse() * grad.transpose();
       //if( i == maxIter - 1){ LaplaceHess_inv = -1 * hess.inverse(); }
       //X = X - 0.01*hess.inverse() * grad.transpose();


       X = X + 0.0000005 * grad.transpose();

       for(int j = 0; j < X.size(); ++j){
           if(X[j] < 0.){
               X[j] = 0;
           }
       }

       //X = X + 0.00005 * grad.transpose();


       //std::cout << "X \n" << X << "\n\n";

       for(int d = 0; d < plotParaIndex.size(); ++d){
           NRFile << X[ plotParaIndex[d] ] << " ";
       } NRFile << "\n";
    }

    for(int i = 0; i < paraIndex.size() ; ++i){

        thetaHat_b.setZero();
        thetaHat_b( i, 0 ) = 1;
        hess_b = -1 * thetaHat_b.transpose( ) * PriorCovMatrixInv;

      //du_dtheta_b = L * dudtheta_iFunc( X, k_inv, dKdTheta_iFunc (X,  paraIndex[i] ),  u_n )  ;
        du_dtheta_b = L * dudtheta_iFunc( X, k_inv, dKdTheta_iFunc (X,  i ),  u_n )  ;

        // compute du2_dthetab_dTheta and drop unobserved dofs

        //du2_dthetab_dTheta = L * du2_dthetab_ThetaFunc( X,  k_inv, u_n , paraIndex[i] );
        du2_dthetab_dTheta = L * du2_dthetab_ThetaFunc( X,  k_inv, u_n , i );


        for(int j = 0; j < trueSampleDisp.rows(); ++j){

            for(int k = 0; k < trueSampleDisp.cols();++k ){
                          y_i(k,0)= trueSampleDisp(j, k);
                      }

            hess_b -= (y_i - u).transpose() * CovMatrixNoiseInv * -1 * du2_dthetab_dTheta;
            hess_b -= du_dtheta_b.transpose() * CovMatrixNoiseInv * du_dTheta;

        }
        for(int d = 0; d < DimPara; ++d){

            hess( i, d) = hess_b( 0, d );
        }
    }
    LaplaceHess_inv = -1 * hess.inverse();

    std::cout << "Done Newton Ralphson" << std::endl;


    NRFile.close();
    Eigen::MatrixXd negLogHess = -1 * hess;
    //std::cout << "LogHess\n" <<-1*negLogHess << std::endl;
    Eigen::MatrixXd stdLaplaceInv = negLogHess;
    Eigen::MatrixXd stdLaplace = negLogHess.inverse();

    std::cout << "LaplaceMAP \n" << X << "\n\n";
    std::cout << "grad \n" << grad << "\n\n";
    std::cout << "hess \n" << hess << "\n\n";
    std::cout << "LaplaceHess_inv \n" << LaplaceHess_inv << "\n\n";
    //std::cout << "LaplaceHess_inv \n" << LaplaceHess_inv.sqrt() << "\n\n";
    //return 0;
    Eigen::VectorXd LaplaceMAP = X;
//Compute Matrix of 2nd der p(theta|y_i, Sig)----------------------------------------------

    //produce k(theta)
//    for(int j = 0; j < X.size(); ++j){
//        TrussFem.modA(j, X[j]);
//    }
//    TrussFem.assembleS();
//    k = TrussFem.getK();
//    TrussFem.FEMClassReset(false);
//
//    //produce dk(theta)/dtheta
//   for(int j = 0; j < X.size(); ++j){
//       TrussFem.modA(j, 1);
//   }
//   TrussFem.assembleS();
//   dk_dtheta = TrussFem.getK();
//   TrussFem.FEMClassReset(false);
//
//   Eigen::MatrixXd covLaplaceAppInv = -1. * HessPost(X,k, dk_dtheta );
//
//   std::cout << "LaplaceMAP  = \n" << LaplaceMAP << "\n\n";
//   std::cout << "covLaplaceApp  = \n" << covLaplaceAppInv.inverse().sqrt() << "\n\n";
//


//Eval True Pdf to plot ---------------------------------------------------------

    std::cout << "Computing scatter points true pdf \n-----------------------------------------------" << '\n';


    Eigen::VectorXd xPost( DimPara ); xPost = LaplaceMAP;

    int dimEval = 2;
    double LikVals;
    double maxVal = -9e30;
    Eigen::VectorXd maxXPost(DimPara); maxXPost.setConstant(0);
    double Vol = 0;

    std::ofstream myFile;
    myFile.open("pdfResults.dat");

    double a = 0.01;//-0.08;
    double b = 0.08;

    double c = 0.01;//-0.08;
    double d = 0.08;

    int samplesX = 1 * 1e2;
    int samplesY = 1 * 1e2;
    //int samplesY = 1;


    double dx = (double) (b - a) / samplesX;
    double dy = (double) (d - c) / samplesY;
    //double dy = 1.;

    double bottomLim = -9e40;

    //Eigen::MatrixXd evaluations ( samplesX * samplesY , 2);
    Eigen::MatrixXd evaluations ( samplesX * samplesY , dimEval + 1);

    unsigned ctr = 0;

    for(int i = 0; i < samplesX; ++i){

        //std::cout << "Here-Fault\n";
        xPost[ plotParaIndex[0] ] = a + (double) dx * ( i + 1) ;

        for(int j = 0; j < samplesY; ++j){

           xPost[ plotParaIndex[1] ] = c + (double) dy * ( j + 1) ;


            LikVals =  PostFunc.Eval( xPost ) ;
            //std::cout << LikVals << std::endl;


            if(LikVals > maxVal){
                maxVal = LikVals;
                maxXPost   = xPost;
            }

            evaluations(ctr, 0) = xPost[ plotParaIndex[0] ];
            evaluations(ctr, 1) = xPost[ plotParaIndex[1] ];
            evaluations(ctr, 2) = LikVals;

            ctr++;
        }
    }

    for(int i = 0; i < evaluations.rows(); ++i){
        evaluations(i, dimEval ) = std::exp(evaluations(i, dimEval ) - maxVal) ;
        //Vol += evaluations(i, 1) * dx;
        Vol += evaluations(i, dimEval) * dx * dy;
    }

    for(int i = 0; i < evaluations.rows(); ++i){
        evaluations(i, dimEval) = evaluations(i, dimEval) / Vol;
        //std::cout << evaluations(i, dimEval) << std::endl;
        if( evaluations(i, dimEval) > bottomLim ){
            myFile << evaluations(i, 0) << " " << evaluations(i, 1) << " " << evaluations(i, 2) << std::endl;
        }
        //myFile << evaluations(i, 0) << " " << evaluations(i, 1) << std::endl;
    }

    //std::cout << "Here-Fault\n";

    std::cout << "maxVal = " << maxVal <<"\n"<< " maxX = \n" << maxXPost << " " << std::endl;
    //return 0;

    std::cout << "Generated true pdf points" << std::endl;
//Eval Laplace Approx --------------------------------------

    Eigen::VectorXd xGauss (DimPara); xGauss = LaplaceMAP;
    Eigen::MatrixXd EvalsLaplApp ( samplesX * samplesY , dimEval + 1);
    std::ofstream myFile3;
    myFile3.open("pdfLaplaceEval.dat");
    double probDensVal;
    std::cout << "-------------------" << '\n';

    int ctr2 = 0;


    //std::cout << "LaplaceMap = \n" << LaplaceMAP << " LaplaceHess_inv.sqrt() = \n" << LaplaceHess_inv.sqrt() << std::endl;
    for(int i = 0; i < samplesX; ++i){

        xGauss[ plotParaIndex[0] ] = a + (double) dx * ( i + 1) ;

        for(int j = 0; j < samplesY; ++j){


            xGauss[ plotParaIndex[1] ] = c + (double) dy * ( j + 1) ;

            probDensVal = 1. / ( std::sqrt( pow(2*M_PI, 2) * negLogHess.inverse().determinant() )   ) *
                                      std::exp( - 1./2. * (xGauss - LaplaceMAP).transpose() * negLogHess * (xGauss - LaplaceMAP) );

            EvalsLaplApp(ctr2, 0) = xGauss[ plotParaIndex[0] ];
            EvalsLaplApp(ctr2, 1) = xGauss[ plotParaIndex[1] ];
            EvalsLaplApp(ctr2, 2) = probDensVal;
            //std::cout << xGauss[ 0 ] << " " << xGauss[1] << " " << probDensVal << std::endl;

            if( probDensVal > bottomLim ){
                myFile3 << xGauss[ plotParaIndex[0] ] << " " << xGauss[ plotParaIndex[1] ] << " " << probDensVal << std::endl;
            }
            //myFile3 << xGauss[0] << " " << probDensVal << std::endl;
            ctr2 ++;
        }
    }
    myFile3.close();

    std::cout << "KLDiv lapalce to True = "<< KLDiv(EvalsLaplApp, evaluations) << std::endl;
    std::cout << "L2Norm lapalce to True = "<<L2Norm(EvalsLaplApp, evaluations) << std::endl;

    return 0;

}


