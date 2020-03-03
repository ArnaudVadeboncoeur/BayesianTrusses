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

#include "ThreeDTruss23Elm.hpp"
#include "sampleGen.hpp"
#include "PdfEval.hpp"

int main(){

    std::tuple<Eigen::MatrixXd, std::vector<double> > trueSamples;
    trueSampleGen( trueSamples );

    Eigen::MatrixXd trueSampleDisp = std::get<0>( trueSamples );

    constexpr unsigned DimObs  = 18;
    constexpr unsigned DimPara =  2;

    std::vector<int> paraIndex {0, 1};

    std::ofstream myFile;
    myFile.open("results.dat");
    myFile.close();


    Eigen::VectorXi nodesFree(6); nodesFree << 1, 2, 3, 6, 7, 8;
        Eigen::VectorXi sampleDof( nodesFree.size() * 3 );
        for(int j = 0; j < nodesFree.size(); ++j){

            sampleDof[ j*3 + 0] = nodesFree[j]*3 + 0;
            sampleDof[ j*3 + 1] = nodesFree[j]*3 + 1;
            sampleDof[ j*3 + 2] = nodesFree[j]*3 + 2;
        }

    //init prior information
    double noiseLikStd = 0.0005;

    Eigen::VectorXd priorMeans(DimPara); priorMeans.setConstant(0.0025);

    PdfEval< DimObs, DimPara , Eigen::VectorXd> PostFunc ( noiseLikStd, trueSampleDisp, sampleDof, priorMeans );

    Eigen::MatrixXd CovMatrixNoise (DimObs,DimObs);
    CovMatrixNoise.setIdentity();
    CovMatrixNoise = CovMatrixNoise * std::pow(noiseLikStd, 2);
    Eigen::MatrixXd CovMatrixNoiseInv = CovMatrixNoise.inverse();

    Eigen::MatrixXd PriorCovMatrix (DimPara,DimPara);
    PriorCovMatrix.setIdentity();
    PriorCovMatrix = PriorCovMatrix * 0.1;
    Eigen::MatrixXd PriorCovMatrixInv = PriorCovMatrix.inverse();

    //init FEM model
    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();
    FEMClass TrussFem(false, TrussDef );
    TrussFem.assembleS();
    Eigen::MatrixXd f = TrussFem.getForce();
    TrussFem.FEMClassReset(false);

//--------------------------------------------------------------------------------------

   //Lambda function to compute u(Theta)
    std::function < Eigen::VectorXd ( const Eigen::VectorXd) > uTheta;
    uTheta = [ &TrussFem ](const Eigen::VectorXd& X ){
        for(int j = 0; j < X.size(); ++j){
            TrussFem.modA(j, X[j] );
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
        for(int j = 0; j < X.size(); ++j){
            TrussFem.modA(j, X[j]);
        }
        TrussFem.assembleS();
        K = TrussFem.getK();
        TrussFem.FEMClassReset(false);

        return K;
    };

    //Lambda function to compute dK/dTheta_i
    std::function < Eigen::MatrixXd (Eigen::VectorXd, int) > dKdTheta_iFunc;
    dKdTheta_iFunc = [ &TrussFem, paraIndex ]( Eigen::VectorXd X, int index ){
        Eigen::MatrixXd dKdtheta_i;
        //produce  dKdTheta_i
        for(int j = 0; j < X.size(); ++j){
            TrussFem.modA(j, 0 );
        }
        TrussFem.modA(index, 1 );
        TrussFem.assembleS( );
        dKdtheta_i = TrussFem.getK();
        TrussFem.FEMClassReset(false);

        return dKdtheta_i;
    };

   //Labmda function to compute dudtheta_i
   std::function < Eigen::VectorXd      ( Eigen::VectorXd, Eigen::MatrixXd,
                                          Eigen::MatrixXd, Eigen::VectorXd, int ) > dudtheta_iFunc;
   dudtheta_iFunc = [ ]( const Eigen::VectorXd& X,          const Eigen::MatrixXd& K_inv,
                           const Eigen::MatrixXd& dKdtheta_b, const Eigen::VectorXd& u,
                           int indexTheta ){

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

           dudtheta_i = dudtheta_iFunc( X, K_inv, dKdTheta_iFunc( X, i), u, i );
           for(int j = 0; j < u.rows(); ++j ){
               dudTheta(j, i) = dudtheta_i(j);
               }
           }

       return dudTheta;
   };

   //Labmda function to compute du2_dthetab_Theta
      std::function < Eigen::MatrixXd    ( const Eigen::VectorXd,
                                            const Eigen::MatrixXd, const Eigen::VectorXd, int , int) > du2_dthetab_ThetaFunc;

      du2_dthetab_ThetaFunc = [ &dudtheta_iFunc, &dKdTheta_iFunc ]( const Eigen::VectorXd& X, const Eigen::MatrixXd& K_inv,
                                                                     const Eigen::VectorXd& u, int indexThetai , int indexThetaj){

          Eigen::MatrixXd du2_dthetab_Theta( K_inv.rows(),  X.rows() );

          Eigen::MatrixXd du2_dthetai_dthetaj( X.rows(), 1 );

          for(int i = 0; i < X.rows(); ++i ){

              du2_dthetai_dthetaj = dudtheta_iFunc( X, K_inv, dKdTheta_iFunc( X, i), u, i );
              for(int j = 0; j < u.rows(); ++j ){
                  du2_dthetab_Theta(j, i) = du2_dthetai_dthetaj(j);
                  }
              }

          return   du2_dthetai_thetaj;
      };

//Lambda function to compute d^2/dTheta^2 log P(Theta | y, Sig)


//Newton Ralphson to find MAP--------------------------------------------


    Eigen::VectorXd X = priorMeans;
    Eigen::MatrixXd k(DimObs, DimObs);
    Eigen::MatrixXd k_inv(DimObs, DimObs);
    Eigen::MatrixXd dk_dtheta(DimPara, DimPara);
    Eigen::MatrixXd y_i(trueSampleDisp.cols(), 1);
    Eigen::MatrixXd u( trueSampleDisp.cols(),  1);
    Eigen::MatrixXd du_dTheta( DimObs, DimPara  );

    Eigen::MatrixXd grad(1, DimPara);
    grad.setZero();


    //N-R iterations
    for(int i = 0; i < 1; ++i){

        k         = KThetaFunc ( X );
        k_inv     = k.inverse();
        u         = uTheta(X);
        du_dTheta = dudThetaFunc(X, k_inv, u );

        for(int j = 0; j < trueSampleDisp.rows(); ++j){

            for(int k = 0; k <trueSampleDisp.cols();++k ){
                y_i(k,0)= trueSampleDisp(j, k);
            }

            grad += (y_i - u).transpose() * CovMatrixNoiseInv * -1. * du_dTheta ;
        }

       //X = X - HessPost(X,k, dk_dtheta ).inverse() * gradPost(X,k, dk_dtheta );

    }
    std::cout << "grad" << grad << std::endl;

    return 0;
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

    Eigen::VectorXd xPost( DimPara ); xPost = LaplaceMAP;

    int dimEval = 2;
    double LikVals;
    double maxVal = -9e30;
    Eigen::VectorXd maxXPost(DimPara); maxXPost.setConstant(0);
    double Vol = 0;

    std::ofstream myFile2;
    myFile.open("pdfResults.dat");

    double a = 0.0000001;
    double b = 0.1;

    double c = 0.0000001;
    double d = 0.1;

    int samplesX = 1e2;
    int samplesY = 1e2;
    //int samplesY = 1;


    double dx = (double) (b - a) / samplesX;
    double dy = (double) (d - c) / samplesY;
    //double dy = 1.;


    //Eigen::MatrixXd evaluations ( samplesX * samplesY , 2);
    Eigen::MatrixXd evaluations ( samplesX * samplesY , dimEval + 1);


    unsigned ctr = 0;

    for(int i = 0; i < samplesX; ++i){

        xPost[ 0 ] = a + (double) dx * ( i + 1) ;

        for(int j = 0; j < samplesY; ++j){

           xPost[ 1 ] = c + (double) dy * ( j + 1) ;


            LikVals =  PostFunc.Eval( xPost ) ;

            if(LikVals > maxVal){
                maxVal = LikVals;
                maxXPost   = xPost;
            }

            evaluations(ctr, 0) = xPost[0];
            evaluations(ctr, 1) = xPost[1];
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
        myFile << evaluations(i, 0) << " " << evaluations(i, 1) << " " << evaluations(i, 2) << std::endl;
        //myFile << evaluations(i, 0) << " " << evaluations(i, 1) << std::endl;
    }


    std::cout << "maxVal = " << maxVal <<"\n"<< " maxX = \n" << maxXPost << " " << std::endl;

//Eval Laplace Approx --------------------------------------

    Eigen::VectorXd xGauss (DimPara); xGauss = LaplaceMAP;
    Eigen::MatrixXd EvalsLaplApp ( samplesX * samplesY , dimEval + 1);
    std::ofstream myFile3;
    myFile3.open("pdfLaplceEval.dat");
    double probDensVal;
    std::cout << "-------------------" << '\n';

    int ctr2 = 0;


    //std::cout << "LaplaceMap = \n" << LaplaceMAP << " \nstdLaplace = \n" << stdLaplace.sqrt() << std::endl;
    for(int i = 0; i < samplesX; ++i){

        xGauss[ 0 ] = a + (double) dx * ( i + 1) ;

        for(int j = 0; j < samplesY; ++j){


            xGauss[ 1 ] = c + (double) dy * ( j + 1) ;

            probDensVal = 0;
//            probDensVal = 1. / ( std::sqrt( pow(2*M_PI, 2) * covLaplaceAppInv.inverse().determinant() )   ) *
//                          std::exp( - 1./2. * (xGauss - LaplaceMAP).transpose() * covLaplaceAppInv * (xGauss - LaplaceMAP)    );

            EvalsLaplApp(ctr2, 0) = xGauss[0];
            EvalsLaplApp(ctr2, 1) = xGauss[1];
            EvalsLaplApp(ctr2, 2) = probDensVal;
            myFile3 << xGauss[ 0 ] << " " << xGauss[1] << " " << probDensVal << std::endl;
            //myFile3 << xGauss[0] << " " << probDensVal << std::endl;
            ctr2 ++;
        }
    }
    myFile3.close();


    std::cout << "KLDiv lapalce to True = "<< KLDiv(EvalsLaplApp, evaluations) << std::endl;
    std::cout << "L2Norm lapalce to True = "<<L2Norm(EvalsLaplApp, evaluations) << std::endl;

    return 0;

}


