/*
 * BTrussMCMC_MH.cpp
 *
 *  Created on: 12 Dec 2019
 *      Author: arnaudv
 */

#include "../../src/FEMClass.hpp"
#include "../../src/statTools/KLDiv.hpp"

#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>


#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <math.h>

#include "ThreeDTruss.hpp"
#include "sampleGen.hpp"
#include "PdfEval.hpp"

int main(){


    std::tuple<Eigen::MatrixXd, std::vector<double> > trueSamples = trueSampleGen();
    Eigen::MatrixXd trueSampleDisp = std::get<0>( trueSamples );

    using Vec  = Eigen::VectorXd;

    constexpr unsigned Dim = 3;

    Vec x(Dim); x << 0.0001, 0.0001, 0.02;

    double noiseLikStd = 0.0005;
    double logLikVal;

    std::ofstream myFile;
    myFile.open("results.dat");

    Vec priorMeans(Dim); priorMeans << 0.05, 0.03, 0.02;
    Eigen::VectorXi sampleDof(Dim); sampleDof << 9, 10, 11;

    PdfEval< Dim, Vec> PostFunc ( noiseLikStd, trueSampleDisp, sampleDof, priorMeans );

//simulated annealing to find MAP--------------------------------------------
    Vec xProp(Dim);
    xProp = x;
    double yProp, yCurr, ratio;
    double randNum;
    yCurr = PostFunc.Eval( x );
    double valMax = -9e30;
    Eigen::VectorXd xMax(Dim);

    std::normal_distribution<double> normal (0, 0.01);
    std::uniform_real_distribution<double> uniformDistribution( 0, 1 );

    std::random_device rd;
    std::mt19937 engine( rd() );

    for(int i = 0; i < 50; ++i){

        xProp[0] = x[0] + normal ( engine );
        xProp[1] = x[1] + normal ( engine );

        yProp = PostFunc.Eval( xProp );

        double t_0 = 1;
        double C = 0.8;
        double alpha =  ( yProp - yCurr ) / ( pow( ( t_0 * C), i ));
        ratio   = std::min( 0.0 ,  alpha  );

        randNum = uniformDistribution( engine ) ;

        if( std::log(randNum) < ratio ){

           x      = xProp;
           yCurr   = yProp;

           if ( yProp > valMax ){
                   valMax = yProp;
                   xMax    = x;
           }
        }
        myFile << x[0] << " " << x[1] << " " << yCurr << '\n';
    }
    myFile.close();
    Eigen::VectorXd LaplaceMAP(2) ; LaplaceMAP << xMax[0], xMax[1];
    //std::cout << valMax << '\n' << xMax << std::endl;

//Computing Entries of Hessian----------------------------------------------

    unsigned hessDim = 2;
    Eigen::MatrixXd negLogHess(hessDim, hessDim);
    negLogHess.setZero();

    using Func = std::function <double( int, double)>;

    double h = 0.001 ;

    Eigen::VectorXd ys(16);

    Eigen::MatrixXd shiftingH (16, 2);
    shiftingH << 2, 2,
                 2, 1,
                 2, -1,
                 2, -2,

                 1, 2,
                 1, 1,
                 1, -1,
                 1, -2,

                 -1, 2,
                 -1, 1,
                 -1, -1,
                 -1, -2,

                 -2, 2,
                 -2, 1,
                 -2, -1,
                 -2, -2;
    shiftingH = shiftingH * h;

    std::function <Eigen::VectorXd(int, int, double , double )> modMAP;
    modMAP = [ xMax ](int i, int j, double dx, double dy ){
        Eigen::VectorXd evalX;
        evalX = xMax;
        evalX[i] += dx; //std::cout << evalX[i];
        evalX[j] += dy; //std::cout << evalX[j] << std::endl;
        return evalX;
    };

    for(int i = 0; i < negLogHess.rows(); ++i){
        for(int j = 0; j<negLogHess.cols(); ++j){

            if( i > j){
                negLogHess(i,j) = negLogHess(j,i);
            }

            else if(i == j){

                 ys.setZero();
                 ys[0] = PostFunc.Eval( modMAP(i, j,   2*h,  0. ) );
                 ys[1] = PostFunc.Eval( modMAP(i, j,   h,    0. ) );
                 ys[2] = PostFunc.Eval( modMAP(i, j,   0.,   0. ) );
                 ys[3] = PostFunc.Eval( modMAP(i, j,  -h,    0. ) );
                 ys[4] = PostFunc.Eval( modMAP(i, j,  -2*h,  0. ) );

                 negLogHess(i, i) = (  -ys[0]  +  16 * ys[1]  -  30*ys[2]  +  16*ys[3]  -  ys[4] )  / ( 12. * h * h );

            }

            else if(i != j){

                ys.setZero();
                for(int k = 0; k < shiftingH.rows(); ++k){
                    ys[k] = PostFunc.Eval( modMAP(i, j,   shiftingH(k, 0),  shiftingH(k, 1) ) );
                    //std::cout << shiftingH(k, 0) << " " <<  shiftingH(k, 1) << " " << ys[k] << std::endl;
                }
                //std::cout << "\n\n "<< xMax << "\n\n" << ys << std::endl;

                negLogHess(i, j) = (    -    ( ys[0]  + 8*ys[1]  - 8*ys[2]  + ys[3] )
		                                + 8 *( ys[4]  + 8*ys[5]  - 8*ys[6]  + ys[7] )
		                                - 8 *( ys[8]  + 8*ys[9]  - 8*ys[10] + ys[11] )
		                                + 1 *( ys[12] + 8*ys[13] - 8*ys[14] + ys[15] )    ) 
		                                
		                                / (144. * h * h ) ;

            }
        }
    }

    Eigen::MatrixXd Iden(hessDim, hessDim);
    Iden.setIdentity();
    Iden = Iden * 1e-8;
    negLogHess = -1. * negLogHess ;
    Eigen::MatrixXd stdLaplaceInv = negLogHess;
    Eigen::MatrixXd stdLaplace = negLogHess.inverse();// + Iden;

//Eval True Pdf to plot ---------------------------------------------------------

    Eigen::VectorXd xPost(3); xPost << 0.00,  0.00,  0.02;

    double LikVals;
    double maxVal = -9e30;
    Eigen::VectorXd maxXPost(2); maxXPost << 0., 0.;
    double Vol = 0;

    std::ofstream myFile2;
    myFile.open("pdfResults.dat");

    double a = 0.0000001;
    double b = 0.1;

    double c = 0.0000001;
    double d = 0.1;

    int samplesX = 1e2;
    int samplesY = 1e2;

    double dx = (double) (b - a) / samplesX;
    double dy = (double) (d - c) / samplesY;

    Eigen::MatrixXd evaluations ( samplesX * samplesY , 3);

    unsigned ctr = 0;

    for(int i = 0; i < samplesX; ++i){

        xPost[0] = a + (double) dx * ( i + 1) ;

        for(int j = 0; j < samplesY; ++j){

            xPost[1] = c + (double) dy * ( j + 1) ;

            LikVals =  PostFunc.Eval( xPost ) ;

            if(LikVals > maxVal){
                maxVal = LikVals;
                maxXPost[0]   = xPost[0];
                maxXPost[1]   = xPost[1];
            }

            evaluations(ctr, 0) = xPost[0];
            evaluations(ctr, 1) = xPost[1];
            evaluations(ctr, 2) = LikVals;

            ctr++;
        }
    }

    for(int i = 0; i < evaluations.rows(); ++i){
        evaluations(i, 2) = std::exp(evaluations(i, 2) - maxVal) ;
        Vol += evaluations(i, 2) * dx * dy;
    }

    for(int i = 0; i < evaluations.rows(); ++i){
        evaluations(i, 2) = evaluations(i, 2) / Vol;
        myFile << evaluations(i, 0) << " " << evaluations(i, 1) << " " << evaluations(i, 2) << std::endl;
    }


    std::cout << "maxVal = " << maxVal <<"\n"<< " maxX = " << maxXPost[0] <<" "<< maxXPost[1] << std::endl;

//Eval Laplace Approx --------------------------------------

    Eigen::VectorXd xGauss (2);
    Eigen::MatrixXd EvalsLaplApp ( samplesX * samplesY , 3);
    std::ofstream myFile3;
    myFile3.open("pdfLaplceEval.dat");
    double probDensVal;
    std::cout << "-------------------" << '\n';

    int ctr2 = 0;


    std::cout << "LaplaceMap = \n" << LaplaceMAP << " \nstdLaplace = \n" << stdLaplace.sqrt() << std::endl;
    for(int i = 0; i < samplesX; ++i){

        xGauss[0] = a + (double) dx * ( i + 1) ;

        for(int j = 0; j < samplesY; ++j){

            xGauss[1] = c + (double) dy * ( j + 1) ;


            probDensVal = 1. / ( std::sqrt( pow(2*M_PI, 2) * negLogHess.inverse().determinant() )   ) *
                          std::exp( - 1./2. * (xGauss - LaplaceMAP).transpose() * negLogHess * (xGauss - LaplaceMAP)    );

            EvalsLaplApp(ctr2, 0) = xGauss[0];
            EvalsLaplApp(ctr2, 1) = xGauss[1];
            EvalsLaplApp(ctr2, 2) = probDensVal;
            myFile3 << xGauss[0] << " " << xGauss[1] << " " << probDensVal << std::endl;
            ctr2 ++;
    }}
    myFile3.close();


    std::cout << "KLDiv lapalce to True = "<<  KLDiv(EvalsLaplApp, evaluations) << std::endl;
    std::cout << "L2Norm lapalce to True = "<< L2Norm(EvalsLaplApp, evaluations) << std::endl;

    return 0;

}


