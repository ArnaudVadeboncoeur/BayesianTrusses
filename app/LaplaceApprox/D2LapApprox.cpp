/*
 * BTrussMCMC_MH.cpp
 *
 *  Created on: 12 Dec 2019
 *      Author: arnaudv
 */

#include "../../src/FEMClass.hpp"

#include <Eigen/Dense>

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

    Vec priorMeans(Dim); priorMeans << 0.03, 0.03, 0.03;
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
    std::cout << valMax << '\n' << xMax << std::endl;

//Computing Entries of Hessian----------------------------------------------

    unsigned hessDim = 2;
    Eigen::MatrixXd negLogHess(hessDim, hessDim);
    negLogHess.setZero();

    using Func = std::function <double( int, double)>;

    double h = 0.00001;
    Eigen::VectorXd x1(Dim);
    Eigen::VectorXd x2(Dim);
    Eigen::VectorXd x3(Dim);
    Eigen::VectorXd x4(Dim);
    double y1,y2,y3,y4;

    for(int i = 0; i < negLogHess.rows(); ++i){
        for(int j = 0; j<negLogHess.cols(); ++j){

            if( i > j){
                negLogHess(i,j) = negLogHess(j,i);
                std::cout << " Here1" << std::endl;
            }

            else if(i == j){
                 x1 = xMax; x1[i] -=  h ;
                 x2 = xMax; x2[i] +=  h ;
                 y1 = PostFunc.Eval(x1);
                 y2 = PostFunc.Eval(x2);
                 negLogHess(i, i) = (  y1  - 2 *  valMax  +  y2 )  / std::pow(h, 2);
                 std::cout << " Here2" << std::endl;

            }

            else if(i != j){
                x1 = xMax;
                x1[i] += h ;
                x1[j] += h ;
                y1 = PostFunc.Eval(x1);

                x2 = xMax;
                x2[i] += h ;
                x2[j] -= h ;
                y2 = PostFunc.Eval(x2);

                x3 = xMax;
                x3[i] -= h ;
                x3[j] += h ;
                y3 = PostFunc.Eval(x3);

                x4 = xMax;
                x4[i] -= h ;
                x4[j] -= h ;
                y4 = PostFunc.Eval(x4);

                negLogHess(i, j) = ( y1  - y2 - y3 +  y4 )  / ( 4. * std::pow(h, 2) );
                std::cout << " Here3" << std::endl;

            }
        }
    }

    negLogHess = -1. * negLogHess;
    std::cout << negLogHess << std::endl;
   // std::cout << negLogHess.inverse() << std::endl;
    Eigen::MatrixXd stdLaplaceInv = negLogHess;
    Eigen::MatrixXd stdLaplace = negLogHess.inverse();
    //std::cout << std::sqrt( negLogHess.inverse()(0,0) ) << std::endl;

//Eval True Pdf to plot ---------------------------------------------------------

    Eigen::VectorXd xPost(3); xPost << 0.00,  0.00,  0.02;

    double LikVals;
    double maxVal = -9e30;
    Eigen::VectorXd maxXPost(2); maxXPost << 0., 0.;
    double Vol = 0;

    std::ofstream myFile2;
    myFile.open("pdfResults.dat");

    double a = 0.0000001;
    double b = 0.15;

    double c = 0.0000001;
    double d = 0.15;

    int samplesX = 1e2;
    int samplesY = 1e2;

    double dx = (double) (b - a) / samplesX;
    double dy = (double) (d - c) / samplesY;

    //std::cout << dx << std::endl;

    Eigen::MatrixXd evaluations ( samplesX * samplesY , 3);

    unsigned ctr = 0;

    for(int i = 0; i < samplesX; ++i){

        xPost[0] = a + (double) dx * i ;

        for(int j = 0; j < samplesY; ++j){

            xPost[1] = c + (double) dy * j ;
            //x[1] = 0.04 ;

            LikVals =  PostFunc.Eval( xPost ) ;
            //std::cout << xPost << "\n" <<LikVals << std::endl << std::endl;

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
    std::ofstream myFile3;
    myFile3.open("pdfLaplceEval.dat");
    double probDensVal;
    std::cout << "-------------------" << '\n';
    std::cout << "LaplaceMap = " << LaplaceMAP << " stdLaplace = " << stdLaplace << std::endl;
    for(int i = 0; i < samplesX; ++i){

        xGauss[0] = a + (double) dx * i ;

        for(int j = 0; j < samplesX; ++j){

            xGauss[1] = c + (double) dy * j ;


            probDensVal = 1. / ( std::sqrt( pow(2*M_PI, 2) * negLogHess.inverse().determinant() )   ) *
                          std::exp( - 1./2. * (xGauss - LaplaceMAP).transpose() * negLogHess * (xGauss - LaplaceMAP)    );


            myFile3 << xGauss[0] << " " << xGauss[1] << " " << probDensVal << std::endl;
    }}
    myFile3.close();



    return 0;

}


