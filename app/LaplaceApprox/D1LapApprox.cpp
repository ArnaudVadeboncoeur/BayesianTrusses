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

    Vec x(Dim); x << 0.0001, 0.04, 0.02;

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
        myFile << x[0] << " " << yCurr << '\n';
    }
    myFile.close();
    double LaplaceMAP = xMax[0];
    std::cout << valMax << '\n' << xMax << std::endl;

//Computing Entries of Hessian----------------------------------------------

    unsigned hessDim = 1;
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
    std::cout << negLogHess.inverse() << std::endl;
    double stdLaplace = negLogHess.inverse()(0,0);
    std::cout << std::sqrt( negLogHess.inverse()(0,0) ) << std::endl;

//Eval True Pdf to plot ---------------------------------------------------------

    double LikVals;
    double maxVal = 0;
    double maxX = 0;
    double Area = 0;

    std::ofstream myFile2;
    myFile2.open("pdfResults.dat");

    double a = 0.0001;
    double b = 0.5;
    Vec xPdf(Dim); x << 0.0001, 0.04, 0.02;

    int samples = 1e3;
    double dx = (double) (b - a) / samples;
    std::cout << dx << std::endl;

    Eigen::MatrixXd evaluations (samples, 2);

    for(int i = 0; i < samples; ++i){

        xPdf[0] = a + (double) dx * i ;
        //std::cout << xPdf[0] << " ";
        LikVals =  PostFunc.Eval(xPdf) ;
        //std::cout << LikVals << std::endl;

        if(LikVals > maxVal){
            maxVal = LikVals;
            maxX   = xPdf[0];
        }

        evaluations(i, 0) = xPdf[0];
        evaluations(i, 1) = LikVals;
    }

    for(int i = 0; i < evaluations.rows(); ++i){
        evaluations(i, 1) = std::exp( evaluations(i, 1) - maxVal );
        Area += evaluations(i, 1) * dx ;
    }

    for(int i = 0; i < evaluations.rows(); ++i){

        evaluations(i, 1) = evaluations(i, 1) / Area;
        myFile2 << evaluations(i, 0) << " " << evaluations(i, 1) <<std::endl;
    }

    std::cout << "maxVal = " << maxVal << " maxX = " << maxX << std::endl;
    myFile2.close();

//Eval Laplace Approx --------------------------------------

    double xGauss;
    std::ofstream myFile3;
    myFile3.open("pdfLaplceEval.dat");
    double probDensVal;
    std::cout << "-------------------" << '\n';
    std::cout << "LaplaceMap = " << LaplaceMAP << " stdLaplace = " << stdLaplace << std::endl;
    for(int i = 0; i < samples; ++i){

            xGauss = a + (double) dx * i ;
            probDensVal =  1./ std::sqrt(2. * M_PI * stdLaplace ) * std::exp( -1./(2.*stdLaplace) * pow( ( xGauss -  LaplaceMAP ), 2.)  );
            myFile3 << xGauss << " " << probDensVal << std::endl;
    }
    myFile3.close();



    return 0;

}


