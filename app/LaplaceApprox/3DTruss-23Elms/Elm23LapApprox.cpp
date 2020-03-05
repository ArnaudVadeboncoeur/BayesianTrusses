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

    std::cout << "Here1" << std::endl;

    std::tuple<Eigen::MatrixXd, std::vector<double> > trueSamples;
    trueSampleGen( trueSamples );
    Eigen::MatrixXd trueSampleDisp = std::get<0>( trueSamples );


    using Vec  = Eigen::VectorXd;

    constexpr unsigned DimObs  = 18;
    constexpr unsigned DimPara =  2;
    constexpr unsigned NumTotPara =  3;


    double noiseLikStd = 0.0005;

    double logLikVal;

    std::ofstream myFile;
    myFile.open("results.dat");

    Eigen::VectorXi nodesFree(6); nodesFree << 1, 2, 3, 6, 7, 8;
        Eigen::VectorXi sampleDof( nodesFree.size() * 3 );
        for(int j = 0; j < nodesFree.size(); ++j){

            sampleDof[ j*3 + 0] = nodesFree[j]*3 + 0;
            sampleDof[ j*3 + 1] = nodesFree[j]*3 + 1;
            sampleDof[ j*3 + 2] = nodesFree[j]*3 + 2;
        }


    Vec priorMeans(DimPara); priorMeans.setConstant(0.025);

    PdfEval< DimObs, DimPara , Vec> PostFunc ( noiseLikStd, trueSampleDisp, sampleDof, priorMeans );

//simulated annealing to find MAP--------------------------------------------

    Vec x(DimPara); x.setConstant(0.001); std::cout<<"\n\n"<<x<<"\n\n";
    Vec xProp(DimPara);
    xProp = x;
    double yProp, yCurr, ratio;
    double randNum;

    yCurr = PostFunc.Eval( x );

    double valMax = -9e30;
    Eigen::VectorXd xMax(DimPara);

    std::normal_distribution<double> normal (0, 0.01);
    std::uniform_real_distribution<double> uniformDistribution( 0, 1 );

    std::random_device rd;
    std::mt19937 engine( rd() );

    for(int i = 0; i < 50; ++i){

        for(int j = 0; j < x.size(); ++j){
            xProp[j] = x[j] + normal ( engine );
        }

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

        for(int j = 0; j < x.size(); ++j){
            myFile << x[j] << " " ;
        }
        myFile << yCurr << '\n';
    }

    myFile.close();
    Eigen::VectorXd LaplaceMAP(DimPara) ; LaplaceMAP = xMax;


    std::cout << "valMax \n" << valMax << '\n' <<" xMax \n" << xMax << "\n\n";

//Computing Entries of Hessian----------------------------------------------

    Eigen::MatrixXd negLogHess(DimPara, DimPara);
    negLogHess.setZero();

    using Func = std::function <double( int, double)>;

    //double h = 0.00001;
    double h = 0.0005;

    Eigen::VectorXd x1(DimPara);
    Eigen::VectorXd x2(DimPara);
    Eigen::VectorXd x3(DimPara);
    Eigen::VectorXd x4(DimPara);
    double y1,y2,y3,y4;

    for(int i = 0; i < negLogHess.rows(); ++i){
        for(int j = 0; j<negLogHess.cols(); ++j){

            if( i > j){
                negLogHess(i,j) = negLogHess(j,i);
            }

            else if(i == j){
                 x1 = xMax; x1[i] -=  h ;
                 x2 = xMax; x2[i] +=  h ;
                 y1 = PostFunc.Eval(x1);
                 y2 = PostFunc.Eval(x2);
                 negLogHess(i, i) = (  y1  - 2 *  valMax  +  y2 )  / std::pow(h, 2);
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
            }
        }
    }

    negLogHess = -1. * negLogHess;
    std::cout << "LogHess\n" <<-1*negLogHess << std::endl;
   // std::cout << negLogHess.inverse() << std::endl;
    Eigen::MatrixXd stdLaplaceInv = negLogHess;
    Eigen::MatrixXd stdLaplace = negLogHess.inverse();
    //std::cout << std::sqrt( negLogHess.inverse()(0,0) ) << std::endl;

    std::cout << "Here2" << std::endl;


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
    double b = 0.2;

    double c = 0.0000001;
    double d = 0.2;

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


    std::cout << "LaplaceMap = \n" << LaplaceMAP << " \nstdLaplace = \n" << stdLaplace.sqrt() << std::endl;
    for(int i = 0; i < samplesX; ++i){

        xGauss[ 0 ] = a + (double) dx * ( i + 1) ;

        for(int j = 0; j < samplesY; ++j){


            xGauss[ 1 ] = c + (double) dy * ( j + 1) ;


            probDensVal = 1. / ( std::sqrt( pow(2*M_PI, 2) * negLogHess.inverse().determinant() )   ) *
                          std::exp( - 1./2. * (xGauss - LaplaceMAP).transpose() * negLogHess * (xGauss - LaplaceMAP) );
            std::cout << " negLogHess.inverse().determinant() " << negLogHess.inverse().determinant() <<std::endl;
            std::cout << " negLogHess " << negLogHess <<std::endl;


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


