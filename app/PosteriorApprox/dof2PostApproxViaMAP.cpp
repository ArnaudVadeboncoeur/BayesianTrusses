/*
 * dof2postEval.cpp
 *
 *  Created on: 3 Feb 2020
 *      Author: arnaudv
 */



#include "../../src/FEMClass.hpp"

#include "../../src/statTools/SampleAnalysis.hpp"
#include "../../src/statTools/histSort.hpp"
#include "../../src/statTools/KLDiv.hpp"
#include "../../src/statTools/SteinDiscrepancy.hpp"

#include "../../src/statTools/MCMC2.hpp"

//#include "../../src/statTools/MCMC3-SimAnnealing.hpp"



#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <math.h>

#include "ThreeDTruss.hpp"

std::tuple<Eigen::MatrixXd, std::vector<double> > trueSampleGen(){

    bool verbosity = false;

    std::ofstream myTrueFile;
    myTrueFile.open("trueResults.dat", std::ios::trunc);

    std::normal_distribution<double> normal( 0, 0.0005 );

    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples = 20;

    std::vector<double> forcing (numSamples) ;

    Eigen::MatrixXd allSamples (numSamples, 3);

    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();

    FEMClass trueTrussFem(false, TrussDef );

    for(int i = 0; i < numSamples ; i++){

        double A1 = 0.06 + normal( engine ) ;
        double A2 = 0.04 + normal( engine ) ;
        double A3 = 0.02 + normal( engine ) ;

        trueTrussFem.modA(0, A1);
        trueTrussFem.modA(1, A2);
        trueTrussFem.modA(2, A3);

        trueTrussFem.assembleS( );
        trueTrussFem.computeDisp( );
        trueTrussFem.computeForce( );



        allSamples(i, 0) = trueTrussFem.getDisp( 9 ) ;
        allSamples(i, 1) = trueTrussFem.getDisp(10 ) ;
        allSamples(i, 2) = trueTrussFem.getDisp(11 ) ;



        myTrueFile << A1 << " " << A2 << " " << A3 << " " << allSamples(i, 0)<< " " << allSamples(i, 1)<<" "<< allSamples(i, 2) << '\n';
        //std::cout << A1 <<" " << A2 << '\n';

        trueTrussFem.FEMClassReset(false);
        if( verbosity == true){if( (numSamples > 100 * 5 ) && ( i % (numSamples / ( 20 ) )  == 0 ) ){std::cout << "computed " << i << " samples " <<'\n';}}
    }

    myTrueFile.close();
    //std::cout << allSamples <<std::endl;


    return std::make_tuple(allSamples, forcing );
}


int main(){


    std::tuple<Eigen::MatrixXd, std::vector<double> > trueSamples = trueSampleGen();
    Eigen::MatrixXd trueSampleDisp = std::get<0>( trueSamples );


    using Vec  = Eigen::VectorXd;
    using Func = std::function <double(Vec)>;

    constexpr unsigned Dim = 3;


    Func pdf = [trueSampleDisp]( Vec x ){


        TupleTrussDef MTrussDef;
        MTrussDef = InitialTrussAssignment( );
        FEMClass MTrussFem( false, MTrussDef );

        MTrussFem.modA( 0, x[0] );
        MTrussFem.modA( 1, x[1] );
        MTrussFem.modA( 2, x[2] );



        MTrussFem.assembleS( );
        MTrussFem.computeDisp( );

        double Mdisp1 = MTrussFem.getDisp( 9 ) ;
        double Mdisp2 = MTrussFem.getDisp( 10 ) ;
        double Mdisp3 = MTrussFem.getDisp( 11 ) ;


        Eigen::Vector3d theta;
        theta << x[0], x[1], x[2];

        Eigen::Vector3d K_thetaInvf;
        K_thetaInvf << Mdisp1, Mdisp2, Mdisp3;

        Eigen::Matrix3d CovMatrixNoise;
        double noise =  0.0005 * 0.0005;
        CovMatrixNoise <<noise,   0,      0,
                          0,      noise,  0,
                          0,      0,      noise;

        double logLik = 0;

        //  p(y|Theta, Sigma)
        logLik += - (double) trueSampleDisp.rows() / 2.0 * std::log( CovMatrixNoise.determinant() ) ;



        //logLik += - 1. / 2.0 * std::log( CovMatrixNoise.determinant() ) ;

        //std::cout << logLik << " ";

        Eigen::Vector3d y_iVec;
        for(int i = 0; i < trueSampleDisp.rows(); ++i){

            y_iVec[0] = trueSampleDisp( i, 0 );
            y_iVec[1] = trueSampleDisp( i, 1 );
            y_iVec[2] = trueSampleDisp( i, 2 );

            logLik += - 1./2. * (y_iVec - K_thetaInvf).transpose() * CovMatrixNoise.inverse() * (y_iVec - K_thetaInvf)   ;

        }
        MTrussFem.FEMClassReset(false);

        if( std::isnan(logLik) ){ logLik = -9e30;}



//------------------------------------------------------------------------------------------//
//        double Theta1R = 0.1;
//        double Theta2R = 0.1;
//
//        Eigen::VectorXd priorSpan(3); priorSpan << Theta1R,  Theta1R, Theta1R;
//
//        Eigen::VectorXd map(3); map << 0.06,  0.04,  0.02;
//
//        double span = 0;
//        for(int i = 0; i <x.size(); ++i){
//            span += pow( x[i] - map[i], 2) / pow( priorSpan[i], 2);
//        }
//        if(span > 1){return -9e30;}
//        else(logLik += 1);



        //-Uniform Prior
//        for( int i = 0; i < Dim; ++i){
//            if( x[i] <= 0 ) {
//                return -9e30;
//            }
//            if( x[i] > 0.5 ){
//                return -9e30;
//           }
//            }logLik += pow( (1. / 0.5), 1./3.);


//----------------------------------------------

//        //Gaussian Prior - Conjugate Prior for Theta p( Theta_0 | Theta_0, sig^2 / k_0 )
//
        Eigen::Vector3d theta_0;

        theta_0 << 0.03, 0.03, 0.03 ;

        //double k_0  = 1e-4 ; // need k_0 to counter weight of prior
        double k_0  = 1e-4;

        logLik +=  - 1./2.* std::log( ( CovMatrixNoise / k_0).determinant() )
                 - 1./2.* (theta - theta_0).transpose() * (CovMatrixNoise / k_0 ).inverse() * (theta - theta_0) ;


 //-----------------------------------------------------//
        //std::cout << logLik << std::endl;
////
//        //std::cout << logLik << " ";
//
//        //Gaussian Prior - Conjugate Prior for sigma Noise
//        double priorPsiSig
//        Eigen::Matrix3d::Identity phi_0;
//
//        phi_0 = phi_0 * priorPsiSig;
//
//        double nu_0   = 4;
//
//        logLik +=    nu_0/2. * std::log( phi_0.determinant( ) )
//                  -( nu_0 + 2. + 1. ) / 2. * std::log( CovMatrixNoise.determinant( ) )
//                  - 1./2. * ( phi_0 * CovMatrixNoise.inverse() ).trace()  ;
//
//        //std::cout << logLik << " " << std::endl;

        return logLik;
    };


    double lower = 1e-5;
    Eigen::VectorXd x(3); x << lower,  lower,  0.02;

    double LikVals;

    Eigen::VectorXd maxVal(3); maxVal << -9e30, -9e30, -9e30;
    Eigen::VectorXd maxX(3);   maxX   << lower, lower, 0.02;
    double Area = 0;

    double a = 0.0000001;
    double b = 0.2;

    int samplesX = 1e2;


    double dx = (double) (b - a) / samplesX;

    Eigen::MatrixXd evals(samplesX, 4);


    for(int iter = 0; iter < 15; ++iter){

        for(int dim = 0 ; dim < 2; ++dim){

            unsigned ctr = 0;

            for(int i = 0; i < samplesX; ++i){

                x[dim] = a + (double) dx * (i + 1) ;

                //LikVals =  std::exp( pdf(x) ) ;
                LikVals =   pdf(x)  ;
                //std::cout << LikVals << std::endl;

                if(LikVals > maxVal[dim]){
                    //std::cout << "yes" <<std::endl;
                    maxVal[dim] = LikVals;
                    maxX[dim]   = x[dim];

                }

                Area += LikVals * dx ;
                //Area = 1;

                evals(ctr, dim*2) = x[dim];
                evals(ctr, dim*2+1) = LikVals;

                ctr++;
            }


            for(int i = 0; i < evals.rows(); ++i){
                evals(i, dim*2+1) = evals(i, dim*2+1) - maxVal[dim];
                evals(i, dim*2+1) = std::exp( evals(i, dim*2+1) );
            }

            double AreaExp = 0;
            for(int i = 0; i < evals.rows(); ++i){
                AreaExp += evals(i, dim*2+1) * dx;
            }
            for(int i = 0; i < evals.rows(); ++i){
                evals(i, dim*2+1) = evals(i, dim*2+1) / AreaExp;
            }



            x = maxX;
            Area = 0;
            maxVal[0] = 0.;
            maxVal[1] = 0.;
        }
    }


    std::cout << "maxVal = " << maxVal << "\n" << " maxX = " << maxX << std::endl;


    std::ofstream myFile;
    myFile.open("pdfResults.dat");

    for(int i = 0; i < samplesX; ++i){
        for(int j = 0; j < 2 * 2; ++j){
    myFile << evals(i, j) << " ";
    }
        myFile << std::endl;
    }

    myFile.close();
    return 0;

}





