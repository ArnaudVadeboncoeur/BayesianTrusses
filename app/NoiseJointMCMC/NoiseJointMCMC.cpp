/*
 * BTrussMCMC_MH.cpp
 *
 *  Created on: 12 Dec 2019
 *      Author: arnaudv
 */

#include "../../src/FEMClass.hpp"

#include "../../src/statTools/SampleAnalysis.hpp"
#include "../../src/statTools/histSort.hpp"
#include "../../src/statTools/KLDiv.hpp"
#include "../../src/statTools/SteinDiscrepancy.hpp"

#include "../../src/statTools/MCMC2.hpp"

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
//    double mu1  = 1;
//    double sig1 = 0.25;
//
//    double mu2  = 1;
//    double sig2 = 0.25;

    std::ofstream myTrueFile;
    myTrueFile.open("trueResults.dat", std::ios::trunc);

    //std::lognormal_distribution<double> lognormal1( mu1, sig1  );
    std::lognormal_distribution<double> lognormal2( -3.688, 1  );
    //std::normal_distribution<double> normal( 0, 0.001 );

    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples = 1e3;

    std::vector<double> forcing (numSamples) ;

    Eigen::MatrixXd allSamples (numSamples, 1);

    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();

    FEMClass trueTrussFem(false, TrussDef );

    for(int i = 0; i < numSamples ; i++){

        //double A1 = 0.025 + normal( engine ) ;
        double A1 = lognormal2( engine );
        trueTrussFem.modA(0, A1);

        trueTrussFem.assembleS( );
        trueTrussFem.computeDisp( );
        trueTrussFem.computeForce( );
        allSamples(i, 0) = trueTrussFem.getDisp(10) ;

        myTrueFile << A1 << '\n';

        trueTrussFem.FEMClassReset(false);
        if( verbosity == true){if( (numSamples > 100 * 5 ) && ( i % (numSamples / ( 20 ) )  == 0 ) ){std::cout << "computed " << i << " samples " <<'\n';}}
    }
    //std::vector<double>  delatXs = findDeltaX(allSamples, 100);
    //HistContainer histPoints = histBin(allSamples, delatXs, true, true);

    myTrueFile.close();

    return std::make_tuple(allSamples, forcing );
}


int main(){

    std::tuple<Eigen::MatrixXd, std::vector<double> > trueSamples = trueSampleGen();
    Eigen::MatrixXd trueSampleDisp = std::get<0>( trueSamples );


    //compute impirical mean and sigma

    double empMean = 0;
    for(unsigned i = 0; i < trueSampleDisp.rows(); ++i){
        empMean += trueSampleDisp(i, 0);
    }
    empMean = (double) empMean / trueSampleDisp.rows();
    std::cout << "empMean = " << empMean << std::endl;

    double empSigma = 0;
    for(unsigned i =0; i <trueSampleDisp.rows(); ++i){
        empSigma += std::pow( trueSampleDisp(i,0) - empMean, 2 );
    }
    empSigma = std::sqrt( (double) empSigma / ( trueSampleDisp.rows() - 1 ) );
    std::cout << "empSigma = " << empSigma << std::endl;

    using Vec = std::vector<double>;
    using Func = std::function <double(Vec)>;

    constexpr unsigned Dim = 2;

    //make log likelihood function sampling work
//    Func pdf = [] ( Vec x) {
//
//        double lik = std::log( (- x[0] * x[0] + 3 ) );
//
//        if( std::isnan(lik) ){
//            lik = -9e30;
//        }
//
//        //std::cout << lik << std::endl;
//        return lik; };


    Func pdf = [trueSampleDisp, empSigma]( Vec x ){

        if( x[0] <= 0 ){ return -9e30;}

        TupleTrussDef MTrussDef;
        MTrussDef = InitialTrussAssignment( );
        FEMClass MTrussFem( false, MTrussDef );
        MTrussFem.modA( 0, x[0] );
        MTrussFem.assembleS( );
        MTrussFem.computeDisp( );
        double Mdisp = MTrussFem.getDisp( 10 ) ;

        double logLik = 0;

        double trueAvgDisp = 0;

        for(int i = 0; i < trueSampleDisp.rows(); ++i){

            logLik += - 1.0 / 2.0 * pow( ( (trueSampleDisp(i,0) - Mdisp) / x[1] ), 2) ;
            trueAvgDisp +=  trueSampleDisp(i,0);
        }
        trueAvgDisp = (double) trueAvgDisp / trueSampleDisp.rows();
        MTrussFem.FEMClassReset(false);


        if( std::isnan(logLik) ){ logLik = -9e30; }


        //Gaussian Prior - Conjugate Prior for Theta
        logLik = logLik + -1./2. * std::log(2.0*M_PI*0.054) - 1.0 / 2.0 * pow( ( ( x[0] - 0.04125 ) / 0.054 ) , 2 );

        //Gaussian Prior - Conjugate Prior for sigma Noise
        logLik = logLik + -1./2. * std::log(2.0*M_PI*0.005) - 1.0 / 2.0 * pow( ( ( x[1] - 0.0054 ) / 0.005 ) , 2 );//pic 1 through 8
        //logLik = logLik + -1./2. * std::log(2.0*M_PI*0.005) - 1.0 / 2.0 * pow( ( ( x[1] - 0.015 ) / 0.05 ) , 2 );//pic 9 and 10
        //logLik = logLik + -1./2. * std::log(2.0*M_PI*0.005) - 1.0 / 2.0 * pow( ( ( x[1] - 0.0054 ) / 0.001 ) , 2 );//pic11
        //logLik = logLik + -1./2. * std::log(2.0*M_PI*0.005) - 1.0 / 2.0 * pow( ( ( x[1] - 0.0054 ) / 0.0005 ) , 2 );//pic12


        return logLik;
    };

    Vec xStart = { 0.015, 0.006 };

    //Vec sigmaInitial = { empSigma/10., empSigma / 1000. };
    //Vec sigmaInitial = { 0.01/10., 0.00052 / 100. };
   // Vec sigmaInitial = { 0.01/5. , 0.00052 / 50. };
    //Vec sigmaInitial = { 0.01/5. , 0.00052 / 1. };//pic v3
    //Vec sigmaInitial = { 0.01 , 0.00052 / 1. };//pic v4
    //Vec sigmaInitial = { 0.01 , 0.00052 * 2. };//pic v5
    //Vec sigmaInitial = { 0.01 , 0.00052 * 5. };//pic v6
    //Vec sigmaInitial = { 0.01 * 5 , 0.00052 * 5. };//pic 7
    Vec sigmaInitial = { 0.01 * 7 , 0.00052 * 10. };//pic8 and pic 9
    //Vec sigmaInitial = { 0.01 * 7 , 0.00052 * 5. };//pic 10
    //Vec sigmaInitial = { 0.01 * 1. , 0.00052 * 2. };//pic 11
    //Vec sigmaInitial = { 0.01 / 1. , 0.00052 / 1. };//pic 12


    Vec lower = {-1000, -1000};
    Vec upper = {+1000, +1000};

    const std::pair< Vec, Vec > bounds = std::make_pair(lower, upper);

    MCMC<Dim,Func, Vec> mcmc(pdf, xStart,sigmaInitial, bounds, false, true);

    std::vector<double> sigmaJump = { 0.1, 0.0001 };

    //mcmc.setSigma(sigmaJump, 100, 1e4, true);

    mcmc.sample( 1e5 );

    //mcmc.thinSamples( 200, 5 );

    std::filebuf myFile;
    myFile.open("resultsModel.dat",std::ios::out);
    std::ostream osOut (&myFile);
    mcmc.writeResult(osOut);
    myFile.close();

    Vec maxArg = mcmc.getMaxArg();
    std::cout<<".getMaxArg() = : ";

    for(int i = 0; i < Dim; ++i ){

        std::cout<<maxArg[i] << " ";

    }std::cout<<std::endl;
    std::cout<< "Here" << std::endl;
    std::cout<<".getMaxVal() = : "<< mcmc.getMaxVal() <<std::endl;


    return 0;

}


