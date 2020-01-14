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

    std::ofstream myTrueFile;
    myTrueFile.open("trueResults.dat", std::ios::trunc);

    std::normal_distribution<double> normal(0., 0.005);

    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples = 1e3;

    std::vector<double> forcing (numSamples) ;

    Eigen::MatrixXd allSamples (numSamples, 1);

    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();

    FEMClass trueTrussFem(false, TrussDef );

    for(int i = 0; i < numSamples ; i++){

        double A1 = 0.05 + normal( engine ) ;
        double A2 = 0.025  + normal( engine );

        trueTrussFem.modA(0, A1);
        trueTrussFem.modA(1, A2);

        trueTrussFem.assembleS( );
        trueTrussFem.computeDisp( );
        trueTrussFem.computeForce( );
        allSamples(i, 0) = trueTrussFem.getDisp(10) ;

        myTrueFile << trueTrussFem.getA(0) << " " << trueTrussFem.getA(1) << '\n';

        trueTrussFem.FEMClassReset(false);
        if( verbosity == true){if( (numSamples > 100 * 5 ) && ( i % (numSamples / ( 20 ) )  == 0 ) ){std::cout << "computed " << i << " samples " <<'\n';}}
    }

    myTrueFile.close();

    return std::make_tuple(allSamples, forcing );
}


int main(){

    std::tuple<Eigen::MatrixXd, std::vector<double> > trueSamples = trueSampleGen();
    Eigen::MatrixXd trueSampleDisp = std::get<0>( trueSamples );

    std::ofstream myTrueDispFile;
    myTrueDispFile.open("trueDispResults.dat", std::ios::trunc);
    for(int i = 0; i < trueSampleDisp.rows(); ++i){
    myTrueDispFile << trueSampleDisp(i,0) << std::endl;
    }
    myTrueDispFile.close();

    //compute empirical mean and sigma

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

    constexpr unsigned Dim = 3;



    Func pdf = [trueSampleDisp, empSigma]( Vec x ){

        for( int i = 0; i < x.size(); ++i){
        if( x[i] <= 0 ){ return -9e30;}
        }

        TupleTrussDef MTrussDef;
        MTrussDef = InitialTrussAssignment( );
        FEMClass MTrussFem( false, MTrussDef );

        MTrussFem.modA( 0, x[0] );
        MTrussFem.modA( 1, x[1] );

        MTrussFem.assembleS( );
        MTrussFem.computeDisp( );
        double Mdisp = MTrussFem.getDisp( 10 ) ;

        double logLik = 0;

        logLik += - (double) trueSampleDisp.rows() / 2.0 *  log ( 2.0 * M_PI) - (double) trueSampleDisp.rows()*log( x[2] ) ;

        double trueAvgDisp = 0;

        for(int i = 0; i < trueSampleDisp.rows(); ++i){

            logLik += - 1.0 / 2.0 * pow( ( (trueSampleDisp(i,0) - Mdisp ) / x[2] ), 2) ;
           }
        MTrussFem.FEMClassReset(false);


        if( std::isnan(logLik) ){ logLik = -9e30; }


        //Gaussian Prior - Conjugate Prior for Theta 1&2
//        double PA1Mu = 0.025;
//        double PA2Mu = 0.05;

        double PA1Mu = 0.1;
        double PA2Mu = 0.1;

        double PA1Std = 1;
        double PA2Std = 1;

        logLik += -1./2. * std::log(2.0*M_PI) - std::log(PA1Std)  -   1.0 / 2.0 * pow( ( ( x[0] - PA1Mu ) / PA1Std ) , 2 );
        logLik += -1./2. * std::log(2.0*M_PI) - std::log(PA2Std)  -   1.0 / 2.0 * pow( ( ( x[1] - PA2Mu ) / PA2Std ) , 2 );

        //Gaussian Prior - Conjugate Prior for sigma Noise - Displacement or Area?
        double PSigMu  = 0.0005;
        double PSigStd = 0.001 ;
        //logLik += -1./2. * std::log(2.0*M_PI) - std::log(PSigStd) - 1.0 / 2.0 * pow( ( ( x[2] - PSigMu ) / PSigStd ) , 2 );
        logLik += std::log( pow( x[2], -2) );

        return logLik;
    };

    Vec xStart = { 0.1, 0.1, 0.001 };

    //Vec sigmaInitial = { 0.0001 , 0.0001, 1e-6 };
    //Vec sigmaInitial = { 0.004 , 0.004, 7e-5 };
    Vec sigmaInitial = { 0.001 , 0.001, 1e-5 };



    Vec lower = {-1e9, -1e9, -1e9};
    Vec upper = {+1e9, +1e9, +1e9};

    const std::pair< Vec, Vec > bounds = std::make_pair(lower, upper);

    MCMC<Dim,Func, Vec> mcmc(pdf, xStart,sigmaInitial, bounds, false, true);

    std::vector<double> sigmaJump = { 0.1, 0.1, 0.001 };

    //mcmc.setSigma(sigmaJump, 100, 1e4, true);

    mcmc.sample( 1 * 1e5 );

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


