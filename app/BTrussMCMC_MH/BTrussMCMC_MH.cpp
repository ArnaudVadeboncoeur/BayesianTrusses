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
#include "../../src/statTools/MCMC.hpp"

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <math.h>

#include "../../src/statTools/SteinDiscrepancy.hpp"
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
    //std::lognormal_distribution<double> lognormal2( mu2, sig2  );
    std::normal_distribution<double> normal( 0, 0.001 );

    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples = 1e5;

    std::vector<double> forcing (numSamples) ;

    Eigen::MatrixXd allSamples (numSamples, 1);

    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();

    FEMClass trueTrussFem(false, TrussDef );

    for(int i = 0; i < numSamples ; i++){

        double A1 = 0.025 + normal( engine ) ;

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

    using Vec = std::vector<double>;
    using Func = std::function <double(Vec)>;

    constexpr unsigned Dim = 1;

    //make log likelihood function sampling work
//    Func pdf = [] ( Vec x) {
//
//        double lik = std::log( (x[0] * x[0] + 3 ) );
//
//        if( std::isnan(lik) ){
//            lik = -9e30;
//        }
//
//        //std::cout << lik << std::endl;
//        return lik; };


    Func pdf = [trueSampleDisp]( Vec x ){

        //std::ofstream ALikFile;
        //ALikFile.open("AandLik.dat", std::ios::app);

        if( x[0] <= 0 ){ return -9e30;}

        TupleTrussDef MTrussDef;
        MTrussDef = InitialTrussAssignment( );
        FEMClass MTrussFem( false, MTrussDef );
        MTrussFem.modA( 0, x[0] );
        MTrussFem.assembleS( );
        MTrussFem.computeDisp( );
        double Mdisp = MTrussFem.getDisp( 10 ) ;

        double logLik = 0;

       // logLik += - (double) trueSampleDisp.rows() / 2.0 * ( log ( 2.0 * M_PI) + log( 1.0 ) );

        double trueAvgDisp = 0;

        for(int i = 0; i < trueSampleDisp.rows(); ++i){

            logLik += - 1.0 / 2.0 * pow( ( (trueSampleDisp(i,0) - Mdisp) / 0.001 ), 2) ;/// 0.01 ;
            //--logLik = logLik * 1.0 / ( std::sqrt( 2.0 * M_PI ) * 1.0) * exp(- 1.0 / 2.0 * pow( ( trueSampleDisp(i,0) - Mdisp ), 2) / 1.0 );
            trueAvgDisp +=  trueSampleDisp(i,0);
        }
        trueAvgDisp = (double) trueAvgDisp / trueSampleDisp.rows();
        MTrussFem.FEMClassReset(false);


        if( std::isnan(logLik) ){ logLik = -9e30; }

//        else if(x[0] >= 0.5 ){ return -9e30;}
//        else {
//            logLik = logLik + std::log( 1. / 0.5 );
//        }

        return logLik;
    };

    Vec xStart = { 0.00001 };

    Vec sigmaInitial = { 0.00001 };

    Vec lower = {-1000};
    Vec upper = {+1000};

    const std::pair< Vec, Vec > bounds = std::make_pair(lower, upper);

    MCMC<Dim,Func, Vec> mcmc(pdf, xStart,sigmaInitial, bounds, 1e3, true, true);

    std::vector<double> sigmaJump = { 0.001 };
    std::cout<<"goign for first mcmc method"<<std::endl;

    mcmc.setSigma(sigmaJump, 100, 1e4, true);

    mcmc.sample(1e5);


    std::filebuf myFile;
    myFile.open("resultsModel.dat",std::ios::out);
    std::ostream osOut (&myFile);
    mcmc.writeResult(osOut);
    myFile.close();

    Vec maxArg = mcmc.getMaxArg();
    std::cout<< "Here" << std::endl;
    std::cout<< maxArg.size() << std::endl;
    std::cout<<".getMaxArg() = : ";

    for(int i = 0; i < Dim; ++i ){

        std::cout<<maxArg[i] << " ";

    }std::cout<<std::endl;
    std::cout<< "Here" << std::endl;
    std::cout<<".getMaxVal() = : "<< mcmc.getMaxVal() <<std::endl;


    return 0;

}


