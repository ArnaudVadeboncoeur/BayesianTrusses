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
#include "../../src/statTools/MCMC2.hpp"

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <math.h>

#include "../../../src/statTools/SteinDiscrepancy.hpp"
#include "ThreeDTruss.hpp"

std::tuple<Eigen::MatrixXd, std::vector<double> > trueSampleGen(){

    bool verbosity = false;

    std::ofstream myTrueFile;
    myTrueFile.open("trueResults.dat", std::ios::trunc);

    std::normal_distribution<double> normal( 0, 0.001 );

    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples = 1e3;

    std::vector<double> forcing (numSamples) ;

    Eigen::MatrixXd allSamples (numSamples, 1);

    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();

    FEMClass trueTrussFem(false, TrussDef );

    for(int i = 0; i < numSamples ; i++){

        double A1 = 0.06 + normal( engine ) ;

        trueTrussFem.modA(0, A1);

        trueTrussFem.assembleS( );
        trueTrussFem.computeDisp( );
        trueTrussFem.computeForce( );
        allSamples(i, 0) = trueTrussFem.getDisp(10) ;

        //std::cout << allSamples(i, 0) << std::endl;

        myTrueFile << A1 << '\n';

        trueTrussFem.FEMClassReset(false);
        if( verbosity == true){if( (numSamples > 100 * 5 ) && ( i % (numSamples / ( 20 ) )  == 0 ) ){std::cout << "computed " << i << " samples " <<'\n';}}
    }

    myTrueFile.close();

    return std::make_tuple(allSamples, forcing );
}


int main(){

    std::tuple<Eigen::MatrixXd, std::vector<double> > trueSamples = trueSampleGen();
    Eigen::MatrixXd trueSampleDisp = std::get<0>( trueSamples );

    using Vec = std::vector<double>;
    using Func = std::function <double(Vec)>;

    constexpr unsigned Dim = 2;

    Func pdf = [trueSampleDisp]( Vec x ){
        //x[0] = A1;
        //x[1] = Sig;

        for( int i = 0; i < x.size(); ++i){
            if( x[i] <= 0 ){ return -9e30;}
            }

        TupleTrussDef MTrussDef;
        MTrussDef = InitialTrussAssignment( );
        FEMClass MTrussFem( false, MTrussDef );
        MTrussFem.modA( 0, x[0] );
        MTrussFem.assembleS( );
        MTrussFem.computeDisp( );
        double Mdisp = MTrussFem.getDisp( 10 ) ;

        double logLik = 0;
        double beta = 1e2;

        logLik += - (double) trueSampleDisp.rows() * log( x[1] ) ;

        for(int i = 0; i < trueSampleDisp.rows(); ++i){

            //logLik += - 1.0 / 2.0 * pow( ( (trueSampleDisp(i,0) - Mdisp) * beta / x[1] ), 2) ;
            logLik += - 1.0 / 2.0 * pow( ( (trueSampleDisp(i,0) - Mdisp) / x[1] ), 2) ;

        }
        MTrussFem.FEMClassReset(false);

        if( std::isnan(logLik) ){ logLik = -9e30; }


        //Gaussian Prior - Conjugate Prior for Theta p( mu | mu_0, sig^2 / k_0 )
        double mu_0 = 0.1 ;
        double k_0  = 1e-4   ; // need k_0 to counter weight of prior

        //logLik += -std::log( x[1] ) - 1. / 2. * k_0 * pow( (x[0] - mu_0) / x[1]  , 2);

        //! Test this One - kappa_o added to first term
        //works
        logLik += -std::log( x[1] ) - 1. / 2. * k_0 * pow( (x[0] - mu_0) / x[1]  , 2);

        //uniform prior trial
//        if( x[0] > 0.5 || x[0] <= 0){
//            logLik += -9e30;
//        }else{ logLik += std::log( 1. / 0.5 ); }


        //Gaussian Prior - Conjugate Prior for sigma Noise
        double sig_0  = 0.001;
        double nu_0   = 1;

        logLik += (- nu_0 - 2) * std::log( x[1] ) - (nu_0 * sig_0 * sig_0 / (2. * x[1]* x[1] ) );


        return logLik;
    };

    Vec xStart = { 0.01 , 0.00001 };

    Vec sigmaInitial = { 0.0005 , 0.000001 };


    Vec lower = {-1e9, -1e9};
    Vec upper = {+1e9, +1e9};

    const std::pair< Vec, Vec > bounds = std::make_pair(lower, upper);

    MCMC<Dim,Func, Vec> mcmc(pdf, xStart,sigmaInitial, bounds, false, true);

    std::vector<double> sigmaJump = { 0.1, 0.0001 };

    //mcmc.setSigma(sigmaJump, 100, 1e4, true);

    mcmc.sample( 1e5 );

    mcmc.thinSamples( 500 , 1 );

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


