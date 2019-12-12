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
#include "../../src/statTools/MCMC.hpp"

#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>

#include "ThreeDTruss.hpp"

std::tuple<Eigen::MatrixXd, std::vector<double> > trueSampleGen(){
    bool verbosity = false;
    double mu1  = 1;
    double sig1 = 0.25;

    double mu2  = 1;
    double sig2 = 0.25;


    std::lognormal_distribution<double> lognormal1( mu1, sig1  );
    std::lognormal_distribution<double> lognormal2( mu2, sig2  );

    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples =  1e4;//1e3;
    std::vector<double> forcing (numSamples) ;

    Eigen::MatrixXd allSamples (numSamples, 1);


    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();

    FEMClass trueTrussFem(false, TrussDef );

    for(int i = 0; i < numSamples ; i++){

        double A1 = lognormal1( engine ) / 100.;

        double disp;

        trueTrussFem.modA(0, A1);

        trueTrussFem.assembleS( );
        trueTrussFem.computeDisp( );
        trueTrussFem.computeForce( );
        allSamples(i, 0) = trueTrussFem.getDisp(10);
        forcing[i] = trueTrussFem.getForce(10);

        trueTrussFem.FEMClassReset(false);
        if( verbosity == true){if( (numSamples > 100 * 5 ) && ( i % (numSamples / ( 20 ) )  == 0 ) ){std::cout << "computed " << i << " samples " <<'\n';}}
    }

    std::vector<double>  delatXs = findDeltaX(allSamples, 100);
    HistContainer histPoints = histBin(allSamples, delatXs, true, false);

    return std::make_tuple(allSamples, forcing );

}




int main(){

    std::tuple<Eigen::MatrixXd, std::vector<double> > trueSamples = trueSampleGen();
    Eigen::MatrixXd trueSampleDisp = std::get<0>( trueSamples );

    using Vec = std::vector<double>;
    using Func = std::function <double(Vec)>;

    constexpr unsigned Dim = 1;

    Func pdf = [trueSampleDisp]( Vec x ){
        if(x[0] < 0 ){ return 0.;}
        double disp;
        TupleTrussDef MTrussDef;
        MTrussDef = InitialTrussAssignment();
        FEMClass MTrussFem(false, MTrussDef );
        MTrussFem.modA(0, x[0]);
        MTrussFem.assembleS( );
        MTrussFem.computeDisp( );
        double Mdisp = MTrussFem.getDisp(10);

        double logLik = 0;
        logLik += - trueSampleDisp.rows() / 2. * log ( 2*M_PI);
        for(int i = 0; i < trueSampleDisp.rows(); ++i){

            logLik += - 1/2. * ( trueSampleDisp(i,0) - Mdisp ) * ( trueSampleDisp(i,0) - Mdisp );
        }

//        std::cout << "Mdisp  = " << Mdisp  <<std::endl;
//        std::cout << "x[0] aka Area = " << x[0]  <<std::endl;
//        std::cout << "logLik = " << logLik <<std::endl;
        MTrussFem.FEMClassReset(false);
        return - logLik;
    };

    Vec xStart = { 0.001 };

    Vec sigmaInitial = { 1 };

    Vec lower = {0};
    Vec upper = {10};

    const std::pair< Vec, Vec > bounds = std::make_pair(lower, upper);

    MCMC<Dim,Func, Vec> mcmc(pdf, xStart,sigmaInitial, bounds, 100, true, true);

    std::vector<double> sigmaJump = { 0.1 };
    std::cout<<"goign for first mcmc method"<<std::endl;

    //mcmc.setSigma(sigmaJump, 1000, 10000, true);



    mcmc.sample(1e5);



    std::filebuf myFile;
    myFile.open("results.dat",std::ios::out);
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


