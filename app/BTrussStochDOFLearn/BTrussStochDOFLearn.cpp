/*
 * BTrussStochDOFLearn.cpp
 *
 *  Created on: 4 Dec 2019
 *      Author: arnaudv
 */

#include "../../src/FEMClass.hpp"
#include "../../src/SampleAnalysis.hpp"
#include "ThreeDTrussDef.hpp"

#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>


Eigen::MatrixXd trueSampleGen(){
    bool verbosity = false;
    double mu  = 1;
    double sig = 0.25;


    std::lognormal_distribution<double> lognormal( mu, sig  );
    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples =  1e4;//1e3;

    Eigen::MatrixXd allSamples (numSamples, 1);


    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();

    FEMClass trueTrussFem(false, TrussDef );

    for(int i = 0; i < numSamples ; i++){

        double A = lognormal( engine );

        double disp;

        trueTrussFem.modA(0, A);
        trueTrussFem.assembleS( );
        trueTrussFem.computeDisp( );
        trueTrussFem.computeForce( );
        allSamples(i, 0) = trueTrussFem.getDisp(10);

        trueTrussFem.FEMClassReset(false);
        if( verbosity == true){if( (numSamples > 100 * 5 ) && ( i % (numSamples / ( 20 ) )  == 0 ) ){std::cout << "computed " << i << " samples " <<'\n';}}
    }

    int nBins = 100;
    histBin(allSamples, nBins, true, true);
    return allSamples;
}


Eigen::MatrixXd ModelSampleGen (int numSamp, double mu, double sig){

    TupleTrussDef TrussDef2;

    TrussDef2 =  InitialTrussAssignment();

    FEMClass ModelTrussFem(false, TrussDef2 );

    std::lognormal_distribution<double> lognormal( mu, sig  );
    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples =  numSamp;

    Eigen::MatrixXd allModelSamples (numSamples,1);

    for(int i = 0; i < numSamples ; i++){

           double A = lognormal( engine );

           ModelTrussFem.modA(0, A);
           ModelTrussFem.assembleS( );
           ModelTrussFem.computeDisp( );
           ModelTrussFem.computeForce( );
           allModelSamples(i, 0) = ModelTrussFem.getDisp(10);
           ModelTrussFem.FEMClassReset(false);
    }


    return allModelSamples;
}


double lossFunction(Eigen::MatrixXd trueSamples , Eigen::MatrixXd ModelSamplesCurr ){

    double TrueSMean = MonteCarloAvgs(trueSamples);

    double MeanModelCurr = MonteCarloAvgs(ModelSamplesCurr);

    return pow(TrueSMean - MeanModelCurr, 2);
}

double proposalKernel(double mu, std::random_device& rd ){

    std::normal_distribution<double> normal( mu, 0.2  );
    std::mt19937 engine( rd() );

    return normal(engine);

}


int main(){

    Eigen::MatrixXd trueSamples = trueSampleGen();



    std::random_device rd;

    unsigned loops = 15;
    double muCurr  = 0;
    double sig = 0.25;

    double muProp;


    Eigen::MatrixXd ModelSamplesCurr;
    Eigen::MatrixXd ModelSamplesProp;

    ModelSamplesCurr = ModelSampleGen (trueSamples.rows(), muCurr, sig);

    //Distance measure taken as squared difference in mean
    //Should be proper Statistical distance of Distributions like KL-div etc..

    double distanceCurr = lossFunction(trueSamples, ModelSamplesCurr ) ;

    double distanceProp;



    for(int i = 0; i < loops; ++i){

        muProp = proposalKernel( muCurr, rd );
        ModelSamplesProp = ModelSampleGen ( trueSamples.rows(), muProp, sig );
        distanceProp = lossFunction(trueSamples, ModelSamplesProp );

        if( distanceProp < distanceCurr){

            distanceCurr = distanceProp;
            muCurr = muProp;
            ModelSamplesCurr = ModelSamplesProp;
            std::cout << "Iter: "<< i <<" muCurr = " << muCurr << std::endl;
        }
        else {std::cout<<"Iter: "<< i << "---" << std::endl;}
    }

    int nBins = 100;
    histBin(ModelSamplesCurr, nBins, true, true);


    return 0;
}














