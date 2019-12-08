/*
 * BTrussStochDOFLearn.cpp
 *
 *  Created on: 4 Dec 2019
 *      Author: arnaudv
 */

#include "../../src/FEMClass.hpp"

#include "../../src/statTools/SampleAnalysis.hpp"
#include "../../src/statTools/histSort.hpp"
#include "../../src/statTools/KLDiv.hpp"
#include "../../src/statTools/SteinDiscrepancy.hpp"

#include "ThreeDTrussDef.hpp"


#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>



Eigen::MatrixXd trueSampleGen(){
    bool verbosity = false;
    double mu  = 10;
    double sig = 0.25;


    std::lognormal_distribution<double> lognormal( mu, sig  );
    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples =  1e2;//1e3;

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

    std::vector<double>  delatXs = findDeltaX(allSamples, 100);
    HistContainer histPoints = histBin(allSamples, delatXs, true, false);

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


double lossFunctionMSEAvg(Eigen::MatrixXd trueSamples , Eigen::MatrixXd ModelSamples ){

    double TrueSMean = MonteCarloAvgs(trueSamples);

    double MeanModelCurr = MonteCarloAvgs(ModelSamples);

    return pow(TrueSMean - MeanModelCurr, 2);
}


double lossFunctionKLDiv(Eigen::MatrixXd trueSamples , Eigen::MatrixXd ModelSamples ){

    std::vector<double>  delatXs = findDeltaX(trueSamples, 100);

    HistContainer trueHistPoints  = histBin(trueSamples,  delatXs, true, false);
    HistContainer ModelHistPoints = histBin(ModelSamples, delatXs, true, false);

    double div = KLDiv(trueHistPoints, ModelHistPoints );

    return div;
}



double proposalKernel(double mu, std::random_device& rd ){

    std::normal_distribution<double> normal( mu, 0.2  );
    std::mt19937 engine( rd() );
    double muProp = -1.;
    while(muProp < 0){
        muProp = normal(engine);

    }

    return muProp ;

}


int main(){

    Eigen::MatrixXd trueSamples = trueSampleGen();



    std::random_device rd;

    unsigned loops = 200;
    double muCurr  = 0;
    double sig = 0.25;

    double muProp;


    Eigen::MatrixXd ModelSamplesCurr;
    Eigen::MatrixXd ModelSamplesProp;

    ModelSamplesCurr = ModelSampleGen (trueSamples.rows(), muCurr, sig);

    //Distance measure taken as squared difference in mean
    //Should be proper Statistical distance of Distributions like KL-div etc..

    //double distanceCurr = lossFunctionMSEAvg(trueSamples, ModelSamplesCurr ) ;
    //double distanceCurr = lossFunctionKLDiv(trueSamples, ModelSamplesCurr ) ;
     double distanceCurr = steinDisc(trueSamples, ModelSamplesCurr ) ;


    double distanceProp;

    int falseAsses = 0;

    for(int i = 0; i < loops; ++i){

        muProp = proposalKernel( muCurr, rd );
        ModelSamplesProp = ModelSampleGen ( trueSamples.rows(), muProp, sig );

        //distanceProp = lossFunctionMSEAvg(trueSamples, ModelSamplesProp );
        //distanceProp = lossFunctionKLDiv(trueSamples, ModelSamplesProp );
        distanceProp = steinDisc(trueSamples, ModelSamplesProp );

        if( distanceProp > distanceCurr && abs(muProp - 10) < abs(muCurr-10) ) {
            std::cout << "x" << std::endl;
            falseAsses++;
        }

        if( distanceProp < distanceCurr){

            distanceCurr = distanceProp;

            if( muCurr < 10 && muProp < muCurr) {

                std::cout << "x" << std::endl;
                falseAsses++;

            }

            muCurr = muProp;
            ModelSamplesCurr = ModelSamplesProp;
            std::cout << "Iter: "<< i <<" muCurr = " << muCurr << " Distance = " << distanceCurr <<std::endl;
        }
        else {
            std::cout<<"Iter: "<< i << "---" <<" muProp = " << muProp << std::endl;}
    }

    std::cout << "MuChosen = " << muCurr << " Distance = " << distanceCurr <<std::endl;

    std::cout << "falseAsses = " << falseAsses << " " << (double) falseAsses / loops * 100 << "%" << std::endl;

    std::vector<double>  delatXs = findDeltaX(ModelSamplesCurr, 100);
    int nBins = 100;
    HistContainer histPoints = histBin(ModelSamplesCurr, delatXs, true, false);


    return 0;
}

