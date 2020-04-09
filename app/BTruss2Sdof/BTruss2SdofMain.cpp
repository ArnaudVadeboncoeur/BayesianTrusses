#include "../../src/FEMClass.hpp"

#include "../../src/statTools/SampleAnalysis.hpp"
#include "../../src/statTools/histSort.hpp"
#include "../../src/statTools/KLDiv.hpp"
#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>

#include "../../src/statTools/SteinDiscrepancy.hpp"
#include "ThreeDTruss3Bar2sdof.hpp"

std::tuple<Eigen::MatrixXd, std::vector<double> > trueSampleGen(){
    bool verbosity = false;
    double mu1  = 1;
    double sig1 = 0.25;

    double mu2  = 1;
    double sig2 = 0.25;


    std::lognormal_distribution<double> lognormal1( mu1, sig1  );
    std::lognormal_distribution<double> lognormal2( mu2, sig2  );
    std::uniform_real_distribution<double> uniform( 0, 1  );

    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples =  1e4;//1e3;
    std::vector<double> forcing (numSamples) ;

    Eigen::MatrixXd allSamples (numSamples, 1);


    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();

    FEMClass trueTrussFem(false, TrussDef );

    for(int i = 0; i < numSamples ; i++){

        double randU = uniform(engine);
        double F;
        if( randU > 0.66 ) { F = -100; }
        else{ F = -200; }


        double A1 = lognormal1( engine );
        //double A2 = A1;
        //double A2 = lognormal2( engine );

        double disp;

        trueTrussFem.modA(0, A1);
        //trueTrussFem.modA(1, A2);

        //trueTrussFem.modForce(3, 1, F);
       // trueTrussFem.modForce(3, 0, F);

        trueTrussFem.assembleS( );
        trueTrussFem.computeDisp( );
        trueTrussFem.computeForce( );
        allSamples(i, 0) = trueTrussFem.getDisp(10);
        forcing[i] = F;

        trueTrussFem.FEMClassReset(false);
        if( verbosity == true){if( (numSamples > 100 * 5 ) && ( i % (numSamples / ( 20 ) )  == 0 ) ){std::cout << "computed " << i << " samples " <<'\n';}}
    }

    std::vector<double>  delatXs = findDeltaX(allSamples, 100);
    HistContainer histPoints = histBin(allSamples, delatXs, true, true);

    return std::make_tuple(allSamples, forcing );

}


std::vector<std::tuple<double, double> >
proposalKernel(std::vector<std::tuple<double, double> > ThetasCurr, std::random_device& rd ){

    std::mt19937 engine( rd() );
    std::normal_distribution<double> normal(0, 0.05);

    //std::vector<std::tuple<double, double> > ThetasProp = {std::make_tuple(0,0),std::make_tuple(0,0)} ;

    std::vector<std::tuple<double, double> > ThetasProp = ThetasCurr;

//   for(int i = 0; i < ThetasCurr.size(); ++i)  {
//
//        std::get<0>(ThetasProp[i]) = -1.;
//        while(std::get<0>(ThetasProp[i]) < 0){
//            std::get<0>(ThetasProp[i]) = normal(engine) + std::get<0>(ThetasCurr[i]);
//        }
//
//        std::get<1>(ThetasProp[i]) = -1.;
//        while(std::get<1>(ThetasProp[i]) < 0){
//            std::get<1>(ThetasProp[i]) = normal(engine) + std::get<1>(ThetasCurr[i]);
//        }
//  }

    std::get<0>(ThetasProp[0]) = -1;
    while(std::get<0>(ThetasProp[0]) < 0){
               std::get<0>(ThetasProp[0]) = normal(engine) + std::get<0>(ThetasCurr[0]);
           }

    std::get<1>(ThetasProp[0]) = -1;
        while(std::get<1>(ThetasProp[0]) < 0){
                   std::get<1>(ThetasProp[0]) = normal(engine) + std::get<1>(ThetasCurr[0]);
               }

    return ThetasProp ;
}



Eigen::MatrixXd ModelSampleGen (int numSamples, std::vector<std::tuple<double, double> > Thetas ,
                                const std::vector < std::pair <double, double> >& forcingCDF ){

    TupleTrussDef TrussDef2;

    TrussDef2 =  InitialTrussAssignment();

    FEMClass ModelTrussFem(false, TrussDef2 );

    std::lognormal_distribution<double> lognormal1( std::get<0>(Thetas[0]), std::get<1>(Thetas[0])  );
    std::lognormal_distribution<double> lognormal2( std::get<0>(Thetas[1]), std::get<1>(Thetas[1])  );

    std::uniform_real_distribution<double> uniform( 0, 1  );

    std::random_device rd;
    std::mt19937 engine( rd() );

    Eigen::MatrixXd allModelSamples (numSamples ,1);
    double forceVal;

    for(int i = 0; i < numSamples ; i++){

           double A1 = lognormal1( engine );
           double A2 = lognormal2( engine );

           double forcingProbVal = uniform ( engine );

           for(unsigned j = 0; j < forcingCDF.size() ; ++j){
               if(forcingProbVal < forcingCDF[j].second){
                   forceVal = forcingCDF[j].first;
                   break;
               }
           }

           ModelTrussFem.modA(0, A1);
           //ModelTrussFem.modA(0, A2);

           //ModelTrussFem.modForce(3, 1, forceVal);
           //trueTrussFem.modForce(3, 0, forceVal);

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

    return  log(TrueSMean - MeanModelCurr) * 2;
}




int main(){

    std::tuple<Eigen::MatrixXd, std::vector<double> > trueSamplesTuple = trueSampleGen();
    Eigen::MatrixXd trueSamples;
    std::vector<double> forcing;

    std::tie (trueSamples, forcing) = trueSamplesTuple;
    //std::cout << trueSamples << std::endl;

    std::vector < std::pair <double, double> > forcingCDF;// < force, ctr >
    double forceVal;
    double tol = 0.01;
    bool presentForceCtr = false;

    std::cout<< "forcing size = " << forcing.size() << std::endl;

    for(unsigned i = 0; i < forcing.size(); ++i){

        forceVal = snap(forcing[i], 1 );
        presentForceCtr = false;

        for(unsigned j = 0; j < forcingCDF.size(); ++j){

            if(  abs(forceVal - forcingCDF[j].first ) < tol ){
                forcingCDF[j].second += 1.0 / forcing.size() ;
                presentForceCtr = true;
                break;
              }
        }
        if ( !presentForceCtr  ){
                forcingCDF.push_back( std::make_pair(forceVal, 1. / forcing.size() ) );
            }
        }

    double cumulProb = 0 ;
    for(unsigned i = 0; i < forcingCDF.size(); ++i){

        cumulProb += forcingCDF[i].second;
        forcingCDF[i].second = cumulProb;

        std::cout << forcingCDF[i].first << " " << forcingCDF[i].second << '\n';
    }


    std::random_device rd;

    unsigned loops = 1e5;

    std::vector<std::tuple<double, double> > ThetasCurr;
    std::vector<std::tuple<double, double> > ThetasProp;

    ThetasCurr = { std::make_tuple(0., 0.01),  std::make_tuple(1., 0.25) };

    Eigen::MatrixXd ModelSamplesCurr;
    Eigen::MatrixXd ModelSamplesProp;



    ModelSamplesCurr = ModelSampleGen (trueSamples.rows(), ThetasCurr , forcingCDF);



    //Distance measure taken as squared difference in mean
    //Should be proper Statistical distance of Distributions like KL-div etc..

    // double distanceCurr = steinDisc(trueSamples, ModelSamplesCurr ) ;
     double distanceCurr = lossFunctionMSEAvg(trueSamples, ModelSamplesCurr ) ;



    double distanceProp;

    for(int i = 0; i < loops; ++i){


        ThetasProp = proposalKernel( ThetasCurr, rd );

        ModelSamplesProp = ModelSampleGen ( trueSamples.rows(), ThetasProp,  forcingCDF );


        //distanceProp = steinDisc(trueSamples, ModelSamplesProp );
        distanceProp = lossFunctionMSEAvg(trueSamples, ModelSamplesProp );
//
//        if( distanceProp > distanceCurr && abs(ThetasProp - 10) < abs(muCurr-10) ) {
//            std::cout << "x" << std::endl;
//            falseAsses++;
//        }

        if( distanceProp < distanceCurr){

            distanceCurr = distanceProp;

//            if( muCurr < 10 && muProp < muCurr) {
//
//                std::cout << "x" << std::endl;
//                falseAsses++;
//
//            }

            ThetasCurr = ThetasProp;
            ModelSamplesCurr = ModelSamplesProp;

            for(unsigned k = 0; k < ThetasCurr.size(); ++k){
            std::cout << "Iter: "<< i <<" muCurr " <<k<<" = "<< std::get<0>(ThetasCurr[k]) << " sigCurr " <<k<<" = "<< std::get<1>(ThetasCurr[k])
                      << " Distance = " << distanceCurr <<std::endl;
            }
            std::cout << std::endl;
        }else{
        for(unsigned k = 0; k < ThetasProp.size(); ++k){
                    std::cout << "Iter: "<< i <<" muProp " <<k<<" = "<< std::get<0>(ThetasProp[k]) << " sigProp " <<k<<" = "<< std::get<1>(ThetasProp[k])
                              << " Distance = " << distanceProp <<std::endl;
                    }
        std::cout << std::endl;
        }
    }

    for(unsigned k = 0; k < ThetasCurr.size(); ++k){
                std::cout <<" muCurr" <<k<<" = "<< std::get<0>(ThetasCurr[k]) << " sigCurr1 " <<k<<" = "<< std::get<1>(ThetasCurr[k])
                          << " Distance = " << distanceCurr <<std::endl;
                }


    std::vector<double>  delatXs = findDeltaX(ModelSamplesCurr, 100);
    int nBins = 100;
    HistContainer histPoints = histBin(ModelSamplesCurr, delatXs, true, true);


    return 0;
}









