/*
 * sampleGen.hpp
 *
 *  Created on: 12 Feb 2020
 *      Author: arnaudv
 */

#ifndef APP_LAPLACEAPPROX_SAMPLEGEN_HPP_
#define APP_LAPLACEAPPROX_SAMPLEGEN_HPP_


#include "../../../src/FEMClass.hpp"

#include "ThreeDTruss37Elm.hpp"
//#include "ThreeDTruss23Elm.hpp"
//#include "ThreeDTruss3Elm.hpp"

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <math.h>

using DataCont =   std::tuple < std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd> > ;


DataCont trueSampleGen( ){

    constexpr double numLoadingCases = 3;

    DataCont trueSamplesTupleContainer;

    bool verbosity = false;

    std::ofstream myTrueFile;
    myTrueFile.open("trueResults.dat", std::ios::trunc);

    std::random_device rd;
    std::mt19937 engine( rd() );

    std::vector <int> numSamples {5, 5, 6};


//---------------------- Determining Indeces of Observed Data ------------------------------------

    //Index of dofs observed
    //std::cout << "Here" << std::endl;

//    Eigen::VectorXi nodesObs( 4 ); nodesObs <<    1, 2, 6, 7;
//    //Eigen::VectorXi nodesObs( 1 ); nodesObs <<  1;
//        Eigen::VectorXi ObsIndex( nodesObs.size() * 3 );
//        for(int j = 0; j < nodesObs.size(); ++j){
//
//            ObsIndex[ j*3 + 0] = nodesObs[j]*3 + 0;
//            ObsIndex[ j*3 + 1] = nodesObs[j]*3 + 1;
//            ObsIndex[ j*3 + 2] = nodesObs[j]*3 + 2;
//        }
//
//    Eigen::VectorXi nodesObs( 10 ); nodesObs <<   1, 2,3,4, 5, 8, 9,10,  11, 12;    Eigen::VectorXi ObsIndex( nodesObs.size() * 2 );
//            for(int j = 0; j < nodesObs.size(); ++j){
//
//                ObsIndex[ j*2 + 0] = nodesObs[j]*3 + 0;
//                ObsIndex[ j*2 + 1] = nodesObs[j]*3 + 1;
//            }

    //all non fixed x and y disp
    Eigen::VectorXi ObsIndex( 24 );
    ObsIndex << 3,  4,
                6,  7,  8,
                9,  10,
                12, 13, 14,
                15, 16,

                25, 25,
                27, 28, 29,
                30, 31,
                33, 34, 35,
                36, 37;

    //Eigen::VectorXi ObsIndex( 10 ); ObsIndex << 4, 7, 10, 13, 16, 25, 28, 31, 34, 37;//all non fixed y disp
    //Eigen::VectorXi ObsIndex( 3 ); ObsIndex << 9, 10, 11;

    //std::cout << ObsIndex<<"\n\n" << std::endl;

//----------------------- Determining Forcing ---------------------------------


    std::vector <Eigen::MatrixXd> forceContainer( numLoadingCases );

    Eigen::MatrixXd forcing1 ( 14 * 3 , 1 ) ;
    forcing1.setZero();
    forcing1( 2 * 3 + 1  , 0 )  = -1e4;
    forcing1( 4 * 3 + 1  , 0 )  = -2e4;
    forcing1( 11 * 3 + 0 , 0 )  = -1e4;

    Eigen::MatrixXd forcing2 ( 14 * 3 , 1 ) ;
    forcing2.setZero();
    forcing2( 4 * 3 + 2 ) = -2e4;
    forcing2( 4 * 3 + 1 ) = -2e4;

    Eigen::MatrixXd forcing3 ( 14 * 3 , 1 ) ;
    forcing3.setZero();
    forcing3( 9 * 3 + 1 ) =  1e4;

    forceContainer[0] = forcing1;
    forceContainer[1] = forcing2;
    forceContainer[2] = forcing3;




//----------------------- Generate Samples ---------------------------------

    std::vector <Eigen::MatrixXd> allSamplesContainer ( numLoadingCases );

    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();
    //List of dofs of reduced system
    std::vector<int> dofK;

    FEMClass trueTrussFem(false, TrussDef );

    for(int f = 0 ; f < numLoadingCases; ++f   ){

        Eigen::MatrixXd allSamples (numSamples[f], ObsIndex.size() );
        std::cout << "Here-Now1" << std::endl;

        trueTrussFem.modForce( forceContainer[f] );
        std::cout << "Here-Now2" << std::endl;

        double A1 = 0.04 ;
        trueTrussFem.modA(13, A1);
        std::cout << "Here-Now3" << std::endl;

        trueTrussFem.assembleS( );
        std::cout << "Here-Now4-1" << std::endl;
        //std::cout << "trueTrussFem.getK() \n" << trueTrussFem.getK() <<std::endl;


        trueTrussFem.computeDisp( );
        //trueTrussFem.computeForce( );
        std::cout << "Here-Now4" << std::endl;

        dofK = trueTrussFem.getFreeDof() ;
        Eigen::VectorXd dispTruss = trueTrussFem.getDisp( );
        std::cout << "Here-Now5" << std::endl;

        double propHalfMax = 0.5;

        //compute Fullwidth half max
        Eigen::VectorXd fwhmStd ( ObsIndex.size() );

        for(int i = 0; i < numSamples[f] ; i++){
            //want to only see data that is in ObsIndex
            for(int j = 0; j < ObsIndex.size(); ++j ){
                for( int l = 0; l < dofK.size() ; ++l){
                    if( ObsIndex[j] == dofK[l] ){

                        fwhmStd[j] =  std::abs( ( dispTruss[ l ] * propHalfMax ) / 2.355 );
                        std::normal_distribution<double> normal( 0, fwhmStd[j] );

                        //std::normal_distribution<double> normal( 0,  0.01* dispTruss.mean() );
                        //std::cout << "0.1* dispTruss.mean() " << 0.01* dispTruss.mean() << std::endl;

                        allSamples(i, j) = dispTruss[ l ] + normal( engine ) ;
                    }
                }

                myTrueFile << allSamples(i, j) << " ";
            }

            myTrueFile << std::endl;

        }
        allSamplesContainer[f] = allSamples;

        myTrueFile << "\n\n";
        trueTrussFem.FEMClassReset(false);
    }

    myTrueFile.close();

    trueSamplesTupleContainer = std::make_tuple( allSamplesContainer, forceContainer );

    //std::cout << "trueTrussFem.getK() \n" << trueTrussFem.getK() <<std::endl;

    return trueSamplesTupleContainer;

}

#endif /* APP_LAPLACEAPPROX_SAMPLEGEN_HPP_ */
