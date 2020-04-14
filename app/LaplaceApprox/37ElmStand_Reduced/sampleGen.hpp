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


void trueSampleGen( std::tuple<Eigen::MatrixXd, std::vector<double> >& trueSamplesTuple ){



    bool verbosity = false;

    std::ofstream myTrueFile;
    myTrueFile.open("trueResults.dat", std::ios::trunc);

    //std::normal_distribution<double> normal( 0, 0.0003 );

    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples = 5;

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
                36, 37;//all non fixed x and y disp

    //Eigen::VectorXi ObsIndex( 10 ); ObsIndex << 4, 7, 10, 13, 16, 25, 28, 31, 34, 37;//all non fixed y disp
    //Eigen::VectorXi ObsIndex( 3 ); ObsIndex << 9, 10, 11;


    //std::cout << ObsIndex<<"\n\n" << std::endl;

    std::vector<double> forcing (numSamples) ;

    Eigen::MatrixXd allSamples (numSamples, ObsIndex.size() );

    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();
    //List of dofs of reduced system
    std::vector<int> dofK;

    FEMClass trueTrussFem(false, TrussDef );



//        double A1 = 0.06 + normal( engine ) ;
//        double A2 = 0.04 + normal( engine ) ;


        double A1 = 0.04 ;
        //double A2 = 0.02 ;//0.04 ;
        //double A3 = 0.02 ;
        //A2 = A1 ;

        trueTrussFem.modA(13, A1);
        //trueTrussFem.modA(1, A1);


        trueTrussFem.assembleS( );
        std::cout << "trueTrussFem.getK() \n" << trueTrussFem.getK() <<std::endl;

        trueTrussFem.computeDisp( );
        trueTrussFem.computeForce( );

        dofK = trueTrussFem.getFreeDof() ;
        Eigen::VectorXd dispTruss = trueTrussFem.getDisp( );

        double propHalfMax = 0.5;

        //compute Fullwidth half max
        Eigen::VectorXd fwhmStd ( ObsIndex.size() );

        for(int i = 0; i < numSamples ; i++){
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


        trueTrussFem.FEMClassReset(false);
    }

    myTrueFile.close();

    std::cout << "fwhmStd\n" << fwhmStd << std::endl;

    trueSamplesTuple = std::make_tuple(allSamples, forcing );

}

#endif /* APP_LAPLACEAPPROX_SAMPLEGEN_HPP_ */
