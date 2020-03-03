/*
 * sampleGen.hpp
 *
 *  Created on: 12 Feb 2020
 *      Author: arnaudv
 */

#ifndef APP_LAPLACEAPPROX_SAMPLEGEN_HPP_
#define APP_LAPLACEAPPROX_SAMPLEGEN_HPP_


#include "../../../src/FEMClass.hpp"
#include "ThreeDTruss23Elm.hpp"

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

    std::normal_distribution<double> normal( 0, 0.0005 );

    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples = 5;

    Eigen::VectorXi nodesFree(6); nodesFree << 1, 2, 3, 6, 7, 8;
    Eigen::VectorXi dofs( nodesFree.size() * 3 );
    for(int j = 0; j < nodesFree.size(); ++j){

        dofs[ j*3 + 0] = nodesFree[j]*3 + 0;
        dofs[ j*3 + 1] = nodesFree[j]*3 + 1;
        dofs[ j*3 + 2] = nodesFree[j]*3 + 2;
    }


    std::vector<double> forcing (numSamples) ;

    Eigen::MatrixXd allSamples (numSamples, dofs.size() );

    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();


    FEMClass trueTrussFem(false, TrussDef );

    for(int i = 0; i < numSamples ; i++){

        double A1 = 0.06 + normal( engine ) ;
        double A2 = 0.04 + normal( engine ) ;


        trueTrussFem.modA(0, A1);
        trueTrussFem.modA(1, A2);


        trueTrussFem.assembleS( );
        trueTrussFem.computeDisp( );
        trueTrussFem.computeForce( );

        myTrueFile << A1 << " " << A2 << " ";
        for(int j =0; j< dofs.size(); ++j){

            allSamples(i, j) = trueTrussFem.getDisp( j ) ;
            myTrueFile << allSamples(i, j) << " ";

        }
        myTrueFile << std::endl;


        trueTrussFem.FEMClassReset(false);
    }

    myTrueFile.close();


    trueSamplesTuple = std::make_tuple(allSamples, forcing );

}

#endif /* APP_LAPLACEAPPROX_SAMPLEGEN_HPP_ */
