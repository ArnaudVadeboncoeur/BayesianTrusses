/*
 * sampleGen.hpp
 *
 *  Created on: 12 Feb 2020
 *      Author: arnaudv
 */

#ifndef APP_LAPLACEAPPROX_SAMPLEGEN_HPP_
#define APP_LAPLACEAPPROX_SAMPLEGEN_HPP_


#include "../../src/FEMClass.hpp"

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

    std::normal_distribution<double> normal( 0, 0.0005 );

    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples = 40;

    std::vector<double> forcing (numSamples) ;

    Eigen::MatrixXd allSamples (numSamples, 3);

    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();

    FEMClass trueTrussFem(false, TrussDef );

    for(int i = 0; i < numSamples ; i++){

        double A1 = 0.06 + normal( engine ) ;
        double A2 = 0.04 + normal( engine ) ;
        double A3 = 0.02 + normal( engine ) ;

        trueTrussFem.modA(0, A1);
        trueTrussFem.modA(1, A2);
        trueTrussFem.modA(2, A3);

        trueTrussFem.assembleS( );
        trueTrussFem.computeDisp( );
        trueTrussFem.computeForce( );

        allSamples(i, 0) = trueTrussFem.getDisp( 9 ) ;
        allSamples(i, 1) = trueTrussFem.getDisp(10 ) ;
        allSamples(i, 2) = trueTrussFem.getDisp(11 ) ;



        myTrueFile << A1 << " " << A2 << " " << A3 << " " << allSamples(i, 0)<< " " << allSamples(i, 1)<<" "<< allSamples(i, 2) << '\n';

        trueTrussFem.FEMClassReset(false);
        if( verbosity == true){if( (numSamples > 100 * 5 ) && ( i % (numSamples / ( 20 ) )  == 0 ) ){std::cout << "computed " << i << " samples " <<'\n';}}
    }

    myTrueFile.close();

    return std::make_tuple(allSamples, forcing );
}

#endif /* APP_LAPLACEAPPROX_SAMPLEGEN_HPP_ */
