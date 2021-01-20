/*
 * sampleGen.hpp
 *
 *  Created on: 12 Feb 2020
 *      Author: arnaudv
 */

#ifndef APP_LAPLACEAPPROX_SAMPLEGEN_HPP_
#define APP_LAPLACEAPPROX_SAMPLEGEN_HPP_


#include "../../../src/FEMClass.hpp"
#include "../Truss37Elm.hpp"

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <math.h>
#include <numeric>



using vecMat   =   std::vector < Eigen::MatrixXd > ;
using DataCont =   std::tuple  < Eigen::MatrixXd, Eigen::MatrixXi ,vecMat, double > ;

DataCont trueSampleGen( Eigen::VectorXi ObsIndex ){

    DataCont trueSamplesTupleContainer;

    bool verbosity = false;

    //std::ofstream myTrueFile;
    //myTrueFile.open("trueResults.dat", std::ios::trunc);

    int numLoadingCases = 5;
    //std::vector <int> numSamples {5,5, 5, 5, 5};
    std::vector <int> numSamples {5,5, 5, 5, 5};

    int numObsVects = std::accumulate(numSamples.begin(), numSamples.end(), 0);
    std::cout << "numObsVects \n" << numObsVects << std::endl;
    Eigen::MatrixXd ObsDisp        (numObsVects, ObsIndex.size() );
    Eigen::MatrixXi yToLoadIndexes ( numObsVects, 3  );

//----------------------- Determining Forcing ---------------------------------


    vecMat forceContainer( numLoadingCases );
    double Kn = 1000 * 10 * 10;
    Eigen::MatrixXd forcing1 ( 14 * 3 , 1 ) ;
    forcing1.setZero();
    forcing1( 2 * 3 + 1  , 0 )  = -1*Kn;
    forcing1( 4 * 3 + 1  , 0 )  = -2*Kn;
    forcing1( 9 * 3 + 1  , 0 )  = -1*Kn;
    forcing1( 11 * 3 + 1  , 0 )  = -2*Kn;

    forceContainer[0] = forcing1;


    Eigen::MatrixXd forcing2 ( 14 * 3 , 1 ) ;
    forcing2.setZero();
    forcing2( 12 * 3 + 1 , 0 ) = -2*Kn;
    forcing2( 5 * 3 + 1 , 0 ) = -1*Kn;
    forcing2( 3 * 3 + 1 , 0 ) = -2*Kn;
	forcing2( 10 * 3 + 1 , 0 ) = -1*Kn;


    forceContainer[1] = forcing2;


    Eigen::MatrixXd forcing3 ( 14 * 3 , 1 ) ;
    forcing3.setZero();
    forcing3( 9 * 3 + 1 , 0 ) =  1*Kn;

    forceContainer[2] = forcing3;

    Eigen::MatrixXd forcing4 ( 14 * 3 , 1 ) ;
    forcing4.setZero();
    forcing4( 4 * 3  + 0  , 0 )   = -2*Kn;
    forcing4( 11 * 3  + 0  , 0 )  = -2*Kn;

    forceContainer[3] = forcing4;

    Eigen::MatrixXd forcing5 ( 14 * 3 , 1 ) ;
    forcing5.setZero();
    forcing5( 11 * 3 + 2 , 0 )  = -2*Kn;
    forcing5( 4 * 3  + 2  , 0 )  =  2*Kn;

    forceContainer[4] = forcing5;



//----------------------- Generate Samples ---------------------------------

    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();
    //List of dofs of reduced system
    std::vector<int> dofK;

    FEMClass trueTrussFem(false, TrussDef );

    //Can measure ex: +- 1cm to 95% confidence
    // 0.001m = 2\sigma; \sigma = 0.0005m
    double sigma_n = 0.0005;
    //95% +- val = 2 * sigma_n
    std::cout << "sigma noise --> +-" << 2 * sigma_n << std::endl;
    std::normal_distribution<double> normal( 0, sigma_n );
    std::random_device rd;
    std::mt19937 engine( rd() );

    int obsCtr = 0;
    for(int f = 0 ; f < numSamples.size() ; ++f ){
    	if( numSamples[f] == 0 ){ continue;}

    	trueTrussFem.FEMClassReset(false);
        trueTrussFem.modForce( forceContainer[f] );

        double A1 = 0.005 ;
        trueTrussFem.modA(13, A1);

        trueTrussFem.assembleS( );

        trueTrussFem.computeDisp( );
        //trueTrussFem.computeForce( );

        dofK = trueTrussFem.getFreeDof() ;
        Eigen::VectorXd dispTruss = trueTrussFem.getDisp( );

        //===new code
        Eigen::VectorXd allDispTruss = trueTrussFem.getAllDisp();
        for(int i = 0; i < numSamples[f]; ++i){
			for(int j = 0; j < ObsIndex.size(); ++j){
				ObsDisp(obsCtr, j )        = allDispTruss[ ObsIndex[j] ] + normal( engine );
			}

			yToLoadIndexes(obsCtr, 0) = f;
			yToLoadIndexes(obsCtr, 1) = 0;//ObsIndex row
			yToLoadIndexes(obsCtr, 2) = numSamples[f];
			obsCtr ++;
        }

//        std::cout << " allDispTruss \n" << allDispTruss << std::endl;
//        std::cout << " ObsIndex \n"     << ObsIndex << std::endl;
//        std::cout << " ObsDisp \n"      << ObsDisp << std::endl;
//        std::cout << " yToLoadIndexes \n"      << yToLoadIndexes << std::endl;
        //===new code

    }
    std::cout << "ytl : \n" << yToLoadIndexes << std::endl;
    trueSamplesTupleContainer = std::make_tuple( ObsDisp, yToLoadIndexes , forceContainer, sigma_n );

    return trueSamplesTupleContainer;

}

#endif /* APP_LAPLACEAPPROX_SAMPLEGEN_HPP_ */
