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
    std::vector <int> numSamples {2, 1, 0, 0, 0};
    //std::vector <int> numSamples {5,5, 5, 5, 5};

    int numObsVects = std::accumulate(numSamples.begin(), numSamples.end(), 0);
    std::cout << "numObsVects \n" << numObsVects << std::endl;
    Eigen::MatrixXd ObsDisp        (1, numObsVects * ObsIndex.size() );
    Eigen::MatrixXi yToLoadIndexes ( numObsVects, 3  );

    std::cout << "Here" << std::endl;

//----------------------- Determining Forcing ---------------------------------


    vecMat forceContainer( numLoadingCases );
    double Kn = 1000 * 10 * 5;
    Eigen::MatrixXd forcing1 ( 14 * 3 , 1 ) ;
    forcing1.setZero();
    forcing1( 2 * 3 + 1  , 0 )  = -1*Kn;
    forcing1( 4 * 3 + 1  , 0 )  = -2*Kn;
    forcing1( 9 * 3 + 1  , 0 )  = -1*Kn;
    forcing1( 11 * 3 + 1  , 0 )  = -2*Kn;

    //forcing1( 11 * 3 + 0 , 0 )  = -1e4;
    //forcing1( 4 * 3  + 0  , 0 )  = -2e4;

    forceContainer[0] = forcing1;


    Eigen::MatrixXd forcing2 ( 14 * 3 , 1 ) ;
    forcing2.setZero();
    forcing2( 2 * 3 + 0 , 0 ) = -2*Kn;
    forcing2( 9 * 3 + 0 , 0 ) = -2*Kn;

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

    //std::vector <Eigen::MatrixXd> allSamplesContainer ( numLoadingCases );

    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();
    //List of dofs of reduced system
    std::vector<int> dofK;

    FEMClass trueTrussFem(false, TrussDef );

    //Can measure ex: +- 1cm to 95% confidence
    // 0.01m = 2\sigma; \sigma = 0.005m
    double sigma_n = 0.005;
    std::normal_distribution<double> normal( 0, sigma_n );
    std::random_device rd;
    std::mt19937 engine( rd() );

    int ctr = 0;
    for(int f = 0 ; f < numSamples.size() ; ++f ){
    	if( numSamples[f] == 0 ){ continue;}

    	trueTrussFem.FEMClassReset(false);
        trueTrussFem.modForce( forceContainer[f] );
        std::cout << "HereA" << std::endl;

        double A1 = 0.04 ;
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
				ObsDisp(0, ctr )        = allDispTruss[ ObsIndex[j] ] + normal( engine );
				ctr++;
			}

			yToLoadIndexes(f + i, 0) = f;
			yToLoadIndexes(f + i, 2) = 0;//ObsIndex row
			yToLoadIndexes(f + i, 2) = numSamples[f];
        }

        std::cout << " allDispTruss \n" << allDispTruss << std::endl;
        std::cout << " ObsIndex \n"     << ObsIndex << std::endl;
        std::cout << " ObsDisp \n"      << ObsDisp << std::endl;
        //===new code

    }



    //std::cout << " \n\n\n\nDone New Code \n\n\n\n\n"  << std::endl;


/*

    for(int f = 0 ; f < numLoadingCases; ++f   ){

        Eigen::MatrixXd allSamples (numSamples[f], ObsIndex.cols() );

        trueTrussFem.modForce( forceContainer[f] );

        double A1 = 0.04 ;
        trueTrussFem.modA(13, A1);

        trueTrussFem.assembleS( );

        trueTrussFem.computeDisp( );
        //trueTrussFem.computeForce( );

        dofK = trueTrussFem.getFreeDof() ;
        Eigen::VectorXd dispTruss = trueTrussFem.getDisp( );

        //===new code
        Eigen::VectorXd allDispTruss = trueTrussFem.getAllDisp();
        int positionCtr = 0;
        for(int i = 0; i < ObsIndex.cols(); ++i){
        	ObsDisp(0, i) = allDispTruss[ObsIndex(0,i)];
			positionCtr ++;
        }


        std::cout << " allDispTruss \n" << allDispTruss << std::endl;
        std::cout << " ObsIndex \n"     << ObsIndex << std::endl;
        std::cout << " ObsDisp \n"      << ObsDisp << std::endl;
        //===new code


        //std::cout << dispTruss << std::endl;

        double propHalfMax = 0.1;

        //compute Fullwidth half max
        Eigen::VectorXd fwhmStd ( ObsIndex.cols() );

        for(int i = 0; i < numSamples[f] ; i++){
            //want to only see data that is in ObsIndex
            for(int j = 0; j < ObsIndex.cols(); ++j ){
                for( int l = 0; l < dofK.size() ; ++l){
                    if( ObsIndex(0,j) == dofK[l] ){

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
*/
    //myTrueFile.close();

    trueSamplesTupleContainer = std::make_tuple( ObsDisp, yToLoadIndexes , forceContainer, sigma_n );

    return trueSamplesTupleContainer;

}

#endif /* APP_LAPLACEAPPROX_SAMPLEGEN_HPP_ */
