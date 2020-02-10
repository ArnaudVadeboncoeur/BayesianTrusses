/*
 * SampleAnalysis.hpp
 *
 *  Created on: 26 Nov 2019
 *      Author: arnaudv
 */

#ifndef SRC_STATTOOLS_SAMPLEANALYSIS_HPP_
#define SRC_STATTOOLS_SAMPLEANALYSIS_HPP_


#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include <Eigen/Dense>

double SampleMean(const Eigen::MatrixXd& allSamples, int column ){
    //Computes Average over range
   int sampleCtr = 0;
   double monteCarloSum  = 0;
   double monteCarloAvg;

   bool verbosity = false;

   for( unsigned i = 0; i < allSamples.rows(); ++i ){

       monteCarloSum += allSamples(i, column);

           sampleCtr ++ ;
       }



   monteCarloAvg =  monteCarloSum / sampleCtr ;


   return monteCarloAvg ;
}

double MonteCarloAvgs(const Eigen::MatrixXd& allSamples, int column ){
    //Computes Average over range
   double upperBound =  1e10;
   double lowerBound = -1e10;
   double valMax = 0;
   double valMin = 0;
   int sampleCtr = 0;
   double monteCarloSum  = 0;

   bool verbosity = false;

   for( unsigned i = 0; i < allSamples.rows(); ++i ){

       if( (allSamples(i, column) > lowerBound) || (allSamples(i, column) < upperBound) ){

           if( allSamples(i, column) > valMax )   {
               valMax = allSamples(i, column);
           }
           if(allSamples(i, column) < valMin) {
               valMin = allSamples(i, column);
           }

           monteCarloSum += allSamples(i, column);

           sampleCtr ++ ;
       }
   }


   monteCarloSum = (valMax - valMin)  *  (double) 1 / sampleCtr  * monteCarloSum;


   if (verbosity){
        std::cout <<"monteCarloSum = " << monteCarloSum << '\n';
        std::cout <<"valMax = " << valMax << '\n';
        std::cout <<"valMin = " << valMin << '\n';
        std::cout <<"sampleCtr = " << sampleCtr << '\n';
   }


   return monteCarloSum ;
}


void FreqIntergral (const Eigen::MatrixXd& allSamples, const double& valMax, const double& valMin) {

    std::cout << "valMax = " << valMax << "\tvalMin = " << valMin << '\n';
    Eigen::VectorXd lower ( allSamples.cols() ); lower << valMin;
    Eigen::VectorXd upper ( allSamples.cols() ); upper << (valMax + valMin) / 2.0;

    double freqInterg = 0;
    std::cout << allSamples.rows() << std::endl;
    for(unsigned i = 0; i < allSamples.rows(); ++i){
        for(unsigned j = 0; j < allSamples.cols(); ++j){

            if( (allSamples(i, j) > lower[j] ) && (allSamples(i, j) < upper[j]) ){
                freqInterg ++;
            }

        }

    }
    freqInterg = (double) freqInterg / ( allSamples.rows() * allSamples.cols() );
    std::cout << "Frequency Intergral = " << freqInterg << std::endl;


    return ;
}



#endif
