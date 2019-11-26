/*
 * SampleAnalysis.hpp
 *
 *  Created on: 26 Nov 2019
 *      Author: arnaudv
 */

#ifndef SRC_SAMPLEANALYSIS_HPP_
#define SRC_SAMPLEANALYSIS_HPP_


#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include <Eigen/Core>


void MonteCarloAvgs(const Eigen::MatrixXd& allSamples){
    //Computes Average over range
   double upperBound =  1e10;
   double lowerBound = -1e10;
   double valMax = 0;
   double valMin = 0;
   int sampleCtr = 0;
   double monteCarloSum  = 0;

   for( unsigned i = 0; i < allSamples.rows(); ++i ){

       if( (allSamples(i, 1) > lowerBound) || (allSamples(i, 1) < upperBound) ){

           if( allSamples(i, 1) > valMax )   {
               valMax = allSamples(i, 1);
           }
           if(allSamples(i, 1) < valMin) {
               valMin = allSamples(i, 1);
           }

           monteCarloSum += allSamples(i, 1);

           sampleCtr ++ ;
       }

   }

   std::cout <<"monteCarloSum = " << monteCarloSum << '\n';
   std::cout <<"valMax = " << valMax << '\n';
   std::cout <<"valMin = " << valMin << '\n';
   std::cout <<"sampleCtr = " << sampleCtr << '\n';


   monteCarloSum = (valMax - valMin)  *  (double) 1 / sampleCtr  * monteCarloSum;

   std::cout <<"monteCarloSum = " << monteCarloSum << '\n';


    return;
}





double snap(double y, double precision){
   double reduced;
   double modulo;

   int n = std::floor( ( y ) / precision );
   modulo = y - n * precision;

   while( modulo > precision){
   modulo =  modulo - precision;
   std::cout << "loop ->0"<<std::endl ;
   }

   if( modulo < precision / 2.0 ){
      reduced = y - modulo;
   }
   else{
      reduced = y + (precision - modulo);
   }

   return reduced;
}




void histBin(const Eigen::MatrixXd& allSamples,
             const double& valMax, const double& valMin,
             bool normalized, bool output){

    std::vector< std::pair < Eigen::VectorXd , double> >  snapMappingFreq;
    std::vector< Eigen::VectorXd > snapMap (allSamples.rows()) ;
    Eigen::VectorXd mapping ( allSamples.cols() );

    int nBins = 100;
    double deltaX = abs(valMax - valMin) / nBins ;

    for(unsigned i = 0; i < allSamples.rows(); ++i){

        for(unsigned int j = 0; j < allSamples.cols() ; ++j){
            mapping[j] = snap(allSamples(i, j), deltaX );
            //std::cout<< mapping[j] <<"\t"<<allSamples(i, j)<<std::endl;
        }
        snapMap[i] = mapping;
    }

    Eigen::VectorXd current;
    for(unsigned i = 0; i < snapMap.size(); ++i){
        current = snapMap[i];
        bool equal = false;

        double tolerence = 1e-10;

        for(unsigned j = 0; j < snapMappingFreq.size(); ++j){

            double difference =0;
            for(unsigned k =0; k < current.size(); ++k){

                difference += abs(current[k] - snapMappingFreq[j].first[k]);
            }

            if( difference < tolerence  ){
                equal = true;
                snapMappingFreq[j].second ++;
                break;
            }
            else{equal = false;}
        }
        if( !equal){
            snapMappingFreq.push_back(std::make_pair(current, 1.));
        }
    }

    if(normalized){
        for(unsigned i = 0; i < snapMappingFreq.size(); ++i){
            snapMappingFreq[i].second =(double) snapMappingFreq[i].second / allSamples.rows();
        }

    }

    if(output){

        std::ofstream myFile;
        myFile.open("results.dat", std::ios::trunc);

        for(int i =0; i < snapMappingFreq.size(); ++i){
            myFile<< snapMappingFreq[i].first <<" "<<snapMappingFreq[i].second << std::endl;

        }
        myFile.close();
    }
    return;
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
