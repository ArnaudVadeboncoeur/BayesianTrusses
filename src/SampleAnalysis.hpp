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

#include <Eigen/Dense>


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

//   while( modulo > precision){
//   modulo =  modulo - precision;
//   std::cout << "loop -> 0"<<std::endl ;
//   }

   if( modulo < precision / 2.0 ){
      reduced = y - modulo;
   }
   else{
      reduced = y + (precision - modulo);
   }

   return reduced;
}




void histBin(const Eigen::MatrixXd& allSamples,int nBins, bool normalized, bool output){

    std::vector< std::pair < Eigen::VectorXd , double> >  snapMappingFreq;


    std::vector< Eigen::VectorXd > snapMap (allSamples.rows()) ;
    Eigen::VectorXd mapping ( allSamples.cols() );

    Eigen::VectorXd valsMax ( allSamples.cols() );
    Eigen::VectorXd valsMin ( allSamples.cols() );

    for(unsigned i = 0; i < valsMax.size(); ++i){
        valsMax[i] = -1e9;
        valsMin[i] = +1e9;
    }

    Eigen::VectorXd deltaXs ( allSamples.cols() );

    for(unsigned     j = 0; j < allSamples.cols() ; ++j){
        for(unsigned i = 0; i < allSamples.rows() ; ++i){

            if( allSamples(i, j) > valsMax[j]) { valsMax[j] = allSamples(i, j) ;}
            if( allSamples(i, j) < valsMin[j]) { valsMin[j] = allSamples(i, j) ;}
        }
        }

    for(unsigned i = 0; i < deltaXs.size(); ++i ){

        deltaXs[i] =  abs(  (double) ((valsMax[i] - valsMin[i]) / nBins ) );
//        std::cout << deltaXs[i] << std::endl;
//        std::cout << valsMax[i] << std::endl;
//        std::cout << valsMin[i] << std::endl;

    }

    std::cout << "\n\nSnaping\n";
    for(unsigned i = 0; i < allSamples.rows(); ++i){

        for(unsigned int j = 0; j < allSamples.cols() ; ++j){
            mapping[j] = snap(allSamples(i, j), deltaXs[j] );
            //std::cout<< mapping[j] <<"\t"<<allSamples(i, j)<<std::endl;
        }
        snapMap[i] = mapping;
    }

    std::cout << "\n\nCounting Frequency\n";
    Eigen::VectorXd current;
    for(unsigned i = 0; i < snapMap.size(); ++i){
        current = snapMap[i];

        bool equal = false;

        double tolerence = 1e-10;

        for(unsigned j = 0; j < snapMappingFreq.size(); ++j){

            double difference = 0;
            for(unsigned k = 0; k < current.size(); ++k){

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

    std::cout << "\n\nSorting by first Dimension\n";
    //std::vector< std::pair < Eigen::VectorXd , double> >
    std::vector< std::pair < Eigen::VectorXd , double> >  sortedMappingFreq ( 1 );
    sortedMappingFreq[0] = snapMappingFreq[0];

    std::pair< Eigen::VectorXd, double> currentS;
    for(int i =1 ; i < snapMappingFreq.size(); ++i){

        currentS = snapMappingFreq[i];

        for(int j = 0; j < sortedMappingFreq.size(); ++j){

            if( currentS.first[0] > sortedMappingFreq[j].first[0]){
           // if( currentS.second > sortedMappingFreq[j].second){

                //std::cout << currentS.second << " > " << sortedMappingFreq[j].second << std::endl;
                sortedMappingFreq.emplace(sortedMappingFreq.begin() + j , currentS );
                break;
            }
            else if( j == sortedMappingFreq.size() -1 ){
                sortedMappingFreq.push_back(currentS);
                break;
            }
        }

    }

    if(output){
        std::cout << "\n\nWriting to file\n";
        std::ofstream myFile;
        myFile.open("results.dat", std::ios::trunc);

        for(int i =0; i < sortedMappingFreq.size(); ++i){
            for(int j =0; j < sortedMappingFreq[i].first.size(); ++j){
            myFile<< sortedMappingFreq[i].first[j] <<" ";
            }
            myFile<<sortedMappingFreq[i].second << std::endl;

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
