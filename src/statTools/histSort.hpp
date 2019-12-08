/*
 * basicStat.hpp
 *
 *  Created on: 6 Dec 2019
 *      Author: arnaudv
 */


#ifndef STATTOOLS_HISTSORT_HPP
#define STATTOOLS_HISTSORT_HPP

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>


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



std::vector<double> findDeltaX (const Eigen::MatrixXd& allSamples, int nBins ){

    std::vector<double > deltaXs ( allSamples.cols() );

    Eigen::VectorXd valsMax ( allSamples.cols() );
    Eigen::VectorXd valsMin ( allSamples.cols() );

    for(unsigned i = 0; i < valsMax.size(); ++i){
        valsMax[i] = -1e9;
        valsMin[i] = +1e9;
    }

    for(unsigned     j = 0; j < allSamples.cols() ; ++j){
        for(unsigned i = 0; i < allSamples.rows() ; ++i){

            if( allSamples(i, j) > valsMax[j]) { valsMax[j] = allSamples(i, j) ;}
            if( allSamples(i, j) < valsMin[j]) { valsMin[j] = allSamples(i, j) ;}
        }
        }

    for(unsigned i = 0; i < deltaXs.size(); ++i ){

        deltaXs[i] =  abs(  (double) ((valsMax[i] - valsMin[i]) / nBins ) );

    }

    return deltaXs;
}


using HistContainer = std::vector< std::pair < Eigen::VectorXd , double> >;

HistContainer histBin(const Eigen::MatrixXd& allSamples,
             const std::vector<double>& deltaXs,
             bool normalized, bool output){

    HistContainer  snapMappingFreq;


    std::vector< Eigen::VectorXd > snapMap (allSamples.rows()) ;
    Eigen::VectorXd mapping ( allSamples.cols() );


    if(output) { std::cout << "\n\nSnaping\n"; }

    for(unsigned i = 0; i < allSamples.rows(); ++i){

        for(unsigned int j = 0; j < allSamples.cols() ; ++j){
            mapping[j] = snap(allSamples(i, j), deltaXs[j] );
            //std::cout<< mapping[j] <<"\t"<<allSamples(i, j)<<std::endl;
        }
        snapMap[i] = mapping;
    }

    if(output) { std::cout << "\n\nCounting Frequency\n";}
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

    if(output) { std::cout << "\n\nSorting by first Dimension\n";}

    //std::vector< std::pair < Eigen::VectorXd , double> >
    HistContainer  sortedMappingFreq ( 1 );
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



    return sortedMappingFreq ;
}

#endif
