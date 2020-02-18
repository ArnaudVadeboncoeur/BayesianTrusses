/*
 * KLDiv.hpp
 *
 *  Created on: 6 Dec 2019
 *      Author: arnaudv
 */

#ifndef STATTOOLS_KLDIV_HPP_
#define STATTOOLS_KLDIV_HPP_

#include <vector>
#include <iostream>
#include <cmath>

#include <Eigen/Dense>

using HistContainer = std::vector< std::pair < Eigen::VectorXd , double> >;

double L2Norm(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q){
    double l2 = 0;

    int PyCol = P.cols() - 1;
    int QyCol = Q.cols() - 1;

        for(int i =0; i< P.rows(); ++i){

            l2 += std::pow( P(i, PyCol) - Q(i, QyCol), 2 ) ;
        }
    l2 = std::sqrt(l2);
    return l2;
}


double KLDiv(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q){
    double div = 0;
    if(P.rows() != Q.rows()){std::cout << "P and Q not same size" << std::endl; return -9999.9; }

    int PyCol = P.cols()- 1;
    int QyCol = Q.cols()- 1;


        for(int i =0; i< P.rows(); ++i){

            if(P(i, PyCol) != 0. && Q(i, QyCol) != 0. ){
            //std::cout << P(i, PyCol) << " " <<  Q(i, QyCol ) << std::endl;

            div += P(i, PyCol) * std::log ( P(i, PyCol) / Q(i, QyCol ) );
        }}
    return div;
}



double KLDiv(const HistContainer& trueHistPoints, const HistContainer& modelHistPoints ){

    double div = 0.;

    //distance of model to true dist
    for( unsigned i = 0; i < trueHistPoints.size() ; ++i){

        div += modelHistPoints[i].second * log ( modelHistPoints[i].second / trueHistPoints[i].second ) ;

    }


    return div;
}


#endif /* SRC_STATTOOLS_KLDIV_HPP_ */
