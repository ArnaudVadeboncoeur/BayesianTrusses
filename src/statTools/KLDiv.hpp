/*
 * KLDiv.hpp
 *
 *  Created on: 6 Dec 2019
 *      Author: arnaudv
 */

#ifndef STATTOOLS_KLDIV_HPP_
#define STATTOOLS_KLDIV_HPP_

#include <vector>
#include <cmath>

#include <Eigen/Dense>

using HistContainer = std::vector< std::pair < Eigen::VectorXd , double> >;

double KLDiv(const HistContainer& trueHistPoints, const HistContainer& modelHistPoints ){

    double div = 0.;

    //distance of model to true dist
    for( unsigned i = 0; i < trueHistPoints.size() ; ++i){

        div += modelHistPoints[i].second * log ( modelHistPoints[i].second / trueHistPoints[i].second ) ;

    }


    return div;
}


#endif /* SRC_STATTOOLS_KLDIV_HPP_ */
