/*
 * SteinDiscrepancy.hpp
 *
 *  Created on: 6 Dec 2019
 *      Author: arnaudv
 */

#ifndef STATTOOLS_STEINDISCREPANCY_HPP_
#define STATTOOLS_STEINDISCREPANCY_HPP_


#include <Eigen/Dense>
#include <cmath>


double steinDisc(Eigen::MatrixXd& trueSamples, Eigen::MatrixXd& ModelSamples){

    //Not proper, equation not well interpreted.

    double sdm = 0;

    for(unsigned i = 0; i<trueSamples.rows(); ++i){

        sdm += abs( trueSamples(i,0) - ModelSamples(i,0) );
    }

    return sdm / trueSamples.rows() ;
}




#endif /* STATTOOLS_STEINDISCREPANCY_HPP_ */
