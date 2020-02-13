/*
 * PdfEval.hpp
 *
 *  Created on: 12 Feb 2020
 *      Author: arnaudv
 */

#ifndef APP_LAPLACEAPPROX_PDFEVAL_HPP_
#define APP_LAPLACEAPPROX_PDFEVAL_HPP_

#include <Eigen/Dense>

#include "../../src/FEMClass.hpp"
#include "ThreeDTruss.hpp"



template < unsigned DIM, typename Vec >
class PdfEval
{
public:

    PdfEval ( ) { };

    PdfEval ( double noiseLik, Eigen::MatrixXd trueSampleDisp, Eigen::VectorXi dispDof, Vec priorMean ) ;

    double Eval( Vec x);

    ~PdfEval () {};

private:

    Eigen::MatrixXd trueSampleDisp_;
    double noiseLikStd_;
    Eigen::VectorXi dispDof_ ;
    Vec priorMean_;

};

template <unsigned DIM, typename Vec >
PdfEval<DIM,Vec>::PdfEval( double noiseLikStd, Eigen::MatrixXd trueSampleDisp, Eigen::VectorXi dispDof, Vec priorMean ){

    noiseLikStd_ = noiseLikStd;
    trueSampleDisp_ = trueSampleDisp;
    dispDof_ = dispDof;
    priorMean_ = priorMean;
}


template < unsigned DIM, typename Vec >
double PdfEval<DIM, Vec>::Eval(Vec x ){

    for(int i = 0; i < DIM; ++i){
        if( x[i] <= 0 ){
            return -9e30;
        }}

    TupleTrussDef MTrussDef;
    MTrussDef = InitialTrussAssignment( );
    FEMClass MTrussFem( false, MTrussDef );

    for(int i =0; i < DIM; ++i){

        MTrussFem.modA( i, x[i] );
    }

    MTrussFem.assembleS( );
    MTrussFem.computeDisp( );

    Eigen::VectorXd K_thetaInvf(DIM);

    for(int i = 0; i < DIM; ++i){

        K_thetaInvf[i] = MTrussFem.getDisp( dispDof_[i] );

    }

    Eigen::VectorXd theta(DIM);
    for(int i =0; i < DIM; ++i){

        theta[i] = x[1];
    }

    Eigen::MatrixXd CovMatrixNoise (DIM,DIM);

    CovMatrixNoise.setIdentity();
    CovMatrixNoise = CovMatrixNoise * std::pow(noiseLikStd_, 2);

    double logLik = 0;

    //  p(y|Theta, Sigma)
    logLik += - (double) trueSampleDisp_.rows() / 2.0 * std::log( CovMatrixNoise.determinant() ) ;

    Eigen::VectorXd y_iVec (DIM);
    for(int i = 0; i < trueSampleDisp_.rows(); ++i){

        for(int j =0; j < DIM; ++j){
            y_iVec[j] = trueSampleDisp_( i, j );
        }

        logLik += - 1./2. * (y_iVec - K_thetaInvf).transpose() * CovMatrixNoise.inverse() * (y_iVec - K_thetaInvf)   ;

    }
    MTrussFem.FEMClassReset(false);

    if( std::isnan(logLik) ){ logLik = -9e30;}

    //Gaussian Prior - Conjugate Prior for Theta p( Theta_0 | Theta_0, sig^2 / k_0 )

    Eigen::VectorXd theta_0 ( DIM );
    for(int i =0; i < DIM; ++i){

        theta_0[i] = priorMean_[i];
    }

    double k_0  = 1e-4 ; // need k_0 to counter weight of prior

    logLik += - 1./2.* std::log( ( CovMatrixNoise / k_0).determinant() )
              - 1./2.* (theta - theta_0).transpose() * (CovMatrixNoise / k_0 ).inverse() * (theta - theta_0) ;

    return logLik;
    }

#endif /* APP_LAPLACEAPPROX_PDFEVAL_HPP_ */
