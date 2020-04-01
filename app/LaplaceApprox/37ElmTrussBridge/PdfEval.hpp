/*
 * PdfEval.hpp
 *
 *  Created on: 12 Feb 2020
 *      Author: arnaudv
 */

#ifndef APP_LAPLACEAPPROX_PDFEVAL_HPP_
#define APP_LAPLACEAPPROX_PDFEVAL_HPP_

#include <Eigen/Dense>

#include "../../../src/FEMClass.hpp"

#include "ThreeDTruss37Elm.hpp"
//#include "ThreeDTruss23Elm.hpp"
//#include "ThreeDTruss3Elm.hpp"



template < unsigned DimObs, unsigned DimPara, typename Vec >
class PdfEval
{
public:

    PdfEval ( ) { };

    PdfEval ( double noiseLik, Eigen::MatrixXd trueSampleDisp,  std::vector<int> paraIndex ,Eigen::VectorXi ObsIndex,
              Vec priorMean, const Eigen::MatrixXd& PriorCovMatrix ) ;

    double Eval( Vec x);

    ~PdfEval () {};

private:

    Eigen::MatrixXd trueSampleDisp_;
    double noiseLikStd_;
    Eigen::VectorXi ObsIndex_ ;
    Eigen::MatrixXd PriorCovMatrix_;
    std::vector<int> paraIndex_;
    Vec priorMean_;

};

template < unsigned DimObs, unsigned DimPara, typename Vec >
PdfEval < DimObs, DimPara, Vec>::PdfEval( double noiseLikStd, Eigen::MatrixXd trueSampleDisp, std::vector<int> paraIndex,
                                          Eigen::VectorXi ObsIndex, Vec priorMean,
                                          const Eigen::MatrixXd& PriorCovMatrix ){

    noiseLikStd_ = noiseLikStd;
    trueSampleDisp_ = trueSampleDisp;
    paraIndex_ = paraIndex;
    ObsIndex_ = ObsIndex;
    priorMean_ = priorMean;
    PriorCovMatrix_ = PriorCovMatrix;
}


template < unsigned DimObs, unsigned DimPara, typename Vec >
double PdfEval< DimObs, DimPara, Vec>::Eval(Vec x ){

//    for(int i = 0; i < DimPara; ++i){
//        if( x[i] <= 0 ){
//
//            return -9e30;
//        }}
    //std::cout << "Here pdfEval1" << std::endl;
    TupleTrussDef MTrussDef;
    MTrussDef = InitialTrussAssignment( );
    FEMClass MTrussFem( false, MTrussDef );

    for(int i =0; i < DimPara; ++i){

        MTrussFem.modA( paraIndex_[i], x[i] );
    }

    MTrussFem.assembleS( );
    MTrussFem.computeDisp( );

    Eigen::VectorXd K_thetaInvf(DimObs);
    std::vector<int> dofK = MTrussFem.getFreeDof() ;
    Eigen::MatrixXd L( DimObs , dofK.size() ); L.setZero();

        for(int i = 0; i < ObsIndex_.size(); ++i ){
            for( int j = 0; j < dofK.size(); ++j ){
                if(dofK[j] == ObsIndex_[i]){
                    L(i, j) = 1;
                    break;
                }
            }
        }

    K_thetaInvf = MTrussFem.getDisp();
    K_thetaInvf = L * K_thetaInvf;

    Eigen::MatrixXd CovMatrixNoise (DimObs,DimObs);

    CovMatrixNoise.setIdentity();
    CovMatrixNoise = CovMatrixNoise * std::pow(noiseLikStd_, 2);

    double logLik = 0;

    //  p(y|Theta, Sigma)
    logLik += - (double) trueSampleDisp_.rows() / 2.0 * std::log( CovMatrixNoise.determinant() ) ;

    //std::cout << trueSampleDisp_ << "\n\n";

    Eigen::VectorXd y_iVec (DimObs);
    for(int i = 0; i < trueSampleDisp_.rows(); ++i){

        for(int j =0; j < DimObs; ++j){
            y_iVec[j] = trueSampleDisp_( i, j );
        }

        //std::cout << "\n y_iVec\n" << y_iVec << std::endl;
        //std::cout << "\ K_thetaInvf\n" << K_thetaInvf << std::endl;

        logLik += - 1./2. * (y_iVec - K_thetaInvf).transpose() * CovMatrixNoise.inverse() * (y_iVec - K_thetaInvf)   ;


    }

    MTrussFem.FEMClassReset(false);

    if( std::isnan(logLik) ){ return -9e30;}


    //Gaussian Prior - Conjugate Prior for Theta p( Theta_0 | Theta_0, sig^2 / k_0 )


    Eigen::VectorXd theta(DimPara);
    for(int i =0; i < DimPara; ++i){
        theta[i] = x[i];
    }

    Eigen::VectorXd theta_0 ( DimPara );
    for(int i =0; i < DimPara; ++i){

        theta_0[i] = priorMean_[i];
    }


    logLik += - 1./2.* std::log( PriorCovMatrix_.determinant() )
              - 1./2.* (theta - theta_0).transpose() * PriorCovMatrix_.inverse() * (theta - theta_0) ;

    return logLik;
    }

#endif /* APP_LAPLACEAPPROX_PDFEVAL_HPP_ */
