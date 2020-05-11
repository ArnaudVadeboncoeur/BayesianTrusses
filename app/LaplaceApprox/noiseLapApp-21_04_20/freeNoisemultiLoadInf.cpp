/*
 * BTrussMCMC_MH.cpp
 *
 *  Created on: 12 Dec 2019
 *      Author: arnaudv
 */

#include "../../../src/FEMClass.hpp"
#include "../../../src/statTools/KLDiv.hpp"

#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>


#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <math.h>

//#include "PdfEval.hpp"
#include "sampleGen.hpp"
#include "ThreeDTruss37Elm.hpp"

int main(){

    DataCont trueSamplesTupleContainer = trueSampleGen();

    std::vector <Eigen::MatrixXd> trueSampleDispC = std::get<0>( trueSamplesTupleContainer );
    std::vector <Eigen::MatrixXd> trueForcingC    = std::get<1>( trueSamplesTupleContainer );

    int num = trueSampleDispC.size();
    for(int i = num - 1; i >= 0; --i){
        if(trueSampleDispC[i].rows() == 0){
            trueSampleDispC.erase(trueSampleDispC.begin()+i);
            trueForcingC.erase(trueForcingC.begin()+i);
        }

    }
    std::cout << trueSampleDispC.size() << std::endl;

    int numForcing = trueSampleDispC.size();

    constexpr unsigned DimK       =  30 ;
    constexpr unsigned DimObs     =  20;//20 ;//1 node 3->x,y,z
    constexpr unsigned DimPara    =  10 ;

    constexpr unsigned NumTotPara =  37;

    std::vector<int> paraIndex     { 12, 13,14, 15, 16, 17,18,19,20,21  };

    bool plot_1_dim = false;
    //If number larger then DimPara, Noise Parameters will be ploted
    std::vector<int> plotParaIndex {1,4};



    //Eigen::VectorXi ObsIndex( 4 ); ObsIndex << 6, 7, 12, 13;

    Eigen::VectorXi nodesObs( 10 ); nodesObs <<   1, 2,3,4, 5, 8, 9,10,  11, 12;    Eigen::VectorXi ObsIndex( nodesObs.size() * 2 );
            for(int j = 0; j < nodesObs.size(); ++j){

                ObsIndex[ j*2 + 0] = nodesObs[j]*3 + 0;
                ObsIndex[ j*2 + 1] = nodesObs[j]*3 + 1;
            }


//    double mu = 4 * std::pow( noisePmean.mean(), 2) /  pow(noisePstd.mean(), 2) + DimObs + 3 ;
//    double mu = DimObs + 4;
//    std::cout << "mu = " << mu << std::endl;
//    for(int i = 0 ; i < DimObs; ++i){
//
//        psi(i,i) = noisePmean[i] * (mu - DimObs - 1.);
//        psi(i,i) = noisePstd[i];
//        std::cout << "Mode " << i << " " << psi(i,i) / (mu + DimObs + 1.);
//        std::cout << "\tStd " << i << " " << std::sqrt( 4 * std::pow( noisePmean.mean(), 2) / (mu - DimObs - 3.) )<< std::endl;
//
//    }


//-------------------init prior information TO BE REMOVED----------------------------

    //find empirical std in data -- to do for every loading case

    std::vector <Eigen::MatrixXd> CovMatrixNoiseInvC(numForcing);
    Eigen::VectorXd empMean( trueSampleDispC[0].cols() );
    Eigen::VectorXd empStd( trueSampleDispC[0].cols() );

    for(int f = 0 ; f < numForcing; ++f){


    empMean.setZero();

    for(int i = 0 ; i < trueSampleDispC[f].rows(); ++i){
        for(int j = 0 ; j < trueSampleDispC[f].cols(); ++j){

            empMean[j]+= trueSampleDispC[f](i, j);
        }
    }
    for(int i = 0 ; i < empMean.size(); ++i){
        empMean[i] = empMean[i] / trueSampleDispC[f].rows();
    }


    empStd.setZero();
    //most reliable way of computing std it seems..
    double propHalfMax = 0.1;
    for(int i = 0 ; i < empStd.size(); ++i){
            empStd[i] = ( std::abs( empMean[i] ) * propHalfMax ) / 2.355;
        }

    Eigen::MatrixXd empStdCovMatrix (DimObs,DimObs); empStdCovMatrix.setZero();
    for(int i = 0; i < DimObs; ++i){
       empStdCovMatrix(i, i) = std::pow(empStd[i], 2) ;//* 0.1;
       //10% std of mean disp values.
       //empStdCovMatrix(i, i) = std::pow( 0.01 * trueSampleDispC[f].mean(), 2) ;
    }

        CovMatrixNoiseInvC[f] = empStdCovMatrix.inverse();

    }


    std::cout << " empMean\n" << empMean << std::endl;
    //Eigen::MatrixXd EmpNoiseFixed(DimObs, DimObs);

    Eigen::MatrixXd CovNoiseAllF(DimObs, DimObs);CovNoiseAllF.setZero();
    CovNoiseAllF = CovNoiseAllF.setIdentity() * pow( 0.1, 2);

    //CovNoiseAllF = CovMatrixNoiseInvC[0];


    //-------------------Prior over Noise parameters ----------------------------

        // prior for inverse Wishart distribution
        Eigen::VectorXd noisePstd( DimObs ); noisePstd.setConstant( std::pow( 1e-3, 2) );
        Eigen::MatrixXd noisePmean( DimObs, 1 ); noisePmean.setConstant( std::pow( 1e-5, 2) );
//        noisePmean = CovMatrixNoiseInvC[0].inverse().diagonal();
//        noisePstd = noisePmean ;

        noisePmean = empStd.cwiseProduct(empStd);

        noisePstd = empStd.cwiseProduct(empStd);;

        Eigen::MatrixXd psi(DimObs, DimObs ); psi.setZero();
        psi = noisePstd.asDiagonal();



    //----------------- Prior over FEM model parameters--------------------------
    Eigen::VectorXd priorMeans(DimPara); priorMeans.setConstant(0.060);

    Eigen::MatrixXd PriorCovMatrix (DimPara,DimPara); PriorCovMatrix.setZero();
    Eigen::VectorXd priorStdVec(DimPara); priorStdVec.setConstant(0.1);
    for(int i = 0; i < priorStdVec.size(); ++i){
        PriorCovMatrix(i, i) = pow(priorStdVec[i], 2) ;//* 0.1;
    }

    Eigen::MatrixXd PriorCovMatrixInv = PriorCovMatrix.inverse();



    //------------------------------------------------------------------------------

    //init FEM model
    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();
    FEMClass TrussFem(false, TrussDef );
    TrussFem.assembleS();
    std::vector<int> dofK = TrussFem.getFreeDof();
    TrussFem.FEMClassReset(false);

//--------------------------------------------------------------------------------------

   //Lambda function to compute u(Theta)
    std::function < Eigen::VectorXd ( const Eigen::VectorXd, int) > uTheta;
    uTheta = [ &TrussFem, paraIndex, trueForcingC ](const Eigen::VectorXd& X, int forcingIndex ){
        TrussFem.FEMClassReset(false);

        for(int j = 0; j < paraIndex.size(); ++j){
            TrussFem.modA(paraIndex[j], X[j] );
        }

        TrussFem.modForce( trueForcingC[forcingIndex] );

        TrussFem.assembleS();

        TrussFem.computeDisp();
        Eigen::VectorXd u = TrussFem.getDisp();
        TrussFem.FEMClassReset(false);

        return u;
    };

   //Lambda function to compute K(Theta)
    std::function < Eigen::MatrixXd (Eigen::VectorXd) > KThetaFunc;
    KThetaFunc = [ &TrussFem, paraIndex ]( Eigen::VectorXd X ){
        Eigen::MatrixXd K;
        //produce k(theta)
        for(int j = 0; j < paraIndex.size(); ++j){
            TrussFem.modA(paraIndex[j], X[j]);
        }
        TrussFem.assembleS();
        K = TrussFem.getK();
        TrussFem.FEMClassReset(false);

        return K;
    };

    //Lambda function to compute dK/dTheta_i
    std::function < Eigen::MatrixXd (Eigen::VectorXd, int) > dKdTheta_iFunc;
    dKdTheta_iFunc = [ &TrussFem, NumTotPara, paraIndex ]( Eigen::VectorXd X, int index ){
        TrussFem.FEMClassReset(false);
        Eigen::MatrixXd dKdtheta_i;
        //produce  dKdTheta_i
        for(int j = 0; j < NumTotPara; ++j){
            TrussFem.modA(j, 0 );
        }
        TrussFem.modA(paraIndex[index], 1 );
        TrussFem.assembleS( );
        dKdtheta_i = TrussFem.getK();
        TrussFem.FEMClassReset(false);

        return dKdtheta_i;
    };

   //Labmda function to compute dudtheta_i
   std::function < Eigen::VectorXd      ( Eigen::VectorXd, Eigen::MatrixXd,
                                          Eigen::MatrixXd, Eigen::VectorXd ) > dudtheta_iFunc;
   dudtheta_iFunc = [ ](   const Eigen::VectorXd& X,          const Eigen::MatrixXd& K_inv,
                           const Eigen::MatrixXd& dKdtheta_b, const Eigen::VectorXd& u ){

        Eigen::VectorXd dudtheta_b( X.rows() );
        dudtheta_b = - K_inv * dKdtheta_b * u ;

        return dudtheta_b;
    };

//Labmda function to compute dudTheta
   std::function < Eigen::MatrixXd    ( const Eigen::VectorXd,
                                         const Eigen::MatrixXd, const Eigen::VectorXd ) > dudThetaFunc;

   dudThetaFunc = [ &dudtheta_iFunc, &dKdTheta_iFunc ]( const Eigen::VectorXd& X, const Eigen::MatrixXd& K_inv,
                                                        const Eigen::VectorXd& u){

       Eigen::MatrixXd dudTheta( K_inv.rows(),  X.rows() );

       Eigen::VectorXd dudtheta_i( X.rows() );

       for(int i = 0; i < X.rows(); ++i ){

           dudtheta_i = dudtheta_iFunc( X, K_inv, dKdTheta_iFunc( X, i), u );
           for(int j = 0; j < u.rows(); ++j ){
               dudTheta(j, i) = dudtheta_i(j);
               }
           }

       return dudTheta;
   };


   //Labmda function to compute du2_dthetab_Theta
	std::function < Eigen::MatrixXd    ( const Eigen::VectorXd,
									     const Eigen::MatrixXd, const Eigen::VectorXd, int, int) > du2_dthetai_dthetajFunc;

	du2_dthetai_dthetajFunc = [ &dudtheta_iFunc, &dKdTheta_iFunc ]( const Eigen::VectorXd& X, const Eigen::MatrixXd& K_inv,
																     const Eigen::VectorXd& u, int index_i, int index_j ){
	 Eigen::MatrixXd du2_dthetai_dthetaj( X.rows(), 1 );

	 Eigen::MatrixXd dudtheta_i(K_inv.rows(), 1 );
	 Eigen::MatrixXd dudtheta_j(K_inv.rows(), 1 );

	 Eigen::MatrixXd dK_dtheta_i(K_inv.rows(), K_inv.rows());
	 Eigen::MatrixXd dK_dtheta_j(K_inv.rows(), K_inv.rows());

	 dK_dtheta_i = dKdTheta_iFunc( X, index_i);
	 dK_dtheta_j = dKdTheta_iFunc( X, index_j);

	 dudtheta_i = dudtheta_iFunc( X, K_inv, dK_dtheta_i, u );
	 dudtheta_j = dudtheta_iFunc( X, K_inv, dK_dtheta_j, u );

	 du2_dthetai_dthetaj = - K_inv * ( dK_dtheta_i * dudtheta_j + dK_dtheta_j * dudtheta_i );

	 return du2_dthetai_dthetaj;
	};



	//Labmda function to compute du2_dthetab_Theta
	std::function < Eigen::MatrixXd    ( const Eigen::VectorXd,
										 const Eigen::MatrixXd, const Eigen::VectorXd, int) > du2_dthetab_ThetaFunc;

	du2_dthetab_ThetaFunc = [ &du2_dthetai_dthetajFunc ]( const Eigen::VectorXd& X, const Eigen::MatrixXd& K_inv,
																         const Eigen::VectorXd& u, int index_b ){

	  Eigen::MatrixXd du2_dthetab_Theta( K_inv.rows(),  X.rows() );
	  Eigen::MatrixXd du2_dthetai_dthetaj(  K_inv.rows(), 1 );

	  for(int i = 0; i < X.rows(); ++i ){

		  du2_dthetai_dthetaj = du2_dthetai_dthetajFunc( X, K_inv, u, index_b, i );
		  for(int j = 0; j < u.rows(); ++j ){
			  du2_dthetab_Theta(j, i) = du2_dthetai_dthetaj(j, 0);

			  }
		  }

	  return   du2_dthetab_Theta;
	};

    //Labmda function to compute Bii
    std::function < Eigen::MatrixXd ( const Eigen::MatrixXd&, int) > BiFunc;

    BiFunc = [DimObs] (const Eigen::MatrixXd& C_ufInv, int index_i){

        Eigen::MatrixXd Bi(DimObs,DimObs);
        Eigen::MatrixXd Jii(DimObs,DimObs); Jii.setZero();
        Jii(index_i, index_i) = 1;

        Bi = -1. * C_ufInv * Jii * C_ufInv;

        return Bi;
    };

    //Labmda function to compute Dji
        std::function < Eigen::MatrixXd ( const Eigen::MatrixXd& , int, int) > DjiFunc;

        DjiFunc = [DimObs, &BiFunc] (const Eigen::MatrixXd& C_ufInv, int index_j, int index_i){

            Eigen::MatrixXd Dji(DimObs,DimObs);
            Eigen::MatrixXd Bj = BiFunc(C_ufInv, index_j);
            Eigen::MatrixXd Jii(DimObs,DimObs); Jii.setZero();
            Jii(index_i, index_i) = 1;

            Dji = -1. * ( Bj * Jii * C_ufInv  + C_ufInv * Jii* Bj );

            return Dji;
        };

//Newton Ralphson to find MAP--------------------------------------------


    //Eigen::VectorXd X(DimPara); X.setConstant(0.07);
    //Eigen::VectorXd XPast(DimPara); XPast.setConstant(0.07);
    double Null = 1e-14 ;
    Eigen::MatrixXd k(DimK, DimK);
    Eigen::MatrixXd k_inv(DimK, DimK);
    Eigen::MatrixXd dk_dtheta(DimPara, DimPara);
    Eigen::MatrixXd y_i(DimObs, 1);
    Eigen::MatrixXd u  ( DimObs,  1 );
    Eigen::MatrixXd uPast  ( DimObs,  1 );
    Eigen::MatrixXd u_n( DimK  ,  1 );
    Eigen::MatrixXd du_dTheta( DimObs, DimPara );

    Eigen::MatrixXd du2_dthetab_dTheta( DimObs, DimPara  );
    Eigen::MatrixXd thetaHat_b ( DimPara, 1);
    Eigen::MatrixXd du_dtheta_b ( DimObs, 1  );

    Eigen::MatrixXd gradTheta(1, DimPara);      gradTheta.setZero();
    Eigen::MatrixXd gradNoise(1, DimObs);       gradNoise.setZero();
    Eigen::MatrixXd grad(1, DimPara + DimObs);  grad.setZero();

    Eigen::MatrixXd WCov_inv  (DimObs, DimObs );
    Eigen::VectorXd W         ( DimPara + DimObs );

    W.block(0,0,DimPara,1)           = priorMeans.block(0,0,DimPara,1);
    W.block(DimPara ,0,DimObs,1)     = noisePmean.block(0,0,DimObs,1) * 10;

    std::cout << "W" << W << std::endl;
    std::cout << "W.tail(DimObs).cwiseSqrt().transpose();\n" << W.tail(DimObs).cwiseSqrt().transpose() << std::endl;

    Eigen::VectorXd WPast = W;

    Eigen::MatrixXd Bi  (DimObs, DimObs);

    std::cout <<"(KThetaFunc ( X )).rows() " << (KThetaFunc (  W.head(DimPara) )).rows() << std::endl;
    std::cout <<"(KThetaFunc (  W.head(DimPara) )).cols() " << (KThetaFunc (  W.head(DimPara) )).cols() << std::endl;

    Eigen::MatrixXd L( DimObs , DimK ); L.setZero();

    for(int i = 0; i < ObsIndex.size(); ++i ){
        for( int j = 0; j < dofK.size(); ++j ){
            if( dofK[j] == ObsIndex[i] ){
                L(i, j) = 1;
                break;
            }
        }
    }
    std::cout << "Done Creating L" << std::endl;
    std::cout << L << std::endl;
    std::cout << "L.rows() " << L.rows() << std::endl;
    std::cout << "L.cols() " << L.cols() << std::endl;

    std::ofstream optFile;
    optFile.open("Opt.dat");
    for(int d = 0; d < plotParaIndex.size(); ++d){
               optFile << W[ plotParaIndex[d] ] << " ";
           } optFile << "\n";

   std::ofstream FullDimOpt;
   FullDimOpt.open("FullDimOpt.dat");
   for(int d = 0; d < W.size(); ++d){
       FullDimOpt << W[d] << " ";
          } FullDimOpt << "\n";


    //N-R iterations
    int maxIter = 5e3;
    double stepTheta = 0.03;
    double stepNoise = std::pow( 3e-4, 2);

    for(int i = 0; i < maxIter; ++i){

        k           = KThetaFunc (  W.head(DimPara) );
        k_inv       = k.inverse();


        gradNoise.setZero();
        gradTheta.setZero();
        grad.setZero();

        //prior grad contribution
        gradNoise = - (  W.tail(DimObs) - noisePmean  ).transpose() * psi.inverse();
        gradTheta = - (  W.head(DimPara) - priorMeans ).transpose() * PriorCovMatrixInv;



        WCov_inv = W.tail(DimObs).asDiagonal().inverse();

        //---Compute gradient---
        for(int f = 0 ; f < numForcing; ++f){
            //---compute Theta grad ---
            u_n         = uTheta( W.head(DimPara), f) ;
            u           = L * u_n;

            du_dTheta   = L * dudThetaFunc( W.head(DimPara), k_inv, u_n );

            for(int j = 0; j < trueSampleDispC[f].rows(); ++j){

                y_i.block(0,0,DimObs,1 ) = trueSampleDispC[f].block(j,0,1, DimObs ).transpose();

                gradTheta -= (y_i - u).transpose() * WCov_inv * -1. * du_dTheta ;
            }

            grad.block(0,0, 1, DimPara) += gradTheta.block(0,0, 1, DimPara);

            //---compute Noise grad ---
            for(int n = 0; n < DimObs; n++){

                gradNoise(0, n) -= trueSampleDispC[f].rows() / 2. *  WCov_inv(n,n) ;

                Bi = BiFunc(WCov_inv , n);

                for(int j = 0; j < trueSampleDispC[f].rows(); ++j){

                    y_i.block(0,0,DimObs,1 ) = trueSampleDispC[f].block(j,0,1, DimObs ).transpose();

                    gradNoise(0, n) -= (1./2. * (y_i - u).transpose() * Bi * (y_i - u))(0,0) ;
                }
                grad(0,DimPara + n ) += gradNoise(0, n);
            }

        }

//        bool gradNull; gradNull = true;
//        for(int j = 0; j < grad.size(); ++j){
//            if( std::abs( grad(0, j) ) > Null ){ gradNull = false; break; }
//        }
//        if( gradNull ){
//            std::cout << "gradNull\n" << grad << std::endl;
//            grad.setConstant(1e-10);
//            break ;
//        }

        if(i % 2 == 0){
            if( ( W.head(DimPara) - WPast.head(DimPara)).norm() < 0.8 * stepTheta ){
                stepTheta = stepTheta * 0.8 ;
            }
            if( (W.tail(DimObs) - WPast.tail(DimObs)).norm() < 0.8 * stepNoise ){
                stepNoise = stepNoise * 0.8 ;
                        }
            WPast = W;
        }

       W.head(DimPara) =  W.head(DimPara) + stepTheta *  1. / gradTheta.norm() *  gradTheta.transpose() ;

       W.tail(DimObs) = W.tail(DimObs) + stepNoise * 1. / gradNoise.norm() * gradNoise.transpose() ;

       for(int j = 0; j <  DimPara; ++j){
           if( W.head(DimPara)[j] < 0.){
               W.head(DimPara) = WPast.head(DimPara);
               stepTheta = stepTheta * 0.9;
               std::cout << "back step \n";
               break;
           }
       }
       for(int j = 0; j <  DimObs; ++j){
           if( W.tail(DimObs)[j] < 0.){
               W.tail(DimObs) = WPast.tail(DimObs);
               stepNoise = stepNoise * 0.9 ;
               std::cout << "back step \n";
               break;
           }
       }

       for(int d = 0; d < plotParaIndex.size(); ++d){
           optFile << W[ plotParaIndex[d] ] << " ";
       } optFile << "\n";

       for(int d = 0; d < W.size(); ++d){
          FullDimOpt << W[d] << " ";
       } FullDimOpt << "\n";

    }


    std::cout << " W.head(DimPara) \n" <<  W.head(DimPara) << "\n\n";
    std::cout << "gradTheta \n" << gradTheta << "\n\n";

    std::cout << "CovNoiseAllFDiagMAP \n" << W.tail(DimObs) << "\n\n";
    std::cout << "gradNoise \n" << gradNoise << "\n\n";

    std::cout << "Done Optimisaiton" << std::endl;

    optFile.close();
    FullDimOpt.close();

    //return 0;

//-----------------Compute Hessian at MAP-------------------

    //std::cout << "psi\n" << psi << std::endl;

    Eigen::MatrixXd hess(DimPara + DimObs, DimPara + DimObs);
    Eigen::MatrixXd LaplaceHess_inv(DimPara + DimObs, DimPara + DimObs);
    hess.setZero();

    Eigen::MatrixXd hess_b( DimPara, 1 );
    hess_b.setZero();

    Eigen::MatrixXd hessQuad1(DimPara, DimPara);


    for(int i = 0; i < paraIndex.size() ; ++i){

        thetaHat_b.setZero();
        thetaHat_b( i, 0 ) = 1;
        hess_b = -1 * thetaHat_b.transpose( ) * PriorCovMatrixInv;

        for(int f = 0 ; f < numForcing; ++f){

            u_n = uTheta( W.head(DimPara), f) ;
            u   = L * u_n;
            du_dTheta   = L * dudThetaFunc( W.head(DimPara), k_inv, u_n );
            du_dtheta_b = L * dudtheta_iFunc(  W.head(DimPara), k_inv, dKdTheta_iFunc ( W.head(DimPara),  i ),  u_n )  ;
            du2_dthetab_dTheta = L * du2_dthetab_ThetaFunc(  W.head(DimPara),  k_inv, u_n , i );

            for(int j = 0; j < trueSampleDispC[f].rows(); ++j){

                y_i.block(0,0,DimObs,1 ) = trueSampleDispC[f].block(j,0,1, DimObs ).transpose();

                hess_b -= (y_i - u).transpose() * CovMatrixNoiseInvC[f] * -1 * du2_dthetab_dTheta;
                hess_b -= du_dtheta_b.transpose() * CovMatrixNoiseInvC[f] * du_dTheta;
            }
        }
        for(int d = 0; d < DimPara; ++d){
            hessQuad1( i, d) = hess_b( 0, d );
        }
    }

    //std::cout << "hessQuad1\n" << hessQuad1 << std::endl;

    //compute quadrant 2 of Hessian
    Eigen::MatrixXd hessQuad2(DimPara, DimObs);
    //vary i of sigma_i
    Eigen::MatrixXd dSig_i_dTheta (1, DimPara);
    for(int i = 0; i < DimObs; ++i){
        WCov_inv = W.tail(DimObs).asDiagonal().inverse();
        Bi = BiFunc(WCov_inv , i);
        dSig_i_dTheta.setZero();

        for(int f = 0 ; f < numForcing; ++f){

            u_n         = uTheta( W.head(DimPara), f) ;
            u           = L * u_n;
            du_dTheta   = L * dudThetaFunc( W.head(DimPara), k_inv, u_n );

            for(int j = 0; j < trueSampleDispC[f].rows(); ++j){

                //std::cout << "Here A" << std::endl;
                //std::cout << "trueSampleDispC[f]\n" << trueSampleDispC[f] << std::endl;
                //std::cout << "trueSampleDispC[f].block(j,0,j, DimObs - 1 );\n" << trueSampleDispC[f].block(j, 0 , 1, DimObs) << std::endl;

                y_i.block(0,0,DimObs,1 ) = trueSampleDispC[f].block(j,0,1, DimObs ).transpose();

                //std::cout << "Here B" << std::endl;

                dSig_i_dTheta -=   ((y_i - u).transpose() * Bi * -1. * du_dTheta) ;
            }
        }
        for(int jj = 0; jj < DimPara; ++jj){

            hessQuad2(jj, i) = dSig_i_dTheta(0, jj);
        }
    }
   // std::cout << "hessQuad2\n" << hessQuad2 << std::endl;

   //Compute upper triangle of quadrant3 of hessian;
    Eigen::MatrixXd hessQuad3 (DimObs, DimObs);hessQuad3.setZero();
    Eigen::MatrixXd Dji (DimObs, DimObs);

    Eigen::MatrixXd sigPrior(1, DimObs);
    Eigen::VectorXd sigHat_i(DimObs); sigHat_i.setZero();

    for(int i = 0; i < DimObs; ++i){
        sigHat_i.setZero();
        sigHat_i( i, 0 ) = 1;
        sigPrior = sigHat_i.transpose() * psi.inverse();
        //std::cout << "sigPrior\n" << sigPrior << std::endl;

        for(int j = 0; j < DimObs; ++j){

            Dji = DjiFunc( WCov_inv, j, i );
            //std::cout << "Dji\n" << Dji << std::endl;

            for(int f = 0 ; f < numForcing; ++f){

                u_n         = uTheta( W.head(DimPara), f) ;
                u           = L * u_n;

                for(int k = 0; k < trueSampleDispC[f].rows(); ++k){

                    y_i.block(0,0,DimObs,1 ) = trueSampleDispC[f].block(k,0,1, DimObs ).transpose();

                    //std::cout << "(1./2. * (y_i - u).transpose() * Dji * (y_i - u))(0,0)\n"
                    //         << (1./2. * (y_i - u).transpose() * Dji * (y_i - u))(0,0) << std::endl;


                    hessQuad3(i, j) -= (1./2. * (y_i - u).transpose() * Dji * (y_i - u))(0,0) ;
                }

            }
            hessQuad3(i, j) -= sigPrior(0, j);
        }
    }
    //std::cout << "hessQuad3\n" << hessQuad3 << std::endl;


    for(int i = 0; i < DimPara + DimObs; ++i){
        for(int j = 0; j < DimPara + DimObs; ++j){

            if(i < DimPara && j < DimPara){

                hess(i, j) = hessQuad1(i, j);
            }
            if(i < DimPara && j >= DimPara){

                hess(i, j) = hessQuad2( i, j - DimPara );
            }
            if(i >= DimPara && j >= DimPara){

                hess(i, j) = hessQuad3( i - DimPara , j - DimPara  );
            }
            if(i >= DimPara && j < DimPara){

                hess(i, j) = hessQuad2.transpose()(i - DimPara, j);

            }


        }
    }



    Eigen::MatrixXd negLogHess = -1 * hess;


    std::cout << "hess \n" << hess << "\n\n";
    std::cout << "LaplaceHess_inv \n" << -1 * hess.inverse() << "\n\n";

    Eigen::VectorXd LaplaceMAP = W;


//Eval True Pdf to plot ---------------------------------------------------------

    std::cout << "Computing scatter points true pdf \n-----------------------------------------------" << '\n';
    //return 0;

    Eigen::VectorXd WPost = W;

    int dimEval = 2;
    double LikVals;
    double maxVal = -9e30;
    Eigen::VectorXd maxXPost( DimPara + DimObs); maxXPost.setConstant(0);
    double Vol = 0;

    std::ofstream myFile;
    myFile.open("pdfResults.dat");
    double a,b,c,d = 0;
    if(plotParaIndex[0] < DimPara){
        a = 0.00;//-0.08;
        b = 0.1;}
    else{
        a = 0;//-0.08;
        b = 1e-6;
    }

    if(plotParaIndex[1] < DimPara){
        c = 0.00;
        d = 0.1;}
    else{
        c = 0;
        d = 1e-6;
    }

    int samplesX = 1 * 1e2;
    if (plot_1_dim) {samplesX = 1;}
    int samplesY = 1 * 1e2;


    double dx = (double) (b - a) / samplesX;
    if (plot_1_dim) {dx = 1;}
    double dy = (double) (d - c) / samplesY;

    a = a - dx*0.99;
    c = c - dy*0.99;

    double bottomLim = 1e-2;

   Eigen::MatrixXd evaluations ( samplesX * samplesY , dimEval + 1);

   Eigen::MatrixXd Sigma = psi;

   Eigen::MatrixXd CovMatrixNoise (DimObs, DimObs); CovMatrixNoise.setZero();

   unsigned ctr = 0;

   double Nan = 0; double NotNan = 0;

   for(int i = 0; i < samplesX; ++i){

       WPost[ plotParaIndex[0] ] = a + (double) dx * ( i + 1) ;

       for(int j = 0; j < samplesY; ++j){

           WPost[ plotParaIndex[1] ] = c + (double) dy * ( j + 1) ;

           //--------Eval True Post-----------


           //LikVals = pow( Sigma.determinant(), -(mu + Sigma.rows()+ 1 )/2.) * std::exp( -1./2. * ( psi * Sigma.inverse() ).trace()  );

           //Gaussian prior over noise parameters
           LikVals = - 1./2.* std::log( psi.determinant() )
                     - 1./2.* (WPost.tail(DimObs) - noisePmean).transpose() * psi.inverse() * (WPost.tail(DimObs) - noisePmean) ;
//           std::cout << "WPost.tail(DimObs)\n" << WPost.tail(DimObs) << std::endl;
//           std::cout << "noisePmean\n" << noisePmean << std::endl;
//           std::cout << "psi.diagonal()\n" << psi.diagonal() << std::endl;


           //Gaussian prior over FEM parameters
           LikVals += - 1./2.* std::log( PriorCovMatrix.determinant() )
                     - 1./2.* (WPost.head(DimPara) - priorMeans).transpose() * PriorCovMatrixInv * (WPost.head(DimPara) - priorMeans) ;

           if( std::isnan(LikVals) ){ std::cout << "is nan 1 \n";}

           CovMatrixNoise = WPost.tail(DimObs).asDiagonal();

           for(int f = 0 ; f < numForcing; ++f){

               u_n         = uTheta( WPost.head(DimPara), f ) ;

               u           = L * u_n;


               LikVals += - (double) trueSampleDispC[f].rows() / 2.0 * std::log( CovMatrixNoise.determinant() ) ;
               //LikVals += - (double) trueSampleDispC[f].rows() / 2.0 * std::log( CovMatrixNoiseInvC[f].inverse().determinant() ) ;

               if( std::isnan(LikVals) ){ std::cout << "is nan 2 \n";}

               for(int l = 0; l < trueSampleDispC[f].rows(); ++l){

                   y_i.block(0,0,DimObs,1 ) = trueSampleDispC[f].block(l,0,1, DimObs ).transpose();

                   LikVals += - ( 1./2. * (y_i - u).transpose() * CovMatrixNoise.inverse() * ( y_i - u ))(0, 0)   ;
                   //LikVals += - ( 1./2. * (y_i - u).transpose() * CovMatrixNoiseInvC[f] * ( y_i - u ))(0, 0)   ;
                   if( std::isnan(LikVals) ){
                       Nan +=1;

                       }
                   else{
                       NotNan+=1;

                   }

               }
           }
           if( std::isnan(LikVals) ){
               LikVals = -9e30;
               std::cout << "is nan 4 \n";}
           //--------Eval True Post-----------
           if(LikVals > maxVal && !std::isnan(LikVals) ){
               maxVal = LikVals;
               maxXPost   = WPost;
           }

           evaluations(ctr, 0) = WPost[ plotParaIndex[0] ];
           evaluations(ctr, 1) = WPost[ plotParaIndex[1] ];
           evaluations(ctr, 2) = LikVals;

           ctr++;
       }
   }
   std::cout << "IS NotNan " << NotNan << std::endl;
   std::cout << "IS NAN " << Nan << std::endl;


   for(int i = 0; i < evaluations.rows(); ++i){
       evaluations(i, dimEval ) = std::exp(evaluations(i, dimEval ) - maxVal) ;
       Vol += evaluations(i, dimEval) * dx * dy;
   }

   for(int i = 0; i < evaluations.rows(); ++i){
       evaluations(i, dimEval) = evaluations(i, dimEval) / Vol;
       if( evaluations(i, dimEval) < bottomLim ){
           evaluations(i, dimEval) = bottomLim;
       }
           myFile << evaluations(i, 0) << " " << evaluations(i, 1) << " " << evaluations(i, 2) << std::endl;

   }


   std::cout << "maxVal = " << maxVal <<"\n"<< " maxX = \n" << maxXPost << " " << std::endl;

   std::cout << "Generated true pdf points" << std::endl;

//   return 0;
//Eval Laplace Approx --------------------------------------

   Eigen::VectorXd xGauss = W;
   Eigen::MatrixXd EvalsLaplApp ( samplesX * samplesY , dimEval + 1);
   std::ofstream myFile3;
   myFile3.open("pdfLaplaceEval.dat");
   double probDensVal;
   std::cout << "-------------------" << '\n';

   int ctr2 = 0;

   for(int i = 0; i < samplesX; ++i){

       xGauss[ plotParaIndex[0] ] = a + (double) dx * ( i + 1) ;

       for(int j = 0; j < samplesY; ++j){


           xGauss[ plotParaIndex[1] ] = c + (double) dy * ( j + 1) ;

           probDensVal = 1. / ( std::sqrt( pow(2 * M_PI, 2) * negLogHess.inverse().determinant() )   ) *
                                     std::exp( - 1./2. * (xGauss - LaplaceMAP).transpose() * negLogHess * (xGauss - LaplaceMAP) );

           EvalsLaplApp(ctr2, 0) = xGauss[ plotParaIndex[0] ];
           EvalsLaplApp(ctr2, 1) = xGauss[ plotParaIndex[1] ];
           EvalsLaplApp(ctr2, 2) = probDensVal;

           if( probDensVal < bottomLim ){
               probDensVal = bottomLim;
           }
               myFile3 << xGauss[ plotParaIndex[0] ] << " " << xGauss[ plotParaIndex[1] ] << " " << probDensVal << std::endl;

           ctr2 ++;
       }
   }
   myFile3.close();

   std::cout << "KLDiv lapalce to True = "<< KLDiv(EvalsLaplApp, evaluations) << std::endl;
   std::cout << "L2Norm lapalce to True = "<<L2Norm(EvalsLaplApp, evaluations) << std::endl;

   return 0;

}


