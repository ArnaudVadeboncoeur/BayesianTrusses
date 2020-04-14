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

#include "ThreeDTruss37Elm.hpp"
//#include "ThreeDTruss23Elm.hpp"
//#include "ThreeDTruss3Elm.hpp"

#include "sampleGen.hpp"
#include "PdfEval.hpp"

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
            //else{std::cout << trueForcingC[i] << std::endl;}

    }
    std::cout << trueSampleDispC.size() << std::endl;

    int numForcing = trueSampleDispC.size();

    constexpr unsigned DimK       =  30;
    constexpr unsigned DimObs     =  20 ;//1 node 3->x,y,z
    constexpr unsigned DimPara    =  3 ;

    constexpr unsigned NumTotPara =  37;

    std::vector<int> paraIndex     { 13, 16, 17};//, 16 };
    bool plot_1_dim = false;
    std::vector<int> plotParaIndex {0, 1};


    std::cout << "Here-main" << std::endl;

    //Index of dofs observed
//    Eigen::VectorXi nodesObs( 5 ); nodesObs <<   1, 2, 4, 9,  11;
//        Eigen::VectorXi ObsIndex( nodesObs.size() * 3 );
//        for(int j = 0; j < nodesObs.size(); ++j){
//
//            ObsIndex[ j*3 + 0] = nodesObs[j]*3 + 0;
//            ObsIndex[ j*3 + 1] = nodesObs[j]*3 + 1;
//            ObsIndex[ j*3 + 2] = nodesObs[j]*3 + 2;
//        }

    //Index of dofs observed -- x and y only
    Eigen::VectorXi nodesObs( 10 ); nodesObs <<   1, 2,3,4, 5, 8, 9,10,  11, 12;
        Eigen::VectorXi ObsIndex( nodesObs.size() * 2 );
        for(int j = 0; j < nodesObs.size(); ++j){

            ObsIndex[ j*2 + 0] = nodesObs[j]*3 + 0;
            ObsIndex[ j*2 + 1] = nodesObs[j]*3 + 1;
        }
//    Eigen::VectorXi ObsIndex( DimObs );
//    ObsIndex << 3,  4,
//                6,  7,  8,
//                9,  10,
//                12, 13, 14,
//                15, 16,
//
//                25, 25,
//                27, 28, 29,
//                30, 31,
//                33, 34, 35,
//                36, 37;//all non fixed x and y disp


    //Eigen::VectorXi ObsIndex( 3 ); ObsIndex << 9, 10, 11;

    std::cout << "Here-2" << std::endl;

//-------------------init prior information ----------------------------

    //find empirical std in data -- to do for every loading case

    std::vector <Eigen::MatrixXd> CovMatrixNoiseInvC(numForcing);

    for(int f = 0 ; f < numForcing; ++f){

    Eigen::VectorXd empMean( trueSampleDispC[f].cols() );
    empMean.setZero();

    for(int i = 0 ; i < trueSampleDispC[f].rows(); ++i){
        for(int j = 0 ; j < trueSampleDispC[f].cols(); ++j){

            empMean[j]+= trueSampleDispC[f](i, j);
        }
    }
    for(int i = 0 ; i < empMean.size(); ++i){
        empMean[i] = empMean[i] / trueSampleDispC[f].rows();
    }

    Eigen::VectorXd empStd( trueSampleDispC[f].cols() );
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


    //PdfEval< DimObs, DimPara , Eigen::VectorXd> PostFunc ( empStdCovMatrix, trueSampleDisp, paraIndex , ObsIndex, priorMeans, PriorCovMatrix );

    Eigen::VectorXd priorMeans(DimPara); priorMeans.setConstant(0.06);

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

//Newton Ralphson to find MAP--------------------------------------------


    Eigen::VectorXd X(DimPara); X.setConstant(0.07);
    Eigen::VectorXd XPast(DimPara); XPast.setConstant(0.07);
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

    Eigen::MatrixXd grad(1, DimPara);
    grad.setZero();


    Eigen::MatrixXd hess(DimPara, DimPara);
    Eigen::MatrixXd LaplaceHess_inv(DimPara, DimPara);
    hess.setZero();

    Eigen::MatrixXd hess_b( DimPara, 1 );
    hess.setZero();


    std::cout << "Not Done Creating L" << std::endl;
    //std::cout << KThetaFunc ( X ) << std::endl;
    std::cout <<"(KThetaFunc ( X )).rows() " << (KThetaFunc ( X )).rows() << std::endl;
    std::cout <<"(KThetaFunc ( X )).cols() " << (KThetaFunc ( X )).cols() << std::endl;
    //creat K to Obsversed Matrix L
    Eigen::MatrixXd L( DimObs , DimK ); L.setZero();



//    std::cout << "ObsIndex" << std::endl;
//    for(int i = 0; i < ObsIndex.size(); ++i){
//            std::cout  << ObsIndex[i]<<" " <<i << std::endl;
//        }
//    for(int i = 0; i < dofK.size(); ++i){
//        std::cout  << dofK[i]<<" "<< i << std::endl;
//    }


    //std::cout << "dofK" << std::endl;

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

//    std::cout << "dofK\n";
//    for(int i = 0; i < dofK.size(); ++i){ std::cout << dofK[i] << std::endl;}
//    std::cout << "ObsIndex\n" << ObsIndex << std::endl;
//    std::cout << "L\n" << L << std::endl;


    std::ofstream NRFile;
    NRFile.open("Newton-RalphsonOpt.dat");
    for(int d = 0; d < plotParaIndex.size(); ++d){
               NRFile << X[ plotParaIndex[d] ] << " ";
           } NRFile << "\n";


    //N-R iterations
    int maxIter = 2000;
    double step = 0.02;

    for(int i = 0; i < maxIter; ++i){

        k           = KThetaFunc ( X );
        //std::cout << "Computed K" << std::endl;
        k_inv       = k.inverse();
        grad = - ( X - priorMeans ).transpose() * PriorCovMatrixInv;


        for(int f = 0 ; f < numForcing; ++f){

            u_n         = uTheta(X, f) ;
            //std::cout << "Computed u_n" << std::endl;
            //std::cout << "\nu_n \n" << u_n << std::endl;

            u           = L * u_n;
            //std::cout << "Computed u reduced" << std::endl;

            du_dTheta   = L * dudThetaFunc(X, k_inv, u_n );
            //std::cout << "Computed u etc." << std::endl;


            //std::cout << grad << " " << std::endl;
            //std::cout << "Computed grad prior term" << std::endl;

            for(int j = 0; j < trueSampleDispC[f].rows(); ++j){

                for(int k = 0; k <trueSampleDispC[f].cols();++k ){
                    y_i(k,0)= trueSampleDispC[f](j, k);
                }
                //std::cout <<"(y_i - u).transpose() * CovMatrixNoiseInvC[f] * -1. * du_dTheta ;" << (y_i - u).transpose() * CovMatrixNoiseInvC[f] * -1. * du_dTheta << std::endl;
                grad -= (y_i - u).transpose() * CovMatrixNoiseInvC[f] * -1. * du_dTheta ;
            }

        }

        bool gradNull; gradNull = true;
        for(int j = 0; j < grad.size(); ++j){
            if( std::abs( grad(0, j) ) > Null ){ gradNull = false; break; }
        }
        if( gradNull ){
            std::cout << "gradNull\n" << grad << std::endl;
            grad.setConstant(1e-10);
            break ;
        }


//       if( 0.00000005 * grad.transpose()(0,0) > 0.1 || 0.00000005 * grad.transpose()(1,0) > 0.1 ){
//
//           std::cout << "big jump \n";
//           std::cout << "grad \n" << grad << "\n";
//           std::cout << "X \n" << X << "\n";
//           std::cout << "u \n" << u << "\n";
//           std::cout << "uPast \n" << uPast << "\n";
//           std::cout << "du_dTheta \n" << du_dTheta << "\n";
//           std::cout << "(y_i - u) \n" << (y_i - u) << "\n";
//           int f = 0;
//           for(int j = 0; j < trueSampleDispC[f].rows(); ++j){
//
//               for(int k = 0; k <trueSampleDispC[f].cols();++k ){
//                   y_i(k,0)= trueSampleDispC[f](j, k);
//               }
//               std::cout <<"(y_i - u).transpose() * CovMatrixNoiseInvC[f] * -1. * du_dTheta ;" << (y_i - u).transpose() * CovMatrixNoiseInvC[f] * -1. * du_dTheta << std::endl;
//               std::cout <<"(y_i - u).transpose() " << (y_i - u).transpose() << std::endl;
//
//           }
//       }
//       uPast = u;

       //X = X + 0.00000005 * grad.transpose();
        if(i % 2 == 0){
            if( (X - XPast).norm() < step/4. ){
                step = step/2.;
            }
            XPast = X;
        }
       X = X + step *  1. / grad.norm() *  grad.transpose() ;
       
//       std::cout <<"grad.transpose() " << grad.transpose() << std::endl;
//       std::cout <<"grad.norm() " << grad.norm() << std::endl;
//       std::cout <<"0.0005 *  1. / grad.norm() *  grad.transpose() ; " << 0.0005 *  1. / grad.norm() *  grad.transpose()  << std::endl;




       for(int j = 0; j < X.size(); ++j){
           if(X[j] < 0.){
               X[j] = 0;
           }
       }

       for(int d = 0; d < plotParaIndex.size(); ++d){
           NRFile << X[ plotParaIndex[d] ] << " ";
       } NRFile << "\n";

    }

    for(int i = 0; i < paraIndex.size() ; ++i){

        thetaHat_b.setZero();
        thetaHat_b( i, 0 ) = 1;
        hess_b = -1 * thetaHat_b.transpose( ) * PriorCovMatrixInv;

        for(int f = 0 ; f < numForcing; ++f){

            u_n = uTheta(X, f) ;

            u   = L * u_n;

            du_dTheta   = L * dudThetaFunc(X, k_inv, u_n );

            du_dtheta_b = L * dudtheta_iFunc( X, k_inv, dKdTheta_iFunc (X,  i ),  u_n )  ;

            du2_dthetab_dTheta = L * du2_dthetab_ThetaFunc( X,  k_inv, u_n , i );


            for(int j = 0; j < trueSampleDispC[f].rows(); ++j){

                for(int k = 0; k < trueSampleDispC[f].cols();++k ){
                    y_i(k,0)= trueSampleDispC[f](j, k);
                }

                hess_b -= (y_i - u).transpose() * CovMatrixNoiseInvC[f] * -1 * du2_dthetab_dTheta;
                hess_b -= du_dtheta_b.transpose() * CovMatrixNoiseInvC[f] * du_dTheta;

            }
        }

        for(int d = 0; d < DimPara; ++d){

            hess( i, d) = hess_b( 0, d );
        }
    }

    std::cout << "Done Newton Ralphson" << std::endl;

    NRFile.close();
    Eigen::MatrixXd negLogHess = -1 * hess;

    std::cout << "LaplaceMAP \n" << X << "\n\n";
    std::cout << "grad \n" << grad << "\n\n";
    std::cout << "hess \n" << hess << "\n\n";
    std::cout << "LaplaceHess_inv \n" << -1 * hess.inverse() << "\n\n";

    Eigen::VectorXd LaplaceMAP = X;


//Eval True Pdf to plot ---------------------------------------------------------

    std::cout << "Computing scatter points true pdf \n-----------------------------------------------" << '\n';
    //return 0;

    Eigen::VectorXd xPost( DimPara ); xPost = LaplaceMAP;

    int dimEval = 2;
    double LikVals;
    double maxVal = -9e30;
    Eigen::VectorXd maxXPost(DimPara); maxXPost.setConstant(0);
    double Vol = 0;

    std::ofstream myFile;
    myFile.open("pdfResults.dat");

    double a = 0.01;//-0.08;
    double b = 0.1;

    double c = 0.01;
    double d = 0.1;

    int samplesX = 1 * 1e2;
    if (plot_1_dim) {samplesX = 1;}
    int samplesY = 1 * 1e2;


    double dx = (double) (b - a) / samplesX;
    if (plot_1_dim) {dx = 1;}
    double dy = (double) (d - c) / samplesY;

    double bottomLim = -1e-3;

   //Eigen::MatrixXd evaluations ( samplesX * samplesY , 2);
   Eigen::MatrixXd evaluations ( samplesX * samplesY , dimEval + 1);

   unsigned ctr = 0;

   double Nan = 0; double NotNan = 0;

   for(int i = 0; i < samplesX; ++i){

       xPost[ plotParaIndex[0] ] = a + (double) dx * ( i + 1) ;

       for(int j = 0; j < samplesY; ++j){

           xPost[ plotParaIndex[1] ] = c + (double) dy * ( j + 1) ;

           //--------Eval True Post-----------

           //LikVals =  PostFunc.Eval( xPost ) ;

           LikVals = - 1./2.* std::log( PriorCovMatrix.determinant() )
                     - 1./2.* (xPost - priorMeans).transpose() * PriorCovMatrixInv * (xPost - priorMeans) ;
           //std::cout << "LikVals1 " << LikVals << std::endl;
           if( std::isnan(LikVals) ){ std::cout << "is nan 1 \n";}

           for(int f = 0 ; f < numForcing; ++f){

               u_n         = uTheta(xPost, f) ;

               u           = L * u_n;

               LikVals += - (double) trueSampleDispC[f].rows() / 2.0 * std::log( CovMatrixNoiseInvC[f].inverse().determinant() ) ;
               //std::cout << "LikVals2 " << LikVals << std::endl;
               if( std::isnan(LikVals) ){ std::cout << "is nan 2 \n";}

               for(int l = 0; l < trueSampleDispC[f].rows(); ++l){

                   for(int k = 0; k < trueSampleDispC[f].cols(); ++k ){
                       y_i(k,0)= trueSampleDispC[f](l, k);
                   }


                   LikVals += - ( 1./2. * (y_i - u).transpose() * CovMatrixNoiseInvC[f] * ( y_i - u ))(0, 0)   ;
                   //std::cout << "LikVals3 " << LikVals << "\n\n";
                   if( std::isnan(LikVals) ){
                       Nan +=1;

//                       std::cout << "IS NAN! " << Nan << std::endl;
//                       std::cout <<  " xPost \n" <<  xPost  << "\n";
//
//                       std::cout <<  " KThetaFunc ( xPost ) \n" <<  KThetaFunc ( xPost )  << "\n";
//
//                       std::cout << "is nan 3 \n";
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
               maxXPost   = xPost;
           }

           evaluations(ctr, 0) = xPost[ plotParaIndex[0] ];
           evaluations(ctr, 1) = xPost[ plotParaIndex[1] ];
           evaluations(ctr, 2) = LikVals;

           ctr++;
       }
   }
   std::cout << "IS NotNan " << NotNan << std::endl;
   std::cout << "IS NAN " << Nan << std::endl;


   for(int i = 0; i < evaluations.rows(); ++i){
       evaluations(i, dimEval ) = std::exp(evaluations(i, dimEval ) - maxVal) ;
       //Vol += evaluations(i, 1) * dx;
       Vol += evaluations(i, dimEval) * dx * dy;
   }

   for(int i = 0; i < evaluations.rows(); ++i){
       evaluations(i, dimEval) = evaluations(i, dimEval) / Vol;
       //std::cout << evaluations(i, dimEval) << std::endl;
       if( evaluations(i, dimEval) > bottomLim ){
           //std::cout << evaluations(i, 0) << " " << evaluations(i, 1) << " " << evaluations(i, 2) << std::endl;
           myFile << evaluations(i, 0) << " " << evaluations(i, 1) << " " << evaluations(i, 2) << std::endl;
       }
       //myFile << evaluations(i, 0) << " " << evaluations(i, 1) << std::endl;
   }


   std::cout << "maxVal = " << maxVal <<"\n"<< " maxX = \n" << maxXPost << " " << std::endl;
   //return 0;

   std::cout << "Generated true pdf points" << std::endl;
//Eval Laplace Approx --------------------------------------

   Eigen::VectorXd xGauss (DimPara); xGauss = LaplaceMAP;
   Eigen::MatrixXd EvalsLaplApp ( samplesX * samplesY , dimEval + 1);
   std::ofstream myFile3;
   myFile3.open("pdfLaplaceEval.dat");
   double probDensVal;
   std::cout << "-------------------" << '\n';

   int ctr2 = 0;


   //std::cout << "LaplaceMap = \n" << LaplaceMAP << " LaplaceHess_inv.sqrt() = \n" << LaplaceHess_inv.sqrt() << std::endl;
   for(int i = 0; i < samplesX; ++i){

       xGauss[ plotParaIndex[0] ] = a + (double) dx * ( i + 1) ;

       for(int j = 0; j < samplesY; ++j){


           xGauss[ plotParaIndex[1] ] = c + (double) dy * ( j + 1) ;

           probDensVal = 1. / ( std::sqrt( pow(2 * M_PI, 2) * negLogHess.inverse().determinant() )   ) *
                                     std::exp( - 1./2. * (xGauss - LaplaceMAP).transpose() * negLogHess * (xGauss - LaplaceMAP) );

           EvalsLaplApp(ctr2, 0) = xGauss[ plotParaIndex[0] ];
           EvalsLaplApp(ctr2, 1) = xGauss[ plotParaIndex[1] ];
           EvalsLaplApp(ctr2, 2) = probDensVal;
           //std::cout << xGauss[ 0 ] << " " << xGauss[1] << " " << probDensVal << std::endl;

           if( probDensVal > bottomLim ){
               myFile3 << xGauss[ plotParaIndex[0] ] << " " << xGauss[ plotParaIndex[1] ] << " " << probDensVal << std::endl;
           }
           //myFile3 << xGauss[0] << " " << probDensVal << std::endl;
           ctr2 ++;
       }
   }
   myFile3.close();

   std::cout << "KLDiv lapalce to True = "<< KLDiv(EvalsLaplApp, evaluations) << std::endl;
   std::cout << "L2Norm lapalce to True = "<<L2Norm(EvalsLaplApp, evaluations) << std::endl;

   return 0;

}


