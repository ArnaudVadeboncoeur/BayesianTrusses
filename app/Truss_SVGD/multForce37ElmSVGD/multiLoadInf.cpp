/*
 * BTrussMCMC_MH.cpp
 *
 *  Created on: 12 Dec 2019
 *      Author: arnaudv
 */

#include "../../../src/FEMClass.hpp"
#include "../../../src/statTools/SVGD.hpp"

#include "trueModelDataGen.hpp"
#include "../Truss37Elm.hpp"

#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <math.h>


int main(){

	using vecMat = std::vector< Eigen::MatrixXd > ;


    constexpr unsigned DimK       =  30 ;
    constexpr unsigned DimObs     =  6 ;
    constexpr unsigned DimPara    =  2 ;

    constexpr unsigned NumTotPara =  37;
    //these worked well --           {12, 13,14, 15, 16, 17  };
    //std::vector<int> paraIndex     { 0, 1, 2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14, 15, 16, 17 , 18, 19, 20, 21 };
    //std::vector<int> paraIndex     { 12, 13,14, 15, 16, 17 , 18, 19, 20, 21 };
    std::vector<int> paraIndex     {  13, 16 };
    bool             plot_1_dim    = false;
    std::vector<int> plotParaIndex {0, 1};

    //Index of dofs observed -- 2 = x and y only
    int Numxyz = 2;//1, 2, 3
    Eigen::MatrixXi nodesObs(1,  3 ); nodesObs <<   1, 2, 3;
        Eigen::VectorXi ObsIndex( nodesObs.size() * Numxyz );
        for(int j = 0; j < nodesObs.size(); ++j){

            ObsIndex[ j*Numxyz + 0] = nodesObs(0, j)*3 + 0;   //x
            ObsIndex[ j*Numxyz + 1] = nodesObs(0, j)*3 + 1;   //y
          //ObsIndex[ j*Numxyz + 2] = nodesObs(0, j)*3 + 2;   //z
        }


	DataCont trueSamplesTupleContainer = trueSampleGen( ObsIndex );

	Eigen::MatrixXd trueSampleDispC    = std::get<0>( trueSamplesTupleContainer );
	Eigen::MatrixXi ytL                = std::get<1>( trueSamplesTupleContainer );
	vecMat          trueForcingC       = std::get<2>( trueSamplesTupleContainer );
	double          sigma_n            = std::get<3>( trueSamplesTupleContainer );


//---------------------------------------Noise----------------------------------

	Eigen::MatrixXd cov_n (DimPara, DimPara);
	cov_n.setZero();
	for(int i = 0; i < DimPara; ++i){
		cov_n(i,i) = sigma_n * sigma_n;
	}
	Eigen::MatrixXd cov_n_Inv = cov_n.inverse();
//---------------------------------------Prior----------------------------------

    Eigen::VectorXd priorMeans(DimPara);
    priorMeans.setConstant(0.06);

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
//Your gonna have to rewrite



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
    std::function < Eigen::MatrixXd ( const Eigen::MatrixXd) > KThetaFunc;
    KThetaFunc = [ &TrussFem, paraIndex ](  const Eigen::MatrixXd& X ){
        Eigen::MatrixXd K;
        //produce k(theta)
        for(int j = 0; j < paraIndex.size(); ++j){
            TrussFem.modA(paraIndex[j], X(0,j));
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
   std::function < Eigen::MatrixXd ( const Eigen::VectorXd,
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
	std::function < Eigen::MatrixXd ( const Eigen::VectorXd,
									  const Eigen::MatrixXd, const Eigen::VectorXd, int, int) > du2_dthetai_dthetajFunc;

	du2_dthetai_dthetajFunc = [ &dudtheta_iFunc, &dKdTheta_iFunc ]
						      ( const Eigen::VectorXd& X, const Eigen::MatrixXd& K_inv,
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



//std::functions needed for SVGD
//	- Particle Initialisation
// 	- G(theta): -- forward model solution for all settings on Xn, matrix: [ n samples x n observed dims ]
// 	- delxG(theta): derivative forward model, vector of matrices: < n samples x [ n Observed x n parameters ] >
// 	- delxlogP(theta): use G(theta), Yobs, delxG(theta),L ,noise cov, prior to compute derivative of logP(x)
// 	results in [n samples x n parameters]
//
//===========================New code -- get samples from SVGD===================================

////Labmda function to compute dudTheta
//   std::function < Eigen::MatrixXd ( const Eigen::VectorXd,
//									 const Eigen::MatrixXd, const Eigen::VectorXd ) > dudThetaFunc;
//
//   dudThetaFunc = [ &dudtheta_iFunc, &dKdTheta_iFunc ]( const Eigen::VectorXd& X, const Eigen::MatrixXd& K_inv,
//														const Eigen::VectorXd& u){
//
//	   Eigen::MatrixXd dudTheta( K_inv.rows(),  X.rows() );
//
//	   Eigen::VectorXd dudtheta_i( X.rows() );
//
//	   for(int i = 0; i < X.rows(); ++i ){
//
//		   dudtheta_i = dudtheta_iFunc( X, K_inv, dKdTheta_iFunc( X, i), u );
//		   for(int j = 0; j < u.rows(); ++j ){
//			   dudTheta(j, i) = dudtheta_i(j);
//			   }
//		   }
//
//	   return dudTheta;
//   };



////Lambda function to compute G(theta)
////- G(theta): -- forward model solution for all settings on Xn, matrix:< nForcing x [ n samples x n observed dims ] >
//=


	Eigen::MatrixXd L( DimObs , DimK ); L.setZero();
	   for(int i = 0; i < ObsIndex.size(); ++i ){
	           for( int j = 0; j < dofK.size(); ++j ){
	               if( dofK[j] == ObsIndex[i] ){
	                   L(i, j) = 1;
	                   break;
	               }
	           }
	       }

	//compute del_xLogP(x)
	std::function < Eigen::MatrixXd (const Eigen::MatrixXd) > delLogP;
	delLogP = [&TrussFem, &KThetaFunc,  &dudThetaFunc, &uTheta,
	           paraIndex, trueForcingC, DimK,DimObs, ObsIndex, ytL,
	           L, cov_n_Inv, priorMeans, PriorCovMatrixInv ]
	           (const  Eigen::MatrixXd& X ){

	   Eigen::MatrixXd delLogPVar (X.rows(), X.cols()); delLogPVar.setZero();
	   Eigen::MatrixXd X_i (1, X.cols());

	   Eigen::MatrixXd k(DimK, DimK);
      Eigen::MatrixXd k_inv(DimK, DimK);

      Eigen::MatrixXd u(1, DimObs);
      Eigen::MatrixXd u_n( DimK  ,  1 );
      Eigen::MatrixXd du_dTheta ( DimObs, DimPara );

	   for(int i = 0; i < delLogPVar.rows(); ++i){

	      X_i   = X.row(i).transpose() ;

	      k     =     KThetaFunc ( X_i ) ;
	      k_inv =     k.inverse();

	      delLogPVar.row(i) += - ( X_i - priorMeans ).transpose() * PriorCovMatrixInv;

	      for(int j = 0 ; j < ytL.rows() ; ++j){

	         if(j > 0 ){
	            if(ytL(j, 0) != ytL(j - 1, 0)){

	            u_n         = uTheta(X, f) ;
	            u           = L * uTheta(X, ytL(j, 0) );
               du_dTheta   = L * dudThetaFunc(X, k_inv, u_n );
	         }}
	         delLogPVar.row(i) -= (trueSampleDispC.row(j).transpose() - u).transpose() * cov_n_Inv * -1. * du_dTheta ;
	         }
	   }

	   return delLogPVar;
	};

return 0;}




/*



//Labmda function to compute dudTheta
//- delxG(theta): derivative forward model, vector of matrices: < n samples x [ n Observed x n parameters ] >
//   std::function < Eigen::MatrixXd ( const Eigen::VectorXd,
//									 const Eigen::MatrixXd, const Eigen::VectorXd ) > dudThetaFunc;
//
//   dudThetaFunc = [ &dudtheta_iFunc, &dKdTheta_iFunc ]( const Eigen::VectorXd& X, const Eigen::MatrixXd& K_inv,
//														const Eigen::VectorXd& u){
//
//	   Eigen::MatrixXd dudTheta( K_inv.rows(),  X.rows() );
//
//	   Eigen::VectorXd dudtheta_i( X.rows() );
//
//	   for(int i = 0; i < X.rows(); ++i ){
//
//		   dudtheta_i = dudtheta_iFunc( X, K_inv, dKdTheta_iFunc( X, i), u );
//		   for(int j = 0; j < u.rows(); ++j ){
//			   dudTheta(j, i) = dudtheta_i(j);
//			   }
//		   }
//
//	   return dudTheta;
//   };
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


	Eigen::MatrixXd L( DimObs , DimK ); L.setZero();
	for(int i = 0; i < ObsIndex.size(); ++i ){
	        for( int j = 0; j < dofK.size(); ++j ){
	            if( dofK[j] == ObsIndex[i] ){
	                L(i, j) = 1;
	                break;
	            }
	        }
	    }


    k = KThetaFunc ( X );
	//std::cout << "Computed K" << std::endl;
	k_inv       = k.inverse();
	grad = - ( X - priorMeans ).transpose() * PriorCovMatrixInv;

	for(int f = 0 ; f < numForcing; ++f){

		u_n         = uTheta(X, f) ;

		u           = L * u_n;

		du_dTheta   = L * dudThetaFunc(X, k_inv, u_n );;

		for(int j = 0; j < trueSampleDispC[f].rows(); ++j){

			for(int k = 0; k <trueSampleDispC[f].cols();++k ){
				y_i(k,0)= trueSampleDispC[f](j, k);
			}
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


    return 0;
}

*/ /*












// Old code ---
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

    for(int i = 0; i < ObsIndex.cols(); ++i ){
        for( int j = 0; j < dofK.size(); ++j ){
            if( dofK[j] == ObsIndex(0,i) ){
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
    NRFile.open("GradBisectionOpt.dat");
    for(int d = 0; d < plotParaIndex.size(); ++d){
               NRFile << X[ plotParaIndex[d] ] << " ";
           } NRFile << "\n";

   std::ofstream FullDimOpt;
   FullDimOpt.open("FullDimOpt.dat");
   for(int d = 0; d < X.size(); ++d){
       FullDimOpt << X[d] << " ";
          } FullDimOpt << "\n";


    //GradBisectionOpt
    int maxIter = 1e3;
    double step = 0.01;

    for(int i = 0; i < maxIter; ++i){

        k           = KThetaFunc ( X );
        //std::cout << "Computed K" << std::endl;
        k_inv       = k.inverse();
        grad = - ( X - priorMeans ).transpose() * PriorCovMatrixInv;

        for(int f = 0 ; f < numForcing; ++f){

            u_n         = uTheta(X, f) ;

            u           = L * u_n;

            du_dTheta   = L * dudThetaFunc(X, k_inv, u_n );;

            for(int j = 0; j < trueSampleDispC[f].rows(); ++j){

                for(int k = 0; k <trueSampleDispC[f].cols();++k ){
                    y_i(k,0)= trueSampleDispC[f](j, k);
                }
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

       //X = X + 0.00000005 * grad.transpose();
        if(i % 2 == 0){
            if( (X - XPast).norm() < 0.8 * step ){
                step = step * 0.8 ;
            }
            XPast = X;
        }
       X = X + step *  1. / grad.norm() *  grad.transpose() ;

       for(int j = 0; j < X.size(); ++j){
           if(X[j] < 0.){
               X[j] = 0;
           }
       }

       for(int d = 0; d < plotParaIndex.size(); ++d){
           NRFile << X[ plotParaIndex[d] ] << " ";
       } NRFile << "\n";

       for(int d = 0; d < X.size(); ++d){
          FullDimOpt << X[d] << " ";
       } FullDimOpt << "\n";

    }

    //Compute Hessian for Laplace approx
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

    std::cout << "Done Laplce Approx" << std::endl;

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

           LikVals = - 1./2.* std::log( PriorCovMatrix.determinant() )
                     - 1./2.* (xPost - priorMeans).transpose() * PriorCovMatrixInv * (xPost - priorMeans) ;

           if( std::isnan(LikVals) ){ std::cout << "is nan 1 \n";}

           for(int f = 0 ; f < numForcing; ++f){

               u_n         = uTheta(xPost, f) ;

               u           = L * u_n;

               LikVals += - (double) trueSampleDispC[f].rows() / 2.0 * std::log( CovMatrixNoiseInvC[f].inverse().determinant() ) ;

               if( std::isnan(LikVals) ){ std::cout << "is nan 2 \n";}

               for(int l = 0; l < trueSampleDispC[f].rows(); ++l){

                   for(int k = 0; k < trueSampleDispC[f].cols(); ++k ){
                       y_i(k,0)= trueSampleDispC[f](l, k);
                   }

                   LikVals += - ( 1./2. * (y_i - u).transpose() * CovMatrixNoiseInvC[f] * ( y_i - u ))(0, 0)   ;
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


   //std::cout << "Generated true pdf points" << std::endl;
   //Eval Laplace Approx can be found in Laplce approx multiload script--------------------------------------

   return 0;

}

*/
