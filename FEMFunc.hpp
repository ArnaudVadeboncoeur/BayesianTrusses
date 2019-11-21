/*
 * FEMFunc.hpp
 *
 *  Created on: 21 Nov 2019
 *      Author: arnaudv
 */

#ifndef FEMCLS_HPP_
#define FEMFUNC_HPP_

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include "ThreeDTrussDef.hpp"



class FEMFUNC{
public:

    FEMFUNC ( ) { };

    FEMFUNC ( bool );

    void assembleS ( ) ;

    void computeDisp ( );

    void computeForce ( );

    //method to modify A, E, Force

    //method to get disp at certain nodes

    //methode to get force at certain members

    ~FEMFUNC ( ) { };

private:

    bool verbosity_;

    //Initialised Variable and containers
    unsigned numberNodes_ ;
    unsigned numberElms_ ;

    Eigen::VectorXd A_ ;
    Eigen::VectorXd E_ ;
    Eigen::VectorXi dof_ ;

    Eigen::MatrixXd nodes_ ;
    Eigen::MatrixXi members_ ;
    Eigen::MatrixXi memberData_ ;
    Eigen::MatrixXd force_ ;

    //Created Variables and containers
    std::vector< Eigen::MatrixXd >  vectorOfK_ ;
    std::vector< Eigen::MatrixXd >  vectorOfLocalK_ ;
    std::vector< Eigen::MatrixXd >  vectorOfT_ ;
    std::vector< std::vector<int> > dofKgs_ ;
    std::vector<int> freeDof_;

    Eigen::VectorXd disp_;
    Eigen::VectorXd allDisp_;

    Eigen::MatrixXd S_;

    Eigen::MatrixXd baseK_ ;


    std::vector<Eigen::Vector2d> vectorOfU_ ;
    std::vector<Eigen::Vector2d> vectorOfQ_ ;
    std::vector<Eigen::VectorXd> vectorOfV_ ;


};

FEMFUNC::FEMFUNC( bool verbosity  ){

    verbosity_ = verbosity ;

    InitialTrussAssignment();

    numberNodes_   = numberNodes;
    numberElms_    = numberElms;

    A_             = A;
    E_             = E;
    dof_           = dof;

    nodes_         = nodes;
    members_       = members;
    memberData_    = memberData;
    force_         = force;

    vectorOfK_.resize(numberElms_);
    vectorOfLocalK_.resize(numberElms_);
    vectorOfT_.resize(numberElms_);
    dofKgs_.resize(numberElms_);

    disp_.resize( numberNodes_ * 3 );

    baseK_.resize(2,2);
    baseK_ << 1, -1, -1, 1;

    S_.resize( numberNodes_ * 3, numberNodes_ * 3 );

    allDisp_.resize( numberNodes_ * 3 );

    vectorOfU_.resize( numberElms_ );
    vectorOfQ_.resize( numberElms_ );
    vectorOfV_.resize( numberElms_ );

    if(verbosity_){std::cout<<"Done Creating local containers"<<'\n';}

}

void FEMFUNC::assembleS(){
    //---------------------------Compuete all Global K matrices-----------------------------//

    for(int i =0; i < numberElms_ ; ++i){
            Eigen::MatrixXd Ki(2,2);

            double l = sqrt(   pow(  nodes(members_(i,1), 0) - nodes(members_(i,0), 0)  , 2 )      //x

                            +  pow(  nodes(members_(i,1), 1) - nodes(members_(i,0), 1)  , 2 )      //y

                            +  pow(  nodes(members_(i,1), 2) - nodes(members_(i,0), 2)  , 2 )   ); //z

            double cosX =  ( nodes(members_(i,1), 0) - nodes(members_(i,0), 0) ) / l;
            double cosY =  ( nodes(members_(i,1), 1) - nodes(members_(i,0), 1) ) / l;
            double cosZ =  ( nodes(members_(i,1), 2) - nodes(members_(i,0), 2) ) / l;

            std::vector<int> dofKi(6);
            for(unsigned ii = 0; ii < dofKi.size(); ++ii ){

                if(ii < 3){ dofKi[ii] = (members_(i,0)) * 3 + ii    ;}
                else      { dofKi[ii] = (members_(i,1)) * 3 + ii % 3;}
            }
            dofKgs_[i] = dofKi;

            Eigen::MatrixXd T (2,6);
            T << cosX, cosY, cosZ,  0,     0,   0,
                 0,    0,    0,     cosX, cosY, cosZ;

            vectorOfT_[i] = T;


            vectorOfLocalK_[i] = E_(memberData(i,0)) * A_(memberData(i,1)) / l * baseK_;

            vectorOfK_[i] = T.transpose() * vectorOfLocalK_[i] * T ;

            if( verbosity_ ) { std::cout <<"Global K"<<i<<"\n"<< vectorOfK_[i] << "\n\n\n"; }
        }



    //------------------------Assemble Structure Stiffness Matrix ---------------------------//

        for(unsigned kid = 0; kid < numberElms_; ++kid){

            for(int i = 0 ; i < 6; ++i){

                for(int j =0; j < 6; ++j){

                    S_(dofKgs_[kid][i],dofKgs_[kid][j]) += vectorOfK_[kid](i,j);
                }
            }

        }

   return;

}


void FEMFUNC::computeDisp( ){
    //---------------------------------compute displacement---------------------------------//


    //remove fixed degrees of freedom
        for(int i = dof_.size() - 1; i >= 0 ; --i){

            if(dof_[i] == 1){

                removeColumn(S_,i);
                removeRow(S_, i);
                removeRow(force_, i);
            }
            else{ freeDof_.insert(freeDof_.begin(),i);}
        }

       if( verbosity_ ) { std::cout <<"Structure Matrix \n" << S_ << '\n'; }

       if( verbosity_ ) { std::cout <<"Force Applied \n" << force_ << '\n'; }

       disp_ =  S_.inverse() * force_ ;


       if(verbosity_){

           std::cout<<"at degrees of freedom disp is:\n\n";

           for(unsigned i =0; i < freeDof_.size(); ++i){

               std::cout<<"dof: "<<freeDof_[i]<<'\t'<<"disp = "<<disp_(i)<<'\n';
           }
       }

          for(int i = 0; i < freeDof_.size(); ++i){
              for(int j = 0; j < numberNodes_ * 3; ++j){
              if(freeDof_[i] == j){
                  allDisp_[j]=disp_[i];
              }
              }
          }

          if( verbosity_ ) { std::cout <<"allDisp :\n"<< allDisp_ << '\n'; }

          if( verbosity_ ) { for(int i =0; i <23; i++){std::cout<<(i+1)*2<<'\n';} }
}

void FEMFUNC::computeForce( ){

    //-------------------------------compute force in members------------------------------//
       for(int kid =0 ; kid < numberElms_; ++kid){

           //put proper global displacement into member disp v vector
           Eigen::VectorXd v(6);
           for(int i = 0; i < freeDof_.size() ; ++i){
               for(int j = 0; j < 6 ; ++j){

                   if(dofKgs_[kid][j] == freeDof_[i]){

                       v[j] = disp_[i];
                   }
               }
           }

           vectorOfV_[kid] = v;

           vectorOfU_[kid] = vectorOfT_[kid]      * vectorOfV_[kid];
           vectorOfQ_[kid] = vectorOfLocalK_[kid] * vectorOfU_[kid];

           if( verbosity_ ){std::cout<< "\n\nQ"<<kid<<" :\n";
                            std::cout<<vectorOfQ_[kid]<<"\n";}
       }


}

#endif /* FEMFUNC_HPP_ */








