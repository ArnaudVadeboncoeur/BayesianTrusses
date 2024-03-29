/*
 * FEMClass.hpp
 *
 *  Created on: 21 Nov 2019
 *      Author: arnaudv
 */

#ifndef FEMClass_HPP_
#define FEMClass_HPP_

#include <iostream>
#include <vector>
#include <tuple>

#include <Eigen/Dense>
#include "matTools.hpp"





using TupleTrussDef = std::tuple <  unsigned, unsigned,
                                    Eigen::VectorXd,
                                    Eigen::VectorXd,
                                    Eigen::MatrixXd,
                                    Eigen::VectorXi,
                                    Eigen::MatrixXi,
                                    Eigen::MatrixXi,
                                    Eigen::MatrixXd  > ;

class FEMClass{
public:

    FEMClass ( ) { };

    FEMClass ( bool, TupleTrussDef ) ;

    void FEMClassReset ( bool );

    void assembleS ( ) ;

    void computeDisp ( );

    void computeForce ( );



    void modA(int index, double Area)                       { A_[index] = Area ; }
    void modE(int index, double Modulus)                    { E_[index] = Modulus ; }
    void modForce( int i_dof, int j_xyz, double forceMod )  { force_(i_dof * 3 + j_xyz, 0) = forceMod ; }
    void modForce( Eigen::MatrixXd Newforcing )             { force_ = Newforcing ; }

    double getDisp(int dofIndex)          { return disp_[dofIndex] ; }
    Eigen::VectorXd getDisp(   )          { return disp_ ; }


    Eigen::VectorXd getDisp( Eigen::VectorXi dofs ) {

    	Eigen::VectorXd disps (dofs.size() );
    	for(int i = 0; i < dofs.size(); ++i){
    		disps[i] = allDisp_[dofs[i]];
    	}
    	return disps ;
    }


    Eigen::VectorXd getAllDisp( )	     {return allDisp_; }

    double getForce(int dofIndex)        { return force_(dofIndex, 0) ; }
    Eigen::MatrixXd getForce(   )        { return force_ ; }

    double getK(int dofi, int dofj)      { return S_(dofi, dofj) ; }
    Eigen::MatrixXd getK(   )            { return S_ ; }

    double getA(int memberIndex )        { return A_( memberIndex ); }
    Eigen::VectorXd getA (   )           { return A_; }

    std::vector< int > getFreeDof( )     { return freeDof_; }

    ~FEMClass ( ) { };

private:

    bool verbosity_;

    //Initialised Variable and containers

    TupleTrussDef TrussDef_;

    unsigned numberNodes_ ;
    unsigned numberElms_ ;

    Eigen::VectorXd A_ ;
    Eigen::VectorXd E_ ;
    Eigen::MatrixXd nodes_ ;
    Eigen::VectorXi dof_ ;
    Eigen::MatrixXi members_ ;
    Eigen::MatrixXi memberData_ ;
    Eigen::MatrixXd force_ ;

    std::vector < double > dofK_ ;

    //Created Variables and containers
    std::vector< Eigen::MatrixXd >  vectorOfK_ ;
    std::vector< Eigen::MatrixXd >  vectorOfLocalK_ ;
    std::vector< Eigen::MatrixXd >  vectorOfT_ ;
    std::vector< std::vector<int> > dofKgs_ ;
    std::vector< int > freeDof_;

    Eigen::VectorXd disp_;
    Eigen::VectorXd allDisp_;
    Eigen::VectorXd allForce_;

    Eigen::MatrixXd S_;

    Eigen::MatrixXd baseK_ ;


    std::vector<Eigen::Vector2d> vectorOfU_ ;
    std::vector<Eigen::Vector2d> vectorOfQ_ ;
    std::vector<Eigen::VectorXd> vectorOfV_ ;


};

FEMClass::FEMClass ( bool verbosity, TupleTrussDef TrussDef ) {

    TrussDef_ = TrussDef;
    FEMClassReset( verbosity);




}

void FEMClass::FEMClassReset( bool verbosity){

    verbosity_ = verbosity;

    std::tie (numberNodes_,
              numberElms_,
              A_, E_,
              nodes_,
              dof_,
              members_,
              memberData_,
              force_) = TrussDef_;


    freeDof_.clear();

    //make element wise copy of matrixXd force_ as this one is reduced by reference in code
    allForce_.resize(force_.size());
    for(int i = 0; i < force_.size(); ++i){ allForce_[i] = force_( i, 0) ;}

    vectorOfK_.resize(numberElms_);
    vectorOfLocalK_.resize(numberElms_);
    vectorOfT_.resize(numberElms_);
    dofKgs_.resize(numberElms_);

    disp_.resize( numberNodes_ * 3 );

    baseK_.resize(2,2);
    baseK_ << 1, -1, -1, 1;

    S_ = Eigen::MatrixXd::Zero( numberNodes_ * 3, numberNodes_ * 3 );
    allDisp_= Eigen::VectorXd::Zero( numberNodes_ * 3 );

    vectorOfU_.resize( numberElms_ );
    vectorOfQ_.resize( numberElms_ );
    vectorOfV_.resize( numberElms_ );

    dofK_.clear();

    if(verbosity_){std::cout<<"Done Creating local containers"<<'\n';}
    return;
}

void FEMClass::assembleS(){
    //---------------------------Compute all Global K matrices-----------------------------//

    for(int i =0; i < numberElms_ ; ++i){
        Eigen::MatrixXd Ki(2,2);

        double l = sqrt(   pow(  nodes_(members_(i,1), 0) - nodes_(members_(i,0), 0)  , 2 )      //x

                        +  pow(  nodes_(members_(i,1), 1) - nodes_(members_(i,0), 1)  , 2 )      //y

                        +  pow(  nodes_(members_(i,1), 2) - nodes_(members_(i,0), 2)  , 2 )   ); //z

        double cosX =  ( nodes_(members_(i,1), 0) - nodes_(members_(i,0), 0) ) / l;
        double cosY =  ( nodes_(members_(i,1), 1) - nodes_(members_(i,0), 1) ) / l;
        double cosZ =  ( nodes_(members_(i,1), 2) - nodes_(members_(i,0), 2) ) / l;

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


        vectorOfLocalK_[i] = E_(memberData_(i,0)) * A_(memberData_(i,1)) / l * baseK_;

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

    //remove fixed degrees of freedom
    for(int i = dof_.size() - 1; i >= 0 ; --i){

        if(dof_[i] == 1){

        	matTools::removeColumn(S_,i);
        	matTools::removeRow(S_, i);
        	matTools::removeRow(force_, i);
        }
        else{ freeDof_.insert(freeDof_.begin(),i);}
    }

    if( verbosity_ ) { std::cout <<"Structure Matrix \n" << S_ << '\n'; }

    if( verbosity_ ) { std::cout <<"Force Applied \n" << force_ << '\n'; }

   return;

}


void FEMClass::computeDisp( ){
    //---------------------------------compute displacement---------------------------------//


    disp_ =  S_.inverse() * force_ ;

    //std::cout << "FemClassDisp\n" << disp_ << std::endl;


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

void FEMClass::computeForce( ){

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


#endif /* FEMClass_HPP_ */

