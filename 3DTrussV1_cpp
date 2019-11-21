#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <math.h>

#include "ThreeDTrussDef.hpp"

int main(){
    bool verbosity = true;

    if( verbosity ) { std::cout << "Creating Gemoetry" << "\n"; }
    TrussAssignment();

//---------------------------Compuete all Global K matrices-----------------------------//
    std::vector< Eigen::MatrixXd >  vectorOfK (numberElms);
    std::vector< Eigen::MatrixXd >  vectorOfLocalK (numberElms);
    std::vector< Eigen::MatrixXd >  vectorOfT (numberElms);
    std::vector< std::vector<int> > dofKgs(numberElms);


    Eigen::MatrixXd baseK(2,2);
    baseK <<  1., -1.,
             -1.,  1.;

    for(int i =0; i < numberElms; ++i){
        Eigen::MatrixXd Ki(2,2);

        double l = sqrt(   pow(  nodes(members(i,1), 0) - nodes(members(i,0), 0)  , 2 )      //x

                        +  pow(  nodes(members(i,1), 1) - nodes(members(i,0), 1)  , 2 )      //y

                        +  pow(  nodes(members(i,1), 2) - nodes(members(i,0), 2)  , 2 )   ); //z

        double cosX =  ( nodes(members(i,1), 0) - nodes(members(i,0), 0) ) / l;
        double cosY =  ( nodes(members(i,1), 1) - nodes(members(i,0), 1) ) / l;
        double cosZ =  ( nodes(members(i,1), 2) - nodes(members(i,0), 2) ) / l;

        std::vector<int> dofKi(6);
        for(unsigned ii = 0; ii < dofKi.size(); ++ii ){

            if(ii < 3){ dofKi[ii] = (members(i,0)) * 3 + ii    ;}
            else      { dofKi[ii] = (members(i,1)) * 3 + ii % 3;}
        }
        dofKgs[i] = dofKi;

        Eigen::MatrixXd T (2,6);
        T << cosX, cosY, cosZ,  0,     0,   0,
             0,    0,    0,     cosX, cosY, cosZ;

        vectorOfT[i] = T;


        vectorOfLocalK[i] = E(memberData(i,0)) * A(memberData(i,1)) / l * baseK;

        vectorOfK[i] = T.transpose() * vectorOfLocalK[i] * T ;

        if( verbosity ) { std::cout <<"Global K"<<i<<"\n"<< vectorOfK[i] << "\n\n\n"; }
    }

//------------------------Assemble Structure Stiffness Matrix ---------------------------//

    Eigen::MatrixXd S(numberNodes * 3, numberNodes * 3);

    for(unsigned kid = 0; kid < numberElms; ++kid){

        for(int i = 0 ; i < 6; ++i){

            for(int j =0; j < 6; ++j){

                S(dofKgs[kid][i],dofKgs[kid][j]) += vectorOfK[kid](i,j);
            }
        }

    }

//---------------------------------compute displacement---------------------------------//

//remove fixed degress of freedom

    std::vector<int> freeDof;
    for(int i = dof.size() - 1; i >= 0 ; --i){

        if(dof[i] == 1){

            removeColumn(S,i);
            removeRow(S, i);
            removeRow(force, i);
        }
        else{ freeDof.insert(freeDof.begin(),i);}
    }

   Eigen::VectorXd disp( numberNodes * 3 );

   if( verbosity ) { std::cout <<"Structure Matrix \n"<< S << '\n'; }

   if( verbosity ) { std::cout << force << '\n'; }

   disp =  S.inverse() * force ;

   std::cout<<"at degrees of freedom disp is:\n\n";

   for(unsigned i =0; i < freeDof.size(); ++i){

       std::cout<<"dof: "<<freeDof[i]<<'\t'<<"disp = "<<disp(i)<<'\n';
   }
//-------------------------------compute force in members------------------------------//

   std::vector<Eigen::Vector2d> vectorOfU ( numberElms );
   std::vector<Eigen::Vector2d> vectorOfQ ( numberElms );
   std::vector<Eigen::VectorXd> vectorOfV ( numberElms );

   for(int kid =0 ; kid < numberElms; ++kid){

       //put proper global displacement into member disp v vector
       Eigen::VectorXd v(6);
       for(int i = 0; i < freeDof.size() ; ++i){
           for(int j = 0; j < 6 ; ++j){

               if(dofKgs[kid][j] == freeDof[i]){

                   v[j] = disp[i];
               }
           }
       }

       vectorOfV[kid] = v;

       vectorOfU[kid] = vectorOfT[kid]      * vectorOfV[kid];
       vectorOfQ[kid] = vectorOfLocalK[kid] * vectorOfU[kid];

       std::cout<< "\n\nQ"<<kid<<" :\n";
       std::cout<<vectorOfQ[kid]<<"\n";
   }

   Eigen::VectorXd allDisp( numberNodes * 3 );

   for(int i = 0; i < freeDof.size(); ++i){
       for(int j = 0; j < numberNodes * 3; ++j){
       if(freeDof[i] == j){
           allDisp[j]=disp[i];
       }
       }
   }

   if( verbosity ) { std::cout <<"allDisp :\n"<< allDisp << '\n'; }

   for(int i =0; i <23; i++){std::cout<<(i+1)*2<<'\n';}

    return 0;
}


