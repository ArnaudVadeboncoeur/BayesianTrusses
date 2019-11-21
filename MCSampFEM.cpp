/*
 * MCSampFEM.cpp
 *
 *  Created on: 21 Nov 2019
 *      Author: arnaudv
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include "FEMFunc.hpp"

#include <Eigen/Dense>


int main(){

   std::ofstream myFile;
   myFile.open("results.dat");

    std::lognormal_distribution<double> lognormal( 0.01, 0.1  );
    std::random_device rd;
    std::mt19937 engine( rd() );


    for(int i = 0; i < 1e3 ; i++){

        FEMFUNC TrussFem(false);
        double A = lognormal( engine );
        TrussFem.modA(1, A);
        TrussFem.assembleS( );
        TrussFem.computeDisp( );
        TrussFem.computeForce( );

        std::cout<<A<<'\n';
        std::cout<<TrussFem.getDisp(6)<<"\n\n";

        myFile<<A<<" "<<TrussFem.getDisp(6)<<" "<<'\n';

    }

    myFile.close();

    return 0;
}
