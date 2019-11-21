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
#include <cmath>
#include "FEMFunc.hpp"

#include <Eigen/Dense>


int main(){

    std::ofstream myFile;
    myFile.open("results.dat");

    double mu = 0;
    double sig = 0.1;

    std::lognormal_distribution<double> lognormal( mu, sig  );
    std::random_device rd;
    std::mt19937 engine( rd() );

    double SumDisp = 0;
    double SumA = 0;
    int Iters = 1e4;
    for(int i = 0; i < Iters ; i++){

        FEMFUNC TrussFem(false);
        double A = lognormal( engine ) / 100;
        double disp;
        TrussFem.modA(1, A);
        TrussFem.assembleS( );
        TrussFem.computeDisp( );
        TrussFem.computeForce( );
        disp = TrussFem.getDisp(6);
        std::cout<<A<<'\n';
        std::cout<<disp<<"\n\n";
        SumDisp +=disp;
        SumA += A;
        myFile<<A<<" "<<disp<<" "<<'\n';

    }
    double expecDisp = SumDisp / Iters;
    double expecA = SumA /Iters;

    myFile.close();
    std::cout<<"expecDisp  = "<<expecDisp<<"\n\n";
    std::cout<<"expecA  = "<<expecA<<"\n\n";

    FEMFUNC TrussFem(false);
    double mean = exp(mu + sig*sig/2.0) / 100;

    //TrussFem.modA(1, mean);
    TrussFem.modA(1, expecA);

    TrussFem.assembleS( );
    TrussFem.computeDisp( );
    TrussFem.computeForce( );
    std::cout<<"mean LogNorm = "<<mean<<'\n';
    std::cout<<"FEMFUNC(mean) = "<<TrussFem.getDisp(6)<<'\n';


    return 0;
}
