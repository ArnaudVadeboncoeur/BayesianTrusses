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

#include "src/FEMClass.hpp"
#include "src/SampleAnalysis.hpp"


#include <Eigen/Dense>


int main(){

    //std::ofstream myFile;
    //myFile.open("results.dat", std::ios::trunc);

    double mu = 0;
    double sig = 1;

    double upperBound =  1e10;
    double lowerBound = -1e10;

    std::lognormal_distribution<double> lognormal( mu, sig  );
    //std::normal_distribution<double> normal( mu, sig  );
    std::random_device rd;
    std::mt19937 engine( rd() );

    double SumDisp = 0;

    int numSamples = 1e5;
    double valMax = -1e6;
    double valMin =  1e6;

    Eigen::MatrixXd allSamples (numSamples, 1);

    FEMClass TrussFem(false);

    for(int i = 0; i < numSamples ; i++){

        double A = lognormal( engine ) / 100.0 ;
        int dof = 6;
        //double A = 0.01;
        double disp;
        TrussFem.modA(1, A);
        TrussFem.assembleS( );
        TrussFem.computeDisp( );
        TrussFem.computeForce( );


        allSamples(i, 0) = TrussFem.getDisp(dof);

        if( TrussFem.getDisp(dof) > valMax ) { valMax =  TrussFem.getDisp(dof); }
        if( TrussFem.getDisp(dof) < valMin ) { valMin =  TrussFem.getDisp(dof); }

        //myFile<<A<<" "<<allSamples(i, 0)<<" "<<'\n';

        TrussFem.FEMClassReset(false);
    }

    //myFile.close();

    histBin(allSamples, valMax, valMin,true, true);
    //FreqIntergral(allSamples, valMax, valMin);


    return 0;
}
