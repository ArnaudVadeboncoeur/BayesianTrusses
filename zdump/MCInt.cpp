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
#include <Eigen/Dense>
#include "../src/statTools/SampleAnalysis.hpp"


int main(){

    //std::ofstream myFile;
    //myFile.open("results.dat", std::ios::trunc);

    double mu = 0;
    double sig = 0.25;

    double upperBound =  1e10;
    double lowerBound = -1e10;

    std::lognormal_distribution<double> lognormal( mu, sig  );
    std::normal_distribution<double> normal( mu, sig  );
    std::uniform_real_distribution<double> uniform ( 0.0, 1.0 );
    std::random_device rd;
    std::mt19937 engine( rd() );

    double SumDisp = 0;

    int numSamples =  1e5;//1e3;
    double valMax = -1e6;
    double valMin =  1e6;

    Eigen::MatrixXd allSamples (numSamples, 1);

    FEMClass TrussFem(false);

    for(int i = 0; i < numSamples ; i++){

        double A = lognormal( engine );// / 100.0 ;
        //double A2 = lognormal( engine )  ;
        //double A2 = 0.001;
        //double A = 0.01;
        double disp;
        //TrussFem.modA(1, A1);
//        double A;
//
//        double randU = uniform(engine);
//
//        if( randU > 0.75 ) { A = 0.4; }
//        else if( randU > 0.25 ){ A = 0.2;}
//        else{ A = 0.1; }

        TrussFem.modA(0, A);
        TrussFem.assembleS( );
        TrussFem.computeDisp( );
        TrussFem.computeForce( );


        allSamples(i, 0) = TrussFem.getDisp(10);

        //allSamples(i, 1) = TrussFem.getDisp(22);

        //myFile<<A<<" "<<allSamples(i, 0)<<" "<<'\n';

        //std::cout << allSamples(i, 0) << '\n';

        TrussFem.FEMClassReset(false);
        if( (numSamples > 100 * 5 ) && ( i % (numSamples / ( 20 ) )  == 0 ) ){std::cout << "computed " << i << " samples " <<'\n';}
    }

    //myFile.close();
    int nBins = 100;
    histBin(allSamples, nBins, true, true);
    //FreqIntergral(allSamples, valMax, valMin);

//    FEMClass DemoT(false);
//    double A = 0.1;
//    DemoT.modA(0, A);
//    DemoT.assembleS( );
//    DemoT.computeDisp( );
//    std::cout <<" DemoT disp dof9 = " << DemoT.getDisp(10) << std::endl;

    return 0;
}
