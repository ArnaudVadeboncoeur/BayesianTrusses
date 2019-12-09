#include "../../src/FEMClass.hpp"

#include "../../src/statTools/SampleAnalysis.hpp"
#include "../../src/statTools/histSort.hpp"
#include "../../src/statTools/KLDiv.hpp"
#include "../../src/statTools/SteinDiscrepancy.hpp"

#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>

#include "ThreeDTruss3Bar2sdof.hpp"

std::tuple<Eigen::MatrixXd, std::vector<double> > trueSampleGen(){
    bool verbosity = false;
    double mu1  = 1;
    double sig1 = 0.25;

    double mu2  = 2;
    double sig2 = 0.5;


    std::lognormal_distribution<double> lognormal1( mu1, sig1  );
    std::lognormal_distribution<double> lognormal2( mu2, sig2  );
    std::uniform_real_distribution<double> uniform( 0, 1  );

    std::random_device rd;
    std::mt19937 engine( rd() );

    int numSamples =  1e4;//1e3;
    std::vector<double> forcing (numSamples) ;

    Eigen::MatrixXd allSamples (numSamples, 1);


    TupleTrussDef TrussDef;
    TrussDef =  InitialTrussAssignment();

    FEMClass trueTrussFem(false, TrussDef );

    for(int i = 0; i < numSamples ; i++){

        double randU = uniform(engine);
        double F;
        if( randU > 0.66 ) { F = -100; }
        else{ F = -200; }


        double A1 = lognormal1( engine );
        double A2 = lognormal2( engine );

        double disp;

        trueTrussFem.modA(0, A1);
        trueTrussFem.modA(1, A2);
        trueTrussFem.modForce(2, 0, F);

        trueTrussFem.assembleS( );
        trueTrussFem.computeDisp( );
        trueTrussFem.computeForce( );
        allSamples(i, 0) = trueTrussFem.getDisp(10);
        forcing[i] = F;

        trueTrussFem.FEMClassReset(false);
        if( verbosity == true){if( (numSamples > 100 * 5 ) && ( i % (numSamples / ( 20 ) )  == 0 ) ){std::cout << "computed " << i << " samples " <<'\n';}}
    }

    std::vector<double>  delatXs = findDeltaX(allSamples, 100);
    HistContainer histPoints = histBin(allSamples, delatXs, true, true);

    return std::make_tuple(allSamples, forcing );

}



int main(){

    std::tuple<Eigen::MatrixXd, std::vector<double> > trueSamplesTuple = trueSampleGen();
    Eigen::MatrixXd trueSamples;
    std::vector<double> forcing;

    std::tie (trueSamples, forcing) = trueSamplesTuple;
    std::cout << trueSamples << std::endl;

    return 0;
}


