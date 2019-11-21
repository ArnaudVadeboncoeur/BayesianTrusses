/*
 * MCSampFEM.cpp
 *
 *  Created on: 21 Nov 2019
 *      Author: arnaudv
 */

#include <iostream>
#include "FEMFunc.hpp"



#include <Eigen/Dense>


int main(){

    FEMFUNC TrussFem(true);
    TrussFem.assembleS();
    TrussFem.computeDisp();
    TrussFem.computeForce();

    return 0;
}
