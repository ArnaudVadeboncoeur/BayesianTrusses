
#ifndef DTRUSSDEF3BAR_HPP_
#define DTRUSSDEF3BAR_HPP_

#include <Eigen/Dense>
#include <tuple>

using TupleTrussDef = std::tuple <  unsigned, unsigned,
                                    Eigen::VectorXd,
                                    Eigen::VectorXd,
                                    Eigen::MatrixXd,
                                    Eigen::VectorXi,
                                    Eigen::MatrixXi,
                                    Eigen::MatrixXi,
                                    Eigen::MatrixXd  > ;

//---------------------define Truss-----------------------//



TupleTrussDef InitialTrussAssignment(){

    unsigned numberNodes = 4;
    unsigned numberElms  = 3;

    Eigen::VectorXd A(3);
    Eigen::VectorXd E(1);
    Eigen::MatrixXd nodes       (numberNodes, 3);
    Eigen::VectorXi dof         (numberNodes * 3);
    Eigen::MatrixXi members     (numberElms, 2);
    Eigen::MatrixXi memberData  (numberElms, 2);
    Eigen::MatrixXd force       (numberNodes * 3, 1);

    //Areas
    A << 0.01, 0.01, 0.01; //m^2

    //Modulus of Elasticity
    E << 2e8; // N/m^2

    //Node coordinates
    nodes << 0.,    0.,   0.,
             1.,    0.,   0.,
             0.5,   1.,   0.,
             0.5,   0.5,  1.;

    //Dof Restrainment, 0 free, 1 restrained
    //DofNum = nodeNum * 3 + (x=0, y=1, z=2)
    dof << 1,1,1,//    node 0 x,y,z
           1,1,1,//    node 1 x,y,z
           1,1,1,//    node 2 x,y,z
           0,0,0;//    node 3 x,y,z

    //Node Connectivity
    members << 0, 3,
               1, 3,
               2, 3;

    //Material Type;
                //E, A
    memberData << 0, 0,
                  0, 0,
                  0, 0;
    //force applied at degree of freedom
    force << 0,    0,    0,//    node 0 x,y,z
             0,    0,    0,//    node 1 x,y,z
             0,    0,    0,//    node 2 x,y,z
             0,    -1000,0;//    node 3 x,y,z

    return std::make_tuple( numberNodes,
                               numberElms,
                               A, E,
                               nodes,
                               dof,
                               members,
                               memberData,
                               force      );
}


#endif /* DTRUSSDEF_HPP_ */
