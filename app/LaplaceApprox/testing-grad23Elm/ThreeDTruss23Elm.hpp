
#ifndef DTRUSSDEF_HPP_
#define DTRUSSDEF_HPP_

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


    unsigned numberNodes = 10;
    unsigned numberElms  = 23;

    Eigen::VectorXd A(4);
    Eigen::VectorXd E(1);
    Eigen::MatrixXd nodes       (numberNodes, 3);
    Eigen::VectorXi dof         (numberNodes * 3);
    Eigen::MatrixXi members     (numberElms, 2);
    Eigen::MatrixXi memberData  (numberElms, 2);
    Eigen::MatrixXd force       (numberNodes * 3, 1);
    //Areas
    A.setConstant(0.025); //m^2

    //Modulus of Elasticity
    E <<    2e8; // N/m^2

    //Node coordinates
    nodes <<
            0.,    0.,   0., //-0
            1.,    0.,   0., //-1
            2.,    0.,   0., //-2
            1.,    1.,   0., //-3
            0.,    1.,   0., //-4

            0.,    0.,   1., //-5
            1.,    0.,   1., //-6
            2.,    0.,   1., //-7
            1.,    1.,   1., //-8
            0.,    1.,   1.; //-9

    //Dof restrainment, 0 free, 1 restrained
    //DofNum = nodeNum * 3 + (x=0, y=1, z=2)
    dof<<
            1,1,1,  //-0
            0,0,0,  //-1
            0,0,0,  //-2
            0,0,0,  //-3
            1,1,1,  //-4

            1,1,1,  //-5
            0,0,0,  //-6
            0,0,0,  //-7
            0,0,0,  //-8
            1,1,1;  //-9


    //Node Connectivity
    members <<
            0, 1,  //-0
            1, 2,  //-1
            2, 3,  //-2
            3, 1,  //-3
            3, 4,  //-4
            3, 0,  //-5

            5, 6,  //-6
            6, 7,  //-7
            7, 8,  //-8
            8, 6,  //-9
            8, 9,  //-10
            8, 5,  //-11

            0, 5,  //-12
            1, 6,  //-13
            2, 7,  //-14
            3, 8,  //-15
            4, 9,  //-16

            5, 1,  //-17
            0, 6,  //-18
            6, 2,  //-19
            1, 7,  //-20
            9, 3,  //-21
            4, 8;  //-22


    //Material Type;
          //E, A
    memberData <<
            0, 0,  //-0
            0, 1,  //-1
            0, 3,  //-2
            0, 3,  //-3
            0, 3,  //-4
            0, 3,  //-5

            0, 3,  //-6
            0, 2,  //-7
            0, 3,  //-8
            0, 3,  //-9
            0, 3,  //-10
            0, 3,  //-11

            0, 3,  //-12
            0, 3,  //-13
            0, 3,  //-14
            0, 3,  //-15
            0, 3,  //-16

            0, 3,  //-17
            0, 3,  //-18
            0, 3,  //-19
            0, 3,  //-20
            0, 3,  //-21
            0, 3;  //-22


    //force applied at degree of fredom
    force <<
            0,     0,     0,  //-0
            0,     0,     0,  //-1
            0,     -1000, 0,  //-2
            0,     0,     0,  //-3
            0,     0,     0,  //-4

            0,     0,     0,  //-5
            0,     0,     0,  //-6
            0,     -1000, 0,  //-7
            0,     0,     0,  //-8
            0,     0,     0;  //-9
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
