
#ifndef DTRUSSDEF37Elm_HPP_
#define DTRUSSDEF37Elm_HPP_

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


    unsigned numberNodes = 14;
    unsigned numberElms  = 37;

    Eigen::VectorXd A( numberElms );
    Eigen::VectorXd E(1);

    Eigen::MatrixXd nodes       (numberNodes, 3);
    Eigen::VectorXi dof         (numberNodes * 3);
    Eigen::MatrixXi members     (numberElms, 2);
    Eigen::MatrixXi memberData  (numberElms, 2);
    Eigen::MatrixXd force       (numberNodes * 3, 1);
    //Areas
    A.setConstant(0.06); //m^2

    //Modulus of Elasticity
    E <<    2e8; // N/m^2

    //Node coordinates
    nodes <<
          //x      y     z
            6.,    0.,   0., //-0
            5.,    2.,   0., //-1
            4.,    0.,   0., //-2
            3.,    2.,   0., //-3
            2.,    0.,   0., //-4
            1.,    2.,   0., //-5
            0.,    0.,   0., //-6

            6.,    0.,   2., //-7
            5.,    2.,   2., //-8
            4.,    0.,   2., //-9
            3.,    2.,   2., //-10
            2.,    0.,   2., //-11
            1.,    2.,   2., //-12
            0.,    0.,   2.; //-13

    //Dof restrained, 0 free, 1 restrained
    //DofNum = nodeNum * 3 + (x=0, y=1, z=2)
    dof<<
          //x     y    z
            1,    1,   1, //-0
            0,    0,   0, //-1
            0,    0,   0, //-2
            0,    0,   0, //-3
            0,    0,   0, //-4
            0,    0,   0, //-5
            1,    1,   1, //-6

            1,    1,   1, //-7
            0,    0,   0, //-8
            0,    0,   0, //-9
            0,    0,   0, //-10
            0,    0,   0, //-11
            0,    0,   0, //-12
            1,    1,   1; //-13


    //Node Connectivity
    members <<
            //front face
            0,   1,  //-0
            1,   2,  //-1
            2,   3,  //-2
            3,   4,  //-3
            4,   5,  //-4
            5,   6,  //-5

            //back face
            7,   8,  //-6
            8,   9,  //-7
            9,   10, //-8
            10,  11, //-9
            11,  12, //-10
            12,  13, //-11

            //bottom chord front
            0,   2, //-12
            2,   4, //-13
            4,   6, //-14

            //bottom chord back
            7,   9,  //-15
            9,   11, //-16
            11,  13, //-17

            //top chord front
            1,   3,  //-18
            3,   5,  //-19

            //top chord back
            8,   10,  //-20
            10,  12,  //-21

            //lateral brace top
            1,   8,  //-22
            3,   10, //-23
            5,   12, //-24

            //lateral brace bottom
            2,   9,  //-25
            4,   11, //-26

            //top cross-brace
            1,   10, //-27
            8,   3,  //-28
            3,   12, //-29
            10,  5,  //-30

            //bottom cross-brace
            0,   9, //-31
            7,   2,  //-32
            2,   11, //-33
            9,   4,  //-34
            4,   13, //-35
            11,  6;  //-36


    //Material Type;

    memberData <<
          //E,   A
            //front face
            0,   0,  //-0
            0,   1,  //-1
            0,   2,  //-2
            0,   3,  //-3
            0,   4,  //-4
            0,   5,  //-5

            //back face
            0,   6,  //-6
            0,   7,  //-7
            0,   8, //-8
            0,   9, //-9
            0,   10, //-10
            0,   11, //-11

            //bottom chord front
            0,   12, //-12
            0,   13, //-13
            0,   14, //-14

            //bottom chord back
            0,   15,  //-15
            0,   16, //-16
            0,   17, //-17

            //top chord front
            0,   18,  //-18
            0,   19,  //-19

            //top chord back
            0,   20,  //-20
            0,   21,  //-21

            //lateral brace top
            0,   22,  //-22
            0,   23, //-23
            0,   24, //-24

            //lateral brace bottom
            0,   25,  //-25
            0,   26, //-26

            //top cross-brace
            0,   27, //-27
            0,   28,  //-28
            0,   29, //-29
            0,   30,  //-30

            //bottom cross-brace
            0,   31, //-31
            0,   32,  //-32
            0,   33, //-33
            0,   34,  //-34
            0,   35, //-35
            0,   36;  //-36


    //force applied at degree of freedom
    force <<
          //x      y     z
            0.,       0.,       0., //-0
            0.,       0.,       0., //-1
            0.,       -1e5,     0., //-2
            0.,       0.,       0., //-3
            0.,       0.,       0., //-4
            0.,       0.,       0., //-5
            0.,       0.,       0., //-6

            0.,       0.,       0., //-7
            0.,       0.,       0., //-8
            0.,       -1e5,     0., //-9
            0.,       0.,       0., //-10
            0.,       0.,       0., //-11
            0.,       0.,       0., //-12
            0.,       0.,       0.; //-13


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
